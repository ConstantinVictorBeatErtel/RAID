"""Raw-image training for the two MAE-LoRA conditions.

The retrieval bank remains the frozen feature bank.  Only the query features
fed to the action decoder are adapted, which makes the LoRA comparison a
visual-adaptation versus retrieval test rather than a moving-neighbour test.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data_libero import _find_img_key

from .config import ExperimentConfig
from .data import FeatureTransitions
from .models import DirectDecoder, RAIDDecoder, trainable_parameter_names
from .raw import assert_lora_isolated, encode_trainable, load_raw_encoder_pair
from .retrieval import TransitionMemory
from .splits import demo_uid, load_and_validate, select_training_ids, transition_uid
from .training import _atomic_torch_save, seed_everything


class RawImageTransitions(Dataset):
    """Selected LIBERO transitions with explicit task/demo/timestep IDs."""

    def __init__(self, cfg: ExperimentConfig, split_name: str):
        split = load_and_validate(cfg.data.split_path)
        self.future_horizon = int(cfg.horizon)
        self.items: list[dict] = []
        action_rows: list[np.ndarray] = []
        task_rows: list[tuple[Path, str, list[int], int]] = []
        for task_ordinal, task in enumerate(split["tasks"]):
            path = Path(task["path"])
            if not path.exists():
                path = Path(cfg.data.dataset_dir) / path.name
            if not path.exists():
                raise FileNotFoundError(f"Dataset file from split is missing: {task['path']}")
            ids = (select_training_ids(task, cfg.data.train_per_task, cfg.data.train_seed)
                   if split_name == "train" else list(map(int, task[split_name])))
            task_rows.append((path, task["task"], ids, task_ordinal))
            if split_name == "train":
                with h5py.File(path, "r") as handle:
                    for demo_id in ids:
                        action_rows.append(np.asarray(handle[f"data/demo_{demo_id}/actions"][: -self.future_horizon], dtype=np.float32))
        if action_rows:
            actions = np.concatenate(action_rows, axis=0)
            self.mean = actions.mean(axis=0).astype(np.float32)
            raw_std = actions.std(axis=0)
            self.std = np.where(raw_std < 1e-6, 1.0, raw_std).astype(np.float32)
        else:
            # Validation/test must use the training-only normalization.  The
            # caller replaces these values after constructing the dataset.
            self.mean = np.zeros(7, dtype=np.float32)
            self.std = np.ones(7, dtype=np.float32)
        for path, task_name, ids, task_ordinal in task_rows:
            with h5py.File(path, "r") as handle:
                for demo_id in ids:
                    group = handle[f"data/demo_{demo_id}"]
                    images = group["obs"][_find_img_key(group["obs"])][:]
                    actions = np.asarray(group["actions"][: -cfg.horizon], dtype=np.float32)
                    for timestep, action in enumerate(actions):
                        self.items.append({
                            "image_t": images[timestep],
                            "image_next": images[timestep + cfg.horizon],
                            "action_raw": action,
                            "task": task_name,
                            "demo_id": demo_uid(task_name, demo_id),
                            "timestep": timestep,
                            "transition_uid": transition_uid(task_name, demo_id, timestep),
                        })
        self.items.sort(key=lambda row: row["transition_uid"])

    def set_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        action = (row["action_raw"] - self.mean) / (self.std + 1e-8)
        return {
            "image_t": torch.from_numpy(row["image_t"]),
            "image_next": torch.from_numpy(row["image_next"]),
            "action": torch.from_numpy(action.astype(np.float32)),
            "demo_id": row["demo_id"],
            "timestep": row["timestep"],
            "task": row["task"],
            "transition_uid": row["transition_uid"],
        }


def _collate(batch):
    return {
        "image_t": torch.stack([x["image_t"] for x in batch]),
        "image_next": torch.stack([x["image_next"] for x in batch]),
        "action": torch.stack([x["action"] for x in batch]),
        "demo_id": torch.tensor([x["demo_id"] for x in batch], dtype=torch.long),
        "timestep": torch.tensor([x["timestep"] for x in batch], dtype=torch.long),
        "task": [x["task"] for x in batch],
        "transition_uid": [x["transition_uid"] for x in batch],
    }


def _fixed_queries(feature_dir, split_path, split_name, horizon, train_per_task=None, train_seed=42, norm_mean=None, norm_std=None):
    dataset = FeatureTransitions(feature_dir, split_path, split_name, train_per_task=train_per_task, train_seed=train_seed, horizon=horizon, norm_mean=norm_mean, norm_std=norm_std)
    return {(x.demo_id, x.timestep): (x.current, x.future) for x in dataset.items}


def run_raw_condition(
    condition: str,
    cfg: ExperimentConfig,
    *,
    mae_checkpoint: str,
    output_dir: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    stop_after_updates: int | None = None,
) -> dict:
    if condition not in {"lora", "lora_raid"}:
        raise ValueError("raw runner is reserved for lora and lora_raid")
    seed_everything(cfg.data.train_seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    train = RawImageTransitions(cfg, "train")
    validation = RawImageTransitions(cfg, "validation")
    validation.set_norm(train.mean, train.std)
    norm_path = Path(output_dir or cfg.output_dir) / "action_normalizer.pt"
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    if norm_path.exists():
        saved = torch.load(norm_path, map_location="cpu", weights_only=False)
        saved_mean = saved["mean"].numpy() if torch.is_tensor(saved["mean"]) else saved["mean"]
        saved_std = saved["std"].numpy() if torch.is_tensor(saved["std"]) else saved["std"]
        train.set_norm(saved_mean, saved_std)
    else:
        torch.save({"mean": torch.from_numpy(train.mean), "std": torch.from_numpy(train.std), "train_seed": cfg.data.train_seed, "train_per_task": cfg.data.train_per_task}, norm_path)
    fixed_train = _fixed_queries(cfg.data.feature_dir, cfg.data.split_path, "train", cfg.horizon, cfg.data.train_per_task, cfg.data.train_seed, torch.from_numpy(train.mean), torch.from_numpy(train.std))
    fixed_val = _fixed_queries(cfg.data.feature_dir, cfg.data.split_path, "validation", cfg.horizon, cfg.data.train_per_task, cfg.data.train_seed, torch.from_numpy(train.mean), torch.from_numpy(train.std))
    memory = None
    if condition == "lora_raid":
        memory = TransitionMemory(cfg.model.feature_dim, cfg.model.action_dim, device)
        action_by_key = {(x["demo_id"], x["timestep"]): x["action_raw"] for x in train.items}
        bank_current, bank_future, bank_actions, bank_ids = [], [], [], []
        for key, (current, future) in fixed_train.items():
            demo_id, timestep = key
            action = action_by_key[key]
            normalized = (action - train.mean) / (train.std + 1e-8)
            bank_current.append(current); bank_future.append(future)
            bank_actions.append(torch.from_numpy(normalized)); bank_ids.append(demo_id)
        memory.extend(torch.stack(bank_current), torch.stack(bank_future), torch.stack(bank_actions), torch.tensor(bank_ids, dtype=torch.long))
    pair = load_raw_encoder_pair(
        mae_checkpoint,
        str(device),
        blocks=cfg.model.lora_target_blocks,
        rank=cfg.model.lora_rank,
        alpha=cfg.model.lora_alpha,
        dropout=cfg.model.lora_dropout,
    )
    assert_lora_isolated(pair)
    if condition == "lora_raid":
        decoder = RAIDDecoder(cfg.model.feature_dim, cfg.model.action_dim, cfg.model.hidden_dim,
                              cfg.model.bottleneck_dim, cfg.model.decoder_dropout,
                              cfg.model.prior_dropout, cfg.model.prior_noise_std).to(device)
    else:
        decoder = DirectDecoder(cfg.model.feature_dim, cfg.model.action_dim, cfg.model.hidden_dim,
                                cfg.model.bottleneck_dim, cfg.model.decoder_dropout).to(device)
    lora_parameters = [p for p in pair.adapted.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": decoder.parameters(), "lr": cfg.training.decoder_lr},
        {"params": lora_parameters, "lr": cfg.training.lora_lr},
    ], weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.updates)
    generator = torch.Generator().manual_seed(int(cfg.data.train_seed))
    order = torch.randperm(len(train), generator=generator).tolist(); cursor = 0
    val_loader = DataLoader(validation, batch_size=cfg.training.microbatch_size, shuffle=False, num_workers=0, collate_fn=_collate)
    curve, best, best_step = [], float("inf"), 0
    start_step = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("config_digest") != cfg.digest():
            raise ValueError("resume checkpoint was produced by a different configuration")
        decoder.load_state_dict(checkpoint["decoder"])
        pair.adapted.mae.load_state_dict(checkpoint["mae"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint.get("step", checkpoint.get("best_step", 0)))
        best = float(checkpoint.get("best_validation_mse", float("inf")))
        best_step = int(checkpoint.get("best_step", 0))
        curve = list(checkpoint.get("curve", []))
        if checkpoint.get("rng_state") is not None: torch.set_rng_state(checkpoint["rng_state"].cpu())
        if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if checkpoint.get("python_rng_state") is not None: random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("numpy_rng_state") is not None: np.random.set_state(checkpoint["numpy_rng_state"])
        if start_step >= cfg.training.updates:
            out = Path(output_dir or cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
            if not (out / f"{condition}.pt").exists():
                raise RuntimeError("completed resume is missing the validation-selected best checkpoint")
            result = {"condition": condition, "status": "complete", "seed": cfg.data.train_seed, "best_validation_mse": best, "best_step": best_step, "seconds": 0.0, "curve": curve, "trainable_parameters": sum(p.numel() for p in decoder.parameters() if p.requires_grad) + sum(p.numel() for p in lora_parameters), "lora_targets": trainable_parameter_names(pair.adapted.mae), "future_mode": "observed", "config_digest": cfg.digest(), "horizon": cfg.horizon, "representation": "recorded_future_raw_image", "demonstrations_per_task": cfg.data.train_per_task, "updates": cfg.training.updates}
            (out / f"{condition}.json").write_text(json.dumps(result, indent=2) + "\n")
            return result

    accumulation = cfg.training.batch_size // cfg.training.microbatch_size
    target_updates = min(cfg.training.updates, int(stop_after_updates)) if stop_after_updates is not None else cfg.training.updates
    if target_updates < 1:
        raise ValueError("stop_after_updates must be positive")
    def next_train_batch():
        nonlocal order, cursor
        indices = []
        while len(indices) < cfg.training.microbatch_size:
            if cursor >= len(order):
                order = torch.randperm(len(train), generator=generator).tolist(); cursor = 0
            take = min(cfg.training.microbatch_size - len(indices), len(order) - cursor)
            indices.extend(order[cursor:cursor + take]); cursor += take
        return _collate([train[i] for i in indices])

    for _ in range(start_step * accumulation):
        next_train_batch()
    started = time.time()
    for step in range(start_step + 1, target_updates + 1):
        optimizer.zero_grad(set_to_none=True); loss_value = 0.0
        for _ in range(cfg.training.batch_size // cfg.training.microbatch_size):
            batch = next_train_batch()
            current = encode_trainable(pair.adapted, batch["image_t"])
            future = encode_trainable(pair.adapted, batch["image_next"], patch_normalized=(cfg.horizon == 3))
            fixed_current = torch.stack([fixed_train[(int(d), int(t))][0] for d, t in zip(batch["demo_id"], batch["timestep"])])
            fixed_future = torch.stack([fixed_train[(int(d), int(t))][1] for d, t in zip(batch["demo_id"], batch["timestep"])])
            result = memory.retrieve(fixed_current.to(device), fixed_future.to(device), k=cfg.model.retrieval_k, query_demo_ids=batch["demo_id"].to(device)) if memory is not None else None
            pred, _ = decoder(current, future, None if result is None else result.actions.to(device), None if result is None else result.valid.to(device))
            loss = torch.nn.functional.mse_loss(pred, batch["action"].to(device))
            loss_value += float(loss.item())
            (loss / (cfg.training.batch_size // cfg.training.microbatch_size)).backward()
        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + lora_parameters, cfg.training.grad_clip); optimizer.step(); scheduler.step()
        if step % cfg.training.validate_every == 0 or step == cfg.training.updates:
            decoder.eval(); pair.adapted.eval(); demo_scores = {}; prediction_rows = []; target_rows = []; prediction_ids = []; neighbor_rows = []
            with torch.no_grad():
                for vb in val_loader:
                    vc = encode_trainable(pair.adapted, vb["image_t"]); vf = encode_trainable(pair.adapted, vb["image_next"], patch_normalized=(cfg.horizon == 3))
                    fc = torch.stack([fixed_val[(int(d), int(t))][0] for d, t in zip(vb["demo_id"], vb["timestep"])])
                    ff = torch.stack([fixed_val[(int(d), int(t))][1] for d, t in zip(vb["demo_id"], vb["timestep"])])
                    vr = memory.retrieve(fc.to(device), ff.to(device), k=cfg.model.retrieval_k, query_demo_ids=vb["demo_id"].to(device)) if memory is not None else None
                    vp, _ = decoder(vc, vf, None if vr is None else vr.actions.to(device), None if vr is None else vr.valid.to(device))
                    errors = (vp - vb["action"].to(device)).pow(2).mean(dim=1).cpu()
                    prediction_rows.append(vp.cpu())
                    target_rows.append(vb["action"].cpu())
                    prediction_ids.extend(zip(vb["task"], vb["demo_id"].tolist(), vb["timestep"].tolist()))
                    if vr is not None:
                        neighbor_rows.append({"indices": vr.indices.cpu(), "similarities": vr.similarities.cpu(), "valid": vr.valid.cpu()})
                    for task, demo_id, error in zip(vb["task"], vb["demo_id"].tolist(), errors.tolist()):
                        demo_scores.setdefault((task, int(demo_id)), []).append(float(error))
            task_scores = {}
            for (task, _demo_id), values in demo_scores.items():
                task_scores.setdefault(task, []).append(float(np.mean(values)))
            val = float(np.mean([np.mean(scores) for scores in task_scores.values()])) if task_scores else float("inf")
            curve.append({"step": step, "train_mse": loss_value / (cfg.training.batch_size // cfg.training.microbatch_size), "validation_mse": val})
            if val < best:
                best, best_step = val, step
                out = Path(output_dir or cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
                _atomic_torch_save({"condition": condition, "decoder": decoder.state_dict(), "mae": pair.adapted.mae.state_dict(), "config_digest": cfg.digest(), "step": step, "best_step": step, "best_validation_mse": val, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(), "curve": curve, "normalizer_mean": train.mean, "normalizer_std": train.std}, out / f"{condition}.pt")
                torch.save({"predictions": torch.cat(prediction_rows), "targets": torch.cat(target_rows), "transition_ids": prediction_ids, "neighbor_batches": neighbor_rows, "validation_mse": val, "step": step}, out / f"{condition}_validation.pt")
            decoder.train(); pair.adapted.eval(); pair.predictor.eval()
        out = Path(output_dir or cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
        if step % 100 == 0 or step == cfg.training.updates:
            _atomic_torch_save({"condition": condition, "decoder": decoder.state_dict(), "mae": pair.adapted.mae.state_dict(), "config_digest": cfg.digest(), "step": step, "best_step": best_step, "best_validation_mse": best, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(), "curve": curve, "normalizer_mean": train.mean, "normalizer_std": train.std}, out / f"{condition}.last.pt")
    result = {"condition": condition, "status": "calibration" if target_updates < cfg.training.updates else "complete", "seed": cfg.data.train_seed, "best_validation_mse": best, "best_step": best_step, "seconds": time.time() - started, "curve": curve, "trainable_parameters": sum(p.numel() for p in decoder.parameters() if p.requires_grad) + sum(p.numel() for p in lora_parameters), "lora_targets": trainable_parameter_names(pair.adapted.mae), "future_mode": "observed", "config_digest": cfg.digest(), "horizon": cfg.horizon, "representation": "recorded_future_raw_image", "demonstrations_per_task": cfg.data.train_per_task, "updates": cfg.training.updates, "completed_updates": target_updates}
    out = Path(output_dir or cfg.output_dir); out.mkdir(parents=True, exist_ok=True); (out / f"{condition}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
