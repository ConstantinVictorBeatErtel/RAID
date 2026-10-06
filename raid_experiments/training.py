"""Legacy pilot training retained for audit; active CLI execution is disabled."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import FeatureTransitions, collate_transitions
from .models import DirectDecoder, RAIDDecoder
from .retrieval import TransitionMemory
from .evaluation import hierarchical_macro_mse

Condition = Literal["direct", "raid", "lora", "lora_raid"]


def _atomic_torch_save(payload, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(condition: Condition, cfg: ExperimentConfig):
    if condition in ("raid", "lora_raid"):
        return RAIDDecoder(cfg.model.feature_dim, cfg.model.action_dim, cfg.model.hidden_dim, cfg.model.bottleneck_dim, cfg.model.decoder_dropout, cfg.model.prior_dropout, cfg.model.prior_noise_std)
    return DirectDecoder(cfg.model.feature_dim, cfg.model.action_dim, cfg.model.hidden_dim, cfg.model.bottleneck_dim, cfg.model.decoder_dropout)


def _memory(dataset: FeatureTransitions, device: torch.device) -> TransitionMemory:
    memory = TransitionMemory(dataset.items[0].current.numel(), dataset.items[0].action.numel(), device)
    memory.extend(torch.stack([x.current for x in dataset.items]),
                  torch.stack([x.future for x in dataset.items]),
                  torch.stack([x.action for x in dataset.items]),
                  torch.tensor([x.demo_id for x in dataset.items], dtype=torch.long))
    return memory


def _nearest(current, future, memory, condition, demo_id, cfg):
    if condition == "direct":
        return None
    if condition == "raid" or condition == "lora_raid":
        return memory.retrieve(current, future, k=cfg.model.retrieval_k, query_demo_ids=demo_id)
    raise ValueError(f"unsupported cached-feature condition: {condition}")


def _scheduled_batches(dataset, microbatch: int, updates: int, seed: int):
    """Yield identical transition IDs for every condition using an isolated RNG."""
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(dataset), generator=generator).tolist()
    cursor = 0
    for _ in range(updates):
        batch_indices = []
        while len(batch_indices) < microbatch:
            if cursor >= len(order):
                order = torch.randperm(len(dataset), generator=generator).tolist()
                cursor = 0
            take = min(microbatch - len(batch_indices), len(order) - cursor)
            batch_indices.extend(order[cursor:cursor + take]); cursor += take
        yield collate_transitions([dataset[i] for i in batch_indices])


def run_condition(condition: Condition, cfg: ExperimentConfig, *, output_dir: str | Path | None = None,
                  resume_checkpoint: str | Path | None = None, stop_after_updates: int | None = None) -> dict:
    """Train one condition and write an auditable result bundle.

    The feature-cache path intentionally exposes ``lora`` as a decoder-side
    smoke adapter only. The publication LoRA condition must call the raw-image
    runner; it must not silently report cached features
    as MAE-adapted features.
    """
    if condition not in {"direct", "raid", "lora", "lora_raid"}:
        raise ValueError(condition)
    seed_everything(cfg.data.train_seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    train = FeatureTransitions(cfg.data.feature_dir, cfg.data.split_path, "train", train_per_task=cfg.data.train_per_task, train_seed=cfg.data.train_seed, horizon=cfg.horizon)
    validation = FeatureTransitions(cfg.data.feature_dir, cfg.data.split_path, "validation", train_per_task=cfg.data.train_per_task, train_seed=cfg.data.train_seed, horizon=cfg.horizon, norm_mean=train.norm_mean, norm_std=train.norm_std)
    if (train.items[0].current.numel() != cfg.model.feature_dim
            or train.items[0].future.numel() != cfg.model.feature_dim
            or train.items[0].action.numel() != cfg.model.action_dim):
        raise ValueError(f"feature cache dimensions do not match config: cache={(train.items[0].current.numel(), train.items[0].action.numel())} config={(cfg.model.feature_dim, cfg.model.action_dim)}")
    memory = _memory(train, device) if condition in {"raid", "lora_raid"} else None
    norm_path = Path(output_dir or cfg.output_dir) / "action_normalizer.pt"
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    if norm_path.exists():
        saved_norm = torch.load(norm_path, map_location="cpu", weights_only=False)
        if not torch.allclose(saved_norm["mean"], train.norm_mean) or not torch.allclose(saved_norm["std"], train.norm_std):
            raise RuntimeError("persisted action normalizer does not match this split, budget, and seed")
    else:
        torch.save({"mean": train.norm_mean, "std": train.norm_std, "train_seed": cfg.data.train_seed, "train_per_task": cfg.data.train_per_task}, norm_path)
    (Path(output_dir or cfg.output_dir) / "training_subset.json").write_text(json.dumps({"seed": cfg.data.train_seed, "demonstrations_per_task": cfg.data.train_per_task, "task_demo_ids": train.selected_demo_ids}, indent=2) + "\n")
    model = build_model(condition, cfg).to(device)
    if condition in {"lora", "lora_raid"}:
        raise RuntimeError("Cached-feature training cannot claim MAE LoRA. Use the raw-image runner with --lora-mae.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.decoder_lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.updates)
    vloader = DataLoader(validation, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=collate_transitions)
    schedule = _scheduled_batches(train, cfg.training.microbatch_size, cfg.training.updates * (cfg.training.batch_size // cfg.training.microbatch_size), cfg.data.train_seed)
    curve = []
    best = float("inf")
    best_step = 0
    start_step = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("config_digest") != cfg.digest():
            raise ValueError("resume checkpoint was produced by a different configuration")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint.get("step", checkpoint.get("best_step", 0)))
        best = float(checkpoint.get("best_validation_mse", float("inf")))
        best_step = int(checkpoint.get("best_step", 0))
        curve = list(checkpoint.get("curve", []))
        for _ in range(start_step * (cfg.training.batch_size // cfg.training.microbatch_size)):
            next(schedule)
        if checkpoint.get("rng_state") is not None:
            torch.set_rng_state(checkpoint["rng_state"].cpu())
        if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if checkpoint.get("schedule_rng_state") is not None:
            # The schedule generator is independent from model RNG. Restore it
            # after advancing the iterator to the checkpoint boundary.
            pass
        if start_step >= cfg.training.updates:
            out_dir = Path(output_dir or cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
            if not (out_dir / f"{condition}.pt").exists():
                raise RuntimeError("completed resume is missing the validation-selected best checkpoint")
            result = {"condition": condition, "seed": cfg.data.train_seed, "best_validation_mse": best, "best_step": best_step, "seconds": 0.0, "curve": curve, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "feature_cache_lora": False, "config_digest": cfg.digest(), "horizon": cfg.horizon, "representation": "observed_future_feature_cache", "demonstrations_per_task": cfg.data.train_per_task, "updates": cfg.training.updates, "status": "complete"}
            (out_dir / f"{condition}.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
    target_updates = min(cfg.training.updates, int(stop_after_updates)) if stop_after_updates is not None else cfg.training.updates
    if target_updates < 1:
        raise ValueError("stop_after_updates must be positive")
    started = time.time()
    for step in range(start_step + 1, target_updates + 1):
        optimizer.zero_grad(set_to_none=True); loss_value = 0.0
        for _ in range(cfg.training.batch_size // cfg.training.microbatch_size):
            batch = next(schedule)
            current, future, action = (batch[x].to(device) for x in ("current", "future", "action"))
            result = _nearest(current, future, memory, condition, batch["demo_id"], cfg)
            pred, _ = model(current, future, None if result is None else result.actions.to(device), None if result is None else result.valid.to(device))
            loss = torch.nn.functional.mse_loss(pred, action)
            loss_value += float(loss.item())
            (loss / (cfg.training.batch_size // cfg.training.microbatch_size)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip); optimizer.step(); scheduler.step()
        if step % cfg.training.validate_every == 0 or step == cfg.training.updates:
            model.eval(); demo_scores = {}; prediction_rows = []; target_rows = []; prediction_ids = []; neighbor_rows = []
            with torch.no_grad():
                for vb in vloader:
                    vc, vf, va = (vb[x].to(device) for x in ("current", "future", "action"))
                    vr = _nearest(vc, vf, memory, condition, vb["demo_id"], cfg)
                    vp, _ = model(vc, vf, None if vr is None else vr.actions.to(device), None if vr is None else vr.valid.to(device))
                    errors = (vp - va).pow(2).mean(dim=1).cpu()
                    prediction_rows.append(vp.cpu());
                    target_rows.append(va.cpu())
                    prediction_ids.extend(zip(vb["task"], vb["demo_id"].tolist(), vb["timestep"].tolist()))
                    if vr is not None:
                        neighbor_rows.append({"indices": vr.indices.cpu(), "similarities": vr.similarities.cpu(), "valid": vr.valid.cpu()})
                    for task, demo_id, error in zip(vb["task"], vb["demo_id"].tolist(), errors.tolist()):
                        demo_scores.setdefault((task, int(demo_id)), []).append(float(error))
            # Demonstrations, then tasks, are the statistical units. A long
            # demonstration must not dominate checkpoint selection.
            val = hierarchical_macro_mse(
                [value for values in demo_scores.values() for value in values],
                [task for (task, _demo) in demo_scores for _ in demo_scores[(task, _demo)]],
                [demo for (_task, demo), values in demo_scores.items() for _ in values],
            ) if demo_scores else float("inf")
            curve.append({"step": step, "train_mse": loss_value / (cfg.training.batch_size // cfg.training.microbatch_size), "validation_mse": val})
            if val < best:
                best, best_step = val, step
                Path(output_dir or cfg.output_dir).mkdir(parents=True, exist_ok=True)
                _atomic_torch_save({"condition": condition, "state_dict": model.state_dict(), "config_digest": cfg.digest(), "step": step, "best_step": step, "best_validation_mse": val, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(), "curve": curve, "normalizer_mean": train.norm_mean, "normalizer_std": train.norm_std}, Path(output_dir or cfg.output_dir) / f"{condition}.pt")
                torch.save({"predictions": torch.cat(prediction_rows), "targets": torch.cat(target_rows), "transition_ids": prediction_ids, "neighbor_batches": neighbor_rows, "validation_mse": val, "step": step}, Path(output_dir or cfg.output_dir) / f"{condition}_validation.pt")
            model.train()
        state = {"condition": condition, "state_dict": model.state_dict(), "config_digest": cfg.digest(), "step": step, "best_step": best_step, "best_validation_mse": best, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(), "curve": curve, "normalizer_mean": train.norm_mean, "normalizer_std": train.norm_std}
        if step % 100 == 0 or step == target_updates:
            _atomic_torch_save(state, Path(output_dir or cfg.output_dir) / f"{condition}.last.pt")
    out = {"condition": condition, "status": "calibration" if target_updates < cfg.training.updates else "complete", "seed": cfg.data.train_seed, "best_validation_mse": best, "best_step": best_step, "seconds": time.time() - started, "curve": curve, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "feature_cache_lora": False, "config_digest": cfg.digest(), "horizon": cfg.horizon, "representation": "observed_future_feature_cache", "demonstrations_per_task": cfg.data.train_per_task, "updates": cfg.training.updates, "completed_updates": target_updates}
    out_dir = Path(output_dir or cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True); (out_dir / f"{condition}.json").write_text(json.dumps(out, indent=2) + "\n")
    return out
