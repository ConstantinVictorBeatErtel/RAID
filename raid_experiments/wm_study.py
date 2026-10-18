"""Protocol v4 data, training, adaptation and recorded-transition diagnostic.

Data contract
  Frame cache: per demonstration, frozen MAE CLS features of every recorded
  frame (T x 768). Source and support shards carry recorded actions (T x 7);
  query shards never do. Query labels are read from HDF5 only inside
  ``evaluate_recorded`` and every read is appended to the query-access ledger.
  Chunk sample at step t (0 <= t <= T-1-H): inputs (o_t, o_t+H), target
  a_t..a_t+H-1, normalized per dimension with the source-training normalizer.

Matching
  Both heads are source-trained from one query stream (same tasks, steps and
  transitions per update). Target adaptation for both arms uses the identical
  support IDs; evaluation uses identical query demonstrations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .models import attach_mae_lora, trainable_parameter_names
from .wm_heads import ChunkMLP, RetrievalMLP, chunk_mse, knn_average, make_bank, retrieval_block
from .wm_protocol import (ACTION_DIM, ARMS, PROTOCOL_V4, SUPPORT_BUDGETS, HeadSpec, RetrievalSpec,
                          TemporalAlignment, validate_wm_protocol)

CACHE_FORMAT = 1
CHECKPOINT_FORMAT = 1
SCIENTIFIC = "scientific"
SYNTHETIC = "synthetic"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def demo_uid(task: str, demo: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{task}\x1f{int(demo)}".encode()).digest()[:8], "big") & ((1 << 63) - 1)


def demo_role(spec: dict, demo: int) -> str:
    demo = int(demo)
    if spec["role"] == "source":
        for role, ids in spec["source"].items():
            if demo in set(map(int, ids)):
                return role
        raise ValueError(f"source demo {demo} has no split role")
    if demo in set(map(int, spec["query"])):
        return "query"
    if demo in set(map(int, spec["support_pool"])):
        return "support_pool"
    raise ValueError(f"held-out demo {demo} has no split role")


def _image_key(obs) -> str:
    for key in ("agentview_rgb", "agentview_image", "image"):
        if key in obs:
            return key
    raise KeyError("demonstration has no agent-view RGB")


def read_frames(path: str | Path, demo: int) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as handle:
        obs = handle[f"data/demo_{int(demo)}/obs"]
        return np.asarray(obs[_image_key(obs)])


def _read_actions(path: str | Path, demo: int) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[f"data/demo_{int(demo)}/actions"], dtype=np.float32)


def read_initial_state(path: str | Path, demo: int) -> np.ndarray:
    """Initial simulator state of a demonstration (no action labels are read)."""
    import h5py
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[f"data/demo_{int(demo)}/states"][0])


@torch.no_grad()
def _encode(encoder, frames: np.ndarray, batch: int = 64) -> torch.Tensor:
    out = []
    for start in range(0, len(frames), batch):
        out.append(encoder.encode_frames(torch.from_numpy(np.ascontiguousarray(frames[start:start + batch]))).float().cpu())
    return torch.cat(out)


def encoder_identity(encoder) -> dict:
    state = getattr(encoder, "mae", encoder).state_dict()
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().float().numpy().tobytes())
    return {"encoder_class": type(encoder).__name__, "weights_sha256": digest.hexdigest(),
            "feature_dim": int(getattr(encoder, "feat_dim", 768))}


def cache_frame_features(protocol: dict, encoder, out_dir: str | Path, *, run_mode: str = SCIENTIFIC) -> dict:
    validate_wm_protocol(protocol)
    out = Path(out_dir)
    identity = encoder_identity(encoder)
    shards = {}
    for task, spec in protocol["tasks"].items():
        if run_mode == SCIENTIFIC and _sha_file(Path(spec["path"])) != spec["source_sha256"]:
            raise ValueError(f"HDF5 source hash mismatch for {task}")
        shards[task] = {}
        for demo in spec["demo_ids"]:
            role = demo_role(spec, demo)
            frames = read_frames(spec["path"], demo)
            payload: dict[str, Any] = {"task": task, "demo_id": int(demo), "role": role,
                                       "features": _encode(encoder, frames), "frame_count": int(len(frames)),
                                       "protocol_digest": protocol["digest"], "encoder": identity}
            if role != "query":
                actions = _read_actions(spec["path"], demo)
                if len(actions) != len(frames):
                    raise ValueError(f"{task}:{demo} frame/action count mismatch")
                payload["actions"] = torch.from_numpy(actions)
            rel = Path(task) / f"demo_{int(demo)}.pt"
            (out / task).mkdir(parents=True, exist_ok=True)
            torch.save(payload, out / rel)
            shards[task][str(int(demo))] = {"path": rel.as_posix(), "role": role, "sha256": _sha_file(out / rel)}
    manifest = {"format": CACHE_FORMAT, "protocol": PROTOCOL_V4, "protocol_digest": protocol["digest"],
                "model_digest": protocol["model_digest"], "execution_mode": run_mode, "encoder": identity,
                "shards": shards, "query_labels_present": False}
    _write_json(out / "manifest.json", manifest)
    return manifest


def load_frame_cache(cache_dir: str | Path, protocol: dict, *, run_mode: str = SCIENTIFIC) -> dict:
    root = Path(cache_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("model_digest") != protocol["model_digest"] or manifest.get("format") != CACHE_FORMAT:
        raise ValueError("frame cache belongs to another protocol")
    if manifest.get("execution_mode") != run_mode:
        raise ValueError(f"{run_mode} execution requires a {run_mode} frame cache")
    cache: dict[str, dict[int, dict]] = {}
    for task, records in manifest["shards"].items():
        cache[task] = {}
        for demo, record in records.items():
            path = root / record["path"]
            if _sha_file(path) != record["sha256"]:
                raise ValueError(f"stale frame shard {path}")
            shard = torch.load(path, map_location="cpu", weights_only=False)
            if shard["role"] == "query" and "actions" in shard:
                raise ValueError("query shard contains forbidden action labels")
            cache[task][int(demo)] = shard
    return {"manifest": manifest, "shards": cache}


def source_normalizer(protocol: dict, cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    values = [cache["shards"][task][int(d)]["actions"] for task, spec in protocol["tasks"].items()
              if spec["role"] == "source" for d in spec["source"]["source_train"]]
    stacked = torch.cat(values).float()
    mean, std = stacked.mean(0), stacked.std(0, unbiased=False)
    return mean, torch.where(std < 1e-6, torch.ones_like(std), std)


@dataclass
class ChunkRows:
    current: torch.Tensor
    future: torch.Tensor
    target: torch.Tensor | None
    demo_uid: torch.Tensor
    demo: list[int]
    step: list[int]
    task: str

    def __len__(self):
        return len(self.current)


def chunk_rows(shard_by_demo: dict[int, dict], demos: list[int], task: str, chunk: int,
               mean: torch.Tensor | None = None, std: torch.Tensor | None = None, *,
               actions_override: dict[int, np.ndarray] | None = None) -> ChunkRows:
    cur, fut, tgt, uid, dd, ss = [], [], [], [], [], []
    for demo in demos:
        shard = shard_by_demo[int(demo)]
        feats = shard["features"].float()
        actions = (torch.as_tensor(actions_override[int(demo)]) if actions_override and int(demo) in actions_override
                   else shard.get("actions"))
        steps = range(0, len(feats) - chunk)
        for t in steps:
            cur.append(feats[t]); fut.append(feats[t + chunk])
            if actions is not None:
                tgt.append(((actions[t:t + chunk].float() - mean) / std) if mean is not None else actions[t:t + chunk].float())
            uid.append(demo_uid(task, demo)); dd.append(int(demo)); ss.append(t)
    if not cur:
        raise ValueError(f"no complete {chunk}-step chunks for {task}:{demos}")
    return ChunkRows(torch.stack(cur), torch.stack(fut), torch.stack(tgt) if tgt else None,
                     torch.tensor(uid), dd, ss, task)


def _source_tasks(protocol: dict) -> list[str]:
    return sorted(t for t, s in protocol["tasks"].items() if s["role"] == "source")


def _validation_bank_demos(protocol: dict, task: str, seed: int, m: int = 5) -> list[int]:
    ids = sorted(map(int, protocol["tasks"][task]["source"]["source_train"]))
    random.Random(f"val-bank:{seed}:{task}").shuffle(ids)
    return ids[:m]


def _schedule(protocol: dict, seed: int, updates: int, batch: int) -> list[dict]:
    """One matched query stream for both heads (task, bank demos, query rows)."""
    rng = random.Random(f"v4-source:{seed}")
    tasks = _source_tasks(protocol)
    plan = []
    for _ in range(updates):
        task = rng.choice(tasks)
        demos = sorted(map(int, protocol["tasks"][task]["source"]["source_train"]))
        sizes = [b for b in SUPPORT_BUDGETS if b < len(demos)]
        if not sizes:
            raise ValueError(f"{task} has too few source-training demos for an episodic bank")
        m = rng.choice(sizes)
        bank = rng.sample(demos, m)
        plan.append({"task": task, "bank": bank, "query_demos": [d for d in demos if d not in bank],
                     "rng": rng.getrandbits(32)})
    return plan


def _sample_rows(rows: ChunkRows, batch: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, len(rows), (batch,), generator=gen)


def _validation(protocol, cache, chunk, mean, std, seed, mlp, raid, retrieval, device) -> dict:
    losses = {"chunk_mlp": [], "retrieval_mlp": []}
    mlp.eval(); raid.eval()
    with torch.no_grad():
        for task in _source_tasks(protocol):
            shards = cache["shards"][task]
            val = chunk_rows(shards, protocol["tasks"][task]["source"]["source_validation"], task, chunk, mean, std)
            bank_demos = _validation_bank_demos(protocol, task, seed)
            br = chunk_rows(shards, bank_demos, task, chunk, mean, std)
            bank = make_bank(br.current, br.future, br.target, br.demo_uid, device)
            cur, fut, tgt = val.current.to(device), val.future.to(device), val.target.to(device)
            losses["chunk_mlp"].append(float(chunk_mse(mlp(cur, fut), tgt).mean()))
            block, _ = retrieval_block(bank, cur, fut, spec=retrieval, chunk_steps=chunk)
            losses["retrieval_mlp"].append(float(chunk_mse(raid(cur, fut, block), tgt).mean()))
    mlp.train(); raid.train()
    return {k: float(np.mean(v)) for k, v in losses.items()}


def train_source(protocol: dict, cache: dict, out_dir: str | Path, *, seed: int, device: str = "cpu",
                 updates: int = 2000, batch: int = 64, lr: float = 3e-4, weight_decay: float = 1e-4,
                 validate_every: int = 100, run_mode: str = SCIENTIFIC) -> dict:
    """Source-train the LoRA-arm ChunkMLP and the RAID RetrievalMLP on a matched stream."""
    validate_wm_protocol(protocol)
    design = protocol["design"]
    temporal = TemporalAlignment(**design["temporal"])
    chunk = temporal.chunk_steps
    if run_mode == SCIENTIFIC and (updates, batch) != (design["training"]["source_updates"], design["training"]["batch_size"]):
        raise ValueError("scientific source training uses the fixed protocol settings")
    retrieval = RetrievalSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in design["retrieval"].items()})
    head = HeadSpec(**design["head"])
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    mean, std = source_normalizer(protocol, cache)
    feature_dim = next(iter(next(iter(cache["shards"].values())).values()))["features"].shape[-1]
    mlp = ChunkMLP(chunk, feature_dim, head).to(dev)
    raid = RetrievalMLP(chunk, feature_dim, head, retrieval).to(dev)
    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    opt_raid = torch.optim.AdamW(raid.parameters(), lr=lr, weight_decay=weight_decay)
    rows_cache: dict[tuple, ChunkRows] = {}
    curve, best = [], {"chunk_mlp": (float("inf"), 0, None), "retrieval_mlp": (float("inf"), 0, None)}
    started = time.perf_counter()
    for step, item in enumerate(_schedule(protocol, seed, updates, batch), start=1):
        task, shards = item["task"], cache["shards"][item["task"]]
        qkey = (task, tuple(item["query_demos"]))
        if qkey not in rows_cache:
            rows_cache[qkey] = chunk_rows(shards, item["query_demos"], task, chunk, mean, std)
        q = rows_cache[qkey]
        idx = _sample_rows(q, batch, item["rng"])
        cur, fut, tgt = q.current[idx].to(dev), q.future[idx].to(dev), q.target[idx].to(dev)
        br = chunk_rows(shards, item["bank"], task, chunk, mean, std)
        bank = make_bank(br.current, br.future, br.target, br.demo_uid, dev)
        block, _ = retrieval_block(bank, cur, fut, spec=retrieval, chunk_steps=chunk,
                                   query_demo_ids=q.demo_uid[idx].to(dev))
        for model, opt, pred in ((mlp, opt_mlp, lambda: mlp(cur, fut)), (raid, opt_raid, lambda: raid(cur, fut, block))):
            opt.zero_grad(set_to_none=True)
            loss = chunk_mse(pred(), tgt).mean()
            loss.backward()
            opt.step()
        if step % validate_every == 0 or step == updates:
            scores = _validation(protocol, cache, chunk, mean, std, seed, mlp, raid, retrieval, dev)
            curve.append({"step": step, **scores})
            for name, model in (("chunk_mlp", mlp), ("retrieval_mlp", raid)):
                if scores[name] < best[name][0]:
                    best[name] = (scores[name], step, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    seconds = time.perf_counter() - started
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"format": CHECKPOINT_FORMAT, "protocol": PROTOCOL_V4, "protocol_digest": protocol["digest"],
               "model_digest": protocol["model_digest"], "execution_mode": run_mode, "seed": seed,
               "chunk_steps": chunk, "feature_dim": feature_dim,
               "normalizer_mean": mean, "normalizer_std": std, "head_spec": design["head"],
               "retrieval_spec": design["retrieval"], "updates": updates, "batch": batch, "lr": lr,
               "weight_decay": weight_decay, "curve": curve,
               "heads": {name: {"state_dict": best[name][2], "best_step": best[name][1],
                                "best_validation_chunk_mse": best[name][0]} for name in best},
               "cache_identity": cache["manifest"]["encoder"], "training_seconds": seconds,
               "scientific_result": False}
    path = out / f"source_v4_seed{seed}.pt"
    torch.save(payload, path)
    summary = {k: v for k, v in payload.items() if k not in {"heads", "normalizer_mean", "normalizer_std"}}
    summary["best"] = {n: {"step": b[1], "validation_chunk_mse": b[0]} for n, b in best.items()}
    summary["checkpoint_sha256"] = _sha_file(path)
    _write_json(out / f"source_v4_seed{seed}.json", summary)
    return {"path": str(path), **summary}


def load_source(path: str | Path, protocol: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_digest") != protocol["model_digest"] or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("source checkpoint belongs to another protocol")
    return payload


def build_heads(source: dict, device: str = "cpu") -> tuple[ChunkMLP, RetrievalMLP]:
    head = HeadSpec(**source["head_spec"])
    retrieval = RetrievalSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in source["retrieval_spec"].items()})
    mlp = ChunkMLP(source["chunk_steps"], source["feature_dim"], head)
    raid = RetrievalMLP(source["chunk_steps"], source["feature_dim"], head, retrieval)
    mlp.load_state_dict(source["heads"]["chunk_mlp"]["state_dict"])
    raid.load_state_dict(source["heads"]["retrieval_mlp"]["state_dict"])
    return mlp.to(device).eval(), raid.to(device).eval()


def support_ids(protocol: dict, task: str, budget: int, seed: int) -> list[int]:
    spec = protocol["tasks"][task]
    if spec["role"] == "source" or budget not in SUPPORT_BUDGETS:
        raise ValueError("support sets exist only for held-out tasks and 2/5/10 budgets")
    return list(map(int, spec["support_budgets_by_seed"][str(seed)][str(budget)]))


def raid_target_bank(protocol: dict, cache: dict, source: dict, task: str, budget: int, seed: int,
                     device: str = "cpu") -> tuple[Any, dict]:
    """RAID arm target adaptation: populate the bank with exactly the support set."""
    started = time.perf_counter()
    ids = support_ids(protocol, task, budget, seed)
    rows = chunk_rows(cache["shards"][task], ids, task, source["chunk_steps"],
                      source["normalizer_mean"], source["normalizer_std"])
    bank = make_bank(rows.current, rows.future, rows.target, rows.demo_uid, device)
    seconds = time.perf_counter() - started
    nbytes = sum(t.element_size() * t.nelement() for t in (bank.current, bank.future, bank.actions, bank.keys, bank.demo_ids))
    return bank, {"arm": "raid_bank_mlp", "bank_membership": {"task": task, "support_demo_ids": ids,
                  "source_transitions": 0}, "bank_entries": len(rows), "bank_bytes": int(nbytes),
                  "bank_construction_seconds": seconds, "gradient_updates": 0}


class _SupportChunks(torch.utils.data.Dataset):
    def __init__(self, frames: dict[int, np.ndarray], actions: dict[int, np.ndarray], chunk: int,
                 mean: torch.Tensor, std: torch.Tensor):
        self.items = [(d, t) for d in sorted(frames) for t in range(len(frames[d]) - chunk)]
        self.frames, self.actions, self.chunk, self.mean, self.std = frames, actions, chunk, mean, std

    def __len__(self):
        return len(self.items)

    def get(self, indices: torch.Tensor):
        cur, fut, tgt = [], [], []
        for i in indices.tolist():
            d, t = self.items[i]
            cur.append(self.frames[d][t]); fut.append(self.frames[d][t + self.chunk])
            tgt.append((torch.from_numpy(self.actions[d][t:t + self.chunk]) - self.mean) / self.std)
        return torch.from_numpy(np.stack(cur)), torch.from_numpy(np.stack(fut)), torch.stack(tgt).float()


def adapt_lora(protocol: dict, source: dict, encoder_factory: Callable[[], Any], task: str, budget: int,
               seed: int, out_dir: str | Path, *, device: str = "cpu", run_mode: str = SCIENTIFIC,
               max_updates: int | None = None, snapshots: tuple[int, ...] | None = None) -> dict:
    """LoRA arm: adapt MAE LoRA (blocks 10-11 qkv/proj) plus the action MLP on the support set."""
    design = protocol["design"]
    cfg, lora = design["training"], design["inverse_model"]["lora"]
    max_updates = max_updates or cfg["target_max_updates"]
    snapshots = tuple(snapshots or cfg["target_snapshots"])
    if run_mode == SCIENTIFIC and (max_updates, list(snapshots)) != (cfg["target_max_updates"], cfg["target_snapshots"]):
        raise ValueError("scientific LoRA adaptation uses the fixed protocol schedule")
    dev = torch.device(device)
    torch.manual_seed(seed)
    encoder = encoder_factory()
    names = attach_mae_lora(encoder.mae, lora["blocks"], lora["rank"], lora["alpha"], lora["dropout"])
    mlp, _ = build_heads(source, device)
    mlp.train()
    chunk = source["chunk_steps"]
    ids = support_ids(protocol, task, budget, seed)
    path = protocol["tasks"][task]["path"]
    frames = {d: read_frames(path, d) for d in ids}
    actions = {d: _read_actions(path, d) for d in ids}
    data = _SupportChunks(frames, actions, chunk, source["normalizer_mean"], source["normalizer_std"])
    lora_params = [p for p in encoder.mae.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": mlp.parameters(), "lr": cfg["head_lr"]},
                             {"params": lora_params, "lr": cfg["lora_lr"]}], weight_decay=cfg["weight_decay"])
    batch, micro = cfg["batch_size"], cfg["microbatch_size"]
    gen = torch.Generator().manual_seed(int(_sha_bytes(f"lora:{task}:{budget}:{seed}".encode())[:8], 16))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, started = [], time.perf_counter()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for step in range(1, max_updates + 1):
        opt.zero_grad(set_to_none=True)
        indices = torch.randint(0, len(data), (batch,), generator=gen)
        for part in indices.split(micro):
            cur, fut, tgt = data.get(part)
            feats = encoder.encode_frames(torch.cat((cur, fut)))
            fc, ff = feats[:len(part)], feats[len(part):]
            loss = chunk_mse(mlp(fc, ff), tgt.to(dev)).mean() * (len(part) / batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(list(mlp.parameters()) + lora_params, cfg["grad_clip"])
        opt.step()
        if step in snapshots:
            snap = {"step": step, "head_state": {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()},
                    "lora_state": {k: v.detach().cpu().clone() for k, v in encoder.mae.state_dict().items() if "lora_" in k},
                    "adaptation_seconds": time.perf_counter() - started}
            snap_path = out / f"lora_{task}_b{budget}_s{seed}_u{step}.pt"
            torch.save({**snap, "protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
                        "task": task, "budget": budget, "seed": seed, "support_demo_ids": ids,
                        "lora_targets": names}, snap_path)
            records.append({"step": step, "path": str(snap_path), "sha256": _sha_file(snap_path),
                            "adaptation_seconds": snap["adaptation_seconds"]})
    trainable = sum(p.numel() for p in mlp.parameters()) + sum(p.numel() for p in lora_params)
    summary = {"arm": "lora_mlp", "task": task, "budget": budget, "seed": seed, "support_demo_ids": ids,
               "snapshots": records, "trainable_parameters": int(trainable), "lora_targets": names,
               "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None,
               "execution_mode": run_mode, "scientific_result": False}
    _write_json(out / f"lora_{task}_b{budget}_s{seed}.json", summary)
    return summary


def load_lora_snapshot(path: str | Path, encoder_factory: Callable[[], Any], source: dict, protocol: dict,
                       device: str = "cpu"):
    snap = torch.load(path, map_location="cpu", weights_only=False)
    if snap.get("model_digest") != protocol["model_digest"]:
        raise ValueError("LoRA snapshot belongs to another protocol")
    lora = protocol["design"]["inverse_model"]["lora"]
    encoder = encoder_factory()
    attach_mae_lora(encoder.mae, lora["blocks"], lora["rank"], lora["alpha"], lora["dropout"])
    missing = encoder.mae.load_state_dict(snap["lora_state"], strict=False)
    if missing.unexpected_keys:
        raise ValueError(f"unexpected LoRA keys {missing.unexpected_keys}")
    mlp, _ = build_heads(source, device)
    mlp.load_state_dict(snap["head_state"])
    encoder.mae.eval()
    return encoder, mlp.eval(), snap


def _macro(errors: torch.Tensor, demos: list[int]) -> float:
    per_demo: dict[int, list[float]] = {}
    for value, demo in zip(errors.tolist(), demos):
        per_demo.setdefault(demo, []).append(value)
    return float(np.mean([np.mean(v) for v in per_demo.values()]))


def _append_query_access(ledger: Path, record: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def evaluate_recorded(protocol: dict, cache: dict, source: dict, task: str, budget: int, seed: int, *,
                      lora_snapshot: str | Path, encoder_factory: Callable[[], Any], out_dir: str | Path,
                      device: str = "cpu") -> dict:
    """Recorded-transition diagnostic: inverse-dynamics error on recorded (o_t, o_t+H) pairs."""
    spec = protocol["tasks"][task]
    if spec["role"] == "source":
        raise ValueError("recorded diagnostic evaluates held-out tasks")
    chunk = source["chunk_steps"]
    mean, std = source["normalizer_mean"], source["normalizer_std"]
    retrieval = RetrievalSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in source["retrieval_spec"].items()})
    query = list(map(int, spec["query"]))
    out = Path(out_dir)
    _append_query_access(out / "query_access.jsonl", {"task": task, "budget": budget, "seed": seed,
                         "demos": query, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "purpose": "recorded_transition_diagnostic"})
    labels = {d: _read_actions(spec["path"], d) for d in query}
    rows = chunk_rows(cache["shards"][task], query, task, chunk, mean, std, actions_override=labels)
    target = rows.target.to(device)
    mlp, raid = build_heads(source, device)
    bank, bank_cost = raid_target_bank(protocol, cache, source, task, budget, seed, device)
    results, costs = {}, {"raid_bank_mlp": bank_cost}
    with torch.no_grad():
        cur, fut = rows.current.to(device), rows.future.to(device)
        t0 = time.perf_counter()
        block, _ = retrieval_block(bank, cur, fut, spec=retrieval, chunk_steps=chunk)
        t1 = time.perf_counter()
        pred = {"raid_bank_mlp": raid(cur, fut, block)}
        t2 = time.perf_counter()
        costs["raid_bank_mlp"].update(retrieval_seconds=t1 - t0, inference_seconds=t2 - t1)
        pred["source_mlp_frozen"] = mlp(cur, fut)
        pred["support_knn_average"] = knn_average(bank, cur, fut, spec=retrieval, chunk_steps=chunk)
        encoder, lora_mlp, snap = load_lora_snapshot(lora_snapshot, encoder_factory, source, protocol, device)
        t3 = time.perf_counter()
        feats = {d: _encode(encoder, read_frames(spec["path"], d)) for d in query}
        lc = torch.stack([feats[d][t] for d, t in zip(rows.demo, rows.step)]).to(device)
        lf = torch.stack([feats[d][t + chunk] for d, t in zip(rows.demo, rows.step)]).to(device)
        t4 = time.perf_counter()
        pred["lora_mlp"] = lora_mlp(lc, lf)
        t5 = time.perf_counter()
        costs["lora_mlp"] = {"arm": "lora_mlp", "adaptation_seconds": snap["adaptation_seconds"],
                             "selected_step": snap["step"], "encoding_seconds": t4 - t3,
                             "inference_seconds": t5 - t4, "gradient_updates": snap["step"]}
    for name, value in pred.items():
        err = chunk_mse(value, target)
        first = F.mse_loss(value[:, 0], target[:, 0], reduction="none").mean(-1)
        grip_pred = torch.sign(value[..., 6] * std[6].to(device) + mean[6].to(device))
        grip_true = torch.sign(target[..., 6] * std[6].to(device) + mean[6].to(device))
        results[name] = {"chunk_mse_macro": _macro(err.cpu(), rows.demo),
                         "first_step_mse_macro": _macro(first.cpu(), rows.demo),
                         "gripper_accuracy": float((grip_pred == grip_true).float().mean()),
                         "role": "experimental_arm" if name in ARMS else "diagnostic_control"}
    report = {"protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
              "execution_mode": source.get("execution_mode", SCIENTIFIC),
              "setting": "recorded_transition_diagnostic",
              "task": task, "role": spec["role"], "budget": budget, "seed": seed, "chunk_steps": chunk,
              "query_demos": query, "support_demo_ids": support_ids(protocol, task, budget, seed),
              "transitions": len(rows), "results": results, "costs": costs,
              "note": "recorded-action error; does not evaluate execution of generated plans",
              "scientific_result": False}
    _write_json(out / f"recorded_{task}_b{budget}_s{seed}_u{snap['step']}.json", report)
    return report


def _as_retrieval(payload: dict) -> RetrievalSpec:
    return RetrievalSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in payload.items()})


def evaluate_retention(protocol: dict, cache: dict, source: dict, task: str, budget: int, seed: int, *,
                       lora_snapshot: str | Path, encoder_factory: Callable[[], Any], out_dir: str | Path,
                       device: str = "cpu", run_mode: str = SCIENTIFIC) -> dict:
    """Source-task retention after the target procedure for one held-out condition.

    LoRA is scored with the target adapter active. RAID is scored with the unchanged
    source retrieval MLP and a source-training bank of the same budget: target
    adaptation does not change its weights, and its target bank holds no source
    transitions. This does not read held-out query labels.
    """
    if protocol["tasks"][task]["role"] == "source":
        raise ValueError("retention is recorded per held-out condition, not on a source task")
    chunk = source["chunk_steps"]
    mean, std = source["normalizer_mean"], source["normalizer_std"]
    retrieval = _as_retrieval(source["retrieval_spec"])
    frozen, raid = build_heads(source, device)
    encoder, lora_mlp, snap = load_lora_snapshot(lora_snapshot, encoder_factory, source, protocol, device)
    per_task: dict[str, dict] = {}
    raid_banks = []
    with torch.no_grad():
        for source_task in _source_tasks(protocol):
            spec = protocol["tasks"][source_task]
            retention_ids = list(map(int, spec["source"]["source_retention"]))
            train_ids = sorted(map(int, spec["source"]["source_train"]))
            if set(retention_ids) & set(train_ids):
                raise ValueError(f"{source_task} retention demos overlap source training")
            if run_mode == SCIENTIFIC and budget > len(train_ids):
                raise ValueError(f"{source_task} has fewer source-training demos than budget {budget}")
            bank_size = min(budget, len(train_ids))
            bank_ids = random.Random(f"retention-bank:{seed}:{source_task}:{budget}").sample(train_ids, bank_size)
            shards = cache["shards"][source_task]
            held = chunk_rows(shards, retention_ids, source_task, chunk, mean, std)
            target = held.target.to(device)
            br = chunk_rows(shards, bank_ids, source_task, chunk, mean, std)
            bank = make_bank(br.current, br.future, br.target, br.demo_uid, device)
            block, _ = retrieval_block(bank, held.current.to(device), held.future.to(device), spec=retrieval,
                                       chunk_steps=chunk, query_demo_ids=held.demo_uid.to(device))
            raid_pred = raid(held.current.to(device), held.future.to(device), block)
            feats = {d: _encode(encoder, read_frames(spec["path"], d)) for d in retention_ids}
            lc = torch.stack([feats[d][t] for d, t in zip(held.demo, held.step)]).to(device)
            lf = torch.stack([feats[d][t + chunk] for d, t in zip(held.demo, held.step)]).to(device)
            lora_pred = lora_mlp(lc, lf)
            frozen_pred = frozen(held.current.to(device), held.future.to(device))
            per_task[source_task] = {
                "retention_demos": retention_ids,
                "raid_bank_demo_ids": bank_ids,
                "raid_bank_mlp": _macro(chunk_mse(raid_pred, target).cpu(), held.demo),
                "lora_mlp": _macro(chunk_mse(lora_pred, target).cpu(), held.demo),
                "source_mlp_frozen": _macro(chunk_mse(frozen_pred, target).cpu(), held.demo),
            }
            raid_banks.append({"task": source_task, "demo_ids": bank_ids, "source_train_only": True,
                               "overlaps_retention": bool(set(bank_ids) & set(retention_ids))})
    def _avg(arm: str) -> float:
        return float(np.mean([row[arm] for row in per_task.values()]))
    report = {"protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
              "execution_mode": source.get("execution_mode", run_mode), "setting": "source_retention",
              "condition_task": task, "budget": budget, "seed": seed, "per_source_task": per_task,
              "results": {
                  "raid_bank_mlp": {"chunk_mse_macro": _avg("raid_bank_mlp"), "gradient_updates": 0,
                                    "weights": "unchanged_source_checkpoint",
                                    "bank": "source_training_episodic", "target_adapter_active": False},
                  "lora_mlp": {"chunk_mse_macro": _avg("lora_mlp"), "gradient_updates": snap["step"],
                               "weights": "target_lora_and_head", "target_adapter_active": True,
                               "selected_step": snap["step"]},
                  "source_mlp_frozen": {"chunk_mse_macro": _avg("source_mlp_frozen"),
                                        "role": "diagnostic_control"},
              },
              "raid_banks": raid_banks, "scientific_result": False}
    if any(row["overlaps_retention"] for row in raid_banks):
        raise ValueError("retention bank includes a retention demonstration")
    _write_json(Path(out_dir) / f"retention_{task}_b{budget}_s{seed}_u{snap['step']}.json", report)
    return report
