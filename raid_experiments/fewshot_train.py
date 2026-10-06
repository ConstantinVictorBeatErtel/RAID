"""Source training, few-shot adaptation, and sealed-query evaluation.

This module only accepts horizon-one recorded-RGB caches carrying the exact
source hashes in a prepared fewshot_idm_adaptation_v3 manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
import zipfile
import platform
from contextvars import ContextVar
from functools import wraps
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .fewshot import LIBERO_COMMIT, PROTOCOL, REPETITION_SEEDS, support_budgets_for_seed, validate_goal_protocol
from .evaluation import action_metrics, hierarchical_macro_mse
from .models import DirectDecoder, LowRankLinear, attach_mae_lora, trainable_parameter_names
from .retrieval import TransitionMemory
from src.mae_encoder import MAE_IMPLEMENTATION, MAE_PREPROCESSING, PREPROCESSING_VERSION


METHODS = {"frozen_source", "target_knn", "head_ft", "lora_head_ft"}
UPDATE_SNAPSHOTS = (200, 500, 1000)
FEATURE_DIM = 768
CHECKPOINT_FORMAT = 4
CONTINUATION_FORMAT = 4
EVIDENCE_FORMAT = 4
ARCHITECTURE = {"feature_dim": FEATURE_DIM, "action_dim": 7, "hidden_dim": 512, "bottleneck_dim": 256}


class _SignalStop:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        if __import__("threading").current_thread() is __import__("threading").main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                self.previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
        return self

    def _handle(self, signum, frame):
        self.requested = True

    def __exit__(self, exc_type, exc, tb):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu or --device mps explicitly")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _same_checkpoint_value(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same_checkpoint_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_checkpoint_value(a, b) for a, b in zip(left, right))
    return left == right


def _source_snapshot_files() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.glob("raid_experiments/*.py"), root / "src/mae_encoder.py",
                    root / "src/cache_mae_features.py", root / "src/data_libero.py",
                    root / "configs/fewshot_libero_goal.yaml", root / "requirements-pilot.txt"])
    return {str(path.relative_to(root)): path.read_bytes() for path in paths if path.is_file()}


def _scientific_config() -> dict:
    payload = yaml.safe_load(Path("configs/fewshot_libero_goal.yaml").read_text())
    for key in ("cloud", "output_dir", "name"):
        payload.pop(key, None)
    return payload


def _code_snapshot_digest() -> str:
    files = _source_snapshot_files()
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()
                if name != "configs/fewshot_libero_goal.yaml"}
    manifest["scientific_config"] = hashlib.sha256(json.dumps(
        _scientific_config(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_adaptation_checkpoint(payload: dict, expected: dict, *, scientific: bool) -> None:
    """Shared structural/provenance checks for learned-condition snapshots."""
    keys = ("method", "task", "budget", "seed", "support_ids", "protocol_digest",
            "source_checkpoint_sha256", "source_feature_digest", "feature_digest",
            "code_snapshot_sha256", "selected_step")
    if (not isinstance(payload, dict) or payload.get("checkpoint_format") != CHECKPOINT_FORMAT
            or payload.get("execution_mode") != "scientific"
            or any(payload.get(key) != expected.get(key) for key in keys)
            or payload.get("study_config_sha256") != _config_digest(_scientific_config())
            or not isinstance(payload.get("config_digest"), str)
            or len(payload["config_digest"]) != 64
            or payload.get("architecture") != ARCHITECTURE):
        raise ValueError("adaptation checkpoint identity, version, or architecture is invalid")
    step = payload.get("selected_step")
    planned = payload.get("planned_updates")
    config = payload.get("condition_config")
    if (step not in UPDATE_SNAPSHOTS or payload.get("optimizer_updates") != step
            or planned not in UPDATE_SNAPSHOTS and planned != 1000):
        raise ValueError("adaptation checkpoint is incomplete or has invalid update metadata")
    if (not isinstance(config, dict) or _config_digest(config) != payload.get("config_digest")
            or config.get("method") != payload.get("method") or config.get("task") != payload.get("task")
            or config.get("budget") != payload.get("budget") or config.get("seed") != payload.get("seed")
            or config.get("updates") != planned
            or config.get("execution_mode") != "scientific"
            or config.get("source_checkpoint_sha256") != payload.get("source_checkpoint_sha256")):
        raise ValueError("adaptation configuration provenance is invalid")
    fixed = _scientific_config()["adaptation"]
    expected_config = {"batch_size": fixed["batch_size"], "microbatch_size": fixed["microbatch_size"],
        "head_lr": fixed["head_lr"], "lora_lr": fixed["lora_lr"],
        "weight_decay": fixed["weight_decay"], "grad_clip": fixed["grad_clip"],
        "lora_rank": fixed["lora_rank"], "lora_alpha": fixed["lora_alpha"],
        "lora_dropout": fixed["lora_dropout"], "lora_blocks": fixed["target_blocks"]}
    if (any(config.get(key) != value for key, value in expected_config.items())
            or config.get("mae_checkpoint_sha256") != payload.get("mae_checkpoint_sha256")
            or config.get("source_feature_digest") != payload.get("source_feature_digest")):
        raise ValueError("adaptation configuration differs from the fixed study settings")
    state = payload.get("state_dict")
    try:
        model = DirectDecoder(768, 7)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f"adaptation checkpoint weights are invalid: {exc}") from exc
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("adaptation checkpoint contains non-finite head weights")
    if payload.get("method") == "lora_head_ft":
        adapter = payload.get("adapter_config")
        expected_shapes = {}
        for block in (10, 11):
            for projection, output_dim in (("qkv", 2304), ("proj", 768)):
                expected_shapes[f"blocks.{block}.attn.{projection}.lora_A"] = (8, 768)
                expected_shapes[f"blocks.{block}.attn.{projection}.lora_B"] = (output_dim, 8)
        actual = payload.get("lora_state_dict")
        initial = payload.get("initial_lora_state_dict")
        if (adapter != {"rank": 8, "alpha": 16, "dropout": 0.0,
                        "blocks": [10, 11], "projections": ["qkv", "proj"]}
                or not isinstance(actual, dict) or not isinstance(initial, dict)
                or set(actual) != set(expected_shapes) or set(initial) != set(expected_shapes)
                or any(tuple(actual[k].shape) != shape or tuple(initial[k].shape) != shape
                       or not torch.isfinite(actual[k]).all() or not torch.isfinite(initial[k]).all()
                       for k, shape in expected_shapes.items())
                or not any(not torch.equal(initial[k], actual[k]) for k in expected_shapes)):
            raise ValueError("LoRA checkpoint tensors or adaptation evidence are invalid")


def _validate_adaptation_evidence(root: Path, artifact: Path, expected: dict) -> tuple[dict, dict]:
    """Shared scientific snapshot, condition-completion, and hash validation."""
    if not artifact.is_file():
        raise FileNotFoundError(f"adaptation checkpoint missing: {artifact}")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    _validate_adaptation_checkpoint(payload, expected, scientific=True)
    method = expected["method"]
    condition_path = root / f"{method}.json"
    condition = json.loads(condition_path.read_text()) if condition_path.is_file() else {}
    required_condition = {key: expected.get(key) for key in
        ("method", "task", "budget", "seed", "protocol_digest", "source_checkpoint_sha256",
         "source_feature_digest", "feature_digest", "code_snapshot_sha256")}
    if (condition.get("execution_mode") != "scientific" or condition.get("status") != "complete"
            or condition.get("updates") != condition.get("planned_updates")
            or condition.get("planned_updates") != payload.get("planned_updates")
            or condition.get("config_digest") != payload.get("config_digest")
            or any(condition.get(key) != value for key, value in required_condition.items())):
        raise ValueError("adaptation condition completion identity is invalid")
    completion_path = root / f"{method}.complete.json"
    completion = json.loads(completion_path.read_text()) if completion_path.is_file() else {}
    required_completion = {"evidence_format": EVIDENCE_FORMAT, "status": "complete",
        **required_condition, "planned_updates": payload.get("planned_updates"),
        "config_digest": payload.get("config_digest")}
    step_name = (artifact.name if ".step" in artifact.name else
                 f"{method}.step{payload.get('selected_step')}.pt")
    if (any(completion.get(key) != value for key, value in required_completion.items())
            or completion.get("snapshot_sha256", {}).get(step_name) != _sha(artifact)):
        raise ValueError("hash-bound adaptation completion evidence is missing or invalid")
    return payload, condition


def _write_source_snapshot(output_dir: Path) -> str:
    files = _source_snapshot_files()
    identity = _code_snapshot_digest()
    provenance = output_dir / "_provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    payload = buffer.getvalue()
    snapshot_hash = hashlib.sha256(payload).hexdigest()
    snapshot = provenance / f"source_snapshot_{identity}_{snapshot_hash[:12]}.zip"
    _write_snapshot_bytes(snapshot, payload)
    manifest = {"identity_sha256": identity,
                "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
                "files": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())},
                "libero_commit": LIBERO_COMMIT,
                "mae_implementation": MAE_IMPLEMENTATION,
                "preprocessing_version": PREPROCESSING_VERSION,
                "scientific_config": _scientific_config(),
                "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__},
                "dependency_lock_sha256": hashlib.sha256(Path("requirements-pilot.txt").read_bytes()).hexdigest()
                    if Path("requirements-pilot.txt").is_file() else None}
    manifest_path = provenance / f"source_snapshot_{identity}_{snapshot_hash[:12]}.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("immutable source snapshot manifest does not match current source")
    if not manifest_path.exists():
        temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temp.write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            os.link(temp, manifest_path)
        except FileExistsError:
            if json.loads(manifest_path.read_text()) != manifest:
                raise RuntimeError("immutable source snapshot manifest raced with different content")
        finally:
            temp.unlink(missing_ok=True)
    return manifest["identity_sha256"]


def _load_cache_map(feature_dir: str | Path, protocol: dict, *, target_task: str | None = None,
                    budget: int | None = None, seed: int = 42, include_query: bool = False,
                    include_retention: bool = False, run_mode: str | None = None) -> tuple[dict[str, dict], str]:
    """Load only stage-authorized, content-verified per-demo feature shards."""
    root = Path(feature_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"feature cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format") != 4 or manifest.get("protocol") != PROTOCOL
            or manifest.get("protocol_digest") != protocol["digest"]):
        raise ValueError("feature cache manifest does not belong to this v3 protocol")
    if run_mode is not None:
        expected_cache_mode = "synthetic" if run_mode == "synthetic" else "scientific"
        if manifest.get("execution_mode") != expected_cache_mode:
            raise ValueError(f"{run_mode} execution requires a {expected_cache_mode} feature cache")
    if manifest.get("horizon") != 1 or manifest.get("future_representation") != "recorded_rgb":
        raise ValueError("few-shot study requires horizon-one recorded RGB feature caches")
    if (manifest.get("preprocessing") != MAE_PREPROCESSING
            or not manifest.get("mae_checkpoint_sha256")
            or manifest.get("preprocessing_version") != "mae_vit_b16_resize224_bicubic_imagenet_v1"
            or manifest.get("action_convention") != "libero_recorded_action_at_t_v1"):
        raise ValueError("feature manifest lacks fixed encoder/preprocessing provenance")
    if target_task is not None and (target_task not in protocol["tasks"] or budget not in (2, 5, 10)):
        raise ValueError("target cache access requires a protocol task and 2/5/10 support budget")
    expected_sources = {task: spec["source_sha256"] for task, spec in protocol["tasks"].items()}
    recorded_sources = {str(row.get("task")): row.get("sha256") for row in manifest.get("dataset_files", [])}
    if recorded_sources != expected_sources:
        raise ValueError("cache manifest source data identity differs from the protocol")
    shard_map = manifest.get("demo_shards", {})
    if set(shard_map) != set(protocol["tasks"]):
        raise ValueError("cache shard task coverage differs from protocol")
    encoder_identity = {key: manifest.get(key) for key in
                        ("execution_mode", "horizon", "future_representation", "mae_checkpoint_sha256",
                         "mae_implementation", "preprocessing", "preprocessing_version", "action_convention", "feature_dim",
                         "libero_commit")}
    if (encoder_identity["libero_commit"] != LIBERO_COMMIT
            or encoder_identity["feature_dim"] != FEATURE_DIM
            or encoder_identity["mae_implementation"] != MAE_IMPLEMENTATION
            or encoder_identity["preprocessing_version"] != PREPROCESSING_VERSION):
        raise ValueError("feature cache upstream code identity differs from the pinned study revisions")
    identity_sha = hashlib.sha256(json.dumps(encoder_identity, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    if manifest.get("identity_sha256") != identity_sha:
        raise ValueError("feature cache encoder identity digest is invalid")
    seen_shard_paths = set()
    declared_shard_paths = set()
    for task, spec in protocol["tasks"].items():
        records = shard_map[task]
        if set(map(int, records)) != set(map(int, spec["demo_ids"])):
            raise ValueError(f"cache shard demonstration coverage differs from protocol for {task}")
        for demo_key, record in records.items():
            relative = Path(record.get("path", ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe cache shard path for {task}:{demo_key}")
            if relative.as_posix() in seen_shard_paths:
                raise ValueError(f"duplicate cache shard path: {relative}")
            seen_shard_paths.add(relative.as_posix())
            declared_shard_paths.add(relative.as_posix())
    actual_shard_paths = {p.relative_to(root).as_posix() for p in root.glob("*/*.pt")}
    if actual_shard_paths != declared_shard_paths:
        raise ValueError("feature cache contains orphaned shards or missing declared files")
    rows, used_shards, source_shards, support_shards = {}, [], [], []
    for task, spec in protocol["tasks"].items():
        if spec["role"] == "source":
            allowed = list(spec["source"]["source_train"]) + list(spec["source"]["source_validation"])
            if include_retention:
                allowed += list(spec["source"]["source_retention"])
        elif task == target_task:
            allowed = list(spec["support_budgets_by_seed"][str(seed)][str(budget)])
            if include_query:
                allowed += list(spec["query"])
        else:
            allowed = []
        feat_t, feat_next, actions, lengths, ids = [], [], [], [], []
        offsets = {}; cursor = 0
        for demo in allowed:
            record = shard_map[task].get(str(int(demo)))
            if not record:
                raise ValueError(f"missing cache shard for allowed demonstration {task}:{demo}")
            expected_role = ("query" if int(demo) in set(spec.get("query", [])) else
                             "support_pool" if spec["role"] != "source" else
                             next(role for role, values in spec["source"].items() if int(demo) in values))
            if record.get("role") != expected_role:
                raise ValueError(f"cache shard role mismatch for {task}:{demo}")
            path = root / record["path"]
            if not path.is_file() or _sha(path) != record.get("sha256"):
                raise ValueError(f"missing or stale feature shard {path}")
            # Query artifacts are opened only by evaluation and must not carry labels.
            shard = torch.load(path, map_location="cpu", weights_only=False)
            if (shard.get("protocol_digest") != protocol["digest"] or shard.get("task") != task
                    or int(shard.get("demo_id", -1)) != int(demo) or shard.get("role") != expected_role
                    or shard.get("source_sha256") != spec["source_sha256"]
                    or shard.get("encoder_identity") != {**encoder_identity, "identity_sha256": identity_sha}
                    or record.get("encoder_identity_sha256") != identity_sha):
                raise ValueError(f"feature shard identity mismatch for {task}:{demo}")
            n = int(spec["transitions"][str(int(demo))])
            expected_transition_ids = [f"{task}:{int(demo)}:{t}" for t in range(n)]
            if (shard.get("transition_count") != n or len(shard.get("transition_ids", [])) != n
                    or shard.get("transition_ids") != expected_transition_ids
                    or shard["feat_t"].shape != (n, 768) or shard["feat_next"].shape != (n, 768)
                    or not torch.isfinite(shard["feat_t"]).all() or not torch.isfinite(shard["feat_next"]).all()):
                raise ValueError(f"feature shard transition coverage is invalid for {task}:{demo}")
            feat_t.append(shard["feat_t"].float()); feat_next.append(shard["feat_next"].float())
            if expected_role == "query":
                if shard.get("has_actions") or any(key in shard for key in ("actions", "action", "actions_raw", "targets")):
                    raise ValueError("query feature shard contains forbidden action labels")
            else:
                if (not shard.get("has_actions") or not isinstance(shard.get("actions_raw"), torch.Tensor)
                        or shard["actions_raw"].shape != (n, 7)):
                    raise ValueError(f"support/source cache labels missing for {task}:{demo}")
                if not torch.isfinite(shard["actions_raw"]).all():
                    raise ValueError(f"nonfinite action label in {task}:{demo}")
                actions.append(shard["actions_raw"].float())
            ids.append(int(demo)); lengths.append(n); offsets[int(demo)] = (cursor, cursor+n); cursor += n
            used_shards.append(record["sha256"])
            if spec["role"] == "source" and int(demo) in set(spec["source"]["source_train"] + spec["source"]["source_validation"]):
                source_shards.append((task, int(demo), record["sha256"]))
            if task == target_task and expected_role == "support_pool":
                support_shards.append((task, int(demo), record["sha256"]))
        rows[task] = {"task_name": task, "feat_t": torch.cat(feat_t) if feat_t else torch.empty((0,768)),
                      "feat_next": torch.cat(feat_next) if feat_next else torch.empty((0,768)),
                      "actions_raw": torch.cat(actions) if actions else torch.empty((0,7)),
                      "demo_ids": ids, "demo_lengths": lengths, "_offsets": offsets,
                      "_encoder_identity": encoder_identity}
    source_identity = hashlib.sha256(json.dumps({"protocol": protocol["digest"],
        "cache_execution_mode": manifest.get("execution_mode"), "shards": sorted(source_shards),
        "encoder": encoder_identity}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for row in rows.values():
        row["_source_feature_digest"] = source_identity
    identity = hashlib.sha256(json.dumps({"protocol": protocol["digest"],
        "cache_execution_mode": manifest.get("execution_mode"),
        "shards": sorted(source_shards + support_shards), "encoder": encoder_identity,
        "task": target_task, "budget": budget, "seed": seed},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return rows, source_identity if target_task is None else identity

def _selected_feature_identity(protocol: dict, caches: dict[str, dict], *,
                               target_task: str | None = None, budget: int | None = None,
                               seed: int = 42) -> str:
    payload = {"protocol_digest": protocol["digest"], "encoder_identity":
               next(iter(caches.values()))["_encoder_identity"], "tasks": {}}
    for task, spec in protocol["tasks"].items():
        if spec["role"] == "source":
            demos = list(spec["source"]["source_train"]) + list(spec["source"]["source_validation"])
        elif task == target_task:
            demos = list(spec["support_budgets_by_seed"][str(seed)][str(budget)])
        else:
            demos = []
        cache = caches[task]
        selected_rows = []
        for demo in demos:
            lo, hi = cache["_offsets"][int(demo)]
            selected_rows.append({key: hashlib.sha256(cache[key][lo:hi].contiguous().numpy().tobytes()).hexdigest()
                                  for key in ("feat_t", "feat_next", "actions_raw")})
        payload["tasks"][task] = {"demo_ids": list(map(int, demos)), "rows": selected_rows}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _demo_rows(cache: dict, demo_ids: list[int], mean: torch.Tensor, std: torch.Tensor,
               task_name: str) -> dict[str, torch.Tensor | list]:
    current, future, target, tids, dids, tasks, raw_actions = [], [], [], [], [], [], []
    for demo in demo_ids:
        lo, hi = cache["_offsets"][int(demo)]
        for t in range(hi - lo):
            idx = lo + t
            action = cache["actions_raw"][idx].float()
            current.append(cache["feat_t"][idx].float())
            future.append(cache["feat_next"][idx].float())
            raw_actions.append(action)
            target.append((action - mean) / std)
            tids.append(f"{task_name}:{int(demo)}:{t}")
            dids.append(_demo_uid(task_name, int(demo)))
            tasks.append(task_name)
    return {"current": torch.stack(current), "future": torch.stack(future),
            "action": torch.stack(target), "raw_action": torch.stack(raw_actions),
            "transition_ids": tids, "demo_ids": torch.tensor(dids), "task_ids": tasks}


def _demo_uid(task: str, demo: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{task}\x1f{demo}".encode()).digest()[:8], "big") & ((1 << 63) - 1)


def _source_normalizer(protocol: dict, caches: dict) -> tuple[torch.Tensor, torch.Tensor]:
    actions = []
    for task, spec in protocol["tasks"].items():
        if spec["role"] != "source":
            continue
        cache = caches[task]
        for demo in spec["source"]["source_train"]:
            lo, hi = cache["_offsets"][int(demo)]
            actions.append(cache["actions_raw"][lo:hi].float())
    values = torch.cat(actions, dim=0)
    mean = values.mean(0)
    std = values.std(0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def _write_immutable_json(path: Path, payload: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(payload, default=str)):
            raise FileExistsError(f"refusing to replace immutable evaluation artifact {path}")
        return
    _write_json(path, payload)


_ACTIVE_PROFILE = ContextVar("raid_training_profile", default=None)


def _profile_clock(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _timed_phase(name):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            active = _ACTIVE_PROFILE.get()
            if active is None:
                return function(*args, **kwargs)
            profile, device, _ = active
            started = _profile_clock(device)
            try:
                return function(*args, **kwargs)
            finally:
                profile.setdefault(name, []).append(_profile_clock(device) - started)
        return measured
    return decorate


def _profile_training(function):
    @wraps(function)
    def measured(*args, **kwargs):
        profile = kwargs.get("timing_profile")
        if profile is None:
            return function(*args, **kwargs)
        device = torch.device(kwargs.get("device", "cuda"))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = _profile_clock(device)
        token = _ACTIVE_PROFILE.set((profile, device, started))
        try:
            return function(*args, **kwargs)
        finally:
            profile["total_seconds"] = _profile_clock(device) - started
            profile["peak_memory_bytes"] = (torch.cuda.max_memory_allocated(device)
                                             if device.type == "cuda" else None)
            _ACTIVE_PROFILE.reset(token)
    return measured


def _profile_ready():
    active = _ACTIVE_PROFILE.get()
    if active:
        profile, device, started = active
        profile["initialization_seconds"] = _profile_clock(device) - started


@_timed_phase("checkpoint_write_seconds")
def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _profile_update_start(profile: dict | None, step: int, device: torch.device):
    if profile is None:
        return None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter(), step <= int(profile.get("warmup_updates", 0))


def _profile_update_end(profile: dict | None, started: tuple | None, device: torch.device) -> None:
    if profile is None or started is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timestamp, warmup = started
    profile.setdefault("warmup_update_seconds" if warmup else "measured_update_seconds", []).append(
        time.perf_counter() - timestamp)


@_timed_phase("checkpoint_serialization_seconds")
def _snapshot_bytes(payload: dict) -> bytes:
    # Pickle memoizes object identity, including strings. An unpickled training
    # curve followed by new entries has a different alias graph from an
    # uninterrupted curve, even when every value and tensor is identical.
    # Normalize that graph before FIRST publication; recovery always copies the
    # already archived bytes and never serializes them again.
    def canonical(value):
        if isinstance(value, str):
            return sys.intern(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous().clone()
        if isinstance(value, dict):
            return {canonical(key): canonical(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, tuple):
            return tuple(canonical(item) for item in value)
        return value
    buffer = io.BytesIO()
    torch.save(canonical(payload), buffer)
    return buffer.getvalue()


@_timed_phase("snapshot_publish_seconds")
def _write_snapshot_bytes(path: Path, content: bytes) -> str:
    """Publish a checkpoint blob once and never replace its referenced bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        if _sha(path) != digest:
            raise FileExistsError(f"refusing to replace immutable snapshot {path}")
        return digest
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(name, path)
        except FileExistsError:
            if _sha(path) != digest:
                raise FileExistsError(f"refusing to replace immutable snapshot {path}")
    finally:
        Path(name).unlink(missing_ok=True)
    return digest


@_timed_phase("source_validation_seconds")
def _validation_score(model: DirectDecoder, batch: dict, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(batch["current"].to(device), batch["future"].to(device))[0].cpu()
    row_mse = (pred - batch["action"]).pow(2).mean(-1).numpy()
    return hierarchical_macro_mse(row_mse, batch["task_ids"], batch["demo_ids"].tolist())


@_profile_training
def train_source(protocol_path: str | Path, feature_dir: str | Path, output_dir: str | Path,
                 *, seed: int = 42, updates: int = 2000, stop_after_updates: int | None = None,
                 device: str = "cuda", hidden: int = 512, bottleneck: int = 256,
                 batch_size: int = 64, microbatch_size: int = 8,
                 head_lr: float = 3e-4, weight_decay: float = 1e-4,
                 resume_checkpoint: str | Path | None = None,
                 run_mode: str = "scientific", timing_profile: dict | None = None) -> dict:
    """Fit the direct source IDM on the six preregistered source tasks."""
    if run_mode not in ("scientific", "calibration", "synthetic"):
        raise ValueError("run_mode must be scientific, calibration, or synthetic")
    if run_mode == "scientific" and updates != 2000:
        raise ValueError("scientific source training is fixed at 2,000 updates; use calibration/synthetic mode for short runs")
    if run_mode == "scientific":
        fixed = _scientific_config()["source_training"]
        if (hidden != 512 or bottleneck != 256 or batch_size != fixed["batch_size"]
                or microbatch_size != fixed["microbatch_size"] or head_lr != fixed["head_lr"]
                or weight_decay != fixed["weight_decay"]):
            raise ValueError("scientific source training settings must match the versioned study configuration")
    protocol = validate_goal_protocol(protocol_path)
    if seed not in REPETITION_SEEDS:
        raise ValueError(f"source seed must be one of {REPETITION_SEEDS}")
    caches, feature_digest = _load_cache_map(feature_dir, protocol, run_mode=run_mode)
    mean, std = _source_normalizer(protocol, caches)
    train_rows, val_rows = [], []
    for task, spec in protocol["tasks"].items():
        if spec["role"] != "source":
            continue
        train_rows.append(_demo_rows(caches[task], spec["source"]["source_train"], mean, std, task))
        val_rows.append(_demo_rows(caches[task], spec["source"]["source_validation"], mean, std, task))
    train = {k: torch.cat([r[k] for r in train_rows]) if k in ("current", "future", "action", "raw_action", "demo_ids") else sum((r[k] for r in train_rows), []) for k in train_rows[0]}
    val = {k: torch.cat([r[k] for r in val_rows]) if k in ("current", "future", "action", "raw_action", "demo_ids") else sum((r[k] for r in val_rows), []) for k in val_rows[0]}
    if batch_size % microbatch_size:
        raise ValueError("batch_size must be divisible by microbatch_size")
    device = _resolve_device(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = DirectDecoder(train["current"].shape[-1], train["action"].shape[-1], hidden, bottleneck).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=head_lr, weight_decay=weight_decay)
    config_digest = _config_digest({"execution_mode": run_mode, "seed": seed, "updates": updates, "hidden": hidden, "bottleneck": bottleneck,
                                    "batch": batch_size, "microbatch": microbatch_size, "lr": head_lr,
                                    "weight_decay": weight_decay})
    source_config = {"execution_mode": run_mode, "seed": seed, "updates": updates,
        "hidden": hidden, "bottleneck": bottleneck, "batch": batch_size,
        "microbatch": microbatch_size, "lr": head_lr, "weight_decay": weight_decay}
    target_updates = min(updates, stop_after_updates) if stop_after_updates is not None else updates
    if target_updates < 1:
        raise ValueError("stop_after_updates must be positive")
    schedule_rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train["action"]), generator=schedule_rng).tolist(); cursor = 0
    curve, best, best_step, best_state, start_step = [], float("inf"), 0, None, 0
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    code_digest = _write_source_snapshot(out)
    normalizer_path = out / "action_normalizer.pt"
    normalizer_payload = {"mean": mean, "std": std, "source_tasks": list(protocol["task_roles"]["source"]),
                          "protocol_digest": protocol["digest"]}
    normalizer_payload["normalizer_sha256"] = hashlib.sha256(
        mean.contiguous().numpy().tobytes() + std.contiguous().numpy().tobytes()).hexdigest()
    _atomic_save(normalizer_payload, normalizer_path)
    bound_study_root = str(out.resolve())
    resumed = None
    if resume_checkpoint is not None:
        resumed = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if (resumed.get("continuation_format") != CONTINUATION_FORMAT
                or resumed.get("execution_mode") != run_mode
                or resumed.get("protocol_digest") != protocol["digest"] or resumed.get("feature_digest") != feature_digest
                or resumed.get("config_digest") != config_digest or resumed.get("code_snapshot_sha256") != code_digest):
            raise ValueError("source continuation identity does not match protocol or feature cache")
        bound_study_root = resumed.get("study_root")
        if (not isinstance(bound_study_root, str) or not Path(bound_study_root).is_absolute()
                or str(Path(bound_study_root).resolve()) != bound_study_root):
            raise ValueError("source continuation has no canonical study-root binding")
        model.load_state_dict(resumed["state_dict"]); optimizer.load_state_dict(resumed["optimizer"])
        start_step = int(resumed["step"]); cursor = int(resumed["cursor"]); order = list(resumed["order"])
        schedule_rng.set_state(resumed["schedule_rng_state"].cpu())
        torch.set_rng_state(resumed["torch_rng_state"].cpu()); np.random.set_state(resumed["numpy_rng_state"])
        random.setstate(resumed["python_rng_state"])
        if torch.cuda.is_available() and resumed.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all([state.cpu() for state in resumed["cuda_rng_state"]])
        curve = resumed.get("curve", []); best = float(resumed.get("best_validation_mse", float("inf")))
        best_step = int(resumed.get("best_step", 0)); best_state = resumed.get("best_state_dict")
        if start_step > target_updates:
            raise ValueError("continuation is already beyond requested total updates")
    started = time.time()
    _profile_ready()
    with _SignalStop() as signal_stop:
        for step in range(start_step + 1, target_updates + 1):
            update_started = _profile_update_start(timing_profile, step, device)
            model.train(); optimizer.zero_grad(set_to_none=True); loss_acc = 0.0
            for _ in range(batch_size // microbatch_size):
                idxs = []
                while len(idxs) < microbatch_size:
                    if cursor == len(order):
                        order = torch.randperm(len(train["action"]), generator=schedule_rng).tolist(); cursor = 0
                    take = min(microbatch_size - len(idxs), len(order) - cursor)
                    idxs.extend(order[cursor:cursor + take]); cursor += take
                ix = torch.tensor(idxs)
                pred, _ = model(train["current"][ix].to(device), train["future"][ix].to(device))
                loss = torch.nn.functional.mse_loss(pred, train["action"][ix].to(device))
                (loss / (batch_size // microbatch_size)).backward(); loss_acc += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            _profile_update_end(timing_profile, update_started, device)
            if step % 100 == 0 or step == updates:
                score = _validation_score(model, val, device)
                curve.append({"step": step, "train_mse": loss_acc / (batch_size // microbatch_size), "validation_mse": score})
                if score < best:
                    best, best_step = score, step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    _atomic_save({"protocol": PROTOCOL, "protocol_digest": protocol["digest"], "feature_digest": feature_digest,
                                  "config_digest": config_digest,
                                  "code_snapshot_sha256": code_digest,
                                  "seed": seed, "state_dict": best_state,
                                  "architecture": {"feature_dim": int(train["current"].shape[-1]),
                                                   "action_dim": int(train["action"].shape[-1]),
                                                   "hidden_dim": hidden, "bottleneck_dim": bottleneck},
                                  "best_step": best_step, "best_validation_mse": best, "normalizer_mean": mean,
                                  "normalizer_std": std, "curve": curve}, out / f"source_seed{seed}.validation_candidate.pt")
            if step % 100 == 0 or step == target_updates or signal_stop.requested:
                _atomic_save({"continuation_format": CONTINUATION_FORMAT, "execution_mode": run_mode,
                              "study_root": bound_study_root,
                              "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                              "cursor": cursor, "order": order, "schedule_rng_state": schedule_rng.get_state(),
                              "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
                              "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                              "python_rng_state": random.getstate(), "best_state_dict": best_state,
                              "best_step": best_step, "best_validation_mse": best, "curve": curve,
                              "protocol_digest": protocol["digest"], "feature_digest": feature_digest,
                              "config_digest": config_digest, "code_snapshot_sha256": code_digest}, out / f"source_seed{seed}.last.pt")
            if signal_stop.requested:
                break
    if signal_stop.requested and step < target_updates:
        result = {"status": "interrupted", "execution_mode": run_mode, "protocol": PROTOCOL,
                  "seed": seed, "updates": step, "planned_updates": updates,
                  "feature_digest": feature_digest, "code_snapshot_sha256": code_digest,
                  "continuation_checkpoint": str(out / f"source_seed{seed}.last.pt")}
        _write_json(out / f"source_seed{seed}.json", result)
        return result
    if best_state is None:
        if target_updates < updates:
            result = {"status": "calibration", "protocol": PROTOCOL, "seed": seed,
                      "updates": target_updates, "requested_updates": updates,
                      "feature_digest": feature_digest, "code_snapshot_sha256": code_digest,
                      "continuation_checkpoint": str(out / f"source_seed{seed}.last.pt"),
                      "selection": "no source-validation candidate reached the fixed schedule"}
            _write_json(out / f"source_seed{seed}.json", result)
            return result
        raise RuntimeError("no scheduled source-validation checkpoint was selected")
    scientific_complete = run_mode == "scientific" and target_updates == 2000 and updates == 2000
    best_payload = {"checkpoint_format": CHECKPOINT_FORMAT,
                    "study_root": bound_study_root,
                    "study_config_sha256": _config_digest(_scientific_config()),
                    "protocol": PROTOCOL, "protocol_digest": protocol["digest"],
                    "feature_digest": feature_digest, "source_feature_digest": feature_digest,
                    "config_digest": config_digest, "code_snapshot_sha256": code_digest,
                    "source_config": source_config,
                    "seed": seed, "state_dict": best_state, "best_step": best_step,
                    "execution_mode": run_mode, "training_updates": target_updates,
                    "planned_updates": updates, "training_complete": target_updates == updates,
                    "scientific_complete": scientific_complete,
                    "architecture": {"feature_dim": int(train["current"].shape[-1]),
                                     "action_dim": int(train["action"].shape[-1]),
                                     "hidden_dim": hidden, "bottleneck_dim": bottleneck},
                    "best_validation_mse": best, "normalizer_mean": mean, "normalizer_std": std,
                    "curve": curve}
    source_path = out / f"source_seed{seed}.pt"
    archived = resumed.get("source_snapshot") if resumed and start_step == target_updates else None
    if archived is not None:
        source_bytes = archived.get("content")
        if (not isinstance(source_bytes, bytes) or hashlib.sha256(source_bytes).hexdigest() != archived.get("sha256")
                or not _same_checkpoint_value(torch.load(io.BytesIO(source_bytes), map_location="cpu", weights_only=False), best_payload)):
            raise ValueError("source continuation archived snapshot is corrupt or conflicts with training evidence")
    else:
        source_bytes = _snapshot_bytes(best_payload)
    source_sha = _write_snapshot_bytes(source_path, source_bytes)
    source_completion = {"evidence_format": EVIDENCE_FORMAT, "status": "complete",
        "checkpoint_sha256": source_sha, "protocol_digest": protocol["digest"],
        "seed": seed, "source_feature_digest": feature_digest,
        "code_snapshot_sha256": code_digest, "config_digest": config_digest,
        "planned_updates": updates, "training_updates": target_updates,
        "best_step": best_step, "best_validation_mse": best,
        "execution_mode": run_mode}
    _write_immutable_json(out / f"source_seed{seed}.complete.json", source_completion)
    # Preserve the exact first-published bytes so completed source recovery into
    # a new directory neither reserializes the snapshot nor changes its ledger.
    continuation_path = out / f"source_seed{seed}.last.pt"
    continuation = (torch.load(continuation_path, map_location="cpu", weights_only=False)
                    if continuation_path.is_file() else resumed)
    continuation = dict(continuation, source_snapshot={"content": source_bytes, "sha256": source_sha})
    _atomic_save(continuation, continuation_path)
    result = {"status": "complete" if scientific_complete else run_mode,
              "protocol": PROTOCOL,
              "seed": seed, "updates": target_updates, "requested_updates": updates, "best_step": best_step,
              "best_validation_mse": best, "seconds": time.time() - started,
              "trainable_parameters": sum(p.numel() for p in model.parameters()), "feature_digest": feature_digest,
              "code_snapshot_sha256": code_digest,
              "source_checkpoint_sha256": source_sha}
    _write_json(out / f"source_seed{seed}.json", result)
    return result


class RawSelectedRows(Dataset):
    """Lazy RGB/action rows from only one role-approved demo set."""
    def __init__(self, task_path: str | Path, task: str, demo_ids: list[int], mean: torch.Tensor, std: torch.Tensor):
        self.path, self.task, self.mean, self.std = str(task_path), task, mean.numpy(), std.numpy()
        self.items = []
        with h5py.File(self.path, "r") as h:
            for demo in demo_ids:
                g = h[f"data/demo_{int(demo)}"]
                obs = g["obs"]
                key = next((x for x in ("agentview_rgb", "agentview_image", "image") if x in obs), None)
                if key is None:
                    raise ValueError(f"{task} demo {demo} has no agent RGB")
                n = min(len(g["actions"]), len(obs[key])) - 1
                for timestep in range(n):
                    self.items.append((int(demo), timestep, key))

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        demo, timestep, image_key = self.items[index]
        with h5py.File(self.path, "r") as h:
            group = h[f"data/demo_{demo}"]
            image_t = np.asarray(group[f"obs/{image_key}"][timestep])
            image_next = np.asarray(group[f"obs/{image_key}"][timestep + 1])
            action = np.asarray(group["actions"][timestep], dtype=np.float32)
        return {"image_t": torch.from_numpy(image_t), "image_next": torch.from_numpy(image_next),
                "action": torch.from_numpy(((action - self.mean) / self.std).astype(np.float32)),
                "demo": demo, "timestep": timestep,
                "transition_id": f"{self.task}:{demo}:{timestep}"}


def _load_encoder_with_lora(mae_checkpoint: str, device: torch.device,
                            blocks: tuple[int, ...] = (10, 11), rank: int = 8,
                            alpha: int = 16, dropout: float = 0.0,
                            **unused):
    from src.mae_encoder import MAEImageEncoder
    encoder = MAEImageEncoder.from_checkpoint(mae_checkpoint, device)
    names = attach_mae_lora(encoder.mae, blocks, rank, alpha, dropout)
    actual = set(trainable_parameter_names(encoder.mae))
    expected = {f"blocks.{block}.attn.{projection}.lora_{side}"
                for block in blocks for projection in ("qkv", "proj") for side in ("A", "B")}
    if set(names) != {f"blocks.{block}.attn.{projection}" for block in blocks for projection in ("qkv", "proj")} or actual != expected:
        raise RuntimeError(f"LoRA allowlist mismatch: {actual}")
    return encoder, names


def _encode_live(encoder, images: torch.Tensor) -> torch.Tensor:
    return encoder.encode_frames(images)


def _validate_source_checkpoint_payload(payload: dict, protocol: dict, seed: int, *,
                                        feature_digest: str | None = None,
                                        checkpoint_path: str | Path | None = None,
                                        scientific: bool = True) -> None:
    if not isinstance(payload, dict) or payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError("source checkpoint format is invalid")
    if scientific and (payload.get("execution_mode") != "scientific"
            or payload.get("study_config_sha256") != _config_digest(_scientific_config())
            or payload.get("scientific_complete") is not True
            or payload.get("training_complete") is not True
            or payload.get("training_updates") != 2000 or payload.get("planned_updates") != 2000):
        raise ValueError("scientific use requires a completed 2,000-update scientific source checkpoint")
    if (payload.get("protocol") != PROTOCOL or payload.get("protocol_digest") != protocol["digest"]
            or int(payload.get("seed", -1)) != int(seed)):
        raise ValueError(f"source checkpoint protocol or seed identity is invalid for seed {seed}")
    if feature_digest is not None and payload.get("feature_digest") != feature_digest:
        raise ValueError("source checkpoint was trained with different feature caches")
    if not isinstance(payload.get("feature_digest"), str) or len(payload["feature_digest"]) != 64:
        raise ValueError("source checkpoint feature-cache digest is invalid")
    if payload.get("source_feature_digest") != payload.get("feature_digest"):
        raise ValueError("source feature-cache identity is invalid")
    source_config = payload.get("source_config")
    if (not isinstance(source_config, dict) or _config_digest(source_config) != payload.get("config_digest")
            or source_config.get("seed") != seed
            or source_config.get("execution_mode") != payload.get("execution_mode")):
        raise ValueError("source training configuration provenance is invalid")
    if scientific:
        expected_config = {"execution_mode": "scientific", "seed": seed, "updates": 2000,
            "hidden": 512, "bottleneck": 256, "batch": 64, "microbatch": 8,
            "lr": 0.0003, "weight_decay": 0.0001}
        if source_config != expected_config:
            raise ValueError("source training configuration differs from the fixed study settings")
    if payload.get("code_snapshot_sha256") != _code_snapshot_digest():
        raise ValueError("source checkpoint code snapshot differs from the current effective source tree")
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError("source checkpoint architecture does not match the fixed inverse-dynamics model")
    study_root = payload.get("study_root")
    if (not isinstance(study_root, str) or not Path(study_root).is_absolute()
            or str(Path(study_root).resolve()) != study_root):
        raise ValueError("source checkpoint has no canonical study-root binding")
    mean, std = payload.get("normalizer_mean"), payload.get("normalizer_std")
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor) or mean.shape != (7,) or std.shape != (7,):
        raise ValueError("source checkpoint has no valid embedded action normalizer")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("source checkpoint action normalizer is invalid")
    curve = payload.get("curve")
    best_step = payload.get("best_step")
    best_score = payload.get("best_validation_mse")
    if (not isinstance(curve, list) or not curve or best_step not in {item.get("step") for item in curve}
            or not isinstance(best_score, (int, float)) or not math.isfinite(float(best_score))):
        raise ValueError("source checkpoint validation selection evidence is incomplete")
    model = DirectDecoder(768, 7)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:
        raise ValueError(f"source checkpoint weights are invalid: {exc}") from exc
    if any(not torch.isfinite(tensor).all() for tensor in payload["state_dict"].values()):
        raise ValueError("source checkpoint contains non-finite weights")
    if checkpoint_path is not None:
        if scientific:
            completion_path = Path(checkpoint_path).parent / f"source_seed{seed}.complete.json"
            completion = json.loads(completion_path.read_text()) if completion_path.is_file() else {}
            expected_completion = {"evidence_format": EVIDENCE_FORMAT, "status": "complete",
                "checkpoint_sha256": _sha(Path(checkpoint_path)),
                "protocol_digest": protocol["digest"], "seed": seed,
                "source_feature_digest": payload["source_feature_digest"],
                "code_snapshot_sha256": payload["code_snapshot_sha256"],
                "config_digest": payload["config_digest"], "planned_updates": 2000,
                "training_updates": 2000, "best_step": payload["best_step"],
                "best_validation_mse": payload["best_validation_mse"],
                "execution_mode": "scientific"}
            if any(completion.get(key) != value for key, value in expected_completion.items()):
                raise ValueError("source checkpoint completion receipt is missing or hash-mismatched")
        sidecar_path = Path(checkpoint_path).parent / "action_normalizer.pt"
        if sidecar_path.exists():
            sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
            expected = hashlib.sha256(mean.cpu().contiguous().numpy().tobytes() + std.cpu().contiguous().numpy().tobytes()).hexdigest()
            if (sidecar.get("normalizer_sha256") != expected or not torch.equal(sidecar.get("mean"), mean.cpu())
                    or not torch.equal(sidecar.get("std"), std.cpu())):
                raise ValueError("action normalizer sidecar differs from the source checkpoint")


def _load_source(source_checkpoint: str | Path, protocol: dict, caches: dict, device: torch.device,
                 *, run_mode: str = "scientific"):
    payload = torch.load(source_checkpoint, map_location=device, weights_only=False)
    expected_digest = next(iter(caches.values())).get("_source_feature_digest")
    _validate_source_checkpoint_payload(payload, protocol, payload.get("seed", -1),
        feature_digest=expected_digest, checkpoint_path=source_checkpoint,
        scientific=run_mode == "scientific")
    return payload


def _support_memory(cache: dict, task: str, demo_ids: list[int], mean: torch.Tensor, std: torch.Tensor,
                    device: torch.device, source_rows: list[dict] | None = None) -> TransitionMemory:
    rows = _demo_rows(cache, demo_ids, mean, std, task)
    memory = TransitionMemory(rows["current"].shape[-1], rows["action"].shape[-1], device)
    current, future, actions, ids = rows["current"], rows["future"], rows["action"], rows["demo_ids"]
    if source_rows:
        current = torch.cat([r["current"] for r in source_rows] + [current])
        future = torch.cat([r["future"] for r in source_rows] + [future])
        actions = torch.cat([r["action"] for r in source_rows] + [actions])
        ids = torch.cat([r["demo_ids"] for r in source_rows] + [ids])
    memory.extend(current.to(device), future.to(device), actions.to(device), ids.to(device))
    return memory


@_profile_training
def adapt_target(protocol_path: str | Path, feature_dir: str | Path, source_checkpoint: str | Path,
                 output_dir: str | Path, *, task: str, method: str, budget: int, seed: int = 42,
                 updates: int = 1000, stop_after_updates: int | None = None, device: str = "cuda",
                 mae_checkpoint: str | None = None,
                 batch_size: int = 64, microbatch_size: int = 8, selected_step: int | None = None,
                 freeze_file: str | Path | None = None,
                 confirmation_file: str | Path | None = None,
                 resume_checkpoint: str | Path | None = None,
                 run_mode: str = "scientific",
                 head_lr: float = 3e-4, lora_lr: float = 1e-4, weight_decay: float = 1e-4,
                 grad_clip: float = 1.0, lora_rank: int = 8, lora_alpha: int = 16,
                 lora_dropout: float = 0.0, lora_blocks: tuple[int, ...] = (10, 11),
                 timing_profile: dict | None = None) -> dict:
    if method not in METHODS:
        raise ValueError(f"unsupported method {method}; expected {sorted(METHODS)}")
    if run_mode not in ("scientific", "calibration", "synthetic"):
        raise ValueError("run_mode must be scientific, calibration, or synthetic")
    if run_mode == "scientific":
        fixed = _scientific_config()["adaptation"]
        if (method not in ("head_ft", "lora_head_ft") or batch_size != fixed["batch_size"]
                or microbatch_size != fixed["microbatch_size"] or head_lr != fixed["head_lr"]
                or lora_lr != fixed["lora_lr"] or weight_decay != fixed["weight_decay"]
                or grad_clip != fixed["grad_clip"] or lora_rank != fixed["lora_rank"]
                or lora_alpha != fixed["lora_alpha"] or lora_dropout != fixed["lora_dropout"]
                or list(lora_blocks) != fixed["target_blocks"]):
            raise ValueError("scientific adaptation settings must match the versioned study configuration")
    if run_mode == "scientific" and method in ("head_ft", "lora_head_ft") and updates not in UPDATE_SNAPSHOTS:
        raise ValueError("scientific adaptation candidates must use 200, 500, or 1,000 updates")
    protocol = validate_goal_protocol(protocol_path); caches, feature_digest = _load_cache_map(
        feature_dir, protocol, target_task=task, budget=budget, seed=seed, run_mode=run_mode)
    spec = protocol["tasks"].get(task)
    if spec is None or spec["role"] not in ("development", "final_test"):
        raise ValueError("adapt_target requires a preregistered held-out task")
    if budget not in (2, 5, 10):
        raise ValueError("support budget must be one of 2, 5, 10")
    if batch_size % microbatch_size:
        raise ValueError("batch_size must be divisible by microbatch_size")
    if spec["role"] == "final_test":
        final_record = confirmation_file or freeze_file
        if not final_record or not Path(final_record).is_file():
            raise RuntimeError("final-test adaptation requires a registered confirmation record")
        frozen = json.loads(Path(final_record).read_text())
        body = dict(frozen); claimed = body.pop("confirmation_digest", None)
        expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (frozen.get("protocol_digest") != protocol["digest"] or claimed != expected
                or frozen.get("confirmation_registered") is not True
                or frozen.get("source_checkpoints", {}).get(str(seed)) != _sha(Path(source_checkpoint))
                or budget not in frozen.get("confirmation_scope", {}).get("support_budgets", [])):
            raise RuntimeError("final-test adaptation confirmation record is invalid")
        locked_step = frozen.get("selected_updates", {}).get(method)
        if method in ("head_ft", "lora_head_ft"):
            if locked_step not in UPDATE_SNAPSHOTS:
                raise RuntimeError(f"freeze record has no selected update count for {method}")
            if selected_step is not None and selected_step != locked_step:
                raise ValueError("requested adaptation step differs from the frozen development setting")
            selected_step = int(locked_step)
    cache_manifest = json.loads((Path(feature_dir) / "manifest.json").read_text())
    for checkpoint, key in ((mae_checkpoint, "mae_checkpoint_sha256"),):
        if checkpoint and _sha(Path(checkpoint)) != cache_manifest.get(key):
            raise ValueError(f"{key.removesuffix('_sha256')} does not match the frozen-feature encoder")
    if seed not in REPETITION_SEEDS:
        raise ValueError(f"adaptation seed must be one of {REPETITION_SEEDS}")
    support = list(map(int, spec["support_budgets_by_seed"][str(seed)][str(budget)]))
    source_payload = _load_source(source_checkpoint, protocol, caches, torch.device("cpu"), run_mode=run_mode)
    if int(source_payload.get("seed", -1)) != protocol["repetitions"][str(seed)]["source_seed"]:
        raise ValueError("source checkpoint seed does not match this paired repetition")
    mean, std = source_payload["normalizer_mean"].cpu(), source_payload["normalizer_std"].cpu()
    normalizer_path = Path(source_checkpoint).parent / "action_normalizer.pt"
    if normalizer_path.exists():
        sidecar = torch.load(normalizer_path, map_location="cpu", weights_only=False)
        normalizer_sha = hashlib.sha256(mean.contiguous().numpy().tobytes() + std.contiguous().numpy().tobytes()).hexdigest()
        if (sidecar.get("normalizer_sha256") != normalizer_sha
                or not torch.equal(sidecar.get("mean"), mean) or not torch.equal(sidecar.get("std"), std)):
            raise ValueError("action normalizer sidecar differs from the source checkpoint")
    device = _resolve_device(device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    source = source_payload
    source_feature_digest = str(source_payload["source_feature_digest"])
    model = DirectDecoder(768, 7).to(device); model.load_state_dict(source["state_dict"])
    model.eval()
    out = Path(output_dir) / task / f"{budget}shot" / f"seed{seed}"; out.mkdir(parents=True, exist_ok=True)
    started = time.time(); best_step = 0; curve = []; start_step = 0
    if run_mode == "scientific" and spec["role"] == "development" and method in ("head_ft", "lora_head_ft") and updates != 1000:
        raise ValueError("scientific development adaptation must complete the shared 1,000-update schedule")
    planned_updates = (int(selected_step) if spec["role"] == "final_test"
                       and method in ("head_ft", "lora_head_ft") else int(updates))
    condition_config = {"execution_mode": run_mode, "method": method, "task": task, "budget": budget,
        "seed": seed, "updates": planned_updates, "batch_size": batch_size,
        "microbatch_size": microbatch_size, "source_checkpoint_sha256": _sha(Path(source_checkpoint)),
        "source_feature_digest": source_feature_digest,
        "head_lr": head_lr, "lora_lr": lora_lr, "weight_decay": weight_decay,
        "grad_clip": grad_clip, "lora_rank": lora_rank, "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout, "lora_blocks": list(lora_blocks),
        "mae_checkpoint_sha256": _sha(Path(mae_checkpoint)) if mae_checkpoint else None}
    config_digest = _config_digest(condition_config)
    adapter_config = {"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout,
                      "blocks": list(lora_blocks), "projections": ["qkv", "proj"]}
    cap = min(planned_updates, stop_after_updates) if stop_after_updates is not None else planned_updates
    if cap < 1: raise ValueError("adaptation update count must be positive")
    code_digest = _write_source_snapshot(Path(output_dir))
    lora_encoder = None; lora_names = []; initial_lora_state = None; frozen_encoder_state = None
    if method == "lora_head_ft":
        if not mae_checkpoint:
            raise ValueError("lora_head_ft requires --mae-ckpt")
        lora_encoder, lora_names = _load_encoder_with_lora(mae_checkpoint, device,
                                                           lora_blocks, lora_rank, lora_alpha, lora_dropout)
        initial_lora_state = {k: v.detach().cpu().clone() for k, v in lora_encoder.mae.state_dict().items()
                              if "lora_" in k}
        frozen_encoder_state = {k: v.detach().cpu().clone() for k, v in lora_encoder.mae.state_dict().items()
                                if "lora_" not in k}
        raw_train = RawSelectedRows(spec["path"], task, support, mean, std)
        order_rng = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.AdamW([{"params": model.parameters(), "lr": head_lr},
                                       {"params": [p for p in lora_encoder.mae.parameters() if p.requires_grad], "lr": lora_lr}], weight_decay=weight_decay)
    elif method == "head_ft":
        rows = _demo_rows(caches[task], support, mean, std, task)
        order_rng = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=head_lr, weight_decay=weight_decay)
    else:
        initial_lora_state = None
        optimizer = None; rows = _demo_rows(caches[task], support, mean, std, task)
    if method in ("head_ft", "lora_head_ft"):
        resumed = None; latest_snapshot = None; snapshot_archive = {}; snapshot_archive_bytes = {}
        max_lora_gradient_norm = 0.0
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if method == "lora_head_ft":
            if batch_size % microbatch_size: raise ValueError("batch size must divide microbatch size")
            order = torch.randperm(len(raw_train), generator=order_rng).tolist(); cursor = 0
        else:
            order = torch.randperm(len(rows["action"]), generator=order_rng).tolist(); cursor = 0
        if resume_checkpoint is not None:
            resumed = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            if (resumed.get("continuation_format") != CONTINUATION_FORMAT
                    or resumed.get("execution_mode") != run_mode
                    or resumed.get("protocol_digest") != protocol["digest"] or resumed.get("feature_digest") != feature_digest
                    or resumed.get("config_digest") != config_digest or resumed.get("code_snapshot_sha256") != code_digest or resumed.get("method") != method
                    or resumed.get("task") != task or resumed.get("budget") != budget
                    or resumed.get("seed") != seed
                    or resumed.get("support_ids") != support or resumed.get("source_checkpoint_sha256") != _sha(Path(source_checkpoint))):
                raise ValueError("adaptation continuation identity does not match the requested condition")
            if resumed.get("adapter_config") != (adapter_config if method == "lora_head_ft" else None):
                raise ValueError("adaptation continuation LoRA configuration differs from the requested condition")
            model.load_state_dict(resumed["state_dict"]); optimizer.load_state_dict(resumed["optimizer"])
            if lora_encoder:
                expected_lora_keys = {k for k in lora_encoder.mae.state_dict() if "lora_" in k}
                if set(resumed.get("lora_state_dict") or {}) != expected_lora_keys:
                    raise ValueError("adaptation continuation LoRA tensor keys do not match the allowlist")
                lora_encoder.mae.load_state_dict(resumed["lora_state_dict"], strict=False)
            start_step = int(resumed["step"]); order = list(resumed["order"]); cursor = int(resumed["cursor"])
            order_rng.set_state(resumed["order_rng_state"].cpu())
            resume_rng_state = resumed["torch_rng_state"].cpu()
            resume_numpy_rng_state = resumed["numpy_rng_state"]
            resume_python_rng_state = resumed["python_rng_state"]
            resume_cuda_rng_state = resumed.get("cuda_rng_state")
            curve = resumed.get("curve", []); best_step = int(resumed.get("best_step", 0))
            latest_snapshot = resumed.get("latest_snapshot")
            snapshot_archive = resumed.get("snapshot_archive", {})
            snapshot_archive_bytes = resumed.get("snapshot_archive_bytes", {})
            if set(snapshot_archive_bytes) != set(snapshot_archive):
                raise ValueError("continuation is missing immutable snapshot bytes")
            max_lora_gradient_norm = float(resumed.get("max_lora_gradient_norm", 0.0))
            if latest_snapshot and latest_snapshot.get("selected_step"):
                snapshot_archive.setdefault(str(latest_snapshot["selected_step"]), latest_snapshot)
            for snapshot_key, archived in snapshot_archive.items():
                if (not isinstance(archived, dict) or str(archived.get("selected_step")) != str(snapshot_key)
                        or archived.get("checkpoint_format") != CHECKPOINT_FORMAT
                        or archived.get("execution_mode") != run_mode
                        or archived.get("protocol_digest") != protocol["digest"]
                        or archived.get("source_checkpoint_sha256") != _sha(Path(source_checkpoint))
                        or archived.get("support_ids") != support
                        or archived.get("method") != method or archived.get("task") != task
                        or archived.get("budget") != budget or archived.get("seed") != seed
                        or archived.get("feature_digest") != feature_digest
                        or archived.get("code_snapshot_sha256") != code_digest
                        or archived.get("condition_config") != condition_config
                        or archived.get("config_digest") != config_digest):
                    raise ValueError("continuation snapshot archive contains incompatible evidence")
                try:
                    archive_model = DirectDecoder(768, 7)
                    archive_model.load_state_dict(archived["state_dict"], strict=True)
                except Exception as exc:
                    raise ValueError("continuation snapshot archive has invalid head weights") from exc
                if any(not torch.isfinite(value).all() for value in archived["state_dict"].values()):
                    raise ValueError("continuation snapshot archive contains non-finite head weights")
                archived_path = out / f"{method}.step{snapshot_key}.pt"
                archived_bytes = snapshot_archive_bytes[snapshot_key].get("content")
                archived_sha = snapshot_archive_bytes[snapshot_key].get("sha256")
                if (not isinstance(archived_bytes, bytes)
                        or hashlib.sha256(archived_bytes).hexdigest() != archived_sha):
                    raise ValueError("continuation snapshot bytes are missing or corrupt")
                if archived_path.exists():
                    if _sha(archived_path) != archived_sha:
                        raise ValueError(f"existing snapshot differs from continuation archive: {archived_path}")
                else:
                    _write_snapshot_bytes(archived_path, archived_bytes)
            if start_step > cap:
                raise ValueError("adaptation continuation is already beyond requested total updates")
            # Recovery validation constructs temporary models. Restore the
            # serialized training RNG after those checks so it is the exact
            # state seen by the next optimizer update.
            torch.set_rng_state(resume_rng_state)
            np.random.set_state(resume_numpy_rng_state)
            random.setstate(resume_python_rng_state)
            if device.type == "cuda" and resume_cuda_rng_state:
                torch.cuda.set_rng_state_all([state.cpu() for state in resume_cuda_rng_state])
        _profile_ready()
        with _SignalStop() as signal_stop:
            for step in range(start_step + 1, cap + 1):
                update_started = _profile_update_start(timing_profile, step, device)
                model.train(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.0
                for _ in range(batch_size // microbatch_size):
                    selected = []
                    while len(selected) < microbatch_size:
                        if cursor >= len(order): order = torch.randperm(len(raw_train) if method == "lora_head_ft" else len(rows["action"]), generator=order_rng).tolist(); cursor = 0
                        take = min(microbatch_size - len(selected), len(order) - cursor); selected.extend(order[cursor:cursor+take]); cursor += take
                    if method == "lora_head_ft":
                        batch = [raw_train[i] for i in selected]
                        images_t = torch.stack([x["image_t"] for x in batch]); images_n = torch.stack([x["image_next"] for x in batch])
                        current, future = _encode_live(lora_encoder, images_t), _encode_live(lora_encoder, images_n)
                        target = torch.stack([x["action"] for x in batch]).to(device)
                    else:
                        ix = torch.tensor(selected); current=rows["current"][ix].to(device); future=rows["future"][ix].to(device); target=rows["action"][ix].to(device)
                    pred, _ = model(current, future); loss = torch.nn.functional.mse_loss(pred, target)
                    (loss / (batch_size // microbatch_size)).backward(); loss_sum += float(loss.detach())
                if lora_encoder:
                    grad_sq = sum(float(p.grad.detach().float().pow(2).sum()) for p in lora_encoder.mae.parameters()
                                  if p.requires_grad and p.grad is not None)
                    max_lora_gradient_norm = max(max_lora_gradient_norm, grad_sq ** 0.5)
                allowed = list(model.parameters()) + ([p for p in lora_encoder.mae.parameters() if p.requires_grad] if lora_encoder else [])
                torch.nn.utils.clip_grad_norm_(allowed, grad_clip); optimizer.step()
                _profile_update_end(timing_profile, update_started, device)
                if step in UPDATE_SNAPSHOTS or (updates not in UPDATE_SNAPSHOTS and step == updates):
                    curve.append({"step": step, "support_train_mse": loss_sum / (batch_size // microbatch_size)})
                    best_step = step
                    snap = {"checkpoint_format": CHECKPOINT_FORMAT, "execution_mode": run_mode,
                            "study_config_sha256": _config_digest(_scientific_config()),
                            "config_digest": config_digest, "condition_config": condition_config,
                            "planned_updates": planned_updates, "optimizer_updates": step,
                            "protocol": PROTOCOL, "protocol_digest": protocol["digest"], "feature_digest": feature_digest,
                            "code_snapshot_sha256": code_digest,
                            "method": method, "task": task, "budget": budget, "seed": seed,
                            "source_checkpoint_sha256": _sha(Path(source_checkpoint)), "support_ids": support,
                            "source_feature_digest": source_feature_digest,
                            "architecture": ARCHITECTURE,
                            "adapter_config": adapter_config if method == "lora_head_ft" else None,
                            "mae_checkpoint_sha256": cache_manifest.get("mae_checkpoint_sha256"),
                            "selected_step": step,
                            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            "lora_state_dict": ({k:v.detach().cpu().clone() for k,v in lora_encoder.mae.state_dict().items() if "lora_" in k} if lora_encoder else None),
                            "initial_lora_state_dict": ({k: v.clone() for k, v in initial_lora_state.items()} if initial_lora_state else None),
                            "max_lora_gradient_norm": max_lora_gradient_norm,
                            "lora_targets": lora_names, "curve": [dict(item) for item in curve]}
                    if run_mode == "calibration" and lora_encoder is not None:
                        # Preserve a prediction made by the live, trained model before
                        # serializing the snapshot so smoke can compare it after reload.
                        probe = raw_train[0]
                        was_training = model.training
                        lora_was_training = lora_encoder.training
                        model.eval(); lora_encoder.eval()
                        with torch.no_grad():
                            probe_current = _encode_live(lora_encoder, probe["image_t"].unsqueeze(0).to(device))
                            probe_future = _encode_live(lora_encoder, probe["image_next"].unsqueeze(0).to(device))
                            snap["calibration_probe"] = {
                                "transition_id": probe["transition_id"],
                                "prediction": model(probe_current, probe_future)[0].detach().cpu().clone()}
                        model.train(was_training)
                        lora_encoder.train(lora_was_training)
                    snap_path = out / f"{method}.step{step}.pt"
                    snap_bytes = _snapshot_bytes(snap)
                    snap_sha = _write_snapshot_bytes(snap_path, snap_bytes)
                    latest_snapshot = snap
                    snapshot_archive[str(step)] = snap
                    snapshot_archive_bytes[str(step)] = {"sha256": snap_sha, "content": snap_bytes}
                if step % 100 == 0 or step == cap or signal_stop.requested:
                    continuation = {"continuation_format": CONTINUATION_FORMAT, "execution_mode": run_mode,
                                    "protocol_digest": protocol["digest"], "feature_digest": feature_digest,
                                    "config_digest": config_digest, "method": method, "task": task,
                                    "code_snapshot_sha256": code_digest,
                                    "budget": budget, "seed": seed, "source_checkpoint_sha256": _sha(Path(source_checkpoint)),
                                    "source_feature_digest": source_feature_digest,
                                    "adapter_config": adapter_config if method == "lora_head_ft" else None,
                                    "support_ids": support, "state_dict": model.state_dict(),
                                    "lora_state_dict": ({k: v.detach().cpu() for k, v in lora_encoder.mae.state_dict().items() if "lora_" in k} if lora_encoder else None),
                                    "optimizer": optimizer.state_dict(), "step": step, "order": order, "cursor": cursor,
                                    "order_rng_state": order_rng.get_state(), "torch_rng_state": torch.get_rng_state(),
                                    "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
                                    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                                    "best_step": best_step, "curve": curve,
                                    "latest_snapshot": latest_snapshot, "snapshot_archive": snapshot_archive,
                                    "snapshot_archive_bytes": snapshot_archive_bytes,
                                    "max_lora_gradient_norm": max_lora_gradient_norm}
                    _atomic_save(continuation, out / f"{method}.last.pt")
                if signal_stop.requested:
                    break
        if signal_stop.requested and step < cap:
            result = {"status": "interrupted", "execution_mode": run_mode,
                      "method": method, "task": task, "budget": budget, "seed": seed,
                      "updates": step, "planned_updates": planned_updates,
                      "condition_training_complete": False,
                      "continuation_checkpoint": str(out / f"{method}.last.pt"),
                      "protocol_digest": protocol["digest"], "source_checkpoint_sha256": _sha(Path(source_checkpoint))}
            _write_json(out / f"{method}.json", result)
            return result
        chosen_step = selected_step or (updates if updates not in UPDATE_SNAPSHOTS else max((s for s in UPDATE_SNAPSHOTS if s <= cap), default=0))
        if chosen_step == 0:
            raise ValueError("no scheduled adaptation snapshot is available")
        selected_path = out / f"{method}.step{chosen_step}.pt"
        if not selected_path.exists() and str(chosen_step) in snapshot_archive:
            record = snapshot_archive_bytes[str(chosen_step)]
            _write_snapshot_bytes(selected_path, record["content"])
        if cap < planned_updates and selected_step is None and not (out / f"{method}.step{chosen_step}.pt").exists():
            _write_json(out / f"{method}.json", {"status": "calibration", "method": method,
                "task": task, "budget": budget, "seed": seed, "updates": cap,
                "continuation_checkpoint": str(out / f"{method}.last.pt"),
                "protocol_digest": protocol["digest"], "feature_digest": feature_digest,
                "source_checkpoint_sha256": _sha(Path(source_checkpoint))})
            return json.loads((out / f"{method}.json").read_text())
        if not selected_path.exists():
            raise ValueError(f"selected update {chosen_step} was not saved; available snapshots: {sorted(p.name for p in out.glob(method + '.step*.pt'))}")
        if run_mode == "scientific" and cap == planned_updates:
            hashes = {path.name: _sha(path) for path in sorted(out.glob(f"{method}.step*.pt"))}
            _write_immutable_json(out / f"{method}.complete.json", {
                "evidence_format": EVIDENCE_FORMAT, "status": "complete",
                "method": method, "task": task, "budget": budget, "seed": seed,
                "protocol_digest": protocol["digest"], "source_checkpoint_sha256": _sha(Path(source_checkpoint)),
                "feature_digest": feature_digest, "code_snapshot_sha256": code_digest,
                "source_feature_digest": source_feature_digest,
                "planned_updates": planned_updates, "config_digest": config_digest,
                "snapshot_sha256": hashes})
        selected_payload = torch.load(selected_path, map_location="cpu", weights_only=False)
        _write_snapshot_bytes(out / f"{method}.selected.pt", selected_path.read_bytes())
        model.eval()
    optimizer_updates = cap if optimizer is not None else 0
    frozen_backbone_unchanged = (all(torch.equal(value, lora_encoder.mae.state_dict()[key].detach().cpu())
                                     for key, value in frozen_encoder_state.items())
                                 if frozen_encoder_state is not None else None)
    status = ("complete" if run_mode == "scientific" and cap == planned_updates
              else "calibration" if run_mode == "calibration" or cap != planned_updates else "synthetic")
    result = {"status": status, "execution_mode": run_mode,
              "condition_training_complete": run_mode == "scientific" and cap == planned_updates,
              "planned_updates": planned_updates,
              "study_config_sha256": _config_digest(_scientific_config()), "config_digest": config_digest,
              "condition_config": condition_config,
              "method": method, "task": task,
              "budget": budget, "seed": seed, "support_ids": support, "updates": optimizer_updates,
              "optimizer_updates": optimizer_updates,
              "selected_step": (selected_step or (cap if cap in UPDATE_SNAPSHOTS or updates not in UPDATE_SNAPSHOTS else best_step)) if optimizer is not None else None,
              "best_step": best_step,
              "seconds": time.time()-started, "trainable_parameters": (sum(p.numel() for p in model.parameters() if p.requires_grad) + (sum(p.numel() for p in lora_encoder.mae.parameters() if p.requires_grad) if lora_encoder else 0)) if optimizer is not None else 0,
              "lora_targets": lora_names, "protocol_digest": protocol["digest"], "feature_digest": feature_digest,
              "source_feature_digest": source_feature_digest,
              "max_lora_gradient_norm": max_lora_gradient_norm if method == "lora_head_ft" else 0.0,
              "lora_parameters_changed": (any(not torch.equal(initial_lora_state[k], selected_payload["lora_state_dict"][k])
                                               for k in initial_lora_state)
                                          if method == "lora_head_ft" else False),
              "frozen_backbone_unchanged": frozen_backbone_unchanged,
              "code_snapshot_sha256": code_digest,
              "source_checkpoint_sha256": _sha(Path(source_checkpoint)), "curve": curve}
    _write_json(out / f"{method}.json", result)
    return result


def _predict_cached(model, query, support_rows, memory, method, device, temperature=0.1,
                    source_support_memory=None) -> torch.Tensor:
    if method == "frozen_source" or method in ("head_ft", "lora_head_ft"):
        with torch.no_grad(): return model(query["current"].to(device), query["future"].to(device))[0].cpu()
    qcur, qfut = query["current"].to(device), query["future"].to(device)
    if method == "target_knn":
        result = memory.retrieve(qcur, qfut, k=3, temperature=temperature)
        return result.actions.sum(1).cpu()
    if method == "current_weighted_knn":
        result = memory.retrieve(qcur, qfut, k=3, current_only=True, temperature=temperature)
        return result.actions.sum(1).cpu()
    if method == "transition_1nn":
        result = memory.retrieve(qcur, qfut, k=1)
        return memory.actions[result.indices[:, 0]].cpu()
    if method == "support_mean":
        return support_rows["action"].mean(0, keepdim=True).expand(len(qcur), -1).cpu()
    if method == "source_support_knn":
        if source_support_memory is None:
            raise ValueError("source_support_knn requires the pooled source and target-support bank")
        result = source_support_memory.retrieve(qcur, qfut, k=3, temperature=temperature)
        return result.actions.sum(1).cpu()
    raise ValueError(method)


def _query_rows_from_hdf5(cache: dict, spec: dict, task: str, mean: torch.Tensor,
                          std: torch.Tensor) -> dict:
    """Join evaluation-only HDF5 action labels to label-free cached query features."""
    features, actions, transition_ids, demo_ids, task_ids = [], [], [], [], []
    raw = RawSelectedRows(spec["path"], task, list(map(int, spec["query"])), mean, std)
    for row in raw:
        demo, timestep = int(row["demo"]), int(row["timestep"])
        lo, hi = cache["_offsets"][demo]
        if timestep >= hi - lo:
            raise ValueError(f"query transition exceeds cached coverage: {task}:{demo}:{timestep}")
        features.append((cache["feat_t"][lo+timestep].float(), cache["feat_next"][lo+timestep].float()))
        actions.append(row["action"].float()); transition_ids.append(row["transition_id"])
        demo_ids.append(_demo_uid(task, demo)); task_ids.append(task)
    if not features:
        raise ValueError(f"query set is empty for {task}")
    return {"current": torch.stack([x[0] for x in features]), "future": torch.stack([x[1] for x in features]),
            "action": torch.stack(actions), "transition_ids": transition_ids,
            "demo_ids": torch.tensor(demo_ids), "task_ids": task_ids}


def _retention_metrics(model, protocol: dict, caches: dict, mean: torch.Tensor, std: torch.Tensor,
                       device: torch.device, encoder=None, *, target_adapter_active=True) -> dict:
    """Evaluate source-retention demonstrations with the reported weights active."""
    per_task = {}
    model.eval()
    for task, spec in protocol["tasks"].items():
        if spec["role"] != "source":
            continue
        rows = _demo_rows(caches[task], spec["source"]["source_retention"], mean, std, task)
        if encoder is None:
            with torch.no_grad():
                prediction = model(rows["current"].to(device), rows["future"].to(device))[0].cpu()
            raw_action = rows["action"]
            demo_ids = rows["demo_ids"].tolist()
        else:
            loader = torch.utils.data.DataLoader(RawSelectedRows(spec["path"], task,
                          list(map(int, spec["source"]["source_retention"])), mean, std),
                          batch_size=16, shuffle=False, num_workers=0)
            predictions, targets, demos = [], [], []
            encoder.eval()
            with torch.no_grad():
                for batch in loader:
                    current = _encode_live(encoder, batch["image_t"])
                    future = _encode_live(encoder, batch["image_next"])
                    predictions.append(model(current, future)[0].cpu())
                    targets.append(batch["action"])
                    demos.extend([_demo_uid(task, int(x)) for x in batch["demo"].tolist()])
            prediction = torch.cat(predictions); raw_action = torch.cat(targets); demo_ids = demos
        row_mse = (prediction - raw_action).pow(2).mean(-1).numpy()
        per_task[task] = hierarchical_macro_mse(row_mse, [task] * len(row_mse), demo_ids)
    return {"hierarchical_source_normalized_mse": sum(per_task.values()) / len(per_task),
            "per_task_source_normalized_mse": per_task,
            "target_adapter_active": bool(target_adapter_active)}


def evaluate_task(protocol_path: str | Path, feature_dir: str | Path, source_checkpoint: str | Path,
                  output_dir: str | Path, *, task: str, budget: int, seed: int = 42,
                  device: str = "cuda", methods: tuple[str, ...] = tuple(METHODS),
                  source_support_knn: bool = False, mae_checkpoint: str | None = None,
                  final_lock: str | Path | None = None,
                  confirmation_file: str | Path | None = None,
                  selected_updates: dict[str, int] | None = None,
                  run_mode: str = "scientific",
                  lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0,
                  lora_blocks: tuple[int, ...] = (10, 11)) -> dict:
    if run_mode not in ("scientific", "calibration", "synthetic"):
        raise ValueError("run_mode must be scientific, calibration, or synthetic")
    selected_updates = {key: value for key, value in (selected_updates or {}).items() if value is not None}
    protocol = validate_goal_protocol(protocol_path); spec = protocol["tasks"].get(task)
    if spec is None or spec["role"] not in ("development", "final_test"):
        raise ValueError("evaluation requires a preregistered development or final-test task")
    lock = None
    if spec["role"] == "final_test":
        final_record = confirmation_file or final_lock
        if not final_record or not Path(final_record).is_file(): raise RuntimeError("final-test evaluation requires a registered confirmation record")
        lock = json.loads(Path(final_record).read_text())
        lock_body = dict(lock); claimed_lock_digest = lock_body.pop("confirmation_digest", None)
        calculated_lock_digest = hashlib.sha256(json.dumps(lock_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        registered = lock.get("source_checkpoints", {}).get(str(seed))
        if (lock.get("protocol_digest") != protocol["digest"]
                or lock.get("confirmation_format") != 4
                or lock.get("confirmation_registered") is not True
                or claimed_lock_digest != calculated_lock_digest or registered != _sha(Path(source_checkpoint))):
            raise RuntimeError("final-test freeze file is invalid")
        if budget not in lock.get("confirmation_scope", {}).get("support_budgets", []):
            raise ValueError("requested support budget is outside the registered confirmation scope")
        locked_updates = lock.get("selected_updates", {})
        if set(locked_updates) != {"head_ft", "lora_head_ft"} or any(v not in UPDATE_SNAPSHOTS for v in locked_updates.values()):
            raise ValueError("confirmation record has invalid selected update counts")
        if selected_updates and any(lock.get("selected_updates", {}).get(k) != v for k, v in selected_updates.items()):
            raise ValueError("requested update count differs from the development freeze record")
        selected_updates = dict(locked_updates)
    allowed = METHODS | {"support_mean", "transition_1nn", "current_weighted_knn", "source_support_knn"}
    for method in methods:
        if method in ("head_ft", "lora_head_ft") and method not in (selected_updates or {}):
            if spec["role"] == "development":
                raise ValueError(f"development {method} evaluation must identify its candidate update snapshot")
        if method not in allowed: raise ValueError(f"unsupported method {method}")
    caches, feature_digest = _load_cache_map(feature_dir, protocol, target_task=task, budget=budget, seed=seed,
                                             include_query=True, include_retention=True, run_mode=run_mode)
    device = _resolve_device(device)
    source = _load_source(source_checkpoint, protocol, caches, device, run_mode=run_mode)
    if int(source.get("seed", -1)) != int(seed):
        raise ValueError("source checkpoint repetition seed does not match evaluation seed")
    mean = source["normalizer_mean"].cpu(); std = source["normalizer_std"].cpu()
    support_ids = list(map(int, spec["support_budgets_by_seed"][str(seed)][str(budget)]))
    condition_root = Path(output_dir) / task / f"{budget}shot" / f"seed{seed}"
    cache_manifest = json.loads((Path(feature_dir) / "manifest.json").read_text())
    # Validate every requested learned artifact and its required encoder before
    # any final-query action labels are opened.
    for method in methods:
        if run_mode != "scientific" or method not in ("head_ft", "lora_head_ft"):
            continue
        step = selected_updates.get(method)
        if step not in UPDATE_SNAPSHOTS:
            raise ValueError(f"{method} evaluation requires a registered snapshot update")
        artifact = (condition_root / f"{method}.step{step}.pt" if spec["role"] == "development"
                    else condition_root / f"{method}.selected.pt")
        expected_identity = {"method": method, "task": task, "budget": budget, "seed": seed,
            "support_ids": support_ids, "protocol_digest": protocol["digest"],
            "source_checkpoint_sha256": _sha(Path(source_checkpoint)),
            "source_feature_digest": source["source_feature_digest"],
            "feature_digest": feature_digest, "code_snapshot_sha256": _code_snapshot_digest(),
            "selected_step": step}
        _validate_adaptation_evidence(condition_root, artifact, expected_identity)
        if method == "lora_head_ft":
            if not mae_checkpoint or not Path(mae_checkpoint).is_file():
                raise ValueError("LoRA evaluation requires an available --mae-ckpt")
            if _sha(Path(mae_checkpoint)) != cache_manifest.get("mae_checkpoint_sha256"):
                raise ValueError("LoRA evaluation MAE checkpoint differs from the feature cache")
    if spec["role"] == "final_test":
        from .fewshot_report import canonical_study_root, study_access_lock
        access_root = canonical_study_root({seed: source_checkpoint})
        if lock.get("study_root") != str(access_root):
            raise RuntimeError("confirmation record is bound to another canonical study root")
        if run_mode == "scientific":
            registry_path = access_root / "confirmation_registry.json"
            if (not registry_path.is_file()
                    or json.loads(registry_path.read_text()) != lock):
                raise RuntimeError("confirmation record does not match the canonical study registry")
        access_path = access_root / "final_query_access.json"
        condition = {"task": task, "budget": budget, "seed": seed,
                     "source_checkpoint_sha256": lock["source_checkpoints"][str(seed)]}
        with study_access_lock(access_root):
            access = json.loads(access_path.read_text()) if access_path.exists() else {
                "evidence_format": EVIDENCE_FORMAT, "protocol_digest": protocol["digest"], "conditions": []}
            if access.get("protocol_digest") != protocol["digest"] or access.get("evidence_format") != EVIDENCE_FORMAT:
                raise RuntimeError("final-query access ledger has an incompatible identity")
            if condition not in access["conditions"]:
                access["conditions"].append(condition)
                tmp_access = access_path.with_suffix(".json.tmp")
                tmp_access.write_text(json.dumps(access, indent=2, sort_keys=True) + "\n"); tmp_access.replace(access_path)
    query_started = time.perf_counter()
    query = _query_rows_from_hdf5(caches[task], spec, task, mean, std)
    query_evaluation_seconds = time.perf_counter() - query_started
    support = _demo_rows(caches[task], support_ids, mean, std, task)
    retrieval_methods = {"target_knn", "current_weighted_knn", "transition_1nn"}
    memory = None
    retrieval_bank_started = time.perf_counter()
    if retrieval_methods.intersection(methods):
        memory = TransitionMemory(768, 7, device)
        memory.extend(support["current"].to(device), support["future"].to(device),
                      support["action"].to(device), support["demo_ids"].to(device))
        memory.transition_ids = list(support["transition_ids"])
    source_bank_rows = None
    source_support_knn_memory = None
    if source_support_knn or "source_support_knn" in methods:
        source_bank_rows = []
        for src_task, src_spec in protocol["tasks"].items():
            if src_spec["role"] == "source":
                source_bank_rows.append(_demo_rows(caches[src_task], src_spec["source"]["source_train"], mean, std, src_task))
        source_support_knn_memory = TransitionMemory(768, 7, device)
        source_support_knn_memory.extend(torch.cat([r["current"] for r in source_bank_rows]+[support["current"]]).to(device),
                                         torch.cat([r["future"] for r in source_bank_rows]+[support["future"]]).to(device),
                                         torch.cat([r["action"] for r in source_bank_rows]+[support["action"]]).to(device),
                                         torch.cat([r["demo_ids"] for r in source_bank_rows]+[support["demo_ids"]]).to(device))
        source_support_knn_memory.transition_ids = sum((r["transition_ids"] for r in source_bank_rows), []) + support["transition_ids"]
    retrieval_bank_seconds = time.perf_counter() - retrieval_bank_started
    root = condition_root; root.mkdir(parents=True, exist_ok=True)
    report = {"evidence_format": EVIDENCE_FORMAT, "execution_mode": run_mode,
              "protocol": PROTOCOL, "protocol_digest": protocol["digest"], "task": task,
              "status": "complete" if run_mode == "scientific" else run_mode,
              "role": spec["role"], "budget": budget, "seed": seed,
              "query_ids": query["transition_ids"], "support_ids": support_ids,
              "normalizer_sha256": hashlib.sha256(mean.contiguous().numpy().tobytes() + std.contiguous().numpy().tobytes()).hexdigest(),
              "source_checkpoint_sha256": _sha(Path(source_checkpoint)),
              "source_feature_digest": source["source_feature_digest"],
              "feature_digest": feature_digest,
              "code_snapshot_sha256": source.get("code_snapshot_sha256"),
              "phase_seconds": {"query_label_and_feature_loading": query_evaluation_seconds,
                                "retrieval_bank_construction": retrieval_bank_seconds},
              "selected_updates": dict(selected_updates or {}), "methods": {}}
    evaluation_config = {"protocol_digest": protocol["digest"],
        "task": task, "budget": budget, "seed": seed, "methods": sorted(methods),
        "selected_updates": dict(selected_updates or {}), "source_checkpoint_sha256": _sha(Path(source_checkpoint)),
        "source_feature_digest": source["source_feature_digest"], "feature_digest": feature_digest,
        "code_snapshot_sha256": source.get("code_snapshot_sha256")}
    report["evaluation_config_sha256"] = _config_digest(evaluation_config)
    candidate_methods = set(methods) & {"head_ft", "lora_head_ft"}
    if run_mode == "scientific" and spec["role"] == "development" and candidate_methods:
        if len(methods) != 1 or len(candidate_methods) != 1:
            raise ValueError("evaluate each learned development snapshot separately after the baseline bundle")
        baseline_path = root / "evaluation_baselines.json"
        if not baseline_path.is_file():
            raise FileNotFoundError("evaluate the six development controls once before candidate snapshots")
        baseline = json.loads(baseline_path.read_text())
        controls = {"frozen_source", "target_knn", "support_mean", "transition_1nn",
                    "current_weighted_knn", "source_support_knn"}
        if (baseline.get("execution_mode") != "scientific" or baseline.get("status") != "complete"
                or baseline.get("protocol_digest") != protocol["digest"]
                or baseline.get("task") != task or baseline.get("budget") != budget
                or baseline.get("seed") != seed or baseline.get("query_ids") != query["transition_ids"]
                or baseline.get("support_ids") != support_ids
                or baseline.get("source_checkpoint_sha256") != _sha(Path(source_checkpoint))
                or baseline.get("source_feature_digest") != source["source_feature_digest"]
                or set(baseline.get("methods", {})) != controls
                or set(baseline.get("prediction_artifacts", {})) != controls):
            raise ValueError("development baseline bundle identity or method coverage is invalid")
        report["baseline_evidence"] = {
            "evaluation_sha256": _sha(baseline_path),
            "prediction_artifact_sha256": {name: baseline["prediction_artifacts"][name]["sha256"]
                                            for name in sorted(controls)}}
    for method in methods:
        if device.type == "cuda":
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        model = DirectDecoder(768, 7).to(device); model.load_state_dict(source["state_dict"]); model.eval()
        predictions = None; adaptation_encoder = None; adaptation_payload = None; trained = None
        raw_encoding_seconds = 0.0
        if method == "target_knn":
            infer_started = time.perf_counter()
            predictions = _predict_cached(model, query, support, memory, method, device)
        elif method in ("head_ft", "lora_head_ft"):
            selected_step = (selected_updates or {}).get(method)
            artifact = (root / f"{method}.step{selected_step}.pt" if selected_step is not None
                        and spec["role"] == "development" else root / f"{method}.selected.pt")
            if not artifact.exists():
                raise FileNotFoundError(f"adaptation checkpoint missing: {artifact}; run adapt first")
            adaptation_record = root / f"{method}.json"
            condition = None
            if adaptation_record.is_file():
                condition = json.loads(adaptation_record.read_text())
                if (condition.get("execution_mode") != run_mode or condition.get("method") != method
                        or condition.get("task") != task or condition.get("budget") != budget
                        or condition.get("seed") != seed):
                    raise ValueError("adaptation completion record does not match the requested condition")
                if run_mode == "scientific" and (condition.get("status") != "complete"
                        or condition.get("updates") != condition.get("planned_updates")
                        or condition.get("planned_updates") != (1000 if spec["role"] == "development" else selected_updates.get(method))
                        or condition.get("config_digest") is None):
                    raise ValueError("scientific evaluation requires completed condition training")
                if run_mode == "scientific":
                    completion_path = root / f"{method}.complete.json"
                    completion = json.loads(completion_path.read_text()) if completion_path.is_file() else {}
                    expected_completion = {"evidence_format": EVIDENCE_FORMAT, "status": "complete",
                        "method": method, "task": task, "budget": budget, "seed": seed,
                        "protocol_digest": protocol["digest"],
                        "source_checkpoint_sha256": _sha(Path(source_checkpoint)),
                        "source_feature_digest": source["source_feature_digest"],
                        "feature_digest": feature_digest, "code_snapshot_sha256": _code_snapshot_digest(),
                        "planned_updates": condition.get("planned_updates"),
                        "config_digest": condition.get("config_digest")}
                    snapshot_name = (artifact.name if ".step" in artifact.name else
                                     f"{method}.step{condition.get('selected_step')}.pt")
                    if (any(completion.get(key) != value for key, value in expected_completion.items())
                            or completion.get("snapshot_sha256", {}).get(snapshot_name) != _sha(artifact)):
                        raise ValueError("scientific evaluation requires hash-bound adaptation completion evidence")
            elif run_mode == "scientific":
                raise FileNotFoundError(f"adaptation completion record missing: {adaptation_record}")
            trained = torch.load(artifact, map_location=device, weights_only=False)
            adaptation_payload = trained
            if (trained.get("checkpoint_format") != CHECKPOINT_FORMAT or trained.get("execution_mode") != run_mode
                    or trained.get("protocol_digest") != protocol["digest"] or trained.get("source_checkpoint_sha256") != _sha(Path(source_checkpoint))
                    or trained.get("support_ids") != support_ids or trained.get("feature_digest") != feature_digest
                    or trained.get("code_snapshot_sha256") != _code_snapshot_digest()
                    or trained.get("method") != method or trained.get("task") != task
                    or trained.get("budget") != budget or trained.get("seed") != seed
                    or (condition and trained.get("config_digest") != condition.get("config_digest"))
                    or (condition and trained.get("condition_config") != condition.get("condition_config"))
                    or (run_mode == "scientific" and trained.get("study_config_sha256") != _config_digest(_scientific_config()))
                    or trained.get("architecture") != ARCHITECTURE):
                raise ValueError("adaptation checkpoint identity does not match this evaluation")
            if selected_step is not None and trained.get("selected_step") != selected_step:
                raise ValueError(f"{method} checkpoint update count differs from requested development snapshot")
            if spec["role"] == "final_test" and trained.get("selected_step") != lock.get("selected_updates", {}).get(method):
                raise ValueError("final-test checkpoint does not use the frozen development update count")
            model.load_state_dict(trained["state_dict"]); model.eval()
            if method == "lora_head_ft":
                if not mae_checkpoint: raise ValueError("LoRA evaluation requires --mae-ckpt")
                expected_adapter = {"rank": lora_rank, "alpha": lora_alpha, "dropout": lora_dropout,
                                    "blocks": list(lora_blocks), "projections": ["qkv", "proj"]}
                if (trained.get("adapter_config") != expected_adapter
                        or trained.get("mae_checkpoint_sha256") != _sha(Path(mae_checkpoint))):
                    raise ValueError("LoRA adapter configuration/base checkpoint identity mismatch")
                encoder, _ = _load_encoder_with_lora(mae_checkpoint, device,
                                                     lora_blocks, lora_rank, lora_alpha, lora_dropout)
                expected_keys = {k for k in encoder.mae.state_dict() if "lora_" in k}
                if set(trained.get("lora_state_dict") or {}) != expected_keys:
                    raise ValueError("LoRA adapter tensor keys do not match the configured allowlist")
                encoder.mae.load_state_dict(trained["lora_state_dict"], strict=False); encoder.eval()
                adaptation_encoder = encoder
                raw_query = RawSelectedRows(spec["path"], task, spec["query"], mean, std)
                preds=[]
                if device.type == "cuda": torch.cuda.synchronize(device)
                infer_started = time.perf_counter()
                with torch.no_grad():
                    for row in raw_query:
                        encoding_started = time.perf_counter()
                        current=_encode_live(encoder,row["image_t"].unsqueeze(0)); future=_encode_live(encoder,row["image_next"].unsqueeze(0))
                        if device.type == "cuda": torch.cuda.synchronize(device)
                        raw_encoding_seconds += time.perf_counter() - encoding_started
                        preds.append(model(current,future)[0].squeeze(0).cpu())
                predictions=torch.stack(preds)
            else:
                infer_started = time.perf_counter()
                predictions=_predict_cached(model, query, support, memory, method, device,
                                             source_support_memory=source_support_knn_memory)
        else:
            infer_started = time.perf_counter()
            predictions=_predict_cached(model, query, support, memory, method, device,
                                        source_support_memory=source_support_knn_memory)
        if device.type == "cuda": torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - infer_started
        if method in allowed:
            pred_np=predictions.numpy(); target_np=query["action"].numpy()
            row_error=((predictions-query["action"]).pow(2).mean(-1)).numpy()
            metrics=action_metrics(pred_np,target_np,mean=mean.numpy(),std=std.numpy(),task_ids=query["task_ids"],demo_ids=query["demo_ids"].tolist())
            metrics["hierarchical_source_normalized_mse"]=hierarchical_macro_mse(row_error,query["task_ids"],query["demo_ids"].tolist())
            metrics["cached_compute_ms_per_transition"] = 1000.0 * inference_seconds / max(1, len(query["action"]))
            metrics["raw_frame_encoding_seconds"] = raw_encoding_seconds if method == "lora_head_ft" else None
            metrics["end_to_end_latency_ms_per_transition"] = None
            metrics["latency_note"] = "not comparable across methods until raw-image calibration completes"
            metrics["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            adaptation_result = root / f"{method}.json"
            if method in ("head_ft", "lora_head_ft") and adaptation_result.is_file():
                adaptation_payload = json.loads(adaptation_result.read_text())
                metrics["adaptation_seconds"] = adaptation_payload.get("seconds")
                metrics["trainable_parameters"] = adaptation_payload.get("trainable_parameters")
                metrics["selected_update"] = (trained.get("selected_step") if trained is not None else adaptation_payload.get("selected_step"))
                if trained is not None:
                    metrics["adaptation_config_digest"] = trained.get("config_digest")
            else:
                metrics["adaptation_seconds"] = 0.0
                metrics["trainable_parameters"] = 0
            report["methods"][method]=metrics
            if method in ("head_ft", "lora_head_ft", "frozen_source"):
                retention_started = time.perf_counter()
                metrics["source_retention"] = _retention_metrics(
                    model, protocol, caches, mean, std, device, adaptation_encoder,
                    target_adapter_active=(method != "frozen_source"))
                metrics["source_retention_seconds"] = time.perf_counter() - retention_started
            neighbor_ids = similarities = None
            if method in ("target_knn", "current_weighted_knn", "transition_1nn", "source_support_knn"):
                retrieval_memory = source_support_knn_memory if method == "source_support_knn" else memory
                neighbor_result = retrieval_memory.retrieve(
                    query["current"].to(device), query["future"].to(device),
                    k=1 if method == "transition_1nn" else 3,
                    current_only=method == "current_weighted_knn")
                neighbor_ids = [[retrieval_memory.transition_ids[i] if 0 <= i < len(retrieval_memory.transition_ids) else None
                                 for i in row] for row in neighbor_result.indices.cpu().tolist()]
                similarities = neighbor_result.similarities.cpu()
            suffix = (f"_head{(selected_updates or {}).get('head_ft', 'na')}"
                      f"_lora{(selected_updates or {}).get('lora_head_ft', 'na')}" if selected_updates else "")
            prediction_path = root / f"{method}_predictions{suffix}.pt"
            adapter_path = root / f"{method}.step{trained['selected_step']}.pt" if method in ("head_ft", "lora_head_ft") and trained is not None else None
            prediction_payload = {"prediction_format": 4, "execution_mode": run_mode,
                "predictions": predictions, "targets": query["action"],
                "transition_ids": query["transition_ids"], "support_ids": support_ids,
                "normalizer_mean": mean, "normalizer_std": std, "protocol_digest": protocol["digest"],
                "source_checkpoint_sha256": _sha(Path(source_checkpoint)),
                "source_feature_digest": source["source_feature_digest"], "feature_digest": feature_digest,
                "code_snapshot_sha256": source.get("code_snapshot_sha256"),
                "study_config_sha256": _config_digest(_scientific_config()),
                "evaluation_config": evaluation_config,
                "evaluation_config_sha256": report["evaluation_config_sha256"],
                "adaptation_artifact_sha256": _sha(adapter_path) if adapter_path and adapter_path.is_file() else None,
                "method": method, "task": task, "budget": budget, "seed": seed,
                "selected_update": trained.get("selected_step") if trained is not None else None,
                "retrieval_neighbor_ids": neighbor_ids, "retrieval_similarities": similarities}
            if not prediction_path.exists():
                _atomic_save(prediction_payload, prediction_path)
            else:
                saved = torch.load(prediction_path, map_location="cpu", weights_only=False)
                if (saved.get("protocol_digest") != protocol["digest"]
                        or saved.get("prediction_format") != 4
                        or saved.get("execution_mode") != run_mode
                        or saved.get("transition_ids") != query["transition_ids"]
                        or saved.get("support_ids") != support_ids
                        or saved.get("source_checkpoint_sha256") != _sha(Path(source_checkpoint))
                        or saved.get("source_feature_digest") != source["source_feature_digest"]
                        or saved.get("feature_digest") != feature_digest
                        or saved.get("code_snapshot_sha256") != source.get("code_snapshot_sha256")
                        or saved.get("evaluation_config_sha256") != prediction_payload["evaluation_config_sha256"]
                        or saved.get("method") != method or saved.get("task") != task
                        or saved.get("seed") != seed or saved.get("budget") != budget
                        or saved.get("adaptation_artifact_sha256") != prediction_payload["adaptation_artifact_sha256"]
                        or not torch.equal(saved.get("predictions"), predictions)
                        or not torch.equal(saved.get("targets"), query["action"])):
                    raise ValueError(f"prediction artifact identity collision: {prediction_path}")
            report.setdefault("prediction_artifacts", {})[method] = {
                "path": prediction_path.name, "sha256": _sha(prediction_path)}
            report.setdefault("prediction_config_sha256", {})[method] = report["evaluation_config_sha256"]
    selected_updates = dict(selected_updates or {})
    if spec["role"] == "development" and set(selected_updates) == {"head_ft"}:
        report_path = root / f"evaluation_head{selected_updates['head_ft']}.json"
    elif spec["role"] == "development" and set(selected_updates) == {"lora_head_ft"}:
        report_path = root / f"evaluation_lora{selected_updates['lora_head_ft']}.json"
    elif spec["role"] == "development" and not selected_updates:
        report_path = root / "evaluation_baselines.json"
    elif spec["role"] == "development" and selected_updates:
        report_path = root / f"evaluation_head{selected_updates.get('head_ft', 'na')}_lora{selected_updates.get('lora_head_ft', 'na')}.json"
    else:
        report_path = root / "evaluation.json"
    report["evaluation_config_sha256"] = _config_digest({"protocol_digest": protocol["digest"],
        "task": task, "budget": budget, "seed": seed, "methods": sorted(report["methods"]),
        "selected_updates": dict(report.get("selected_updates", {})),
        "source_checkpoint_sha256": report["source_checkpoint_sha256"],
        "source_feature_digest": report["source_feature_digest"], "feature_digest": report["feature_digest"],
        "code_snapshot_sha256": report["code_snapshot_sha256"]})
    performance = {}
    for method, metrics in report["methods"].items():
        performance[method] = {key: metrics.pop(key) for key in
            ("cached_compute_ms_per_transition", "end_to_end_latency_ms_per_transition", "latency_note",
             "peak_memory_bytes", "adaptation_seconds", "raw_frame_encoding_seconds",
             "source_retention_seconds") if key in metrics}
    performance_record = {"protocol_digest": protocol["digest"], "task": task,
        "budget": budget, "seed": seed, "phase_seconds": report.pop("phase_seconds"),
        "measurements": performance}
    _write_json(root / f"{report_path.stem}_performance.json", performance_record)
    _write_immutable_json(report_path,report)
    # Timings describe this attempt and must not change immutable acceptance
    # evidence. Return them explicitly for the calibration collector.
    return {**report, "performance": performance_record}
