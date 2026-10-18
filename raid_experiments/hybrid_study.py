"""Protocol v6 data, training, adaptation and recorded diagnostics.

Frame cache holds frozen reference-MAE features, proprio q, and actions.
Query shards never carry action labels. Source training fits one common
controller with a 0.5/0.5 mix of retrieval-present and masked-memory losses.
Target adaptation for lora_mlp and raid_lora_mlp shares ordering, steps, loss
and init; only the retrieval input differs. Reference features always come
from the frozen cache.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .models import attach_mae_lora
from .hybrid_heads import CommonController, chunk_mse, knn_average, make_bank, weight_hash
from .hybrid_protocol import (ACTION_DIM, CONDITION_NAMES, PRIMARY_BUDGET, PROTOCOL_V6, PROPRIO_DIM,
                              SUPPORT_BUDGETS, HeadSpec, RetrievalSpec, TemporalAlignment,
                              condition_flags, temporal_alignment, validate_hybrid_protocol)
from .hybrid_retrieval import null_block, retrieval_block

CACHE_FORMAT = 6
CHECKPOINT_FORMAT = 6
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
    with tmp.open('w') as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        handle.flush();os.fsync(handle.fileno())
    tmp.replace(path)


def _atomic_torch_save(path: Path, payload: dict) -> dict:
    """Commit a checkpoint atomically, then publish its checksum completion record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush();os.fsync(handle.fileno())
    tmp.replace(path)
    record = {"status": "complete", "file": path.name, "size_bytes": path.stat().st_size,
              "sha256": _sha_file(path), "protocol_digest": payload.get("protocol_digest"),
              "model_digest": payload.get("model_digest"), "scientific_result": False}
    _write_json(path.with_suffix(path.suffix + ".complete.json"), record)
    return record


def _verify_checkpoint_commit(path: str | Path, protocol: dict) -> dict:
    path = Path(path)
    marker = path.with_suffix(path.suffix + ".complete.json")
    if not marker.is_file():
        raise ValueError(f"checkpoint has no atomic completion record: {path}")
    record = json.loads(marker.read_text())
    if (record.get("status") != "complete" or record.get("file") != path.name
            or record.get("size_bytes") != path.stat().st_size
            or record.get("sha256") != _sha_file(path)
            or record.get("protocol_digest") != protocol.get("digest")
            or record.get("model_digest") != protocol.get("model_digest")):
        raise ValueError(f"checkpoint completion record or checksum is invalid: {path}")
    return record


def _training_binding(protocol: dict, cache: dict | None, settings: dict) -> dict:
    """Bind restart state to real dependencies, never just an output filename."""
    return {"protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
            "implementation_sha256": _sha_file(Path(__file__)),
            "runtime": {"python":list(sys.version_info[:3]),"torch":str(torch.__version__),
                        "cuda":torch.version.cuda,"container_image":os.environ.get('RAID_CONTAINER_IMAGE_DIGEST')},
            "cache_manifest_sha256": _sha_bytes(json.dumps(cache["manifest"], sort_keys=True).encode()) if cache else None,
            "settings": settings}


def _rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA training state cannot resume on a CPU runtime")
        torch.cuda.set_rng_state_all(state["cuda"])


def _load_training_restart(path: Path, protocol: dict, binding: dict) -> dict | None:
    pointer=path.with_suffix(path.suffix + ".latest.json")
    if pointer.exists():
        published=json.loads(pointer.read_text())
        name=published.get("file")
        if not isinstance(name,str) or Path(name).name!=name or not name.startswith(path.stem+"_u"):
            raise ValueError("unsafe training restart pointer")
        path=path.parent/name
        if _sha_file(path)!=published.get("sha256"):
            raise ValueError("training restart pointer checksum differs")
    marker=path.with_suffix(path.suffix + ".complete.json")
    if not path.exists() and not marker.exists():
        return None
    _verify_checkpoint_commit(path, protocol)
    state=torch.load(path, map_location="cpu", weights_only=False)
    if state.get("binding") != binding or state.get("scheduler") is not None:
        raise ValueError("training restart dependencies or scheduler differ")
    return state


def _save_training_restart(path: Path, protocol: dict, binding: dict, **state) -> None:
    import uuid
    version=path.with_name(f"{path.stem}_u{state['update_count']:06d}_{uuid.uuid4().hex[:8]}{path.suffix}")
    record=_atomic_torch_save(version, {"protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
                             "binding": binding, "scheduler": None, "rng": _rng_state(),
                             "scientific_result": False, **state})
    # A crash between file and receipt publication leaves the previous pointer
    # intact. Unpublished partial versions cannot become completed progress.
    _write_json(path.with_suffix(path.suffix + ".latest.json"), {"file":version.name,"sha256":record['sha256']})


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


def read_view_frames(protocol: dict, path: str | Path, demo: int) -> np.ndarray:
    """Recorded frames in the protocol's controller view (VPP2: the generated 4:3 band)."""
    from .hybrid_vpp2 import controller_view_fn
    return controller_view_fn(protocol)(read_frames(path, demo))


def read_actions(path: str | Path, demo: int) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[f"data/demo_{int(demo)}/actions"], dtype=np.float32)


def read_proprio(path: str | Path, demo: int, protocol: dict | None = None) -> np.ndarray:
    """Read joint7 + gripper2. Falls back to robot_states[:, :9] when obs keys are absent."""
    import h5py
    schema = (protocol or {}).get("design", {}).get("state_schema", {})
    joint_key = schema.get("hdf5_joint_key", "obs/joint_states")
    grip_key = schema.get("hdf5_gripper_key", "obs/gripper_states")
    with h5py.File(path, "r") as handle:
        root = handle[f"data/demo_{int(demo)}"]
        if joint_key.startswith("obs/"):
            jk, gk = joint_key.split("/", 1)[1], grip_key.split("/", 1)[1]
            if "obs" in root and jk in root["obs"] and gk in root["obs"]:
                j = np.asarray(root["obs"][jk], dtype=np.float32)
                g = np.asarray(root["obs"][gk], dtype=np.float32)
                return np.concatenate([j, g], axis=-1)
        if "robot_states" in root:
            rs = np.asarray(root["robot_states"], dtype=np.float32)
            if rs.shape[-1] >= PROPRIO_DIM:
                return rs[:, :PROPRIO_DIM]
        raise KeyError(f"{path} demo {demo} has no proprioception matching the state schema")


def read_initial_state(path: str | Path, demo: int) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[f"data/demo_{int(demo)}/states"][0])


@torch.no_grad()
def _encode(encoder, frames: np.ndarray, batch: int = 64) -> torch.Tensor:
    out = []
    for start in range(0, len(frames), batch):
        out.append(encoder.encode_frames(
            torch.from_numpy(np.ascontiguousarray(frames[start:start + batch]))).float().cpu())
    return torch.cat(out)


def encoder_identity(encoder) -> dict:
    state = getattr(encoder, "mae", encoder).state_dict()
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().float().numpy().tobytes())
    return {"encoder_class": type(encoder).__name__, "weights_sha256": digest.hexdigest(),
            "feature_dim": int(getattr(encoder, "feat_dim", 768))}


FINAL_STAGE = "narrow_final_confirmation_after_freeze"


def final_access(protocol: dict, freeze_file: str | Path | None) -> dict:
    """Controlled stage-3 access: a frozen amendment bound to this protocol unlocks the sealed final tasks.

    The record must say status "frozen" and stage FINAL_STAGE, and carry this protocol's model digest. Returns the
    record plus its SHA-256 (written into every final-stage output)."""
    if not freeze_file:
        raise ValueError("final-test access requires a freeze file")
    path = Path(freeze_file)
    record = json.loads(path.read_text())
    if (record.get("status") != "frozen" or record.get("stage") != FINAL_STAGE
            or record.get("model_digest") != protocol.get("model_digest")):
        raise ValueError("freeze record does not unlock the final stage of this protocol")
    return {**record, "freeze_sha256": _sha_file(path)}


def _final_allowed(protocol: dict, names, final: dict | None) -> bool:
    """True when the task set has no final task, or a verified final-access record is given."""
    if not any(protocol["tasks"][n]["role"] == "final_test" for n in names if n in protocol["tasks"]):
        return True
    return bool(final) and final.get("status") == "frozen" and final.get("stage") == FINAL_STAGE \
        and final.get("model_digest") == protocol.get("model_digest")


def cache_frame_features(protocol: dict, encoder, out_dir: str | Path, *,
                         run_mode: str = SCIENTIFIC,
                         task_allowlist: list[str] | tuple[str, ...] | None = None,
                         final: dict | None = None) -> dict:
    """Build the v6 reference cache (features + proprio + actions). New format; not a v4 cache.

    Final tasks need ``final`` (from final_access); development and final tasks never share one cache."""
    validate_hybrid_protocol(protocol)
    out = Path(out_dir)
    identity = encoder_identity(encoder)
    if task_allowlist is None:
        if run_mode == SCIENTIFIC:
            raise ValueError("scientific feature caching requires an explicit stage task allowlist")
        task_allowlist = tuple(protocol["tasks"])
    task_allowlist = tuple(dict.fromkeys(map(str, task_allowlist)))
    unknown = set(task_allowlist) - set(protocol["tasks"])
    if unknown:
        raise ValueError(f"cache task allowlist contains unknown tasks: {sorted(unknown)}")
    if run_mode == SCIENTIFIC and not _final_allowed(protocol, task_allowlist, final):
        raise ValueError("scientific stage cache may not include sealed final tasks")
    roles = {protocol["tasks"][name]["role"] for name in task_allowlist}
    if run_mode == SCIENTIFIC and {"development", "final_test"} <= roles:
        raise ValueError("development and final tasks never share one cache")
    if not task_allowlist:
        raise ValueError("cache task allowlist must not be empty")
    shards = {}
    for task in task_allowlist:
        spec = protocol["tasks"][task]
        if run_mode == SCIENTIFIC and _sha_file(Path(spec["path"])) != spec["source_sha256"]:
            raise ValueError(f"HDF5 source hash mismatch for {task}")
        shards[task] = {}
        for demo in spec["demo_ids"]:
            role = demo_role(spec, demo)
            frames = read_view_frames(protocol, spec["path"], demo)
            proprio = read_proprio(spec["path"], demo, protocol)
            if len(proprio) != len(frames):
                raise ValueError(f"{task}:{demo} frame/proprio count mismatch")
            payload: dict[str, Any] = {
                "task": task, "demo_id": int(demo), "role": role,
                "features": _encode(encoder, frames),
                "proprio": torch.from_numpy(np.asarray(proprio, dtype=np.float32)),
                "frame_count": int(len(frames)),
                "protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
                "encoder": identity,
            }
            if role != "query":
                actions = read_actions(spec["path"], demo)
                if len(actions) != len(frames):
                    raise ValueError(f"{task}:{demo} frame/action count mismatch")
                payload["actions"] = torch.from_numpy(actions)
            rel = Path(task) / f"demo_{int(demo)}.pt"
            (out / task).mkdir(parents=True, exist_ok=True)
            shard_path = out / rel
            tmp = shard_path.with_suffix(shard_path.suffix + ".tmp")
            torch.save(payload, tmp)
            tmp.replace(shard_path)
            shards[task][str(int(demo))] = {"path": rel.as_posix(), "role": role, "sha256": _sha_file(out / rel)}
    manifest = {"format": CACHE_FORMAT, "protocol": PROTOCOL_V6, "protocol_digest": protocol["digest"],
                "model_digest": protocol["model_digest"], "execution_mode": run_mode, "encoder": identity,
                **({"final_freeze_sha256": final["freeze_sha256"]} if final and "final_test" in roles else {}),
                "stage_task_allowlist": list(task_allowlist), "shards": shards,
                "query_labels_present": False, "includes_proprio": True}
    _write_json(out / "manifest.json", manifest)
    return manifest


def load_frame_cache(cache_dir: str | Path, protocol: dict, *, run_mode: str = SCIENTIFIC,
                     final: dict | None = None) -> dict:
    root = Path(cache_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("model_digest") != protocol["model_digest"] or manifest.get("format") != CACHE_FORMAT:
        raise ValueError("frame cache belongs to another protocol (v4 caches are not reusable)")
    if manifest.get("execution_mode") != run_mode:
        raise ValueError(f"{run_mode} execution requires a {run_mode} frame cache")
    cached_tasks = set(manifest.get("shards", {}))
    if run_mode == SCIENTIFIC and (
            not cached_tasks
            or any(name not in protocol["tasks"] for name in cached_tasks)
            or not _final_allowed(protocol, cached_tasks, final)
            or set(manifest.get("stage_task_allowlist", [])) != cached_tasks):
        raise ValueError("scientific cache contains unregistered tasks or sealed final-task data")
    cache: dict[str, dict[int, dict]] = {}
    for task, records in manifest["shards"].items():
        cache[task] = {}
        if task not in protocol['tasks'] or set(map(int,records))!=set(map(int,protocol['tasks'][task]['demo_ids'])):
            raise ValueError('cache task/demo membership differs from registered inputs')
        for demo, record in records.items():
            relative=Path(record["path"])
            path = (root / relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root.resolve()):
                raise ValueError('cache shard path escapes collection root')
            if _sha_file(path) != record["sha256"]:
                raise ValueError(f"stale frame shard {path}")
            shard = torch.load(path, map_location="cpu", weights_only=False)
            expected_role=demo_role(protocol['tasks'][task],int(demo))
            if (record.get('role')!=expected_role or shard.get('role')!=expected_role
                    or shard.get('task')!=task or shard.get('demo_id')!=int(demo)
                    or shard.get('protocol_digest')!=protocol['digest']
                    or shard.get('model_digest')!=protocol['model_digest']
                    or shard.get('encoder')!=manifest['encoder']):
                raise ValueError('cache shard input, split or encoder provenance differs')
            if shard["role"] == "query" and "actions" in shard:
                raise ValueError("query shard contains forbidden action labels")
            cache[task][int(demo)] = shard
    return {"manifest": manifest, "shards": cache}


def source_normalizers(protocol: dict, cache: dict) -> dict[str, torch.Tensor]:
    """Fit action and proprio normalizers on source-training demos only."""
    actions, proprios = [], []
    for task, spec in protocol["tasks"].items():
        if spec["role"] != "source":
            continue
        for d in spec["source"]["source_train"]:
            shard = cache["shards"][task][int(d)]
            actions.append(shard["actions"])
            proprios.append(shard["proprio"])
    a = torch.cat(actions).float()
    q = torch.cat(proprios).float()
    a_mean, a_std = a.mean(0), a.std(0, unbiased=False)
    q_mean, q_std = q.mean(0), q.std(0, unbiased=False)
    a_std = torch.where(a_std < 1e-6, torch.ones_like(a_std), a_std)
    q_std = torch.where(q_std < 1e-6, torch.ones_like(q_std), q_std)
    return {"action_mean": a_mean, "action_std": a_std, "proprio_mean": q_mean, "proprio_std": q_std}


@dataclass
class ChunkRows:
    current: torch.Tensor
    future: torch.Tensor
    q: torch.Tensor
    target: torch.Tensor | None
    duration: torch.Tensor
    group_id: torch.Tensor
    demo: list[int]
    step: list[int]
    task: str

    def __len__(self):
        return len(self.current)


def chunk_rows(shard_by_demo: dict[int, dict], demos: list[int], task: str, chunk: int,
               norms: dict[str, torch.Tensor] | None = None, *,
               actions_override: dict[int, np.ndarray] | None = None,
               require_complete: bool = True) -> ChunkRows:
    cur, fut, qs, tgt, dur, gid, dd, ss = [], [], [], [], [], [], [], []
    a_mean = norms["action_mean"] if norms else None
    a_std = norms["action_std"] if norms else None
    q_mean = norms["proprio_mean"] if norms else None
    q_std = norms["proprio_std"] if norms else None
    for demo in demos:
        shard = shard_by_demo[int(demo)]
        feats = shard["features"].float()
        proprio = shard["proprio"].float()
        actions = (torch.as_tensor(actions_override[int(demo)]) if actions_override and int(demo) in actions_override
                   else shard.get("actions"))
        if len(feats) <= chunk:
            if require_complete:
                continue
            raise ValueError(f"incomplete chunk for {task}:{demo}")
        for t in range(0, len(feats) - chunk):
            if actions is not None and t + chunk > len(actions):
                raise ValueError(f"incomplete action chunk for {task}:{demo} at {t}")
            cur.append(feats[t]); fut.append(feats[t + chunk])
            q = proprio[t]
            if q_mean is not None:
                q = (q - q_mean) / q_std
            qs.append(q)
            if actions is not None:
                chunk_a = actions[t:t + chunk].float()
                if a_mean is not None:
                    chunk_a = (chunk_a - a_mean) / a_std
                tgt.append(chunk_a)
            dur.append(float(chunk))
            gid.append(demo_uid(task, demo))
            dd.append(int(demo)); ss.append(t)
    if not cur:
        raise ValueError(f"no complete {chunk}-step chunks for {task}:{demos}")
    return ChunkRows(
        torch.stack(cur), torch.stack(fut), torch.stack(qs),
        torch.stack(tgt) if tgt else None,
        torch.tensor(dur, dtype=torch.float32),
        torch.tensor(gid, dtype=torch.long), dd, ss, task)


def _source_tasks(protocol: dict) -> list[str]:
    return sorted(t for t, s in protocol["tasks"].items() if s["role"] == "source")


def _validation_bank_demos(protocol: dict, task: str, seed: int, m: int = 5) -> list[int]:
    ids = sorted(map(int, protocol["tasks"][task]["source"]["source_train"]))
    random.Random(f"val-bank:{seed}:{task}").shuffle(ids)
    return ids[:m]


def _schedule(protocol: dict, seed: int, updates: int) -> list[dict]:
    rng = random.Random(f"v6-source:{seed}")
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


def _as_retrieval(payload: dict) -> RetrievalSpec:
    return RetrievalSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in payload.items()})


def _validation(protocol, cache, chunk, norms, seed, controller, retrieval, device) -> dict:
    losses = {"retrieval_present": [], "masked_memory": [], "mean": []}
    controller.eval()
    with torch.no_grad():
        for task in _source_tasks(protocol):
            shards = cache["shards"][task]
            val = chunk_rows(shards, protocol["tasks"][task]["source"]["source_validation"],
                             task, chunk, norms)
            bank_demos = _validation_bank_demos(protocol, task, seed)
            br = chunk_rows(shards, bank_demos, task, chunk, norms)
            bank = make_bank(br.current, br.future, br.q, br.target, br.duration, br.group_id, device)
            cur = val.current.to(device); fut = val.future.to(device)
            q = val.q.to(device); dur = val.duration.to(device); tgt = val.target.to(device)
            pred_r, _ = controller.forward_with_memory(cur, fut, q, dur, bank=bank, use_retrieval=True,
                                                       exclude_groups=val.group_id.to(device))
            pred_m, _ = controller.forward_with_memory(cur, fut, q, dur, bank=None, use_retrieval=False)
            lr = float(chunk_mse(pred_r, tgt).mean())
            lm = float(chunk_mse(pred_m, tgt).mean())
            losses["retrieval_present"].append(lr)
            losses["masked_memory"].append(lm)
            losses["mean"].append(0.5 * lr + 0.5 * lm)
    controller.train()
    return {k: float(np.mean(v)) for k, v in losses.items()}


def train_source(protocol: dict, cache: dict, out_dir: str | Path, *, seed: int, device: str = "cpu",
                 updates: int = 2000, batch: int = 64, lr: float = 3e-4, weight_decay: float = 1e-4,
                 validate_every: int = 100, run_mode: str = SCIENTIFIC, head_kind: str = "mlp_v6",
                 retrieval_override: RetrievalSpec | None = None) -> dict:
    """Source-train one common controller with dual-mode (retrieval + null) objective."""
    validate_hybrid_protocol(protocol)
    design = protocol["design"]
    temporal = temporal_alignment(protocol)  # Cosmos or VPP2 timing class
    chunk = temporal.chunk_steps
    if run_mode == SCIENTIFIC and (updates, batch) != (design["training"]["source_updates"],
                                                       design["training"]["batch_size"]):
        raise ValueError("scientific source training uses the fixed protocol settings")
    retrieval = retrieval_override or _as_retrieval(design["retrieval"])
    head = HeadSpec(**design["head"])
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = torch.device(device)
    norms = source_normalizers(protocol, cache)
    feature_dim = next(iter(next(iter(cache["shards"].values())).values()))["features"].shape[-1]
    proprio_dim = next(iter(next(iter(cache["shards"].values())).values()))["proprio"].shape[-1]
    controller = _new_controller(head_kind, chunk, feature_dim, proprio_dim, head, retrieval).to(dev)
    opt = torch.optim.AdamW(controller.parameters(), lr=lr, weight_decay=weight_decay)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    restart_path = out / f"source_v6_seed{seed}.restart.pt"
    extra = {} if head_kind == "mlp_v6" else {"head_kind": head_kind, "retrieval": asdict(retrieval)}
    binding = _training_binding(protocol, cache, {"kind":"source", "seed":seed, "updates":updates, **extra,
        "batch":batch, "lr":lr, "weight_decay":weight_decay, "validate_every":validate_every,
        "execution_mode":run_mode, "device":str(dev)})
    restart = _load_training_restart(restart_path, protocol, binding)
    rows_cache: dict[tuple, ChunkRows] = {}
    curve, best = [], (float("inf"), 0, None)
    forward_passes = 0
    completed, elapsed = 0, 0.0
    if restart:
        controller.load_state_dict(restart["controller"])
        opt.load_state_dict(restart["optimizer"])
        completed, elapsed = restart["sampler_position"], restart["elapsed_seconds"]
        if not isinstance(completed,int) or not 0<=completed<=updates:
            raise ValueError("invalid source restart sampler position")
        curve, best, forward_passes = restart["curve"], restart["best"], restart["forward_passes"]
        _restore_rng(restart["rng"])
    started = time.perf_counter()
    for step, item in enumerate(_schedule(protocol, seed, updates), start=1):
        if step<=completed:
            continue
        task, shards = item["task"], cache["shards"][item["task"]]
        qkey = (task, tuple(item["query_demos"]))
        if qkey not in rows_cache:
            rows_cache[qkey] = chunk_rows(shards, item["query_demos"], task, chunk, norms)
        qrows = rows_cache[qkey]
        idx = _sample_rows(qrows, batch, item["rng"])
        cur = qrows.current[idx].to(dev); fut = qrows.future[idx].to(dev)
        q = qrows.q[idx].to(dev); dur = qrows.duration[idx].to(dev)
        tgt = qrows.target[idx].to(dev)
        groups = qrows.group_id[idx].to(dev)
        br = chunk_rows(shards, item["bank"], task, chunk, norms)
        bank = make_bank(br.current, br.future, br.q, br.target, br.duration, br.group_id, dev)
        opt.zero_grad(set_to_none=True)
        pred_r, info_r = controller.forward_with_memory(cur, fut, q, dur, bank=bank, use_retrieval=True,
                                                        exclude_groups=groups)
        pred_m, _ = controller.forward_with_memory(cur, fut, q, dur, bank=None, use_retrieval=False)
        forward_passes += 2
        loss = 0.5 * chunk_mse(pred_r, tgt).mean() + 0.5 * chunk_mse(pred_m, tgt).mean()
        loss = loss + _selection_term(controller, info_r, tgt.reshape(len(tgt), -1))
        loss.backward()
        opt.step()
        if step % validate_every == 0 or step == updates:
            scores = _validation(protocol, cache, chunk, norms, seed, controller, retrieval, dev)
            curve.append({"step": step, **scores})
            if scores["mean"] < best[0]:
                best = (scores["mean"], step,
                        {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()})
            _save_training_restart(restart_path, protocol, binding,
                controller=controller.state_dict(), optimizer=opt.state_dict(),
                sampler_position=step, update_count=step, curve=curve, best=best,
                forward_passes=forward_passes, elapsed_seconds=elapsed+time.perf_counter()-started)
    seconds = elapsed + time.perf_counter() - started
    payload = {
        "format": CHECKPOINT_FORMAT, "protocol": PROTOCOL_V6,
        "protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
        "execution_mode": run_mode, "seed": seed, "chunk_steps": chunk,
        "feature_dim": feature_dim, "proprio_dim": proprio_dim,
        "normalizers": norms, "head_spec": design["head"],
        "retrieval_spec": design["retrieval"] if retrieval_override is None else asdict(retrieval),
        **({} if head_kind == "mlp_v6" else {"head_kind": head_kind}),
        "updates": updates, "batch": batch, "lr": lr, "weight_decay": weight_decay,
        "curve": curve, "forward_passes": forward_passes,
        "controller": {"state_dict": best[2], "best_step": best[1], "best_validation_mean": best[0]},
        "cache_identity": cache["manifest"]["encoder"], "training_seconds": seconds,
        "training_dependency_binding": binding,
        "scientific_result": False,
    }
    path = out / f"source_v6_seed{seed}.pt"
    _atomic_torch_save(path, payload)
    summary = {k: v for k, v in payload.items() if k not in {"controller", "normalizers"}}
    summary["best"] = {"step": best[1], "validation_mean": best[0]}
    summary["checkpoint_sha256"] = _sha_file(path)
    _write_json(out / f"source_v6_seed{seed}.json", summary)
    return {"path": str(path), **summary}


def load_source(path: str | Path, protocol: dict) -> dict:
    _verify_checkpoint_commit(path, protocol)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_digest") != protocol["model_digest"] or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("source checkpoint belongs to another protocol")
    binding=payload.get('training_dependency_binding')
    if payload.get('execution_mode')==SCIENTIFIC and (not binding or binding.get('implementation_sha256')!=_sha_file(Path(__file__))):
        raise ValueError('scientific source checkpoint lacks matching training implementation dependencies')
    return payload


def _new_controller(head_kind: str, chunk: int, feature_dim: int, proprio_dim: int, head: HeadSpec,
                    retrieval: RetrievalSpec):
    if head_kind == "mlp_v6":
        return CommonController(chunk, feature_dim, proprio_dim, head, retrieval)
    from .hybrid_v7 import HEAD_KIND, AttentionGateController
    if head_kind != HEAD_KIND:
        raise ValueError(f"unknown controller head kind {head_kind}")
    return AttentionGateController(chunk, feature_dim, proprio_dim, head, retrieval)


def _selection_term(controller, info: dict, target: torch.Tensor) -> torch.Tensor:
    """v7 only: weighted selection loss; v6 heads add nothing."""
    if getattr(controller, "head_kind", "mlp_v6") == "mlp_v6":
        return target.new_zeros(())
    from .hybrid_v7 import SELECTION_WEIGHT, selection_loss
    return SELECTION_WEIGHT * selection_loss(info, target)


def build_controller(source: dict, device: str = "cpu") -> CommonController:
    head = HeadSpec(**source["head_spec"])
    retrieval = _as_retrieval(source["retrieval_spec"])
    controller = _new_controller(source.get("head_kind", "mlp_v6"), source["chunk_steps"], source["feature_dim"],
                                 source["proprio_dim"], head, retrieval)
    controller.load_state_dict(source["controller"]["state_dict"])
    return controller.to(device).eval()


def support_ids(protocol: dict, task: str, budget: int, seed: int) -> list[int]:
    spec = protocol["tasks"][task]
    if spec["role"] == "source" or budget not in SUPPORT_BUDGETS:
        raise ValueError("support sets exist only for held-out tasks and registered budgets")
    return list(map(int, spec["support_budgets_by_seed"][str(seed)][str(budget)]))


def build_support_bank(protocol: dict, cache: dict, source: dict, task: str, budget: int,
                       seed: int, device: str = "cpu",
                       exclude_demo: int | None = None) -> tuple[Any, dict]:
    """Populate a reference bank from the support set (optionally excluding one parent demo)."""
    started = time.perf_counter()
    ids = support_ids(protocol, task, budget, seed)
    if exclude_demo is not None:
        ids = [d for d in ids if d != int(exclude_demo)]
    rows = chunk_rows(cache["shards"][task], ids, task, source["chunk_steps"], source["normalizers"])
    bank = make_bank(rows.current, rows.future, rows.q, rows.target, rows.duration, rows.group_id, device)
    seconds = time.perf_counter() - started
    return bank, {"support_demo_ids": ids, "excluded_demo": exclude_demo,
                  "bank_entries": len(rows), "bank_construction_seconds": seconds,
                  "source_transitions": 0}


class _SupportLive(torch.utils.data.Dataset):
    """Live RGB chunks for LoRA adaptation (controller encoder sees raw frames)."""

    def __init__(self, frames: dict[int, np.ndarray], actions: dict[int, np.ndarray],
                 proprio: dict[int, np.ndarray], chunk: int, norms: dict[str, torch.Tensor]):
        self.items = [(d, t) for d in sorted(frames) for t in range(len(frames[d]) - chunk)]
        self.frames, self.actions, self.proprio = frames, actions, proprio
        self.chunk = chunk
        self.norms = norms

    def __len__(self):
        return len(self.items)

    def get(self, indices: torch.Tensor):
        cur, fut, q, tgt, groups, demos, steps = [], [], [], [], [], [], []
        for i in indices.tolist():
            d, t = self.items[i]
            cur.append(self.frames[d][t]); fut.append(self.frames[d][t + self.chunk])
            qv = (torch.from_numpy(self.proprio[d][t]) - self.norms["proprio_mean"]) / self.norms["proprio_std"]
            q.append(qv)
            a = (torch.from_numpy(self.actions[d][t:t + self.chunk]) - self.norms["action_mean"]) / self.norms["action_std"]
            tgt.append(a.float())
            groups.append(demo_uid("support", d))  # unique per demo; task tag unused for exclusion
            demos.append(d)
            steps.append(t)
        return (torch.from_numpy(np.stack(cur)), torch.from_numpy(np.stack(fut)),
                torch.stack(q).float(), torch.stack(tgt),
                torch.tensor(groups, dtype=torch.long), demos, steps)


def adapt_target(protocol: dict, source: dict, encoder_factory: Callable[[], Any],
                 condition: str, task: str, budget: int, seed: int, out_dir: str | Path, *,
                 cache: dict | None = None, device: str = "cpu", run_mode: str = SCIENTIFIC,
                 updates: int | None = None, source_checkpoint_sha256: str | None = None,
                 cache_manifest_sha256: str | None = None, checkpoint_callback: Callable | None = None,
                 checkpoint_steps: tuple[int, ...] = ()) -> dict:
    """Adapt LoRA + head for lora_mlp / raid_lora_mlp. Identical schedule; retrieval differs.

    ``checkpoint_callback(step, controller, encoder, adaptation_seconds)`` (compute benchmark only) runs after
    the listed updates; its own time is excluded from ``adaptation_seconds``.
    """
    flags = condition_flags(condition)
    if not flags["A"]:
        raise ValueError(f"{condition} has A=0; use bank-only or frozen path")
    design = protocol["design"]
    cfg, lora = design["training"], design["inverse_model"]["lora"]
    updates = updates if updates is not None else cfg["target_updates"]
    if run_mode == SCIENTIFIC and updates != cfg["target_updates"]:
        raise ValueError("scientific adaptation uses the fixed pilot update count")
    if run_mode == SCIENTIFIC and budget != PRIMARY_BUDGET:
        # primary pilot budget is 5; other budgets allowed as recorded diagnostics only when declared
        pass
    retrieval = _as_retrieval(source["retrieval_spec"])
    dev = torch.device(device)
    torch.manual_seed(seed)
    encoder = encoder_factory()
    names = attach_mae_lora(encoder.mae, lora["blocks"], lora["rank"], lora["alpha"], lora["dropout"])
    # Verify zero initial effect: B is zeros so LoRA adds nothing
    controller = build_controller(source, device)
    controller.train()
    init_hash = weight_hash(controller)
    chunk = source["chunk_steps"]
    ids = support_ids(protocol, task, budget, seed)
    path = protocol["tasks"][task]["path"]
    frames = {d: read_view_frames(protocol, path, d) for d in ids}
    actions = {d: read_actions(path, d) for d in ids}
    proprio = {d: read_proprio(path, d, protocol) for d in ids}
    data = _SupportLive(frames, actions, proprio, chunk, source["normalizers"])
    # Reference bank for hybrid: from frozen cache features, excluding the query demo per batch item
    if flags["R"] and cache is None:
        raise ValueError("raid_lora_mlp adaptation requires the reference frame cache")
    lora_params = [p for p in encoder.mae.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": controller.parameters(), "lr": cfg["head_lr"]},
                             {"params": lora_params, "lr": cfg["lora_lr"]}],
                            weight_decay=cfg["weight_decay"])
    batch, micro = cfg["batch_size"], cfg["microbatch_size"]
    gen = torch.Generator().manual_seed(int(_sha_bytes(f"v6:{condition}:{task}:{budget}:{seed}".encode())[:8], 16))
    # Shared stream key so lora_mlp and raid_lora_mlp see the same order when condition string is swapped
    # Use a condition-independent seed for the sample stream:
    gen = torch.Generator().manual_seed(int(_sha_bytes(f"v6adapt:{task}:{budget}:{seed}".encode())[:8], 16))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    restart_path = out / f"{condition}_{task}_b{budget}_s{seed}.restart.pt"
    binding = _training_binding(protocol, cache, {"kind":"adaptation", "task":task,
        "condition":condition,"budget":budget,"seed":seed,"updates":updates,
        "training":cfg,"execution_mode":run_mode,"device":str(dev),
        "source_checkpoint_sha256":source_checkpoint_sha256,
        "source_controller_hash":init_hash,"cache_manifest_sha256":cache_manifest_sha256,
        "mae_weights_hash":weight_hash(encoder.mae)})
    restart = _load_training_restart(restart_path, protocol, binding)
    completed, elapsed = 0, 0.0
    started = time.perf_counter()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    # Build the hybrid bank once from frozen reference cache features.
    bank = None
    if flags["R"]:
        br = chunk_rows(cache["shards"][task], ids, task, chunk, source["normalizers"])
        bank = make_bank(br.current, br.future, br.q, br.target, br.duration, br.group_id, dev)
    if restart:
        controller.load_state_dict(restart["controller"])
        incompatible=encoder.mae.load_state_dict(restart["encoder"],strict=False)
        if incompatible.unexpected_keys or any('lora_' in k for k in incompatible.missing_keys):
            raise ValueError("adaptation restart LoRA structure differs")
        opt.load_state_dict(restart["optimizer"])
        gen.set_state(restart["sampler_rng"])
        completed, elapsed = restart["sampler_position"], restart["elapsed_seconds"]
        if not isinstance(completed,int) or not 0<=completed<=updates:
            raise ValueError("invalid adaptation restart sampler position")
        _restore_rng(restart["rng"])
    for step in range(completed+1, updates + 1):
        opt.zero_grad(set_to_none=True)
        indices = torch.randint(0, len(data), (batch,), generator=gen)
        for part in indices.split(micro):
            cur_img, fut_img, q, tgt, groups, demos, steps = data.get(part)
            feats = encoder.encode_frames(torch.cat((cur_img, fut_img)))
            fc, ff = feats[:len(part)], feats[len(part):]
            dur = torch.full((len(part),), float(chunk), device=dev)
            q = q.to(dev); tgt = tgt.to(dev)
            if flags["R"]:
                # Retrieve with frozen reference features; control with adapted features.
                ref_cur = torch.stack([
                    cache["shards"][task][int(d)]["features"][int(t)]
                    for d, t in zip(demos, steps)
                ]).to(dev)
                ref_fut = torch.stack([
                    cache["shards"][task][int(d)]["features"][int(t) + chunk]
                    for d, t in zip(demos, steps)
                ]).to(dev)
                exclude = torch.tensor([demo_uid(task, d) for d in demos], dtype=torch.long, device=dev)
                pred, info = controller.forward_with_memory(
                    fc, ff, q, dur, bank=bank, use_retrieval=True,
                    exclude_groups=exclude, query_current=ref_cur, query_future=ref_fut)
                selection = _selection_term(controller, info, tgt.reshape(len(tgt), -1))
            else:
                pred, _ = controller.forward_with_memory(fc, ff, q, dur, bank=None, use_retrieval=False)
                selection = tgt.new_zeros(())
            loss = (chunk_mse(pred, tgt).mean() + selection) * (len(part) / batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(list(controller.parameters()) + lora_params, cfg["grad_clip"])
        opt.step()
        if checkpoint_callback is not None and step in checkpoint_steps:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            paused = time.perf_counter()
            checkpoint_callback(step, controller, encoder, elapsed + paused - started)
            started += time.perf_counter() - paused  # exclude evaluation time from adaptation_seconds
        if step%25==0 or step==updates:
            _save_training_restart(restart_path, protocol, binding,
                controller=controller.state_dict(), encoder={k:v for k,v in encoder.mae.state_dict().items() if 'lora_' in k},
                optimizer=opt.state_dict(), sampler_rng=gen.get_state(),
                sampler_position=step, update_count=step,
                elapsed_seconds=elapsed+time.perf_counter()-started)
    snap = {
        "step": updates,
        "condition": condition,
        "head_state": {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()},
        "lora_state": {k: v.detach().cpu().clone() for k, v in encoder.mae.state_dict().items() if "lora_" in k},
        "adaptation_seconds": elapsed + time.perf_counter() - started,
        "init_controller_hash": init_hash,
        "final_controller_hash": weight_hash(controller),
    }
    snap_path = out / f"{condition}_{task}_b{budget}_s{seed}_u{updates}.pt"
    snap_payload = {**snap, "protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
                    "task": task, "budget": budget, "seed": seed, "support_demo_ids": ids,
                    "lora_targets": names, "R": flags["R"], "A": flags["A"],
                    "source_checkpoint_sha256": source_checkpoint_sha256,
                    "training_dependency_binding": binding,
                    "cache_manifest_sha256": cache_manifest_sha256,
                    "cache_encoder_sha256": cache["manifest"]["encoder"]["weights_sha256"] if cache else None}
    _atomic_torch_save(snap_path, snap_payload)
    trainable = sum(p.numel() for p in controller.parameters()) + sum(p.numel() for p in lora_params)
    summary = {"condition": condition, "arm": condition, "task": task, "budget": budget, "seed": seed,
               "support_demo_ids": ids, "updates": updates, "path": str(snap_path),
               "sha256": _sha_file(snap_path), "trainable_parameters": int(trainable),
               "lora_targets": names, "R": flags["R"], "A": flags["A"],
               "adaptation_seconds": snap["adaptation_seconds"],
               "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None,
               "execution_mode": run_mode, "scientific_result": False}
    _write_json(out / f"{condition}_{task}_b{budget}_s{seed}.json", summary)
    return summary


def prepare_frozen_or_raid(protocol: dict, source: dict, cache: dict, condition: str,
                           task: str, budget: int, seed: int, out_dir: str | Path,
                           device: str = "cpu") -> dict:
    """No-gradient conditions: record weight hashes and optional bank membership."""
    flags = condition_flags(condition)
    if flags["A"]:
        raise ValueError(f"{condition} has A=1; use adapt_target")
    controller = build_controller(source, device)
    payload = {"condition": condition, "R": flags["R"], "A": 0, "task": task, "budget": budget,
               "seed": seed, "controller_hash": weight_hash(controller),
               "gradient_updates": 0, "scientific_result": False,
               "protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"]}
    if flags["R"]:
        bank, info = build_support_bank(protocol, cache, source, task, budget, seed, device)
        payload["bank"] = info
        payload["bank_entries"] = info["bank_entries"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / f"{condition}_{task}_b{budget}_s{seed}.json", payload)
    return payload


def load_adapted(path: str | Path, encoder_factory: Callable[[], Any], source: dict,
                 protocol: dict, device: str = "cpu"):
    _verify_checkpoint_commit(path, protocol)
    snap = torch.load(path, map_location="cpu", weights_only=False)
    if snap.get("model_digest") != protocol["model_digest"]:
        raise ValueError("adaptation snapshot belongs to another protocol")
    lora = protocol["design"]["inverse_model"]["lora"]
    encoder = encoder_factory()
    attach_mae_lora(encoder.mae, lora["blocks"], lora["rank"], lora["alpha"], lora["dropout"])
    missing = encoder.mae.load_state_dict(snap["lora_state"], strict=False)
    if missing.unexpected_keys:
        raise ValueError(f"unexpected LoRA keys {missing.unexpected_keys}")
    controller = build_controller(source, device)
    controller.load_state_dict(snap["head_state"])
    encoder.mae.eval()
    return encoder, controller.eval(), snap


def _macro(errors: torch.Tensor, demos: list[int]) -> float:
    per_demo: dict[int, list[float]] = {}
    for value, demo in zip(errors.tolist(), demos):
        per_demo.setdefault(demo, []).append(value)
    return float(np.mean([np.mean(v) for v in per_demo.values()]))


def _append_query_access(ledger: Path, record: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def evaluate_recorded(protocol: dict, cache: dict, source: dict, task: str, budget: int,
                      seed: int, *, out_dir: str | Path, encoder_factory: Callable[[], Any],
                      adapted: dict[str, str | Path] | None = None, device: str = "cpu") -> dict:
    """Recorded-transition diagnostic for all four conditions (+ diagnostics)."""
    spec = protocol["tasks"][task]
    if spec["role"] == "source":
        raise ValueError("recorded diagnostic evaluates held-out tasks")
    chunk = source["chunk_steps"]
    norms = source["normalizers"]
    retrieval = _as_retrieval(source["retrieval_spec"])
    query = list(map(int, spec["query"]))
    out = Path(out_dir)
    _append_query_access(out / "query_access.jsonl", {
        "task": task, "budget": budget, "seed": seed, "demos": query,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "recorded_transition_diagnostic_v6",
    })
    labels = {d: read_actions(spec["path"], d) for d in query}
    rows = chunk_rows(cache["shards"][task], query, task, chunk, norms, actions_override=labels)
    target = rows.target.to(device)
    cur = rows.current.to(device); fut = rows.future.to(device)
    q = rows.q.to(device); dur = rows.duration.to(device)
    controller = build_controller(source, device)
    bank, bank_info = build_support_bank(protocol, cache, source, task, budget, seed, device)
    results, costs = {}, {"bank": bank_info}
    adapted = adapted or {}
    with torch.no_grad():
        # source_frozen
        pred, _ = controller.forward_with_memory(cur, fut, q, dur, bank=None, use_retrieval=False)
        results["source_frozen"] = {"chunk_mse_macro": _macro(chunk_mse(pred, target).cpu(), rows.demo),
                                    "R": 0, "A": 0, "role": "experimental_condition"}
        # raid_bank_mlp
        pred, info = controller.forward_with_memory(cur, fut, q, dur, bank=bank, use_retrieval=True)
        results["raid_bank_mlp"] = {"chunk_mse_macro": _macro(chunk_mse(pred, target).cpu(), rows.demo),
                                    "R": 1, "A": 0, "role": "experimental_condition",
                                    "controller_hash_unchanged": True}
        # knn diagnostic
        knn = knn_average(bank, cur, fut, q, dur, spec=retrieval, chunk_steps=chunk)
        results["support_knn_average"] = {"chunk_mse_macro": _macro(chunk_mse(knn, target).cpu(), rows.demo),
                                          "role": "diagnostic_control"}
        for cond in ("lora_mlp", "raid_lora_mlp"):
            if cond not in adapted:
                continue
            enc, head, snap = load_adapted(adapted[cond], encoder_factory, source, protocol, device)
            # Controller features from the adapted encoder; retrieval uses frozen cache features.
            feats = {d: _encode(enc, read_view_frames(protocol, spec["path"], d)) for d in query}
            lc = torch.stack([feats[d][t] for d, t in zip(rows.demo, rows.step)]).to(device)
            lf = torch.stack([feats[d][t + chunk] for d, t in zip(rows.demo, rows.step)]).to(device)
            use_r = condition_flags(cond)["R"]
            pred, _ = head.forward_with_memory(
                lc, lf, q, dur, bank=bank if use_r else None, use_retrieval=use_r,
                query_current=cur if use_r else None, query_future=fut if use_r else None)
            results[cond] = {"chunk_mse_macro": _macro(chunk_mse(pred, target).cpu(), rows.demo),
                             "R": condition_flags(cond)["R"], "A": 1,
                             "selected_step": snap["step"], "role": "experimental_condition"}
    report = {
        "protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
        "execution_mode": source.get("execution_mode", SCIENTIFIC),
        "setting": "recorded_transition_diagnostic",
        "task": task, "role": spec["role"], "budget": budget, "seed": seed,
        "chunk_steps": chunk, "query_demos": query,
        "support_demo_ids": support_ids(protocol, task, budget, seed),
        "transitions": len(rows), "results": results, "costs": costs,
        "note": "recorded-action error; does not evaluate execution of generated plans",
        "scientific_result": False,
    }
    step = next((results[c]["selected_step"] for c in ("lora_mlp", "raid_lora_mlp") if c in results), 0)
    _write_json(out / f"recorded_{task}_b{budget}_s{seed}_u{step}.json", report)
    return report


def evaluate_retention(protocol: dict, cache: dict, source: dict, task: str, budget: int,
                       seed: int, *, out_dir: str | Path, encoder_factory: Callable[[], Any],
                       adapted: dict[str, str | Path] | None = None, device: str = "cpu",
                       run_mode: str = SCIENTIFIC) -> dict:
    """Source retention with each condition's actual target configuration active."""
    if protocol["tasks"][task]["role"] == "source":
        raise ValueError("retention is recorded per held-out condition")
    chunk = source["chunk_steps"]
    norms = source["normalizers"]
    adapted = adapted or {}
    frozen = build_controller(source, device)
    target_bank, _ = build_support_bank(protocol, cache, source, task, budget, seed, device)
    per_condition: dict[str, dict] = {}
    with torch.no_grad():
        for source_task in _source_tasks(protocol):
            spec = protocol["tasks"][source_task]
            retention_ids = list(map(int, spec["source"]["source_retention"]))
            held = chunk_rows(cache["shards"][source_task], retention_ids, source_task, chunk, norms)
            cur = held.current.to(device); fut = held.future.to(device)
            q = held.q.to(device); dur = held.duration.to(device); target = held.target.to(device)
            # source_frozen: null memory, source weights
            pred, _ = frozen.forward_with_memory(cur, fut, q, dur, bank=None, use_retrieval=False)
            row = {"source_frozen": _macro(chunk_mse(pred, target).cpu(), held.demo)}
            # raid: source weights + target bank (may be empty of source transitions — still the target config)
            pred, _ = frozen.forward_with_memory(cur, fut, q, dur, bank=target_bank, use_retrieval=True)
            row["raid_bank_mlp"] = _macro(chunk_mse(pred, target).cpu(), held.demo)
            for cond in ("lora_mlp", "raid_lora_mlp"):
                if cond not in adapted:
                    continue
                enc, head, snap = load_adapted(adapted[cond], encoder_factory, source, protocol, device)
                feats = {d: _encode(enc, read_view_frames(protocol, spec["path"], d)) for d in retention_ids}
                lc = torch.stack([feats[d][t] for d, t in zip(held.demo, held.step)]).to(device)
                lf = torch.stack([feats[d][t + chunk] for d, t in zip(held.demo, held.step)]).to(device)
                use_r = condition_flags(cond)["R"]
                pred, _ = head.forward_with_memory(
                    lc, lf, q, dur, bank=target_bank if use_r else None, use_retrieval=use_r,
                    query_current=cur if use_r else None, query_future=fut if use_r else None)
                row[cond] = _macro(chunk_mse(pred, target).cpu(), held.demo)
            # reset reference: source bank restored, adapter removed
            train_ids = sorted(map(int, spec["source"]["source_train"]))
            bank_ids = random.Random(f"retention-reset:{seed}:{source_task}:{budget}").sample(
                train_ids, min(budget, len(train_ids)))
            br = chunk_rows(cache["shards"][source_task], bank_ids, source_task, chunk, norms)
            reset_bank = make_bank(br.current, br.future, br.q, br.target, br.duration, br.group_id, device)
            pred, _ = frozen.forward_with_memory(cur, fut, q, dur, bank=reset_bank, use_retrieval=True)
            row["reset_reference"] = _macro(chunk_mse(pred, target).cpu(), held.demo)
            per_condition[source_task] = row
    def _avg(name: str) -> float | None:
        vals = [r[name] for r in per_condition.values() if name in r]
        return float(np.mean(vals)) if vals else None
    report = {
        "protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
        "setting": "source_retention", "condition_task": task, "budget": budget, "seed": seed,
        "per_source_task": per_condition,
        "results": {name: {"chunk_mse_macro": _avg(name)}
                    for name in list(CONDITION_NAMES) + ["reset_reference"] if _avg(name) is not None},
        "note": "target bank and/or adapter active per condition; reset_reference is separate",
        "scientific_result": False,
    }
    _write_json(Path(out_dir) / f"retention_{task}_b{budget}_s{seed}.json", report)
    return report


def coverage_distances(protocol: dict, cache: dict, source: dict, task: str, budget: int,
                       seed: int) -> dict:
    """Fixed reference-space coverage distances (development-only bin edges elsewhere)."""
    bank, _ = build_support_bank(protocol, cache, source, task, budget, seed)
    query = list(map(int, protocol["tasks"][task]["query"]))
    rows = chunk_rows(cache["shards"][task], query, task, source["chunk_steps"], source["normalizers"],
                      actions_override={d: read_actions(protocol["tasks"][task]["path"], d) for d in query})
    retrieval = _as_retrieval(source["retrieval_spec"])
    dist = bank.distance(rows.current, rows.future, rows.q, spec=retrieval)
    # nearest neighbour distance per query row
    nn = dist.min(dim=-1).values
    return {"task": task, "budget": budget, "seed": seed,
            "nn_distances": nn.tolist(),
            "mean_nn": float(nn.mean()), "median_nn": float(nn.median()),
            "geometry": "frozen_reference_mae", "scientific_result": False}


def zero_effect_lora_check(encoder_factory: Callable[[], Any], protocol: dict,
                           source: dict, frames: np.ndarray) -> dict:
    """Verify LoRA with zero B has identical predictions to the frozen encoder."""
    from .models import attach_mae_lora
    lora = protocol["design"]["inverse_model"]["lora"]
    base = encoder_factory()
    adapted = encoder_factory()
    attach_mae_lora(adapted.mae, lora["blocks"], lora["rank"], lora["alpha"], lora["dropout"])
    # Force B to zero (already zero-init) and compare
    with torch.no_grad():
        for name, p in adapted.mae.named_parameters():
            if name.endswith("lora_B"):
                p.zero_()
        a = base.encode_frames(torch.from_numpy(frames))
        b = adapted.encode_frames(torch.from_numpy(frames))
    max_diff = float((a - b).abs().max())
    return {"max_abs_diff": max_diff, "identical": max_diff < 1e-6,
            "lora_targets": [n for n, _ in adapted.mae.named_parameters() if "lora_" in n]}
