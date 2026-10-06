"""Immutable demonstration-level LIBERO split generation and validation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


def demo_uid(task: str, demo_id: int) -> int:
    """Stable cross-loader identity for a whole demonstration."""
    payload = f"{task}\x1f{int(demo_id)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def transition_uid(task: str, demo_id: int, timestep: int) -> str:
    return f"{task}:{int(demo_id)}:{int(timestep)}"


def _task_name(path: Path) -> str:
    return path.stem


def inspect_hdf5(path: str | Path) -> dict[str, Any]:
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        demos = sorted(handle["data"].keys(), key=lambda x: int(x.replace("demo_", "")))
        demo_ids = [int(x.replace("demo_", "")) for x in demos]
        lengths = []
        for demo in demos:
            actions = handle[f"data/{demo}/actions"]
            if len(actions.shape) != 2 or actions.shape[1] != 7:
                raise ValueError(f"{path}:{demo} has action shape {actions.shape}; expected (T,7)")
            obs = handle[f"data/{demo}/obs"]
            image_key = next((key for key in ("agentview_rgb", "agentview_image", "image") if key in obs), None)
            if image_key is None:
                raise ValueError(f"{path}:{demo} has no supported agent-view image")
            if int(obs[image_key].shape[0]) < int(actions.shape[0]):
                raise ValueError(f"{path}:{demo} has fewer images than actions")
            lengths.append(int(actions.shape[0]))
    return {"task": _task_name(path), "path": str(path), "demos": len(demos), "demo_ids": demo_ids, "lengths": lengths}


def create_split(
    task_files: list[str | Path],
    output: str | Path,
    *,
    seed: int = 20260920,
    validation_per_task: int = 5,
    test_per_task: int = 5,
    train_pool_per_task: int | None = None,
    expected_demos_per_task: int | None = 50,
    horizon: int = 1,
) -> dict[str, Any]:
    """Create and persist a deterministic demonstration-level split.

    Demonstration IDs are disjoint across train/validation/test. The manifest
    includes transition counts and a digest so jobs can refuse stale splits.
    """
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    for raw_path in sorted(map(Path, task_files)):
        meta = inspect_hdf5(raw_path)
        if expected_demos_per_task is not None and meta["demos"] != expected_demos_per_task:
            raise ValueError(f"{raw_path} has {meta['demos']} demonstrations; expected {expected_demos_per_task}")
        ids = list(meta["demo_ids"])
        if len(ids) <= validation_per_task + test_per_task:
            raise ValueError(f"{raw_path} has too few demonstrations: {len(ids)}")
        rng.shuffle(ids)
        test = sorted(ids[:test_per_task])
        validation = sorted(ids[test_per_task:test_per_task + validation_per_task])
        pool = ids[test_per_task + validation_per_task:]
        if train_pool_per_task is not None:
            pool = pool[:train_pool_per_task]
        train = sorted(pool)
        length_by_id = dict(zip(meta["demo_ids"], meta["lengths"]))
        tasks.append({
            "task": meta["task"],
            "path": str(raw_path),
            "train": train,
            "validation": validation,
            "test": test,
            "transition_counts": {
                "train": sum(max(0, length_by_id[i] - horizon) for i in train),
                "validation": sum(max(0, length_by_id[i] - horizon) for i in validation),
                "test": sum(max(0, length_by_id[i] - horizon) for i in test),
            },
            "transition_ids": {
                "train": [{"demo_id": i, "timesteps": list(range(max(0, length_by_id[i] - horizon)))} for i in train],
                "validation": [{"demo_id": i, "timesteps": list(range(max(0, length_by_id[i] - horizon)))} for i in validation],
                "test": [{"demo_id": i, "timesteps": list(range(max(0, length_by_id[i] - horizon)))} for i in test],
            },
        })
    manifest: dict[str, Any] = {
        "format": 1,
        "seed": seed,
        "validation_per_task": validation_per_task,
        "test_per_task": test_per_task,
        "horizon": horizon,
        "tasks": tasks,
    }
    manifest["digest"] = _digest(manifest)
    output = Path(output)
    if output.exists():
        existing = load_and_validate(output)
        if existing.get("digest") != manifest["digest"]:
            raise FileExistsError(f"Refusing to overwrite immutable split manifest: {output}")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _digest(manifest: dict[str, Any]) -> str:
    copied = dict(manifest)
    copied.pop("digest", None)
    return hashlib.sha256(json.dumps(copied, sort_keys=True).encode()).hexdigest()


def load_and_validate(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text())
    expected = manifest.get("digest")
    if expected != _digest(manifest):
        raise ValueError(f"Split manifest digest mismatch: {path}")
    for task in manifest.get("tasks", []):
        sets = [set(task.get(key, [])) for key in ("train", "validation", "test")]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError(f"Demonstration leakage in split for {task.get('task')}")
    return manifest


def select_training_ids(task: dict[str, Any], budget: int | None, seed: int) -> list[int]:
    """Return a deterministic nested subset of the persisted training pool.

    The split file deliberately stores the complete pool.  A budget is a
    protocol choice, so its selection must be derived from the pool with a
    separate seed and shared by every condition.
    """
    pool = [int(x) for x in task.get("train", [])]
    if budget is None:
        return sorted(pool)
    if budget < 1 or budget > len(pool):
        raise ValueError(f"training budget {budget} is outside the pool of {len(pool)} demos")
    task_seed = seed + sum((i + 1) * ord(ch) for i, ch in enumerate(str(task.get("task", ""))))
    rng = random.Random(task_seed)
    rng.shuffle(pool)
    return sorted(pool[:budget])


def resolve_task_files(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    paths = sorted(root.rglob("*.hdf5")) + sorted(root.rglob("*.h5"))
    paths = [p for p in paths if "features" not in p.parts]
    if not paths:
        raise FileNotFoundError(f"No LIBERO HDF5 files found below {root}")
    return paths
