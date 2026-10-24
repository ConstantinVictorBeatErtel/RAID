"""LIBERO suites, task roles, demo splits and HDF5 readers for RAID v8.

Roles (docs/RAID_V8_PLAN.md): source = LIBERO-90, development = LIBERO-Goal, final (sealed) = LIBERO-10,
LIBERO-Spatial and LIBERO-Object. Final files are only read by a run that holds a verified freeze record.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

HF_REPO = "yifengzhu-hf/LIBERO-datasets"
SOURCE_SUITES = ("libero_90",)
DEV_SUITES = ("libero_goal",)
FINAL_SUITES = ("libero_10", "libero_spatial", "libero_object")
ROLE = {**{s: "source" for s in SOURCE_SUITES}, **{s: "development" for s in DEV_SUITES},
        **{s: "final" for s in FINAL_SUITES}}
CHUNK = 8
SUPPORT_POOL, QUERY = 30, 20
SOURCE_VAL = 5
SPLIT_SEED = 20261022
SUPPORT_SEEDS = (42, 43, 44)
K_SHOT = 5


@dataclass(frozen=True)
class Task:
    suite: str
    name: str            # file stem without "_demo"
    path: Path

    @property
    def role(self) -> str:
        return ROLE[self.suite]

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.name}"


def list_tasks(root: str | Path, suites) -> list[Task]:
    root = Path(root)
    out = []
    for suite in suites:
        files = sorted((root / suite).glob("*_demo.hdf5"))
        if not files:
            raise FileNotFoundError(f"no LIBERO files for {suite} under {root}")
        out += [Task(suite, f.name[: -len("_demo.hdf5")], f) for f in files]
    return out


def demo_ids(path: str | Path) -> list[int]:
    import h5py
    with h5py.File(path, "r") as handle:
        ids = sorted(int(k.split("_")[1]) for k in handle["data"].keys() if k.startswith("demo_"))
    return ids


def _rng(task: Task, salt: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{task.key}:{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def target_split(task: Task, demos: list[int]) -> dict:
    """Seeded permutation of the demos: support pool (30) and query set (20)."""
    if len(demos) < SUPPORT_POOL + QUERY:
        raise ValueError(f"{task.key} has {len(demos)} demos, needs {SUPPORT_POOL + QUERY}")
    order = [int(d) for d in _rng(task, SPLIT_SEED).permutation(sorted(demos))]
    return {"support_pool": sorted(order[:SUPPORT_POOL]), "query": sorted(order[SUPPORT_POOL:SUPPORT_POOL + QUERY])}


def support_set(task: Task, pool: list[int], seed: int, k: int = K_SHOT) -> list[int]:
    return sorted(int(d) for d in _rng(task, seed).choice(sorted(pool), size=k, replace=False))


def source_split(task: Task, demos: list[int]) -> dict:
    order = [int(d) for d in _rng(task, SPLIT_SEED).permutation(sorted(demos))]
    return {"train": sorted(order[SOURCE_VAL:]), "val": sorted(order[:SOURCE_VAL])}


def _image_key(obs) -> str:
    for key in ("agentview_rgb", "agentview_image", "image"):
        if key in obs:
            return key
    raise KeyError("demonstration has no agent-view RGB")


def read_demo(path: str | Path, demo: int, *, frames: bool = True) -> dict:
    """Frames (T,H,W,3 uint8), actions (T,7), proprio (T,9 = joint7 + gripper2), first simulator state."""
    import h5py
    with h5py.File(path, "r") as handle:
        root = handle[f"data/demo_{int(demo)}"]
        obs = root["obs"]
        out = {"actions": np.asarray(root["actions"], dtype=np.float32),
               "proprio": np.concatenate([np.asarray(obs["joint_states"], dtype=np.float32),
                                          np.asarray(obs["gripper_states"], dtype=np.float32)], axis=-1),
               "state0": np.asarray(root["states"][0])}
        if frames:
            out["frames"] = np.asarray(obs[_image_key(obs)])
    if len(out["actions"]) != len(out["proprio"]):
        raise ValueError(f"{path} demo {demo}: actions and proprio lengths differ")
    return out


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def download(suites, root: str | Path, *, record: str | Path) -> dict:
    """Download whole suites from the Hugging Face mirror and record each file's SHA-256 (pod-side)."""
    from huggingface_hub import snapshot_download
    root = Path(root)
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=str(root), allow_patterns=[f"{s}/*" for s in suites],
                      max_workers=16)
    hashes = {}
    for suite in suites:
        for f in sorted((root / suite).glob("*_demo.hdf5")):
            hashes[f"{suite}/{f.name}"] = {"sha256": sha256_file(f), "bytes": f.stat().st_size}
    Path(record).write_text(json.dumps(hashes, indent=1, sort_keys=True) + "\n")
    return hashes
