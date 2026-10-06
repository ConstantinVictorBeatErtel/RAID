"""
LIBERO dataset loader for RAID visual integration.

Loads LIBERO HDF5 files and returns:
  - image_t:    (H, W, 3) uint8  current frame
  - image_next: (H, W, 3) uint8  next frame
  - action:     (7,)      float32 normalised action
  - language:   str       task language instruction

Each LIBERO task has one HDF5 file with 50 demonstrations.
LIBERO-Spatial has 10 tasks → 500 episodes.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Observation image key names used in LIBERO HDF5 files
# ---------------------------------------------------------------------------
# LIBERO stores images at:  data/demo_X/obs/agentview_rgb  (or agentview_image)
# We'll try both and use whichever exists.
IMG_KEY_CANDIDATES = ["agentview_rgb", "agentview_image", "image"]


def _find_img_key(obs_group: h5py.Group) -> str:
    for k in IMG_KEY_CANDIDATES:
        if k in obs_group:
            return k
    # fallback: first key that looks like an image (has 3 dims + last dim 3 or 4)
    for k in obs_group.keys():
        shape = obs_group[k].shape
        if len(shape) >= 3 and shape[-1] in (3, 4):
            return k
    raise KeyError(f"No image key found in obs group. Keys: {list(obs_group.keys())}")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def compute_norm_stats(hdf5_paths: list[str], max_demos: int = 999999) -> dict:
    """Compute per-dim mean/std over actions across all demos."""
    all_actions = []
    count = 0
    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(),
                           key=lambda x: int(x.replace("demo_", "")))
            for demo in demos:
                if count >= max_demos:
                    break
                actions = f[f"data/{demo}/actions"][:]  # (T, 7)
                all_actions.append(actions)
                count += 1
        if count >= max_demos:
            break
    all_actions = np.concatenate(all_actions, axis=0)  # (N, 7)
    action_mean = all_actions.mean(axis=0).astype(np.float32)
    action_std  = all_actions.std(axis=0).astype(np.float32)
    action_std  = np.where(action_std < 1e-6, 1.0, action_std)
    return {
        "action_mean": torch.tensor(action_mean),
        "action_std":  torch.tensor(action_std),
    }


def save_norm_stats(stats: dict, path: str | Path) -> None:
    torch.save(stats, path)


def load_norm_stats(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LiberoTransitionDataset(Dataset):
    """
    Flat dataset of (image_t, image_next, action_norm, language) transitions.

    Each index corresponds to one (t, t+1) step from any demo.
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        norm_stats: dict,
        n_demos: Optional[int] = None,
        image_size: int = 128,
        skip_last: bool = True,
        future_horizon: int = 1,
    ):
        self.norm_stats = norm_stats
        self.image_size = image_size
        self.future_horizon = int(future_horizon)
        if self.future_horizon < 1:
            raise ValueError("future_horizon must be positive")

        self.images_t:    list[np.ndarray] = []   # (H, W, 3) uint8
        self.images_next: list[np.ndarray] = []
        self.actions:     list[np.ndarray] = []   # (7,) float32 normalised
        self.actions_raw: list[np.ndarray] = []
        self.languages:   list[str]        = []

        action_mean = norm_stats["action_mean"].numpy()
        action_std  = norm_stats["action_std"].numpy()

        total_demos = 0
        for path in hdf5_paths:
            with h5py.File(path, "r") as f:
                # Language instruction stored in dataset attributes
                problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
                lang = problem_info.get("language_instruction", "robot manipulation")
                lang = lang.strip('"\'')

                demos = sorted(f["data"].keys(),
                               key=lambda x: int(x.replace("demo_", "")))
                for demo in demos:
                    if n_demos is not None and total_demos >= n_demos:
                        break
                    grp = f[f"data/{demo}"]
                    obs_grp = grp["obs"]
                    img_key = _find_img_key(obs_grp)

                    images  = obs_grp[img_key][:]          # (T, H, W, 3) uint8
                    actions = grp["actions"][:]             # (T, 7) float32

                    T = len(actions)
                    end = T - self.future_horizon if skip_last else T

                    for t in range(end):
                        img_t    = images[t]
                        future_index = t + self.future_horizon
                        img_next = images[future_index] if future_index < len(images) else images[t]
                        act      = (actions[t] - action_mean) / (action_std + 1e-8)

                        self.images_t.append(img_t)
                        self.images_next.append(img_next)
                        self.actions.append(act.astype(np.float32))
                        self.actions_raw.append(actions[t].astype(np.float32))
                        self.languages.append(lang)

                    total_demos += 1

                if n_demos is not None and total_demos >= n_demos:
                    break

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> dict:
        return {
            "image_t":    torch.from_numpy(self.images_t[idx]),     # (H, W, 3) uint8
            "image_next": torch.from_numpy(self.images_next[idx]),  # (H, W, 3) uint8
            "action":     torch.from_numpy(self.actions[idx]),      # (7,)
            "action_raw": torch.from_numpy(self.actions_raw[idx]),
            "language":   self.languages[idx],
        }


class LazyLiberoTransitionDataset(Dataset):
    """HDF5 transition reader that keeps only demo/timestep indexes in RAM."""

    def __init__(self, hdf5_path: str | Path, norm_stats: dict, future_horizon: int = 1,
                 demo_ids: set[int] | list[int] | None = None, include_actions: bool = True):
        self.path = str(hdf5_path)
        self.future_horizon = int(future_horizon)
        if self.future_horizon < 1:
            raise ValueError("future_horizon must be positive")
        self.mean = norm_stats["action_mean"].cpu().numpy().astype(np.float32)
        self.std = norm_stats["action_std"].cpu().numpy().astype(np.float32)
        self.include_actions = bool(include_actions)
        allowed = None if demo_ids is None else set(map(int, demo_ids))
        self.items: list[tuple[int, str, int]] = []
        with h5py.File(self.path, "r") as handle:
            for key in sorted(handle["data"].keys(), key=lambda x: int(x.replace("demo_", ""))):
                demo = int(key.replace("demo_", "")); group = handle[f"data/{key}"]
                if allowed is not None and demo not in allowed:
                    continue
                actions, obs = group["actions"], group["obs"]
                image_key = _find_img_key(obs)
                if obs[image_key].shape[0] < actions.shape[0]:
                    raise ValueError(f"{self.path}:{key} has fewer images than actions")
                transition_count = min(actions.shape[0], obs[image_key].shape[0]) - self.future_horizon
                self.items.extend((demo, image_key, timestep) for timestep in range(transition_count))
        self._handle = None
        self._pid = None

    def __len__(self) -> int:
        return len(self.items)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None; state["_pid"] = None
        return state

    def __getitem__(self, index: int) -> dict:
        import os
        if self._handle is None or self._pid != os.getpid():
            if self._handle is not None:
                self._handle.close()
            self._handle = h5py.File(self.path, "r")
            self._pid = os.getpid()
        demo, image_key, timestep = self.items[index]
        group = self._handle[f"data/demo_{demo}"]
        current = np.asarray(group[f"obs/{image_key}"][timestep])
        future = np.asarray(group[f"obs/{image_key}"][timestep + self.future_horizon])
        result = {"image_t": torch.from_numpy(current), "image_next": torch.from_numpy(future),
                  "demo_id": demo, "timestep": timestep}
        if self.include_actions:
            action = np.asarray(group["actions"][timestep], dtype=np.float32)
            result["action"] = torch.from_numpy(((action - self.mean) / (self.std + 1e-8)).astype(np.float32))
            result["action_raw"] = torch.from_numpy(action)
        return result

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()


# ---------------------------------------------------------------------------
# Historical combined-cache loader. The active format-4 loader is in fewshot_train.
# ---------------------------------------------------------------------------

class CachedFeatureDataset(Dataset):
    """
    Fail-closed compatibility stub for the removed combined-cache schema.

    Active v4 inputs are per-demonstration shards validated by
    ``raid_experiments.fewshot_train._load_cache_map``. A combined file cannot
    establish protocol roles or sealed-query label isolation.
    """

    def __init__(self, cache_path: str | Path, n_demos: Optional[int] = None):
        raise ValueError("combined legacy feature caches are rejected; use the protocol-bound format-4 MAE shard loader")


# ---------------------------------------------------------------------------
# Utility: find all HDF5 files in a LIBERO dataset directory
# ---------------------------------------------------------------------------

def find_hdf5_files(dataset_dir: str | Path) -> list[str]:
    """Recursively find all .hdf5 files in a directory."""
    pattern = str(Path(dataset_dir) / "**" / "*.hdf5")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        # also try .h5
        pattern = str(Path(dataset_dir) / "**" / "*.h5")
        files = sorted(glob.glob(pattern, recursive=True))
    return files


def make_train_val_split(
    hdf5_paths: list[str],
    norm_stats: dict,
    n_demos: int,
    val_frac: float = 0.2,
) -> tuple[CachedFeatureDataset | LiberoTransitionDataset,
           CachedFeatureDataset | LiberoTransitionDataset]:
    """Split demos 80/20 train/val (used before caching)."""
    n_val  = max(1, int(n_demos * val_frac))
    n_train = n_demos - n_val

    # Val uses last n_val demos (approximate: just pick last files / demos)
    all_demos = []
    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            demo_keys = sorted(f["data"].keys(),
                               key=lambda x: int(x.replace("demo_", "")))
            all_demos.extend([(path, dk) for dk in demo_keys])

    all_demos = all_demos[:n_demos]
    train_demos = all_demos[:n_train]
    val_demos   = all_demos[n_train:]

    def _build(demo_list):
        ds = LiberoTransitionDataset.__new__(LiberoTransitionDataset)
        ds.norm_stats  = norm_stats
        ds.image_size  = 128
        ds.images_t    = []
        ds.images_next = []
        ds.actions     = []
        ds.languages   = []
        action_mean = norm_stats["action_mean"].numpy()
        action_std  = norm_stats["action_std"].numpy()

        for path, demo in demo_list:
            with h5py.File(path, "r") as f:
                problem_info = json.loads(f["data"].attrs.get("problem_info", "{}"))
                lang = problem_info.get("language_instruction", "robot manipulation").strip('"\'')
                grp     = f[f"data/{demo}"]
                obs_grp = grp["obs"]
                img_key = _find_img_key(obs_grp)
                images  = obs_grp[img_key][:]
                actions = grp["actions"][:]
                for t in range(len(actions) - 1):
                    act = (actions[t] - action_mean) / (action_std + 1e-8)
                    ds.images_t.append(images[t])
                    ds.images_next.append(images[t + 1])
                    ds.actions.append(act.astype(np.float32))
                    ds.languages.append(lang)
        return ds

    return _build(train_demos), _build(val_demos)
