"""Feature-cache datasets with immutable demonstration provenance."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
import torch
from torch.utils.data import Dataset

from .splits import demo_uid, load_and_validate, select_training_ids, transition_uid


@dataclass(frozen=True)
class Transition:
    current: torch.Tensor
    future: torch.Tensor
    action: torch.Tensor
    task: str
    demo_id: int
    timestep: int
    transition_uid: str


def _cache_paths(feature_dir: Path) -> list[Path]:
    manifest = feature_dir / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        paths = [Path(p) for p in payload["cached_files"]]
        return [p if p.exists() else feature_dir / p.name for p in paths]
    return sorted(feature_dir.glob("*_features.pt"))


class FeatureTransitions(Dataset):
    """A flat transition dataset that retains task/demo/timestep IDs."""
    def __init__(self, feature_dir: str | Path, split_path: str | Path, split_name: str,
                 train_per_task: int | None = None, horizon: int | None = None,
                 train_seed: int = 42, norm_mean: torch.Tensor | None = None,
                 norm_std: torch.Tensor | None = None):
        self.items: list[Transition] = []
        split = load_and_validate(split_path)
        train_allowed = {x["task"]: set(select_training_ids(x, train_per_task, train_seed)) for x in split["tasks"]}
        self.selected_demo_ids = {task: sorted(ids) for task, ids in train_allowed.items()}
        allowed = {x["task"]: (set(train_allowed[x["task"]]) if split_name == "train" else set(x[split_name])) for x in split["tasks"]}
        paths = _cache_paths(Path(feature_dir))
        cache_by_task = {}
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            task_name = payload.get("task_name", path.stem.replace("_features", ""))
            if task_name in cache_by_task:
                raise ValueError(f"duplicate feature cache for task {task_name}")
            cache_by_task[task_name] = (path, payload)
        expected_tasks = {x["task"] for x in split["tasks"]}
        if set(cache_by_task) != expected_tasks:
            raise ValueError(f"feature tasks {sorted(cache_by_task)} do not match split tasks {sorted(expected_tasks)}")
        # Fit normalization only from the selected training demonstrations,
        # regardless of whether this dataset instance serves train or val.
        if norm_mean is None or norm_std is None:
            norm_rows = []
            for task_name in sorted(expected_tasks):
                path, data = cache_by_task[task_name]
                if horizon is not None and int(data.get("horizon", -1)) != int(horizon):
                    raise ValueError(f"Feature cache {path} has horizon {data.get('horizon')}; expected {horizon}")
                if horizon != 1 or data.get("future_representation") not in (None, "recorded_rgb"):
                    raise ValueError(f"Only horizon-one recorded-frame caches are supported: {path}")
                if "actions_raw" not in data:
                    raise ValueError(f"Feature cache {path} has no raw actions; rebuild it before the pilot")
                task = task_name
                demo_ids = list(map(int, data.get("demo_ids", range(len(data.get("demo_lengths", []))))))
                lengths = list(map(int, data.get("demo_lengths", [])))
                if len(demo_ids) != len(lengths):
                    raise ValueError(f"{path} has mismatched demo_ids/demo_lengths")
                if int(data["actions_raw"].shape[0]) != sum(lengths):
                    raise ValueError(f"{path} action rows do not match demo_lengths")
                if not torch.isfinite(data["actions_raw"]).all():
                    raise ValueError(f"{path} contains non-finite actions")
                offset = 0
                for demo_id, length in zip(demo_ids, lengths):
                    if demo_id in train_allowed.get(task, set()):
                        norm_rows.append(data["actions_raw"][offset:offset + length].float())
                    offset += length
            if not norm_rows:
                raise ValueError(f"No training actions available to fit normalization in {feature_dir}")
            norm = torch.cat(norm_rows, dim=0)
            self.norm_mean = norm.mean(0)
            raw_std = norm.std(0, unbiased=False)
            self.norm_std = torch.where(raw_std < 1e-6, torch.ones_like(raw_std), raw_std)
        else:
            self.norm_mean = torch.as_tensor(norm_mean, dtype=torch.float32).clone()
            self.norm_std = torch.as_tensor(norm_std, dtype=torch.float32).clone()
            if torch.any(self.norm_std <= 0):
                raise ValueError("action normalizer standard deviations must be positive")
        for task_name in sorted(expected_tasks):
            path, data = cache_by_task[task_name]
            if horizon is not None and int(data.get("horizon", -1)) != int(horizon):
                raise ValueError(f"Feature cache {path} has horizon {data.get('horizon')}; expected {horizon}")
            if horizon != 1 or data.get("future_representation") not in (None, "recorded_rgb"):
                raise ValueError(f"Only horizon-one recorded-frame caches are supported: {path}")
            task = task_name
            selected = allowed.get(task, set())
            lengths = list(map(int, data.get("demo_lengths", [])))
            if "demo_ids" not in data:
                referenced = set().union(*(set(map(int, task.get(split_name, []))) for task in split["tasks"] if task.get("task") == task_name))
                if any(i not in range(len(lengths)) for i in referenced):
                    raise ValueError(f"{path} lacks explicit non-contiguous demonstration IDs")
            demo_ids = list(map(int, data.get("demo_ids", range(len(lengths)))))
            if len(demo_ids) != len(lengths):
                raise ValueError(f"{path} has mismatched demo_ids/demo_lengths")
            for key in ("feat_t", "feat_next", "actions_raw"):
                if int(data[key].shape[0]) != sum(lengths):
                    raise ValueError(f"{path} {key} rows do not match demo_lengths")
                if not torch.isfinite(data[key]).all():
                    raise ValueError(f"{path} contains non-finite {key}")
            if not lengths:
                raise ValueError(f"Cache {path} has no demo_lengths; refusing ambiguous split")
            offset = 0
            for demo_id, length in zip(demo_ids, lengths):
                if demo_id not in selected:
                    offset += length
                    continue
                for t in range(length):
                    # Keep task and demonstration in one integer key so a
                    # demo_0 in two tasks can never be confused for the same
                    # retrieval-exclusion group.
                    global_demo_id = demo_uid(task, demo_id)
                    raw_action = data["actions_raw"][offset + t].float()
                    action = (raw_action - self.norm_mean) / self.norm_std
                    self.items.append(Transition(data["feat_t"][offset + t].float(), data["feat_next"][offset + t].float(), action, task, global_demo_id, t, transition_uid(task, demo_id, t)))
                offset += length
        self.items.sort(key=lambda x: x.transition_uid)
        if not self.items:
            raise ValueError(f"No {split_name} transitions found in {feature_dir}")

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        x = self.items[index]
        return {"current": x.current, "future": x.future, "action": x.action, "task": x.task, "demo_id": x.demo_id, "timestep": x.timestep, "transition_uid": x.transition_uid}


def collate_transitions(batch: list[dict]) -> dict:
    return {
        "current": torch.stack([x["current"] for x in batch]),
        "future": torch.stack([x["future"] for x in batch]),
        "action": torch.stack([x["action"] for x in batch]),
        "task": [x["task"] for x in batch],
        "demo_id": torch.tensor([x["demo_id"] for x in batch], dtype=torch.long),
        "timestep": torch.tensor([x["timestep"] for x in batch], dtype=torch.long),
        "transition_uid": [x["transition_uid"] for x in batch],
    }


def write_manifest(path: str | Path, config_digest: str, source_files: Iterable[str]) -> None:
    payload = {"config_digest": config_digest, "source_files": sorted(map(str, source_files))}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
