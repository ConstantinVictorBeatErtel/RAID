"""Metrics shared by observed-future and predicted-future evaluations."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def hierarchical_macro_mse(errors, task_ids, demo_ids=None) -> float:
    """Average transitions within demos, demos within tasks, tasks equally."""
    values = defaultdict(list)
    for index, (task, error) in enumerate(zip(task_ids, np.asarray(errors))):
        demo = demo_ids[index] if demo_ids is not None else index
        values[(str(task), str(demo))].append(float(np.asarray(error).mean()))
    by_task = defaultdict(list)
    for (task, _demo), rows in values.items():
        by_task[task].append(float(np.mean(rows)))
    return float(np.mean([np.mean(rows) for rows in by_task.values()])) if by_task else float("nan")


def action_metrics(predicted, target, *, mean=None, std=None, task_ids=None, demo_ids=None) -> dict:
    """Return normalized, physical-unit, per-action, and task-macro errors."""
    pred = np.asarray(predicted, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    normalized = (pred - true) ** 2
    result = {
        "normalized_mse": float(normalized.mean()),
        "normalized_per_action_mse": normalized.mean(axis=0).tolist(),
    }
    if pred.shape[-1] >= 1:
        # Gripper classification is defined in physical action space. The
        # decoder outputs normalized actions, so undo normalization first.
        if mean is not None and std is not None:
            pred_grip = pred[..., -1] * float(np.asarray(std)[-1]) + float(np.asarray(mean)[-1])
            true_grip = true[..., -1] * float(np.asarray(std)[-1]) + float(np.asarray(mean)[-1])
        else:
            pred_grip, true_grip = pred[..., -1], true[..., -1]
        result["gripper_accuracy"] = float(np.mean((pred_grip >= 0.0) == (true_grip >= 0.0))) if pred.shape[0] else float("nan")
    if mean is not None and std is not None:
        physical = ((pred - true) * np.asarray(std)) ** 2
        result["unnormalized_per_action_mse"] = physical.mean(axis=0).tolist()
        result["unnormalized_mse"] = float(physical.mean())
    if task_ids is not None:
        groups = defaultdict(list)
        for task, row in zip(task_ids, normalized):
            groups[str(task)].append(float(row.mean()))
        result["task_macro_normalized_mse"] = float(np.mean([np.mean(v) for v in groups.values()]))
        result["task_normalized_mse"] = {k: float(np.mean(v)) for k, v in groups.items()}
        if demo_ids is not None:
            result["demonstration_task_macro_normalized_mse"] = hierarchical_macro_mse(normalized, task_ids, demo_ids)
    return result


def save_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")
