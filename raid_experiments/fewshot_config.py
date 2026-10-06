"""Loader and validation for the launch-disabled LIBERO-Goal study config."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from .fewshot import LIBERO_COMMIT, PROTOCOL
from .fewshot_train import METHODS


CONTROLS = {"support_mean", "transition_1nn", "current_weighted_knn", "source_support_knn"}


def load_fewshot_config(path: str | Path = "configs/fewshot_libero_goal.yaml") -> tuple[dict, str]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text())
    if payload.get("protocol") != PROTOCOL or payload.get("launch_enabled") is not None:
        raise ValueError("study config protocol identity is invalid")
    if payload.get("protocol_version") != 3:
        raise ValueError("study config protocol_version must match the active v3 protocol")
    if payload.get("cloud", {}).get("launch_enabled") is not False:
        raise ValueError("paid launch disabled: cloud.launch_enabled must remain false")
    if set(payload.get("cloud", {})) != {"launch_enabled"}:
        raise ValueError("unsupported cloud configuration fields are not allowed in the study config")
    allowed = {"name", "protocol", "protocol_version", "benchmark", "libero_commit",
        "split_hash_seed", "expected_demos_per_task", "source_tasks", "development_tasks", "final_test_tasks",
        "source_demonstrations", "source_validation_demonstrations", "source_retention_demonstrations",
        "target_query_demonstrations", "target_support_pool_demonstrations", "support_budgets", "methods",
        "lookup_controls", "horizon", "future_representation", "feature_dim", "action_dim", "normalizer",
        "source_training", "adaptation", "retrieval", "development_seed", "confirmation_seeds",
        "planning_budget_usd", "protected_reserve_usd", "workload_margin", "output_dir", "cloud"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unrecognized study configuration fields: {sorted(unknown)}")
    if (len(payload.get("methods", [])) != len(set(payload.get("methods", [])))
            or set(payload.get("methods", [])) != METHODS):
        raise ValueError("study config method registry must match the implemented adaptation methods")
    if (len(payload.get("lookup_controls", [])) != len(set(payload.get("lookup_controls", [])))
            or set(payload.get("lookup_controls", [])) != CONTROLS):
        raise ValueError("study config retrieval controls differ from implemented controls")
    expected = {
        "benchmark": "libero_goal", "libero_commit": LIBERO_COMMIT,
        "split_hash_seed": 20260925, "expected_demos_per_task": 50,
        "source_tasks": 6, "development_tasks": 2, "final_test_tasks": 2,
        "source_demonstrations": 40, "source_validation_demonstrations": 5,
        "source_retention_demonstrations": 5, "target_query_demonstrations": 20,
        "target_support_pool_demonstrations": 30, "support_budgets": [2, 5, 10],
        "horizon": 1, "future_representation": "recorded_rgb", "feature_dim": 768,
        "action_dim": 7, "normalizer": "source_training_only", "development_seed": 42,
        "confirmation_seeds": [42, 43, 44], "planning_budget_usd": 250.0,
        "protected_reserve_usd": 25.0, "workload_margin": 0.25,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"study config field {key!r} must be fixed to {value!r}")
    source = payload.get("source_training", {})
    source_expected = {"updates": 2000, "batch_size": 64, "microbatch_size": 8,
                       "head_lr": 0.0003, "weight_decay": 0.0001}
    adaptation = payload.get("adaptation", {})
    adaptation_expected = {"max_updates": 1000, "batch_size": 64, "microbatch_size": 8,
                           "snapshot_updates": [200, 500, 1000], "head_lr": 0.0003,
                           "lora_lr": 0.0001, "lora_rank": 8, "lora_alpha": 16,
                           "lora_dropout": 0.0, "target_blocks": [10, 11],
                           "target_projections": ["qkv", "proj"], "weight_decay": 0.0001,
                           "grad_clip": 1.0}
    retrieval = payload.get("retrieval", {})
    if set(source) != set(source_expected) or set(adaptation) != set(adaptation_expected) or set(retrieval) != {"k", "temperature"}:
        raise ValueError("study config exposes unsupported or missing scientific settings")
    if any(source.get(k) != v for k, v in source_expected.items()):
        raise ValueError("source-training settings differ from the fixed study protocol")
    if any(adaptation.get(k) != v for k, v in adaptation_expected.items()):
        raise ValueError("adaptation settings differ from the fixed study protocol")
    if retrieval != {"k": 3, "temperature": 0.1}:
        raise ValueError("retrieval settings differ from the fixed study protocol")
    if payload.get("cloud", {}).get("launch_enabled") is not False:
        raise ValueError("paid launch disabled: cloud.launch_enabled must remain false")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload, digest
