"""Protocol preparation and validation for the LIBERO-Goal few-shot study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .splits import inspect_hdf5

PROTOCOL = "fewshot_idm_adaptation_v3"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SPLIT_HASH_SEED = 20260925
REPETITION_SEEDS = (42, 43, 44)
OFFICIAL_GOAL_TASKS = (
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl",
    "turn_on_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
)
SOURCE_TASKS = (
    "put_the_wine_bottle_on_the_rack",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_plate",
    "turn_on_the_stove",
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
)
DEVELOPMENT_TASKS = (
    "put_the_cream_cheese_in_the_bowl",
    "open_the_top_drawer_and_put_the_bowl_inside",
)
FINAL_TEST_TASKS = (
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_wine_bottle_on_top_of_the_cabinet",
)
ROLE_ASSIGNMENTS = {"source": SOURCE_TASKS, "development": DEVELOPMENT_TASKS, "final_test": FINAL_TEST_TASKS}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _digest(payload: dict[str, Any]) -> str:
    # File locations are operational metadata, not experimental identity. The
    # content hash and official task/demo identities bind the actual data.
    def without_locations(value):
        if isinstance(value, dict):
            # JSON object keys are strings. Normalize before sorting so integer
            # demo IDs have the same digest before and after a disk roundtrip.
            return {str(k): without_locations(v) for k, v in value.items() if k != "path"}
        if isinstance(value, list):
            return [without_locations(item) for item in value]
        return value
    body = without_locations(payload)
    body.pop("digest", None)
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _demo_order(task: str, ids: list[int], tag: str) -> list[int]:
    return sorted(ids, key=lambda demo: _sha(f"{SPLIT_HASH_SEED}:{tag}:{task}:{demo}"))


def support_budgets_for_seed(task: str, pool: list[int], seed: int) -> dict[str, list[int]]:
    ordered = sorted(map(int, pool), key=lambda demo: _sha(f"support:{int(seed)}:{task}:{demo}"))
    return {str(count): ordered[:count] for count in (2, 5, 10)}


def workload_inventory(scope: str = "all", selected_updates: dict | None = None) -> dict:
    """Canonical execution and forecast counts for the preregistered study."""
    if scope not in ("all", "five_shot"):
        raise ValueError("scope must be all or five_shot")
    development_conditions = len(DEVELOPMENT_TASKS) * 3
    confirmation_conditions = len(FINAL_TEST_TASKS) * (1 if scope == "five_shot" else 3) * len(REPETITION_SEEDS)
    selected = dict(selected_updates) if selected_updates is not None else {"head_ft": 1000, "lora_head_ft": 1000}
    if set(selected) != {"head_ft", "lora_head_ft"} or any(step not in (200, 500, 1000) for step in selected.values()):
        raise ValueError("confirmation updates must contain both frozen learned-method selections")
    source_runs = len(REPETITION_SEEDS)
    # Count writes, not just training invocations. Source validation candidates
    # may improve at every validation, so reserve all twenty candidate writes.
    source_writes = source_runs * (1 + 20 + 20 + 1 + 1)  # normalizer, candidates, continuations, final, archived final
    development_writes = 2 * development_conditions * (10 + 3 + 1)  # continuations, snapshots, alias
    confirmation_writes = confirmation_conditions * sum(
        step // 100 + sum(snapshot <= step for snapshot in (200, 500, 1000)) + 1
        for step in selected.values())
    return {"scope": scope, "source_training_runs": len(REPETITION_SEEDS),
        "confirmation_updates": selected,
        "adaptation_updates_by_method": {method: development_conditions * 1000 + confirmation_conditions * step
                                          for method, step in selected.items()},
        "source_validation_passes": source_runs * 20,
        "checkpoint_writes": source_writes + development_writes + confirmation_writes,
        "checkpoint_write_breakdown": {"source": source_writes, "development": development_writes,
                                         "confirmation": confirmation_writes},
        "development_conditions": development_conditions,
        "development_baseline_evaluations": development_conditions,
        "development_learned_snapshot_evaluations": development_conditions * 6,
        "canonical_evaluation_conditions": development_conditions + confirmation_conditions,
        "confirmation_conditions": confirmation_conditions,
        "adaptation_runs_per_method": development_conditions + confirmation_conditions,
        "development_retention_passes": development_conditions * 7,
        "confirmation_retention_passes": confirmation_conditions * 3,
        "retrieval_bank_builds": (development_conditions + confirmation_conditions) * 2,
        "checkpoint_io_runs": len(REPETITION_SEEDS) + 2 * (development_conditions + confirmation_conditions)}


def split_goal_task_names(seed: int = SPLIT_HASH_SEED) -> dict[str, list[str]]:
    """Assign official task names by SHA-256 order, independent of filenames."""
    order = sorted(OFFICIAL_GOAL_TASKS, key=lambda task: _sha(f"{seed}:{task}"))
    roles = {"source": order[:6], "development": order[6:8], "final_test": order[8:]}
    # Guard the frozen preregistered assignment against an accidental change.
    if seed == SPLIT_HASH_SEED and roles != {k: list(v) for k, v in ROLE_ASSIGNMENTS.items()}:
        raise RuntimeError("protocol task assignment changed; update the preregistration deliberately")
    return roles


def _locate_task_files(dataset_dir: str | Path) -> dict[str, Path]:
    root = Path(dataset_dir)
    paths = sorted(root.rglob("*.hdf5"))
    mapped: dict[str, list[Path]] = {task: [] for task in OFFICIAL_GOAL_TASKS}
    for path in paths:
        stem = path.stem.removesuffix("_demo")
        for task in OFFICIAL_GOAL_TASKS:
            if stem == task:
                mapped[task].append(path)
    result = {}
    for task, matches in mapped.items():
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one HDF5 for official LIBERO-Goal task {task}; found {[str(p) for p in matches]}")
        result[task] = matches[0]
    return result


def create_goal_protocol(dataset_dir: str | Path, output: str | Path, *, expected_demos: int = 50,
                          seed: int = SPLIT_HASH_SEED, overwrite: bool = False) -> dict[str, Any]:
    """Build the fixed 6/2/2 task split and demo-level source/target manifests."""
    if seed != SPLIT_HASH_SEED or expected_demos != 50:
        raise ValueError(f"The preregistered task split seed is fixed at {SPLIT_HASH_SEED}")
    task_files = _locate_task_files(dataset_dir)
    roles = split_goal_task_names(seed)
    tasks: dict[str, Any] = {}
    for role, names in roles.items():
        for task_name in names:
            path = task_files[task_name]
            meta = inspect_hdf5(path)
            if meta["demos"] != expected_demos:
                raise ValueError(f"{task_name} contains {meta['demos']} demos; protocol requires {expected_demos}")
            import h5py
            import numpy as np
            with h5py.File(path, "r") as handle:
                for demo_id in meta["demo_ids"]:
                    group = handle[f"data/demo_{demo_id}"]
                    actions = group["actions"][:]
                    if not np.isfinite(actions).all():
                        raise ValueError(f"{task_name}:demo_{demo_id} contains non-finite actions")
                    key = next(k for k in ("agentview_rgb", "agentview_image", "image") if k in group["obs"])
                    shape = group[f"obs/{key}"].shape
                    if len(shape) != 4 or shape[-1] != 3:
                        raise ValueError(f"{task_name}:demo_{demo_id} must contain 3-channel RGB frames, got {shape}")
            ids = list(meta["demo_ids"])
            length_by_id = dict(zip(ids, meta["lengths"]))
            if any(length < 2 for length in meta["lengths"]):
                raise ValueError(f"{task_name} has a demonstration too short for one-step inverse dynamics")
            if role == "source":
                ordered = _demo_order(task_name, ids, "source-role")
                parts = {"source_train": ordered[:40], "source_validation": ordered[40:45], "source_retention": ordered[45:]}
                support_pool = []
                query = []
            else:
                ordered = _demo_order(task_name, ids, f"{role}-query-role")
                query = ordered[:20]
                support_pool = _demo_order(task_name, ordered[20:], "target-support-pool")
                parts = {}
            if len(set(query) & set(support_pool)) or set().union(*(set(x) for x in parts.values())) & (set(query) | set(support_pool)):
                raise AssertionError(f"demonstration overlap in {task_name}")
            tasks[task_name] = {
                "role": role,
                "path": str(path.resolve()),
                "source_sha256": _file_sha256(path),
                "demo_ids": ids,
                "demo_lengths": length_by_id,
                "source": parts,
                "support_pool": support_pool,
                "support_budgets": support_budgets_for_seed(task_name, support_pool, 42) if role != "source" else {},
                "support_budgets_by_seed": ({str(seed): support_budgets_for_seed(task_name, support_pool, seed)
                                              for seed in REPETITION_SEEDS} if role != "source" else {}),
                "query": query,
                "transitions": {str(demo): length_by_id[demo] - 1 for demo in ids},
            }
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "benchmark": "libero_goal",
        "libero_commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
        "split_seed": seed,
        "repetitions": {str(seed): {"source_seed": seed, "support_seed": seed} for seed in REPETITION_SEEDS},
        "expected_demos_per_task": expected_demos,
        "horizon": 1,
        "task_roles": roles,
        "support_budgets": [2, 5, 10],
        "source_counts": {"source_train": 40, "source_validation": 5, "source_retention": 5},
        "target_counts": {"query": 20, "support_pool": 30},
        "tasks": tasks,
        "warning": "Two final-test tasks are narrow; conclusions apply to these tasks only.",
    }
    payload["digest"] = _digest(payload)
    output = Path(output)
    if output.exists() and not overwrite:
        old = validate_goal_protocol(output)
        if old["digest"] != payload["digest"]:
            raise FileExistsError(f"Refusing to replace immutable protocol manifest {output}")
        return old
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(output)
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_goal_protocol(path: str | Path, *, verify_sources: bool = True) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("wrong protocol identifier")
    if manifest.get("digest") != _digest(manifest):
        raise ValueError("protocol manifest digest mismatch")
    if (manifest.get("benchmark") != "libero_goal"
            or manifest.get("libero_commit") != LIBERO_COMMIT
            or manifest.get("split_seed") != SPLIT_HASH_SEED
            or manifest.get("expected_demos_per_task") != 50
            or manifest.get("horizon") != 1
            or manifest.get("support_budgets") != [2, 5, 10]
            or manifest.get("source_counts") != {"source_train": 40, "source_validation": 5, "source_retention": 5}
            or manifest.get("target_counts") != {"query": 20, "support_pool": 30}
            or manifest.get("repetitions") != {str(seed): {"source_seed": seed, "support_seed": seed}
                                               for seed in REPETITION_SEEDS}):
        raise ValueError("manifest constants differ from the versioned v3 study protocol")
    roles = manifest.get("task_roles", {})
    flat = [task for role in ("source", "development", "final_test") for task in roles.get(role, [])]
    if len(flat) != 10 or len(set(flat)) != 10 or set(flat) != set(OFFICIAL_GOAL_TASKS):
        raise ValueError("task roles must partition the ten official LIBERO-Goal tasks")
    if roles != {k: list(v) for k, v in ROLE_ASSIGNMENTS.items()}:
        raise ValueError("task role assignment differs from the fixed preregistered split")
    if set(manifest.get("tasks", {})) != set(flat):
        raise ValueError("manifest task records do not match task roles")
    for task_name, task in manifest.get("tasks", {}).items():
        if task_name not in flat or task["role"] not in roles or task_name not in roles[task["role"]]:
            raise ValueError(f"invalid role for {task_name}")
        ids = list(map(int, task["demo_ids"]))
        if len(ids) != len(set(ids)) or len(ids) != manifest["expected_demos_per_task"]:
            raise ValueError(f"bad HDF5 demonstration IDs for {task_name}")
        if len(task.get("demo_lengths", {})) != len(ids) or set(map(int, task["demo_lengths"])) != set(ids):
            raise ValueError(f"demo length index is invalid for {task_name}")
        if set(task.get("transitions", {})) != {str(demo) for demo in ids}:
            raise ValueError(f"transition index is incomplete for {task_name}")
        if any(int(task["demo_lengths"][str(demo)]) < 2 for demo in ids):
            raise ValueError(f"demonstration is too short for one-step inverse dynamics in {task_name}")
        if any(int(task["transitions"][str(demo)]) != int(task["demo_lengths"][str(demo)]) - 1
               or int(task["transitions"][str(demo)]) < 1 for demo in ids):
            raise ValueError(f"invalid one-step transition count for {task_name}")
        source = task["source"]
        if task["role"] == "source":
            if task.get("support_pool") or task.get("query") or task.get("support_budgets") or task.get("support_budgets_by_seed"):
                raise ValueError(f"source task unexpectedly contains target roles: {task_name}")
            groups = [set(source[k]) for k in ("source_train", "source_validation", "source_retention")]
            if list(map(len, groups)) != [40, 5, 5] or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
                raise ValueError(f"source train/validation/retention overlap for {task_name}")
            if set.union(*groups) != set(ids):
                raise ValueError(f"source demonstrations do not partition task {task_name}")
            expected = _demo_order(task_name, ids, "source-role")
            if ([list(map(int, source[k])) for k in ("source_train", "source_validation", "source_retention")]
                    != [expected[:40], expected[40:45], expected[45:]]):
                raise ValueError(f"source assignments differ from the deterministic protocol allocation for {task_name}")
        else:
            if source:
                raise ValueError(f"target task unexpectedly contains source roles: {task_name}")
            support_values, query_values = list(map(int, task["support_pool"])), list(map(int, task["query"]))
            support_pool, query = set(support_values), set(query_values)
            if len(support_values) != len(support_pool) or len(query_values) != len(query):
                raise ValueError(f"duplicate support or query demonstration IDs for {task_name}")
            if len(support_pool) != 30 or len(query) != 20 or support_pool & query or support_pool | query != set(ids):
                raise ValueError(f"support/query split invalid for {task_name}")
            expected_task_order = _demo_order(task_name, ids, f"{task['role']}-query-role")
            if (list(map(int, task["query"])) != expected_task_order[:20]
                    or list(map(int, task["support_pool"])) !=
                    _demo_order(task_name, expected_task_order[20:], "target-support-pool")):
                raise ValueError(f"query/support assignments differ from the deterministic protocol allocation for {task_name}")
            expected_seed_budgets = {str(seed): support_budgets_for_seed(task_name, task["support_pool"], seed)
                                     for seed in REPETITION_SEEDS}
            if task.get("support_budgets_by_seed") != expected_seed_budgets:
                raise ValueError(f"support selections do not match the fixed repetition seeds for {task_name}")
            if task.get("support_budgets") != expected_seed_budgets["42"]:
                raise ValueError(f"default support sets do not match seed 42 for {task_name}")
            if any(len(set(expected_seed_budgets[str(seed)][str(n)])) != n
                   for seed in REPETITION_SEEDS for n in (2, 5, 10)):
                raise ValueError(f"support set contains duplicates for {task_name}")
        if verify_sources:
            source_path = Path(task["path"])
            if not source_path.is_file() or _file_sha256(source_path) != task["source_sha256"]:
                raise ValueError(f"source HDF5 missing or changed for {task_name}")
            actual = inspect_hdf5(source_path)
            actual_lengths = dict(zip(actual["demo_ids"], actual["lengths"]))
            if actual["demo_ids"] != ids or actual_lengths != {int(k): int(v) for k, v in task["demo_lengths"].items()}:
                raise ValueError(f"HDF5 demonstration IDs or lengths changed for {task_name}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m raid_experiments.fewshot")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    prepare.add_argument("--dataset-dir", required=True)
    prepare.add_argument("--output", default="artifacts/fewshot_libero_goal_v3/protocol.json")
    prepare.add_argument("--expected-demos", type=int, default=50)
    validate = sub.add_parser("validate-protocol")
    validate.add_argument("--manifest", default="artifacts/fewshot_libero_goal_v3/protocol.json")
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        manifest = create_goal_protocol(args.dataset_dir, args.output, expected_demos=args.expected_demos)
    else:
        manifest = validate_goal_protocol(args.manifest)
    print(json.dumps({"protocol": manifest["protocol"], "digest": manifest["digest"],
                      "task_roles": manifest["task_roles"], "manifest": str(args.output if args.command == "prepare-protocol" else args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
