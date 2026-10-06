"""Evidence validation, development selection, confirmation registration, and reporting."""
from __future__ import annotations

import hashlib
import json
import math
import os
import fcntl
from contextlib import contextmanager
from pathlib import Path
import uuid

import numpy as np
import torch

from .fewshot import PROTOCOL, REPETITION_SEEDS, validate_goal_protocol, workload_inventory
from .fewshot_train import (ARCHITECTURE, CHECKPOINT_FORMAT, EVIDENCE_FORMAT, METHODS as LEARNED_METHODS,
                            _code_snapshot_digest, _config_digest, _scientific_config, _sha,
                            _validate_adaptation_checkpoint, _validate_adaptation_evidence,
                            _validate_source_checkpoint_payload)
from .models import DirectDecoder

SNAPSHOTS = (200, 500, 1000)
CONTROLS = ("frozen_source", "target_knn", "support_mean", "transition_1nn",
            "current_weighted_knn", "source_support_knn")
METHODS = CONTROLS + ("head_ft", "lora_head_ft")


@contextmanager
def study_access_lock(study_root: str | Path):
    root = Path(study_root); root.mkdir(parents=True, exist_ok=True)
    with (root / ".final_query_access.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def canonical_study_root(source_checkpoints: dict[int, str | Path]) -> Path:
    """Resolve the study root from the registered source checkpoint registry."""
    roots = set()
    for path in source_checkpoints.values():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        root = payload.get("study_root")
        if (payload.get("checkpoint_format") != CHECKPOINT_FORMAT or not isinstance(root, str)
                or not Path(root).is_absolute() or str(Path(root).resolve()) != root):
            raise ValueError("source checkpoint lacks a valid canonical study-root binding")
        roots.add(Path(root))
    if len(roots) != 1:
        raise ValueError("all registered source checkpoints must belong to one canonical study root")
    return roots.pop()
def _freeze_digest(payload: dict) -> str:
    body = dict(payload); body.pop("freeze_digest", None)
    body = json.loads(json.dumps(body, sort_keys=True))
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_immutable(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise FileExistsError(f"refusing to replace immutable evidence {path}")
        return
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        try:
            # Hard-link creation is atomic and fails if another writer won.
            os.link(tmp, path)
        except FileExistsError:
            if json.loads(path.read_text()) != normalized:
                raise FileExistsError(f"refusing to replace immutable evidence {path}")
    finally:
        tmp.unlink(missing_ok=True)


def _expected_query_ids(task: str, spec: dict) -> list[str]:
    return [f"{task}:{int(demo)}:{t}" for demo in spec["query"]
            for t in range(int(spec["transitions"][str(int(demo))]))]


def _normalizer_sha(mean: torch.Tensor, std: torch.Tensor) -> str:
    return hashlib.sha256(mean.detach().cpu().contiguous().numpy().tobytes()
                          + std.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _evaluation_config_digest(report: dict) -> str:
    return _config_digest({"protocol_digest": report.get("protocol_digest"),
        "task": report.get("task"), "budget": report.get("budget"), "seed": report.get("seed"),
        "methods": sorted(report.get("methods", {})),
        "selected_updates": report.get("selected_updates", {}),
        "source_checkpoint_sha256": report.get("source_checkpoint_sha256"),
        "source_feature_digest": report.get("source_feature_digest"),
        "feature_digest": report.get("feature_digest"),
        "code_snapshot_sha256": report.get("code_snapshot_sha256")})


def _recorded_query_targets(task: str, spec: dict, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    from .fewshot_train import RawSelectedRows
    rows = RawSelectedRows(spec["path"], task, list(map(int, spec["query"])), mean, std)
    return torch.stack([row["action"] for row in rows])


def _prediction(path: Path, record: dict, report: dict, method: str,
                expected_query_ids: list[str], reference_targets: torch.Tensor | None = None,
                expected_normalizer_sha: str | None = None):
    if not isinstance(record, dict) or not record.get("path"):
        raise ValueError(f"missing prediction artifact for {method}")
    artifact = (path.parent / record["path"]).resolve()
    try:
        artifact.relative_to(path.parent.resolve())
    except ValueError as exc:
        raise ValueError(f"prediction path escapes condition directory for {method}") from exc
    if not artifact.is_file() or _sha(artifact) != record.get("sha256"):
        raise ValueError(f"missing or corrupt prediction artifact for {method}")
    data = torch.load(artifact, map_location="cpu", weights_only=False)
    if data.get("prediction_format") != 4 or data.get("execution_mode") != "scientific":
        raise ValueError(f"unsupported prediction format or non-scientific prediction for {method}")
    ids = data.get("transition_ids")
    pred, target = data.get("predictions"), data.get("targets")
    if ids != expected_query_ids or ids != report.get("query_ids"):
        raise ValueError(f"incomplete or mismatched query coverage for {method}")
    if (not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor)
            or pred.shape != (len(ids), 7) or target.shape != pred.shape
            or not torch.isfinite(pred).all() or not torch.isfinite(target).all()):
        raise ValueError(f"prediction shape or finite-value check failed for {method}")
    prediction_config = data.get("evaluation_config")
    prediction_config_sha = _config_digest(prediction_config) if isinstance(prediction_config, dict) else None
    if (data.get("protocol_digest") != report.get("protocol_digest")
            or data.get("source_checkpoint_sha256") != report.get("source_checkpoint_sha256")
            or data.get("source_feature_digest") != report.get("source_feature_digest")
            or data.get("feature_digest") != report.get("feature_digest")
            or data.get("code_snapshot_sha256") != report.get("code_snapshot_sha256")
            or data.get("study_config_sha256") != _config_digest(_scientific_config())
            or prediction_config_sha != data.get("evaluation_config_sha256")
            or report.get("prediction_config_sha256", {}).get(method) != prediction_config_sha
            or data.get("support_ids") != report.get("support_ids")
            or data.get("method") != method or data.get("task") != report.get("task")
            or data.get("seed") != report.get("seed") or data.get("budget") != report.get("budget")):
        raise ValueError(f"prediction provenance mismatch for {method}")
    if (prediction_config.get("protocol_digest") != report.get("protocol_digest")
            or prediction_config.get("task") != report.get("task")
            or prediction_config.get("budget") != report.get("budget")
            or prediction_config.get("seed") != report.get("seed")
            or prediction_config.get("source_checkpoint_sha256") != report.get("source_checkpoint_sha256")
            or prediction_config.get("source_feature_digest") != report.get("source_feature_digest")
            or prediction_config.get("feature_digest") != report.get("feature_digest")
            or prediction_config.get("code_snapshot_sha256") != report.get("code_snapshot_sha256")
            or method not in prediction_config.get("methods", [])):
        raise ValueError(f"evaluation configuration provenance is invalid for {method}")
    if (method in ("head_ft", "lora_head_ft")
            and prediction_config.get("selected_updates", {}).get(method) != data.get("selected_update")):
        raise ValueError(f"prediction snapshot is not bound to its evaluation update config for {method}")
    mean, std = data.get("normalizer_mean"), data.get("normalizer_std")
    if (not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor)
            or mean.shape != (7,) or std.shape != (7,) or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError(f"prediction normalizer is missing or invalid for {method}")
    normalizer_sha = _normalizer_sha(mean, std)
    if normalizer_sha != report.get("normalizer_sha256"):
        raise ValueError(f"prediction normalizer identity mismatch for {method}")
    if expected_normalizer_sha is not None and normalizer_sha != expected_normalizer_sha:
        raise ValueError(f"prediction normalizer differs from source checkpoint for {method}")
    if method in ("head_ft", "lora_head_ft"):
        step = data.get("selected_update")
        if step not in SNAPSHOTS or report.get("methods", {}).get(method, {}).get("selected_update") != step:
            raise ValueError(f"selected snapshot identity mismatch for {method}")
        snapshot = path.parent / f"{method}.step{step}.pt"
        if (not snapshot.is_file() or _sha(snapshot) != data.get("adaptation_artifact_sha256")):
            raise ValueError(f"adaptation checkpoint provenance is missing or mismatched for {method}")
        payload = torch.load(snapshot, map_location="cpu", weights_only=False)
        expected_identity = {"method": method, "task": report.get("task"),
            "budget": report.get("budget"), "seed": report.get("seed"),
            "support_ids": report.get("support_ids"), "protocol_digest": report.get("protocol_digest"),
            "source_checkpoint_sha256": report.get("source_checkpoint_sha256"),
            "source_feature_digest": report.get("source_feature_digest"),
            "feature_digest": report.get("feature_digest"),
            "code_snapshot_sha256": report.get("code_snapshot_sha256"), "selected_step": step}
        try:
            _validate_adaptation_checkpoint(payload, expected_identity, scientific=True)
            _validate_adaptation_evidence(path.parent, snapshot, expected_identity)
        except Exception as exc:
            raise ValueError(f"adaptation checkpoint identity or weights are invalid for {method}: {exc}") from exc
        if report.get("methods", {}).get(method, {}).get("adaptation_config_digest") != payload.get("config_digest"):
            raise ValueError(f"adaptation config identity mismatch for {method}")
    if reference_targets is not None and (target.shape != reference_targets.shape or
                                          not torch.allclose(target, reference_targets, rtol=0, atol=1e-6)):
        raise ValueError(f"saved query targets differ from HDF5 evaluation labels for {method}")
    return data


def _mse_from_predictions(data: dict) -> float:
    pred = data["predictions"].float(); target = data["targets"].float()
    ids = data["transition_ids"]
    by_demo = {}
    for i, transition in enumerate(ids):
        demo = transition.rsplit(":", 2)[-2]
        by_demo.setdefault(demo, []).append(i)
    demo_errors = [float((pred[idx] - target[idx]).pow(2).mean(-1).mean())
                   for idx in by_demo.values()]
    if not demo_errors:
        raise ValueError("prediction artifact contains no demonstrations")
    return float(np.mean(demo_errors))


def _validate_source_checkpoint(path: Path, protocol: dict, seed: int) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"source checkpoint missing for seed {seed}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_source_checkpoint_payload(payload, protocol, seed, checkpoint_path=path, scientific=True)
    return _sha(path)


def _load_candidate(path: Path, protocol: dict, task: str, budget: int,
                    source_hash: str, expected_methods: set[str], expected_query: list[str],
                    expected_normalizer_sha: str | None = None,
                    expected_targets: torch.Tensor | None = None,
                    expected_feature_digest: str | None = None,
                    expected_code_digest: str | None = None) -> tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing development evidence: {path}")
    report = json.loads(path.read_text())
    if (report.get("evidence_format") != EVIDENCE_FORMAT
            or report.get("status") != "complete" or report.get("execution_mode") != "scientific"
            or report.get("protocol_digest") != protocol["digest"]
            or report.get("task") != task or report.get("role") != "development"
            or report.get("budget") != budget or report.get("seed") != 42
            or report.get("source_checkpoint_sha256") != source_hash
            or (expected_feature_digest is not None and report.get("source_feature_digest") != expected_feature_digest)
            or (expected_code_digest is not None and report.get("code_snapshot_sha256") != expected_code_digest)
            or report.get("query_ids") != expected_query):
        raise ValueError(f"development evidence identity/status/query mismatch: {path}")
    expected_support = list(map(int, protocol["tasks"][task]["support_budgets_by_seed"]["42"][str(budget)]))
    if report.get("support_ids") != expected_support:
        raise ValueError(f"development evidence uses the wrong support set: {path}")
    if set(report.get("methods", {})) != expected_methods:
        raise ValueError(f"method set in {path} must be exactly {sorted(expected_methods)}")
    if report.get("evaluation_config_sha256") != _evaluation_config_digest(report):
        raise ValueError(f"evaluation call configuration hash is invalid: {path}")
    targets = None; predictions = {}; normalizer = None
    for method in sorted(expected_methods):
        data = _prediction(path, report.get("prediction_artifacts", {}).get(method), report, method,
                           expected_query, expected_targets, expected_normalizer_sha)
        targets = data["targets"] if targets is None else targets
        pair = (data["normalizer_mean"], data["normalizer_std"])
        if normalizer is not None and (not torch.equal(normalizer[0], pair[0]) or not torch.equal(normalizer[1], pair[1])):
            raise ValueError(f"normalizer differs across methods in {path}")
        normalizer = pair
        calculated = _mse_from_predictions(data)
        metric = report["methods"][method].get("hierarchical_source_normalized_mse")
        try: metric = float(metric)
        except (TypeError, ValueError): raise ValueError(f"missing metric for {method} in {path}")
        if not math.isfinite(metric) or not math.isclose(metric, calculated, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"reported metric disagrees with predictions for {method} in {path}")
        predictions[method] = data
    return report, predictions


def freeze_development(protocol_path, output_dir, source_checkpoint, selected_step,
                       freeze_path, *, source_checkpoints=None):
    """Choose each learned method's snapshot from independent full-query evidence."""
    if selected_step is not None:
        raise ValueError("selected update counts are computed from development evidence")
    protocol = validate_goal_protocol(protocol_path); root = Path(output_dir); freeze = Path(freeze_path)
    source42 = Path(source_checkpoint)
    source_hash = _validate_source_checkpoint(source42, protocol, 42)
    source_payload = torch.load(source42, map_location="cpu", weights_only=False)
    source_normalizer_sha = _normalizer_sha(source_payload["normalizer_mean"], source_payload["normalizer_std"])
    if source_checkpoints:
        registry = {int(k): Path(v) for k, v in source_checkpoints.items()}
        if set(registry) not in (set(REPETITION_SEEDS), {42}):
            raise ValueError("optional source registry must contain seed 42 or all seeds 42, 43, and 44")
        if registry[42].resolve() != source42.resolve():
            raise ValueError("seed-42 source checkpoint must match --source-checkpoint")
        for seed, checkpoint in registry.items():
            _validate_source_checkpoint(checkpoint, protocol, seed)
    baselines = {}; candidates = {"head_ft": {}, "lora_head_ft": {}}
    for task in protocol["task_roles"]["development"]:
        spec = protocol["tasks"][task]; expected_query = _expected_query_ids(task, spec)
        recorded_targets = _recorded_query_targets(task, spec, source_payload["normalizer_mean"],
                                                   source_payload["normalizer_std"])
        for budget in (2, 5, 10):
            base = root / task / f"{budget}shot" / "seed42"
            baseline, baseline_preds = _load_candidate(base / "evaluation_baselines.json", protocol, task,
                budget, source_hash, set(CONTROLS), expected_query, source_normalizer_sha, recorded_targets,
                source_payload["feature_digest"], source_payload["code_snapshot_sha256"])
            baselines[(task, budget)] = (baseline, baseline_preds)
            for method in candidates:
                for step in SNAPSHOTS:
                    filename = f"evaluation_{'head' if method == 'head_ft' else 'lora'}{step}.json"
                    expected = {method}
                    report, preds = _load_candidate(base / filename, protocol, task, budget,
                                                    source_hash, expected, expected_query, source_normalizer_sha,
                                                    recorded_targets, source_payload["feature_digest"],
                                                    source_payload["code_snapshot_sha256"])
                    baseline_report_path = base / "evaluation_baselines.json"
                    baseline_record = report.get("baseline_evidence", {})
                    if (baseline_record.get("evaluation_sha256") != _sha(baseline_report_path)
                            or baseline_record.get("prediction_artifact_sha256") !=
                               {control: baselines[(task, budget)][0]["prediction_artifacts"][control]["sha256"]
                                for control in CONTROLS}):
                        raise ValueError(f"candidate is not bound to the validated baseline bundle: {filename}")
                    candidate_data = preds[method]
                    base_data = baseline_preds["target_knn"]
                    if (candidate_data["transition_ids"] != base_data["transition_ids"]
                            or not torch.equal(candidate_data["targets"], base_data["targets"])):
                        raise ValueError(f"paired query evidence differs in {filename}")
                    candidates[method][(task, budget, step)] = (report, preds)
    means = {}
    selected = {}
    for method, values in candidates.items():
        means[method] = {step: float(np.mean([
            _mse_from_predictions(values[(task, budget, step)][1][method])
            for task in protocol["task_roles"]["development"] for budget in (2, 5, 10)]))
            for step in SNAPSHOTS}
        selected[method] = min(SNAPSHOTS, key=lambda step: (means[method][step], step))
    candidate_hashes = {}
    for task in protocol["task_roles"]["development"]:
        for budget in (2, 5, 10):
            base = root / task / f"{budget}shot" / "seed42"
            for name in ("evaluation_baselines.json", *(f"evaluation_head{s}.json" for s in SNAPSHOTS),
                         *(f"evaluation_lora{s}.json" for s in SNAPSHOTS)):
                p = base / name; candidate_hashes[str(p.relative_to(root))] = _sha(p)
            baseline = baselines[(task, budget)][0]
            combined = dict(baseline); combined["status"] = "complete"
            combined["selected_updates"] = dict(selected)
            combined["methods"] = dict(baseline["methods"])
            combined["prediction_artifacts"] = dict(baseline["prediction_artifacts"])
            combined["prediction_config_sha256"] = dict(baseline.get("prediction_config_sha256", {}))
            for method, prefix in (("head_ft", "head"), ("lora_head_ft", "lora")):
                report, _ = candidates[method][(task, budget, selected[method])]
                combined["methods"][method] = report["methods"][method]
                combined["prediction_artifacts"][method] = report["prediction_artifacts"][method]
                combined["prediction_config_sha256"][method] = report["prediction_config_sha256"][method]
            combined["evaluation_config_sha256"] = _evaluation_config_digest(combined)
            combined["evaluation_config_composition"] = "frozen_baselines_plus_selected_snapshots"
            _write_immutable(base / "evaluation.json", combined)
    payload = {"freeze_format": 4, "protocol": PROTOCOL, "protocol_digest": protocol["digest"],
               "study_root": str(canonical_study_root({42: source42})),
               "development_source_checkpoint_sha256": source_hash,
               "development_source_feature_digest": source_payload["source_feature_digest"],
               "source_normalizer_sha256": source_normalizer_sha,
               "development_source_normalizer": {"mean": source_payload["normalizer_mean"].tolist(),
                                                  "std": source_payload["normalizer_std"].tolist()},
               "development_candidate_evaluation_sha256": candidate_hashes,
               "development_macro_source_normalized_mse": means,
               "selected_updates": selected, "final_test_unopened": True,
               "confirmation_registered": False,
               "statement": "Development settings frozen from complete query evidence; final-test checkpoint registry is separate."}
    payload["freeze_digest"] = _freeze_digest(payload)
    if freeze.exists():
        existing = json.loads(freeze.read_text())
        normalized = json.loads(json.dumps(payload, sort_keys=True))
        if existing != normalized:
            raise FileExistsError(f"refusing to replace immutable freeze file {freeze}")
        return existing
    _write_immutable(freeze, payload)
    return json.loads(json.dumps(payload, sort_keys=True))


def register_confirmation(protocol_path, freeze_path, source_checkpoints, output_path, *, scope="all",
                          study_root=None):
    """Bind frozen settings to valid source checkpoints before final-query access."""
    protocol = validate_goal_protocol(protocol_path); freeze = json.loads(Path(freeze_path).read_text())
    if (freeze.get("freeze_format") != 4 or freeze.get("protocol_digest") != protocol["digest"]
            or freeze.get("freeze_digest") != _freeze_digest(freeze)):
        raise ValueError("development freeze record is invalid")
    if scope not in ("all", "five_shot"):
        raise ValueError("confirmation scope must be 'all' or 'five_shot'")
    registry = {int(k): Path(v) for k, v in source_checkpoints.items()}
    if set(registry) != set(REPETITION_SEEDS):
        raise ValueError("confirmation requires source checkpoints for seeds 42, 43, and 44")
    access_root = canonical_study_root(registry)
    if study_root is not None and Path(study_root).expanduser().resolve() != access_root:
        raise ValueError("confirmation ledger root must match the registered source checkpoint root")
    if freeze.get("study_root") != str(access_root):
        raise ValueError("registered source checkpoints do not belong to the frozen study root")
    access_path = access_root / "final_query_access.json"
    hashes = {str(seed): _validate_source_checkpoint(path, protocol, seed)
              for seed, path in sorted(registry.items())}
    if hashes.get("42") != freeze.get("development_source_checkpoint_sha256"):
        raise ValueError("seed-42 source checkpoint differs from the development-frozen checkpoint")
    normalizers = {}
    normalizer_values = {}
    feature_digests = {}; code_digests = {}
    for seed, checkpoint in sorted(registry.items()):
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        mean, std = data["normalizer_mean"], data["normalizer_std"]
        normalizers[str(seed)] = _normalizer_sha(mean, std)
        normalizer_values[str(seed)] = {"mean": mean.tolist(), "std": std.tolist(),
                                        "sha256": normalizers[str(seed)]}
        feature_digests[str(seed)] = data["source_feature_digest"]
        code_digests[str(seed)] = data["code_snapshot_sha256"]
    if feature_digests.get("42") != freeze.get("development_source_feature_digest"):
        raise ValueError("seed-42 source features differ from the development freeze")
    payload = {"confirmation_format": 4, "protocol": PROTOCOL, "protocol_digest": protocol["digest"],
               "study_root": str(access_root),
               "freeze_digest": freeze["freeze_digest"], "source_checkpoints": hashes,
               "source_normalizer_sha256": normalizers,
               "source_normalizers": normalizer_values,
               "source_feature_digests": feature_digests,
               "source_code_digests": code_digests,
               "selected_updates": freeze["selected_updates"],
               "workload_inventory": workload_inventory(scope, freeze["selected_updates"]),
               "confirmation_scope": {"support_budgets": [2, 5, 10] if scope == "all" else [5],
                                      "seeds": list(REPETITION_SEEDS), "paired": True},
               "final_test_unopened": True, "confirmation_registered": True}
    payload["confirmation_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    destination = Path(output_path)
    registry_path = access_root / "confirmation_registry.json"
    with study_access_lock(access_root):
        if registry_path.is_file():
            registered = json.loads(registry_path.read_text())
            if registered != json.loads(json.dumps(payload, sort_keys=True)):
                raise FileExistsError("confirmation registry is already bound to different evidence")
            _write_immutable(destination, payload)
            return registered
        if destination.exists():
            existing = json.loads(destination.read_text())
            if existing != json.loads(json.dumps(payload, sort_keys=True)):
                raise FileExistsError(f"refusing to replace registered confirmation {destination}")
        if access_path.is_file() and json.loads(access_path.read_text()).get("conditions"):
            raise ValueError("cannot register confirmation after final-query access was recorded")
        _write_immutable(registry_path, payload)
        _write_immutable(destination, payload)
    return payload


def _paired_bootstrap(root: Path, records: dict, protocol: dict, role: str, budget: int,
                      left: str, right: str, draws=10000) -> dict | None:
    tasks = protocol["task_roles"][role]
    seeds = (42,) if role == "development" else REPETITION_SEEDS
    samples = []; rng = np.random.default_rng(20260925 + budget + len(left) + len(right))
    condition_data = {}; task_demo_ids = {}
    for task in tasks:
        for seed in seeds:
            item = records.get((task, budget, seed))
            if not item:
                return None
            base = root / task / f"{budget}shot" / f"seed{seed}"
            loaded = {}
            for method in (left, right):
                rec = item["prediction_artifacts"][method]
                loaded[method] = torch.load(base / rec["path"], map_location="cpu", weights_only=False)
            a, b = loaded[left], loaded[right]
            if a["transition_ids"] != b["transition_ids"] or not torch.equal(a["targets"], b["targets"]):
                return None
            by_demo = {}
            for idx, tid in enumerate(a["transition_ids"]):
                demo = tid.rsplit(":", 2)[-2]; by_demo.setdefault(demo, []).append(idx)
            deltas = {}
            for demo, indexes in by_demo.items():
                idx = torch.tensor(indexes)
                ea = (a["predictions"][idx].float() - a["targets"][idx].float()).pow(2).mean(-1).mean()
                eb = (b["predictions"][idx].float() - b["targets"][idx].float()).pow(2).mean(-1).mean()
                deltas[demo] = float(ea - eb)
            if task in task_demo_ids and set(task_demo_ids[task]) != set(deltas):
                return None
            task_demo_ids.setdefault(task, sorted(deltas))
            condition_data[(task, seed)] = deltas
    observed = float(np.mean([np.mean(list(v.values())) for v in condition_data.values()]))
    for _ in range(draws):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_demos = {task: rng.choice(ids, size=len(ids), replace=True).tolist()
                         for task, ids in task_demo_ids.items()}
        task_means = []
        for task in tasks:
            repetition_means = []
            for seed in sampled_seeds:
                values = condition_data[(task, int(seed))]
                # Reuse each task's sampled demonstrations across paired methods
                # and every repetition drawn in this bootstrap replicate.
                repetition_means.append(float(np.mean([values[demo] for demo in sampled_demos[task]])))
            task_means.append(float(np.mean(repetition_means)))
        samples.append(float(np.mean(task_means)))
    return {"estimate": observed, "lower_95": float(np.quantile(samples, .025)),
            "upper_95": float(np.quantile(samples, .975)), "draws": draws,
            "tasks_fixed": list(tasks), "seeds": list(seeds)}


def report_fewshot(output_dir, protocol_path=None, *, confirmation_file=None, study_root=None):
    root = Path(output_dir); protocol = validate_goal_protocol(protocol_path) if protocol_path else None
    confirmation = None
    freeze = None
    freeze_path = root / "development_freeze.json"
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text())
        if (not protocol or freeze.get("freeze_format") != 4
                or freeze.get("protocol_digest") != protocol["digest"]
                or freeze.get("freeze_digest") != _freeze_digest(freeze)):
            raise ValueError("development freeze record is invalid")
    requested_study_root = Path(study_root).expanduser().resolve() if study_root else None
    confirmation_path = (Path(confirmation_file) if confirmation_file else
                         (requested_study_root / "confirmation_registry.json" if requested_study_root else
                          root / "confirmation_registry.json"))
    canonical_root = requested_study_root or root.resolve()
    if confirmation_path.is_file():
        confirmation = json.loads(confirmation_path.read_text())
        body = dict(confirmation); claimed = body.pop("confirmation_digest", None)
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (not protocol or confirmation.get("confirmation_format") != 4
                or confirmation.get("protocol_digest") != protocol["digest"]
                or confirmation.get("confirmation_registered") is not True or claimed != digest):
            raise ValueError("confirmation registry is invalid")
        registered_root = Path(confirmation.get("study_root", "")).expanduser().resolve()
        if requested_study_root and registered_root != requested_study_root:
            raise ValueError("confirmation registry belongs to another canonical study root")
        canonical_root = registered_root
        canonical_registry_path = canonical_root / "confirmation_registry.json"
        if (not canonical_registry_path.is_file()
                or json.loads(canonical_registry_path.read_text()) != confirmation):
            raise ValueError("confirmation file does not match the canonical study registry")
    access_path = canonical_root / "final_query_access.json"
    access = json.loads(access_path.read_text()) if access_path.is_file() else None
    if access and (not protocol or access.get("protocol_digest") != protocol.get("digest")
                   or access.get("evidence_format") != EVIDENCE_FORMAT):
        raise ValueError("final-query access ledger identity is invalid")
    paths = sorted(root.rglob("evaluation.json")); records = {}; invalid = []
    for path in paths:
        try:
            item = json.loads(path.read_text()); task = item.get("task"); role = item.get("role")
            budget = item.get("budget"); seed = item.get("seed")
            if (item.get("evidence_format") != EVIDENCE_FORMAT or item.get("status") != "complete"
                    or item.get("execution_mode") != "scientific"):
                raise ValueError("unsupported evidence format, calibration, or non-complete evidence cannot count")
            if protocol is None: raise ValueError("a protocol manifest is required for scientific reporting")
            if (task not in protocol["tasks"] or protocol["tasks"][task]["role"] != role
                    or budget not in (2, 5, 10) or seed not in (42, 43, 44)
                    or item.get("protocol_digest") != protocol["digest"]):
                raise ValueError("protocol/task/role/budget/seed identity mismatch")
            if (role == "development" and seed != 42) or (role == "final_test" and seed not in REPETITION_SEEDS):
                raise ValueError("invalid role and repetition pairing")
            if role == "final_test" and (not confirmation
                    or confirmation.get("source_checkpoints", {}).get(str(seed)) != item.get("source_checkpoint_sha256")
                    or confirmation.get("source_feature_digests", {}).get(str(seed)) != item.get("source_feature_digest")
                    or confirmation.get("source_code_digests", {}).get(str(seed)) != item.get("code_snapshot_sha256")
                    or item.get("selected_updates") != confirmation.get("selected_updates")):
                raise ValueError("final-test evidence differs from the registered source/settings")
            if role == "final_test":
                access_condition = {"task": task, "budget": budget, "seed": seed,
                    "source_checkpoint_sha256": item.get("source_checkpoint_sha256")}
                if not access or access_condition not in access.get("conditions", []):
                    raise ValueError("final-test evidence has no matching query-access ledger entry")
            expected_ids = _expected_query_ids(task, protocol["tasks"][task])
            if item.get("query_ids") != expected_ids: raise ValueError("query coverage is incomplete")
            if set(item.get("methods", {})) != set(METHODS): raise ValueError("completed evaluation lacks required methods")
            if set(item.get("prediction_artifacts", {})) != set(METHODS): raise ValueError("completed evaluation lacks prediction artifacts")
            if (set(item.get("prediction_config_sha256", {})) != set(METHODS)
                    or (item.get("evaluation_config_composition") != "frozen_baselines_plus_selected_snapshots"
                        and item.get("evaluation_config_sha256") != _evaluation_config_digest(item))):
                raise ValueError("evaluation configuration provenance is incomplete")
            if role == "final_test":
                normalizer = (confirmation or {}).get("source_normalizers", {}).get(str(seed))
                if not isinstance(normalizer, dict):
                    raise ValueError("confirmation record lacks the registered source normalizer")
                mean, std = torch.tensor(normalizer.get("mean")), torch.tensor(normalizer.get("std"))
                normalizer_sha = _normalizer_sha(mean, std)
                if (normalizer_sha != normalizer.get("sha256")
                        or normalizer_sha != confirmation.get("source_normalizer_sha256", {}).get(str(seed))):
                    raise ValueError("confirmation source normalizer identity is invalid")
            else:
                normalizer = (freeze or {}).get("development_source_normalizer")
                if not isinstance(normalizer, dict):
                    raise ValueError("development report requires the validated development freeze record")
                mean, std = torch.tensor(normalizer.get("mean")), torch.tensor(normalizer.get("std"))
                normalizer_sha = _normalizer_sha(mean, std)
                if normalizer_sha != freeze.get("source_normalizer_sha256"):
                    raise ValueError("development freeze source normalizer identity is invalid")
            expected_targets = _recorded_query_targets(task, protocol["tasks"][task], mean, std)
            for method in METHODS:
                data = _prediction(path, item["prediction_artifacts"][method], item, method,
                                   expected_ids, expected_targets, normalizer_sha)
                metric = item["methods"][method].get("hierarchical_source_normalized_mse")
                if not math.isclose(float(metric), _mse_from_predictions(data), rel_tol=1e-5, abs_tol=1e-6):
                    raise ValueError(f"reported score differs from predictions for {method}")
            key = (task, budget, seed)
            if key in records: raise ValueError(f"duplicate canonical condition {key}")
            records[key] = item
        except Exception as exc:
            invalid.append({"path": str(path.relative_to(root)), "error": str(exc)})
    missing = []
    if protocol:
        final_budgets = (confirmation or {}).get("confirmation_scope", {}).get("support_budgets", [2, 5, 10])
        for role, seeds, budgets in (("development", (42,), (2, 5, 10)),
                                     ("final_test", REPETITION_SEEDS, tuple(final_budgets))):
            for task in protocol["task_roles"][role]:
                for budget in budgets:
                    for seed in seeds:
                        if (task, budget, seed) not in records:
                            missing.append({"task": task, "role": role, "budget": budget, "seed": seed, "status": "missing"})
    aggregates = {}
    if protocol:
        final_budgets = (confirmation or {}).get("confirmation_scope", {}).get("support_budgets", [2, 5, 10])
        for role, seeds, budgets in (("development", (42,), (2, 5, 10)),
                                     ("final_test", REPETITION_SEEDS, tuple(final_budgets))):
            for budget in budgets:
                by_task = {}
                for task in protocol["task_roles"][role]:
                    values = [records[(task, budget, seed)]["methods"] for seed in seeds
                              if (task, budget, seed) in records]
                    if values: by_task[task] = {m: float(np.mean([v[m]["hierarchical_source_normalized_mse"] for v in values])) for m in METHODS}
                aggregates[f"{role}:{budget}shot"] = {m: (float(np.mean([v[m] for v in by_task.values()]))
                    if set(by_task) == set(protocol["task_roles"][role]) else None) for m in METHODS}
    comparisons = [("head_ft", "target_knn"), ("lora_head_ft", "target_knn"),
                   ("lora_head_ft", "head_ft"), ("head_ft", "frozen_source"), ("lora_head_ft", "frozen_source")]
    bootstrap = {}
    if protocol:
        for role, budgets in (("development", (2, 5, 10)),
                              ("final_test", tuple((confirmation or {}).get("confirmation_scope", {}).get("support_budgets", [2, 5, 10])))):
            for budget in budgets:
                for left, right in comparisons:
                    result = _paired_bootstrap(root, records, protocol, role, budget, left, right)
                    if result is not None: bootstrap[f"{role}:{budget}shot:{left}-{right}"] = result
    return {"protocol": PROTOCOL, "valid_evaluation_artifacts": len(records),
            "expected_canonical_evaluations": (len(protocol["task_roles"]["development"])*3
                + len(protocol["task_roles"]["final_test"])*len((confirmation or {}).get("confirmation_scope", {}).get("support_budgets", [2,5,10]))*3) if protocol else 0,
            "missing_conditions": missing, "invalid_or_failed_artifacts": invalid,
            "aggregates_by_role_and_budget": aggregates, "paired_bootstrap_95_ci": bootstrap,
            "records": [{"task": k[0], "budget": k[1], "seed": k[2], "status": "complete"} for k in records],
            "calibration_is_not_a_completed_study": True}
