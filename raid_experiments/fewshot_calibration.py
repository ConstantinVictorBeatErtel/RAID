"""Versioned calibration collection using the production few-shot paths."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import time
from pathlib import Path

import torch

from .fewshot import DEVELOPMENT_TASKS, SOURCE_TASKS, validate_goal_protocol
from .fewshot_train import (_code_snapshot_digest, _config_digest, _same_checkpoint_value,
                            _scientific_config, _sha, adapt_target, evaluate_task, train_source)
from src.mae_encoder import MAE_IMPLEMENTATION


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_measurements(measurements: dict) -> None:
    """Reject incomplete/nonfinite measurements before publishing a receipt."""
    methods = {"frozen_source", "target_knn", "support_mean", "transition_1nn",
               "current_weighted_knn", "source_support_knn", "head_ft", "lora_head_ft"}

    def positive(value, label):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"missing or invalid calibration measurement: {label}")

    for learner in ("source", "head_ft", "lora_head_ft"):
        for phase, count in (("warmup_update_seconds", 5), ("update_seconds", 20)):
            values = measurements.get(f"{learner}_{phase}")
            if not isinstance(values, list) or len(values) != count:
                raise ValueError(f"{learner} must record {count} {phase} samples in one session")
            for value in values:
                positive(value, f"{learner}_{phase}")
        positive(measurements.get(f"{learner}_initialization_and_data_prep_seconds"), f"{learner} initialization")
        profile = measurements.get("training_profiles", {}).get(learner, {})
        positive(profile.get("peak_memory_bytes"), f"{learner} peak memory")
        for phase in ("checkpoint_write_seconds", "checkpoint_serialization_seconds", "snapshot_publish_seconds"):
            values = profile.get(phase)
            if not isinstance(values, list) or not values:
                raise ValueError(f"missing actual {learner} {phase}")
            for value in values:
                positive(value, f"{learner} {phase}")
    validations = measurements["training_profiles"]["source"].get("source_validation_seconds")
    if not isinstance(validations, list) or not validations:
        raise ValueError("missing source validation timing")
    for value in validations:
        positive(value, "source validation")
    timings = measurements.get("per_method_query_ms_per_transition", {})
    if set(timings) != methods:
        raise ValueError("calibration must time all eight methods")
    for method, value in timings.items():
        positive(value, method)
    for field, expected in (("source_retention_seconds", {"frozen_source", "head_ft", "lora_head_ft"}),
                            ("adaptation_recovery_seconds", {"head_ft", "lora_head_ft"})):
        values = measurements.get(field, {})
        if set(values) != expected:
            raise ValueError(f"incomplete {field}")
        for method, value in values.items():
            positive(value, f"{field}/{method}")
    for field in ("mae_encoder_initialization_seconds", "feature_cache_wall_seconds",
                  "raw_frame_encoding_seconds", "retrieval_bank_construction_seconds",
                  "query_label_and_feature_loading_seconds", "eight_method_query_evaluation_seconds",
                  "checkpoint_io_copy_seconds", "source_recovery_seconds", "peak_memory_bytes", "artifact_storage_bytes"):
        positive(measurements.get(field), field)
    for field in ("static_asset_cache_bytes", "source_run_storage_bytes", "head_ft_run_storage_bytes", "lora_head_ft_run_storage_bytes"):
        positive(measurements.get(field), field)


def _collect_calibration(protocol_path, feature_dir, mae_checkpoint, output_dir, *, device="cuda",
                        libero_repo=None) -> dict:
    """Collect a complete calibration bundle, or retain diagnostics without a receipt.

    Each production learner runs five warmup updates and twenty measured
    optimizer updates in one uninterrupted session. CUDA completion is required
    because these measurements size the accelerator workload.
    """
    out = Path(output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    result = {"measurement_format": 3, "collector_version": 3,
              "status": "partial", "protocol": "fewshot_idm_adaptation_v3",
              "device": str(dev), "scientific_result": False,
              "completion_receipt_written": False, "measurements": {},
              "required_components": ["same_session_warmup_and_steady_state", "eight_method_equal_input_matrix",
                  "initialization", "encoding", "retrieval_bank_construction", "query_evaluation",
                  "source_retention", "checkpoint_io", "recovery", "peak_memory", "artifact_storage"]}
    result_path = out / "collect_calibration.json"
    if result_path.exists() or (out / "calibration_completion.json").exists():
        raise FileExistsError("calibration output already contains evidence; choose a fresh output directory")
    if dev.type != "cuda" or not torch.cuda.is_available():
        result.update(status="blocked", reason="CUDA is required for completed calibration evidence")
        _atomic_json(result_path, result)
        return result

    protocol = validate_goal_protocol(protocol_path)
    if not Path(mae_checkpoint).is_file():
        raise FileNotFoundError(f"MAE checkpoint missing: {mae_checkpoint}")
    from .fewshot_train import _load_cache_map
    from .fewshot_assets import audit_complete_feature_cache
    cache_manifest_path = Path(feature_dir) / "manifest.json"
    json.loads(cache_manifest_path.read_text())
    _load_cache_map(feature_dir, protocol, run_mode="calibration")
    audit_complete_feature_cache(feature_dir, protocol)
    from src.mae_encoder import MAEImageEncoder

    _sync(dev); start = time.perf_counter()
    encoder = MAEImageEncoder.from_checkpoint(mae_checkpoint, device=dev)
    _sync(dev)
    result["measurements"]["mae_encoder_initialization_seconds"] = time.perf_counter() - start
    del encoder
    torch.cuda.reset_peak_memory_stats(dev)

    workspace = out / "production_runs"
    workspace.mkdir(exist_ok=True)
    if not libero_repo:
        raise ValueError("complete calibration requires the pinned --libero-repo checkout")
    from src.cache_mae_features import cache_protocol_tasks
    dataset_roots = {Path(spec["path"]).expanduser().resolve().parent
                     for spec in protocol["tasks"].values()}
    if len(dataset_roots) != 1:
        raise ValueError("protocol task files do not share one dataset root")
    calibrated_cache_dir = workspace / "cache_cuda"
    cache_started = time.perf_counter()
    cache = cache_protocol_tasks(protocol_path, next(iter(dataset_roots)), calibrated_cache_dir,
        mae_checkpoint, libero_repo, device=str(dev), batch_size=64)
    _sync(dev)
    feature_dir = calibrated_cache_dir
    cache_manifest_path = calibrated_cache_dir / "manifest.json"
    _load_cache_map(feature_dir, protocol, run_mode="calibration")
    audit_complete_feature_cache(feature_dir, protocol)
    result["measurements"]["feature_cache_wall_seconds"] = cache["measurements"]["feature_cache_wall_seconds"]
    result["measurements"]["feature_cache_encoder_initialization_seconds"] = cache["measurements"]["encoder_initialization_seconds"]
    result["measurements"]["feature_cache_total_wall_seconds"] = time.perf_counter() - cache_started
    target = DEVELOPMENT_TASKS[0]
    source_dir = workspace / "source"
    source_profile = {"warmup_updates": 5, "measured_update_seconds": []}
    _sync(dev); started = time.perf_counter()
    source_result = train_source(protocol_path, feature_dir, source_dir, seed=42, updates=25,
        batch_size=64, microbatch_size=8, device=str(dev), run_mode="calibration",
        timing_profile=source_profile)
    _sync(dev)
    result["measurements"]["source_initialization_and_data_prep_seconds"] = source_profile["initialization_seconds"]
    result["measurements"]["source_update_seconds"] = list(source_profile["measured_update_seconds"])
    result["measurements"]["source_warmup_update_seconds"] = list(source_profile["warmup_update_seconds"])
    result["measurements"]["training_profiles"] = {"source": source_profile}
    source_checkpoint = source_dir / "source_seed42.pt"
    if not source_checkpoint.is_file():
        raise RuntimeError("source calibration did not produce a checkpoint")

    run_root = workspace / "runs"
    adaptation_profiles = {}
    for method in ("head_ft", "lora_head_ft"):
        profile = {"warmup_updates": 5, "measured_update_seconds": []}
        adaptation_profiles[method] = profile
        _sync(dev); started = time.perf_counter()
        adapt_result = adapt_target(protocol_path, feature_dir, source_checkpoint, run_root,
            task=target, method=method, budget=5, seed=42, updates=25,
            batch_size=64, microbatch_size=8, device=str(dev),
            mae_checkpoint=mae_checkpoint if method == "lora_head_ft" else None,
            run_mode="calibration", timing_profile=profile)
        _sync(dev)
        result["measurements"][f"{method}_initialization_and_data_prep_seconds"] = profile["initialization_seconds"]
        result["measurements"][f"{method}_update_seconds"] = list(profile["measured_update_seconds"])
        result["measurements"][f"{method}_warmup_update_seconds"] = list(profile["warmup_update_seconds"])
        result["measurements"]["training_profiles"][method] = profile
        if adapt_result.get("status") != "calibration" or len(profile["measured_update_seconds"]) != 20:
            raise RuntimeError(f"{method} calibration did not finish five warmup and twenty measured updates")

    condition_dir = run_root / target / "5shot" / "seed42"
    snapshot_io_started = time.perf_counter()
    for method in ("head_ft", "lora_head_ft"):
        snapshot = condition_dir / f"{method}.step25.pt"
        with snapshot.open("rb") as source, (workspace / f"{method}.io-copy.pt").open("wb") as destination:
            shutil.copyfileobj(source, destination)
    _sync(dev)
    result["measurements"]["checkpoint_io_copy_seconds"] = time.perf_counter() - snapshot_io_started

    evaluation_root = run_root
    input_spec = protocol["tasks"][target]
    from .fewshot_train import RawSelectedRows
    from torch.utils.data import DataLoader
    normalizer = torch.load(source_dir / "action_normalizer.pt", map_location="cpu", weights_only=False)
    query_rows = RawSelectedRows(input_spec["path"], target, input_spec["query"],
                                 normalizer["mean"], normalizer["std"])
    input_hasher = hashlib.sha256(); input_ids = []
    for query_batch in DataLoader(query_rows, batch_size=64, shuffle=False, num_workers=0):
        input_hasher.update(query_batch["image_t"].numpy().tobytes())
        input_hasher.update(query_batch["image_next"].numpy().tobytes())
        input_ids.extend(query_batch["transition_id"])
    input_hash = input_hasher.hexdigest()
    evaluate_started = time.perf_counter()
    evaluation = evaluate_task(protocol_path, feature_dir, source_checkpoint, evaluation_root,
        task=target, budget=5, seed=42, methods=("frozen_source", "target_knn", "support_mean",
        "transition_1nn", "current_weighted_knn", "source_support_knn", "head_ft", "lora_head_ft"),
        selected_updates={"head_ft": 25, "lora_head_ft": 25}, device=str(dev),
        mae_checkpoint=mae_checkpoint, run_mode="calibration")
    _sync(dev)
    result["measurements"]["eight_method_query_evaluation_seconds"] = time.perf_counter() - evaluate_started
    if set(evaluation.get("methods", {})) != {"frozen_source", "target_knn", "support_mean",
            "transition_1nn", "current_weighted_knn", "source_support_knn", "head_ft", "lora_head_ft"}:
        raise RuntimeError("calibration evaluation did not produce all eight methods")
    if evaluation.get("query_ids") != input_ids:
        raise RuntimeError("calibration raw RGB IDs differ from evaluated query IDs")
    performance = evaluation["performance"]["measurements"]
    for method, record in evaluation["prediction_artifacts"].items():
        payload = torch.load(condition_dir / record["path"], map_location="cpu", weights_only=False)
        if payload.get("transition_ids") != input_ids or _sha(condition_dir / record["path"]) != record["sha256"]:
            raise RuntimeError(f"{method} calibration prediction input/hash mismatch")
    result["measurements"]["per_method_query_ms_per_transition"] = {
        method: performance[method].get("cached_compute_ms_per_transition")
        for method in evaluation["methods"]}
    result["measurements"]["source_retention"] = {
        method: evaluation["methods"][method].get("source_retention")
        for method in ("frozen_source", "head_ft", "lora_head_ft")}
    result["measurements"]["source_retention_seconds"] = {
        method: performance[method].get("source_retention_seconds")
        for method in ("frozen_source", "head_ft", "lora_head_ft")}
    result["measurements"]["raw_frame_encoding_seconds"] = performance["lora_head_ft"].get(
        "raw_frame_encoding_seconds")
    result["measurements"]["retrieval_bank_construction_seconds"] = evaluation["performance"].get(
        "phase_seconds", {}).get("retrieval_bank_construction")
    result["measurements"]["query_label_and_feature_loading_seconds"] = evaluation["performance"].get(
        "phase_seconds", {}).get("query_label_and_feature_loading")
    result["measurements"]["peak_memory_bytes"] = max(
        [int(profile["peak_memory_bytes"]) for profile in result["measurements"]["training_profiles"].values()]
        + [int(performance[method]["peak_memory_bytes"]) for method in performance])

    # Exercise source and both adaptation continuation paths into fresh dirs.
    source_prefix = workspace / "source_prefix"
    recovered_source_dir = workspace / "source_recovered"
    recovery_start = time.perf_counter()
    train_source(protocol_path, feature_dir, source_prefix, seed=42, updates=25,
        stop_after_updates=5, batch_size=64, microbatch_size=8, device=str(dev), run_mode="calibration")
    train_source(protocol_path, feature_dir, recovered_source_dir, seed=42, updates=25,
        batch_size=64, microbatch_size=8, device=str(dev), run_mode="calibration",
        resume_checkpoint=source_prefix / "source_seed42.last.pt")
    _sync(dev)
    result["measurements"]["source_recovery_seconds"] = time.perf_counter() - recovery_start
    expected_source = torch.load(source_dir / "source_seed42.last.pt", map_location="cpu", weights_only=False)
    recovered_source = torch.load(recovered_source_dir / "source_seed42.last.pt", map_location="cpu", weights_only=False)
    for key in ("state_dict", "best_state_dict", "optimizer", "order", "cursor", "schedule_rng_state",
                "torch_rng_state", "numpy_rng_state", "python_rng_state", "cuda_rng_state"):
        if not _same_checkpoint_value(expected_source[key], recovered_source[key]):
            raise RuntimeError(f"source continuation recovery changed {key}")
    recovery_measurements = {}
    for method in ("head_ft", "lora_head_ft"):
        recovery_dir = workspace / f"recovery_{method}"
        recovery_start = time.perf_counter()
        prefix_dir = workspace / f"prefix_{method}"
        adapt_target(protocol_path, feature_dir, source_checkpoint, prefix_dir,
            task=target, method=method, budget=5, seed=42, updates=25, stop_after_updates=5,
            batch_size=64, microbatch_size=8, device=str(dev),
            mae_checkpoint=mae_checkpoint if method == "lora_head_ft" else None,
            run_mode="calibration")
        adapt_target(protocol_path, feature_dir, source_checkpoint, recovery_dir,
            task=target, method=method, budget=5, seed=42, updates=25,
            batch_size=64, microbatch_size=8, device=str(dev),
            mae_checkpoint=mae_checkpoint if method == "lora_head_ft" else None,
            resume_checkpoint=prefix_dir / target / "5shot" / "seed42" / f"{method}.last.pt",
            run_mode="calibration")
        _sync(dev)
        recovery_measurements[method] = time.perf_counter() - recovery_start
        expected = torch.load(condition_dir / f"{method}.selected.pt", map_location="cpu", weights_only=False)
        recovered = torch.load(recovery_dir / target / "5shot" / "seed42" / f"{method}.selected.pt",
                               map_location="cpu", weights_only=False)
        if any(not torch.equal(expected["state_dict"][key], recovered["state_dict"][key])
               for key in expected["state_dict"]):
            raise RuntimeError(f"{method} continuation recovery changed the trained snapshot")
        expected_last = torch.load(condition_dir / f"{method}.last.pt", map_location="cpu", weights_only=False)
        recovered_last = torch.load(recovery_dir / target / "5shot" / "seed42" / f"{method}.last.pt",
                                    map_location="cpu", weights_only=False)
        for key in ("optimizer", "order", "cursor", "order_rng_state", "torch_rng_state",
                    "numpy_rng_state", "python_rng_state", "cuda_rng_state", "lora_state_dict"):
            if not _same_checkpoint_value(expected_last.get(key), recovered_last.get(key)):
                raise RuntimeError(f"{method} continuation recovery changed {key}")
        original_snapshot = condition_dir / f"{method}.step25.pt"
        recovered_snapshot = recovery_dir / target / "5shot" / "seed42" / f"{method}.step25.pt"
        if _sha(original_snapshot) != _sha(recovered_snapshot):
            raise RuntimeError(f"{method} recovered snapshot bytes differ from the uninterrupted run")
    result["measurements"]["adaptation_recovery_seconds"] = recovery_measurements
    result["measurements"]["artifact_storage_bytes"] = _tree_bytes(workspace)
    result["measurements"]["static_asset_cache_bytes"] = (_tree_bytes(feature_dir)
        + Path(mae_checkpoint).stat().st_size + _tree_bytes(Path(libero_repo))
        + sum(Path(spec["path"]).stat().st_size for spec in protocol["tasks"].values()))
    result["measurements"]["source_run_storage_bytes"] = _tree_bytes(source_dir)
    for method in ("head_ft", "lora_head_ft"):
        result["measurements"][f"{method}_run_storage_bytes"] = sum(
            path.stat().st_size for path in condition_dir.glob(f"{method}.*") if path.is_file())

    result.update({"status": "complete", "protocol_digest": protocol["digest"],
        "study_config_sha256": _config_digest(_scientific_config()),
        "code_snapshot_sha256": _code_snapshot_digest(),
        "cache_identity_sha256": cache.get("identity_sha256"),
        "asset_hashes": {"mae": _sha(Path(mae_checkpoint)),
            "dataset_files": {str(row["task"]): row["sha256"] for row in cache.get("dataset_files", [])}},
        "mae_checkpoint_sha256": _sha(Path(mae_checkpoint)), "mae_implementation": MAE_IMPLEMENTATION,
        "raw_input_ids": input_ids, "raw_input_sha256": input_hash,
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
            "cuda_device": torch.cuda.get_device_name(dev), "torch": torch.__version__,
            "timm": MAE_IMPLEMENTATION.split(":", 1)[0]},
        "cache_manifest_sha256": _sha(cache_manifest_path), "completion_receipt_written": False,
        "feature_cache_wall_seconds": cache.get("measurements", {}).get("feature_cache_wall_seconds")})
    # All required measurements must be present and finite before either
    # immutable artifact is published.
    _validate_measurements(result["measurements"])
    result["completion_receipt_written"] = True
    _atomic_json(result_path, result)
    receipt = {"receipt_format": 1, "measurement_sha256": _sha(result_path),
        "protocol_digest": result["protocol_digest"], "study_config_sha256": result["study_config_sha256"],
        "code_snapshot_sha256": result["code_snapshot_sha256"],
        "cache_identity_sha256": result["cache_identity_sha256"],
        "mae_checkpoint_sha256": result["mae_checkpoint_sha256"],
        "asset_hashes": result["asset_hashes"],
        "cache_manifest_sha256": result["cache_manifest_sha256"],
        "raw_input_ids": result["raw_input_ids"], "raw_input_sha256": input_hash,
        "hardware": result["hardware"],
        "status": "complete"}
    _atomic_json(out / "calibration_completion.json", receipt)
    return result


def collect_calibration(protocol_path, feature_dir, mae_checkpoint, output_dir, *, device="cuda",
                        libero_repo=None) -> dict:
    """Run calibration and retain a failure diagnostic without certifying it."""
    try:
        return _collect_calibration(protocol_path, feature_dir, mae_checkpoint, output_dir,
                                     device=device, libero_repo=libero_repo)
    except Exception as exc:
        out = Path(output_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
        receipt = out / "calibration_completion.json"
        diagnostic = {"measurement_format": 3, "collector_version": 3, "status": "failed",
            "device": str(device), "scientific_result": False, "completion_receipt_written": False,
            "reason": f"{type(exc).__name__}: {exc}"}
        if not receipt.exists() and not (out / "collect_calibration.json").exists():
            _atomic_json(out / "collect_calibration.json", diagnostic)
        return diagnostic
