"""Read-only asset gates, real-checkpoint smoke, and conservative cost calibration."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import subprocess
import platform
import sys
import time
from pathlib import Path

import torch

from .fewshot import LIBERO_COMMIT, PROTOCOL, validate_goal_protocol, workload_inventory



def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    commands = [["git", "-C", str(repo), *args]]
    fallback = Path("/Users/ConstiX/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git")
    if fallback.is_file():
        commands.append([str(fallback), "-C", str(repo), *args])
    last = None
    for command in commands:
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        except Exception as exc:
            last = exc
    raise RuntimeError(str(last))


def audit_complete_feature_cache(feature_dir, protocol: dict) -> dict:
    """Verify every expected shard, including sealed-query label exclusion."""
    root = Path(feature_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format") != 4 or manifest.get("protocol_digest") != protocol["digest"]
            or manifest.get("feature_dim") != 768 or manifest.get("horizon") != 1):
        raise ValueError("complete feature cache manifest identity is invalid")
    encoder_keys = ("execution_mode", "horizon", "future_representation", "mae_checkpoint_sha256",
        "mae_implementation", "preprocessing", "preprocessing_version", "action_convention",
        "feature_dim", "libero_commit")
    encoder_identity = {key: manifest.get(key) for key in encoder_keys}
    expected_encoder_sha = hashlib.sha256(json.dumps(encoder_identity, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if manifest.get("identity_sha256") != expected_encoder_sha:
        raise ValueError("feature cache encoder identity digest is invalid")
    expected_encoder = {**encoder_identity, "identity_sha256": expected_encoder_sha}
    shard_map = manifest.get("demo_shards")
    if not isinstance(shard_map, dict) or set(shard_map) != set(protocol["tasks"]):
        raise ValueError("feature cache does not enumerate every protocol task")
    checked = query_count = label_count = 0
    for task, spec in protocol["tasks"].items():
        expected_roles = {}
        if spec["role"] == "source":
            for group, demos in spec["source"].items():
                for demo in demos:
                    expected_roles[str(int(demo))] = group
        else:
            query_ids = {str(int(demo)) for demo in spec["query"]}
            for demo in spec.get("support_pool", []):
                expected_roles[str(int(demo))] = "support_pool"
            for demo in query_ids:
                expected_roles[demo] = "query"
        records = shard_map[task]
        if set(records) != set(expected_roles):
            raise ValueError(f"cache shard enumeration is incomplete or unexpected for {task}")
        for demo, role in expected_roles.items():
            record = records[demo]
            path = root / record.get("path", "")
            if (record.get("role") != role or not path.is_file()
                    or _sha(path) != record.get("sha256")):
                raise ValueError(f"cache shard role, presence, or hash is invalid for {task}:{demo}")
            shard = torch.load(path, map_location="cpu", weights_only=False)
            count = int(spec["transitions"][demo])
            ids = [f"{task}:{demo}:{step}" for step in range(count)]
            if (shard.get("protocol_digest") != protocol["digest"]
                    or shard.get("task") != task or str(shard.get("demo_id")) != demo
                    or shard.get("role") != role or shard.get("source_sha256") != spec["source_sha256"]
                    or shard.get("encoder_identity") != expected_encoder
                    or record.get("encoder_identity_sha256") != expected_encoder_sha
                    or shard.get("transition_count") != count or shard.get("transition_ids") != ids
                    or tuple(shard.get("feat_t", torch.empty(0)).shape) != (count, 768)
                    or tuple(shard.get("feat_next", torch.empty(0)).shape) != (count, 768)
                    or not torch.isfinite(shard["feat_t"]).all()
                    or not torch.isfinite(shard["feat_next"]).all()):
                raise ValueError(f"cache shard tensor or provenance validation failed for {task}:{demo}")
            has_labels = role != "query"
            forbidden = {"actions", "action", "actions_raw", "targets"}
            if bool(shard.get("has_actions")) != has_labels or bool(record.get("has_actions")) != has_labels:
                raise ValueError(f"cache label flag is wrong for {task}:{demo}")
            if has_labels:
                actions = shard.get("actions_raw")
                if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != (count, 7) or not torch.isfinite(actions).all():
                    raise ValueError(f"cache action labels are invalid for {task}:{demo}")
                label_count += 1
            elif forbidden.intersection(shard):
                raise ValueError(f"sealed query shard contains labels for {task}:{demo}")
            checked += 1
            query_count += role == "query"
    return {"valid": True, "format": 4, "checked_shards": checked,
            "query_shards": query_count, "labeled_shards": label_count,
            "cache_manifest_sha256": _sha(manifest_path)}


def verify_assets(protocol_path, feature_dir, mae_checkpoint=None, *, libero_repo=None, asset_notes=None, require_cache=False):
    checks, errors = {}, []
    try:
        protocol = validate_goal_protocol(protocol_path)
        checks["protocol"] = {"valid": True, "version": protocol["protocol"], "digest": protocol["digest"]}
    except Exception as exc:
        protocol = None
        checks["protocol"] = {"valid": False, "error": str(exc)}
        errors.append("protocol manifest missing or invalid")
    checks["dataset"] = {"valid": protocol is not None, "task_count": len(protocol["tasks"]) if protocol else 0}
    checkpoint = Path(mae_checkpoint).expanduser() if mae_checkpoint else None
    checks["mae_checkpoint"] = {"path": str(checkpoint) if checkpoint else None,
        "exists": bool(checkpoint and checkpoint.is_file()), "sha256": _sha(checkpoint) if checkpoint else None}
    if not checks["mae_checkpoint"]["exists"]: errors.append("standalone MAE ViT-B/16 checkpoint unavailable")
    repo = Path(libero_repo).expanduser().resolve() if libero_repo else None
    try:
        if not repo or not repo.is_dir(): raise FileNotFoundError("pinned LIBERO checkout is absent")
        commit, dirty = _git(repo, "rev-parse", "HEAD"), bool(_git(repo, "status", "--porcelain"))
        checks["libero_repo"] = {"path": str(repo), "commit": commit, "expected_commit": LIBERO_COMMIT,
            "pinned": commit == LIBERO_COMMIT, "clean": not dirty}
    except Exception as exc:
        checks["libero_repo"] = {"path": str(repo) if repo else None, "pinned": False, "clean": False, "error": str(exc)}
    if not checks["libero_repo"].get("pinned") or not checks["libero_repo"].get("clean"):
        errors.append("LIBERO checkout is absent, dirty, or not at the pinned revision")
    try:
        notes = json.loads(Path(asset_notes).read_text()) if asset_notes else None
        record = notes.get("mae", {}) if isinstance(notes, dict) else {}
        if (not record.get("origin") or not record.get("pretraining_exposure")
                or record.get("sha256") != checks["mae_checkpoint"].get("sha256")):
            raise ValueError("MAE origin, matching hash, and pretraining exposure note are required")
        if protocol is None:
            raise ValueError("protocol manifest is required to validate dataset provenance")
        dataset_note = notes.get("libero_goal", {})
        source_note = notes.get("libero_source", {})
        expected_files = {task: {"path": spec["path"], "sha256": spec["source_sha256"]}
                          for task, spec in protocol["tasks"].items()}
        recorded_files = dataset_note.get("files", {})
        if (not dataset_note.get("origin") or not dataset_note.get("fallback_origin")
                or dataset_note.get("fallback_revision") != "f13aa24a3da8c43c7225569f28c562979fa0e35a"
                or recorded_files != expected_files
                or source_note.get("revision") != LIBERO_COMMIT or source_note.get("clean_checkout") is not True):
            raise ValueError("LIBERO dataset/source origin, revision, or file hashes do not match the protocol")
        checks["checkpoint_exposure_notes"] = {"valid": True, "path": str(asset_notes), "notes": notes}
    except Exception as exc:
        checks["checkpoint_exposure_notes"] = {"valid": False, "path": str(asset_notes) if asset_notes else None, "error": str(exc)}
        errors.append("MAE checkpoint origin/hash/pretraining-exposure notes are missing or invalid")
    cache_manifest = Path(feature_dir) / "manifest.json"
    cache_ok = False
    try:
        cache = json.loads(cache_manifest.read_text())
        from .fewshot_train import _load_cache_map
        if protocol is None: raise ValueError("protocol invalid")
        if cache.get("mae_checkpoint_sha256") != checks["mae_checkpoint"].get("sha256"):
            raise ValueError("cache was built with another MAE checkpoint")
        _load_cache_map(feature_dir, protocol, run_mode="scientific")
        cache_audit = audit_complete_feature_cache(feature_dir, protocol)
        cache_ok = True
        checks["feature_cache"] = {"path": str(cache_manifest), "exists": True, "valid": True,
            "format": cache.get("format"), "encoder_identity_sha256": cache.get("identity_sha256"),
            "complete_audit": cache_audit}
    except Exception as exc:
        checks["feature_cache"] = {"path": str(cache_manifest), "exists": cache_manifest.is_file(),
            "valid": False, "error": str(exc)}
    if require_cache and not cache_ok: errors.append("format-4 standalone-MAE feature cache absent or invalid")
    ready = (protocol is not None and checks["mae_checkpoint"]["exists"]
        and checks["libero_repo"].get("pinned") and checks["libero_repo"].get("clean")
        and checks["checkpoint_exposure_notes"].get("valid"))
    checks["checkpoint_cache_match"] = (not cache_manifest.is_file() or not checks["mae_checkpoint"]["exists"]
        or (json.loads(cache_manifest.read_text()).get("mae_checkpoint_sha256") == checks["mae_checkpoint"].get("sha256")))
    return {"protocol": PROTOCOL, "ready": ready and (cache_ok if require_cache else True),
        "assets_ready": ready, "cache_ready": cache_ok, "checks": checks,
        "blocking_reasons": errors, "paid_execution": "disabled"}


def _smoke_real_impl(protocol_path, feature_dir, mae_checkpoint,
               *, libero_repo=None, asset_notes=None, device="cpu",
               output_dir="artifacts/fewshot_libero_goal_v3") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    result_path = out / "smoke_real.json"
    result = {"status": "running", "stage": "smoke-real", "device": device,
              "paid_execution": "disabled"}
    phase = "asset verification"
    def mark_phase(value):
        nonlocal phase
        phase = value; result["phase"] = value
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    mark_phase(phase)
    assets = verify_assets(protocol_path, feature_dir, mae_checkpoint,
                           libero_repo=libero_repo, asset_notes=asset_notes, require_cache=True)
    if not assets["ready"]:
        result.update(status="blocked", asset_gate=assets,
                      reason="verified LIBERO data, standalone MAE checkpoint, pinned LIBERO source, notes, and format-4 cache are required")
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose cpu or mps explicitly")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    from .fewshot import DEVELOPMENT_TASKS, SOURCE_TASKS, validate_goal_protocol
    from .fewshot_train import (_encode_live, _load_encoder_with_lora, _sha,
                                adapt_target, train_source)
    from .models import DirectDecoder, attach_mae_lora
    from src.mae_encoder import MAEImageEncoder
    from src.data_libero import LazyLiberoTransitionDataset
    from torch.utils.data import DataLoader
    protocol = validate_goal_protocol(protocol_path)
    cache_manifest = json.loads((Path(feature_dir) / "manifest.json").read_text())
    source_task = SOURCE_TASKS[0]
    source_demo = int(protocol["tasks"][source_task]["source"]["source_train"][0])
    spec = protocol["tasks"][source_task]
    mark_phase("checkpoint loading and preprocessing")
    encoder = MAEImageEncoder.from_checkpoint(mae_checkpoint, device=dev)
    encoder.eval()
    mark_phase("raw/cache feature comparison")
    frozen_base = {key: value.detach().cpu().clone() for key, value in encoder.mae.state_dict().items()}
    norm = {"action_mean": torch.zeros(7), "action_std": torch.ones(7)}
    dataset = LazyLiberoTransitionDataset(spec["path"], norm, demo_ids={source_demo}, include_actions=True)
    batch = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=False, num_workers=0)))
    with torch.no_grad():
        raw_t, raw_n = encoder.encode_frames(batch["image_t"].to(dev)), encoder.encode_frames(batch["image_next"].to(dev))
    record = cache_manifest["demo_shards"][source_task][str(source_demo)]
    shard = torch.load(Path(feature_dir) / record["path"], map_location="cpu", weights_only=False)
    raw_cache_error = max(float((raw_t.cpu() - shard["feat_t"][:len(raw_t)]).abs().max()),
                          float((raw_n.cpu() - shard["feat_next"][:len(raw_n)]).abs().max()))
    if raw_cache_error > 2e-4:
        raise RuntimeError(f"raw/cache image embeddings differ by {raw_cache_error:.6g}")
    attach_mae_lora(encoder.mae, (10, 11), rank=8, alpha=16, dropout=0.0)
    encoder.eval()
    with torch.no_grad():
        zero_lora_t = encoder.encode_frames(batch["image_t"].to(dev))
        zero_lora_n = encoder.encode_frames(batch["image_next"].to(dev))
    zero_adapter_error = max(float((raw_t - zero_lora_t).abs().max()),
                             float((raw_n - zero_lora_n).abs().max()))
    if zero_adapter_error > 2e-4:
        raise RuntimeError(f"zero-initialized real LoRA changed base features by {zero_adapter_error:.6g}")
    timings = {}; source_out = None
    try:
        work = out / "smoke_workspace"
        work.mkdir(parents=True, exist_ok=True)
        source_out = work / "source"
        mark_phase("production source training")
        start = time.perf_counter()
        source_result = train_source(protocol_path, feature_dir, source_out, seed=42, updates=2,
                                     batch_size=8, microbatch_size=4, device=str(dev), run_mode="calibration")
        timings["source_training_2_updates_seconds"] = time.perf_counter() - start
        source_checkpoint = source_out / "source_seed42.pt"
        target_task = DEVELOPMENT_TASKS[0]
        mark_phase("production head adaptation")
        head_start = time.perf_counter()
        head = adapt_target(protocol_path, feature_dir, source_checkpoint, work / "runs",
            task=target_task, method="head_ft", budget=5, seed=42, updates=2,
            batch_size=8, microbatch_size=4, device=str(dev), run_mode="calibration")
        timings["head_adaptation_2_updates_seconds"] = time.perf_counter() - head_start
        mark_phase("production LoRA adaptation")
        lora_start = time.perf_counter()
        lora = adapt_target(protocol_path, feature_dir, source_checkpoint, work / "runs",
            task=target_task, method="lora_head_ft", budget=5, seed=42, updates=2,
            batch_size=8, microbatch_size=4, device=str(dev), mae_checkpoint=mae_checkpoint,
            run_mode="calibration")
        timings["lora_adaptation_2_updates_seconds"] = time.perf_counter() - lora_start
        head_file = work / "runs" / target_task / "5shot" / "seed42" / "head_ft.step2.pt"
        lora_file = work / "runs" / target_task / "5shot" / "seed42" / "lora_head_ft.step2.pt"
        hp = torch.load(head_file, map_location="cpu", weights_only=False)
        lp = torch.load(lora_file, map_location="cpu", weights_only=False)
        adapter_changed = any(not torch.equal(lp["initial_lora_state_dict"][key], value)
                              for key, value in lp["lora_state_dict"].items())
        if (head["status"] != "calibration" or lora["status"] != "calibration"
                or not lora.get("lora_parameters_changed") or not adapter_changed
                or not lora.get("frozen_backbone_unchanged")
                or not torch.isfinite(torch.tensor(lora.get("max_lora_gradient_norm", float("nan"))))
                or lora.get("max_lora_gradient_norm", 0.0) <= 0):
            raise RuntimeError("production adaptation smoke did not complete or update LoRA")
        mark_phase("checkpoint prediction reload")
        model_b = DirectDecoder(768, 7).to(dev); model_b.load_state_dict(lp["state_dict"]); model_b.eval()
        encoder_b, _ = _load_encoder_with_lora(mae_checkpoint, dev,
                                                (10, 11), 8, 16, 0.0)
        expected_keys = set(lp["lora_state_dict"])
        if expected_keys != {k for k in encoder_b.mae.state_dict() if "lora_" in k}:
            raise RuntimeError("saved LoRA adapter keys do not match real base model")
        encoder_b.mae.load_state_dict(lp["lora_state_dict"], strict=False); encoder_b.eval()
        reloaded_state = encoder_b.mae.state_dict()
        for key, value in frozen_base.items():
            wrapped = key.replace(".qkv.", ".qkv.base.").replace(".proj.", ".proj.base.")
            actual = reloaded_state[wrapped if wrapped in reloaded_state else key].detach().cpu()
            if not torch.equal(actual, value):
                raise RuntimeError(f"frozen MAE parameter changed during LoRA adaptation: {key}")
        source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        rows = LazyLiberoTransitionDataset(protocol["tasks"][target_task]["path"],
            {"action_mean": source_payload["normalizer_mean"], "action_std": source_payload["normalizer_std"]},
            demo_ids={int(x) for x in protocol["tasks"][target_task]["support_budgets_by_seed"]["42"]["5"]},
            include_actions=True)
        serialized_probe = lp.get("calibration_probe", {})
        probe_parts = str(serialized_probe.get("transition_id", "")).rsplit(":", 2)
        if len(probe_parts) != 3 or probe_parts[0] != target_task:
            raise RuntimeError("LoRA snapshot is missing a valid live prediction probe")
        probe_demo, probe_timestep = int(probe_parts[1]), int(probe_parts[2])
        probe_index = next((i for i, (demo, _key, timestep) in enumerate(rows.items)
                            if demo == probe_demo and timestep == probe_timestep), None)
        if probe_index is None:
            raise RuntimeError("saved LoRA probe is not present in the selected support demonstrations")
        sample = next(iter(DataLoader(torch.utils.data.Subset(rows, [probe_index]),
                                      batch_size=1, shuffle=False, num_workers=0)))
        with torch.no_grad():
            current = _encode_live(encoder_b, sample["image_t"])
            future = _encode_live(encoder_b, sample["image_next"])
            prediction_after_reload = model_b(current, future)[0]
            encoder.mae.load_state_dict(lp["lora_state_dict"], strict=False); encoder.eval()
            prediction_before_reload = model_b(_encode_live(encoder, sample["image_t"]),
                                               _encode_live(encoder, sample["image_next"]))[0]
        if (not torch.isfinite(prediction_after_reload).all()
                or not torch.allclose(prediction_before_reload, prediction_after_reload, rtol=0, atol=1e-6)):
            raise RuntimeError("reloaded real LoRA model produced nonfinite predictions")
        sample_transition_id = f"{target_task}:{int(sample['demo_id'][0])}:{int(sample['timestep'][0])}"
        if (serialized_probe.get("transition_id") != sample_transition_id
                or not isinstance(serialized_probe.get("prediction"), torch.Tensor)
                or not torch.allclose(serialized_probe["prediction"], prediction_after_reload.cpu(),
                                      rtol=0, atol=1e-6)):
            raise RuntimeError("pre-save and post-reload LoRA predictions differ")
    except Exception as exc:
        result.update(status="failed", phase=phase, error=f"{type(exc).__name__}: {exc}", asset_gate=assets,
                      raw_cache_max_abs_error=raw_cache_error, stage_timings=timings,
                      diagnostic_directory=str(out / "smoke_workspace"))
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
    from .fewshot_train import _code_snapshot_digest, _config_digest, _scientific_config
    try:
        import timm
        timm_version = timm.__version__
    except Exception:
        timm_version = None
    result = {"status": "passed", "device": str(dev), "protocol_digest": protocol["digest"],
              "asset_gate": assets, "source_sample": f"{source_task}:{source_demo}",
              "mae_checkpoint_sha256": assets["checks"]["mae_checkpoint"]["sha256"],
              "raw_cache_max_abs_error": raw_cache_error,
              "zero_adapter_max_abs_error": zero_adapter_error,
              "real_zero_adapter_equivalence": True,
              "production_source_training": True, "production_head_adaptation": True,
              "production_lora_adaptation": True, "recorded_action_targets": True,
              "adapter_reload_prediction_finite": True, "adapter_reload_predictions_match": True,
              "pre_save_prediction_matches_reloaded_checkpoint": True,
              "frozen_backbone_unchanged": True,
              "diagnostic_directory": str(out / "smoke_workspace"), "stage_timings": timings,
              "code_snapshot_sha256": _code_snapshot_digest(),
              "study_config_sha256": _config_digest(_scientific_config()),
              "cache_manifest_sha256": _sha(Path(feature_dir) / "manifest.json"),
              "runtime": {"python": sys.version, "platform": platform.platform(),
                          "torch": torch.__version__, "timm": timm_version},
              "cuda_verified": dev.type == "cuda", "scientific_result": False,
              "paid_execution": "disabled"}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def smoke_real(protocol_path, feature_dir, mae_checkpoint,
               *, libero_repo=None, asset_notes=None, device="cpu",
               output_dir="artifacts/fewshot_libero_goal_v3") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    result_path = out / "smoke_real.json"
    try:
        return _smoke_real_impl(protocol_path, feature_dir, mae_checkpoint,
            libero_repo=libero_repo, asset_notes=asset_notes, device=device,
            output_dir=out)
    except Exception as exc:
        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        result.update({"status": "failed", "stage": "smoke-real", "device": device,
                  "error": f"{type(exc).__name__}: {exc}", "scientific_result": False,
                  "paid_execution": "disabled"})
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        return result


def _forecast_calibration_workload(measurements: dict, pricing: dict, budget: dict,
                                   spent_usd: float, *, scope: str, selected_updates: dict | None = None) -> dict:
    inventory = workload_inventory(scope, selected_updates)
    development_conditions = inventory["development_conditions"]
    confirmation_conditions = inventory["confirmation_conditions"]
    adaptation_runs_per_method = inventory["adaptation_runs_per_method"]
    source_runs = inventory["source_training_runs"]
    candidate_snapshot_evaluations = inventory["development_learned_snapshot_evaluations"]
    # Development evaluates the frozen source plus all six learned snapshots
    # for every task/budget condition. Confirmation evaluates three frozen
    # models (one per repetition) for every preregistered condition.
    development_retention_passes = inventory["development_retention_passes"]
    confirmation_retention_passes = inventory["confirmation_retention_passes"]
    retention_passes = development_retention_passes + confirmation_retention_passes
    # Target-support memory is shared by three controls; pooled source+support
    # memory is a second bank used by its own control.
    retrieval_conditions = inventory["retrieval_bank_builds"]
    source_seconds = measurements["source_training_20_updates_seconds"]
    head_seconds = measurements["head_adaptation_20_updates_seconds"]
    lora_seconds = measurements["lora_adaptation_20_updates_seconds"]
    source_hours = source_seconds * (2000 / 20) * source_runs / 3600
    adaptation_hours = sum(seconds * inventory["adaptation_updates_by_method"][method] / 20
        for method, seconds in (("head_ft", head_seconds), ("lora_head_ft", lora_seconds))) / 3600
    cache_hours = measurements["feature_cache_wall_seconds"] / 3600
    initialization_seconds = (
        measurements.get("feature_cache_encoder_initialization_seconds", 0.0)
        + measurements.get("source_initialization_and_data_prep_seconds", 0.0) * source_runs
        + (measurements.get("head_ft_initialization_and_data_prep_seconds", 0.0)
           + measurements.get("lora_head_ft_initialization_and_data_prep_seconds", 0.0))
          * adaptation_runs_per_method)
    initialization_hours = initialization_seconds / 3600
    evaluation_seconds = (
        measurements["development_baseline_bundle_seconds"] * development_conditions
        + measurements["development_candidate_bundle_seconds"] * candidate_snapshot_evaluations
        + measurements["confirmation_bundle_seconds"] * confirmation_conditions)
    retrieval_seconds = measurements["retrieval_bank_build_seconds"] * retrieval_conditions
    retention_seconds = measurements["retention_pass_seconds"] * retention_passes
    training_runs = source_runs + 2 * adaptation_runs_per_method
    checkpoint_seconds = (measurements["checkpoint_write_seconds"] * inventory["checkpoint_writes"]
        if "checkpoint_write_seconds" in measurements else measurements["checkpoint_io_bundle_seconds"] * training_runs)
    validation_seconds = measurements.get("source_validation_pass_seconds", 0) * inventory["source_validation_passes"]
    recovery_seconds = (measurements["recovery_seconds_per_adaptation"] * (2 * adaptation_runs_per_method)
                        + measurements.get("source_recovery_seconds", 0) * source_runs)
    shared_hours = (evaluation_seconds + retrieval_seconds + retention_seconds
                    + checkpoint_seconds + validation_seconds + recovery_seconds) / 3600
    raw_hours = source_hours + adaptation_hours + cache_hours + initialization_hours + shared_hours
    workload_hours = raw_hours * (1 + budget["workload_margin"])
    storage_bytes = measurements["total_artifact_storage_bytes"]
    if "static_asset_cache_bytes" in measurements:
        # Three scheduled snapshots versus one in calibration; reserve recovery
        # copies as well. Source continuation/candidate files are overwritten.
        storage_bytes = (measurements["static_asset_cache_bytes"]
            + 2 * source_runs * measurements["source_run_storage_bytes"]
            + 2 * 3 * adaptation_runs_per_method * (
                measurements["head_ft_run_storage_bytes"] + measurements["lora_head_ft_run_storage_bytes"]))
    storage_gb = storage_bytes / (1024 ** 3)
    storage_cost = storage_gb * pricing["storage_gb_month_usd"] * pricing.get("retention_months", 1)
    compute_cost = workload_hours * pricing["gpu_hourly_usd"]
    return {"source_training_hours": source_hours, "adaptation_hours": adaptation_hours,
        "initialization_and_data_prep_hours": initialization_hours,
        "source_validation_hours": validation_seconds / 3600,
        "checkpoint_write_hours": checkpoint_seconds / 3600,
        "retrieval_evaluation_retention_io_recovery_hours": shared_hours,
        "feature_cache_hours": cache_hours, "workload_hours_with_margin": workload_hours,
        "storage_gb": storage_gb, "storage_cost_usd": storage_cost,
        "compute_cost_usd": compute_cost, "total_cost_usd": compute_cost + storage_cost,
        "available_before_protected_reserve_usd": budget["planning_budget_usd"]
            - budget["protected_reserve_usd"] - spent_usd,
        "confirmation_scope": scope,
        "condition_matrix": {"source_training_runs": source_runs,
            "development_conditions": development_conditions,
            "development_baseline_evaluations": development_conditions,
            "development_snapshot_evaluations": candidate_snapshot_evaluations,
            "confirmation_conditions": confirmation_conditions,
            "adaptation_runs_per_method": adaptation_runs_per_method,
            "development_retention_passes": development_retention_passes,
            "confirmation_retention_passes": confirmation_retention_passes,
            "retention_passes": retention_passes, "retrieval_bank_builds": retrieval_conditions,
        "checkpoint_io_runs": training_runs,
        "checkpoint_writes": inventory["checkpoint_writes"],
        "checkpoint_write_breakdown": inventory["checkpoint_write_breakdown"],
        "source_validation_passes": inventory["source_validation_passes"],
        "confirmation_updates": inventory["confirmation_updates"],
        "canonical_evaluation_conditions": inventory["canonical_evaluation_conditions"]}}


def calibrate_study(output_dir, config: dict, *, device="cpu", feature_dir=None,
                    pricing_file=None, spent_usd=0.0) -> dict:
    out = Path(output_dir); smoke_path = out / "smoke_real.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.is_file() else None
    missing = []; measurements = {}
    budget_values = {}
    try:
        budget_values = {key: float(config[key]) for key in
                         ("planning_budget_usd", "protected_reserve_usd", "workload_margin")}
        if (not all(math.isfinite(value) for value in budget_values.values())
                or budget_values["planning_budget_usd"] <= 0
                or budget_values["planning_budget_usd"] > 250
                or budget_values["protected_reserve_usd"] < 25
                or budget_values["protected_reserve_usd"] >= budget_values["planning_budget_usd"]
                or budget_values["workload_margin"] < 0.25):
            raise ValueError("budget must be at most $250, reserve at least $25, and workload margin at least 25%")
    except Exception as exc:
        missing.append(f"invalid budget configuration: {exc}")
    if not smoke or smoke.get("status") != "passed": missing.append("passed real-checkpoint smoke")
    if not smoke or smoke.get("cuda_verified") is not True:
        missing.append("measured CUDA workload and memory")
    if not torch.cuda.is_available() or str(device) != "cuda":
        missing.append("CUDA hardware is not available for this calibration run")
    cache_manifest_path = Path(feature_dir) / "manifest.json" if feature_dir else None
    cache = None
    if cache_manifest_path and cache_manifest_path.is_file():
        cache = json.loads(cache_manifest_path.read_text())
    bundle_path = out / "collect_calibration.json"
    bundle = None
    try:
        bundle = json.loads(bundle_path.read_text())
        if bundle.get("measurement_format") != 3 or bundle.get("status") != "complete":
            raise ValueError("calibration bundle is partial, failed, or an unsupported version")
        from .fewshot_train import _code_snapshot_digest, _config_digest, _scientific_config
        expected_identity = {"measurement_format": 3, "collector_version": 3,
            "protocol_digest": cache.get("protocol_digest") if cache else None,
            "study_config_sha256": _config_digest(_scientific_config()),
            "cache_identity_sha256": cache.get("identity_sha256") if cache else None,
            "code_snapshot_sha256": _code_snapshot_digest(),
            "device": "cuda"}
        if any(bundle.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("measurement identity is incompatible with protocol, config, cache, code, or device")
        asset_hashes = {"mae": smoke.get("mae_checkpoint_sha256") if smoke else None,
            "dataset_files": {str(row.get("task")): row.get("sha256")
                              for row in (cache.get("dataset_files", []) if cache else [])}}
        input_ids = bundle.get("raw_input_ids")
        hardware = bundle.get("hardware")
        if (not isinstance(bundle.get("raw_input_sha256"), str) or len(bundle["raw_input_sha256"]) != 64
                or not isinstance(input_ids, list) or not input_ids
                or any(not isinstance(value, str) or not value for value in input_ids)
                or len(input_ids) != len(set(input_ids))
                or bundle.get("asset_hashes") != asset_hashes
                or not isinstance(hardware, dict) or not hardware.get("cuda_device")
                or bundle.get("mae_checkpoint_sha256") != asset_hashes["mae"]):
            raise ValueError("measurement inputs, hardware, or asset identity is incomplete")
        receipt_path = out / "calibration_completion.json"
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("receipt_format") != 1 or receipt.get("status") != "complete"
                or receipt.get("measurement_sha256") != _sha(bundle_path)
                or receipt.get("protocol_digest") != bundle.get("protocol_digest")
                or receipt.get("study_config_sha256") != bundle.get("study_config_sha256")
                or receipt.get("code_snapshot_sha256") != bundle.get("code_snapshot_sha256")
                or receipt.get("mae_checkpoint_sha256") != asset_hashes["mae"]
                or receipt.get("asset_hashes") != bundle.get("asset_hashes")
                or receipt.get("cache_identity_sha256") != bundle.get("cache_identity_sha256")
                or receipt.get("cache_manifest_sha256") != bundle.get("cache_manifest_sha256")
                or receipt.get("raw_input_ids") != input_ids
                or receipt.get("raw_input_sha256") != bundle.get("raw_input_sha256")
                or receipt.get("raw_input_sha256") != bundle.get("raw_input_sha256")
                or receipt.get("hardware") != hardware):
            raise ValueError("measurements lack a matching completed collector receipt")
        recorded = bundle.get("measurements", {})
        from .fewshot_calibration import _validate_measurements
        _validate_measurements(recorded)
        cache_timing = cache.get("measurements", {}).get("feature_cache_wall_seconds") if cache else None
        update_values = [recorded.get(f"{method}_update_seconds") for method in
                         ("source", "head_ft", "lora_head_ft")]
        if any(not isinstance(values, list) or len(values) != 20 for values in update_values):
            raise ValueError("calibration did not record twenty same-session updates for every learner")
        for values in update_values:
            if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0
                   for value in values):
                raise ValueError("calibration update timing samples are invalid")
        measurements = {
            "feature_cache_wall_seconds": cache_timing,
            "feature_cache_encoder_initialization_seconds": recorded.get("feature_cache_encoder_initialization_seconds"),
            "source_initialization_and_data_prep_seconds": recorded.get("source_initialization_and_data_prep_seconds"),
            "head_ft_initialization_and_data_prep_seconds": recorded.get("head_ft_initialization_and_data_prep_seconds"),
            "lora_head_ft_initialization_and_data_prep_seconds": recorded.get("lora_head_ft_initialization_and_data_prep_seconds"),
            "source_training_20_updates_seconds": sum(recorded["source_update_seconds"]),
            "head_adaptation_20_updates_seconds": sum(recorded["head_ft_update_seconds"]),
            "lora_adaptation_20_updates_seconds": sum(recorded["lora_head_ft_update_seconds"]),
            "retrieval_bank_build_seconds": recorded.get("retrieval_bank_construction_seconds"),
            "development_baseline_bundle_seconds": recorded.get("eight_method_query_evaluation_seconds"),
            "development_candidate_bundle_seconds": recorded.get("eight_method_query_evaluation_seconds"),
            "confirmation_bundle_seconds": recorded.get("eight_method_query_evaluation_seconds"),
            "retention_pass_seconds": sum(recorded.get("source_retention_seconds", {}).values()) / 3,
            "checkpoint_io_bundle_seconds": recorded.get("checkpoint_io_copy_seconds"),
            "source_validation_pass_seconds": max(recorded["training_profiles"]["source"]["source_validation_seconds"]),
            # A continuation at 1,000 updates contains three archived snapshots;
            # calibration contains one. Use the slowest observed write and
            # reserve a factor of three for the larger continuation payload.
            "checkpoint_write_seconds": 3 * max(
                value for profile in recorded["training_profiles"].values()
                for value in (profile["checkpoint_write_seconds"] + [
                    max(profile["checkpoint_serialization_seconds"]) + max(profile["snapshot_publish_seconds"])])),
            "source_recovery_seconds": recorded["source_recovery_seconds"],
            "recovery_seconds_per_adaptation": max(recorded.get("adaptation_recovery_seconds", {}).values()),
            "peak_memory_bytes": recorded.get("peak_memory_bytes"),
            "total_artifact_storage_bytes": recorded.get("artifact_storage_bytes"),
            **{key: recorded[key] for key in ("static_asset_cache_bytes", "source_run_storage_bytes",
                                               "head_ft_run_storage_bytes", "lora_head_ft_run_storage_bytes")}}
    except Exception as exc:
        missing.append(f"valid bound calibration measurements: {exc}")
    if not cache or not cache.get("measurements", {}).get("feature_cache_wall_seconds"):
        missing.append("measured format-4 feature-cache creation")
    elif measurements.get("feature_cache_wall_seconds") != cache.get("measurements", {}).get("feature_cache_wall_seconds"):
        missing.append("calibration feature-cache timing differs from the verified cache manifest")
    if cache and smoke:
        if (smoke.get("protocol_digest") != cache.get("protocol_digest")
                or smoke.get("mae_checkpoint_sha256") != cache.get("mae_checkpoint_sha256")):
            missing.append("smoke and cache protocol/checkpoint identities differ")
    required_measurements = ("feature_cache_wall_seconds", "feature_cache_encoder_initialization_seconds",
        "source_initialization_and_data_prep_seconds", "head_ft_initialization_and_data_prep_seconds",
        "lora_head_ft_initialization_and_data_prep_seconds", "source_training_20_updates_seconds",
        "head_adaptation_20_updates_seconds", "lora_adaptation_20_updates_seconds",
        "retrieval_bank_build_seconds", "development_baseline_bundle_seconds",
        "development_candidate_bundle_seconds", "confirmation_bundle_seconds",
        "retention_pass_seconds", "checkpoint_io_bundle_seconds",
        "recovery_seconds_per_adaptation",
        "peak_memory_bytes", "total_artifact_storage_bytes")
    missing.extend(f"measured {key.replace('_', ' ')}" for key in required_measurements if key not in measurements)
    pricing = None
    try:
        if pricing_file:
            pricing = json.loads(Path(pricing_file).read_text())
            price_values = (float(pricing.get("gpu_hourly_usd", float("nan"))),
                            float(pricing.get("storage_gb_month_usd", float("nan"))),
                            float(pricing.get("retention_months", 1)))
            if (not pricing.get("verified") or not pricing.get("source_url")
                    or not all(math.isfinite(x) for x in price_values)
                    or price_values[0] <= 0 or price_values[1] < 0 or price_values[2] <= 0):
                missing.append("verified GPU and storage pricing evidence")
        else:
            missing.append("verified GPU and storage pricing evidence")
    except Exception:
        pricing = None
        missing.append("verified GPU and storage pricing evidence")
    try:
        spent_value = float(spent_usd)
        if not math.isfinite(spent_value) or spent_value < 0:
            raise ValueError("negative or non-finite amount")
    except Exception:
        spent_value = float("nan")
        missing.append("prior spending must be finite and non-negative")
    numeric_measurements = [value for key, value in measurements.items()
                            if key.endswith("_seconds") or key.endswith("_bytes")]
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0
           for value in numeric_measurements):
        missing.append("all supplied measurements must be finite and non-negative")
    for key in required_measurements:
        value = measurements.get(key)
        if isinstance(value, (int, float)) and (not math.isfinite(float(value)) or float(value) <= 0):
            missing.append(f"measured {key.replace('_', ' ')} must be finite and positive")
    if cache and cache.get("measurements", {}).get("device") != "cuda":
        missing.append("feature cache timing must come from CUDA for a GPU forecast")
    if smoke and smoke.get("device") != "cuda":
        missing.append("real smoke timing must come from CUDA for a GPU forecast")
    source_seconds = measurements.get("source_training_20_updates_seconds")
    head_seconds = measurements.get("head_adaptation_20_updates_seconds")
    lora_seconds = measurements.get("lora_adaptation_20_updates_seconds")
    forecast = None; forecasts = {}
    required_timing_values = (source_seconds, head_seconds, lora_seconds)
    if all(isinstance(x, (int, float)) and math.isfinite(float(x)) and float(x) > 0
           for x in required_timing_values) and pricing and not missing and budget_values:
        forecasts = {scope: _forecast_calibration_workload(measurements, pricing, budget_values, spent_value,
                                                           scope=scope) for scope in ("all", "five_shot")}
        forecast = forecasts["all"]
        if forecast["total_cost_usd"] > forecast["available_before_protected_reserve_usd"]:
            fallback = forecasts["five_shot"]
            if fallback["total_cost_usd"] <= fallback["available_before_protected_reserve_usd"]:
                forecast = fallback
            else:
                missing.append("full and five-shot workloads exceed allocation while preserving reserve")
    if forecast is None:
        missing.append("complete finite workload forecast is unavailable")
    forecast_ready = forecast is not None and math.isfinite(float(forecast["total_cost_usd"]))
    result = {"status": "ready" if not missing and forecast_ready else "blocked", "protocol": PROTOCOL,
              "device_requested": device, "cuda_available": torch.cuda.is_available(),
              "smoke_artifact": str(smoke_path) if smoke else None, "missing_evidence": missing,
              "measurements": measurements, "pricing_evidence": pricing,
              "forecast": forecast, "forecasts_by_scope": forecasts, "spent_usd": spent_value,
              "budget": {**budget_values, "paid_execution": "disabled"},
              "stage_breakdown_required": ["feature caching", "source training", "head adaptation",
                  "LoRA adaptation", "retrieval-bank construction", "evaluation", "retention",
                  "checkpoint I/O", "storage", "recovery"],
              "next_experiment": {"task": "one development task", "support_demonstrations": 5,
                  "seed": 42, "methods": sorted(["frozen_source", "target_knn", "head_ft", "lora_head_ft",
                      "support_mean", "transition_1nn", "current_weighted_knn", "source_support_knn"]),
                  "gate": "run only after real-input smoke and complete cost forecast pass"}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibrate_study.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
