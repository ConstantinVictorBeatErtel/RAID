"""Command line entry point used by Luna and Runpod."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os

from .cloud import (account, choose_gpu, create_pod, delete_network_volume, delete_pod,
                    estimate, render_runbook, status, upload_repository, prepare_remote_environment,
                    upload_file, start_remote_job, wait_remote_job, collect_remote_results)
from .config import enumerate_expansion, load_config
from .report import build_report
from .splits import create_split, resolve_task_files, load_and_validate
from .training import run_condition


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m raid_experiments")
    p.add_argument("command", choices=["preflight", "estimate", "run", "run-local", "resume", "status", "collect", "cleanup", "make-split", "enumerate-expansion", "prepare-protocol", "validate-protocol", "cache-features", "train-source", "adapt-target", "evaluate-task", "freeze-development", "register-confirmation", "report-fewshot", "verify-assets", "smoke-real", "collect-calibration", "calibrate-study"])
    p.add_argument("--config", default="configs/fewshot_libero_goal.yaml")
    p.add_argument("--pod-id")
    p.add_argument("--volume-id")
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mae-ckpt", help="Pinned MAE checkpoint for raw-image LoRA runs")
    p.add_argument("--provider", choices=["runpod"], help="Cloud provider override")
    p.add_argument("--budget-usd", type=float, help="Project ceiling override")
    p.add_argument("--experiment-dir", help="Existing experiment directory for resume/collect")
    p.add_argument("--output-dir", help="Override output directory for an isolated remote run")
    p.add_argument("--resume-dir", help="Existing local result directory containing condition checkpoints")
    p.add_argument("--stop-after-updates", type=int, help="Calibration stop point; preserves the full schedule for later resume")
    p.add_argument("--delete-volume", action="store_true", help="Delete the owned volume after verified collection")
    p.add_argument("--dataset-dir", help="LIBERO-Goal HDF5 root for protocol preparation")
    p.add_argument("--manifest", help="Few-shot protocol manifest path")
    p.add_argument("--expected-demos", type=int, default=50)
    p.add_argument("--feature-dir", help="Versioned horizon-one standalone MAE feature cache directory")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size used by MAE feature-cache generation")
    p.add_argument("--source-checkpoint", help="Source IDM checkpoint for target stages")
    p.add_argument("--source-checkpoints", help="Comma-separated seed42,seed43,seed44 source checkpoint paths for freeze-development")
    p.add_argument("--task", help="Official LIBERO-Goal task name")
    p.add_argument("--method", help="Adaptation method name")
    p.add_argument("--methods", help="Comma-separated evaluation methods (defaults to configured study methods and controls)")
    p.add_argument("--budget", type=int, choices=[2, 5, 10], help="Target support demonstration budget")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--updates", type=int)
    p.add_argument("--run-mode", choices=["scientific", "calibration"], default="scientific")
    p.add_argument("--selected-step", type=int, help="Development-selected adaptation update count")
    p.add_argument("--head-step", type=int, choices=[200, 500, 1000], help="Frozen development update count for head_ft")
    p.add_argument("--lora-step", type=int, choices=[200, 500, 1000], help="Frozen development update count for lora_head_ft")
    p.add_argument("--device", default="cuda")
    p.add_argument("--freeze-file", help="Development freeze record used to authorize final-test evaluation")
    p.add_argument("--confirmation-scope", choices=["all", "five_shot"], default="all")
    p.add_argument("--confirmation-file", help="Registered confirmation input for final stages; output for register-confirmation")
    p.add_argument("--study-root", help="Canonical source-checkpoint directory for confirmation and final-query ledger")
    p.add_argument("--pricing-file", help="JSON with verified GPU and storage rates plus source URL")
    p.add_argument("--spent-usd", type=float, default=0.0, help="Amount already charged to the study budget")
    p.add_argument("--resume-checkpoint", help="Matching source/adaptation continuation checkpoint")
    p.add_argument("--study-config", default="configs/fewshot_libero_goal.yaml",
                   help="Versioned config for the few-shot LIBERO-Goal offline stages")
    p.add_argument("--libero-repo", help="Pinned LIBERO source checkout")
    p.add_argument("--asset-notes", help="JSON recording checkpoint origins and pretraining exposure")
    return p


def preflight(cfg) -> dict:
    cfg.validate()
    split = Path(cfg.data.split_path)
    features = Path(cfg.data.feature_dir)
    report = {"config_digest": cfg.digest(), "config": cfg.to_dict(), "checks": {}}
    report["checks"]["split"] = load_and_validate(split) if split.exists() else "missing (run make-split after downloading data)"
    if isinstance(report["checks"]["split"], dict):
        report["checks"]["split_horizon_matches"] = report["checks"]["split"].get("horizon", 1) == cfg.horizon
        report["checks"]["task_count_matches"] = len(report["checks"]["split"].get("tasks", [])) == cfg.data.tasks
    report["checks"]["features"] = features.exists()
    if features.exists():
        manifest = features / "manifest.json"
        report["checks"]["feature_manifest"] = json.loads(manifest.read_text()) if manifest.exists() else "missing (run feature cache preparation)"
        if isinstance(report["checks"]["feature_manifest"], dict):
            report["checks"]["feature_horizon_matches"] = (
                report["checks"]["feature_manifest"].get("horizon") == cfg.horizon
                and report["checks"]["feature_manifest"].get("future_representation") == "recorded_rgb"
            )
            dataset_checks = []
            for entry in report["checks"]["feature_manifest"].get("dataset_files", []):
                path = Path(entry["path"])
                dataset_checks.append(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry.get("sha256"))
            report["checks"]["feature_source_hashes"] = bool(dataset_checks) and all(dataset_checks)
    git_result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if git_result.returncode != 0:
        fallback = Path("/Users/ConstiX/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git")
        if fallback.exists():
            git_result = subprocess.run([str(fallback), "rev-parse", "HEAD"], capture_output=True, text=True)
    report["checks"]["git"] = git_result.stdout.strip() or "unavailable"
    lock = Path("requirements-pilot.txt")
    report["checks"]["dependency_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else "missing"
    try:
        account_payload = account()
        report["checks"]["runpod_account"] = {k: account_payload.get(k) for k in ("clientBalance", "spendLimit", "currentSpendPerHr")}
        report["checks"]["gpu"] = choose_gpu(cfg)
    except Exception as exc:
        report["checks"]["runpod"] = f"unavailable: {exc}"
    report["checks"]["lora_scope"] = "MAE final blocks only in raw-image runner; cached-feature runner refuses LoRA"
    report["checks"]["predictor"] = cfg.predictor.__dict__
    output = Path(cfg.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def _require_ready(report: dict) -> None:
    checks = report.get("checks", {})
    if not isinstance(checks.get("split"), dict):
        raise RuntimeError("preflight failed: immutable split manifest is missing or invalid")
    if checks.get("split_horizon_matches") is not True:
        raise RuntimeError("preflight failed: split transition counts were generated for another horizon")
    if checks.get("task_count_matches") is not True:
        raise RuntimeError("preflight failed: configured benchmark task count does not match split")
    if checks.get("features") is not True:
        raise RuntimeError("preflight failed: versioned feature caches are missing")
    if checks.get("feature_horizon_matches") is not True:
        raise RuntimeError("preflight failed: feature cache horizon/representation does not match config")
    if checks.get("feature_source_hashes") is not True:
        raise RuntimeError("preflight failed: feature cache source data hashes do not match")


def _write_experiment_metadata(cfg, path: Path, *, stamp: str | None = None) -> None:
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (path / "run.json").write_text(json.dumps({"experiment_id": path.name, "config_digest": cfg.digest(), "created_at": stamp, "method_registry": ["direct", "raid", "lora", "lora_raid"]}, indent=2) + "\n")
    (path / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str) + "\n")
    lock = Path("requirements-pilot.txt")
    source = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if source.returncode != 0:
        fallback = Path("/Users/ConstiX/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git")
        if fallback.exists():
            source = subprocess.run([str(fallback), "rev-parse", "HEAD"], capture_output=True, text=True)
    (path / "provenance.json").write_text(json.dumps({
        "source_commit": source.stdout.strip() or "unavailable",
        "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else "missing",
        "libero_commit": cfg.libero_commit,
    }, indent=2) + "\n")


def _new_experiment_dir(cfg) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(cfg.output_dir) / f"{cfg.name}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    _write_experiment_metadata(cfg, path, stamp=stamp)
    return path


def _write_collection_manifest(root: Path) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "collection_manifest.json":
            continue
        files.append({"path": str(path.relative_to(root)), "size": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    payload = {"format": 1, "experiment_id": root.name, "files": files}
    tmp = root / "collection_manifest.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(root / "collection_manifest.json")


def _reports_under(root: Path) -> list[dict]:
    report_roots = {root}
    report_roots.update(p.parent for p in root.rglob("summary.json"))
    return [build_report(path) for path in sorted(report_roots) if any(path.glob("*.json"))]


def _reserved_attempt_budget(run_dir: Path) -> float:
    total = 0.0
    for ledger in run_dir.glob("cloud_ledger*.json"):
        try:
            total += float(json.loads(ledger.read_text()).get("quote", {}).get("with_25_percent_margin", 0.0))
        except (OSError, ValueError, TypeError):
            continue
    return total


def _reserved_project_budget(output_dir: Path) -> float:
    """Count every prior reservation under this project before provisioning."""
    total = 0.0
    for ledger in output_dir.rglob("cloud_ledger*.json") if output_dir.exists() else []:
        try:
            payload = json.loads(ledger.read_text())
            total += float(payload.get("quote", {}).get("with_25_percent_margin", 0.0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise RuntimeError(f"cannot verify project budget ledger: {ledger}")
    return total


def _active_project_run(output_dir: Path) -> Path | None:
    for ledger in output_dir.rglob("cloud_ledger*.json") if output_dir.exists() else []:
        try:
            payload = json.loads(ledger.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            raise RuntimeError(f"cannot verify project resource ledger: {ledger}")
        if payload.get("status") in {"provisioning", "created", "running", "creation_uncertain", "termination_uncertain"}:
            return ledger
    return None


def _retryable_failure(exc: Exception) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError, subprocess.TimeoutExpired))


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    active_commands = {"prepare-protocol", "validate-protocol", "cache-features", "train-source",
        "adapt-target", "evaluate-task", "freeze-development", "register-confirmation",
        "report-fewshot", "verify-assets", "smoke-real", "collect-calibration", "calibrate-study"}
    if args.command not in active_commands:
        raise SystemExit("This legacy pilot command is disabled. Use the v3 standalone-MAE recorded-frame study commands.")
    if args.command == "prepare-protocol":
        if not args.dataset_dir:
            raise SystemExit("prepare-protocol requires --dataset-dir")
        from .fewshot import create_goal_protocol
        output = args.manifest or "artifacts/fewshot_libero_goal_v3/protocol.json"
        manifest = create_goal_protocol(args.dataset_dir, output, expected_demos=args.expected_demos)
        print(json.dumps({"protocol": manifest["protocol"], "digest": manifest["digest"],
                          "task_roles": manifest["task_roles"], "manifest": output}, indent=2))
        return 0
    if args.command == "validate-protocol":
        if not args.manifest:
            raise SystemExit("validate-protocol requires --manifest")
        from .fewshot import validate_goal_protocol
        manifest = validate_goal_protocol(args.manifest)
        print(json.dumps({"protocol": manifest["protocol"], "digest": manifest["digest"],
                          "task_roles": manifest["task_roles"], "manifest": args.manifest}, indent=2))
        return 0
    if args.command == "cache-features":
        if not args.dataset_dir or not args.mae_ckpt or not args.feature_dir or not args.libero_repo:
            raise SystemExit("cache-features requires --dataset-dir, --feature-dir, --mae-ckpt, and pinned --libero-repo")
        if not args.manifest:
            raise SystemExit("cache-features requires --manifest")
        subprocess.run([sys.executable, "src/cache_mae_features.py", "--dataset-dir", args.dataset_dir,
                        "--feature-dir", args.feature_dir, "--mae-ckpt", args.mae_ckpt,
                        "--manifest", args.manifest, "--libero-repo", args.libero_repo,
                        "--device", args.device, "--batch-size", str(args.batch_size)], check=True)
        return 0
    if args.command in {"verify-assets", "smoke-real", "collect-calibration", "calibrate-study"}:
        from .fewshot_assets import verify_assets, smoke_real, calibrate_study
        manifest = args.manifest or "artifacts/fewshot_libero_goal_v3/protocol.json"
        feature_dir = args.feature_dir or "data/libero_goal/features"
        out_dir = Path(args.output_dir or "artifacts/fewshot_libero_goal_v3")
        cfg, _ = __import__("raid_experiments.fewshot_config", fromlist=["load_fewshot_config"]).load_fewshot_config(args.study_config)
        if args.command == "verify-assets":
            report = verify_assets(manifest, feature_dir, args.mae_ckpt,
                                   libero_repo=args.libero_repo, asset_notes=args.asset_notes, require_cache=False)
            print(json.dumps(report, indent=2)); return 0 if report["ready"] else 2
        if args.command == "smoke-real":
            report = smoke_real(manifest, feature_dir, args.mae_ckpt,
                                libero_repo=args.libero_repo, asset_notes=args.asset_notes,
                                device=args.device, output_dir=out_dir)
            print(json.dumps(report, indent=2)); return 0 if report["status"] == "passed" else 2
        if args.command == "collect-calibration":
            if not args.libero_repo:
                raise SystemExit("collect-calibration requires --libero-repo for the pinned production cache pass")
            from .fewshot_calibration import collect_calibration
            report = collect_calibration(manifest, feature_dir, args.mae_ckpt, out_dir,
                                         device=args.device, libero_repo=args.libero_repo)
            print(json.dumps(report, indent=2)); return 0 if report.get("status") == "complete" else 2
        calibration_cache = out_dir / "production_runs" / "cache_cuda"
        if not (calibration_cache / "manifest.json").is_file():
            calibration_cache = Path(feature_dir)
        report = calibrate_study(out_dir, cfg, device=args.device, feature_dir=calibration_cache,
                                 pricing_file=args.pricing_file, spent_usd=args.spent_usd)
        print(json.dumps(report, indent=2)); return 0 if report["status"] == "ready" else 2
    if args.command in {"train-source", "adapt-target", "evaluate-task", "freeze-development", "register-confirmation", "report-fewshot"}:
        from .fewshot_train import METHODS, evaluate_task, adapt_target, train_source
        from .fewshot_report import freeze_development, register_confirmation, report_fewshot
        from .fewshot_config import load_fewshot_config
        study_cfg, study_config_digest = load_fewshot_config(args.study_config)
        manifest = args.manifest or "artifacts/fewshot_libero_goal_v3/protocol.json"
        feature_dir = args.feature_dir or "data/libero_goal/features"
        output_dir = args.output_dir or "artifacts/fewshot_libero_goal_v3"
        if study_cfg.get("output_dir") and not args.output_dir:
            output_dir = study_cfg["output_dir"]
        if args.command == "train-source":
            print(json.dumps(train_source(manifest, feature_dir, output_dir, seed=args.seed,
                                          updates=args.updates or study_cfg["source_training"]["updates"],
                                          batch_size=study_cfg["source_training"]["batch_size"],
                                          microbatch_size=study_cfg["source_training"]["microbatch_size"],
                                          head_lr=study_cfg["source_training"]["head_lr"],
                                          weight_decay=study_cfg["source_training"]["weight_decay"],
                                          stop_after_updates=args.stop_after_updates,
                                          device=args.device, resume_checkpoint=args.resume_checkpoint,
                                          run_mode=args.run_mode), indent=2, default=str)); return 0
        if args.command == "adapt-target":
            if not args.source_checkpoint or not args.task or not args.method or args.budget is None:
                raise SystemExit("adapt-target requires --source-checkpoint, --task, --method, and --budget")
            print(json.dumps(adapt_target(manifest, feature_dir, args.source_checkpoint, output_dir,
                                          task=args.task, method=args.method, budget=args.budget,
                                          seed=args.seed, updates=args.updates or study_cfg["adaptation"]["max_updates"],
                                          stop_after_updates=args.stop_after_updates,
                                          batch_size=study_cfg["adaptation"]["batch_size"],
                                          microbatch_size=study_cfg["adaptation"]["microbatch_size"],
                                          head_lr=study_cfg["adaptation"]["head_lr"],
                                          lora_lr=study_cfg["adaptation"]["lora_lr"],
                                          weight_decay=study_cfg["adaptation"]["weight_decay"],
                                          grad_clip=study_cfg["adaptation"]["grad_clip"],
                                          lora_rank=study_cfg["adaptation"]["lora_rank"],
                                          lora_alpha=study_cfg["adaptation"]["lora_alpha"],
                                          lora_dropout=study_cfg["adaptation"]["lora_dropout"],
                                          lora_blocks=tuple(study_cfg["adaptation"]["target_blocks"]),
                                          device=args.device, mae_checkpoint=args.mae_ckpt,
                                          selected_step=args.selected_step,
                                          freeze_file=args.freeze_file,
                                          confirmation_file=args.confirmation_file,
                                          resume_checkpoint=args.resume_checkpoint,
                                          run_mode=args.run_mode), indent=2, default=str)); return 0
        if args.command == "evaluate-task":
            if not args.source_checkpoint or not args.task or args.budget is None:
                raise SystemExit("evaluate-task requires --source-checkpoint, --task, and --budget")
            methods = tuple(args.methods.split(",")) if args.methods else tuple(sorted(set(study_cfg["methods"]) | set(study_cfg["lookup_controls"])))
            selected_updates = ({method: step for method, step in
                                 (("head_ft", args.head_step), ("lora_head_ft", args.lora_step))
                                 if step is not None}
                                if args.head_step is not None or args.lora_step is not None else None)
            print(json.dumps(evaluate_task(manifest, feature_dir, args.source_checkpoint, output_dir,
                                           task=args.task, budget=args.budget, seed=args.seed,
                                           device=args.device, methods=methods,
                                           mae_checkpoint=args.mae_ckpt,
                                           lora_rank=study_cfg["adaptation"]["lora_rank"],
                                           lora_alpha=study_cfg["adaptation"]["lora_alpha"],
                                           lora_dropout=study_cfg["adaptation"]["lora_dropout"],
                                           lora_blocks=tuple(study_cfg["adaptation"]["target_blocks"]),
                                           final_lock=args.freeze_file, confirmation_file=args.confirmation_file,
                                           selected_updates=selected_updates,
                                           run_mode=args.run_mode), indent=2, default=str)); return 0
        if args.command == "register-confirmation":
            if not args.freeze_file or not args.source_checkpoints:
                raise SystemExit("register-confirmation requires --freeze-file and --source-checkpoints")
            paths = [x for x in args.source_checkpoints.split(",") if x]
            if len(paths) != 3:
                raise SystemExit("--source-checkpoints requires three paths in seed order 42,43,44")
            registry = {seed: path for seed, path in zip((42, 43, 44), paths)}
            target = args.confirmation_file or str(Path(output_dir) / "confirmation.json")
            print(json.dumps(register_confirmation(manifest, args.freeze_file, registry, target,
                scope=args.confirmation_scope, study_root=args.study_root), indent=2)); return 0
        if args.command == "freeze-development":
            if not args.source_checkpoint:
                raise SystemExit("freeze-development requires --source-checkpoint")
            if args.head_step is not None or args.lora_step is not None or args.selected_step is not None:
                raise SystemExit("freeze-development selects update counts from the complete candidate evidence; omit step overrides")
            freeze_path = args.freeze_file or str(Path(output_dir) / "development_freeze.json")
            registry = None
            if args.source_checkpoints:
                paths = [x for x in args.source_checkpoints.split(",") if x]
                if len(paths) != 3:
                    raise SystemExit("--source-checkpoints requires exactly three paths in seed order 42,43,44")
                registry = {seed: path for seed, path in zip((42,43,44), paths)}
            print(json.dumps(freeze_development(manifest, output_dir, args.source_checkpoint,
                                                None, freeze_path, source_checkpoints=registry), indent=2)); return 0
        report_dir = Path(args.output_dir or output_dir)
        print(json.dumps(report_fewshot(report_dir, manifest,
            confirmation_file=args.confirmation_file, study_root=args.study_root), indent=2, default=str)); return 0
    if args.command == "enumerate-expansion":
        print(json.dumps(enumerate_expansion(args.config), indent=2)); return 0
    cfg = load_config(args.config)
    if args.output_dir:
        cfg = replace(cfg, output_dir=args.output_dir)
    if args.provider or args.budget_usd is not None:
        cloud = cfg.cloud
        if args.provider:
            cloud = replace(cloud, provider=args.provider)
        if args.budget_usd is not None:
            cloud = replace(cloud, budget_usd=args.budget_usd)
        cfg = replace(cfg, cloud=cloud)
    if args.command == "preflight":
        print(json.dumps(preflight(cfg), indent=2, default=str)); return 0
    if args.command == "estimate":
        print(json.dumps(estimate(cfg, hours=args.hours), indent=2)); return 0
    if args.command == "make-split":
        manifest = create_split(resolve_task_files(cfg.data.dataset_dir), cfg.data.split_path, seed=cfg.data.split_seed, validation_per_task=cfg.data.validation_per_task, test_per_task=cfg.data.test_per_task, train_pool_per_task=40, expected_demos_per_task=cfg.data.demonstrations_per_task, horizon=cfg.horizon)
        print(json.dumps({"path": cfg.data.split_path, "digest": manifest["digest"]}, indent=2)); return 0
    if args.command == "status":
        print(json.dumps(status(cfg), indent=2)); return 0
    if args.command == "collect":
        roots = [Path(args.experiment_dir)] if args.experiment_dir else [Path(cfg.output_dir)] + [p for p in Path(cfg.output_dir).glob(f"{cfg.name}_*") if p.is_dir()]
        reports = [report for root in roots for report in _reports_under(root)]
        print(json.dumps({"experiments": reports}, indent=2)); return 0
    if args.command == "cleanup":
        if not args.pod_id: raise SystemExit("cleanup requires --pod-id")
        cleanup_name = cfg.name
        if args.experiment_dir and (Path(args.experiment_dir) / "run.json").exists():
            cleanup_name = json.loads((Path(args.experiment_dir) / "run.json").read_text()).get("experiment_id", cleanup_name)
        if args.delete_volume and not args.experiment_dir:
            raise SystemExit("cleanup with --delete-volume requires --experiment-dir so collection can be verified")
        if args.delete_volume:
            root = Path(args.experiment_dir)
            receipt = root / "collection_receipt.json"
            if not receipt.exists():
                raise SystemExit("refusing volume deletion before collection receipt verification")
        delete_pod(args.pod_id, cleanup_name)
        if args.volume_id and args.delete_volume:
            delete_network_volume(args.volume_id, cleanup_name, receipt_path=Path(args.experiment_dir) / "collection_receipt.json")
        print(f"Deleted experiment pod {args.pod_id}" + (f" and volume {args.volume_id}" if args.volume_id else "")); return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
