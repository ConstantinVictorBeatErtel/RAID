#!/usr/bin/env python3
"""Direct A40 pod runner for the RAID few-shot study.

Drives the existing scientific CLI chain. Does not use fewshot-cloud launch,
does not authorize spending, and never relaxes scientific integrity checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


IMAGE_DIGEST = "sha256:2594f25beeef751aeebcc207ba4c45a382ba174655f01e15cbbfad0e2bdb5ef8"
LOCK_RELATIVE = "artifacts/fewshot_libero_goal_v3/readiness_acceptance_20261004/requirements-linux-cu118.lock"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
MAE_SHA256 = "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
SMOKE_MAX_ABS = 2e-4
DEFAULT_HOURLY_USD = 0.59
DEFAULT_PROJECT_CAP_USD = 12.0
DEFAULT_MARGIN = 1.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def study_paths(root: str | Path) -> dict[str, str]:
    """Canonical absolute paths under one study root."""
    base = str(Path(root).resolve())
    output = base + "/collection/runs"
    source = base + "/collection/source"
    return {
        "STUDY_ROOT": base,
        "REPO": base + "/repo",
        "PROTOCOL": base + "/inputs/protocol-cloud.json",
        "DATASET": base + "/inputs/dataset",
        "FEATURES": base + "/inputs/features",
        "MAE": base + "/inputs/files/mae_checkpoint",
        "ASSET_NOTES": base + "/inputs/files/asset_notes",
        "LIBERO": base + "/inputs/libero_repo",
        "OUTPUT": output,
        "SOURCE_ROOT": source,
        "FREEZE": output + "/development_freeze.json",
        "CONFIRMATION": output + "/confirmation.json",
        "VENV": base + "/environments/venv",
        "LOGS": base + "/collection/logs",
        "LEDGER": base + "/run_ledger.json",
        "SCOPE": base + "/scope_decision.json",
        "OUTPUTS_HASH": base + "/outputs_sha256.json",
        "DONE": base + "/DONE",
        "FAILED": base + "/FAILED",
        "SMOKE_DIR": base + "/collection/smoke_cuda",
        "LOCK": base + "/repo/" + LOCK_RELATIVE,
    }


def redact(text: str) -> str:
    """Strip Runpod API tokens from error text written to markers/logs."""
    return re.sub(r"rpa_[A-Za-z0-9]+", "rpa_REDACTED", str(text))


def write_failed_marker(paths: dict[str, str], *, stage: str, error: str) -> None:
    payload = {
        "status": "failed",
        "stage": stage,
        "error": redact(error),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_result": False,
        "paid_execution": "direct_pod_runner",
    }
    Path(paths["FAILED"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rewrite_protocol_paths(protocol: dict, dataset_root: str | Path) -> dict:
    """Rewrite operational HDF5 paths; digest must stay unchanged."""
    from raid_experiments.fewshot import _digest

    dataset_root = Path(dataset_root).resolve()
    remote = json.loads(json.dumps(protocol))
    original_digest = remote.get("digest")
    task_paths = {}
    for task, spec in remote["tasks"].items():
        original = Path(spec["path"])
        matches = [item for item in dataset_root.rglob(original.name)
                   if item.is_file() and sha256_file(item) == spec["source_sha256"]]
        if len(matches) != 1:
            raise ValueError(f"cannot uniquely map pinned HDF5 source for {task}")
        relative = matches[0].relative_to(dataset_root).as_posix()
        spec["path"] = str(dataset_root / relative)
        task_paths[task] = spec["path"]
    if _digest(remote) != original_digest:
        raise ValueError("operational path rewrite changed the frozen protocol digest")
    return remote, task_paths


def rewrite_asset_notes(notes: dict, task_paths: dict[str, str], protocol: dict) -> dict:
    remote = json.loads(json.dumps(notes))
    for task, path_value in task_paths.items():
        entry = remote.get("libero_goal", {}).get("files", {}).get(task)
        if not isinstance(entry, dict) or entry.get("sha256") != protocol["tasks"][task]["source_sha256"]:
            raise ValueError(f"asset exposure notes do not match staged dataset provenance for {task}")
        entry["path"] = path_value
    return remote


def resolve_argv(argv: list[str], mapping: dict[str, str]) -> list[str]:
    resolved = []
    for item in argv:
        text = item
        for key, value in mapping.items():
            text = text.replace("${" + key + "}", value)
        if "${" in text:
            raise ValueError(f"unresolved placeholder in argv item: {text}")
        resolved.append(text)
    return resolved


def inventory_counts(commands: list[dict]) -> dict[str, int]:
    counts = {
        "source": 0,
        "baseline_evaluation": 0,
        "candidate_evaluation": 0,
        "confirmation_evaluation": 0,
        "condition_validation": 0,
        "development_freeze": 0,
        "confirmation_report": 0,
        "head_ft": 0,
        "lora_head_ft": 0,
    }
    for entry in commands:
        op = entry["operation"]
        if op in counts:
            counts[op] += 1
    counts["development_baseline_bundles"] = counts["baseline_evaluation"]
    counts["development_candidates"] = counts["candidate_evaluation"]
    counts["final_conditions"] = counts["confirmation_evaluation"]
    counts["sources"] = counts["source"]
    return counts


def assert_inventory(commands: list[dict], *, scope: str) -> dict[str, int]:
    counts = inventory_counts(commands)
    expected_final = 18 if scope == "all" else 6
    if counts["sources"] != 3:
        raise AssertionError(f"expected 3 sources, got {counts['sources']}")
    if counts["development_baseline_bundles"] != 6:
        raise AssertionError(f"expected 6 baseline bundles, got {counts['development_baseline_bundles']}")
    if counts["development_candidates"] != 36:
        raise AssertionError(f"expected 36 candidates, got {counts['development_candidates']}")
    if counts["final_conditions"] != expected_final:
        raise AssertionError(f"expected {expected_final} final conditions, got {counts['final_conditions']}")
    if counts["development_freeze"] != 1:
        raise AssertionError("expected one development freeze")
    if counts["confirmation_report"] != 1:
        raise AssertionError("expected one confirmation report")
    return counts


def choose_scope(first_timings: list[dict], *, hourly_usd: float,
                 spent_seconds: float | None = None,
                 project_cap_usd: float = DEFAULT_PROJECT_CAP_USD,
                 margin: float = DEFAULT_MARGIN) -> dict:
    """Preregistered affordability rule: all, else five_shot, else stop."""
    if hourly_usd <= 0 or project_cap_usd <= 0 or margin < 1:
        raise ValueError("hourly rate, project cap, and margin must be positive")
    by_op: dict[str, float] = {}
    for row in first_timings:
        op = row["operation"]
        seconds = float(row["seconds"])
        if seconds < 0 or not seconds == seconds:
            raise ValueError(f"invalid timing seconds for {op}")
        by_op[op] = by_op.get(op, 0.0) + seconds
    measured = sum(by_op.values()) if spent_seconds is None else float(spent_seconds)
    source = by_op.get("source", 0.0)
    adapt = by_op.get("head_ft", 0.0) + by_op.get("lora_head_ft", 0.0)
    baseline = by_op.get("baseline_evaluation", 0.0)
    candidate = by_op.get("candidate_evaluation", 0.0)
    validation = by_op.get("condition_validation", 0.0)
    if source <= 0 or adapt <= 0 or baseline <= 0:
        raise ValueError("first_condition timings lack source/adapt/baseline measurements")
    rem_dev = 5 * (adapt + baseline + candidate + validation) + max(validation, 60.0)
    sources = 2 * source
    conf_eval = baseline + 2.0 * (candidate / 6.0 if candidate > 0 else baseline / 6.0)
    conf_cond = adapt + conf_eval
    report = max(validation, 60.0)

    def project(scope: str) -> dict:
        n_conf = 18 if scope == "all" else 6
        remaining = rem_dev + sources + n_conf * conf_cond + report
        total_seconds = measured + remaining
        cost = (total_seconds / 3600.0) * hourly_usd * margin
        return {
            "scope": scope,
            "measured_seconds": measured,
            "remaining_seconds": remaining,
            "total_seconds": total_seconds,
            "projected_cost_usd": cost,
            "fits": cost <= project_cap_usd,
        }

    all_proj = project("all")
    five_proj = project("five_shot")
    if all_proj["fits"]:
        decision = "all"
    elif five_proj["fits"]:
        decision = "five_shot"
    else:
        decision = "stop"
    return {
        "status": "ready" if decision != "stop" else "blocked",
        "decision": decision,
        "project_cap_usd": project_cap_usd,
        "hourly_usd": hourly_usd,
        "margin": margin,
        "projections": {"all": all_proj, "five_shot": five_proj},
        "rule": "all budgets if projected cost fits cap; else five_shot; else stop",
    }


def command_key(entry: dict) -> str:
    payload = {"operation": entry["operation"], "identity": entry.get("identity", {}),
               "argv": entry["argv"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_ledger(path: Path) -> dict:
    if not path.is_file():
        return {"format": 1, "commands": {}, "scientific_result": False}
    return json.loads(path.read_text())


def save_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def build_full_command_list(scope: str = "all") -> list[dict]:
    from raid_experiments.fewshot_execution import scientific_sequence

    commands = []
    for phase in ("first_condition", "remaining_development", "confirmation_sources"):
        for entry in scientific_sequence(phase)["commands"]:
            commands.append({**entry, "phase": phase})
    registration = {
        "confirmation_scope": {
            "support_budgets": [2, 5, 10] if scope == "all" else [5],
            "seeds": [42, 43, 44],
            "paired": True,
        },
        "selected_updates": {"head_ft": 1000, "lora_head_ft": 1000},
    }
    commands.append({
        "argv": ["python", "-m", "raid_experiments", "register-confirmation",
                 "--manifest", "${PROTOCOL}", "--freeze-file", "${FREEZE}",
                 "--source-checkpoints",
                 "${SOURCE_ROOT}/source_seed42.pt,${SOURCE_ROOT}/source_seed43.pt,${SOURCE_ROOT}/source_seed44.pt",
                 "--confirmation-file", "${CONFIRMATION}",
                 "--confirmation-scope", scope, "--study-root", "${SOURCE_ROOT}",
                 "--output-dir", "${OUTPUT}"],
        "operation": "register_confirmation",
        "identity": {"scope": scope},
        "phase": "register_confirmation",
    })
    for entry in scientific_sequence("confirmation", registration=registration)["commands"]:
        commands.append({**entry, "phase": "confirmation"})
    return commands


def dry_run(root: str | Path, *, scope: str = "all") -> dict:
    paths = study_paths(root)
    commands = build_full_command_list(scope=scope)
    resolved = []
    for entry in commands:
        argv = resolve_argv(entry["argv"], paths)
        resolved.append({"phase": entry["phase"], "operation": entry["operation"],
                         "identity": entry.get("identity", {}), "argv": argv})
    counts = assert_inventory(commands, scope=scope)
    return {"status": "dry_run", "scope": scope, "root": paths["STUDY_ROOT"],
            "command_count": len(resolved), "inventory": counts, "commands": resolved,
            "scientific_result": False}


def _python(paths: dict[str, str]) -> str:
    candidate = Path(paths["VENV"]) / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def bootstrap_venv(paths: dict[str, str], *, image_digest: str = IMAGE_DIGEST) -> None:
    lock = Path(paths["LOCK"])
    if not lock.is_file():
        raise FileNotFoundError(f"pinned CUDA lock missing: {lock}")
    venv = Path(paths["VENV"])
    if not (venv / "bin" / "python").is_file():
        subprocess.run(["python3.10", "-m", "venv", str(venv)], check=True)
    python = str(venv / "bin" / "python")
    subprocess.run([python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([
        python, "-m", "pip", "install", "--require-hashes",
        "--extra-index-url", "https://download.pytorch.org/whl/cu118",
        "-r", str(lock),
    ], check=True)
    subprocess.run([python, "-m", "pip", "check"], check=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["RAID_CONTAINER_IMAGE_DIGEST"] = image_digest
    Path(paths["LOGS"]).mkdir(parents=True, exist_ok=True)


def prepare_protocol_and_notes(paths: dict[str, str], *, source_protocol: Path,
                               source_notes: Path) -> None:
    protocol = json.loads(source_protocol.read_text())
    remote_protocol, task_paths = rewrite_protocol_paths(protocol, paths["DATASET"])
    Path(paths["PROTOCOL"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["PROTOCOL"]).write_text(json.dumps(remote_protocol, indent=2, sort_keys=True) + "\n")
    notes = json.loads(source_notes.read_text())
    if notes.get("mae", {}).get("sha256") != MAE_SHA256:
        raise ValueError("MAE asset notes hash does not match the pinned checkpoint")
    remote_notes = rewrite_asset_notes(notes, task_paths, remote_protocol)
    Path(paths["ASSET_NOTES"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["ASSET_NOTES"]).write_text(json.dumps(remote_notes, indent=2, sort_keys=True) + "\n")
    mae = Path(paths["MAE"])
    if not mae.is_file() or sha256_file(mae) != MAE_SHA256:
        raise ValueError("MAE checkpoint missing or hash mismatch")


def run_logged_command(paths: dict[str, str], entry: dict, *,
                       runner: Callable | None = None,
                       cwd: str | None = None) -> dict:
    ledger_path = Path(paths["LEDGER"])
    ledger = load_ledger(ledger_path)
    key = command_key(entry)
    prior = ledger["commands"].get(key)
    if prior and prior.get("exit_code") == 0:
        return {**prior, "skipped": True}
    argv = resolve_argv(entry["argv"], paths)
    if argv and argv[0] == "python":
        argv = [_python(paths)] + argv[1:]
    log_dir = Path(paths["LOGS"])
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time())}_{entry['operation']}_{key[:12]}"
    stdout_path = log_dir / f"{stamp}.stdout.log"
    stderr_path = log_dir / f"{stamp}.stderr.log"
    started = time.time()
    if runner is not None:
        completed = runner(argv, cwd=cwd or paths["REPO"])
        exit_code = int(completed.get("exit_code", 1))
        stdout_path.write_text(completed.get("stdout", ""))
        stderr_path.write_text(completed.get("stderr", ""))
    else:
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            proc = subprocess.run(argv, cwd=cwd or paths["REPO"], stdout=out, stderr=err)
        exit_code = proc.returncode
    seconds = time.time() - started
    record = {
        "key": key,
        "phase": entry.get("phase"),
        "operation": entry["operation"],
        "identity": entry.get("identity", {}),
        "argv": argv,
        "exit_code": exit_code,
        "seconds": seconds,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "skipped": False,
    }
    ledger["commands"][key] = record
    save_ledger(ledger_path, ledger)
    if exit_code != 0:
        raise RuntimeError(f"command failed ({entry['operation']}): exit {exit_code}; see {stderr_path}")
    return record


def verify_and_calibrate(paths: dict[str, str], *, runner: Callable | None = None) -> list[dict]:
    records = []
    verify = {
        "phase": "bootstrap",
        "operation": "verify_assets",
        "identity": {},
        "argv": ["python", "-m", "raid_experiments", "verify-assets",
                 "--manifest", "${PROTOCOL}", "--feature-dir", "${FEATURES}",
                 "--mae-ckpt", "${MAE}", "--libero-repo", "${LIBERO}",
                 "--asset-notes", "${ASSET_NOTES}"],
    }
    records.append(run_logged_command(paths, verify, runner=runner))
    cache = {
        "phase": "bootstrap",
        "operation": "cache_features",
        "identity": {},
        "argv": ["python", "-m", "raid_experiments", "cache-features",
                 "--dataset-dir", "${DATASET}", "--feature-dir", "${FEATURES}",
                 "--mae-ckpt", "${MAE}", "--manifest", "${PROTOCOL}",
                 "--libero-repo", "${LIBERO}", "--device", "cuda"],
    }
    records.append(run_logged_command(paths, cache, runner=runner))
    smoke = {
        "phase": "bootstrap",
        "operation": "smoke_real",
        "identity": {},
        "argv": ["python", "-m", "raid_experiments", "smoke-real",
                 "--manifest", "${PROTOCOL}", "--feature-dir", "${FEATURES}",
                 "--mae-ckpt", "${MAE}", "--libero-repo", "${LIBERO}",
                 "--asset-notes", "${ASSET_NOTES}", "--device", "cuda",
                 "--output-dir", "${SMOKE_DIR}"],
    }
    records.append(run_logged_command(paths, smoke, runner=runner))
    smoke_path = Path(paths["SMOKE_DIR"]) / "smoke_real.json"
    if not smoke_path.is_file():
        raise RuntimeError("CUDA smoke result missing")
    report = json.loads(smoke_path.read_text())
    if report.get("status") != "passed":
        raise RuntimeError("CUDA smoke did not pass")
    for key in ("raw_cache_max_abs_error", "zero_adapter_max_abs_error"):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value > SMOKE_MAX_ABS:
            raise RuntimeError(f"CUDA smoke {key}={value} exceeds {SMOKE_MAX_ABS}")
    if report.get("frozen_backbone_unchanged") is not True:
        raise RuntimeError("CUDA smoke frozen backbone check failed")
    if report.get("adapter_reload_predictions_match") is not True:
        raise RuntimeError("CUDA smoke reload prediction check failed")
    return records


def first_condition_timings_from_ledger(paths: dict[str, str]) -> list[dict]:
    """Restart-safe timings: every successful first_condition command in the ledger."""
    ledger = load_ledger(Path(paths["LEDGER"]))
    rows = []
    for record in ledger.get("commands", {}).values():
        if record.get("phase") != "first_condition" or record.get("exit_code") != 0:
            continue
        rows.append({"operation": record["operation"], "seconds": float(record["seconds"])})
    return rows


def load_or_choose_scope(paths: dict[str, str], *, hourly_usd: float,
                         project_cap_usd: float, started_at: float | None) -> dict:
    existing = Path(paths["SCOPE"])
    if existing.is_file():
        prior = json.loads(existing.read_text())
        if prior.get("decision") in {"all", "five_shot"}:
            return prior
    timings = first_condition_timings_from_ledger(paths)
    spent = None if started_at is None else max(0.0, time.time() - float(started_at))
    report = choose_scope(timings, hourly_usd=hourly_usd, spent_seconds=spent,
                          project_cap_usd=project_cap_usd)
    existing.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_phase_commands(paths: dict[str, str], phase: str, *,
                       registration: dict | None = None,
                       runner: Callable | None = None) -> list[dict]:
    from raid_experiments.fewshot_execution import scientific_sequence

    payload = scientific_sequence(phase, registration=registration)
    records = []
    for entry in payload["commands"]:
        records.append(run_logged_command(paths, {**entry, "phase": phase}, runner=runner))
    return records


def hash_outputs(paths: dict[str, str]) -> dict:
    root = Path(paths["STUDY_ROOT"]) / "collection"
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith("logs/") or rel.startswith("smoke_cuda/") or rel == "runner.log":
            continue
        files[rel] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    payload = {"format": 1, "root": str(root), "files": files,
               "file_count": len(files), "scientific_result": False}
    Path(paths["OUTPUTS_HASH"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def run_study(root: str | Path, *, hourly_usd: float = DEFAULT_HOURLY_USD,
              project_cap_usd: float = DEFAULT_PROJECT_CAP_USD,
              image_digest: str = IMAGE_DIGEST,
              source_protocol: str | Path | None = None,
              source_notes: str | Path | None = None,
              skip_bootstrap: bool = False,
              started_at: float | None = None,
              runner: Callable | None = None) -> dict:
    paths = study_paths(root)
    for key in ("OUTPUT", "SOURCE_ROOT", "LOGS", "FEATURES", "DATASET", "SMOKE_DIR"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    if Path(paths["DONE"]).is_file():
        return {"status": "already_done", "root": paths["STUDY_ROOT"], "scientific_result": False}

    stage = "bootstrap"
    try:
        if not skip_bootstrap and runner is None:
            bootstrap_venv(paths, image_digest=image_digest)
        if source_protocol and source_notes:
            prepare_protocol_and_notes(paths, source_protocol=Path(source_protocol),
                                       source_notes=Path(source_notes))
        elif not Path(paths["PROTOCOL"]).is_file():
            raise FileNotFoundError(
                "protocol-cloud.json missing; pass --source-protocol/--source-notes or stage inputs")

        stage = "calibration"
        verify_and_calibrate(paths, runner=runner)
        stage = "first_condition"
        run_phase_commands(paths, "first_condition", runner=runner)
        stage = "cost_gate"
        scope_report = load_or_choose_scope(
            paths, hourly_usd=hourly_usd, project_cap_usd=project_cap_usd, started_at=started_at)
        if scope_report["decision"] == "stop":
            write_failed_marker(paths, stage="cost_gate",
                                error="projected cost exceeds project cap; study stopped")
            return {"status": "blocked_by_cost_gate", "scope": scope_report,
                    "root": paths["STUDY_ROOT"], "scientific_result": False}

        stage = "remaining_development"
        run_phase_commands(paths, "remaining_development", runner=runner)
        stage = "confirmation_sources"
        run_phase_commands(paths, "confirmation_sources", runner=runner)
        scope = scope_report["decision"]
        stage = "register_confirmation"
        register = {
            "phase": "register_confirmation",
            "operation": "register_confirmation",
            "identity": {"scope": scope},
            "argv": ["python", "-m", "raid_experiments", "register-confirmation",
                     "--manifest", "${PROTOCOL}", "--freeze-file", "${FREEZE}",
                     "--source-checkpoints",
                     "${SOURCE_ROOT}/source_seed42.pt,${SOURCE_ROOT}/source_seed43.pt,"
                     "${SOURCE_ROOT}/source_seed44.pt",
                     "--confirmation-file", "${CONFIRMATION}",
                     "--confirmation-scope", scope, "--study-root", "${SOURCE_ROOT}",
                     "--output-dir", "${OUTPUT}"],
        }
        run_logged_command(paths, register, runner=runner)
        registration = json.loads(Path(paths["CONFIRMATION"]).read_text())
        stage = "confirmation"
        run_phase_commands(paths, "confirmation", registration=registration, runner=runner)
        outputs = hash_outputs(paths)
        Path(paths["DONE"]).write_text(json.dumps({
            "status": "complete", "scope": scope,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "outputs_sha256": paths["OUTPUTS_HASH"], "file_count": outputs["file_count"],
            "scientific_result": False, "paid_execution": "direct_pod_runner",
        }, indent=2, sort_keys=True) + "\n")
        return {"status": "complete", "scope": scope, "scope_report": scope_report,
                "outputs": outputs, "root": paths["STUDY_ROOT"], "scientific_result": False}
    except Exception as exc:
        write_failed_marker(paths, stage=stage, error=f"{type(exc).__name__}: {exc}")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Study root, e.g. /workspace/raid-study/<id>")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scope", choices=["all", "five_shot"], default="all",
                        help="Dry-run inventory scope (live runs choose scope after first_condition)")
    parser.add_argument("--hourly-usd", type=float, default=DEFAULT_HOURLY_USD)
    parser.add_argument("--project-cap-usd", type=float, default=DEFAULT_PROJECT_CAP_USD)
    parser.add_argument("--image-digest", default=IMAGE_DIGEST)
    parser.add_argument("--source-protocol", help="Local protocol_v3.json to rewrite into the study root")
    parser.add_argument("--source-notes", help="Local mae_asset_notes.json to rewrite into the study root")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--started-at", type=float, default=None,
                        help="Unix epoch when the pod became billable; used by the cost gate")
    args = parser.parse_args(argv)
    if args.dry_run:
        report = dry_run(args.root, scope=args.scope)
        print(json.dumps(report, indent=2))
        return 0
    try:
        report = run_study(
            args.root, hourly_usd=args.hourly_usd, project_cap_usd=args.project_cap_usd,
            image_digest=args.image_digest, source_protocol=args.source_protocol,
            source_notes=args.source_notes, skip_bootstrap=args.skip_bootstrap,
            started_at=args.started_at)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": redact(f"{type(exc).__name__}: {exc}"),
                          "scientific_result": False}, indent=2))
        return 2
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("status") in {"complete", "already_done"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
