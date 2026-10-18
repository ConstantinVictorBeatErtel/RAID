"""Local collection contract for protocol v4.

A bundle is accepted only for the scope it declares. Synthetic acceptance is an
experiment test, not a scientific result. A scientific scope stays closed until
the container image is pinned, generation cost is measured, CUDA evidence is
present and the Cosmos3-Edge exposure audit is resolved.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .wm_protocol import TemporalAlignment


@dataclass(frozen=True)
class CollectionScope:
    name: str
    execution_mode: str
    budgets: tuple[int, ...]
    seeds: tuple[int, ...]
    lora_steps: tuple[int, ...]
    require_retention: bool = True
    require_rollout: bool = True
    rollout_cases_per_task: int | None = None
    development_tasks: tuple[str, ...] | None = None


def synthetic_smoke_scope(protocol: dict, *, budgets: tuple[int, ...] = (2,),
                          steps: tuple[int, ...] = (2, 4), seed: int = 42,
                          rollout_cases_per_task: int = 1) -> CollectionScope:
    return CollectionScope("synthetic_development_smoke", "synthetic", budgets, (seed,), steps,
                           rollout_cases_per_task=rollout_cases_per_task)


def scientific_development_scope(protocol: dict) -> CollectionScope:
    tasks = tuple(name for name, spec in protocol["tasks"].items() if spec["role"] == "development")
    return CollectionScope("development", "scientific", (2, 5, 10), (42,), (200, 500, 1000),
                           rollout_cases_per_task=None, development_tasks=tasks)


def scientific_blockers(protocol: dict, root: Path, asset_notes: dict | None) -> list[str]:
    """Gates that must pass before a bundle can be called a scientific result."""
    blockers = []
    try:
        TemporalAlignment(**protocol["design"]["temporal"]).require_calibrated()
    except (RuntimeError, ValueError):
        blockers.append("time_scale_uncalibrated")
    audit = (asset_notes or {}).get("pretraining_exposure_audit", {})
    if audit.get("status") != "resolved":
        blockers.append("exposure_audit_unresolved")
    runtime = (asset_notes or {}).get("runtime") or {}
    notes_image = str((asset_notes or {}).get("container_image_digest") or runtime.get("container_image_digest") or "")
    if not notes_image.startswith("sha256:") or len(notes_image) < len("sha256:") + 64:
        blockers.append("container_image_unpinned")
    costs_path = root / "costs.json"
    measured = False
    if costs_path.is_file():
        costs = json.loads(costs_path.read_text())
        measured = bool(costs.get("measured")) and float(costs.get("generation_seconds_per_clip", 0)) > 0
    if not measured:
        blockers.append("generation_cost_unmeasured")
    cuda_path = root / "cuda_evidence.json"
    if not cuda_path.is_file() or json.loads(cuda_path.read_text()).get("status") != "passed":
        blockers.append("cuda_evidence_missing")
    return blockers


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _dev_tasks(protocol: dict, scope: CollectionScope) -> tuple[str, ...]:
    if scope.development_tasks is not None:
        return scope.development_tasks
    return tuple(name for name, spec in protocol["tasks"].items() if spec["role"] == "development")


def validate_collection(root: str | Path, protocol: dict, scope: CollectionScope, *,
                        asset_notes: dict | None = None) -> dict:
    root = Path(root)
    errors: list[str] = []
    protocol_path = root / "protocol.json"
    if not protocol_path.is_file():
        errors.append("protocol_json_missing")
    else:
        on_disk = _load(protocol_path)
        if on_disk.get("digest") != protocol["digest"] or on_disk.get("model_digest") != protocol["model_digest"]:
            errors.append("protocol_digest_mismatch")
    uncalibrated = root / "protocol_uncalibrated.json"
    if not uncalibrated.is_file():
        errors.append("uncalibrated_protocol_missing")
    elif _load(uncalibrated).get("model_digest") != protocol.get("model_digest"):
        errors.append("calibration_changed_model_digest")

    for seed in scope.seeds:
        ckpt = root / "source" / f"source_v4_seed{seed}.pt"
        if not ckpt.is_file():
            errors.append(f"source_checkpoint_missing:{seed}")
            continue
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        if payload.get("model_digest") != protocol["model_digest"]:
            errors.append(f"source_checkpoint_protocol_mismatch:{seed}")
        if payload.get("execution_mode") != scope.execution_mode:
            errors.append(f"source_execution_mode:{payload.get('execution_mode')}")
    manifest = root / "cache" / "manifest.json"
    if not manifest.is_file():
        errors.append("frame_cache_missing")
    elif _load(manifest).get("model_digest") != protocol["model_digest"]:
        errors.append("frame_cache_protocol_mismatch")
    elif _load(manifest).get("execution_mode") != scope.execution_mode:
        errors.append("frame_cache_execution_mode")

    reports = root / "reports"
    dev_tasks = _dev_tasks(protocol, scope)
    final_tasks = {name for name, spec in protocol["tasks"].items() if spec["role"] == "final_test"}
    recorded: list[dict] = []
    for task in dev_tasks:
        for budget in scope.budgets:
            for seed in scope.seeds:
                for step in scope.lora_steps:
                    path = reports / f"recorded_{task}_b{budget}_s{seed}_u{step}.json"
                    if not path.is_file():
                        errors.append(f"recorded_missing:{path.name}")
                        continue
                    report = _load(path)
                    recorded.append(report)
                    if report.get("model_digest") != protocol["model_digest"]:
                        errors.append(f"recorded_protocol_mismatch:{path.name}")
                    if report.get("scientific_result") is not False:
                        errors.append(f"scientific_result_asserted:{path.name}")
                    raid = report.get("costs", {}).get("raid_bank_mlp", {})
                    if raid.get("gradient_updates") != 0:
                        errors.append(f"raid_target_gradients:{path.name}")
                    membership = raid.get("bank_membership", {}).get("support_demo_ids")
                    if list(membership or []) != list(report.get("support_demo_ids") or []):
                        errors.append(f"support_mismatch:{path.name}")
                    if set(report["results"]) < {"raid_bank_mlp", "lora_mlp"}:
                        errors.append(f"arm_missing:{path.name}")

    freeze_path = root / "freeze.json"
    selected = None
    if not freeze_path.is_file():
        errors.append("freeze_missing")
    else:
        freeze = _load(freeze_path)
        selected = freeze.get("selected_lora_updates")
        if freeze.get("status") != "frozen" or freeze.get("model_digest") != protocol.get("model_digest"):
            errors.append("freeze_mismatch")
        if selected not in scope.lora_steps:
            errors.append("freeze_step_outside_scope")
        if freeze.get("scientific_result") is not False:
            errors.append("scientific_result_asserted:freeze")

    if scope.require_retention:
        if selected is None:
            errors.append("retention_without_freeze")
        else:
            for task in dev_tasks:
                for budget in scope.budgets:
                    for seed in scope.seeds:
                        path = reports / f"retention_{task}_b{budget}_s{seed}_u{selected}.json"
                        if not path.is_file():
                            errors.append(f"retention_missing:{path.name}")
                            continue
                        report = _load(path)
                        raid = report.get("results", {}).get("raid_bank_mlp", {})
                        lora = report.get("results", {}).get("lora_mlp", {})
                        if raid.get("gradient_updates") != 0 or raid.get("target_adapter_active") is not False:
                            errors.append(f"raid_retention_adapted:{path.name}")
                        if lora.get("target_adapter_active") is not True or lora.get("selected_step") != selected:
                            errors.append(f"lora_retention_inactive:{path.name}")
                        for bank in report.get("raid_banks", []):
                            if bank.get("overlaps_retention") or bank.get("source_train_only") is not True:
                                errors.append(f"retention_bank_invalid:{path.name}")
                        if report.get("scientific_result") is not False:
                            errors.append(f"scientific_result_asserted:{path.name}")

    ledger_path = reports / "query_access.jsonl"
    if not ledger_path.is_file():
        errors.append("query_ledger_missing")
    else:
        lines = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
        opened = {(line.get("task"), tuple(line.get("demos", []))) for line in lines}
        for line in lines:
            if line.get("task") in final_tasks:
                errors.append("final_test_query_opened")
        for report in recorded:
            key = (report.get("task"), tuple(report.get("query_demos", [])))
            if key not in opened:
                errors.append(f"query_ledger_missing_case:{key[0]}")

    if scope.require_rollout:
        try:
            TemporalAlignment(**protocol["design"]["temporal"]).require_calibrated()
        except (RuntimeError, ValueError):
            errors.append("time_scale_uncalibrated")
        evidence_path = root / "calibration" / "temporal_calibration.json"
        if not evidence_path.is_file():
            errors.append("calibration_evidence_missing")
        else:
            evidence = _load(evidence_path)
            if evidence.get("data") != "source_validation_only" or evidence.get("plausible_fraction", 0) < 0.8:
                errors.append("calibration_evidence_invalid")
            held_out = {name for name, spec in protocol["tasks"].items() if spec["role"] != "source"}
            if any(row.get("task") in held_out for row in evidence.get("cases", [])):
                errors.append("calibration_used_held_out_task")
        summary_path = root / "rollouts" / "summary.json"
        if not summary_path.is_file():
            errors.append("rollout_summary_missing")
        else:
            summary = _load(summary_path)
            if summary.get("scientific_result") is not False:
                errors.append("scientific_result_asserted:rollout")
            cases = summary.get("cases", [])
            by_task: dict[str, list] = {}
            for case in cases:
                by_task.setdefault(case.get("task"), []).append(case)
                if case.get("shared_initial_plan") is not True:
                    errors.append(f"rollout_plan_not_shared:{case.get('case_id')}")
                for arm, record in case.get("arms", {}).items():
                    video = record.get("video")
                    if not video or not Path(video).is_file():
                        errors.append(f"rollout_video_missing:{case.get('case_id')}:{arm}")
            for task in dev_tasks:
                queries = protocol["tasks"][task]["query"]
                expected = len(queries) if scope.rollout_cases_per_task is None else scope.rollout_cases_per_task
                if len(by_task.get(task, [])) < expected:
                    errors.append(f"rollout_cases_missing:{task}")
            if scope.execution_mode == "scientific":
                height = protocol["design"]["generator"]["height"]
                for case in cases:
                    for record in case.get("arms", {}).values():
                        spec = (record.get("plans") or [{}])[0].get("generator_height")
                        if spec != height:
                            errors.append("generator_resolution_mismatch")
                            break

    blockers = scientific_blockers(protocol, root, asset_notes)
    if scope.execution_mode == "scientific":
        errors.extend(blockers)
    audit = (asset_notes or {}).get("pretraining_exposure_audit") or {}
    eligible = bool(audit.get("status") == "resolved" and audit.get("held_out_claims_eligible") is True)
    report = {"ok": not errors, "errors": errors, "scientific_blockers": blockers,
              "scientific_result": False, "scope": scope.name, "execution_mode": scope.execution_mode,
              "protocol_digest": protocol.get("digest"), "model_digest": protocol.get("model_digest"),
              "held_out_claims_eligible": eligible}
    if scope.execution_mode == "scientific" and report["ok"] and not blockers:
        report["scientific_result"] = True
    if not eligible:
        report["claims_limitation"] = (
            "held_out_claims_eligible=false: do not assert LIBERO-Goal tasks are unseen by Cosmos3-Edge"
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "collection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def collect_cli(protocol: dict, args) -> int:
    notes_path = Path(args.asset_notes) if args.asset_notes else None
    notes = json.loads(notes_path.read_text()) if notes_path and notes_path.is_file() else {}
    budgets = tuple(int(part) for part in args.budgets.split(",") if part)
    steps = tuple(int(part) for part in args.lora_steps.split(",") if part)
    if args.run_mode == "scientific":
        scope = scientific_development_scope(protocol)
    else:
        scope = synthetic_smoke_scope(protocol, budgets=budgets, steps=steps, seed=args.seed,
                                      rollout_cases_per_task=1)
    report = validate_collection(args.root or args.output, protocol, scope, asset_notes=notes)
    print(json.dumps({key: report[key] for key in ("ok", "errors", "scientific_result", "scientific_blockers", "scope")},
                     indent=2))
    return 0 if report["ok"] else 2
