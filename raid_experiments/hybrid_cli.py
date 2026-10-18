"""Protocol v6 command line: ``python -m raid_experiments.hybrid_cli <command>``.

Commands run locally or on an already-provisioned machine. None of them rents
compute. Final-test tasks are refused without a development freeze record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from . import hybrid_collect, hybrid_feasibility
from .hybrid_protocol import (PRIMARY_BUDGET, PROTOCOL_V6, build_hybrid_protocol,
                              validate_hybrid_protocol, with_verified_state_parity,
                              with_verified_timing)


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _write(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mae_factory(checkpoint: str, device: str):
    from src.mae_encoder import MAEImageEncoder
    return lambda: MAEImageEncoder.from_checkpoint(checkpoint, device)


def require_stage_access(protocol: dict, task: str, freeze_file: str | None) -> dict | None:
    role = protocol["tasks"][task]["role"]
    if role == "source":
        raise SystemExit("held-out evaluation commands take development or final-test tasks")
    if role == "development":
        return None
    if not freeze_file:
        raise SystemExit("final-test access requires --freeze-file")
    freeze = _load(freeze_file)
    if freeze.get("status") != "frozen" or freeze.get("model_digest") != protocol.get("model_digest"):
        raise SystemExit("freeze record does not belong to this protocol")
    return freeze


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m raid_experiments.hybrid_cli",
                                description="Protocol v6 (raid_hybrid_generalization_v6) stages.")
    p.add_argument("--protocol", default="artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-protocol", help="Build digest-bound protocol_v6.json from the v3 manifest")
    b.add_argument("--v3-manifest", default="data/assets/libero/protocol_v3.json")
    b.add_argument("--v4-protocol", default="artifacts/raid_wm_libero_goal_v4/protocol_v4.json")
    b.add_argument("--output", default="artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json")

    sub.add_parser("feasibility", help="Print unmeasured cost forecast for pilot and final options")

    sp = sub.add_parser("source-panel-plan", help="Print stage-1 source-panel pod plan (no launch)")
    sp.add_argument("--study-root", default="/workspace/raid-wm/source_panel")

    spanel = sub.add_parser("source-panel", help="Run the 30-case source generator panel (needs CUDA + model)")
    spanel.add_argument("--mae", required=True)
    spanel.add_argument("--output", default="collection")
    spanel.add_argument("--device", default="cuda")
    spanel.add_argument("--hf-cache", default=None)
    spanel.add_argument("--load-model", action="store_true",
                        help="Load the pinned Cosmos3-Edge weights (downloads if missing)")
    spanel.add_argument("--expert-replays", default=None,
                        help="Optional expert_replays.json to merge into panel_summary")

    rp = sub.add_parser("resolve-paths", help="Bind staged HDF5 paths by content hash")
    rp.add_argument("--data-root", required=True)
    rp.add_argument("--output", required=True)
    rp.add_argument("--stage", choices=["all", "source-panel", "pilot", "final"], default="all")
    rp.add_argument("--freeze-file", default=None, help="frozen amendment; required for --stage final")

    ir = sub.add_parser("ingest-rubric", help="Ingest named human ratings from annotation_sheet.csv")
    ir.add_argument("--panel-dir", required=True)
    ir.add_argument("--sheet", default=None)

    st = sub.add_parser("stamp-timing", help="Stamp verified timing after panel + rubric gates pass")
    st.add_argument("--panel-dir", required=True)
    st.add_argument("--output", required=True, help="Where to write the stamped protocol JSON")

    ti = sub.add_parser("validate-timing-interface", help="Write source-only simulator frame/action evidence")
    ti.add_argument("--panel-dir", required=True)
    ti.add_argument("--task", required=True)
    ti.add_argument("--demo", type=int, required=True)
    ti.add_argument("--libero-root", default="data/assets/libero/upstream")

    cp = sub.add_parser("check-state-parity", help="Compare HDF5 proprio with live simulator after set_init_state")
    cp.add_argument("--libero-root", default="data/assets/libero/upstream")
    cp.add_argument("--task", default="put_the_cream_cheese_in_the_bowl")
    cp.add_argument("--demo", type=int, default=0)
    cp.add_argument("--steps", type=int, default=5)
    cp.add_argument("--output", default="artifacts/raid_hybrid_libero_goal_v6/local_checks/state_parity.json")

    rp = sub.add_parser("replay-experts", help="Replay recorded actions from initial states; official success")
    rp.add_argument("--libero-root", default="data/assets/libero/upstream")
    rp.add_argument("--cases", type=int, default=30)
    rp.add_argument("--output", default="artifacts/raid_hybrid_libero_goal_v6/local_checks/expert_replays.json")

    c = sub.add_parser("cache", help="Build v6 reference frame cache (requires encoder)")
    c.add_argument("--mae", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--device", default="cpu")
    c.add_argument("--run-mode", choices=["scientific", "synthetic"], default="scientific")
    c.add_argument("--tasks", nargs="+", default=None,
                   help="Explicit stage task allowlist; required for scientific caching")

    t = sub.add_parser("train-source", help="Dual-mode source training of the common controller")
    t.add_argument("--cache", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="cpu")
    t.add_argument("--updates", type=int, default=None)
    t.add_argument("--run-mode", choices=["scientific", "synthetic"], default="scientific")

    a = sub.add_parser("adapt", help="Adapt LoRA+head for lora_mlp or raid_lora_mlp")
    a.add_argument("--source", required=True)
    a.add_argument("--cache", required=True)
    a.add_argument("--mae", required=True)
    a.add_argument("--condition", choices=["lora_mlp", "raid_lora_mlp"], required=True)
    a.add_argument("--task", required=True)
    a.add_argument("--budget", type=int, default=PRIMARY_BUDGET)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--out", required=True)
    a.add_argument("--device", default="cpu")
    a.add_argument("--updates", type=int, default=None)
    a.add_argument("--run-mode", choices=["scientific", "synthetic"], default="scientific")
    a.add_argument("--freeze-file", default=None)

    e = sub.add_parser("evaluate-recorded", help="Recorded-transition diagnostic for all conditions")
    e.add_argument("--source", required=True)
    e.add_argument("--cache", required=True)
    e.add_argument("--mae", required=True)
    e.add_argument("--task", required=True)
    e.add_argument("--budget", type=int, default=PRIMARY_BUDGET)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--out", required=True)
    e.add_argument("--lora", default=None)
    e.add_argument("--raid-lora", default=None)
    e.add_argument("--device", default="cpu")
    e.add_argument("--freeze-file", default=None)

    r = sub.add_parser("retention", help="Source retention with target configuration active")
    r.add_argument("--source", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--mae", required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--budget", type=int, default=PRIMARY_BUDGET)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out", required=True)
    r.add_argument("--lora", default=None)
    r.add_argument("--raid-lora", default=None)
    r.add_argument("--device", default="cpu")

    az = sub.add_parser("analyze", help="Compute hybrid contrasts from matched episode JSON files")
    az.add_argument("--episodes-dir", required=True)

    ro = sub.add_parser("rollout", help="Closed-loop four-condition rollout for selected pilot starts")
    ro.add_argument("--source", required=True)
    ro.add_argument("--cache", required=True)
    ro.add_argument("--mae", required=True)
    ro.add_argument("--task", required=True)
    ro.add_argument("--budget", type=int, default=PRIMARY_BUDGET)
    ro.add_argument("--seed", type=int, default=42)
    ro.add_argument("--out", required=True)
    ro.add_argument("--libero-root", default="data/assets/libero/upstream")
    ro.add_argument("--lora", default=None)
    ro.add_argument("--raid-lora", default=None)
    ro.add_argument("--device", default="cpu")
    ro.add_argument("--run-mode", choices=["scientific", "synthetic"], default="scientific")
    ro.add_argument("--freeze-file", default=None)
    ro.add_argument("--panel-dir", default=None,
                    help="Required for scientific mode: source_panel with passed gates")
    ro.add_argument("--max-steps", type=int, default=300)

    pilot = sub.add_parser("pilot", help="Run the gated resumable 40-episode development pilot")
    pilot.add_argument("--mae", required=True)
    pilot.add_argument("--libero-root", default="data/assets/libero/upstream")
    pilot.add_argument("--panel-dir", required=True,
                       help="Source panel with technical, replay, named-rating, and timing gates passed")
    pilot.add_argument("--out", required=True)
    pilot.add_argument("--device", default="cuda")

    f = sub.add_parser("freeze", help="Freeze pilot adaptation defaults after development diagnostics")
    f.add_argument("--report-dir", required=True)
    f.add_argument("--output", required=True)

    col = sub.add_parser("collect", help="Validate a collection bundle against v6 gates")
    col.add_argument("--root", required=True)
    col.add_argument("--run-mode", choices=["scientific", "synthetic"], default="synthetic")
    col.add_argument("--asset-notes", default="configs/cosmos3_edge_asset_notes.json")

    args = p.parse_args(argv)

    if args.command == "build-protocol":
        v3 = _load(args.v3_manifest)
        v4_digest = _load(args.v4_protocol)["digest"] if Path(args.v4_protocol).is_file() else None
        protocol = build_hybrid_protocol(v3, v4_digest=v4_digest)
        validate_hybrid_protocol(protocol)
        _write(args.output, protocol)
        hybrid_collect.write_rubric_template(Path(args.output).parent)
        print(json.dumps({"digest": protocol["digest"], "model_digest": protocol["model_digest"],
                          "output": args.output}, indent=2))
        return 0

    if args.command == "feasibility":
        print(json.dumps(hybrid_feasibility.feasibility(), indent=2))
        return 0

    if args.command == "source-panel-plan":
        from .wm_pod import source_panel_plan
        print(json.dumps(source_panel_plan(study_root=args.study_root), indent=2))
        return 0

    if args.command == "resolve-paths":
        from .hybrid_protocol import bind_task_paths, find_task_paths
        protocol = validate_hybrid_protocol(_load(args.protocol))
        roles = {"source-panel": {"source"}, "pilot": {"source", "development"}, "all": None,
                 "final": {"source", "final_test"}}[args.stage]
        if args.stage == "final":  # controlled stage-3 access; refuses without a frozen amendment
            from .hybrid_study import final_access
            try:
                final_access(protocol, args.freeze_file)
            except (ValueError, OSError) as error:
                raise SystemExit(f"final-stage access refused: {error}")
        selected = [name for name, row in protocol["tasks"].items()
                    if roles is None or row["role"] in roles]
        paths = find_task_paths(protocol, args.data_root, task_names=selected)
        rebound = bind_task_paths(protocol, paths, task_names=selected)
        validate_hybrid_protocol(rebound)
        _write(args.output, rebound)
        print(json.dumps({"status": "resolved", "protocol_digest": rebound["digest"],
                          "model_digest": rebound["model_digest"], "tasks": len(paths),
                          "output": args.output}, indent=2))
        return 0

    protocol = validate_hybrid_protocol(_load(args.protocol))

    if args.command == "pilot":
        from .hybrid_runner import run_development_pilot
        report = run_development_pilot(protocol, mae_path=args.mae,
                                       libero_root=args.libero_root, panel_dir=args.panel_dir,
                                       output=args.out, device=args.device)
        print(json.dumps(report, indent=2, default=str))
        return 0

    if args.command == "check-state-parity":
        from .hybrid_local_checks import check_state_parity
        report = check_state_parity(protocol, args.task, args.demo, args.libero_root, steps=args.steps)
        _write(args.output, report)
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("parity_ok") else 1

    if args.command == "replay-experts":
        from .hybrid_local_checks import replay_source_panel
        report = replay_source_panel(protocol, args.libero_root, cases=args.cases,
                                     progress_path=str(args.output) + ".progress.json")
        _write(args.output, report)
        print(json.dumps({k: report[k] for k in report if k != "cases"}, indent=2, default=str))
        return 0 if report.get("gate_pass") else 1

    if args.command == "source-panel":
        from .hybrid_protocol import is_vpp2
        if is_vpp2(protocol):
            raise SystemExit("VPP2 protocols use raid_experiments.hybrid_vpp2_panel for the panel and timing calibration")
        from . import hybrid_panel
        from .hybrid_protocol import GeneratorSpec, TemporalAlignment
        from .world_model import FrozenWorldModel
        import time as _time
        if not args.load_model:
            raise SystemExit("source-panel requires --load-model (or inject a fake backend in tests)")
        from .world_model import Cosmos3EdgeBackend
        # Prefer wm_protocol GeneratorSpec for FrozenWorldModel compatibility
        from .wm_protocol import GeneratorSpec as WMGeneratorSpec, TemporalAlignment as WMTemporalAlignment
        spec = WMGeneratorSpec()
        temporal = WMTemporalAlignment(
            control_hz=protocol["design"]["temporal"]["control_hz"],
            generation_fps=protocol["design"]["temporal"]["generation_fps"],
            num_frames=protocol["design"]["temporal"]["num_frames"],
            chunk_steps=protocol["design"]["temporal"]["chunk_steps"],
            chunks_per_plan=protocol["design"]["temporal"]["chunks_per_plan"],
        )
        t0 = _time.perf_counter()
        backend = Cosmos3EdgeBackend(spec, device=args.device, cache_dir=args.hf_cache)
        backend.load()
        load_seconds = _time.perf_counter() - t0
        world = FrozenWorldModel(backend, Path(args.output) / "generation_cache",
                                 spec=spec, temporal=temporal)
        encoder = _mae_factory(args.mae, args.device)()
        summary = hybrid_panel.run_source_panel(protocol, world, encoder, args.output,
                                                load_seconds=load_seconds)
        if args.expert_replays:
            hybrid_panel.attach_expert_replays(Path(args.output) / "source_panel",
                                               _load(args.expert_replays))
        print(json.dumps({k: summary[k] for k in summary if k != "cases"}, indent=2))
        return 0 if summary.get("technical_decodes_ok") else 2

    if args.command == "ingest-rubric":
        from . import hybrid_panel
        summary = hybrid_panel.ingest_rubric(args.panel_dir, args.sheet)
        print(json.dumps({"rubric_passes": summary["rubric_passes"],
                          "annotators": summary["annotators"],
                          "rubric_gate_pass": summary["rubric_gate_pass"]}, indent=2))
        return 0

    if args.command == "validate-timing-interface":
        from .hybrid_protocol import is_vpp2
        if is_vpp2(protocol):
            raise SystemExit("VPP2 protocols use raid_experiments.hybrid_vpp2_panel for the panel and timing calibration")
        from .hybrid_local_checks import check_frame_action_timing
        task_spec = protocol["tasks"].get(args.task)
        if not task_spec or task_spec["role"] != "source":
            raise SystemExit("timing-interface evidence must use a registered source task")
        if int(args.demo) not in set(map(int, task_spec["source"]["source_validation"])):
            raise SystemExit("timing-interface demo must be source-validation data")
        evidence = check_frame_action_timing(protocol, args.task, args.demo, args.libero_root)
        evidence.update({"protocol_digest": protocol["digest"], "demo": int(args.demo),
                         "source_sha256": task_spec["source_sha256"],
                         "requested_future_frame": int(round(
                             protocol["design"]["temporal"]["chunk_steps"]
                             / protocol["design"]["temporal"]["control_hz"]
                             * protocol["design"]["temporal"]["generation_fps"]
                             / protocol["design"]["temporal"]["time_scale"]))})
        target = Path(args.panel_dir) / "timing_interface.json"
        _write(target, evidence)
        print(json.dumps({"status": "passed" if evidence["rows"] and all(
            r["action_changed_state"] for r in evidence["rows"]) else "blocked",
                          "output": str(target), "rows": len(evidence["rows"])}, indent=2))
        return 0 if evidence["rows"] and all(r["action_changed_state"] for r in evidence["rows"]) else 2

    if args.command == "stamp-timing":
        from . import hybrid_panel
        result = hybrid_panel.stamp_timing(protocol, args.panel_dir)
        if result["status"] != "stamped":
            print(json.dumps(result, indent=2))
            return 2
        _write(args.output, result["protocol"])
        print(json.dumps({"status": "stamped", "output": args.output,
                          "digest": result["protocol"]["digest"]}, indent=2))
        return 0

    if args.command == "cache":
        from . import hybrid_study
        factory = _mae_factory(args.mae, args.device)
        manifest = hybrid_study.cache_frame_features(protocol, factory(), args.out,
                                                     run_mode=args.run_mode,
                                                     task_allowlist=args.tasks)
        print(json.dumps({"shards": sum(len(v) for v in manifest["shards"].values()),
                          "encoder": manifest["encoder"]["weights_sha256"][:16]}, indent=2))
        return 0

    if args.command == "train-source":
        from . import hybrid_study
        cache = hybrid_study.load_frame_cache(args.cache, protocol, run_mode=args.run_mode)
        kwargs = {}
        if args.updates is not None:
            kwargs["updates"] = args.updates
        summary = hybrid_study.train_source(protocol, cache, args.out, seed=args.seed,
                                            device=args.device, run_mode=args.run_mode, **kwargs)
        print(json.dumps({"path": summary["path"], "best": summary["best"]}, indent=2))
        return 0

    if args.command == "adapt":
        from . import hybrid_study
        require_stage_access(protocol, args.task, args.freeze_file)
        source = hybrid_study.load_source(args.source, protocol)
        cache = hybrid_study.load_frame_cache(args.cache, protocol, run_mode=args.run_mode)
        kwargs = {}
        if args.updates is not None:
            kwargs["updates"] = args.updates
        summary = hybrid_study.adapt_target(
            protocol, source, _mae_factory(args.mae, args.device), args.condition,
            args.task, args.budget, args.seed, args.out, cache=cache, device=args.device,
            run_mode=args.run_mode, **kwargs)
        print(json.dumps({"path": summary["path"], "updates": summary["updates"]}, indent=2))
        return 0

    if args.command == "evaluate-recorded":
        from . import hybrid_study
        require_stage_access(protocol, args.task, args.freeze_file)
        source = hybrid_study.load_source(args.source, protocol)
        cache = hybrid_study.load_frame_cache(args.cache, protocol,
                                             run_mode=source.get("execution_mode", "scientific"))
        adapted = {}
        if args.lora:
            adapted["lora_mlp"] = args.lora
        if args.raid_lora:
            adapted["raid_lora_mlp"] = args.raid_lora
        report = hybrid_study.evaluate_recorded(
            protocol, cache, source, args.task, args.budget, args.seed,
            out_dir=args.out, encoder_factory=_mae_factory(args.mae, args.device),
            adapted=adapted, device=args.device)
        print(json.dumps({"task": report["task"], "results": {k: v.get("chunk_mse_macro")
                                                              for k, v in report["results"].items()}}, indent=2))
        return 0

    if args.command == "retention":
        from . import hybrid_study
        source = hybrid_study.load_source(args.source, protocol)
        cache = hybrid_study.load_frame_cache(args.cache, protocol,
                                             run_mode=source.get("execution_mode", "scientific"))
        adapted = {}
        if args.lora:
            adapted["lora_mlp"] = args.lora
        if args.raid_lora:
            adapted["raid_lora_mlp"] = args.raid_lora
        report = hybrid_study.evaluate_retention(
            protocol, cache, source, args.task, args.budget, args.seed,
            out_dir=args.out, encoder_factory=_mae_factory(args.mae, args.device),
            adapted=adapted, device=args.device)
        print(json.dumps(report["results"], indent=2))
        return 0

    if args.command == "analyze":
        from . import hybrid_analysis
        episodes = []
        for path in sorted(Path(args.episodes_dir).glob("*.json")):
            payload = _load(path)
            if "conditions" in payload:
                episodes.append(payload)
        report = hybrid_analysis.write_analysis(episodes, Path(args.episodes_dir) / "analysis.json")
        print(json.dumps(report["macro"], indent=2))
        return 0

    if args.command == "rollout":
        require_stage_access(protocol, args.task, args.freeze_file)
        if args.run_mode == "scientific":
            if not args.panel_dir:
                raise SystemExit("scientific rollout requires --panel-dir with passed source-panel gates")
            from .hybrid_admission import validate_scientific_admission
            validate_scientific_admission(protocol, args.panel_dir, stage="rollout")
        from .hybrid_runner import run_pilot_task
        result = run_pilot_task(protocol, task=args.task, source_path=args.source,
                                cache_path=args.cache, mae_path=args.mae,
                                lora_path=args.lora, raid_lora_path=args.raid_lora,
                                libero_root=args.libero_root, output=args.out,
                                device=args.device, seed=args.seed, budget=args.budget,
                                max_steps=args.max_steps, run_mode=args.run_mode)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "freeze":
        report_dir = Path(args.report_dir)
        if not report_dir.is_dir():
            raise SystemExit(f"freeze --report-dir is not a directory: {report_dir}")
        recorded = sorted(report_dir.glob("recorded_*_b*_s*_u*.json"))
        if not recorded:
            raise SystemExit(f"freeze found no recorded_* reports under {report_dir}")
        for path in recorded:
            payload = _load(path)
            if payload.get("model_digest") != protocol.get("model_digest"):
                raise SystemExit(f"report {path.name} belongs to another protocol")
        record = {
            "status": "frozen",
            "protocol_digest": protocol["digest"],
            "model_digest": protocol["model_digest"],
            "selected_target_updates": protocol["design"]["training"]["target_updates"],
            "primary_budget": PRIMARY_BUDGET,
            "development_seed": 42,
            "report_dir": str(report_dir),
            "recorded_reports": [p.name for p in recorded],
            "rule": "pilot defaults fixed at 500 updates; no selection from final performance",
            "scientific_result": False,
        }
        _write(args.output, record)
        print(json.dumps(record, indent=2))
        return 0

    if args.command == "collect":
        return hybrid_collect.collect_cli(protocol, args)

    raise SystemExit(f"unimplemented command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
