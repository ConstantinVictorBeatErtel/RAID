#!/usr/bin/env python3
"""Session entry for protocol v4 validation and v6 source-panel pods.

``launch`` creates a pod only when ``--authorize-spend`` is passed. ``plan``
prints the selected plan and contacts nobody.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))




def launch_validation(*args, **kwargs):
    from raid_experiments.wm_pod import launch_validation as operation
    return operation(*args, **kwargs)


def launch_source_panel(*args, **kwargs):
    from raid_experiments.wm_pod import launch_source_panel as operation
    return operation(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pod_wm_session")
    parser.add_argument("command", choices=["plan", "prepare", "launch", "status", "watch", "collect", "resume", "teardown"])
    parser.add_argument("--mode", choices=["validation", "source-panel", "operational-acceptance", "pilot"], default="validation")
    parser.add_argument("--session", default=None)
    parser.add_argument("--authorize-spend", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=10)
    parser.add_argument("--max-wait-seconds", type=int, default=7200)
    parser.add_argument("--acceptance-session", default=None)
    parser.add_argument("--continue-setup", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "operational-acceptance":
        from raid_experiments.operational_acceptance import dispatch
        result = dispatch(args.command, Path(args.session or "artifacts/raid_hybrid_libero_goal_v6/operational_acceptance"),
                          authorize=args.authorize_spend, interval=args.interval_seconds, max_wait=args.max_wait_seconds)
        print(json.dumps(result, indent=2))
        return 0
    if args.mode == "pilot":
        print(f"{args.mode} lifecycle is not implemented in this checkout; refusing to launch", file=sys.stderr)
        return 2

    if args.mode == 'source-panel':
        from raid_experiments import managed_source
        session=Path(args.session or 'artifacts/raid_hybrid_libero_goal_v6/source_session')
        if args.command=='plan':
            from raid_experiments.wm_pod import source_panel_plan
            print(json.dumps(source_panel_plan(),indent=2));return 0
        if args.command=='prepare':result=managed_source.prepare(session)
        elif args.command=='launch':
            if not args.authorize_spend:
                print('launch refused: --authorize-spend required',file=sys.stderr);return 2
            if not args.acceptance_session:
                print('source launch requires authenticated --acceptance-session of this exact release',file=sys.stderr);return 2
            result=managed_source.launch(session,args.acceptance_session)
        elif args.command=='status':result=managed_source.status(session)
        elif args.command=='resume':result=managed_source.resume(session,setup=args.continue_setup)
        elif args.command=='collect':result=managed_source.collect(session)
        elif args.command=='watch':result=managed_source.watch(session,args.max_wait_seconds)
        elif args.command=='teardown':
            from raid_experiments.operational_acceptance import service
            result=managed_source.status(session)
            service('/v2/sessions/'+result['request']['sessionId']+'/terminate',{})
            result=managed_source.watch(session,args.max_wait_seconds)
        else:raise ValueError('unsupported source lifecycle command')
        print(json.dumps(result,indent=2));return 0

    from raid_experiments.wm_pod import (  # noqa: E402
        collect_source_panel, source_panel_plan,
        source_panel_status, teardown_source_panel, validation_plan, watch_source_panel,
        resume_source_panel,
    )
    if args.mode == "source-panel":
        session = Path(args.session or "artifacts/raid_wm_source_panel_session")
        if args.command in {"status", "watch", "collect", "resume", "teardown"}:
            from pod_provider import RunpodProvider
            provider = RunpodProvider()
            if args.command == "status":
                print(json.dumps(source_panel_status(session, provider), indent=2))
                return 0
            if args.command == "collect":
                print(json.dumps(collect_source_panel(session, provider), indent=2))
                return 0
            if args.command == "teardown":
                print(json.dumps(teardown_source_panel(session, provider), indent=2))
                return 0
            operation = (resume_source_panel if args.command == "resume" else watch_source_panel)
            print(json.dumps(operation(
                session, provider, interval_seconds=args.interval_seconds,
                max_wait_seconds=args.max_wait_seconds), indent=2))
            return 0
        if args.command == "plan":
            print(json.dumps(source_panel_plan(), indent=2))
            return 0
        if not args.authorize_spend:
            print("launch refused: pass --authorize-spend to rent the source-panel pod",
                  file=sys.stderr)
            return 2
        from pod_provider import RunpodProvider  # imported only after authorization
        launch_source_panel(session, RunpodProvider(), authorize_spend=True)
        return 0
    if args.command not in {"plan", "launch"}:
        print(f"{args.command} is not supported for validation mode", file=sys.stderr)
        return 2
    session = Path(args.session or "artifacts/raid_wm_validation_session")
    if args.command == "plan":
        print(json.dumps(validation_plan(), indent=2))
        return 0
    if not args.authorize_spend:
        print("launch refused: pass --authorize-spend to rent the validation pod", file=sys.stderr)
        return 2
    from pod_provider import RunpodProvider  # imported only after authorization
    launch_validation(session, RunpodProvider(), authorize_spend=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
