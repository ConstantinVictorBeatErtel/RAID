"""Compute-aware offline benchmark: frozen, RAID (v6), LoRA, RAID+LoRA (v6), RAID (v7), RAID+LoRA (v7).

User-approved on 15 October 2026. Development tasks only (final tasks stay sealed), registered support seeds
42/43/44, five-shot support. Quality = recorded-transition chunk MSE on the 20 registered query demos per task
(real recorded future frames; no generator, no simulator). Compute = target-time adaptation GPU-seconds
(evaluation time excluded), FLOPs per update and per decision, decision latency and memory, all on one pod.
Quality is read at fixed update checkpoints, so each method gives a quality-vs-compute curve; RAID points sit at
zero adaptation compute.

Suites: "v1" (the six original conditions), "full" (v1 plus the re-implemented baselines), "v7head" (exploratory,
development only: the zero-training conditions plus the frozen-encoder head_only and raid_v7_head, which pair with
the bitwise-identical source controllers of earlier runs) and "final" (the frozen method set: full plus
raid_v7_head). Stage "final" reads the sealed final tasks instead of the development tasks and needs a frozen
amendment (hybrid_study.final_access); development and final tasks never share one run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import torch

from .hybrid_heads import chunk_mse, knn_average, make_bank
from .hybrid_protocol import PRIMARY_BUDGET, RetrievalSpec
from . import hybrid_study as hs
from . import hybrid_baselines as hb
from .hybrid_v7 import HEAD_KIND, describe, v7_retrieval

SEEDS = (42, 43, 44)
CHECKPOINTS = (25, 50, 100, 200, 350, 500)
ADAPTED = (("lora_mlp", "mlp_v6", "lora_mlp"), ("raid_lora_mlp", "mlp_v6", "raid_lora_mlp"),
           ("raid_lora_v7", HEAD_KIND, "raid_lora_mlp"))
SUITES = {"v1": (True, False, ()), "full": (True, True, ()), "v7head": (False, False, ("head_only", "raid_v7_head")),
          "final": (True, True, ("raid_v7_head",))}  # (LoRA conditions, full baseline set, extra frozen-encoder methods)
STAGE_ROLE = {"development": "development", "final": "final_test"}
TIMING_REPEATS = 50
# Many-core pods (255 cores) make PyTorch's default one-thread-per-core setting extremely slow (oversubscription);
# a fixed thread count also keeps CPU-side timing comparable across pods.
CPU_THREADS = 16


def _atomic(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _sync(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _flops(fn) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return int(counter.get_total_flops())


class TaskData:
    """Query rows, labels and raw query frames of one development task (loaded once)."""

    def __init__(self, protocol: dict, cache: dict, source: dict, task: str, *, final: dict | None = None):
        spec = protocol["tasks"][task]
        if spec["role"] not in ("development", "final_test"):
            raise ValueError("the compute benchmark reads held-out tasks only")
        if spec["role"] == "final_test" and not hs._final_allowed(protocol, [task], final):
            raise ValueError("final tasks need a verified final-access record")
        self.task, self.chunk = task, source["chunk_steps"]
        self.query = list(map(int, spec["query"]))
        labels = {d: hs.read_actions(spec["path"], d) for d in self.query}
        self.rows = hs.chunk_rows(cache["shards"][task], self.query, task, self.chunk, source["normalizers"],
                                  actions_override=labels)
        self.frames = {d: hs.read_view_frames(protocol, spec["path"], d) for d in self.query}


def evaluate(data: TaskData, controller, *, encoder=None, bank=None, use_retrieval: bool, device: str) -> dict:
    """Recorded-transition chunk MSE (macro over query demos); retrieval keys always use frozen features."""
    rows = data.rows
    ref_cur, ref_fut = rows.current.to(device), rows.future.to(device)
    was = controller.training
    controller.eval()
    mae_was = encoder.mae.training if encoder is not None else None
    with torch.no_grad():
        if encoder is None:
            cur, fut = ref_cur, ref_fut
        else:
            encoder.mae.eval()
            feats = {d: hs._encode(encoder, data.frames[d]) for d in data.query}
            cur = torch.stack([feats[d][t] for d, t in zip(rows.demo, rows.step)]).to(device)
            fut = torch.stack([feats[d][t + data.chunk] for d, t in zip(rows.demo, rows.step)]).to(device)
        pred, _ = controller.forward_with_memory(
            cur, fut, rows.q.to(device), rows.duration.to(device), bank=bank if use_retrieval else None,
            use_retrieval=use_retrieval, query_current=ref_cur if (use_retrieval and encoder is not None) else None,
            query_future=ref_fut if (use_retrieval and encoder is not None) else None)
        errors = chunk_mse(pred, rows.target.to(device)).cpu()
        r2 = hb.r2_score(pred.cpu(), rows.target)
    controller.train(was)
    if encoder is not None:
        encoder.mae.train(mae_was)
    per_demo: dict[int, list[float]] = {}
    for value, demo in zip(errors.tolist(), rows.demo):
        per_demo.setdefault(demo, []).append(value)
    by_demo = {str(d): float(np.mean(v)) for d, v in per_demo.items()}
    return {"mse_macro": float(np.mean(list(by_demo.values()))), "mse_by_demo": by_demo, "r2": r2}


def decision_cost(data: TaskData, controller, *, encoder, reference_encoder=None, bank=None,
                  use_retrieval: bool, device: str) -> dict:
    """One control decision with batch 1: encode current+future frames, (reference encode), retrieve, head."""
    demo = data.query[0]
    frames = torch.from_numpy(np.ascontiguousarray(np.stack([data.frames[demo][0], data.frames[demo][data.chunk]])))
    q = data.rows.q[:1].to(device)
    dur = data.rows.duration[:1].to(device)

    def decide():
        f = encoder.encode_frames(frames).float().to(device)
        ref = reference_encoder.encode_frames(frames).float().to(device) if reference_encoder is not None else f
        return controller.forward_with_memory(f[:1], f[1:], q, dur, bank=bank if use_retrieval else None,
                                              use_retrieval=use_retrieval, query_current=ref[:1],
                                              query_future=ref[1:])

    controller.eval()
    encoder.mae.eval()
    with torch.no_grad():
        for _ in range(5):
            decide()
        _sync(device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(TIMING_REPEATS):
            decide()
        _sync(device)
        seconds = (time.perf_counter() - started) / TIMING_REPEATS
        flops = _flops(decide)
    return {"seconds_per_decision": seconds, "flops_per_decision": flops,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "bank_entries": len(bank) if (bank is not None and use_retrieval) else 0,
            "includes": "2-frame encode" + (" + reference encode" if reference_encoder is not None else "")
                        + (" + kNN retrieval" if use_retrieval else "") + " + head"}


def update_flops(protocol: dict, source: dict, task: str, seed: int, controller, encoder, *, cache: dict,
                 use_retrieval: bool, device: str) -> int:
    """FLOPs of one adaptation update (batch 64 = 8 microbatches of 8, forward + backward)."""
    cfg = protocol["design"]["training"]
    ids = hs.support_ids(protocol, task, PRIMARY_BUDGET, seed)
    path = protocol["tasks"][task]["path"]
    chunk = source["chunk_steps"]
    data = hs._SupportLive({d: hs.read_view_frames(protocol, path, d) for d in ids},
                           {d: hs.read_actions(path, d) for d in ids},
                           {d: hs.read_proprio(path, d, protocol) for d in ids}, chunk, source["normalizers"])
    bank = None
    if use_retrieval:
        br = hs.chunk_rows(cache["shards"][task], ids, task, chunk, source["normalizers"])
        bank = make_bank(br.current, br.future, br.q, br.target, br.duration, br.group_id, device)
    part = torch.arange(cfg["microbatch_size"])

    def micro():
        cur_img, fut_img, q, tgt, _, demos, steps = data.get(part)
        feats = encoder.encode_frames(torch.cat((cur_img, fut_img)))
        n = len(part)
        dur = torch.full((n,), float(chunk), device=device)
        kwargs = {}
        if use_retrieval:
            kwargs = {"query_current": torch.stack([cache["shards"][task][int(d)]["features"][int(t)]
                                                    for d, t in zip(demos, steps)]).to(device),
                      "query_future": torch.stack([cache["shards"][task][int(d)]["features"][int(t) + chunk]
                                                   for d, t in zip(demos, steps)]).to(device)}
        pred, info = controller.forward_with_memory(feats[:n], feats[n:], q.to(device), dur, bank=bank,
                                                    use_retrieval=use_retrieval, **kwargs)
        loss = chunk_mse(pred, tgt.to(device)).mean()
        if use_retrieval:
            loss = loss + hs._selection_term(controller, info, tgt.to(device).reshape(n, -1))
        loss.backward()

    flops = _flops(micro)
    for p in list(controller.parameters()) + list(encoder.mae.parameters()):
        p.grad = None  # adapt_target zeroes gradients at every update; nothing leaks into training
    return flops * (cfg["batch_size"] // cfg["microbatch_size"])


def _final_ledger(collection: Path, final: dict, tasks: list[str]) -> None:
    """One final-stage run: the ledger binds the freeze record; a resumed stage must carry the same record."""
    path = collection / "final_access.json"
    if path.is_file():
        if json.loads(path.read_text())["freeze_sha256"] != final["freeze_sha256"]:
            raise ValueError("final tasks were already read under another freeze record")
        return
    _atomic(path, {"freeze_sha256": final["freeze_sha256"], "amendment_digest": final.get("digest"),
                   "tasks": tasks, "first_access_unix": time.time(), "held_out_claims_eligible": False,
                   "rule": "one frozen run; nothing is tuned after it"})


def run(root: str | Path, *, device: str = "cuda", run_mode: str = hs.SCIENTIFIC, suite: str = "v1",
        stage: str = "development", freeze_file: str | None = None) -> dict:
    """See the module docstring for suites and stages."""
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {sorted(SUITES)}")
    if stage not in STAGE_ROLE:
        raise ValueError("stage must be development or final")
    if (stage == "final") != (suite == "final"):
        raise ValueError("the final stage runs exactly the frozen final suite")
    run_lora, run_full, extra_methods = SUITES[suite]
    torch.set_num_threads(CPU_THREADS)
    root = Path(root)
    protocol = json.loads((root / "collection/protocol_bench_pod.json").read_text())
    final = hs.final_access(protocol, freeze_file) if stage == "final" else None
    out = root / "work/bench"
    collection = root / "collection/bench"
    mae_path = str(root / "weights/mae_pretrain_vit_base.pth")
    from .hybrid_cli import _mae_factory
    encoder_factory = _mae_factory(mae_path, device)
    role = STAGE_ROLE[stage]
    tasks = sorted(t for t, s in protocol["tasks"].items() if s["role"] == role)
    if final is not None:
        _final_ledger(collection, final, tasks)
    cache_dir = out / "cache"
    if not (cache_dir / "manifest.json").is_file():
        started = time.perf_counter()
        hs.cache_frame_features(protocol, encoder_factory(), cache_dir, run_mode=run_mode,
                                task_allowlist=[t for t, s in protocol["tasks"].items()
                                                if s["role"] in {"source", role}], final=final)
        _atomic(collection / "frame_cache_cost.json", {"seconds": time.perf_counter() - started,
                                                        "note": "shared one-time reference feature cache"})
    cache = hs.load_frame_cache(cache_dir, protocol, run_mode=run_mode, final=final)
    training = protocol["design"]["training"]
    total = int(training["target_updates"])
    steps = tuple(sorted({c for c in CHECKPOINTS if c < total} | {total}))
    base = hs._as_retrieval(protocol["design"]["retrieval"])
    _atomic(collection / "bench_design.json", {"v7": describe(), "seeds": SEEDS, "checkpoints": steps, "suite": suite,
                                               "baselines": hb.describe() if run_full else None,
                                               "extra_methods": list(extra_methods), "stage": stage,
                                               "final_freeze_sha256": final["freeze_sha256"] if final else None,
                                               "cpu_threads": torch.get_num_threads(),
                                               "tasks": tasks, "budget": PRIMARY_BUDGET,
                                               "adapted": [a[0] for a in ADAPTED] if run_lora else [],
                                               "protocol_digest": protocol["digest"]})
    for seed in SEEDS:
        sources, source_cost = {}, {}
        for kind in ("mlp_v6", HEAD_KIND):
            sdir = out / f"source_{kind}_s{seed}"
            path = sdir / f"source_v6_seed{seed}.pt"
            if not path.is_file():
                hs.train_source(protocol, cache, sdir, seed=seed, device=device,
                                updates=int(training["source_updates"]), batch=int(training["batch_size"]),
                                validate_every=int(training["validation_every"]), run_mode=run_mode,
                                head_kind=kind, retrieval_override=None if kind == "mlp_v6" else v7_retrieval(base))
            sources[kind] = hs.load_source(path, protocol)
            source_cost[kind] = {"training_seconds": sources[kind]["training_seconds"],
                                 "forward_passes": sources[kind]["forward_passes"],
                                 "best_step": sources[kind]["controller"]["best_step"],
                                 "best_validation_mean": sources[kind]["controller"]["best_validation_mean"]}
        _atomic(collection / f"source_cost_s{seed}.json", source_cost)
        for task in tasks:
            unit = collection / f"unit_{task}_s{seed}.json"
            if unit.is_file():
                continue
            data = TaskData(protocol, cache, sources["mlp_v6"], task, final=final)
            frozen_encoder = encoder_factory()
            v6 = hs.build_controller(sources["mlp_v6"], device)
            v7 = hs.build_controller(sources[HEAD_KIND], device)
            started = time.perf_counter()
            bank6, info6 = hs.build_support_bank(protocol, cache, sources["mlp_v6"], task, PRIMARY_BUDGET, seed, device)
            bank_seconds = time.perf_counter() - started
            bank7, _ = hs.build_support_bank(protocol, cache, sources[HEAD_KIND], task, PRIMARY_BUDGET, seed, device)
            result = {"task": task, "seed": seed, "support_demo_ids": info6["support_demo_ids"],
                      "bank_entries": info6["bank_entries"], "bank_construction_seconds": bank_seconds,
                      "zero_adaptation": {
                          "source_frozen": evaluate(data, v6, use_retrieval=False, device=device),
                          "raid_bank_mlp": evaluate(data, v6, bank=bank6, use_retrieval=True, device=device),
                          "raid_bank_v7": evaluate(data, v7, bank=bank7, use_retrieval=True, device=device)},
                      "decision_cost": {
                          "source_frozen": decision_cost(data, v6, encoder=frozen_encoder, use_retrieval=False, device=device),
                          "raid_bank_mlp": decision_cost(data, v6, encoder=frozen_encoder, bank=bank6, use_retrieval=True, device=device),
                          "raid_bank_v7": decision_cost(data, v7, encoder=frozen_encoder, bank=bank7, use_retrieval=True, device=device)},
                      "curves": {}, "adaptation": {}}
            with torch.no_grad():
                knn = knn_average(bank6, data.rows.current.to(device), data.rows.future.to(device),
                                  data.rows.q.to(device), data.rows.duration.to(device),
                                  spec=hs._as_retrieval(sources["mlp_v6"]["retrieval_spec"]), chunk_steps=data.chunk)
                errors = chunk_mse(knn, data.rows.target.to(device)).cpu()
                by_demo: dict[str, list[float]] = {}
                for value, demo in zip(errors.tolist(), data.rows.demo):
                    by_demo.setdefault(str(demo), []).append(value)
                result["zero_adaptation"]["support_knn_average"] = {
                    "mse_macro": hs._macro(errors, data.rows.demo),
                    "mse_by_demo": {d: float(np.mean(v)) for d, v in by_demo.items()},
                    "r2": hb.r2_score(knn.cpu(), data.rows.target)}
            start_point = {"lora_mlp": "source_frozen", "raid_lora_mlp": "raid_bank_mlp", "raid_lora_v7": "raid_bank_v7"}
            for name, kind, condition in (ADAPTED if run_lora else ()):
                use_r = condition == "raid_lora_mlp"
                adir = out / f"adapt_{name}_{task}_s{seed}"
                shutil.rmtree(adir, ignore_errors=True)  # a unit always restarts cleanly (curve points are in memory)
                bank = bank6 if kind == "mlp_v6" else bank7
                points = [{"updates": 0, "adaptation_seconds": 0.0,
                           **result["zero_adaptation"][start_point[name]]}]
                extra = {}

                def checkpoint(step, controller, encoder, seconds, name=name, use_r=use_r, bank=bank,
                               points=points, extra=extra, kind=kind):
                    points.append({"updates": step, "adaptation_seconds": seconds,
                                   **evaluate(data, controller, encoder=encoder, bank=bank, use_retrieval=use_r,
                                              device=device)})
                    if step == steps[0]:
                        extra["flops_per_update"] = update_flops(protocol, sources[kind], task, seed, controller, encoder,
                                                                 cache=cache, use_retrieval=use_r, device=device)
                    if step == steps[-1]:
                        extra["decision_cost"] = decision_cost(data, controller, encoder=encoder,
                                                               reference_encoder=frozen_encoder if use_r else None,
                                                               bank=bank, use_retrieval=use_r, device=device)

                summary = hs.adapt_target(protocol, sources[kind], encoder_factory, condition, task, PRIMARY_BUDGET, seed,
                                          adir, cache=cache, device=device, run_mode=run_mode,
                                          checkpoint_callback=checkpoint, checkpoint_steps=steps)
                for point in points:
                    point["flops"] = extra.get("flops_per_update", 0) * point["updates"]
                result["curves"][name] = points
                result["adaptation"][name] = {"adaptation_seconds": summary["adaptation_seconds"],
                                              "trainable_parameters": summary["trainable_parameters"],
                                              "peak_gpu_bytes": summary["peak_gpu_bytes"],
                                              "flops_per_update": extra.get("flops_per_update")}
                result["decision_cost"][name] = extra.get("decision_cost")
            if run_full or extra_methods:
                _run_baselines(result, protocol, sources, encoder_factory, frozen_encoder, data, task, seed, cache,
                               bank6, steps, device, names=(hb.TRAINED if run_full else ()) + tuple(
                                   n for n in extra_methods if not (run_full and n in hb.TRAINED)),
                               zero_training=run_full, bank7=bank7)
            if final is not None:
                result["final_freeze_sha256"] = final["freeze_sha256"]
            _atomic(unit, result)
    return report(root)


def _run_baselines(result, protocol, sources, encoder_factory, frozen_encoder, data, task, seed, cache, bank6,
                   steps, device, *, names=hb.TRAINED, zero_training=True, bank7=None) -> None:
    v6 = hs.build_controller(sources["mlp_v6"], device)
    base_retrieval = hs._as_retrieval(sources["mlp_v6"]["retrieval_spec"])
    for name, cls in ((("retrieve_and_play", hb.RetrieveAndPlay), ("regent_blend", hb.RegentBlend))
                      if zero_training else ()):
        model = cls(v6, base_retrieval, data.chunk)
        result["zero_adaptation"][name] = evaluate(data, model, bank=bank6, use_retrieval=True, device=device)
        result["decision_cost"][name] = decision_cost(data, model, encoder=frozen_encoder, bank=bank6,
                                                      use_retrieval=True, device=device)
    for name in names:
        retrieval = name in hb.EXPLORATORY  # raid_v7_head: v7 source, support bank, retrieval at every step
        encoder_mode = (hb.EXPLORATORY if retrieval else hb.SPECS)[name][0]
        source = sources[HEAD_KIND] if retrieval else sources["mlp_v6"]
        bank = bank7 if retrieval else None
        points, extra = [], {}

        def checkpoint(step, controller, encoder, seconds, points=points, extra=extra, encoder_mode=encoder_mode,
                       source=source, bank=bank):
            points.append({"updates": step, "adaptation_seconds": seconds,
                           **evaluate(data, controller, encoder=encoder, bank=bank, use_retrieval=bank is not None,
                                      device=device)})
            if step == steps[0]:
                extra["flops_per_update"] = (
                    hb.frozen_update_flops(protocol, source, controller, cache, task, seed, device, bank=bank)
                    if encoder_mode == "frozen" else
                    update_flops(protocol, source, task, seed, controller, encoder, cache=cache,
                                 use_retrieval=False, device=device))
            if step == steps[-1]:
                extra["decision_cost"] = decision_cost(data, controller, encoder=encoder or frozen_encoder, bank=bank,
                                                       use_retrieval=bank is not None, device=device)

        summary = hb.train_variant(name, protocol, source, encoder_factory, task, seed, cache=cache,
                                   device=device, steps=(0,) + tuple(steps), callback=checkpoint, bank=bank)
        for point in points:
            point["flops"] = extra.get("flops_per_update", 0) * point["updates"]
        result["curves"][name] = points
        result["adaptation"][name] = {**summary, "flops_per_update": extra.get("flops_per_update")}
        result["decision_cost"][name] = extra.get("decision_cost")


def _time_to_target(points: list[dict], target: float) -> float | None:
    """Adaptation seconds at which the curve first reaches the target MSE (linear between checkpoints)."""
    for a, b in zip(points, points[1:]):
        if a["mse_macro"] <= target:
            return a["adaptation_seconds"]
        if b["mse_macro"] <= target:
            span = a["mse_macro"] - b["mse_macro"]
            frac = (a["mse_macro"] - target) / span if span > 0 else 1.0
            return a["adaptation_seconds"] + frac * (b["adaptation_seconds"] - a["adaptation_seconds"])
    return points[0]["adaptation_seconds"] if points and points[0]["mse_macro"] <= target else None


def report(root: str | Path) -> dict:
    root = Path(root)
    collection = root / "collection/bench"
    units = [json.loads(p.read_text()) for p in sorted(collection.glob("unit_*_s*.json"))]
    rows = []
    for u in units:
        targets = {"raid_bank_mlp": u["zero_adaptation"]["raid_bank_mlp"]["mse_macro"],
                   "raid_bank_v7": u["zero_adaptation"]["raid_bank_v7"]["mse_macro"]}
        row = {"task": u["task"], "seed": u["seed"],
               "zero": {k: v["mse_macro"] for k, v in u["zero_adaptation"].items()},
               "final": {k: v[-1]["mse_macro"] for k, v in u["curves"].items()},
               "seconds_to_match_raid_v6": {k: _time_to_target(v, targets["raid_bank_mlp"]) for k, v in u["curves"].items()},
               "seconds_to_match_raid_v7": {k: _time_to_target(v, targets["raid_bank_v7"]) for k, v in u["curves"].items()}}
        rows.append(row)
    conditions = list(units[0]["zero_adaptation"]) if units else []
    adapted = list(units[0]["curves"]) if units else []
    summary = {"units": len(units), "rows": rows,
               "mean_zero_adaptation_mse": {c: statistics.mean(r["zero"][c] for r in rows) for c in conditions} if rows else {},
               "mean_final_mse": {a: statistics.mean(r["final"][a] for r in rows) for a in adapted} if rows else {},
               "mean_final_r2": {a: statistics.mean(u["curves"][a][-1]["r2"] for u in units) for a in adapted} if units else {},
               "mean_zero_adaptation_r2": {c: statistics.mean(u["zero_adaptation"][c].get("r2", float("nan")) for u in units)
                                           for c in conditions} if units else {},
               "scientific_result": False, "held_out_claims_eligible": False,
               "final_freeze_sha256": units[0].get("final_freeze_sha256") if units else None,
               "scope": ("sealed final tasks, one frozen run" if units and units[0].get("final_freeze_sha256")
                         else "development tasks only") + "; recorded transitions; registered seeds 42/43/44; five-shot"}
    _atomic(collection / "bench_summary.json", summary)
    return {"units": len(units)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--suite", default="v1", choices=sorted(SUITES))
    parser.add_argument("--stage", default="development", choices=sorted(STAGE_ROLE))
    parser.add_argument("--freeze-file", default=None)
    a = parser.parse_args(argv)
    print(json.dumps(run(a.root, device=a.device, suite=a.suite, stage=a.stage, freeze_file=a.freeze_file)
                     if a.command == "run" else report(a.root)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
