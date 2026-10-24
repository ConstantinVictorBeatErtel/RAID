"""RAID v8 stage driver (development or final). Resumable: finished units are skipped.

    python -m raid_experiments.v8.bench --stage development --config CONFIG.json --data DIR --cache DIR --out DIR
    python -m raid_experiments.v8.bench --stage final --config AMENDMENT.json --sources DIR ...

Steps: download -> cache -> source -> offline -> encoder -> closed_loop -> timing. The development stage never
touches the final suites; the final stage requires a frozen amendment and the source checkpoints it binds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from . import cache as c
from . import data as d
from . import encoders as enc
from . import source as s
from . import target as tg

STEPS = ("download", "cache", "source", "offline", "encoder", "closed_loop", "timing")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def config_digest(cfg: dict) -> str:
    body = {k: v for k, v in cfg.items() if k not in ("digest", "frozen_at")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def target_suites(stage: str, cfg: dict) -> tuple[str, ...]:
    if stage == "development":
        return d.DEV_SUITES
    if cfg.get("status") != "frozen" or cfg.get("digest") != config_digest(cfg):
        raise SystemExit("final stage needs a frozen amendment with a matching digest")
    return d.FINAL_SUITES


def source_ckpt(sources: Path, encoder: str, kind: str, seed: int) -> Path:
    return sources / encoder / f"{kind}_s{seed}.pt"


def load_source(sources: Path, encoder: str, kind: str, seed: int, device: str):
    from .heads import build
    payload = torch.load(source_ckpt(sources, encoder, kind, seed), map_location="cpu", weights_only=False)
    model = build(payload["head"], d.CHUNK, enc.FEATURE_DIM, 9, corrupt=payload["corrupt"])
    model.load_state_dict(payload["state"])
    return model.to(device).eval(), payload


def step_download(stage, cfg, args):
    suites = (d.SOURCE_SUITES + d.DEV_SUITES) if stage == "development" else d.FINAL_SUITES
    rec = Path(args.out) / f"data_sha256_{stage}.json"
    if rec.is_file():
        return
    log("download", suites)
    d.download(suites, args.data, record=rec)


def step_cache(stage, cfg, args):
    """One HDF5 read per demo; all encoders encode it (features fp16 per encoder and task)."""
    suites = (d.SOURCE_SUITES + d.DEV_SUITES) if stage == "development" else target_suites(stage, cfg)
    tasks = d.list_tasks(args.data, suites)
    encoders = {}
    for name in cfg["encoders"]:
        encoders[name], identity = enc.load(name, args.device, root=args.root)
        c.write_json(Path(args.out) / "encoders" / f"{name}.json", identity)
    t0, frames = time.time(), 0
    for task in tasks:
        r = c.build_task_cache(task, encoders, args.cache)
        if not r.get("skipped"):
            frames += r["frames"]
            log("cache", task.key, r["frames"], r["seconds"])
    c.write_json(Path(args.out) / "timing" / f"cache_{stage}.json",
                 {"encoders": cfg["encoders"], "tasks": len(tasks), "frames": frames, "seconds": time.time() - t0})
    del encoders
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def _source_inputs(cfg, args, encoder: str):
    tasks = d.list_tasks(args.data, d.SOURCE_SUITES)
    caches = [c.load_task(args.cache, encoder, t) for t in tasks]
    splits = [d.source_split(t, sorted(tc["demos"])) for t, tc in zip(tasks, caches)]
    return tasks, caches, splits


def step_source(stage, cfg, args):
    if stage != "development":
        return  # the final stage reuses the source checkpoints bound by the amendment
    sources = Path(args.out) / "sources"
    norms = None
    for name in cfg["encoders"]:
        tasks, caches, splits = _source_inputs(cfg, args, name)
        if norms is None:
            norms = c.normalizers([(tc, sp["train"]) for tc, sp in zip(caches, splits)])
            sources.mkdir(parents=True, exist_ok=True)
            torch.save(norms, sources / "normalizers.pt")
        src = s.SourceData(caches, splits, norms, args.device)
        for kind, corrupt in cfg["source_heads"]:
            for seed in cfg["seeds"]:
                out = source_ckpt(sources, name, kind, seed)
                if out.is_file():
                    continue
                model, record = s.train(kind.split("_")[0], src, seed=seed, updates=cfg["source_updates"],
                                        corrupt=corrupt, val_every=cfg.get("source_val_every", 250), log=log)
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"head": kind.split("_")[0], "label": kind, "corrupt": corrupt, "seed": seed,
                            "encoder": name, "state": model.state_dict(), "record": record}, out)
                c.write_json(out.with_suffix(".json"), record)
        del src
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def _target_rows(task, tc, norms, seed, task_index):
    split = d.target_split(task, sorted(tc["demos"]))
    support = d.support_set(task, split["support_pool"], seed)
    return (c.rows(tc, support, norms, task_index=task_index), c.rows(tc, split["query"], norms, task_index=task_index),
            support, split)


def step_offline(stage, cfg, args):
    """Training-free methods and frozen-feature head updates, all encoders, all seeds."""
    sources = Path(args.sources or Path(args.out) / "sources")
    norms = torch.load(sources / "normalizers.pt")
    tasks = d.list_tasks(args.data, target_suites(stage, cfg))
    for name in cfg["encoders"]:
        for seed in cfg["seeds"]:
            models = {kind: load_source(sources, name, kind, seed, args.device)[0] for kind, _ in cfg["source_heads"]}
            base = models[cfg["base_head"]]
            for ti, task in enumerate(tasks):
                out = Path(args.out) / "offline" / name / f"{task.suite}__{task.name}__s{seed}.json"
                if out.is_file():
                    continue
                tc = c.load_task(args.cache, name, task)
                sup, qry, support, split = _target_rows(task, tc, norms, seed, 500 + ti)
                sup, qry = sup.to(args.device), qry.to(args.device)
                bank = tg.support_bank(sup, args.device)
                res = {"task": task.key, "encoder": name, "seed": seed, "support": support, "query": split["query"],
                       "zero": {"knn": tg.score(tg.knn(bank, qry), qry),
                                "regent_blend": tg.score(tg.regent_blend(base, bank, qry), qry),
                                "head_only": tg.score(tg.predict(base, qry, use_retrieval=False), qry)},
                       "adapted": {}}
                for kind in models:
                    res["zero"][kind] = tg.score(tg.predict(models[kind], qry, bank=bank, use_retrieval=True), qry)
                table = cfg["head_lrs"].get(name, cfg["head_lrs"])  # final amendments store learning rates per encoder
                for method, kind, retrieval in cfg["head_methods"]:
                    for lr in table[method]:
                        r = tg.adapt_head(models[kind], sup, qry, use_retrieval=retrieval, lr=lr, seed=seed,
                                          device=args.device, checkpoints=tuple(cfg.get("head_checkpoints", tg.HEAD_CHECKPOINTS)))
                        res["adapted"][f"{method}@{lr:g}"] = r["curve"]
                c.write_json(out, res)
                log("offline", name, task.key, seed, "knn", round(res["zero"]["knn"]["mse"], 4),
                    {k: round(v["mse"], 4) for k, v in res["zero"].items()})


def _frames(task, demos):
    return {int(dm): d.read_demo(task.path, dm)["frames"] for dm in demos}


def step_encoder(stage, cfg, args):
    """MAE LoRA and full fine-tuning (retrieval off), one seed, learning-rate grid from the config."""
    if "mae" not in cfg["encoders"]:
        return
    sources = Path(args.sources or Path(args.out) / "sources")
    norms = torch.load(sources / "normalizers.pt")
    tasks = d.list_tasks(args.data, target_suites(stage, cfg))
    tasks = [t for t in tasks if not cfg.get("encoder_tasks") or t.key in cfg["encoder_tasks"]]
    seed = cfg["encoder_seed"]
    base = load_source(sources, "mae", cfg["base_head"], seed, args.device)[0]
    factory = lambda: enc.load("mae", args.device, root=args.root)[0]
    for ti, task in enumerate(tasks):
        out = Path(args.out) / "encoder" / f"{task.suite}__{task.name}__s{seed}.json"
        if out.is_file():
            continue
        tc = c.load_task(args.cache, "mae", task)
        sup, qry, support, split = _target_rows(task, tc, norms, seed, 500 + ti)
        sup, qry = sup.to(args.device), qry.to(args.device)
        sf, qf = _frames(task, support), _frames(task, split["query"])
        res = {"task": task.key, "seed": seed, "runs": {}}
        for mode in ("lora", "full"):
            for lr in cfg["encoder_lrs"][mode]:
                r = tg.adapt_encoder(base, factory, sup, sf, qry, qf, mode=mode, lr=lr, head_lr=cfg["encoder_head_lr"],
                                     seed=seed, device=args.device,
                                     checkpoints=tuple(cfg.get("encoder_checkpoints", tg.ENCODER_CHECKPOINTS)))
                res["runs"][f"{mode}@{lr:g}"] = {"curve": r["curve"], "trainable_params": r["trainable_params"]}
                log("encoder", task.key, mode, lr, [round(p["mse"], 4) for p in r["curve"]])
                del r
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        c.write_json(out, res)


def step_closed_loop(stage, cfg, args):
    from . import closed_loop as cl
    sources = Path(args.sources or Path(args.out) / "sources")
    norms = torch.load(sources / "normalizers.pt")
    seed, name = cfg["closed_loop"]["seed"], "mae"
    tasks = d.list_tasks(args.data, target_suites(stage, cfg))
    frozen = enc.load(name, args.device, root=args.root)[0]
    models = {kind: load_source(sources, name, kind, seed, args.device)[0] for kind, _ in cfg["source_heads"]}
    for ti, task in enumerate(tasks):
        out = Path(args.out) / "closed_loop" / f"{task.suite}__{task.name}__s{seed}.json"
        if out.is_file():
            continue
        tc = c.load_task(args.cache, name, task)
        sup, qry, support, split = _target_rows(task, tc, norms, seed, 500 + ti)
        sup = sup.to(args.device)
        bank = tg.support_bank(sup, args.device)
        demos = split["query"][: cfg["closed_loop"]["episodes"]]
        policies = {"knn": cl.knn_policy(bank, device=args.device)}
        for kind in models:
            policies[f"{kind}_zero"] = cl.head_policy(models[kind], frozen, bank=bank, use_retrieval=True,
                                                      device=args.device)
        qry = qry.to(args.device)
        for method, kind, retrieval, lr, updates in cfg["closed_loop"]["adapted"]:
            r = tg.adapt_head(models[kind], sup, qry, use_retrieval=retrieval, lr=lr, seed=seed, device=args.device,
                              checkpoints=(updates,))
            policies[f"{method}@{updates}"] = cl.head_policy(r["model"], frozen, bank=r["bank"],
                                                             use_retrieval=retrieval, device=args.device)
        for mode, lr, updates in cfg["closed_loop"]["encoder"]:
            sf, qf = _frames(task, support), _frames(task, split["query"])
            r = tg.adapt_encoder(models[cfg["base_head"]], lambda: enc.load(name, args.device, root=args.root)[0],
                                 sup, sf, qry, qf, mode=mode, lr=lr, head_lr=cfg["encoder_head_lr"], seed=seed,
                                 device=args.device, checkpoints=(updates,))
            r["encoder"].model.eval()
            policies[f"{mode}@{updates}"] = cl.head_policy(r["model"], frozen, use_retrieval=False,
                                                           adapted_encoder=r["encoder"], device=args.device)
        res = {"task": task.key, "seed": seed, "methods": {}}
        env = cl.vector_env(args.libero_root, task, len(demos))
        try:
            for method, policy in policies.items():
                res["methods"][method] = cl.run_task(task, demos, tc, policy, frozen, norms,
                                                     libero_root=args.libero_root, env=env,
                                                     gripper=cfg["closed_loop"].get("gripper", "raw"))
                log("closed_loop", task.key, method, res["methods"][method]["success_rate"])
        finally:
            env.close()
        c.write_json(out, res)


def step_timing(stage, cfg, args):
    """Per-update GPU time and FLOPs of each adaptation method, and per-decision latency (batch 1)."""
    out = Path(args.out) / "timing" / f"methods_{args.gpu_label}.json"
    if out.is_file():
        return
    sources = Path(args.sources or Path(args.out) / "sources")
    norms = torch.load(sources / "normalizers.pt")
    task = d.list_tasks(args.data, target_suites(stage, cfg))[0]
    seed = cfg["seeds"][0]
    tc = c.load_task(args.cache, "mae", task)
    sup, qry, support, split = _target_rows(task, tc, norms, seed, 500)
    sup, qry = sup.to(args.device), qry.to(args.device)
    models = {kind: load_source(sources, "mae", kind, seed, args.device)[0] for kind, _ in cfg["source_heads"]}
    rec = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "methods": {}}
    for method, kind, retrieval in cfg["head_methods"]:
        r = tg.adapt_head(models[kind], sup, qry, use_retrieval=retrieval, lr=3e-4, seed=seed, device=args.device,
                          checkpoints=(0, 100))
        rec["methods"][method] = {"seconds_per_update": r["curve"][-1]["gpu_seconds"] / 100,
                                  "trainable_params": sum(p.numel() for p in models[kind].parameters())}
    factory = lambda: enc.load("mae", args.device, root=args.root)[0]
    sf, qf = _frames(task, support), _frames(task, split["query"][:2])
    small_q = qry.select(torch.tensor([i for i, dm in enumerate(qry.demo) if dm in qf]))
    for mode in ("lora", "full"):
        r = tg.adapt_encoder(models[cfg["base_head"]], factory, sup, sf, small_q, qf, mode=mode, lr=1e-4,
                             head_lr=3e-4, seed=seed, device=args.device, checkpoints=(0, 50))
        rec["methods"][mode] = {"seconds_per_update": r["curve"][-1]["gpu_seconds"] / 50,
                                "trainable_params": r["trainable_params"]}
    c.write_json(out, rec)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("development", "final"), required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sources", default=None)
    p.add_argument("--libero-root", default=None)
    p.add_argument("--root", default=".")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu-label", default="primary")
    p.add_argument("--steps", default=",".join(STEPS))
    args = p.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    for step in args.steps.split(","):
        log("STEP", step)
        t0 = time.time()
        try:
            globals()[f"step_{step}"](args.stage, cfg, args)
        except Exception:
            c.write_json(Path(args.out) / "FAILED.json", {"step": step, "traceback": traceback.format_exc()})
            raise
        c.write_json(Path(args.out) / "timing" / f"step_{args.stage}_{step}.json", {"seconds": time.time() - t0})
    c.write_json(Path(args.out) / "DONE.json", {"stage": args.stage, "steps": args.steps, "at": time.time()})


if __name__ == "__main__":
    main()
