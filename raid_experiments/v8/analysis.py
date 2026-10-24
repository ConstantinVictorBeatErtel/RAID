"""RAID v8 analysis: development selection (writes the final configuration) and the pre-registered final tests.

    python -m raid_experiments.v8.analysis select --dev OUT_DIR --config DEV_CONFIG --write AMENDMENT.json
    python -m raid_experiments.v8.analysis final --final OUT_DIR --amendment AMENDMENT.json --write RESULTS.json

Selection rules (fixed before the development run):
  * v8 variant: the source head with the lower mean zero-training MSE on development tasks (MAE, all seeds).
  * training-free baseline for H2: k-NN or REGENT blend, lower mean development MSE (MAE).
  * learning rate of each head method / encoder: lowest mean MSE after 25 updates (head methods) or 200 updates
    (LoRA, full fine-tuning) on development tasks.
Final tests: per task, the paired difference is averaged over seeds; 95% percentile bootstrap over tasks
(10,000 resamples, seed 0); Holm correction over H1-H3 at alpha = 0.05.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

HEAD_BUDGET, ENCODER_BUDGET = 25, 200
BOOT, BOOT_SEED, ALPHA = 10_000, 0, 0.05


def _units(folder: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]


def at(curve: list[dict], updates: int) -> float:
    return next(p["mse"] for p in curve if p["updates"] == updates)


def per_task(units: list[dict], value) -> dict[str, float]:
    """Mean over seeds of value(unit) for each task."""
    out: dict[str, list[float]] = {}
    for u in units:
        out.setdefault(u["task"], []).append(value(u))
    return {t: float(np.mean(v)) for t, v in out.items()}


def bootstrap(diffs: np.ndarray) -> tuple[float, float, float]:
    """Mean, and the 2.5% / 97.5% percentiles of the bootstrap mean, plus a one-sided p (share of means >= 0)."""
    rng = np.random.default_rng(BOOT_SEED)
    means = diffs[rng.integers(0, len(diffs), (BOOT, len(diffs)))].mean(1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(
        (means >= 0).mean())


def compare(a: dict, b: dict, label: str) -> dict:
    tasks = sorted(set(a) & set(b))
    diffs = np.array([a[t] - b[t] for t in tasks])
    mean, lo, hi, p = bootstrap(diffs)
    return {"comparison": label, "tasks": len(tasks), "mean_diff": mean, "ci95": [lo, hi], "p_one_sided": p,
            "tasks_better": int((diffs < 0).sum()), "worst_task_diff": float(diffs.max()),
            "per_task": {t: float(x) for t, x in zip(tasks, diffs)}}


def holm(results: list[dict]) -> None:
    order = sorted(range(len(results)), key=lambda i: results[i]["p_one_sided"])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (len(results) - rank) * results[i]["p_one_sided"])
        running = max(running, adj)
        results[i]["p_holm"] = running
        results[i]["verdict"] = "supported" if running < ALPHA and results[i]["ci95"][1] < 0 else "not supported"


def select(dev: Path, cfg: dict) -> dict:
    hb, eb = cfg.get("select_head_budget", HEAD_BUDGET), cfg.get("select_encoder_budget", ENCODER_BUDGET)
    off = _units(dev / "offline" / "mae")
    zero = {k: per_task(off, lambda u, k=k: u["zero"][k]["mse"]) for k in off[0]["zero"]}
    v8_labels = [label for label, _ in cfg["source_heads"] if label.startswith("v8")]
    v8 = min(v8_labels, key=lambda label: np.mean(list(zero[label].values())))
    baseline = min(("knn", "regent_blend"), key=lambda k: np.mean(list(zero[k].values())))
    lrs = {}
    for enc in cfg["encoders"]:
        units = _units(dev / "offline" / enc)
        for method, _, _ in cfg["head_methods"]:
            scores = {lr: np.mean([at(u["adapted"][f"{method}@{lr:g}"], hb) for u in units])
                      for lr in cfg["head_lrs"][method]}
            lrs.setdefault(enc, {})[method] = float(min(scores, key=scores.get))
    enc_units = _units(dev / "encoder")
    enc_lrs = {}
    for mode in ("lora", "full"):
        scores = {lr: np.mean([at(u["runs"][f"{mode}@{lr:g}"]["curve"], eb) for u in enc_units])
                  for lr in cfg["encoder_lrs"][mode]}
        enc_lrs[mode] = float(min(scores, key=scores.get))
    v8_method = "raid_" + v8
    final = {
        "encoders": cfg["encoders"], "seeds": cfg["seeds"], "source_updates": cfg["source_updates"],
        "source_heads": [h for h in cfg["source_heads"] if h[0] in ("v7", v8)], "base_head": v8,
        "head_methods": [["head_only", v8, False], ["raid_v7", "v7", True], ["raid_v8", v8, True]],
        "head_lrs": {enc: {"head_only": [lrs[enc]["head_only"]], "raid_v7": [lrs[enc]["raid_v7"]],
                           "raid_v8": [lrs[enc][v8_method]]} for enc in cfg["encoders"]},
        "encoder_seed": cfg["encoder_seed"], "encoder_lrs": {m: [enc_lrs[m]] for m in enc_lrs},
        "encoder_head_lr": cfg["encoder_head_lr"],
        # gripper "binary": executed gripper command rounded to open/closed for every method (development decision,
        # 23 Oct 2026: regression outputs in-between values that fail to grasp; local test 0.3->0.6 head-only, 0.9->1.0 k-NN)
        "closed_loop": {"seed": 42, "episodes": 20, "gripper": "binary",
                        "adapted": [["head_only", v8, False, lrs["mae"]["head_only"], HEAD_BUDGET],
                                    ["raid_v8", v8, True, lrs["mae"][v8_method], HEAD_BUDGET]],
                        "encoder": [["lora", enc_lrs["lora"], ENCODER_BUDGET], ["full", enc_lrs["full"], ENCODER_BUDGET]]},
        "hypotheses": {"H1": ["zero", v8, "zero", "v7"], "H2": ["zero", v8, "zero", baseline],
                       "H3": ["adapted", f"raid_v8@{HEAD_BUDGET}", "adapted", f"head_only@{HEAD_BUDGET}"]},
        "selection": {"v8_variant": v8, "training_free_baseline": baseline,
                      "dev_zero_mse_mae": {k: float(np.mean(list(v.values()))) for k, v in zero.items()},
                      "head_lrs": lrs, "encoder_lrs": enc_lrs},
        "analysis": {"bootstrap": BOOT, "seed": BOOT_SEED, "alpha": ALPHA, "correction": "Holm over H1-H3",
                     "unit": "task (mean over seeds)", "primary_encoder": "mae"},
    }
    return final


def freeze(body: dict) -> dict:
    """Bind the final configuration to the v8 code and seal it (bench.target_suites checks the digest)."""
    import hashlib
    from .bench import config_digest
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    body = {**body, "stage": "final", "status": "frozen", "code_sha256": code,
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    body["digest"] = config_digest(body)
    return body


def _value(unit: dict, kind: str, name: str):
    if kind == "zero":
        return unit["zero"][name]["mse"]
    method, budget = name.split("@")
    curve = next(v for k, v in unit["adapted"].items() if k.startswith(method + "@"))
    return at(curve, int(budget))


def final_tests(final: Path, amend: dict) -> dict:
    out = {"generated_at": time.time(), "amendment_digest": amend["digest"], "encoders": {}}
    for enc in amend["encoders"]:
        units = _units(final / "offline" / enc)
        results = []
        for name, (ka, a, kb, b) in amend["hypotheses"].items():
            va = per_task(units, lambda u: _value(u, ka, a))
            vb = per_task(units, lambda u: _value(u, kb, b))
            results.append({"hypothesis": name, **compare(va, vb, f"{a} minus {b}")})
        holm(results)
        means = {}
        for k in units[0]["zero"]:
            means[f"zero/{k}"] = float(np.mean(list(per_task(units, lambda u, k=k: u["zero"][k]["mse"]).values())))
        for k, curve in units[0]["adapted"].items():
            for p in curve:
                means[f"{k.split('@')[0]}@{p['updates']}"] = float(np.mean(list(
                    per_task(units, lambda u, k=k, n=p["updates"]: at(u["adapted"][k], n)).values())))
        out["encoders"][enc] = {"tests": results, "means": means, "units": len(units)}
    enc_units = _units(final / "encoder")
    if enc_units:
        out["encoder_tuning"] = {k: {str(p["updates"]): float(np.mean([at(u["runs"][k]["curve"], p["updates"])
                                                                     for u in enc_units]))
                                     for p in enc_units[0]["runs"][k]["curve"]} for k in enc_units[0]["runs"]}
    cl_units = _units(final / "closed_loop")
    if cl_units:
        out["closed_loop"] = {m: float(np.mean([u["methods"][m]["success_rate"] for u in cl_units]))
                              for m in cl_units[0]["methods"]}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("select", "final"))
    p.add_argument("--dev"); p.add_argument("--final"); p.add_argument("--config"); p.add_argument("--amendment")
    p.add_argument("--write", required=True)
    a = p.parse_args(argv)
    if a.mode == "select":
        body = select(Path(a.dev), json.loads(Path(a.config).read_text()))
        body = freeze(body)
    else:
        body = final_tests(Path(a.final), json.loads(Path(a.amendment).read_text()))
    Path(a.write).write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    print(a.write)


if __name__ == "__main__":
    main()
