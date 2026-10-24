"""Every number in the RAID v8 manuscript, computed from the collected run outputs only.

    python3 paper/v8_numbers.py          (repository root)  ->  paper/v8_numbers.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "artifacts/raid_v8/final_20261023/collection/v8_final"
DEV = ROOT / "artifacts/raid_v8/dev_20261022/collection/v8_dev"
ENCODERS = ("mae", "dinov2", "siglip")
UPDATES = (0, 10, 25, 50, 100, 200)
BOOT = 10_000


def units(folder: Path) -> list[dict]:
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(folder / "*.json")))]


def at(curve, n, key="mse"):
    return next(p[key] for p in curve if p["updates"] == n)


def per_task(rows, fn):
    out = {}
    for u in rows:
        out.setdefault(u["task"], []).append(fn(u))
    return {t: float(np.mean(v)) for t, v in out.items()}


def boot_ci(values, seed=0):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (BOOT, len(values)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def offline(folder: Path) -> dict:
    out = {}
    for enc in ENCODERS:
        rows = units(folder / "offline" / enc)
        zero = {k: per_task(rows, lambda u, k=k: u["zero"][k]["mse"]) for k in rows[0]["zero"]}
        zero_r2 = {k: float(np.mean([u["zero"][k]["r2"] for u in rows])) for k in rows[0]["zero"]}
        adapted = {}
        keys = list(rows[0]["adapted"])
        for key in keys:
            method = key.split("@")[0]
            if sum(k.split("@")[0] == method for k in keys) > 1:  # development grid: keep every learning rate
                method = key
            adapted[method] = {
                "lr": float(key.split("@")[1]),
                "mse": {n: float(np.mean(list(per_task(rows, lambda u, n=n, key=key: at(u["adapted"][key], n)).values())))
                        for n in UPDATES},
                "gpu_seconds": {n: float(np.mean([at(u["adapted"][key], n, "gpu_seconds") for u in rows])) for n in UPDATES},
                "per_task_25": per_task(rows, lambda u, key=key: at(u["adapted"][key], 25))}
        out[enc] = {"units": len(rows), "tasks": len(zero["knn"]),
                    "zero_mean": {k: float(np.mean(list(v.values()))) for k, v in zero.items()},
                    "zero_r2": zero_r2, "zero_per_task": zero, "adapted": adapted}
    return out


def main():
    res = json.loads((FINAL / "final_results.json").read_text())
    amend = json.loads((ROOT / "artifacts/raid_v8/final_20261023/amendment.json").read_text())
    final = offline(FINAL)
    for enc in ENCODERS:
        final[enc]["tests"] = res["encoders"][enc]["tests"]
        z = final[enc]["zero_per_task"]
        for other in ("knn", "v7", "regent_blend"):
            d = np.array([z["v8_c25"][t] - z[other][t] for t in z["knn"]])
            final[enc][f"v8_minus_{other}"] = {"mean": float(d.mean()), "ci95": boot_ci(d), "tasks_better": int((d < 0).sum()),
                                               "worst": float(d.max())}
    enc_rows = units(FINAL / "encoder")
    encoder_tuning = {k: {"lr": float(k.split("@")[1]),
                          "mse": {n: float(np.mean([at(u["runs"][k]["curve"], n) for u in enc_rows])) for n in (0, 25, 50, 100, 200)},
                          "gpu_seconds": {n: float(np.mean([at(u["runs"][k]["curve"], n, "gpu_seconds") for u in enc_rows]))
                                          for n in (0, 25, 50, 100, 200)},
                          "trainable_params": enc_rows[0]["runs"][k]["trainable_params"]}
                      for k in enc_rows[0]["runs"]}
    # closed loop
    cl = units(FINAL / "closed_loop")
    methods = list(cl[0]["methods"])
    success = {m: {u["task"]: u["methods"][m]["success_rate"] for u in cl} for m in methods}
    suites = sorted({t.split("/")[0] for t in success["knn"]})
    closed = {"episodes_per_task": len(cl[0]["methods"]["knn"]["episodes"]), "tasks": len(cl),
              "gripper": cl[0]["methods"]["knn"]["gripper"],
              "mean": {m: float(np.mean(list(v.values()))) for m, v in success.items()},
              "ci95": {m: boot_ci(list(v.values())) for m, v in success.items()},
              "by_suite": {s: {m: float(np.mean([x for t, x in v.items() if t.startswith(s)])) for m, v in success.items()}
                           for s in suites}}
    for a, b in (("raid_v8@25", "head_only@25"), ("v8_c25_zero", "v7_zero"), ("raid_v8@25", "knn"), ("raid_v8@25", "lora@200")):
        d = np.array([success[a][t] - success[b][t] for t in success["knn"]])
        closed[f"{a} minus {b}"] = {"mean": float(d.mean()), "ci95": boot_ci(d), "tasks_better": int((d > 0).sum()),
                                    "tasks_worse": int((d < 0).sum())}
    # offline error of the same (task, method) pairs, MAE, seed 42 -> rank correlation with success
    mae42 = {u["task"]: u for u in units(FINAL / "offline" / "mae") if u["seed"] == 42}
    enc42 = {u["task"]: u for u in enc_rows}
    lookup = {"knn": lambda t: mae42[t]["zero"]["knn"]["mse"], "v7_zero": lambda t: mae42[t]["zero"]["v7"]["mse"],
              "v8_c25_zero": lambda t: mae42[t]["zero"]["v8_c25"]["mse"],
              "head_only@25": lambda t: at(next(v for k, v in mae42[t]["adapted"].items() if k.startswith("head_only")), 25),
              "raid_v8@25": lambda t: at(next(v for k, v in mae42[t]["adapted"].items() if k.startswith("raid_v8")), 25),
              "lora@200": lambda t: at(next(v["curve"] for k, v in enc42[t]["runs"].items() if k.startswith("lora")), 200),
              "full@200": lambda t: at(next(v["curve"] for k, v in enc42[t]["runs"].items() if k.startswith("full")), 200)}
    xs, ys, within = [], [], []
    for t in success["knn"]:
        e = [lookup[m](t) for m in methods]; s = [success[m][t] for m in methods]
        xs += e; ys += s
        if len(set(s)) > 1:
            within.append(spearman(np.array(e), -np.array(s)))
    closed["offline_vs_success"] = {"spearman_all_pairs": spearman(np.array(xs), -np.array(ys)), "pairs": len(xs),
                                    "mean_within_task_spearman": float(np.mean(within)), "tasks_used": len(within),
                                    "note": "positive = lower offline error goes with higher success"}
    closed["method_offline_mae"] = {m: float(np.mean([lookup[m](t) for t in success["knn"]])) for m in methods}
    timing = json.loads((FINAL / "timing/methods_primary.json").read_text())
    steps = {Path(f).stem.replace("step_final_", ""): json.loads(Path(f).read_text())["seconds"]
             for f in glob.glob(str(FINAL / "timing/step_final_*.json"))}
    dev_steps = {Path(f).stem.replace("step_development_", ""): json.loads(Path(f).read_text())["seconds"]
                 for f in glob.glob(str(DEV / "timing/step_development_*.json"))}
    dev = offline(DEV)
    dev_cl = units(DEV / "closed_loop")
    dev_closed = {m: float(np.mean([u["methods"][m]["success_rate"] for u in dev_cl])) for m in dev_cl[0]["methods"]}
    sources = [json.loads(Path(f).read_text()) for f in glob.glob(str(DEV / "sources/*/*.json"))]
    out = {"amendment_digest": amend["digest"], "selection": amend["selection"], "final": final,
           "encoder_tuning": encoder_tuning, "closed_loop": closed, "timing": timing, "final_steps_seconds": steps,
           "cache_final": json.loads((FINAL / "timing/cache_final.json").read_text()),
           "cache_dev": json.loads((DEV / "timing/cache_development.json").read_text()),
           "dev": {"zero_mean": {e: dev[e]["zero_mean"] for e in ENCODERS},
                   "adapted_25_selected_lr": {e: {m: dev[e]["adapted"][f"{m}@{lr:g}"]["mse"][25]
                                                  for m, lr in amend["selection"]["head_lrs"][e].items()} for e in ENCODERS},
                   "closed_loop_raw_gripper": dev_closed, "steps_seconds": dev_steps,
                   "source_models": len(sources), "source_seconds_mean": float(np.mean([s["seconds"] for s in sources])),
                   "source_best_val": {f"{s['kind']}_c{s['corrupt']}_s{s['seed']}": s["best_score"] for s in sources}},
           "data_files_final": len(json.loads((FINAL / "data_sha256_final.json").read_text())),
           "data_files_dev": len(json.loads((DEV / "data_sha256_development.json").read_text()))}
    (ROOT / "paper/v8_numbers.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("wrote paper/v8_numbers.json")


if __name__ == "__main__":
    main()
