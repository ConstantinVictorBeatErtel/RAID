"""FINAL_RESULTS.md and final_results.html from the verified final collection, exactly as the frozen amendment
says (P1, P2, P3; cluster bootstrap; settings won). Standard library only; no network; nothing is re-selected."""
import json, statistics as st, sys
from pathlib import Path
P = Path(__file__).resolve().parent; ROOT = P.parents[2]; sys.path.insert(0, str(ROOT))
from raid_experiments import hybrid_bench_analysis as a
from raid_experiments.hybrid_study import _sha_file

amend = json.loads((P / "final_analysis_amendment.json").read_text())
units = a.load_units(P / "collection/bench")
assert len(units) == 6 and all(u["final_freeze_sha256"] == _sha_file(P / "final_analysis_amendment.json") for u in units)
L = a.LABELS
pc = amend["primary_comparisons"]
curves, ad, dc, zero = a.curves(units), a.adaptation(units), a.decision(units), a.zero_table(units)
zmean = {k: m for k, m, _ in zero}
upds = [p[3] for p in curves["lora_mlp"]]
order = sorted(curves, key=lambda k: curves[k][-1][1])


def verdict(r):
    lo, hi = r["cluster_ci"]
    return "better" if hi < 0 else ("worse" if lo > 0 else "no clear difference")


rows = []
tf = pc["P1"]["best_training_free_baseline"]
rows.append(("P1", f"RAID v7 (no training) vs {L[tf]}", a.paired(a.per_demo(units, "raid_bank_v7"), a.per_demo(units, tf))))
for n in (25, 500):
    ours = pc["P2"]["best_raid_variant"][str(n)]
    for b in pc["P2"]["baselines"]:
        other = a.per_demo(units, b) if b in units[0]["zero_adaptation"] and b not in units[0]["curves"] else a.per_demo(units, b, n)
        rows.append(("P2", f"{L[ours]} vs {L[b]}, {n} updates", a.paired(a.per_demo(units, ours, n), other)))
secondary = [("RAID v7 vs " + L[k] + " (no training)", a.paired(a.per_demo(units, "raid_bank_v7"), a.per_demo(units, k)))
             for k in ("support_knn_average", "retrieve_and_play", "regent_blend") if k != tf]
for n in (25, 500):
    for k in ("raid_lora_v7", "raid_lora_mlp", "raid_v7_head"):
        if k != pc["P2"]["best_raid_variant"][str(n)]:
            secondary.append((f"{L[k]} vs LoRA, {n} updates", a.paired(a.per_demo(units, k, n), a.per_demo(units, "lora_mlp", n))))
target = curves["lora_mlp"][-1][1]


def at_budget(k, budget):
    if k in zmean and k not in curves:
        return zmean[k]
    ok = [e for s, e, _, _ in curves[k] if s <= budget]
    return ok[-1] if ok else None


def seconds_to(points, goal):
    for (s0, e0, _, _), (s1, e1, _, _) in zip(points, points[1:]):
        if e0 <= goal:
            return s0
        if e1 <= goal:
            return s0 + (e0 - goal) / (e0 - e1) * (s1 - s0)
    return None


budgets = pc["P3"]["fixed_budgets_gpu_seconds"]
everything = order + [k for k in zmean if k not in curves]
dev = {(u["task"], u["seed"]): u for u in a.load_units(P.parent / "compute_bench_baselines_20261017/collection/bench")}
dev_curves = a.curves(list(dev.values()))
v7h = a.load_units(P.parent / "compute_bench_v7head_20261017/collection/bench")
dev_curves["raid_v7_head"] = a.curves(v7h)["raid_v7_head"]
dev_zero = {k: m for k, m, _ in a.zero_table(list(dev.values()))}
pair_line = lambda tag, n, r: a.fmt_pair(f"{tag}: {n}", r)[:-1] + f"| {verdict(r)} |"
md = ["# Final-task offline benchmark (one frozen run)", "",
      f"Run once on the 2 sealed LIBERO-Goal final tasks x registered seeds 42/43/44 (6 settings), five-shot, 20 query demos per task with real recorded future frames. Analysis exactly as frozen in amendment {amend['digest'][:12]} (written before any final data was read). Nothing was tuned after this run.",
      "held_out_claims_eligible = false: the tasks were never used for design, but the VPP2 generator saw LIBERO in training (VPP2 is not used here).",
      "Metrics: chunk MSE on normalized actions (lower is better), macro over query demos; R2 (higher is better).", "",
      "## Headline comparisons (frozen P1 and P2)", "",
      "Difference = ours minus baseline (negative = ours better). Primary interval: 95% bootstrap over the 6 task x seed clusters. Verdict rule fixed in advance.", "",
      "| Comparison (ours minus baseline) | Mean | 95% CI, task x seed clusters | 95% CI, demos | Settings won | Verdict |",
      "|---|---|---|---|---|---|"] + [pair_line(t, n, r) for t, n, r in rows] + \
     ["", "## Without any target training", "", "| Rank | Method | MSE | R2 | Development MSE |", "|---|---|---|---|---|"] + \
     [f"| {i+1} | {L[k]} | {m:.3f} | {r:.3f} | {dev_zero.get(k, float('nan')):.3f} |" for i, (k, m, r) in enumerate(zero)] + \
     ["", "## With target training: error after N updates", "",
      "| Method | " + " | ".join(str(u) for u in upds) + " | R2 at 500 | Dev. MSE at 500 | GPU s (500 updates) | GFLOP / update | Trainable params |",
      "|---|" + "---|" * (len(upds) + 5)] + \
     [f"| {L[k]} | " + " | ".join(f"{p[1]:.3f}" for p in curves[k]) + f" | {curves[k][-1][2]:.3f} | "
      f"{dev_curves[k][-1][1]:.3f} | {ad[k][0]:.1f} | {ad[k][1] / 1e9:.1f} | {ad[k][2]:,} |" for k in order] + \
     ["", "## P3: error per GPU-second", "", f"Error at fixed adaptation budgets (last checkpoint within the budget) and GPU-seconds to reach LoRA's 500-update error ({target:.3f}).", "",
      "| Method | " + " | ".join(f"MSE at {b:g} s" for b in budgets) + " | GPU s to target |", "|---|" + "---|" * (len(budgets) + 1)] + \
     [f"| {L[k]} | " + " | ".join("-" if (v := at_budget(k, b)) is None else f"{v:.3f}" for b in budgets) + " | "
      + ("0 (no training)" if k not in curves and zmean[k] <= target else
         ("not reached" if k not in curves or (t := seconds_to(curves[k], target)) is None else f"{t:.1f}")) + " |" for k in everything] + \
     ["", "## Per setting (descriptive): no training and 25 updates", "",
      "| Task | Seed | Frozen source | RAID v7 | VINN-style k-NN | REGENT-style blend | RAID+LoRA v7, 25 | LoRA, 25 | RAID v7 head-only, 25 |",
      "|---|---|---|---|---|---|---|---|---|"] + \
     [f"| {u['task'].replace('_', ' ')} | {u['seed']} | " + " | ".join(f"{u['zero_adaptation'][k]['mse_macro']:.3f}" for k in
      ("source_frozen", "raid_bank_v7", "support_knn_average", "regent_blend")) + " | " +
      " | ".join(f"{u['curves'][k][1]['mse_macro']:.3f}" for k in ("raid_lora_v7", "lora_mlp", "raid_v7_head")) + " |" for u in units] + \
     ["", "## Reading", "",
      "- P1 failed: without training, RAID v7 (0.500) is worse than VINN-style k-NN (0.376); verdict worse. On the bowl task RAID v7 is better than k-NN in 3/3 seeds (0.306-0.324 vs 0.319-0.374); on the wine-bottle task it is much worse in 3/3 (0.617-0.716 vs 0.353-0.456). On that task the frozen source controller is very poor, and the learned gate still relies on it; plain retrieval (k-NN) does not. On development, RAID v7 was the best training-free method (0.409); this did not transfer.",
      "- P2 at 25 updates held: RAID+LoRA v7 (0.272) is better than every baseline in 6/6 settings, all intervals below 0 (for example LoRA -0.044, full fine-tuning -0.042, head-only -0.049, k-NN -0.105).",
      "- P2 at 500 updates: no clear difference against head-only, LoRA, LoRA without source training and the fresh-head probe; worse than full fine-tuning (+0.013, 1/6); better than STRAP-style and all training-free baselines.",
      "- P3: the cheapest good point is RAID v7 head-only: 0.277 after 0.5 GPU-seconds (it reaches LoRA's 500-update error in about 0.1 s); plain head-only needs about 1 s. LoRA-based methods need about 10 s (RAID+LoRA v7) to 190 s (LoRA). Full fine-tuning has the lowest error at 100 s (0.266) but 13.6 TFLOP per update.",
      "- The development-to-final pattern for trained methods held (fast early gain from retrieval, convergence by 500 updates); the zero-training result did not.", ""] + \
     ["", "## Secondary comparisons (descriptive)", ""] + a.PAIR_HEADER + [a.fmt_pair(n, r) for n, r in secondary] + \
     ["", "## Cost per control decision (batch 1)", "", "| Method | ms | GFLOP |", "|---|---|---|"] + \
     [f"| {L[k]} | {v[0]:.1f} | {v[1]:.0f} |" for k, v in sorted(dc.items(), key=lambda kv: kv[1][0])] + \
     ["", "## Limits", "",
      "Offline recorded-transition action error, not task success. Two final tasks and three seeds: the cluster interval rests on 6 clusters; a percentile bootstrap over so few clusters can be too narrow, so read settings won next to it. RAID v7 head-only and RAID v7 were defined after development results (development was used for design; final tasks were not). No multiplicity correction.", ""]
(P / "FINAL_RESULTS.md").write_text("\n".join(md))
short = lambda n: n.replace("RAID+LoRA v7 (ours) vs ", "vs ").replace(" (no training)", "").replace("STRAP-style retrieved training data", "STRAP-style").replace("Frozen-encoder probe (fresh head)", "fresh-head probe").replace(" updates", " upd.")
bars = [(f"{t} " + short(n), r["mean"], *r["cluster_ci"]) for t, n, r in rows]
html_lines = md[:5] + ["## Error vs adaptation GPU-seconds",
                       a.svg_curves(curves, {"raid_bank_v7": zmean["raid_bank_v7"], tf: zmean[tf]}, ymin=0.2,
                                    ymax=max(0.8, max(p[1] for c in curves.values() for p in c) + 0.05)),
                       "## Frozen comparisons: 95% cluster intervals", a.svg_bars(bars, "ours minus baseline")] + md[5:]
(P / "final_results.html").write_text(a.page("RAID final benchmark", html_lines))
json.dump({"primary": [{"tag": t, "comparison": n, **r, "verdict": verdict(r)} for t, n, r in rows],
           "secondary": {n: r for n, r in secondary}, "mean_curve": curves, "zero": zero,
           "amendment_digest": amend["digest"], "held_out_claims_eligible": False},
          open(P / "final_analysis.json", "w"), indent=2, default=list)
print("written")
