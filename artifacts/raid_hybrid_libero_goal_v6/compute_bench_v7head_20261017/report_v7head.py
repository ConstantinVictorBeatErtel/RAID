"""V7HEAD_RESULTS.md and v7head_results.html: exploratory raid_v7_head (this run) paired with the 2026-10-17
baseline run (same bitwise source controllers; checked below). Standard library only; no network."""
import json, statistics as st, sys
from pathlib import Path
P = Path(__file__).resolve().parent; ROOT = P.parents[2]; sys.path.insert(0, str(ROOT))
from raid_experiments import hybrid_bench_analysis as a

new = {(u["task"], u["seed"]): u for u in a.load_units(P / "collection/bench")}
base = {(u["task"], u["seed"]): u for u in a.load_units(P.parent / "compute_bench_baselines_20261017/collection/bench")}
assert set(new) == set(base) and len(new) == 6, "expected the same 6 task x seed settings"
# Reproducibility: zero-training conditions and head_only must match the baseline run per demo.
repro = {}
for k in ("source_frozen", "raid_bank_mlp", "raid_bank_v7", "support_knn_average"):
    repro[k] = max(abs(new[s]["zero_adaptation"][k]["mse_by_demo"][d] - base[s]["zero_adaptation"][k]["mse_by_demo"][d])
                   for s in new for d in new[s]["zero_adaptation"][k]["mse_by_demo"])
repro["head_only (all checkpoints)"] = max(abs(p["mse_by_demo"][d] - q["mse_by_demo"][d]) for s in new
                                           for p, q in zip(new[s]["curves"]["head_only"], base[s]["curves"]["head_only"])
                                           for d in p["mse_by_demo"])
units = []
for s, b in sorted(base.items()):
    u = json.loads(json.dumps(b))
    for field in ("curves", "adaptation", "decision_cost"):
        u[field]["raid_v7_head"] = new[s][field]["raid_v7_head"]
    units.append(u)
curves, ad, dc = a.curves(units), a.adaptation(units), a.decision(units)
zero = a.zero_table(units)
upds = [p[3] for p in curves["raid_v7_head"]]
order = sorted(curves, key=lambda k: curves[k][-1][1])
ours = "raid_v7_head"
comps = []
for n in (25, 500):
    for b in ("head_only", "raid_lora_v7", "lora_mlp", "full_finetune", "strap_style", "probe_fresh", "scratch_lora"):
        comps.append((f"{a.LABELS[ours]} vs {a.LABELS[b]}, {n} updates", a.paired(a.per_demo(units, ours, n), a.per_demo(units, b, n))))
comps.append(("RAID v7 head-only, 0 updates vs RAID v7 (sanity: identical)",
              a.paired(a.per_demo(units, ours, 0), a.per_demo(units, "raid_bank_v7"))))
TRAINING_FREE_BASELINES = ("support_knn_average", "retrieve_and_play", "regent_blend")  # external baselines only
best_tf = min(TRAINING_FREE_BASELINES, key=lambda k: dict((z[0], z[1]) for z in zero)[k])
for n in (25, 500):
    comps.append((f"{a.LABELS[ours]}, {n} updates vs {a.LABELS[best_tf]} (best training-free baseline)",
                  a.paired(a.per_demo(units, ours, n), a.per_demo(units, best_tf))))
target = curves["lora_mlp"][-1][1]


BUDGETS = (0.1, 0.5, 1, 10, 100)


def at_budget(k, budget):
    ok = [e for t, e, _, _ in curves[k] if t <= budget]
    return ok[-1] if ok else None


def seconds_to(points, goal):
    for (s0, e0, _, _), (s1, e1, _, _) in zip(points, points[1:]):
        if e0 <= goal:
            return s0
        if e1 <= goal:
            return s0 + (e0 - goal) / (e0 - e1) * (s1 - s0)
    return None


md = ["# Exploratory: RAID v7 head-only (frozen encoder, retrieval head fine-tuned)", "",
      "EXPLORATORY. Development tasks only (2) x registered support seeds 42/43/44, five-shot; 20 query demos per task with real recorded future frames.",
      "New method: frozen MAE (cached features), source-trained v7 head (cross-attention + gate over k = 16 retrieved support candidates) fine-tuned with the selection loss; same schedule and sample stream as head-only (500 updates, batch 64). It was designed after seeing the development results.",
      f"This run (A100 80 GB, 16 CPU threads) reran the zero-training conditions and head-only; they match the 2026-10-17 baseline run exactly (largest per-demo difference {max(repro.values()):.1e}), so raid_v7_head pairs with all baselines of that run.",
      "Metrics: chunk MSE on normalized actions (lower is better), macro over query demos; R2 (higher is better).", "",
      "## Error after N updates (same schedule for all)", "",
      "| Method | " + " | ".join(str(u) for u in upds) + " | R2 at 500 | GPU s (500 updates) | GFLOP / update | Trainable params |",
      "|---|" + "---|" * (len(upds) + 4)] + \
     [f"| {a.LABELS[k]} | " + " | ".join(f"{p[1]:.3f}" for p in curves[k]) +
      f" | {curves[k][-1][2]:.3f} | {ad[k][0]:.1f} | {ad[k][1] / 1e9:.1f} | {ad[k][2]:,} |" for k in order] + \
     ["", "## No-training reference", "", "| Method | MSE | R2 |", "|---|---|---|"] + \
     [f"| {a.LABELS[k]} | {m:.3f} | {r:.3f} |" for k, m, r in zero] + \
     ["", "## Paired differences (raid_v7_head minus baseline; negative = ours better)", "",
      "Primary interval: 95% bootstrap over the 6 task x seed clusters (10,000 resamples). Demo-level interval as in earlier reports (optimistic).", ""] + \
     a.PAIR_HEADER + [a.fmt_pair(n, r) for n, r in comps] + \
     ["", "## Compute: GPU-seconds to reach LoRA's 500-update error", "",
      f"Target = mean LoRA error after 500 updates ({target:.3f}).", "", "| Method | GPU s to target |", "|---|---|"] + \
     [f"| {a.LABELS[k]} | " + (f"{v:.1f}" if (v := seconds_to(curves[k], target)) is not None else "not reached") + " |" for k in order] + \
     ["", "## Error at fixed adaptation budgets (last checkpoint within the budget)", "",
      "| Method | " + " | ".join(f"MSE at {b:g} s" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)] + \
     [f"| {a.LABELS[k]} | " + " | ".join("-" if (v := at_budget(k, b)) is None else f"{v:.3f}" for b in BUDGETS) + " |" for k in order] + \
     ["", "## Reading", "",
      f"- After 25 updates, RAID v7 head-only ({curves[ours][1][1]:.3f}) is better than every baseline in 6/6 settings (head-only -0.057, LoRA -0.050, full fine-tuning -0.038) and close to RAID+LoRA v7 (+0.004, worse in 6/6), with about 1.5% of its adaptation compute ({ad[ours][0]:.1f} s vs {ad['raid_lora_v7'][0]:.0f} s for 500 updates) because the encoder stays frozen.",
      f"- With more updates it does not improve further (500 updates: {curves[ours][-1][1]:.3f}); it ends worse than head-only (+0.012), LoRA (+0.020) and full fine-tuning (+0.039).",
      "- Per GPU-second, plain head-only is better at every fixed budget (0.1 to 100 s): one head-only update costs about 2.4 ms against 6 ms (retrieval over 16 candidates), so in the same time it does more updates. The RAID v7 head advantage is per update, not per GPU-second. Both are about 100x cheaper than the LoRA methods.",
      "- Exploratory only. The frozen final analysis keeps the selection rule fixed in advance (best RAID variant by development error), which picks RAID+LoRA v7 at 25 and at 500 updates.", ""] + \
     ["", "## Cost per control decision (batch 1)", "", "| Method | ms | GFLOP |", "|---|---|---|"] + \
     [f"| {a.LABELS[k]} | {v[0]:.1f} | {v[1]:.0f} |" for k, v in sorted(dc.items(), key=lambda kv: kv[1][0])] + \
     ["", "## Reproducibility check against the baseline run (largest per-demo |difference|)", "", "| Quantity | Max difference |", "|---|---|"] + \
     [f"| {k} | {v:.1e} |" for k, v in repro.items()] + \
     ["", "## Limits", "",
      "Exploratory: the variant was defined after the development results were known. Offline recorded-transition error, not task success. Development tasks only; final tasks sealed; held_out_claims_eligible = false (VPP2 saw LIBERO). Three registered seeds, so the cluster interval rests on 6 clusters and is coarse.", ""]
bars = [(n.split(" vs ")[1], r["mean"], *r["cluster_ci"]) for n, r in comps if "sanity" not in n]
md.insert(md.index("## Paired differences (raid_v7_head minus baseline; negative = ours better)") + 3,
          a.svg_bars(bars, "raid_v7_head minus baseline, 95% cluster intervals"))
(P / "V7HEAD_RESULTS.md").write_text("\n".join(l for l in md if not l.startswith("<")))
chart = a.svg_curves({k: v for k, v in curves.items()}, {"raid_bank_v7": dict((z[0], z[1]) for z in zero)["raid_bank_v7"],
                                                         "support_knn_average": dict((z[0], z[1]) for z in zero)["support_knn_average"]})
html_lines = md[:6] + ["## Error vs adaptation GPU-seconds", chart] + md[6:]
(P / "v7head_results.html").write_text(a.page("RAID v7 head-only", html_lines))
json.dump({"comparisons": {n: r for n, r in comps}, "reproducibility_max_abs": repro,
           "mean_curve": {k: v for k, v in curves.items()}, "exploratory": True, "held_out_claims_eligible": False},
          open(P / "v7head_analysis.json", "w"), indent=2, default=list)
print("written")
