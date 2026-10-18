"""Write the frozen final-analysis amendment (JSON + Markdown) from DEVELOPMENT results only.

Runs before any final-task file is hashed, packaged or read. Refuses to overwrite an existing amendment.
Selections use the development means of the baseline run (2026-10-17) and the exploratory v7head run."""
import hashlib, json, statistics as st, sys
from datetime import datetime, timezone
from pathlib import Path
P = Path(__file__).resolve().parent; ROOT = P.parents[2]; sys.path.insert(0, str(ROOT))
from raid_experiments import hybrid_bench_analysis as a, hybrid_study as hs, hybrid_v7, hybrid_baselines as hb

OUT, MD = P / "final_analysis_amendment.json", P / "FINAL_ANALYSIS_PLAN.md"
if OUT.exists() or MD.exists():
    raise SystemExit("amendment exists; registered files are never overwritten")
assert not (P / "release.json").exists() and not (P / "collection").exists(), "final data already touched"
protocol = json.loads((P.parent / "vpp2_pilot_v5_20261014/admission/protocol.json").read_text())
base_amend = json.loads((P.parent / "compute_bench_baselines_20261017/bench_amendment_baselines.json").read_text())
v7h_amend = json.loads((P.parent / "compute_bench_v7head_20261017/bench_amendment_v7head.json").read_text())
dev = a.load_units(P.parent / "compute_bench_baselines_20261017/collection/bench")
v7h = {(u["task"], u["seed"]): u for u in a.load_units(P.parent / "compute_bench_v7head_20261017/collection/bench")}
assert len(dev) == 6 and len(v7h) == 6
mean_at = lambda units, k, n: st.mean(next(p for p in u["curves"][k] if p["updates"] == n)["mse_macro"] for u in units)
TRAINING_FREE = ("support_knn_average", "retrieve_and_play", "regent_blend")
TRAINED = ("head_only", "probe_fresh", "scratch_lora", "full_finetune", "strap_style", "lora_mlp")
RAID_TRAINED = ("raid_lora_v7", "raid_lora_mlp", "raid_v7_head")
zero = {k: st.mean(u["zero_adaptation"][k]["mse_macro"] for u in dev) for k in TRAINING_FREE}
best_tf = min(TRAINING_FREE, key=zero.get)
dev_means = {}
for n in (25, 500):
    dev_means[n] = {k: mean_at(dev, k, n) for k in ("raid_lora_v7", "raid_lora_mlp")}
    dev_means[n]["raid_v7_head"] = mean_at(list(v7h.values()), "raid_v7_head", n)
best_raid = {n: min(RAID_TRAINED, key=dev_means[n].get) for n in (25, 500)}
methods_zero = ["source_frozen", "raid_bank_mlp", "raid_bank_v7", *TRAINING_FREE]
methods_trained = ["lora_mlp", "raid_lora_mlp", "raid_lora_v7", *hb.TRAINED, "raid_v7_head"]
body = {
    "name": "raid_fewshot_idm_final_analysis_plan", "status": "frozen", "stage": hs.FINAL_STAGE,
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "frozen_at": datetime.now(timezone.utc).isoformat(),
    "authority": "user request 17 Oct 2026: freeze the final method set and analysis plan before any final-task data is read; one offline run on the 2 final tasks x 3 seeds; nothing tuned after it",
    "model_digest": protocol["model_digest"], "parent_protocol_digest": protocol["digest"],
    "parent_amendment_digests": {"baselines": base_amend["digest"], "v7head_exploratory": v7h_amend["digest"]},
    "held_out_claims_eligible": False,
    "held_out_claims_note": "The final tasks were never used for design or tuning, but the VPP2 generator saw LIBERO during its training. This offline benchmark does not use VPP2 (real recorded futures); the protocol flag stays false.",
    "scientific_result": False, "final_data_read_before_freeze": False,
    "data": {"tasks": "the 2 registered final_test tasks of protocol v6 (LIBERO-Goal)", "seeds": [42, 43, 44], "support": "five-shot registered support sets",
             "query": "the 20 registered query demos per task, real recorded future frames; no generator, no simulator",
             "package": "source tasks + final tasks only (no development data)", "runs": 1},
    "methods": {"zero_training": methods_zero, "trained": methods_trained,
                "exploratory_included": {"raid_v7_head": "defined after development results; included as a frozen method, labelled exploratory"},
                "baseline_definitions": base_amend["baselines"], "v7": base_amend["v7"], "raid_v7_head": v7h_amend["method"]["raid_v7_head"]},
    "settings": {"training": protocol["design"]["training"], "checkpoints_updates": [0, 25, 50, 100, 200, 350, 500],
                 "cpu_threads": 16, "gpu_class": "80 GB (A100)", "code": "raid_experiments/hybrid_compute_bench.py suite final, stage final"},
    "metrics": {**base_amend["metrics"], "unit_of_analysis": "per (task, seed, query demo) MSE; macro over demos within a task x seed setting"},
    "compute_measures": ["adaptation GPU-seconds (evaluation time excluded)", "FLOPs per update", "trainable parameters",
                         "decision latency (ms, batch 1), FLOPs and peak GPU memory per decision",
                         "one-time shared costs: reference feature cache and source training (reported, not charged to methods)"],
    "primary_comparisons": {
        "P1": {"what": "RAID v7 (raid_bank_v7) vs the best training-free baseline, no training",
               "best_training_free_baseline": best_tf, "rule": "lowest mean development MSE among " + ", ".join(TRAINING_FREE),
               "development_means": zero},
        "P2": {"what": "best RAID variant vs each baseline, at 25 updates and at 500 updates",
               "best_raid_variant": {str(n): best_raid[n] for n in (25, 500)},
               "rule": "lowest mean development MSE at that update count among " + ", ".join(RAID_TRAINED),
               "development_means": {str(n): dev_means[n] for n in (25, 500)},
               "baselines": list(TRAINED) + list(TRAINING_FREE)},
        "P3": {"what": "per-GPU-second curves: mean MSE vs adaptation GPU-seconds for every method (descriptive)",
               "fixed_budgets_gpu_seconds": [0.1, 0.5, 1, 10, 100],
               "error_at_budget_rule": "MSE of the last checkpoint whose adaptation seconds <= budget (training-free methods at 0 s)",
               "time_to_target": "GPU-seconds to reach LoRA's 500-update final-task mean MSE (linear between checkpoints)"}},
    "inference": {"difference": "ours minus baseline per (task, seed, demo); negative = ours better",
                  "primary_interval": "95% percentile bootstrap over the 6 task x seed clusters (cluster value = mean paired difference over its 20 demos), 10,000 resamples, Python random.Random(0)",
                  "secondary_interval": "95% bootstrap over individual demo pairs (optimistic; reported for continuity)",
                  "settings_won": "number of the 6 clusters with mean difference < 0",
                  "decision_rule": "better if the cluster interval lies entirely below 0; worse if entirely above 0; otherwise no clear difference",
                  "multiplicity": "all P1-P2 comparisons are reported; no correction; P1 and P2 at 25 updates are the headline comparisons"},
    "secondary": ["R2 for every method and checkpoint", "development vs final contrast (descriptive)", "all other pairs (descriptive)"],
    "rules": ["one run; resume of the same run after an interruption is allowed, a second run is not",
              "nothing is tuned, re-selected or re-run after the final run; all results are reported, including negative ones",
              "closed-loop task success is separate secondary evidence (optional step 4, needs user approval and funds)"],
}
body["digest"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
OUT.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
L = a.LABELS
md = ["# Frozen final analysis plan (amendment)", "",
      f"Frozen {body['frozen_at']} before any final-task data was hashed, packaged or read. Digest {body['digest']}.",
      f"Parent protocol {protocol['digest']}; model digest {protocol['model_digest']}. held_out_claims_eligible = false (VPP2 saw LIBERO; this benchmark does not use VPP2).", "",
      "## Data", "", "2 final tasks x seeds 42/43/44 = 6 settings; five-shot support; 20 query demos per task with real recorded futures. One run.", "",
      "## Methods (frozen)", "", "No training: " + ", ".join(L[m] for m in methods_zero) + ".",
      "Trained (500 updates, batch 64, checkpoints 0/25/50/100/200/350/500): " + ", ".join(L[m] for m in methods_trained) + ".",
      "RAID v7 head-only is exploratory (defined after development results).", "",
      "## Metrics and compute", "", "Chunk MSE on normalized actions (macro over query demos) and R2. Adaptation GPU-seconds, FLOPs per update, trainable parameters, decision latency and FLOPs.", "",
      "## Primary comparisons", "",
      f"- P1: RAID v7 (no training) vs {L[best_tf]} (best training-free baseline on development, MSE {zero[best_tf]:.3f}).",
      f"- P2: best RAID variant vs each baseline ({', '.join(L[b] for b in list(TRAINED) + list(TRAINING_FREE))}). At 25 updates: {L[best_raid[25]]} (development MSE {dev_means[25][best_raid[25]]:.3f}). At 500 updates: {L[best_raid[500]]} (development MSE {dev_means[500][best_raid[500]]:.3f}).",
      "- P3: per-GPU-second curves (descriptive); error at 0.1, 0.5, 1, 10 and 100 GPU-seconds; GPU-seconds to reach LoRA's 500-update error.", "",
      "## Inference", "", "Paired difference ours minus baseline. Primary 95% interval: percentile bootstrap over the 6 task x seed clusters (10,000 resamples, seed 0). Also: demo-level interval and settings won out of 6.",
      "Decision rule: better if the cluster interval is entirely below 0, worse if entirely above 0, else no clear difference. All comparisons are reported; no tuning after the run.", ""]
MD.write_text("\n".join(md))
(P / "final_analysis_plan.html").write_text(a.page("Final analysis plan", md))
print(json.dumps({"digest": body["digest"], "best_training_free": best_tf, "best_raid": best_raid}))
