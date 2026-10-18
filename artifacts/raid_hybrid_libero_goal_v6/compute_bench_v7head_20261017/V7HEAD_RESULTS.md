# Exploratory: RAID v7 head-only (frozen encoder, retrieval head fine-tuned)

EXPLORATORY. Development tasks only (2) x registered support seeds 42/43/44, five-shot; 20 query demos per task with real recorded future frames.
New method: frozen MAE (cached features), source-trained v7 head (cross-attention + gate over k = 16 retrieved support candidates) fine-tuned with the selection loss; same schedule and sample stream as head-only (500 updates, batch 64). It was designed after seeing the development results.
This run (A100 80 GB, 16 CPU threads) reran the zero-training conditions and head-only; they match the 2026-10-17 baseline run exactly (largest per-demo difference 0.0e+00), so raid_v7_head pairs with all baselines of that run.
Metrics: chunk MSE on normalized actions (lower is better), macro over query demos; R2 (higher is better).

## Error after N updates (same schedule for all)

| Method | 0 | 25 | 50 | 100 | 200 | 350 | 500 | R2 at 500 | GPU s (500 updates) | GFLOP / update | Trainable params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full fine-tuning | 0.756 | 0.351 | 0.291 | 0.270 | 0.268 | 0.263 | 0.260 | 0.512 | 443.5 | 13550.8 | 86,885,944 |
| LoRA, no source training | 1.062 | 0.686 | 0.458 | 0.340 | 0.285 | 0.289 | 0.279 | 0.488 | 182.8 | 5208.4 | 1,161,016 |
| LoRA (TAIL/FADA style) | 0.756 | 0.362 | 0.333 | 0.294 | 0.279 | 0.282 | 0.280 | 0.475 | 187.3 | 5208.4 | 1,161,016 |
| Head-only fine-tuning | 0.756 | 0.369 | 0.327 | 0.310 | 0.296 | 0.293 | 0.288 | 0.475 | 1.2 | 0.3 | 1,087,288 |
| RAID+LoRA v7 (ours) | 0.409 | 0.308 | 0.304 | 0.295 | 0.298 | 0.294 | 0.290 | 0.461 | 204.3 | 5209.4 | 1,494,560 |
| Frozen-encoder probe (fresh head) | 1.062 | 0.708 | 0.470 | 0.356 | 0.308 | 0.314 | 0.294 | 0.474 | 1.1 | 0.3 | 1,087,288 |
| STRAP-style retrieved training data | 0.756 | 0.415 | 0.367 | 0.343 | 0.315 | 0.296 | 0.297 | 0.466 | 1.8 | 0.3 | 1,087,288 |
| RAID v7 head-only (ours) | 0.409 | 0.312 | 0.307 | 0.304 | 0.298 | 0.291 | 0.300 | 0.448 | 3.0 | 1.3 | 1,420,832 |
| RAID+LoRA v6 (ours) | 0.424 | 0.342 | 0.350 | 0.358 | 0.361 | 0.346 | 0.342 | 0.406 | 192.5 | 5208.5 | 1,161,016 |

## No-training reference

| Method | MSE | R2 |
|---|---|---|
| RAID v7 (ours) | 0.409 | 0.373 |
| RAID v6 (ours) | 0.424 | 0.328 |
| VINN-style k-NN | 0.426 | 0.289 |
| REGENT-style blend | 0.452 | 0.248 |
| REGENT retrieve-and-play | 0.476 | 0.212 |
| Frozen source controller | 0.756 | -0.019 |

## Paired differences (raid_v7_head minus baseline; negative = ours better)

Primary interval: 95% bootstrap over the 6 task x seed clusters (10,000 resamples). Demo-level interval as in earlier reports (optimistic).

| Comparison (ours minus baseline) | Mean | 95% CI, task x seed clusters | 95% CI, demos | Settings won |
|---|---|---|---|---|
| RAID v7 head-only (ours) vs Head-only fine-tuning, 25 updates | -0.057 | [-0.062, -0.048] | [-0.064, -0.049] | 6/6 |
| RAID v7 head-only (ours) vs RAID+LoRA v7 (ours), 25 updates | +0.004 | [+0.002, +0.006] | [+0.002, +0.006] | 0/6 |
| RAID v7 head-only (ours) vs LoRA (TAIL/FADA style), 25 updates | -0.050 | [-0.056, -0.044] | [-0.057, -0.044] | 6/6 |
| RAID v7 head-only (ours) vs Full fine-tuning, 25 updates | -0.038 | [-0.044, -0.034] | [-0.045, -0.031] | 6/6 |
| RAID v7 head-only (ours) vs STRAP-style retrieved training data, 25 updates | -0.103 | [-0.123, -0.082] | [-0.111, -0.094] | 6/6 |
| RAID v7 head-only (ours) vs Frozen-encoder probe (fresh head), 25 updates | -0.396 | [-0.447, -0.343] | [-0.417, -0.376] | 6/6 |
| RAID v7 head-only (ours) vs LoRA, no source training, 25 updates | -0.374 | [-0.425, -0.320] | [-0.395, -0.354] | 6/6 |
| RAID v7 head-only (ours) vs Head-only fine-tuning, 500 updates | +0.012 | [+0.005, +0.019] | [+0.007, +0.016] | 0/6 |
| RAID v7 head-only (ours) vs RAID+LoRA v7 (ours), 500 updates | +0.010 | [+0.001, +0.019] | [+0.005, +0.015] | 1/6 |
| RAID v7 head-only (ours) vs LoRA (TAIL/FADA style), 500 updates | +0.020 | [+0.009, +0.030] | [+0.014, +0.025] | 1/6 |
| RAID v7 head-only (ours) vs Full fine-tuning, 500 updates | +0.039 | [+0.028, +0.053] | [+0.031, +0.048] | 0/6 |
| RAID v7 head-only (ours) vs STRAP-style retrieved training data, 500 updates | +0.003 | [-0.009, +0.012] | [-0.004, +0.010] | 2/6 |
| RAID v7 head-only (ours) vs Frozen-encoder probe (fresh head), 500 updates | +0.006 | [-0.003, +0.015] | [-0.003, +0.013] | 3/6 |
| RAID v7 head-only (ours) vs LoRA, no source training, 500 updates | +0.021 | [+0.004, +0.034] | [+0.013, +0.029] | 1/6 |
| RAID v7 head-only, 0 updates vs RAID v7 (sanity: identical) | +0.000 | [+0.000, +0.000] | [+0.000, +0.000] | 0/6 |
| RAID v7 head-only (ours), 25 updates vs VINN-style k-NN (best training-free baseline) | -0.114 | [-0.144, -0.086] | [-0.129, -0.101] | 6/6 |
| RAID v7 head-only (ours), 500 updates vs VINN-style k-NN (best training-free baseline) | -0.127 | [-0.148, -0.106] | [-0.142, -0.112] | 6/6 |

## Compute: GPU-seconds to reach LoRA's 500-update error

Target = mean LoRA error after 500 updates (0.280).

| Method | GPU s to target |
|---|---|
| Full fine-tuning | 66.8 |
| LoRA, no source training | 174.8 |
| LoRA (TAIL/FADA style) | 72.4 |
| Head-only fine-tuning | not reached |
| RAID+LoRA v7 (ours) | not reached |
| Frozen-encoder probe (fresh head) | not reached |
| STRAP-style retrieved training data | not reached |
| RAID v7 head-only (ours) | not reached |
| RAID+LoRA v6 (ours) | not reached |

## Error at fixed adaptation budgets (last checkpoint within the budget)

| Method | MSE at 0.1 s | MSE at 0.5 s | MSE at 1 s | MSE at 10 s | MSE at 100 s |
|---|---|---|---|---|---|
| Full fine-tuning | 0.756 | 0.756 | 0.756 | 0.756 | 0.270 |
| LoRA, no source training | 1.062 | 1.062 | 1.062 | 0.686 | 0.285 |
| LoRA (TAIL/FADA style) | 0.756 | 0.756 | 0.756 | 0.362 | 0.279 |
| Head-only fine-tuning | 0.369 | 0.296 | 0.293 | 0.288 | 0.288 |
| RAID+LoRA v7 (ours) | 0.409 | 0.409 | 0.409 | 0.308 | 0.298 |
| Frozen-encoder probe (fresh head) | 0.708 | 0.308 | 0.314 | 0.294 | 0.294 |
| STRAP-style retrieved training data | 0.415 | 0.343 | 0.315 | 0.297 | 0.297 |
| RAID v7 head-only (ours) | 0.409 | 0.307 | 0.304 | 0.300 | 0.300 |
| RAID+LoRA v6 (ours) | 0.424 | 0.424 | 0.424 | 0.342 | 0.361 |

## Reading

- After 25 updates, RAID v7 head-only (0.312) is better than every baseline in 6/6 settings (head-only -0.057, LoRA -0.050, full fine-tuning -0.038) and close to RAID+LoRA v7 (+0.004, worse in 6/6), with about 1.5% of its adaptation compute (3.0 s vs 204 s for 500 updates) because the encoder stays frozen.
- With more updates it does not improve further (500 updates: 0.300); it ends worse than head-only (+0.012), LoRA (+0.020) and full fine-tuning (+0.039).
- Per GPU-second, plain head-only is better at every fixed budget (0.1 to 100 s): one head-only update costs about 2.4 ms against 6 ms (retrieval over 16 candidates), so in the same time it does more updates. The RAID v7 head advantage is per update, not per GPU-second. Both are about 100x cheaper than the LoRA methods.
- Exploratory only. The frozen final analysis keeps the selection rule fixed in advance (best RAID variant by development error), which picks RAID+LoRA v7 at 25 and at 500 updates.


## Cost per control decision (batch 1)

| Method | ms | GFLOP |
|---|---|---|
| Full fine-tuning | 6.7 | 70 |
| Head-only fine-tuning | 6.8 | 70 |
| Frozen-encoder probe (fresh head) | 6.8 | 70 |
| LoRA (TAIL/FADA style) | 7.0 | 70 |
| LoRA, no source training | 7.0 | 70 |
| RAID v6 (ours) | 7.1 | 70 |
| STRAP-style retrieved training data | 7.1 | 70 |
| REGENT retrieve-and-play | 7.1 | 70 |
| REGENT-style blend | 7.1 | 70 |
| RAID v7 head-only (ours) | 7.2 | 70 |
| Frozen source controller | 7.3 | 70 |
| RAID v7 (ours) | 7.3 | 70 |
| RAID+LoRA v6 (ours) | 14.1 | 141 |
| RAID+LoRA v7 (ours) | 14.1 | 141 |

## Reproducibility check against the baseline run (largest per-demo |difference|)

| Quantity | Max difference |
|---|---|
| source_frozen | 0.0e+00 |
| raid_bank_mlp | 0.0e+00 |
| raid_bank_v7 | 0.0e+00 |
| support_knn_average | 0.0e+00 |
| head_only (all checkpoints) | 0.0e+00 |

## Limits

Exploratory: the variant was defined after the development results were known. Offline recorded-transition error, not task success. Development tasks only; final tasks sealed; held_out_claims_eligible = false (VPP2 saw LIBERO). Three registered seeds, so the cluster interval rests on 6 clusters and is coarse.
