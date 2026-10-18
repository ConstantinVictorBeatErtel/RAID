# Final-task offline benchmark (one frozen run)

Run once on the 2 sealed LIBERO-Goal final tasks x registered seeds 42/43/44 (6 settings), five-shot, 20 query demos per task with real recorded future frames. Analysis exactly as frozen in amendment 1b42ba599a9b (written before any final data was read). Nothing was tuned after this run.
held_out_claims_eligible = false: the tasks were never used for design, but the VPP2 generator saw LIBERO in training (VPP2 is not used here).
Metrics: chunk MSE on normalized actions (lower is better), macro over query demos; R2 (higher is better).

## Headline comparisons (frozen P1 and P2)

Difference = ours minus baseline (negative = ours better). Primary interval: 95% bootstrap over the 6 task x seed clusters. Verdict rule fixed in advance.

| Comparison (ours minus baseline) | Mean | 95% CI, task x seed clusters | 95% CI, demos | Settings won | Verdict |
|---|---|---|---|---|---|
| P1: RAID v7 (no training) vs VINN-style k-NN | +0.123 | [+0.007, +0.250] | [+0.086, +0.161] | 3/6 | worse |
| P2: RAID+LoRA v7 (ours) vs Head-only fine-tuning, 25 updates | -0.049 | [-0.067, -0.030] | [-0.056, -0.041] | 6/6 | better |
| P2: RAID+LoRA v7 (ours) vs Frozen-encoder probe (fresh head), 25 updates | -0.534 | [-0.563, -0.504] | [-0.551, -0.516] | 6/6 | better |
| P2: RAID+LoRA v7 (ours) vs LoRA, no source training, 25 updates | -0.527 | [-0.557, -0.491] | [-0.545, -0.509] | 6/6 | better |
| P2: RAID+LoRA v7 (ours) vs Full fine-tuning, 25 updates | -0.042 | [-0.051, -0.033] | [-0.048, -0.036] | 6/6 | better |
| P2: RAID+LoRA v7 (ours) vs STRAP-style retrieved training data, 25 updates | -0.129 | [-0.182, -0.079] | [-0.145, -0.113] | 6/6 | better |
| P2: RAID+LoRA v7 (ours) vs LoRA (TAIL/FADA style), 25 updates | -0.044 | [-0.061, -0.027] | [-0.050, -0.037] | 6/6 | better |
| P2: RAID+LoRA v7 (ours) vs VINN-style k-NN, 25 updates | -0.105 | [-0.129, -0.083] | [-0.119, -0.091] | 6/6 | better |
| P2: RAID+LoRA v7 (ours) vs REGENT retrieve-and-play, 25 updates | -0.177 | [-0.219, -0.134] | [-0.197, -0.158] | 6/6 | better |
| P2: RAID+LoRA v7 (ours) vs REGENT-style blend, 25 updates | -0.164 | [-0.206, -0.121] | [-0.183, -0.145] | 6/6 | better |
| P2: RAID+LoRA v7 (ours) vs Head-only fine-tuning, 500 updates | +0.000 | [-0.009, +0.008] | [-0.005, +0.006] | 2/6 | no clear difference |
| P2: RAID+LoRA v7 (ours) vs Frozen-encoder probe (fresh head), 500 updates | -0.014 | [-0.029, +0.002] | [-0.022, -0.005] | 4/6 | no clear difference |
| P2: RAID+LoRA v7 (ours) vs LoRA, no source training, 500 updates | +0.007 | [-0.005, +0.018] | [+0.000, +0.014] | 2/6 | no clear difference |
| P2: RAID+LoRA v7 (ours) vs Full fine-tuning, 500 updates | +0.013 | [+0.004, +0.021] | [+0.003, +0.022] | 1/6 | worse |
| P2: RAID+LoRA v7 (ours) vs STRAP-style retrieved training data, 500 updates | -0.045 | [-0.085, -0.005] | [-0.060, -0.031] | 4/6 | better |
| P2: RAID+LoRA v7 (ours) vs LoRA (TAIL/FADA style), 500 updates | -0.002 | [-0.012, +0.007] | [-0.007, +0.004] | 3/6 | no clear difference |
| P2: RAID+LoRA v7 (ours) vs VINN-style k-NN, 500 updates | -0.095 | [-0.113, -0.077] | [-0.112, -0.079] | 6/6 | better |
| P2: RAID+LoRA v7 (ours) vs REGENT retrieve-and-play, 500 updates | -0.167 | [-0.202, -0.133] | [-0.187, -0.147] | 6/6 | better |
| P2: RAID+LoRA v7 (ours) vs REGENT-style blend, 500 updates | -0.154 | [-0.190, -0.120] | [-0.173, -0.135] | 6/6 | better |

## Without any target training

| Rank | Method | MSE | R2 | Development MSE |
|---|---|---|---|---|
| 1 | VINN-style k-NN | 0.376 | 0.300 | 0.426 |
| 2 | REGENT-style blend | 0.435 | 0.212 | 0.452 |
| 3 | REGENT retrieve-and-play | 0.448 | 0.187 | 0.476 |
| 4 | RAID v7 (ours) | 0.500 | -0.041 | 0.409 |
| 5 | RAID v6 (ours) | 0.535 | 0.153 | 0.424 |
| 6 | Frozen source controller | 1.055 | -1.116 | 0.756 |

## With target training: error after N updates

| Method | 0 | 25 | 50 | 100 | 200 | 350 | 500 | R2 at 500 | Dev. MSE at 500 | GPU s (500 updates) | GFLOP / update | Trainable params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full fine-tuning | 1.055 | 0.313 | 0.281 | 0.266 | 0.269 | 0.267 | 0.268 | 0.462 | 0.260 | 444.2 | 13550.8 | 86,885,944 |
| LoRA, no source training | 1.339 | 0.799 | 0.461 | 0.298 | 0.278 | 0.275 | 0.274 | 0.460 | 0.279 | 183.8 | 5208.4 | 1,161,016 |
| Head-only fine-tuning | 1.055 | 0.320 | 0.298 | 0.284 | 0.292 | 0.292 | 0.280 | 0.449 | 0.288 | 1.1 | 0.3 | 1,087,288 |
| RAID+LoRA v7 (ours) | 0.500 | 0.272 | 0.285 | 0.280 | 0.291 | 0.289 | 0.281 | 0.441 | 0.290 | 201.3 | 5209.3 | 1,494,560 |
| LoRA (TAIL/FADA style) | 1.055 | 0.315 | 0.294 | 0.292 | 0.287 | 0.285 | 0.283 | 0.446 | 0.280 | 188.6 | 5208.4 | 1,161,016 |
| RAID v7 head-only (ours) | 0.500 | 0.273 | 0.277 | 0.284 | 0.290 | 0.293 | 0.287 | 0.441 | 0.300 | 2.7 | 1.2 | 1,420,832 |
| Frozen-encoder probe (fresh head) | 1.339 | 0.806 | 0.464 | 0.305 | 0.286 | 0.291 | 0.295 | 0.438 | 0.294 | 1.1 | 0.3 | 1,087,288 |
| STRAP-style retrieved training data | 1.055 | 0.401 | 0.357 | 0.325 | 0.331 | 0.319 | 0.326 | 0.258 | 0.297 | 1.7 | 0.3 | 1,087,288 |
| RAID+LoRA v6 (ours) | 0.535 | 0.297 | 0.317 | 0.331 | 0.350 | 0.344 | 0.335 | 0.384 | 0.342 | 193.4 | 5208.5 | 1,161,016 |

## P3: error per GPU-second

Error at fixed adaptation budgets (last checkpoint within the budget) and GPU-seconds to reach LoRA's 500-update error (0.283).

| Method | MSE at 0.1 s | MSE at 0.5 s | MSE at 1 s | MSE at 10 s | MSE at 100 s | GPU s to target |
|---|---|---|---|---|---|---|
| Full fine-tuning | 1.055 | 1.055 | 1.055 | 1.055 | 0.266 | 43.5 |
| LoRA, no source training | 1.339 | 1.339 | 1.339 | 0.799 | 0.278 | 64.7 |
| Head-only fine-tuning | 0.320 | 0.292 | 0.292 | 0.280 | 0.280 | 1.0 |
| RAID+LoRA v7 (ours) | 0.500 | 0.500 | 0.500 | 0.272 | 0.291 | 9.4 |
| LoRA (TAIL/FADA style) | 1.055 | 1.055 | 1.055 | 0.315 | 0.287 | 187.3 |
| RAID v7 head-only (ours) | 0.500 | 0.277 | 0.284 | 0.287 | 0.287 | 0.1 |
| Frozen-encoder probe (fresh head) | 0.806 | 0.286 | 0.291 | 0.295 | 0.295 | not reached |
| STRAP-style retrieved training data | 0.401 | 0.325 | 0.331 | 0.326 | 0.326 | not reached |
| RAID+LoRA v6 (ours) | 0.535 | 0.535 | 0.535 | 0.297 | 0.350 | not reached |
| VINN-style k-NN | 0.376 | 0.376 | 0.376 | 0.376 | 0.376 | not reached |
| REGENT-style blend | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 | not reached |
| REGENT retrieve-and-play | 0.448 | 0.448 | 0.448 | 0.448 | 0.448 | not reached |
| RAID v7 (ours) | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | not reached |
| RAID v6 (ours) | 0.535 | 0.535 | 0.535 | 0.535 | 0.535 | not reached |
| Frozen source controller | 1.055 | 1.055 | 1.055 | 1.055 | 1.055 | not reached |

## Per setting (descriptive): no training and 25 updates

| Task | Seed | Frozen source | RAID v7 | VINN-style k-NN | REGENT-style blend | RAID+LoRA v7, 25 | LoRA, 25 | RAID v7 head-only, 25 |
|---|---|---|---|---|---|---|---|---|
| put the bowl on top of the cabinet | 42 | 0.640 | 0.324 | 0.374 | 0.408 | 0.252 | 0.279 | 0.251 |
| put the bowl on top of the cabinet | 43 | 0.736 | 0.306 | 0.319 | 0.346 | 0.244 | 0.275 | 0.243 |
| put the bowl on top of the cabinet | 44 | 0.598 | 0.320 | 0.346 | 0.391 | 0.272 | 0.294 | 0.268 |
| put the wine bottle on top of the cabinet | 42 | 1.456 | 0.716 | 0.353 | 0.406 | 0.277 | 0.353 | 0.280 |
| put the wine bottle on top of the cabinet | 43 | 1.521 | 0.714 | 0.456 | 0.560 | 0.324 | 0.353 | 0.330 |
| put the wine bottle on top of the cabinet | 44 | 1.382 | 0.617 | 0.409 | 0.499 | 0.261 | 0.337 | 0.266 |

## Reading

- P1 failed: without training, RAID v7 (0.500) is worse than VINN-style k-NN (0.376); verdict worse. On the bowl task RAID v7 is better than k-NN in 3/3 seeds (0.306-0.324 vs 0.319-0.374); on the wine-bottle task it is much worse in 3/3 (0.617-0.716 vs 0.353-0.456). On that task the frozen source controller is very poor, and the learned gate still relies on it; plain retrieval (k-NN) does not. On development, RAID v7 was the best training-free method (0.409); this did not transfer.
- P2 at 25 updates held: RAID+LoRA v7 (0.272) is better than every baseline in 6/6 settings, all intervals below 0 (for example LoRA -0.044, full fine-tuning -0.042, head-only -0.049, k-NN -0.105).
- P2 at 500 updates: no clear difference against head-only, LoRA, LoRA without source training and the fresh-head probe; worse than full fine-tuning (+0.013, 1/6); better than STRAP-style and all training-free baselines.
- P3: the cheapest good point is RAID v7 head-only: 0.277 after 0.5 GPU-seconds (it reaches LoRA's 500-update error in about 0.1 s); plain head-only needs about 1 s. LoRA-based methods need about 10 s (RAID+LoRA v7) to 190 s (LoRA). Full fine-tuning has the lowest error at 100 s (0.266) but 13.6 TFLOP per update.
- The development-to-final pattern for trained methods held (fast early gain from retrieval, convergence by 500 updates); the zero-training result did not.


## Secondary comparisons (descriptive)

| Comparison (ours minus baseline) | Mean | 95% CI, task x seed clusters | 95% CI, demos | Settings won |
|---|---|---|---|---|
| RAID v7 vs REGENT retrieve-and-play (no training) | +0.051 | [-0.054, +0.174] | [+0.015, +0.089] | 3/6 |
| RAID v7 vs REGENT-style blend (no training) | +0.064 | [-0.040, +0.186] | [+0.029, +0.101] | 3/6 |
| RAID+LoRA v6 (ours) vs LoRA, 25 updates | -0.018 | [-0.042, +0.001] | [-0.027, -0.009] | 3/6 |
| RAID v7 head-only (ours) vs LoRA, 25 updates | -0.042 | [-0.059, -0.027] | [-0.049, -0.036] | 6/6 |
| RAID+LoRA v6 (ours) vs LoRA, 500 updates | +0.052 | [+0.043, +0.063] | [+0.039, +0.065] | 0/6 |
| RAID v7 head-only (ours) vs LoRA, 500 updates | +0.004 | [-0.002, +0.011] | [-0.001, +0.010] | 2/6 |

## Cost per control decision (batch 1)

| Method | ms | GFLOP |
|---|---|---|
| Frozen-encoder probe (fresh head) | 6.8 | 70 |
| Full fine-tuning | 6.8 | 70 |
| Head-only fine-tuning | 6.8 | 70 |
| LoRA, no source training | 6.9 | 70 |
| LoRA (TAIL/FADA style) | 6.9 | 70 |
| REGENT retrieve-and-play | 6.9 | 70 |
| RAID v6 (ours) | 7.1 | 70 |
| REGENT-style blend | 7.1 | 70 |
| Frozen source controller | 7.2 | 70 |
| RAID v7 (ours) | 7.2 | 70 |
| STRAP-style retrieved training data | 7.3 | 70 |
| RAID v7 head-only (ours) | 7.5 | 70 |
| RAID+LoRA v6 (ours) | 13.9 | 141 |
| RAID+LoRA v7 (ours) | 14.0 | 141 |

## Limits

Offline recorded-transition action error, not task success. Two final tasks and three seeds: the cluster interval rests on 6 clusters; a percentile bootstrap over so few clusters can be too narrow, so read settings won next to it. RAID v7 head-only and RAID v7 were defined after development results (development was used for design; final tasks were not). No multiplicity correction.
