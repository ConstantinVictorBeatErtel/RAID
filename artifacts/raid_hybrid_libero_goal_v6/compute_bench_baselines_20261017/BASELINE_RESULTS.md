# Few-shot inverse-dynamics benchmark: our methods vs re-implemented baselines

Development tasks (2) x registered support seeds 42/43/44, five demos per task; 20 query demos per task with real recorded future frames.
One pod (A100-SXM4-80GB, 16 CPU threads) for all methods. Metrics: chunk MSE on normalized actions (lower is better) and R2 (higher is better).
Baselines are faithful re-implementations of each method family in this setting, not reproductions of the papers.

## Without any target training

| Rank | Method | MSE | R2 |
|---|---|---|---|
| 1 | RAID v7 (ours) | 0.409 | 0.373 |
| 2 | RAID v6 (ours) | 0.424 | 0.328 |
| 3 | VINN-style k-NN | 0.426 | 0.289 |
| 4 | REGENT-style blend | 0.452 | 0.248 |
| 5 | REGENT retrieve-and-play | 0.476 | 0.212 |
| 6 | Frozen source controller | 0.756 | -0.019 |

## With target training: error after N updates (same schedule for all)

| Method | 0 | 25 | 50 | 100 | 200 | 350 | 500 | GPU s (500 updates) | TFLOP / update | Trainable params |
|---|---|---|---|---|---|---|---|---|---|---|
| Full fine-tuning | 0.756 | 0.351 | 0.291 | 0.270 | 0.268 | 0.263 | 0.260 | 443 | 13.55 | 86,885,944 |
| LoRA, no source training | 1.062 | 0.686 | 0.458 | 0.340 | 0.285 | 0.289 | 0.279 | 183 | 5.21 | 1,161,016 |
| LoRA (TAIL/FADA style) | 0.756 | 0.362 | 0.333 | 0.294 | 0.279 | 0.282 | 0.280 | 187 | 5.21 | 1,161,016 |
| Head-only fine-tuning | 0.756 | 0.369 | 0.327 | 0.310 | 0.296 | 0.293 | 0.288 | 1 | 0.00 | 1,087,288 |
| RAID+LoRA v7 (ours) | 0.409 | 0.308 | 0.304 | 0.295 | 0.298 | 0.294 | 0.290 | 204 | 5.21 | 1,494,560 |
| Frozen-encoder probe (fresh head) | 1.062 | 0.708 | 0.470 | 0.356 | 0.308 | 0.314 | 0.294 | 1 | 0.00 | 1,087,288 |
| STRAP-style retrieved training data | 0.756 | 0.415 | 0.367 | 0.343 | 0.315 | 0.296 | 0.297 | 2 | 0.00 | 1,087,288 |
| RAID+LoRA v6 (ours) | 0.424 | 0.342 | 0.350 | 0.358 | 0.361 | 0.346 | 0.342 | 192 | 5.21 | 1,161,016 |

## Paired differences (second method minus first; negative = ours better), 95% bootstrap CI, settings won of 6

| Comparison | Mean | 95% CI | Settings won |
|---|---|---|---|
| RAID v7 vs VINN-style k-NN (no training) | -0.017 | [-0.035, +0.000] | 3/6 |
| RAID v7 vs REGENT-style blend (no training) | -0.042 | [-0.058, -0.027] | 5/6 |
| RAID v7 vs REGENT retrieve-and-play (no training) | -0.066 | [-0.086, -0.048] | 6/6 |
| RAID v7 vs RAID v6 (no training) | -0.014 | [-0.021, -0.007] | 5/6 |
| RAID+LoRA v7 vs Head-only fine-tuning, 25 updates | -0.060 | [-0.068, -0.053] | 6/6 |
| RAID+LoRA v7 vs LoRA (TAIL/FADA style), 25 updates | -0.054 | [-0.061, -0.048] | 6/6 |
| RAID+LoRA v7 vs Full fine-tuning, 25 updates | -0.042 | [-0.049, -0.035] | 6/6 |
| RAID+LoRA v7 vs STRAP-style retrieved training data, 25 updates | -0.107 | [-0.115, -0.098] | 6/6 |
| RAID+LoRA v7 vs Head-only fine-tuning, 500 updates | +0.002 | [-0.004, +0.007] | 2/6 |
| RAID+LoRA v7 vs LoRA (TAIL/FADA style), 500 updates | +0.009 | [+0.005, +0.014] | 2/6 |
| RAID+LoRA v7 vs Full fine-tuning, 500 updates | +0.029 | [+0.021, +0.037] | 0/6 |
| RAID+LoRA v7 vs STRAP-style retrieved training data, 500 updates | -0.007 | [-0.015, -0.000] | 4/6 |

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
| Frozen source controller | 7.3 | 70 |
| RAID v7 (ours) | 7.3 | 70 |
| RAID+LoRA v6 (ours) | 14.1 | 141 |
| RAID+LoRA v7 (ours) | 14.1 | 141 |

## Reading

- Without training, RAID v7 has the lowest error of all methods and beats REGENT-style blending and retrieve-and-play clearly; against VINN-style k-NN the gain is small (-0.017, CI touches 0, 3/6 settings).
- With few updates, RAID+LoRA v7 is the best method (25 updates: better than every baseline in 6/6 settings).
- With full training, full fine-tuning is best (0.260); RAID+LoRA v7 is equal to head-only and slightly worse than LoRA.
- Per GPU-second, head-only training on cached frozen features is extremely cheap (about 1 s for 500 updates) and reaches 0.288; LoRA-based methods need about 200 s because every update runs the image encoder.

## Limits

Offline recorded-transition error on development tasks only (final tasks sealed). Three registered seeds. Bootstrap intervals treat demos as independent (optimistic). v7 and these baselines were chosen after seeing earlier development results, so this is exploratory until a frozen final-task run.
