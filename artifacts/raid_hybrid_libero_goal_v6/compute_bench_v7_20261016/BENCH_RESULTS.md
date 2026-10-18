# Compute-aware benchmark: RAID, LoRA, RAID+LoRA (v6) and RAID+LoRA (v7)

Offline recorded-transition action error (chunk MSE, lower is better) on the 20 registered query demos of each development task;
registered support seeds 42/43/44; five-shot; one pod, NVIDIA A100 80GB PCIe (the pilot used A100-SXM4-80GB). Mean over 6 settings (2 tasks x 3 seeds).

## No target training (instant)

| Method | Error |
|---|---|
| Frozen | 0.756 |
| RAID v6 | 0.424 |
| RAID v7 | 0.409 |
| k-NN average | 0.426 |

## Error vs adaptation budget (updates; GPU-seconds, evaluation excluded)

| Updates | Seconds | LoRA | RAID+LoRA v6 | RAID+LoRA v7 |
|---|---|---|---|---|
| 0 | 0 | 0.756 | 0.424 | 0.409 |
| 25 | 10 | 0.362 | 0.342 | 0.308 |
| 50 | 21 | 0.333 | 0.350 | 0.304 |
| 100 | 43 | 0.295 | 0.358 | 0.296 |
| 200 | 85 | 0.281 | 0.361 | 0.293 |
| 350 | 150 | 0.280 | 0.346 | 0.287 |
| 500 | 214 | 0.282 | 0.342 | 0.288 |

## Paired differences (second minus first; negative = second is better), 95% bootstrap CI over demo x setting pairs

| Comparison | Mean difference | 95% CI | n |
|---|---|---|---|
| RAID v7 vs RAID v6 (no training) | -0.014 | [-0.021, -0.007] | 120 |
| RAID+LoRA v7 vs LoRA, 25 updates | -0.054 | [-0.061, -0.048] | 120 |
| RAID+LoRA v7 vs LoRA, 100 updates | +0.001 | [-0.004, +0.005] | 120 |
| RAID+LoRA v7 vs LoRA, 500 updates | +0.006 | [+0.002, +0.011] | 120 |
| RAID+LoRA v6 vs LoRA, 500 updates | +0.060 | [+0.050, +0.072] | 120 |
| RAID+LoRA v7 vs RAID+LoRA v6, 500 updates | -0.054 | [-0.065, -0.044] | 120 |

## Compute

- Shared one-time costs: reference feature cache 160 s; source training 200 s (v6 head) and 197 s (v7 head), 2,000 updates each.
- RAID bank build per task: 0.027 s (466-981 entries). No training.

| Method | Adaptation s (500 updates) | TFLOP per update | Trainable parameters |
|---|---|---|---|
| LoRA | 215 | 5.21 | 1,161,016 |
| RAID+LoRA v6 | 219 | 5.21 | 1,161,016 |
| RAID+LoRA v7 | 223 | 5.21 | 1,494,560 |

| Method | ms per decision | GFLOP per decision | Peak GPU GB |
|---|---|---|---|
| Frozen | 6.7 | 70 | 0.41 |
| RAID v6 | 6.9 | 70 | 0.41 |
| RAID v7 | 7.0 | 70 | 0.41 |
| LoRA | 6.9 | 70 | 0.78 |
| RAID+LoRA v6 | 13.8 | 141 | 0.79 |
| RAID+LoRA v7 | 13.9 | 141 | 0.79 |

RAID+LoRA needs two encodes per decision (LoRA controller features plus frozen retrieval keys), so its decision cost is about twice that of the other methods.

## Limits

Offline error on recorded transitions, not task success. Development tasks only; final tasks sealed; held_out_claims_eligible = false. 3 registered seeds. The bootstrap intervals treat demos as independent although they share tasks, seeds and source controllers, so they are optimistic. v7 was designed after seeing v6 pilot results (exploratory).
