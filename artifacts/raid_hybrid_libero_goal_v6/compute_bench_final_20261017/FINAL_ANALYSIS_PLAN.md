# Frozen final analysis plan (amendment)

Frozen 2026-10-17T20:15:19.891327+00:00 before any final-task data was hashed, packaged or read. Digest 1b42ba599a9b983d6c67befdfb27841d3dbc49aa26977410ad6acb3fd26e69bf.
Parent protocol f555d02d6e32446aedfa7e0e34c1d12f2a090f04570f9719f71a48ee66c3d041; model digest 68c5cb35bb9f6d2f0fab5f80530b9dea5ec9a1dad8d95672ac13b45f865eb341. held_out_claims_eligible = false (VPP2 saw LIBERO; this benchmark does not use VPP2).

## Data

2 final tasks x seeds 42/43/44 = 6 settings; five-shot support; 20 query demos per task with real recorded futures. One run.

## Methods (frozen)

No training: Frozen source controller, RAID v6 (ours), RAID v7 (ours), VINN-style k-NN, REGENT retrieve-and-play, REGENT-style blend.
Trained (500 updates, batch 64, checkpoints 0/25/50/100/200/350/500): LoRA (TAIL/FADA style), RAID+LoRA v6 (ours), RAID+LoRA v7 (ours), Head-only fine-tuning, Frozen-encoder probe (fresh head), LoRA, no source training, Full fine-tuning, STRAP-style retrieved training data, RAID v7 head-only (ours).
RAID v7 head-only is exploratory (defined after development results).

## Metrics and compute

Chunk MSE on normalized actions (macro over query demos) and R2. Adaptation GPU-seconds, FLOPs per update, trainable parameters, decision latency and FLOPs.

## Primary comparisons

- P1: RAID v7 (no training) vs VINN-style k-NN (best training-free baseline on development, MSE 0.426).
- P2: best RAID variant vs each baseline (Head-only fine-tuning, Frozen-encoder probe (fresh head), LoRA, no source training, Full fine-tuning, STRAP-style retrieved training data, LoRA (TAIL/FADA style), VINN-style k-NN, REGENT retrieve-and-play, REGENT-style blend). At 25 updates: RAID+LoRA v7 (ours) (development MSE 0.308). At 500 updates: RAID+LoRA v7 (ours) (development MSE 0.290).
- P3: per-GPU-second curves (descriptive); error at 0.1, 0.5, 1, 10 and 100 GPU-seconds; GPU-seconds to reach LoRA's 500-update error.

## Inference

Paired difference ours minus baseline. Primary 95% interval: percentile bootstrap over the 6 task x seed clusters (10,000 resamples, seed 0). Also: demo-level interval and settings won out of 6.
Decision rule: better if the cluster interval is entirely below 0, worse if entirely above 0, else no clear difference. All comparisons are reported; no tuning after the run.
