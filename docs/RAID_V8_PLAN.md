# RAID v8 study plan (written 22 October 2026, before any v8 data is read)

## The one claim

> **Learned retrieval for few-shot inverse dynamics fails when the source-trained prediction is wrong.
> A disagreement-aware gate fixes this failure: retrieval then never does worse than plain nearest-neighbour
> regression, and with a few head updates on frozen features it reaches the error of encoder fine-tuning at a
> small fraction of the compute — across 30 held-out LIBERO tasks.**

Everything in the study serves this claim. The v7 result on LIBERO-Goal (the wine-bottle failure) is the
motivation, not a result of this study.

## What is removed (bolt-off)

RAID v6, RAID+LoRA v6, RAID+LoRA v7, the STRAP-style, retrieve-and-play, frozen-encoder-probe and
LoRA-without-source baselines, the VPP2 pilot and generator, and the 15-method table. They do not test the claim.

## Methods (bolt-on and keep)

| Method | Role | Training on target |
|---|---|---|
| VINN-style k-NN | strongest training-free baseline | none |
| REGENT-style blend | heuristic distance gate (the natural competitor to a learned gate) | none |
| RAID v7 | learned gate without confidence inputs (the failing design) | none / head updates |
| **RAID v8 (ours)** | gate sees retrieval confidence and direct–retrieval disagreement; trained with direct-path corruption | none / head updates |
| Head-only fine-tuning | same source network, retrieval off | head updates |
| LoRA, full fine-tuning (MAE only) | encoder-tuning references | encoder + head updates |

RAID v8 gate input: the controller state plus `c = [top-1 similarity, mean valid similarity, attention entropy,
|direct − prior| per action element (detached)]`. During source training, with probability `p_corrupt`, the
direct prediction of a row is replaced by the direct prediction of another row of the batch before the mix (the
trunk gets no gradient from that row). The gate must then learn to use the prior when the direct path disagrees
with confident retrieval. The final variant (with or without corruption, `p_corrupt`) is selected on
development tasks only.

## Data and splits (step 1)

- Source: all 90 LIBERO-90 tasks (45 training + 5 validation demos each).
- Development: the 10 LIBERO-Goal tasks (already seen in earlier work, therefore used for design only).
- Final, sealed: the 30 tasks of LIBERO-10, LIBERO-Spatial and LIBERO-Object. All 30 are used; none is selected.
  Final files are downloaded and hashed in the final run only, after the freeze.
- Per target task: a seeded permutation of the 50 demos gives a support pool of 30 and a query set of 20.
  Support seeds 42, 43, 44 draw K = 5 support demos. Encoder-tuning references use seed 42 only.
- Data: Hugging Face `yifengzhu-hf/LIBERO-datasets` (about 100 GB in total), downloaded on Runpod only.

## Encoders (step 4)

MAE ViT-B/16 (primary, as before), DINOv2 ViT-B/14 (`facebook/dinov2-base`) and SigLIP ViT-B/16
(`google/siglip-base-patch16-224`). All give 768-d features. Each encoder is its own retrieval reference.
Hypotheses are tested with MAE; the other two encoders are robustness checks reported with the same statistics.

## Fair baselines (step 5)

Learning rates for head-only, RAID heads, LoRA and full fine-tuning are tuned on development tasks only, on a
three-value grid each. Source training length is set from development validation curves.

## Closed-loop test without a generator (step 3)

For each query demo of a final task: reset the simulator to the demo's recorded initial state; at each chunk,
find the demo frame that best matches the current observation (monotone search over a short window), use the
recorded frame 8 steps later as the target, predict 8 actions and execute them. Success is LIBERO's task
predicate within 1.5 × the demo length + 20 steps. This isolates the inverse-dynamics model. It is run for
k-NN, RAID v7, RAID v8 (zero training and 25 updates), head-only (25 updates), LoRA and full fine-tuning (200).

## Compute (step 6)

GPU-seconds per update and to target, FLOPs per update and per decision, and the amortized one-time cost (feature
cache and source training divided by the number of target tasks). The adaptation timing is repeated on a second
GPU type in a short timing-only run.

## Pre-registered hypotheses (frozen in an amendment before the final run)

Primary encoder MAE; 30 final tasks × 3 seeds; the paired difference is averaged per task, and the 95% interval
is a percentile bootstrap over the 30 tasks (10,000 resamples). Holm correction over H1–H3.

- **H1 (the fix):** zero-training RAID v8 has lower MSE than zero-training RAID v7.
- **H2 (no worse than retrieval):** zero-training RAID v8 has lower MSE than the best training-free baseline
  (k-NN or REGENT blend, chosen on development tasks).
- **H3 (few-shot efficiency):** RAID v8 after 25 head updates has lower MSE than head-only fine-tuning after 25
  updates.
- Descriptive: worst-task difference to k-NN; error versus GPU-seconds against LoRA and full fine-tuning;
  closed-loop success per method and its rank correlation with offline MSE; the same tests for DINOv2 and SigLIP.

A failed hypothesis is reported as failed.

## Order of work

1. Code and local CPU tests (no spend).
2. Development run on Runpod: download LIBERO-90 and LIBERO-Goal, cache 3 encoders, train sources, select the v8
   variant and learning rates, verify the closed-loop harness on development tasks.
3. Freeze the amendment (methods, settings, hypotheses, analysis code digest).
4. Final run (once): download and hash the 30 final tasks, run offline benchmark and closed loop.
5. Timing-only run on a second GPU type.
6. Rewrite the paper around the claim.
