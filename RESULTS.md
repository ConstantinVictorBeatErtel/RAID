# RAID results: current evidence boundary

Updated 11 October 2026. The intended protocol is raid_hybrid_generalization_v6, testing frozen, RAID, LoRA and RAID + LoRA. No v6 implementation result, scientific measurement or hybrid advantage has been established. [STATUS.md](STATUS.md) is the current evidence summary.

## Protocol v6: few-shot inverse-dynamics adaptation on held-out LIBERO-Goal tasks (18 October 2026)

One frozen offline run on the 2 sealed final tasks x 3 seeds, analysis plan frozen before the data was read. Chunk MSE on normalized actions (lower is better), mean over 6 settings:

| Method | No training | 25 updates | 500 updates | GPU s (500 updates) |
|---|---|---|---|---|
| RAID+LoRA v7 (ours) | 0.500 | 0.272 | 0.281 | 201 |
| RAID v7 head-only (ours, exploratory) | 0.500 | 0.273 | 0.287 | 2.7 |
| LoRA (TAIL/FADA style) | 1.055 | 0.315 | 0.283 | 189 |
| Full fine-tuning | 1.055 | 0.313 | 0.268 | 444 |
| Head-only fine-tuning | 1.055 | 0.320 | 0.280 | 1.1 |
| VINN-style k-NN | 0.376 | – | – | 0 |

Frozen comparisons: RAID v7 vs k-NN without training is worse (P1 failed); RAID+LoRA v7 at 25 updates is better than all 9 baselines in 6/6 settings; at 500 updates it ties with most baselines and is worse than full fine-tuning. Offline action error only, not task success; `held_out_claims_eligible=false`. Full tables, intervals and limits: [FINAL_RESULTS.md](artifacts/raid_hybrid_libero_goal_v6/compute_bench_final_20261017/FINAL_RESULTS.md); summary in [STATUS.md](STATUS.md).

## Historical results and protocol notes

All numbers below retain their original experiment identity. Prior “active v4” labels refer to the earlier snapshot; neither those results nor v3 implementation checks are v6 evidence.

---

# RAID results and provenance

**Protocol boundary (25 September 2026):** this file records earlier GR-1/LIBERO, GRPO, RoboMimic, and V-JEPA measurements. None evaluates the active few-shot LIBERO-Goal study in [RESEARCH.md](RESEARCH.md). Do not use historical scores as evidence for LoRA adaptation on held-out tasks. No new-protocol results have been produced. Preserve every old number and provenance note as historical evidence.

**Protocol labels (11 October 2026):** every number in this file belongs to an earlier protocol (historical tracks, or v3 implementation evidence). The active protocol is v4 (`raid_world_model_idm_v4`). **No v4 result exists.** v4 results must be recorded separately and never combined with the numbers below.

## Earlier protocol v3 status — 11 October 2026

No scientific development or confirmation result exists for protocol v3. The complete synthetic structural graphs, retained recovery/reporting checks, real CPU smoke and current focused cloud checks are implementation evidence, not measurements of task transfer. CUDA calibration remains unexecuted. See [STATUS.md](STATUS.md) for evidence classes and [CURSOR_HANDOFF.md](CURSOR_HANDOFF.md) for the remaining path to a run. All historical numbers below remain unchanged.

---

This file separates the current-main GR-1/LIBERO artifacts from later draft-PR run notes and from the historical RoboMimic/V-JEPA tracks. Numbers are preserved as recorded; this cleanup does not rerun or scientifically revalidate them.

## Track 1 — GR-1 + RAID on LIBERO-Spatial

The paper-facing implementation freezes 384-dimensional GR-1 features and compares `direct_visual` with `raid_visual`. The committed `configs/results_libero.json` contains one point estimate per condition and nominal demonstration scale. The corresponding loss-curve files contain the same minima.

### Committed `main` results

| Nominal demonstration scale | `direct_visual` | `raid_visual` | Direct/RAID ratio |
| ---: | ---: | ---: | ---: |
| 25 | 0.842 | **0.132** | **6.4x** |
| 50 | 0.637 | **0.154** | 4.1x |
| 100 | 0.570 | **0.169** | 3.4x |
| 200 | 0.552 | **0.171** | 3.2x |

Source: [`configs/results_libero.json`](configs/results_libero.json), with full-precision ratios calculated from that file. The exact values are 0.8419656/0.1319512, 0.6371669/0.1543217, 0.5699734/0.1687350, and 0.5522584/0.1714185. Each value matches the minimum validation loss in its committed `configs/loss_curves_*_libero.json` file.

These are offline validation MSE values on normalized actions. They do not establish closed-loop task success. The implementation questions in [docs/RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md) must be resolved before treating them as publication-grade evidence.

### Later draft-PR run note

The later run note on [PR #1’s inspected branch](https://github.com/ConstantinVictorBeatErtel/RAID/blob/b493d87fe7c8b2c78b738a9ce1167ef52dae396f/GRPO_FINAL_RUN.md) reports a separate fresh sweep:

| Nominal demonstration scale | Direct visual | RAID visual |
| ---: | ---: | ---: |
| 25 | 0.852 | **0.131** |
| 50 | 0.639 | **0.158** |
| 100 | 0.580 | **0.171** |
| 200 | 0.554 | **0.174** |

These values are **reported in a run note, not independently reproduced in this cleanup**. The old README combined the later run’s 25-demo pair with the committed-main values for the other scales; the tables above keep the records separate.

## Track 2 — GRPO evidence

The current-main log is [`configs/grpo_libero_log.json`](configs/grpo_libero_log.json) at commit [`9cb2dd1`](https://github.com/ConstantinVictorBeatErtel/RAID/blob/9cb2dd14d40c7a4aaaa5a7a9da7e14386eb92b62/configs/grpo_libero_log.json). It has 86 logged records, best mean reward approximately `-3.153`, and zero recorded success.

The later draft-PR branch preserves two additional runs and their raw artifacts:

| Run | Records / updates | Best mean reward | Peak group success | Evidence |
| --- | ---: | ---: | ---: | --- |
| Full run | 382 | 0.925 at update 294 | 0.25 | [`GRPO_FINAL_RUN.md`](https://github.com/ConstantinVictorBeatErtel/RAID/blob/b493d87fe7c8b2c78b738a9ce1167ef52dae396f/GRPO_FINAL_RUN.md) |
| Polish run | 195 | 1.226 at update 158 | 0.25 | [`grpo_final_summary.json`](https://github.com/ConstantinVictorBeatErtel/RAID/blob/b493d87fe7c8b2c78b738a9ce1167ef52dae396f/raid_grpo_final/grpo_final_summary.json) |

The polish run’s final 25-update averages returned to zero success. Its peak is a within-group rate over four rollouts, not a stable benchmark success rate. The checkpoint and runner snapshots remain outside `main` until their provenance and compatibility changes are deliberately reconciled.

## Track 3 — RoboMimic Lift (low-dimensional state)

This earlier track uses normalized proprioceptive transitions and a different `src/train.py`/`src/evaluate.py` path. The committed [`configs/results.json`](configs/results.json) reports:

| Condition | 25 demos | 50 demos | 100 demos | 200 demos |
| --- | ---: | ---: | ---: | ---: |
| mean baseline | 0.850 | 1.069 | 1.096 | 0.919 |
| nearest neighbor | 0.617 | 0.744 | 0.717 | 0.567 |
| direct MLP | **0.336** | **0.358** | **0.296** | **0.183** |
| RAID gated decoder | 0.397 | 0.398 | 0.340 | 0.218 |
| RAID cross-attention | 0.404 | 0.399 | 0.347 | 0.219 |

The architecture-search log is [`configs/autoresearch_log.md`](configs/autoresearch_log.md). Its historical instructions are now [docs/history/vjepa-program.md](docs/history/vjepa-program.md); that note is not the source for this low-dimensional table.

## Track 4 — V-JEPA and `v2` exploration

The V-JEPA/DINO/SigLIP metrics and the multi-dataset `v2` matrix remain in their original files. They use different feature encoders, datasets, splits, and runners, so they are not combined with the GR-1/LIBERO table. See [docs/HISTORY.md](docs/HISTORY.md), [`src/autoresearch_libero.py`](src/autoresearch_libero.py), and [v2/README.md](v2/README.md).

## Reproduction pointers

Commands and external prerequisites are documented in [docs/REPRODUCING.md](docs/REPRODUCING.md). Reading these committed files is safe; the experiment commands require datasets, pretrained weights, external checkouts, and a suitable GPU environment. Experiments were not run as part of this cleanup.
