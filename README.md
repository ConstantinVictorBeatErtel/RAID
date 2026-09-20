# RAID: Retrieval-Augmented Inverse Dynamics for Robotic Manipulation

RAID studies the inverse-dynamics step for robot world models: given a current visual state and a dreamed next visual state, what 7-DOF motor command should a robot execute? The current paper track freezes GR-1 visual features, retrieves similar demonstrated transitions, and decodes a normalized action with a direct trunk, cross-attention prior, and per-dimension gate.

**Paper:** [RAID project report](paper/RAID_Report_vf.pdf)

**Project page:** <https://constantinvictorbeatertel.github.io/RAID/>

## Research status

This repository records a completed course-project implementation and its experiment artifacts. It is being cleaned before a publication-oriented rerun. The committed results are useful evidence, but they are not yet a verified arXiv reproduction: the inspected main pipeline has unresolved split, normalization, provenance, and implementation-documentation questions. See [research status](docs/RESEARCH_STATUS.md) for the evidence and deferred work.

The headline values below are the point estimates stored in `configs/results_libero.json`. They are reported as historical project results and should not be read as a claim that the publication experiment has already been rerun.

| Demonstrations | Direct visual MLP | RAID visual | Direct/RAID ratio |
| --- | ---: | ---: | ---: |
| 25 | 0.842 | **0.132** | **6.4x** |
| 50 | 0.637 | **0.154** | 4.1x |
| 100 | 0.570 | **0.169** | 3.4x |
| 200 | 0.552 | **0.171** | 3.2x |

These are offline validation MSE values on normalized actions. They are not closed-loop task-success measurements. The later GRPO artifacts on the draft PR report a different run; both records are kept separately in [RESULTS.md](RESULTS.md).

## Current paper implementation

The active GR-1/LIBERO path is the code imported by the LIBERO trainer and sweep:

| Path | Role |
| --- | --- |
| `src/gr1_encoder.py` | Frozen GR-1/MAE encoder and predicted-next-feature wrapper |
| `src/data_libero.py` | LIBERO HDF5 loading, action normalization, and cached-feature datasets |
| `src/memory.py` | Dense cosine retrieval over `(feat_t, feat_next, action)` transitions |
| `src/models.py` | Direct visual baseline and cross-attention RAID visual decoder |
| `src/train_libero.py` | Offline behavior-cloning trainer |
| `src/run_all_libero.py` | Sweep over `direct_visual` and `raid_visual` |
| `src/rollout_libero.py` | Closed-loop LIBERO rollout helpers using predicted next features |
| `src/grpo_libero.py` | GRPO probe for online refinement |
| `src/cache_gr1_features.py` | GR-1 feature-cache generation |

Offline training uses encoded observed consecutive frames from demonstrations. Rollouts instead call GR-1 to predict the next feature before decoding an action; that train/inference distinction is a documented research issue, not a hidden implementation detail.

`src/models_libero.py`, `src/memory_libero.py`, `src/vjepa_encoder.py`, and `src/autoresearch_libero.py` belong to the earlier V-JEPA exploration. Their names do not identify the active GR-1 implementation. The `v2/` subtree is a separate multi-dataset expansion covering DINOv2, Theia, RoboMimic, LIBERO, transformer, and diffusion-policy experiments; it is preserved as historical context rather than presented as the paper’s main result.

## Reading guide

- [Results and provenance](RESULTS.md) — current-main metrics, later draft-PR run notes, GRPO records, and historical tracks.
- [Research status](docs/RESEARCH_STATUS.md) — implementation/documentation discrepancies that must be resolved before a controlled publication rerun.
- [Reproducing](docs/REPRODUCING.md) — commands and external prerequisites, with environment-specific limitations called out.
- [History](docs/HISTORY.md) — how the low-dimensional, V-JEPA, and `v2` tracks relate.
- [Paper directory](paper/README.md) — status of the committed PDF and missing editable source.

## Historical tracks

The repository contains the original RoboMimic low-dimensional sweep, the V-JEPA/DINO/SigLIP exploration, and the later `v2` multi-dataset expansion. These artifacts explain how the GR-1/LIBERO direction was selected, but their metrics use different encoders, datasets, splits, and evaluation code. They remain available for audit and are not merged into the GR-1 result table.

The MIT license is in [LICENSE](LICENSE). The cleanup intentionally leaves the website, report PDF, executable research code, raw result files, figures, notebooks, and checkpoints unchanged.
