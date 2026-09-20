# RAID experiment artifacts

This directory contains tracked configuration notes and recorded outputs. Generated datasets, feature caches, normalization tensors, checkpoints, and logs are ignored unless a small summary is explicitly committed.

## Configuration and instructions

- [`v6.yaml`](v6.yaml) documents nominal hyperparameters for the original low-dimensional Lift pipeline. The Python CLI remains the executable source of truth.
- [`../docs/history/vjepa-program.md`](../docs/history/vjepa-program.md) records the historical V-JEPA experimental program.
- [`autoresearch_log.md`](autoresearch_log.md) records the low-dimensional RAID decoder search.
- `autoresearch_*_models.py` are snapshots of selected architecture-search configurations.

## Recorded metrics

- [`results_libero.json`](results_libero.json) contains the aggregate GR-1/LIBERO `direct_visual` and `raid_visual` validation MSE values used in [RESULTS.md](../RESULTS.md).
- [`results.json`](results.json) contains the earlier RoboMimic low-dimensional sweep.
- [`results_vjepa.json`](results_vjepa.json) and the `val_mse_*_vjepa.json` files contain V-JEPA-era point metrics.

## Loss curves

- `loss_curves_*_libero.json` are per-epoch curves for the GR-1/LIBERO visual sweep.
- `loss_curves_*_vjepa.json` are curves from the V-JEPA exploration.
- `loss_curves_{condition}_{n}demos.json` without the `_libero` suffix belong to the earlier low-dimensional pipeline.

## GRPO logs

- [`grpo_libero_log.json`](grpo_libero_log.json) is the current-main LIBERO probe log.
- [`grpo_log.json`](grpo_log.json) is an earlier low-dimensional GRPO log.
- The later full and polish GRPO logs are on the draft branch documented in [RESULTS.md](../RESULTS.md), not in `main`.

## Generated and external artifacts

The scripts may produce `norm_stats_*.pt`, model checkpoints under `models/`, feature caches under `data/`, and local logs. Those assets are not required to understand the committed metrics and must be regenerated or restored from the relevant experiment environment. See [REPRODUCING.md](../docs/REPRODUCING.md) for the current command shapes and limitations.
