# Reproducing the recorded RAID tracks

This document describes the current command shapes and their external requirements. It is not a verified one-command installation recipe. No experiment was run during the cleanup.

## Read committed results without running experiments

The safest review path is to read [`configs/results_libero.json`](../configs/results_libero.json), the matching `configs/loss_curves_*_libero.json` files, [`configs/grpo_libero_log.json`](../configs/grpo_libero_log.json), and [RESULTS.md](../RESULTS.md). These are static artifacts. For a syntax-checked JSON view, use `python3 -m json.tool configs/results_libero.json`.

## Historical GR-1 feature extraction

The current-main cache command is:

```bash
python3 src/cache_gr1_features.py \
  --dataset_dir data/libero_spatial/libero_spatial \
  --output_dir data/libero_spatial/features \
  --gr1_ckpt checkpoints/gr1/snapshot_ABCD.pt \
  --mae_ckpt checkpoints/gr1/mae_pretrain_vit_base.pth \
  --device cuda \
  --batch_size 128
```

The parser also accepts the defaults shown in `src/cache_gr1_features.py`. The command requires:

- LIBERO-Spatial HDF5 files under the dataset directory;
- the public GR-1 checkout expected by `src/gr1_encoder.py` at `/home/ubuntu/GR1`;
- the GR-1 checkpoint and MAE ViT-base weights under `checkpoints/gr1/`;
- PyTorch, CUDA, `h5py`, NumPy, torchvision, CLIP, and the GR-1 dependencies.

The cache writes per-task `.pt` files, `manifest.json`, and `norm_stats.pt` under ignored/generated paths. The normalization and split caveats are recorded in [research status](RESEARCH_STATUS.md).

## Historical offline behavior cloning

After the cache exists, the current-main sweep command is:

```bash
python3 src/run_all_libero.py \
  --feature_dir data/libero_spatial/features \
  --device cuda \
  --epochs 100
```

The sweep invokes `src/train_libero.py` for `direct_visual` and `raid_visual` at nominal scales 25, 50, 100, and 200. The trainer’s own options are `--condition`, `--n_demos`, `--feature_dir`, `--epochs`, `--batch_size`, `--lr`, and `--device`; its defaults are visible in the parser. It writes ignored checkpoints under `models/`, loss curves under `configs/`, and the aggregate result file under `configs/results_libero.json`.

The active visual model and retrieval imports are from `src/models.py` and `src/memory.py`. The similarly named `*_libero.py` model/memory files belong to the V-JEPA exploration.

## Current-main GRPO probe

The current-main script has no `--model_path` option. It constructs `models/raid_visual_{n_demos}demos_libero_best.pt` from `--n_demos`, with a fallback to the last matching checkpoint if the exact file is absent. The actual parser options are:

```bash
MUJOCO_GL=egl python3 src/grpo_libero.py \
  --n_demos 200 \
  --n_updates 200 \
  --G 4 \
  --beta 0.04 \
  --task_idx 0 \
  --feature_dir data/libero_spatial/features \
  --dataset_dir data/libero_spatial/libero_spatial \
  --gr1_ckpt checkpoints/gr1/snapshot_ABCD.pt \
  --mae_ckpt checkpoints/gr1/mae_pretrain_vit_base.pth \
  --device cuda \
  --log_every 5
```

This requires a working LIBERO/MuJoCo installation, EGL-capable rendering, the external GR-1 checkout, cached features, normalization statistics at `data/libero_spatial/libero_spatial/norm_stats.pt` for the command above, and a behavior-cloning checkpoint in `models/`. The rollout helper contains the hardcoded Ubuntu BDDL path `/home/ubuntu/LIBERO/libero/libero/bddl_files/libero_spatial`; this is an environment limitation, not a portable setup instruction.

The later 382-update full run and 195-update polish run are preserved on the draft branch [`b493d87`](https://github.com/ConstantinVictorBeatErtel/RAID/tree/b493d87fe7c8b2c78b738a9ce1167ef52dae396f), together with a GR-1 compatibility patch and runner snapshots. The branch-specific command shape was:

```bash
cd ~/RAID
MUJOCO_GL=egl python3 raid_grpo_final/grpo_libero_remote_final.py
```

Treat that branch as historical evidence until its code and checkpoint provenance are reconciled with `main`.

## Historical setup script and assets

[`scripts/setup_and_run.sh`](../scripts/setup_and_run.sh) is an environment-specific historical script. It downloads checkpoints, installs packages, invokes system package management, checks EGL, caches features, runs the BC sweep, and generates videos. It assumes `~/RAID`, Ubuntu tools, network access, and `sudo`; it is not a verified installation recipe and should not be run as part of this cleanup.

Keep these asset classes separate when reconstructing an experiment:

| Asset | Current location or source |
| --- | --- |
| Dataset | `data/libero_spatial/libero_spatial/` (ignored, not committed) |
| GR-1 checkout | `/home/ubuntu/GR1` in the inspected code |
| LIBERO checkout | `/home/ubuntu/LIBERO/libero/libero` in rollout code |
| Pretrained weights | `checkpoints/gr1/` (ignored) |
| Cached features and normalization | `data/libero_spatial/features/`, `data/libero_spatial/libero_spatial/norm_stats.pt` (ignored) |
| Trained BC checkpoints | `models/` (ignored) |
| Recorded metrics | `configs/` (tracked) |

For the separate V-JEPA and `v2` tracks, start with [HISTORY.md](HISTORY.md), [the historical program note](history/vjepa-program.md), and [`v2/README.md`](../v2/README.md). Their encoders, data adapters, and artifact roots are different from the paper-facing GR-1 path.
