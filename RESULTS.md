# RAID — Results & experiment log

Consolidated results for both experiment tracks in this repository. Every number below comes from committed artifacts (`configs/results.json`, training logs, or the README tables) — nothing invented.

For architecture search iteration detail on the low-dimensional track, see [`configs/autoresearch_log.md`](configs/autoresearch_log.md).

---

## Track 1 — RoboMimic Lift (low-dimensional state)

Controlled comparison of inverse-dynamics models on proprioceptive \((s_t, s_{t+1})\) pairs. Metric: **validation MSE** on normalized 7-DOF actions. Train/val split: **80/20 by demonstration** at scales **25 / 50 / 100 / 200** demos, seed **42**.

### Final evaluation (`configs/results.json`)

| Condition | 25 demos | 50 demos | 100 demos | 200 demos |
|-----------|----------|----------|-----------|-----------|
| mean baseline | 0.850 | 1.069 | 1.096 | 0.919 |
| nearest neighbor (kNN pooled) | 0.617 | 0.744 | 0.717 | 0.567 |
| **direct_mlp** | **0.336** | **0.358** | **0.296** | **0.183** |
| **raid** (gated decoder) | **0.397** | **0.398** | **0.340** | **0.218** |
| raid_crossattn | 0.404 | 0.399 | 0.347 | 0.219 |

Retrieval hit rate for RAID conditions: **1.0** at all scales.

### Autoresearch on the RAID decoder

Eight architecture iterations on `RAIDDecoder` only (`program.md`), each trained @25 demos:

| Model @25 demos | Best val MSE |
|-----------------|--------------|
| Direct MLP (baseline) | **0.336** |
| RAID (original concat decoder) | ~0.444 |
| **RAID after autoresearch** | **0.397** (0.396789) |

Accepted design: sigmoid gate blending a transition-only inverse branch with the pooled retrieval prior, plus prior-path dropout (p=0.5) and Gaussian prior noise (σ=0.1) during training. RAID improved materially (0.44 → 0.40) but did not surpass direct MLP at the autoresearch metric.

Figures: `python3 notebooks/02_results.py` → `notebooks/figures/`.

---

## Track 2 — LIBERO-Spatial (GR-1 visual encoder)

Frozen **GR-1** (384-dim features) + cross-attention **RAIDDecoderVisual** vs a **DirectMLPVisual** baseline. LIBERO-Spatial: 10 pick-and-place tasks, 50 demonstrations each.

### Stage 1 — offline behaviour cloning (validation MSE ↓)

| Demo scale | `direct_visual` | `raid_visual` | Improvement |
|------------|----------------|---------------|-------------|
| 25 demos | 0.842 | **0.132** | **6.4×** |
| 50 demos | 0.637 | **0.154** | 4.1× |
| 100 demos | 0.570 | **0.169** | 3.4× |
| 200 demos | 0.552 | **0.171** | 3.2× |

Retrieval-augmented cross-attention provides a **6× MSE reduction** at 25 demos.

### Stage 2 — GRPO online fine-tuning

| Metric | Value |
|--------|-------|
| Updates completed | 86 / 100 |
| Starting mean reward | −3.881 |
| Best mean reward | −3.153 (update 59) |
| Improvement | +18.8% |
| Success rate | **0.00** |

The policy learned to move the end-effector closer to the target (shaped reach reward improved 18.8%), but **never completed the task** (SR = 0 throughout).

### Why GRPO stalled (SR = 0)

1. **Episode horizon too short** — pick-and-place needs ~80–150 steps; osmesa CPU rendering limited rollouts to **max_steps=30** (~337 ms/step).
2. **Weak GRPO signal** — with only shaped reach reward and no successes, there is no success/failure contrast for GRPO to exploit.
3. **Train/inference feature mismatch** — BC used ground-truth `(feat_t, feat_next)`; rollouts use GR-1-predicted `feat_next`.
4. **Rendering constraint** — osmesa on the training GPU cannot be fixed without EGL or a different renderer.

> **TODO:** re-run GRPO with EGL rendering and `max_steps=150` on a machine where GPU-accelerated MuJoCo is available. Checkpoints are not in git — re-train with `src/run_all_libero.py` or restore from local storage.

---

## Reproducing results

**Low-dim sweep:**
```bash
python3 src/run_all.py
python3 notebooks/02_results.py
```

**LIBERO visual sweep:**
```bash
python3 src/cache_gr1_features.py --dataset_dir data/libero_spatial/libero_spatial --output_dir data/libero_spatial/features --device cuda
python3 src/run_all_libero.py --feature_dir data/libero_spatial/features --device cuda
```

**GRPO (needs EGL + longer horizon):**
```bash
python3 src/grpo_libero.py --feature_dir data/libero_spatial/features --model_path models/raid_visual_50demos_best.pt --device cuda
```

---

## Artifacts in git vs local

| In git | Not in git (re-generable) |
|--------|---------------------------|
| `configs/results.json` | `models/*.pt` checkpoints |
| `configs/autoresearch_log.md` | `configs/norm_stats_*.pt` |
| Loss-curve JSON under `configs/` | `logs/*.log` |
| Source under `src/`, `notebooks/` | LIBERO HDF5 under `data/` |

Checkpoints and norm stats are gitignored. Re-run training or copy from your experiment machine.
