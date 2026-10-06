# Implementation evidence and research status

Updated 25 September 2026. This page preserves the audit of the earlier main pipeline. The active direction is the few-shot LIBERO-Goal protocol in [RESEARCH.md](../RESEARCH.md), with status in [STATUS.md](../STATUS.md) and implementation acceptance checks in [ADAPTATION_IMPLEMENTATION.md](ADAPTATION_IMPLEMENTATION.md).

## New protocol status

The new direction compares a frozen source inverse-dynamics head, target transition retrieval, target action-head fine-tuning, and MAE LoRA plus head adaptation. It assigns six source, two development, and two final-test LIBERO-Goal tasks, with disjoint target support and query demonstrations. The protocol-manifest preparation code is in `raid_experiments/fewshot.py`; it verifies the official task set, hashes HDF5 inputs, preserves actual demonstration IDs, and writes a digest-bound immutable manifest. This preparer is separate from the trainer.

Source-head fitting, target head adaptation, retrieval controls, query evaluation artifacts, active-adapter retention metrics, and deterministic CPU continuation have synthetic HDF5/cache coverage. A full 500-shard real cache audit and a fresh real CPU MAE/LoRA reference smoke passed on 26 September 2026. The local suite now has 43 passing tests, including subprocess SIGINT/SIGTERM handling, recovery state comparisons, immutable source artifacts, candidate-baseline references, and hand-calculated bootstrap/forecast cases. This evidence does not establish the production-generated all-eight-method fixture, full-length training regression, every scheduled snapshot recovery, CUDA calibration, or scientific results; see the current acceptance matrix in [STATUS.md](../STATUS.md).

---

# Research status before a publication rerun

This is the implementation audit from the repository cleanup. The active research direction is in [RESEARCH.md](../RESEARCH.md); progress and ordered work are tracked in [STATUS.md](../STATUS.md).

This page records what the inspected `main` implementation and committed artifacts actually establish. It is an audit guide, not a correction of the experiment. Each item states the evidence, why it matters, and what is deferred.

## Findings in the inspected main pipeline

| Finding | Evidence | Implication | Deferred work |
| --- | --- | --- | --- |
| The RAID memory bank is populated from the first `demos_per_task` demonstrations, including demonstrations the loader assigns to validation. | `src/train_libero.py`: `populate_memory_from_cache` takes the first `demos_per_task` entries, while `_load_cached_datasets` reserves the last `n_val_per_task` entries for validation. | Validation retrieval can contain validation demonstrations in the inspected implementation. Do not silently describe the historical headline values as leakage-free. | Reconstruct the exact historical run code and rerun with an explicitly training-only memory bank; do not change this cleanup branch’s executable code. |
| Feature-cache action normalization is computed before the trainer split. | `src/cache_gr1_features.py` calls `compute_norm_stats(hdf5_files)` over the supplied files and saves `norm_stats.pt`; the report describes training-split statistics. | The implementation and report describe different normalization provenance. | Establish the intended split, regenerate statistics, and rerun under a recorded configuration. |
| A nominal `n_demos=25` is distributed across ten task files. | `src/train_libero.py` computes `demos_per_task = max(1, n_demos // n_tasks)`; with ten files this is two demonstrations per task, then one is held out per task. | The label `25 demos` does not mean 25 training demonstrations in this loader; it describes a nominal scale. | Decide and document the counting convention before the controlled rerun. |
| Offline and rollout next features differ. | `src/cache_gr1_features.py` encodes observed `image_next`; `src/rollout_libero.py` calls `GR1Encoder.predict_next_feat`. | Offline validation and closed-loop rollout have a feature-distribution mismatch. | Add an explicit paired evaluation protocol and measure the effect after the split/normalization decisions are fixed. |
| The trainer default learning rate differs from the report. | `src/train_libero.py` defaults to `--lr 3e-4`; the report states `1e-3` in its training recipe. | The exact run configuration is not recoverable from the current source and PDF alone. | Recover the run command/configuration or rerun with an explicitly recorded learning rate. |
| The report describes per-dimension metrics and transition metadata that are not fully emitted by the inspected main pipeline. | `configs/results_libero.json` contains aggregate `val_mse` only; `src/run_all_libero.py` writes aggregate values; cached `.pt` records include features, actions, `demo_lengths`, and `task_name`, but not task/demo/step identifiers. | Some report claims cannot be traced to a committed machine-readable artifact on `main`. | Locate the original visualization/evaluation artifacts or regenerate them with explicit metadata and per-action outputs. |
| Final GRPO artifacts and compatibility fixes are outside `main`. | Draft branch [`b493d87`](https://github.com/ConstantinVictorBeatErtel/RAID/tree/b493d87fe7c8b2c78b738a9ce1167ef52dae396f) contains `raid_grpo_final/`, `GRPO_FINAL_RUN.md`, and `third_party_patches/gr1_lambda_compat.patch`; PR #1 is an open draft with conflicts. | The later 195-update polish run is evidence from a separate branch, not a reproducible current-main result. | Reconcile provenance, code differences, and checkpoint ownership before integrating any artifact. |
| The V-JEPA autoresearch entrypoint imports a helper absent from the current data module. | `src/autoresearch_libero.py` imports `make_train_val_vjepa` from `data_libero.py`; that helper is not present in the inspected module. | The historical entrypoint cannot be treated as a verified runnable recipe. | Repair and test the historical path separately if it is needed for the paper’s background. |

## Publication checklist deferred until after cleanup

- Recover the editable manuscript source; `main` contains `paper/RAID_Report_vf.pdf` but no editable LaTeX or source document.
- Verify every citation, version, dataset statement, and external checkpoint reference against primary sources before adding references for arXiv.
- Decide the intended demonstration count, training-only memory-bank construction, normalization split, learning-rate provenance, and observed-versus-predicted feature protocol.
- Repair the reproducibility entrypoints and record a clean environment, commit, data manifest, seeds, and exact commands.
- Rerun offline baselines and RAID under the agreed protocol, with multiple seeds and complete per-action/per-task outputs where required.
- Rerun GRPO with a documented simulator/rendering setup and report stable success statistics separately from offline MSE.
- Update the manuscript and website only after the evidence is reconciled, then perform the arXiv packaging and final author review.

The historical checklist above records work that was deferred at the time of the original audit. The current repair branch addresses only the subset listed as verified below; it does not claim the external rerun or publication evidence is complete.

## Repair-branch status

The isolated `raid_experiments` package contains repairs for stable cross-loader identities, training-only normalization, atomic checkpoints, budget-derived deadlines, and hash-manifest collection. Twenty-seven repository tests pass, including non-contiguous demonstration IDs, raw-image collation, physical gripper metrics, mocked cloud behavior, and multi-task fixtures. These are regression checks, not proof of external model, simulator, or provider correctness.

The current branch also reproduced and repaired the following implementation defects: raw LoRA collation omitted transition IDs; split generation assumed contiguous demo numbering; raw and cached ordering diverged across tasks; GR-1 selected padded history; physical gripper metrics used normalized signs; and collection/deletion accepted stale or unverified artifacts. Remaining cloud, checkpoint, GR-1, simulator, Cosmos, and benchmark work is the ordered queue in [STATUS.md](../STATUS.md).

The package still requires external verification of checkpoint compatibility, LIBERO expert replay, EGL, and the Cosmos 3 Edge runtime. The default Runpod configuration therefore keeps paid launch disabled. Treat `STATUS.md` as the live queue; this document remains the evidence record for historical artifacts and known limitations.
