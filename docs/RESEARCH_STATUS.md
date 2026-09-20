# Research status before a publication rerun

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

None of this deferred work is performed by the repository cleanup.
