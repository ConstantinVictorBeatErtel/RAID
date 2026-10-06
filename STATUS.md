# RAID acceptance status

Updated 4 October 2026. RAID is **not ready for scientific launch**. Protocol v3,
format-v4 artifacts, standalone MAE ViT-B/16, and recorded RGB remain fixed.
GR-1 is permanently prohibited. Paid execution remains disabled.

| Requirement | Status | Evidence and limits |
| --- | --- | --- |
| Real assets and complete cache | **Passed locally** | Fresh `readiness_acceptance_20261004/verify-assets.json`: ten tasks × 50 usable demos; all 500 shards audited, including 80 label-free query shards. Pinned clean LIBERO checkout and official MAE checkpoint match their recorded hashes. |
| Compile and complete local suite | **Passed locally** | 51 tests and 6 recovery subtests passed in 211.29 seconds; one existing Transformer warning. Logs and JUnit: `artifacts/fewshot_libero_goal_v3/readiness_acceptance_20261004/pytest-final.*`. Compilation and whitespace checks pass. |
| Full-length production regression | **Passed; synthetic only** | Production caching over coherent unequal-length RGB/action HDF5; 2,000 source updates; 1,000 updates for each learned method at batch 64/microbatch 8; all eight methods evaluated at 200/500/1,000. A tiny explicitly synthetic image encoder makes this CPU-feasible. The production decoder, optimizer, dropout, sampling, snapshots, and evaluator run unchanged. It is scientifically ineligible. |
| Source recovery and canonical root | **Passed locally** | Completed source recovery copies the exact archived checkpoint bytes and receipt into a fresh directory, retaining the original canonical study-root binding. Relocated checkpoints resolve the same ledger. |
| Head/LoRA scheduled recovery | **Passed for seed 42, five-shot; synthetic only** | Every 200/500/1,000 snapshot recovers in a fresh directory. Weights, optimizer, all RNG states, predictions, and snapshot hashes match. Real subprocess SIGINT/SIGTERM, immutable aliases, idempotent retries, and conflicting-snapshot rejection pass. |
| Sampling and query isolation | **Passed locally** | All three seeds and all three budgets: nested supports, identical head/LoRA target sampling schedules, unequal demo lengths, and actual HDF5 query-action read instrumentation. Invalid final conditions fail before labels open. |
| Recovery across remaining seed/budget combinations | **Unexecuted** | Sampling parity covers the nine combinations; the full interruption/recovery cross-product does not yet. |
| Production-generated freeze → confirmation → report | **Unexecuted as a complete positive flow** | Generated synthetic artifacts fail scientific entrypoints as required. Positive freeze/registration/report unit tests still assemble structural evidence; they are not production-generated scientific evidence. |
| Real MAE CPU smoke | **Passed; diagnostic only** | Fresh `readiness_cpu_20261004_acceptance/smoke_real.json`: raw/cache max error 5.25e-6, zero-adapter error 0, frozen backbone and saved-live/reloaded predictions pass. Python 3.13.7 / PyTorch 2.11.0 / timm 1.0.27, macOS ARM. `cuda_verified=false`, `scientific_result=false`. |
| Real CPU all-method evaluation | **Passed; calibration only** | Fresh `readiness_cpu_20261004_acceptance/smoke_workspace/runs/put_the_cream_cheese_in_the_bowl/5shot/seed42/evaluation_head2_lora2.json`: all eight methods use identical 2,132 recorded query IDs. Two-update calibration checkpoints; no scientific or CUDA eligibility. |
| Linux CUDA dependency resolution | **Passed; resolver only** | 137 packages resolved for Linux x86-64 / Python 3.10, including torch 2.1.2+cu118 and torchvision 0.16.2+cu118. Hash lock and resolver log retained under `readiness_acceptance_20261004`. No target import/GPU verification is implied. |
| Pinned CUDA runtime, smoke, and calibration | **Unexecuted** | Target imports, driver/CUDA compatibility, real MAE forward/backward, and CUDA smoke/calibration still require target hardware. |
| Local calibration implementation checks | **Passed locally** | Production 5-warmup + 20-measured sessions for source/head/LoRA; initialization separate from updates; source validation and actual serialization/publication timing; all eight evaluator timing outputs; missing/nonfinite/partial measurement rejection. Measurement bundle format 3 requires complete resource/input evidence and a matching receipt. No completed CUDA receipt exists. |
| Workload and budget arithmetic | **Passed locally** | Six baseline bundles, 36 candidate evaluations, 42 development retention passes, 54/18 confirmation retention passes, 60 source-validation passes, 801/465 checkpoint writes before selection. Tests include frozen per-method steps, initialization/recovery/storage, actual prior-spend input, 25% margin, protected $25, and $250 ceiling. |
| Verified prices and real forecasts | **Unexecuted** | No complete CUDA bundle, verified pricing file, or recorded actual prior-spend value. Both scope forecasts remain gated. |
| Complete cloud lifecycle review | **Unexecuted** | Ownership/cleanup and collection unit tests pass; live provider preflight and end-to-end lifecycle review remain outstanding. Paid launch is disabled. |
| Scientific development / confirmation results | **Unexecuted** | No first five-shot development experiment, frozen scientific selection, or confirmation has run. |

Current CPU smoke code digest:
`70d3b73c7d029d57df299745e4e50d69b7a6930b9b5049f35b242a623b80e6a7`.
Study configuration digest:
`acbc84b807392675a7d012973b09e6fb86070b10cd467f1a61dd757127804ba5`.
Protocol digest:
`9a2f54a8ccbdecbb7c487578e0b61aff0f4fa3748cc042712975eebbca5e597d`.
Cache manifest SHA-256:
`f7637833bbe5295d04bca19c2afa33c8db14e1a8628c140d20da1e8e82341661`.

The earlier 44-test/CPU artifacts remain historical diagnostics. Refresh evidence
after source/input changes. The exact remaining implementation and verification
sequence is in [READINESS_HANDOFF.md](docs/READINESS_HANDOFF.md).

Machine-readable, hash-bound local evidence: [readiness.json](artifacts/fewshot_libero_goal_v3/readiness_acceptance_20261004/readiness.json). Its `launch_ready` value remains `false`.
