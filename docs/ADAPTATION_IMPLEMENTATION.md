# Adaptation implementation and evidence

Updated 4 October 2026. Protocol v3, standalone MAE ViT-B/16 recorded-RGB features,
and cache/checkpoint/continuation/prediction format v4 remain active. GR-1 is
prohibited by `AGENTS.md` and `RESEARCH.md`. Paid execution is disabled.

## Repairs verified locally

- Snapshot serialization normalizes incidental Python aliasing before first
  publication. Uninterrupted and resumed snapshots now have identical hashes.
  Recovery and selected aliases copy the archived bytes. Completed source
  continuations also retain their exact snapshot bytes and original study root.
- Canonical ledger resolution follows the source checkpoint's persisted binding,
  including relocated recovered source checkpoints. Scientific source validation
  still checks provenance, architecture, weights, and completion receipts.
- Immutable evaluation reports exclude attempt-specific timing data. Repeated
  evaluations retain identical acceptance evidence; timings are returned and
  stored in separate performance records. Prediction payloads explicitly carry
  execution mode; synthetic predictions cannot pass scientific validation.
- Reporting now computes bootstrap comparisons for both development and final-test
  roles; previously an early return skipped the second role.
- Production timing hooks record five warmup and twenty steady-state updates in
  one optimizer session, initialization separately, source validation, actual
  checkpoint writes/serialization/publication, and peak CUDA allocation. The
  collector consumes all eight evaluator performance records, hashes the full
  evaluated RGB input sequence, and checks per-method query-ID coverage.
- Measurement format 3 validates required timing/resource components for
  completeness and finite positive values before issuing a receipt. Recovery
  includes source and both adaptations. CPU/MPS remains ineligible for a CUDA
  completion receipt.
- The shared workload inventory includes source validation and individual
  checkpoint writes, and accepts each frozen method's confirmation update count.
  Before selection the CLI conservatively uses 1,000 updates. Forecasts report
  both scopes, include initialization and source/adaptation recovery, and scale
  retained artifact storage to the training inventory. Before selection: 801
  all-budget or 465 five-shot checkpoint writes; 60 source-validation passes;
  42 development and 54/18 confirmation retention passes.

## Acceptance evidence

| Gate | Status | Evidence and limit |
| --- | --- | --- |
| Local compile/tests | **Passed** | 51 tests, 6 recovery subtests, one legacy Transformer warning; 211.29 seconds. `readiness_acceptance_20261004/pytest-final.*` retains results. |
| Real assets/cache | **Passed locally** | Fresh audit of all 500 shards, including 80 label-free query shards; ten tasks with 50 demos each, pinned MAE and clean LIBERO checkout. |
| Full-length synthetic training/evaluation | **Passed; synthetic only** | Production HDF5/cache path, 2,000 source updates, both 1,000-update adaptations, all eight methods at every 200/500/1,000 snapshot. Tiny synthetic encoder, real production decoder/optimizer/dropout/sampling and persistence. |
| Scheduled interruption/recovery | **Passed for seed 42/five-shot** | Head and LoRA, all scheduled snapshots, fresh directories, real subprocess signals, identical weights/optimizer/RNG/predictions/hashes, retries and conflicting evidence. Full cross-product over remaining seeds/budgets is **unexecuted**. |
| Sampling/data isolation | **Passed locally** | All seeds × budgets, unequal demo lengths, nested supports, head/LoRA sample parity, instrumented HDF5 query-label reads, invalid final conditions. |
| Generated freeze/registration/report flow | **Unexecuted as complete positive flow** | Synthetic rejection is tested. Positive structural tests still assemble payloads. Do not count those as production scientific evidence. |
| Real CPU smoke | **Passed; diagnostic only** | `readiness_cpu_20261004_acceptance/smoke_real.json`; raw/cache error 5.25e-6, zero-adapter error 0, live frozen-weight and reload equivalence pass. Python 3.13.7, torch 2.11.0, timm 1.0.27. |
| Real CPU all-method evaluation | **Passed; calibration only** | Fresh `readiness_cpu_20261004_acceptance/smoke_workspace/runs/put_the_cream_cheese_in_the_bowl/5shot/seed42/evaluation_head2_lora2.json`: all eight methods use identical 2,132 recorded query IDs. Two-update calibration checkpoints; no scientific or CUDA eligibility. |
| CUDA dependency resolution | **Passed; resolver only** | Linux/Python 3.10/CUDA 11.8 hash lock resolves 137 packages. Target imports and GPU execution are **unexecuted**. |
| Local collector timing/schema | **Passed locally** | Actual production 5+20 hooks and all evaluator timing outputs; incomplete/invalid components rejected. Successful CUDA collector path is **unexecuted**. |
| Real forecast / cloud launch | **Unexecuted** | CUDA calibration, verified prices, actual prior spend, and full cloud lifecycle review remain required. Launch stays disabled. |
| Scientific results | **Unexecuted** | No development or confirmation study has run. |

Current code digest:
`70d3b73c7d029d57df299745e4e50d69b7a6930b9b5049f35b242a623b80e6a7`.
Configuration digest:
`acbc84b807392675a7d012973b09e6fb86070b10cd467f1a61dd757127804ba5`.
See [STATUS.md](../STATUS.md) for exact input hashes and
[READINESS_HANDOFF.md](READINESS_HANDOFF.md) for remaining gates and checks.
Historical artifacts remain audit-only; new code requires fresh evidence.

Machine-readable, hash-bound local evidence: [readiness.json](../artifacts/fewshot_libero_goal_v3/readiness_acceptance_20261004/readiness.json). Its `launch_ready` value remains `false`.
