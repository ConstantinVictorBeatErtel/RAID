# Remaining launch gates (4 October 2026)

Paid execution stays disabled. Preserve protocol v3, format-v4 scientific inputs,
the GR-1 prohibition, downloaded assets, and historical evidence. The new local
regressions and CPU diagnostics are not scientific results.

## Local acceptance remaining

1. Complete the production-generated freeze → confirmation → report fixture.
   The full-length fixture now generates source and both learned snapshots,
   evaluates all eight methods at 200/500/1,000, and proves scientific rejection
   of synthetic inputs. The older positive freeze/registration test still builds
   payloads directly. Do not relabel those payloads or the generated synthetic
   artifacts as scientific to make a test pass. Test structural acceptance and
   scientific eligibility separately, keeping the public scientific entrypoints
   strict. Cover both development tasks, all support budgets, final-test tasks,
   three source seeds, relocated output/confirmation paths, and the canonical
   ledger. Instrument actual HDF5 action reads and use sentinel query labels.
2. Extend interruption/recovery coverage from the completed seed-42/five-shot
   matrix to the remaining seeds/budgets. Sampling parity and nested supports
   already cover all nine seed/budget combinations. Existing full-length tests
   cover both methods, every scheduled snapshot, real subprocess SIGINT/SIGTERM,
   fresh directories, optimizer/RNG/prediction equality, exact snapshot hashes,
   repeat evaluations, and conflicting snapshots. Compare source recovery bytes
   and original study-root binding as well as learned snapshots.
3. Exercise three-seed final-test paired bootstrap with hand-calculated expected
   deltas and missing/mismatched pair cases. The report driver now reaches both
   roles; a development constant-delta arithmetic test exists.
4. Review completion publication failure cases: process death between bundle and
   receipt, concurrent writers, tampered archive bytes versus decoded payload,
   and relocation after a query-access ledger has opened. A receipt must not make
   an incomplete run eligible. Keep recovery state exact across optimizer
   boundaries and preserve the first published artifact bytes.
5. Review the complete CUDA collector against its required-component list before
   hardware execution. Local tests now exercise actual 5+20 update timing hooks,
   initialization, source validation, serialization/publication, and all eight
   evaluator timing outputs. Format-3 bundles require complete finite timings,
   memory, storage, recovery, full query input IDs/hashes, and a matching receipt.
   CPU/MPS and schema unit fixtures must never produce completed CUDA evidence.
   Forecasts count actual checkpoint writes (801 all-budget / 465 five-shot before
   selection), 60 source-validation passes, 42 development retention passes,
   54/18 confirmation retention passes, initialization, source/adaptation
   recovery, and scaled retained storage. The pure inventory/forecast functions
   support frozen per-method steps; the preselection CLI remains conservative at
   1,000 steps. Before reducing a CLI forecast after selection, validate and bind
   it to the actual freeze/registration rather than accepting bare step numbers.

## Pinned target runtime

A resolver-only lock exists at
`artifacts/fewshot_libero_goal_v3/readiness_acceptance_20261004/requirements-linux-cu118.lock`.
The log records 137 resolved packages for Linux x86-64 / Python 3.10, including
PyTorch 2.1.2+cu118 and torchvision 0.16.2+cu118. This is not an import or GPU test.
The CUDA index matches the [official PyTorch 2.1.2 instructions](https://docs.pytorch.org/get-started/previous-versions/#v212).

On an already authorized NVIDIA machine, create a clean environment and install
that lock with hashes and the CUDA 11.8 backend. Do not reuse the Mac environment
or silently substitute newer torch/CUDA versions. Record Python, `pip freeze`,
`torch.__version__`, `torch.version.cuda`, device model, driver, and the lock hash.
Run real MAE forward, frozen-weight verification through LoRA wrapper names,
zero-adapter equivalence, nonzero adapter gradients/updates, and saved-live versus
reloaded predictions on the same CUDA device. Run `smoke-real --device cuda`
with the frozen manifest and pinned assets in a fresh output directory.

## Calibration, price, and launch decision

Run `collect-calibration --device cuda` in a fresh directory only after the
runtime gate passes. Inspect actual measurement components and full input-ID
coverage; do not accept a receipt solely because a file exists. The successful
CUDA collector path remains unexecuted locally. Keep the resulting cache,
smoke, measurements, and receipt bound to the same effective code and inputs.

Use a current verified provider price and the user's actual prior spending.
Do not assume zero spend. Require both forecast scopes to be reported and
independently check 25% compute-workload margin, protected $25 reserve, and $250
total ceiling. The five-shot scope still contains all six development baseline
bundles and all 36 learned-candidate evaluations. Retained-storage estimates
include static assets/cache and training/recovery copies.

Review cloud resource ownership, a budget-derived absolute termination deadline,
collection hashes, interruption recovery, and cleanup refusal for unrelated
resources. No resource creation, paid run, first five-shot development experiment,
or confirmation execution is authorized by this readiness handoff.

## Evidence refresh

After any source/input change, use fresh output directories and run:

```bash
python3 -m compileall -q raid_experiments src tests
PYTHONPATH=. python3 -m pytest -q
python3 -m raid_experiments verify-assets \
  --manifest data/assets/libero/protocol_v3.json \
  --feature-dir data/assets/libero/features-v4 \
  --mae-ckpt data/assets/mae/mae_pretrain_vit_base.pth \
  --libero-repo data/assets/libero/upstream \
  --asset-notes configs/mae_asset_notes.json
```

Then repeat real CPU smoke (and CUDA smoke when available), bind reports to
`_code_snapshot_digest()`, `_config_digest(_scientific_config())`, cache/asset
hashes, and the acceptance-test hashes. Update all three readiness documents
from recorded results. Mark missing gates unexecuted, not passed by inference.
