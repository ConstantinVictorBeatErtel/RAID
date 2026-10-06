# Reproducing the active RAID study

The active study uses recorded RGB transitions and standalone MAE ViT-B/16 features under `fewshot_idm_adaptation_v3`. The CLS feature is 768-dimensional and normalized. Frames use deterministic 224×224 bicubic resizing and ImageNet normalization. This protocol does not predict future frames. Paid execution is disabled.

## Assets and fresh cache

Keep downloaded assets outside upstream checkouts. The pinned LIBERO downloader lists the original LIBERO-Goal Box URL and a Hugging Face fallback in [`download_utils.py`](https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/utils/download_utils.py). The original Box URL currently returns 404; the fallback is the same upstream-published `libero_goal` set. Keep the LIBERO checkout at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, the official MAE ViT-B checkpoint, and an asset note recording origins, hashes, and documented pretraining exposure. Do not overwrite existing assets.

```bash
python3 -m raid_experiments prepare-protocol \
  --dataset-dir data/assets/libero/libero_goal \
  --manifest artifacts/fewshot_libero_goal_v3/protocol.json --expected-demos 50
python3 -m raid_experiments cache-features \
  --manifest artifacts/fewshot_libero_goal_v3/protocol.json \
  --dataset-dir data/assets/libero/libero_goal \
  --feature-dir data/assets/libero/features-v4 \
  --mae-ckpt data/assets/mae/mae_pretrain_vit_base.pth \
  --libero-repo data/assets/libero/upstream --device cpu
python3 -m raid_experiments verify-assets \
  --manifest artifacts/fewshot_libero_goal_v3/protocol.json \
  --feature-dir data/assets/libero/features-v4 \
  --mae-ckpt data/assets/mae/mae_pretrain_vit_base.pth \
  --libero-repo data/assets/libero/upstream \
  --asset-notes configs/mae_asset_notes.json
```

Cache format 4 binds source files, MAE weights and implementation, preprocessing, protocol, and pinned LIBERO code. Query shards contain no action labels. Old caches and checkpoints remain untouched and cannot be active inputs. The frozen manifest is `data/assets/libero/protocol_v3.json`; the verified cache is `data/assets/libero/features-v4`. The fresh complete cache audit passes for all 500 shards, including 80 label-free query shards. The current real CPU reference smoke is [`readiness_cpu_20261004_acceptance/smoke_real.json`](../artifacts/fewshot_libero_goal_v3/readiness_cpu_20261004_acceptance/smoke_real.json); its short runs are diagnostic and cannot satisfy scientific gates. It records code digest `70d3b73c7d029d57df299745e4e50d69b7a6930b9b5049f35b242a623b80e6a7`, config digest `acbc84b807392675a7d012973b09e6fb86070b10cd467f1a61dd757127804ba5`, and the local runtime (Python 3.13.7, PyTorch 2.11.0, timm 1.0.27). The code digest must be refreshed after any code change.

## Local acceptance and calibration

Run the local checks with:

```bash
python3 -m compileall -q raid_experiments src tests
PYTHONPATH=. python3 -m pytest -q
```

The complete suite passed **51 tests and 6 recovery subtests** in 211.29 seconds, with one existing Transformer warning. Logs and JUnit output are in `artifacts/fewshot_libero_goal_v3/readiness_acceptance_20261004/pytest-final.*`. The coherent unequal-length HDF5 fixture calls production caching, runs 2,000 source updates and both 1,000-update adaptations, and evaluates all eight methods at 200/500/1,000. A tiny explicitly synthetic encoder keeps the regression practical; the actual production decoder, optimizer, dropout, sampling, checkpoint and evaluation routines run. Synthetic outputs remain scientifically ineligible.

Fresh-directory recovery compares weights, optimizer/RNG state, predictions and exact snapshot hashes at every scheduled step for both methods (seed 42/five-shot), with real subprocess SIGINT/SIGTERM, repeat evaluation and conflicting-evidence rejection. Completed source recovery preserves snapshot bytes and the original canonical ledger binding. Sampling parity, nested supports, unequal lengths and instrumented HDF5 query-label isolation cover all three seeds and budgets. The complete recovery cross-product and a production-generated positive freeze/confirmation/report flow remain unexecuted; see [READINESS_HANDOFF.md](READINESS_HANDOFF.md).

Production calibration hooks now measure initialization separately and run five warmup plus twenty measured updates in one session. Validation, actual checkpoint serialization/publication and per-method evaluation timing are exposed. Format-3 measurement bundles require complete finite timing/resource components, all evaluated RGB IDs/hashes, and a matching receipt. CPU/MPS cannot satisfy the CUDA gate. Forecast arithmetic tests cover both scopes, source validation, actual writes, frozen update counts, scaled storage, prior spend, 25% margin, the protected $25 and $250 ceiling.

A hash lock for Linux x86-64 / Python 3.10 / CUDA 11.8 resolved 137 packages, including torch 2.1.2+cu118 and torchvision 0.16.2+cu118. It is retained at `artifacts/fewshot_libero_goal_v3/readiness_acceptance_20261004/requirements-linux-cu118.lock`. This establishes resolution only: target imports, GPU MAE checks and CUDA calibration remain unexecuted. The CUDA index follows the [official PyTorch 2.1.2 installation instructions](https://docs.pytorch.org/get-started/previous-versions/#v212). The current real CPU smoke used Python 3.13.7, torch 2.11.0 and timm 1.0.27.

```bash
python3 -m raid_experiments smoke-real --manifest data/assets/libero/protocol_v3.json \
  --feature-dir data/assets/libero/features-v4 --mae-ckpt data/assets/mae/mae_pretrain_vit_base.pth \
  --libero-repo data/assets/libero/upstream --asset-notes configs/mae_asset_notes.json --device cpu
python3 -m raid_experiments collect-calibration --manifest data/assets/libero/protocol_v3.json \
  --feature-dir data/assets/libero/features-v4 --mae-ckpt data/assets/mae/mae_pretrain_vit_base.pth \
  --libero-repo data/assets/libero/upstream \
  --output-dir artifacts/fewshot_libero_goal_v3/calibration --device cuda
python3 -m raid_experiments calibrate-study --feature-dir data/assets/libero/features-v4 \
  --pricing-file configs/verified_pricing.json --spent-usd ACTUAL_PRIOR_SPENDING_USD
```

Use fresh output directories for smoke and calibration after any code or input change. Do not write estimated or unverified prices into `configs/verified_pricing.json`; the file is currently absent. A completed forecast requires the CUDA completion receipt, verified current prices, and actual prior spending. A blocked forecast does not authorize paid execution.

Forecast and confirmation registration share one workload inventory: 24 canonical evaluation conditions for all-budget confirmation or 12 for five-shot, six development baseline bundles, 36 learned-snapshot evaluations, 42 development retention passes, and 54/18 confirmation retention passes. Arithmetic tests include prior spending, a 25% workload margin, and the protected $25 reserve under the $250 ceiling. Before selection, confirmation is budgeted conservatively at 1,000 updates. The real price-backed forecast remains blocked until successful CUDA calibration and verified pricing; a blocked forecast does not authorize a paid run.

## Study execution gates

The first five-shot development run remains gated on the pinned CUDA environment, complete local acceptance, CUDA smoke and calibration, a safe price-backed forecast, and a recorded cloud lifecycle safeguard review. Paid execution stays disabled until these gates pass. Development selection still requires its complete preregistered scope: two development tasks at 2-, 5-, and 10-shot with seed 42; evaluate the six-control baseline bundle once per condition and each learned snapshot independently. Freeze selected steps before registering confirmation, then evaluate only the registered scope. See [STATUS.md](../STATUS.md), [RESEARCH.md](../RESEARCH.md), and [ADAPTATION_IMPLEMENTATION.md](ADAPTATION_IMPLEMENTATION.md) for current evidence and remaining gates.

Historical pilot records remain available in [RESULTS.md](../RESULTS.md) for audit and do not count as evidence for this protocol.

The fresh eight-method real CPU evaluation used identical 2,132 recorded query transitions at five-shot/seed 42 with two-update calibration snapshots. Evidence and SHA-256 bindings are in [`readiness.json`](../artifacts/fewshot_libero_goal_v3/readiness_acceptance_20261004/readiness.json). It records `launch_ready=false`; the local CPU results cannot authorize CUDA or scientific execution.
