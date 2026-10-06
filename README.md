# RAID: few-shot visual inverse-dynamics adaptation

RAID is now investigating whether adapting a visual inverse-dynamics model through LoRA helps it generalize to a new robot task from a few demonstrations, compared with retrieving examples or fine-tuning only its action head. The first study uses recorded transitions from the ten official LIBERO-Goal tasks. It tests offline action prediction; it does not by itself show that a policy completes robot tasks.

The scientific protocol lives in [RESEARCH.md](RESEARCH.md), evidence and blockers in [STATUS.md](STATUS.md), the literature record in [docs/RELATED_WORK.md](docs/RELATED_WORK.md), and implementation acceptance checks in [docs/ADAPTATION_IMPLEMENTATION.md](docs/ADAPTATION_IMPLEMENTATION.md). The initial configuration is [configs/fewshot_libero_goal.yaml](configs/fewshot_libero_goal.yaml); its Runpod launch remains disabled.

## Current readiness

The active protocol is v3. Standalone MAE ViT-B/16 supplies normalized 768-dimensional CLS features from deterministically preprocessed recorded RGB frames. Source-head training, head and MAE LoRA adaptation, retrieval controls, evaluation, and evidence-gated reporting live in `raid_experiments/fewshot.py` and `raid_experiments/fewshot_train.py`. Synthetic tests pass; the official MAE checkpoint and pinned LIBERO checkout are present, and original LIBERO-Goal data has been acquired from the pinned downloader's documented fallback. A fresh format-4 cache and CPU smoke remain in progress; CUDA measurements and cost calibration remain blocked. The earlier pilot CLI is disabled.

No LIBERO-Goal results establish the new hypothesis. No paid run is approved. The $250 figure is a planning ceiling, including a $25 protected reserve; measure all costs before scheduling a full comparison.

## Historical evidence

The May report, historical GR-1/LIBERO validation numbers, GRPO records, low-dimensional experiments, V-JEPA exploration, and `v2` matrix are preserved for audit. They use older protocols and cannot be combined with results from the new study. See [RESULTS.md](RESULTS.md) and [docs/HISTORY.md](docs/HISTORY.md). The editable manuscript source is not in the repository; [paper/README.md](paper/README.md) records this limitation.

Use only the explicit v3 commands in [docs/REPRODUCING.md](docs/REPRODUCING.md) for the active study. Paid execution remains disabled. Do not infer scientific readiness from synthetic tests or CPU/MPS smoke.
