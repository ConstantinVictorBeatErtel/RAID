# RAID experiment history

The repository contains several related but non-identical experiments. The current paper-facing result is the GR-1/LIBERO track; the others explain earlier decisions and remain available for audit.

## RoboMimic Lift

The original low-dimensional path uses proprioceptive state transitions and the `src/data.py`, `src/models.py`, `src/train.py`, and `src/evaluate.py` pipeline. Its committed metrics are in [`configs/results.json`](../configs/results.json), and its architecture-search notes are in [`configs/autoresearch_log.md`](../configs/autoresearch_log.md). These results do not use GR-1 visual features.

## V-JEPA/DINO/SigLIP exploration

The next exploration tested frozen visual features, retrieval variants, and alternative inverse-dynamics heads. Its historical instructions are preserved unchanged in [history/vjepa-program.md](history/vjepa-program.md). The corresponding helpers include `src/vjepa_encoder.py`, `src/cache_vjepa_features.py`, `src/models_libero.py`, `src/memory_libero.py`, and `src/autoresearch_libero.py`. The current autoresearch entrypoint has an unresolved helper import documented in [research status](RESEARCH_STATUS.md), so its instructions are historical rather than a verified reproduction recipe.

## GR-1 + LIBERO-Spatial

The paper-facing track freezes GR-1/MAE visual features and evaluates direct and retrieval-augmented decoders on LIBERO-Spatial. The active path is documented in the root [README](../README.md), while result provenance and implementation caveats are in [RESULTS.md](../RESULTS.md) and [RESEARCH_STATUS.md](RESEARCH_STATUS.md).

## `v2` multi-dataset expansion

The `v2/` subtree adds Drive-backed artifact management, RoboMimic/LIBERO adapters, DINOv2/Theia features, transformer and diffusion heads, and a phase matrix. It is additive historical work and has its own requirements and artifact root. Its README now states that its claims are historical and require fresh verification before publication use.

## Draft PR artifacts

The open draft PR branch [`b493d87`](https://github.com/ConstantinVictorBeatErtel/RAID/tree/b493d87fe7c8b2c78b738a9ce1167ef52dae396f) contains transition visualizations, GRPO logs/checkpoints, and a GR-1 compatibility patch. It is documented for provenance but is not merged or integrated by this cleanup.
