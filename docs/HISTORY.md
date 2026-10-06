# RAID experiment history

**Current direction (25 September 2026):** few-shot visual inverse-dynamics adaptation versus transition retrieval on held-out LIBERO-Goal tasks. Its question and preregistered protocol are in [RESEARCH.md](../RESEARCH.md). The historical tracks below are retained to explain earlier work; their scores must not be merged with the new protocol.

## RoboMimic Lift

The original low-dimensional path uses proprioceptive state transitions and the `src/data.py`, `src/models.py`, `src/train.py`, and `src/evaluate.py` pipeline. Its committed metrics are in [`configs/results.json`](../configs/results.json), and its search notes are in [`configs/autoresearch_log.md`](../configs/autoresearch_log.md). It does not use GR-1 visual features.

## V-JEPA/DINO/SigLIP exploration

This historical exploration tested frozen visual features, retrieval variants, and inverse-dynamics heads. Its research program remains at [history/vjepa-program.md](history/vjepa-program.md). The autoresearch entry point has an unresolved helper import documented in [research status](RESEARCH_STATUS.md); these notes are not verified reproduction instructions.

## GR-1 + LIBERO-Spatial pilot

The prior paper-facing track froze GR-1/MAE features and compared direct and retrieval-augmented decoders on LIBERO-Spatial. The repaired `raid_experiments` package was intended for this historical pilot. It is not the active source-task/target-task adaptation pipeline. Its recorded metrics and caveats are in [RESULTS.md](../RESULTS.md) and [RESEARCH_STATUS.md](RESEARCH_STATUS.md).

## `v2` multi-dataset expansion

The `v2/` subtree adds Drive-backed artifact management, RoboMimic/LIBERO adapters, DINOv2/Theia features, transformer and diffusion heads, and a matrix runner. It remains an independent historical code path.

## Draft PR artifacts

The draft branch [`b493d87`](https://github.com/ConstantinVictorBeatErtel/RAID/tree/b493d87fe7c8b2c78b738a9ce1167ef52dae396f) preserves transition visualizations, GRPO records, and a GR-1 compatibility patch. It is not current evidence for the adaptation study.
