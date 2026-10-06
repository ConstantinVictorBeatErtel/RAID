# Experiment configurations and artifacts

## Active research configuration

[`fewshot_libero_goal.yaml`](fewshot_libero_goal.yaml) records the active few-shot adaptation design: LIBERO-Goal, six source tasks, two development tasks, two final-test tasks, support budgets 2/5/10, and frozen-source/retrieval/head-fine-tuning/LoRA-plus-head conditions. Paid launch is explicitly disabled. The protocol and evidence gates are in [RESEARCH.md](../RESEARCH.md); code checks are in [docs/ADAPTATION_IMPLEMENTATION.md](../docs/ADAPTATION_IMPLEMENTATION.md).

The protocol, source fitting, support-only adaptation, evaluation, and reporting CLI stages are implemented and have synthetic integration coverage. The real-data/checkpoint smoke, complete leakage suite, and cost gates remain open; the configuration is not proof that those gates passed.

## Historical configurations

- [`pilot_100.yaml`](pilot_100.yaml), [`expansion_study.yaml`](expansion_study.yaml), and [`cosmos3_edge_feasibility.yaml`](cosmos3_edge_feasibility.yaml) describe earlier or deferred experiment plans. They do not select the new LIBERO-Goal protocol.
- [`v6.yaml`](v6.yaml), `autoresearch_*_models.py`, and [`autoresearch_log.md`](autoresearch_log.md) record the original low-dimensional decoder experiments.
- `results_libero.json`, `results.json`, `results_vjepa.json`, the GRPO logs, and loss curves contain historical measurements. Their protocols differ; see [RESULTS.md](../RESULTS.md).
- Generated datasets, caches, normalizers, checkpoints, and logs are not all tracked. Rebuild or restore them only using the recipe that matches their protocol and source hashes.
