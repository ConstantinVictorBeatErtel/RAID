# Few-shot visual inverse-dynamics adaptation

## Permanent model constraint

RAID must never use, download, load, integrate, or recommend GR-1, including its checkpoints, image projection weights, source dependencies, or derived representations. Historical records may remain for audit only. The active study uses standalone MAE ViT-B/16 recorded-image features and does not use a world model or predicted futures.

This is the active research protocol for RAID. It replaces the earlier planned four-condition pilot as the next study. Historical results remain in [RESULTS.md](RESULTS.md); current implementation readiness is tracked in [STATUS.md](STATUS.md).

## Research question

Given the same few demonstrations of a previously unseen LIBERO task, does adapting a visual inverse-dynamics model through LoRA generalize better than retrieving demonstrated transitions, and at what computational cost?

The primary claim concerns action prediction from recorded current/next RGB frames. This measures transfer of inverse dynamics to held-out tasks after examples are supplied. It does not demonstrate that a learned video predictor or policy completes tasks. Predicted futures and closed-loop control are later studies with their own integration and evaluation gates.

LoRA is central to the comparison; it is not assumed to be novel by itself. TAIL studies parameter-efficient robot imitation, HAND fine-tunes on retrieved trajectories, OpenVLA examines efficient policy adaptation, and VLA-Pro retrieves LoRA adapters. FADA adapts an inverse-dynamics model with a frozen planner. Read and compare these methods and WIZARD before making a novelty claim. The candidate contribution is evidence about when visual encoder adaptation helps beyond head-only fine-tuning or retrieving the same examples, and what that benefit costs.

### Research stages

1. **Establish the gap:** read the closest methods and write a claim/evidence/overlap table. Record unresolved implementation and data-access differences.
2. **Freeze the protocol:** verify benchmark data and task names, create the task and demonstration manifest, agree on methods, budgets, metrics, and tuning rules. Hash and preserve the manifest before model selection.
3. **Verify implementation:** implement role-aware data access and source/adaptation stages; pass regression and synthetic end-to-end checks.
4. **Validate inputs and cost:** load the actual pinned checkpoints, exercise real LIBERO images, verify gradients and preprocessing, then measure runtime and peak memory. Forecast the complete next comparison with 25% workload margin.
5. **Develop:** start with five support demonstrations, one development task, and seed 42. Then run all development tasks and 2/5/10 support budgets; select settings only on development queries.
6. **Confirm:** lock code, configuration, source checkpoint, and selected update counts. Evaluate paired seeds 42, 43, and 44 on final task queries once.
7. **Interpret:** decide from the measured evidence whether LoRA adds value over retrieval and head-only fine-tuning. A clear null result is a valid outcome.
8. **Extend:** validate predicted futures, then closed-loop control, only after their predictor/simulator contracts pass. Consider DROID as a later real-world recorded-data study.
9. **Write and reproduce:** reconcile every paper claim with saved predictions and provenance; recover editable manuscript source, review citations, and reproduce reports from artifacts.

## Initial protocol

### Benchmark and splits

Use the ten official LIBERO-Goal tasks from the pinned LIBERO revision. Hash-sort task names by SHA-256 of `20260925:<task_name>`: six source tasks, the next two for development, and the last two for final testing. Persist task names, not task indices. Under the pinned task map, development holds out cream-cheese-in-bowl and top-drawer-and-bowl; final test holds out bowl-on-cabinet and wine-bottle-on-cabinet. This initial two-task test is narrow; do not generalize beyond the measured tasks.

Verify each task has 50 usable demonstrations before freezing the manifest. Hash-sort demo IDs within each task with a recorded seed. On source tasks allocate 40 source-training, five source-validation, and five retention-evaluation demonstrations. On each held-out task, reserve 20 query demonstrations and use the remaining 30 as a shared support pool. Select nested support sets of 2, 5, and 10 demonstrations. Use the exact same support/query IDs for all methods. Preserve real HDF5 demonstration keys and transition IDs. Missing or unusable data blocks protocol freezing; do not quietly replace tasks or reduce holdouts.

Audit MAE pretraining exposure. “Held out from this study’s training” is not evidence that a task was unseen during upstream pretraining. Final query actions are sealed from all training, normalization, memory construction, and checkpoint selection.

### Model and conditions

Start with horizon-one **recorded RGB** frames, predicting the recorded seven-dimensional action. Use standalone MAE ViT-B/16 directly, with its normalized 768-dimensional CLS feature. Apply deterministic 224×224 bicubic preprocessing and ImageNet normalization to both frames. Verify checkpoint origin and pretraining exposure. Do not use GR-1 code, weights, projections, or derived features; do not construct a future predictor to evaluate recorded transitions.

Train a direct inverse-dynamics head on source tasks from frozen visual features. For each target task, budget, and seed, reset from the same source checkpoint. Compare:

| Condition | Target-task access |
|---|---|
| `frozen_source` | No target updates and no target retrieval. |
| `target_knn` | Similarity-weighted retrieval from the target support transitions; no gradient updates. |
| `head_ft` | Fine-tune only the source-trained action head using target support. |
| `lora_head_ft` | Fine-tune the same head and rank-8 LoRA in MAE blocks 10–11, `qkv` and `proj`; freeze all other weights. |

Call the last condition **LoRA plus head adaptation**, not LoRA-only. Retrieval uses cosine transition keys, `k=3`, temperature `0.1`, with valid-neighbor weight renormalization. Also report support mean action, transition 1-NN, current-frame weighted k-NN, and source-plus-support transition retrieval. The retrieval bank can contain source experience; identify bank membership in artifacts.

The first study excludes learned RAID blending, LoRA-plus-retrieval, adapter routing, online updates, Cosmos, DROID, and GRPO. Add them only if the initial evidence motivates a distinct question.

### Training, selection, and metrics

Fit one action normalizer on source-training actions; persist and reuse it for every method and target budget. Set the scale of constant dimensions to one. Source training uses AdamW, 2,000 updates, batch 64, microbatch 8, and action-head learning rate `3e-4`; select its checkpoint on source validation only.

Target adaptation runs up to 1,000 updates with head learning rate `3e-4`, LoRA learning rate `1e-4`, rank 8, alpha 16, dropout 0, weight decay `1e-4`, and gradient clipping 1. Head-only and LoRA conditions share support IDs, sample schedule, and effective batch size. Compare 200-, 500-, and 1,000-update snapshots on development queries. Select one update count per method by average query error across development tasks and budgets; break ties toward fewer updates. Do not inspect final query results until the settings are locked.

Development uses seed 42. Confirmation uses paired seeds 42, 43, and 44 and records both source initialization and support-selection seeds. Reset from the same source state for every target/budget/repetition; never carry updates across targets or initialize a larger budget from a smaller one.

Primary metric is source-normalized action MSE, averaged by transition within demonstration, then by demonstration within task, then by task. Also report physical per-action errors, gripper accuracy, per-task outcomes, paired method differences, support count, adaptation time, inference latency, peak memory, and trainable parameter count. Evaluate source retention with the target adapter active. Quantify uncertainty over whole demonstrations and paired seeds, never treating correlated frames as independent. Limit claims to evaluated tasks.

If measured cost cannot fit the full confirmation matrix within the **$250 planning ceiling**, including $25 protected reserve and 25% workload margin, retain a complete paired five-demonstration comparison across all methods and seeds before expanding to 2/10-shot settings. Record the reduced design before opening final queries. The ceiling is planning guidance, not authorization to rent compute; paid launch remains disabled until all gates pass.

## Evidence and publication standards

A few-shot adaptation claim requires target examples used only as support, separate unseen query demonstrations and new initial states, fair target-data access, and a retrieval baseline. A robot-control claim additionally requires predicted-future validation, simulator expert replay, paired complete rollouts, the official success predicate, and failures. A robustness claim requires perturbations or genuinely distinct held-out domains. A low compute claim requires measured wall time, GPU memory, inference latency, trainable parameters, storage, and reconciled charges.

The historical LIBERO results in [RESULTS.md](RESULTS.md) are offline validation numbers with known leakage/protocol limitations. They motivate this question but are not results for this protocol. Preserve them as historical evidence and never mix their measurements with the new study.

## Ownership

| Document | Owns |
|---|---|
| This file | Research question, fixed protocol, decision gates, publication claims. |
| [STATUS.md](STATUS.md) | Current implementation state and ordered blockers. |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | Literature comparison and reading record. |
| [docs/ADAPTATION_IMPLEMENTATION.md](docs/ADAPTATION_IMPLEMENTATION.md) | Code work packages and verification acceptance checklist. |
| [RESULTS.md](RESULTS.md) | Measurements and provenance, separated by protocol. |
| [docs/REPRODUCING.md](docs/REPRODUCING.md) | Commands and environment requirements, including limitations. |
| [paper/](paper/README.md) | Manuscript and submission artifacts. |
