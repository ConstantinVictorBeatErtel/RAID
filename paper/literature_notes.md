# Literature reading notes

Reviewed 4 October 2026 for the active recorded-RGB, few-shot visual inverse-dynamics study. These are research notes, not manuscript prose. Citation keys match `references.bib`; links identify the source consulted. “Method review” means the relevant full-text method/evaluation passages were read, not that the implementation was reproduced. Claims below describe the authors' evidence; the implications for RAID are our interpretation.

## Closest comparisons

### TAIL — `tail`

[ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/46e2b525da536030ec1589dddea01feb-Paper-Conference.pdf). **Read:** §§4–5, especially §5.2, pp. 6–9 and Table 3.

Pretrains on 40 LIBERO-90 Kitchen tasks; the main curriculum adapts one adapter per eight-task suite. Each task supplies 50 demonstrations, split 40/10 for training/validation. Adaptation updates adapters, fusion module, and policy head. The frozen-feature baseline also updates fusion and head. Earlier tasks use their saved adapters, unlike evaluating retention with the new adapter active. Table 3 separates a 1.17% trainable-parameter fraction from a 23% GPU-memory reduction at batch 14.

**Use:** Direct precedent for LoRA versus frozen representations in robot imitation. Our head-only comparison is narrower. Do not present parameter fraction as an equivalent memory or runtime saving.

### FADA — `fada`

[arXiv v1](https://arxiv.org/html/2606.28476v1). **Read:** §§3–5, Algorithm 1, §5.5/Table 4, Appendix B.

Studies fixed tasks under dynamics changes. A planner forecasts proprioception; an IDM consumes history and future observations. Target learning freezes both original modules and optimizes **LoRA inside the IDM** with observed transitions. Main adaptation uses approximately 6,000 control steps; evaluation covers humanoid hardware and simulator transfer. Table 4 compares LoRA with full IDM fine-tuning.

**Use:** Strong prior art for few-shot LoRA IDM adaptation. Distinguish visual task transfer, encoder adaptation, and matched transition retrieval. Exact adapter ranks/insertion sites remain unverified; no implementation-equivalence claim is warranted.

### HAND — `hand`

[arXiv v4](https://arxiv.org/html/2505.20455v4). **Read:** §§III-B–III-D, IV-A and IV-B; checked against v1.

Matches human hand paths to robot-play sub-trajectories after visual filtering, then fits LoRA parameters with similarity-weighted behavior cloning. The latest evaluation uses a WidowX kitchen setup and a roughly 50,000-transition play dataset; longer tasks receive one hand demonstration per subtask. Its short adaptation-time demonstration excludes the cost of creating the play dataset and pretrained policy.

**Use:** Retrieval followed by LoRA is established. The query is human task evidence and retrieval selects training trajectories, unlike predicting a query action from the same labeled support set. The old CALVIN description belongs to v1; do not mix its evaluation with v4.

### WIZARD — `wizard`

[arXiv v1](https://arxiv.org/html/2606.07217v1). **Read:** §§3–4, Appendix A/Table 5 and Table 6.

Meta-learns a mapping from language/video task evidence to expert LoRA parameters. Target inference uses no action labels or gradient updates. Beyond held-out suites, Appendix A holds out two tasks per LIBERO suite and trains on 32 remaining tasks. The nearest-neighbor comparator retrieves adapters. Table 6 measures adapter-generation cost, not the entire expert-building and meta-training pipeline.

**Use:** Close task-transfer precedent. Our supervised support adaptation differs in target information and training cost. The paper's warm-start example reports 70 versus 90 steps; the project page says 7 versus 9. Neither number is needed in the draft.

### OpenVLA — `openvla`

[arXiv v3](https://arxiv.org/abs/2406.09246v3). **Read:** §5.3, Table 1 and footnote 4, PDF p. 10.

Compares full, last-layer, frozen-vision, sandwich, and LoRA fine-tuning using rollout success and memory. The “last layer” condition includes the token embedding matrix, so it is not our action-head-only condition. Crucially, the efficient-fine-tuning experiments use a smaller pretraining mixture and SigLIP-only variant, rather than exactly the headline fused-encoder model.

**Use:** Practical precedent for efficient policy adaptation and visual adaptation. Attribute Table 1 to its actual model variant; do not transfer its success or compute figures to MAE.

### OpenVLA-OFT — `oft`

[arXiv v2](https://arxiv.org/abs/2502.19645v2). **Read:** method §IV-B and evaluation §V-A, PDF pp. 4–5.

Studies parallel decoding, action chunking, continuous outputs, and L1 regression. LIBERO training is suite-specific, with 500 available demonstrations per ten-task suite before filtering. Its reported setup uses substantially longer optimization and closed-loop evaluation. Checkpoints are evaluated periodically and the best reported.

**Use:** Shows why action interface, objective, training budget, and selection protocol matter for efficiency comparisons. It is not a held-out-task, 2/5/10-demonstration transition-prediction baseline.

### REGENT — `regent`

[ICLR 2025 paper](https://kaustubhsridhar.github.io/regent-research/static/docs/REGENT.pdf). **Read:** §§3–6, especially §4, PDF pp. 4–7.

Retrieve-and-play copies the nearest action. REGENT supplies 19 retrieved tuples to a transformer and interpolates its output with the nearest action according to distance. The MetaWorld/MuJoCo robotics observations are proprioceptive vectors; image handling appears in game environments. Pretraining excludes the query demonstration from its retrieved context.

**Use:** Strong rationale for a 1-NN control and careful demonstration separation. Do not describe REGENT's robotics evaluation as RGB LIBERO inverse dynamics. Its main no-fine-tuning deployment does not mean the paper never investigates fine-tuning.

### VLA-Pro — `vlapro`

[arXiv v1](https://arxiv.org/html/2605.29562v1). **Read:** §3/algorithm and §4.

Builds task-specific LoRA memories from seen tasks, retrieves them using structured procedural context, and fuses parameters for action generation. Evaluation uses RoboTwin, RLBench and a UR7e setup. The reported procedural extractor and text embedder are external models; their costs belong in any end-to-end accounting.

**Use:** Prior art for retrieval of adapter memories. It retrieves existing task parameters rather than labeled transition actions or fitting a fresh target adapter. No LIBERO evaluation was found in the inspected experimental section.

## Retrieval and representation comparisons

### VINN — `vinn`

[RSS 2022 paper](https://www.roboticsproceedings.org/rss18/p010.pdf). **Read:** method and experimental setup, PDF pp. 3–5.

Learns visual representations and then uses locally weighted regression over demonstrated actions. Includes offline action prediction, a behavior-cloning-on-representations baseline, and real door opening. The encoder may be self-supervised on demonstration images; nonparametric behavior learning does not mean the entire method has no training.

**Use:** Foundational visual retrieval baseline. Our transition key adds a recorded next observation and uses MAE; label it a retrieval baseline inspired by this literature, not a reproduction of VINN.

### FlowRetrieval — `flow`

[PMLR paper](https://proceedings.mlr.press/v270/lin25a.html). **Read:** §§2, 4–5, Algorithm 1, PDF pp. 3–6.

Computes optical flow between current and later frames, embeds it with a VAE, retrieves motion-similar prior data, and co-trains a policy with a flow auxiliary loss. Examples include a LIBERO-Can task with ten target demonstrations and 900 prior trajectories.

**Use:** Motion-based retrieval already uses temporal information. Its retrieval selects a training set; it is not query-time action lookup. Cite proceedings year 2025 consistently, while noting that this is the eighth CoRL conference (2024).

### Behavior Retrieval — `behavior`

[paper](https://arxiv.org/abs/2304.08742). **Read:** §III-A–C and Figure 2.

Learns a state-action similarity representation on prior data, uses a small target dataset to select relevant examples, and trains on target plus retrieved data. “Unlabeled” refers to the lack of task labels, not the absence of recorded robot actions.

**Use:** Foundational precedent for target-guided data selection. Distinguish querying a dataset to train a policy from predicting an action by retrieval.

### STRAP — `strap`

[ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/a06e129e01e0d2ef853e9ff67b911360-Paper-Conference.pdf). **Read:** §§4.4–5, PDF pp. 7–8.

Uses foundation-model image features and subsequence dynamic time warping to match trajectory fragments. A policy is trained on the union of retrieved and target data, with language conditioning used to distinguish task-dependent actions. Evaluation includes LIBERO-10 and real DROID-Kitchen tasks.

**Use:** Adds the missing temporal-retrieval branch to the list. It assumes a prior corpus and a separate policy-training stage, beyond our target-support lookup.

### Importance Weighted Retrieval — `iwr`

[CoRL proceedings](https://proceedings.mlr.press/v305/xie25a.html); [full text reviewed](https://arxiv.org/html/2509.01657v1). **Read:** §§3–4.

Revisits nearest-target-distance retrieval as a limiting density estimate, then estimates target/prior density ratios with kernel density estimators. Selected examples augment policy training. The implementation thresholds importance weights; the paper explicitly distinguishes this from unbiased importance sampling.

**Use:** A limitation of simple retrieval rules, especially for an expanded source bank. It is not evidence that all nearest-neighbor action predictors are statistically invalid, nor an automatic fit to high-dimensional MAE keys.

### RT-Cache — `rtcache`

[arXiv v3](https://arxiv.org/html/2505.09040v3). **Read:** §§III–IV and limitations; compared with v1.

Caches visual embeddings and action sequences, applies hierarchical search, and replays multi-step snippets. The principal hardware endpoint is reaching an object's graspable region, with a separate anchoring study. The newer title is *Training-Free Retrieval for Real-Time Manipulation*. Completion-time aggregates discussed in the paper are conditioned on successful trials.

**Use:** Retrieval cost and replay precedent. Do not generalize the reaching endpoint to broad manipulation success, or compare success-conditioned completion times with per-query inference latency.

### RAEA — `raea`

[arXiv paper](https://arxiv.org/abs/2404.11699v1). **Read:** §§3.2–3.3 and §4.

Retrieves multimodal experience and integrates it into action prediction through a learned cross-attention policy generator trained with behavior cloning. This supports a broad prior-art statement about learned retrieval integration.

**Use:** Relevant to the historical retrieval-augmented method and possible later extensions. Cross-attention to retrieved experience is not a novelty claim for RAID. A reproduction of its attention equations or code is outside this review.

### CLAW — `claw`

[arXiv v1](https://arxiv.org/abs/2606.04130v1). **Read:** §§4.3–4.4 and §5.

Learns a continuous latent-action representation from transitions. During visual planning, nearest-neighbor lookup maps planned latent actions to real controls using labeled reference videos. It also inspects nearest transition neighbors and studies latent-action policy pretraining.

**Use:** A close precedent for retrieving executable actions from transition-derived representations. Our frozen MAE transition features are not CLAW's learned latent actions, and we do not evaluate its world-model planner.

### MAE — `mae`

[CVPR 2022 paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf). **Read:** architecture description and Figure 1.

Uses a masked-image reconstruction objective with an encoder processing visible patches and a lightweight decoder. The decoder is discarded for representation transfer.

**Use:** Cite for the encoder's learning method. Exact checkpoint, CLS extraction, preprocessing, and pretraining-data provenance must be documented from our assets; the paper alone does not verify those implementation choices.

### R3M — `r3m`

[arXiv v3](https://arxiv.org/abs/2203.12601v3). **Read:** §3, especially §3.1 and the objective overview.

Learns reusable visual embeddings from human video with temporal and language supervision, then studies frozen visual features for robot policy learning.

**Use:** Establishes frozen representation transfer as a substantive alternative. Its human-video pretraining and encoder differ from the ImageNet MAE used here; it is context, not a measured baseline.

### VC-1 / CortexBench — `vc1`

[arXiv v2](https://arxiv.org/abs/2303.18240v2). **Read:** primary abstract; full-text adaptation protocol not audited.

The broad benchmark study reports that no visual representation dominates every setting and that task/domain adaptation improves the evaluated model.

**Use:** Supports only the high-level motivation in paragraph two. No particular adaptation algorithm, numerical gain, or direct compatibility with our setup is asserted.

## Inverse dynamics and adaptation foundations

### Behavioral Cloning from Observation — `bco`

[IJCAI 2018 paper](https://www.ijcai.org/proceedings/2018/0687.pdf). **Read:** §§3–4, especially §4.1–4.2.

Learns an agent-specific inverse model from interactions, applies it to state-only expert trajectories to infer actions, and trains a policy using those inferred labels.

**Use:** Historical foundation for action inference from transitions. Unlike BCO, our target support already contains action labels; we evaluate inverse prediction rather than observation-only policy learning.

### UniPi — `unipi`

[paper](https://arxiv.org/abs/2302.00111). **Read:** §3.2 and action-execution description.

Translates video plans into actions through an inverse model trained separately from the planner on a smaller action-labeled dataset. The paper discusses both execution modes but states that its experiments use open-loop execution for efficiency.

**Use:** Precedent for modular video planning plus inverse dynamics. Avoid calling every video-IDM system a demonstrated closed-loop controller.

### Seer — `seer`

[arXiv v1](https://arxiv.org/html/2412.15109v1). **Read:** §§3.1–3.3 and benchmark description.

Jointly trains visual foresight and an action model conditioned on the predicted future representation, with history and task conditioning. Its inverse loss separates arm and gripper terms.

**Use:** Future-conditioned action prediction is established. Our observed next frame removes predictor error and supplies different inference information; offline MSE cannot be compared with Seer's rollout success.

### DreamGen — `dreamgen`

[arXiv v2](https://arxiv.org/html/2505.12705v2). **Read:** pipeline overview and action-extraction method/Figure 3.

Generates synthetic videos, recovers latent or physical pseudo-actions, and uses the resulting trajectories for policy training. Its IDM variant consumes two image frames and predicts intervening action chunks.

**Use:** Demonstrates image-pair action inference for synthetic-data labeling. The active study neither generates videos nor evaluates downstream synthetic-data training.

### Sample efficiency of IDMs — `morin`

[ICML 2026 paper](https://proceedings.mlr.press/v306/morin26a.html). **Read:** §§3–4 and §6 discussion, PDF pp. 3–5 and 9.

Relates video-model/IDM policies and inverse-model labeling in a limiting semi-supervised setting. Explains possible label efficiency through lower inverse-model complexity and stochasticity. The discussion assumes a curated action-free expert distribution and distinguishes generalization from sampling costs.

**Use:** Motivation for measuring label efficiency. It does not prove a LoRA advantage under cross-task visual transfer or imply that lower action MSE guarantees control success.

### LoRA — `lora`

[paper](https://arxiv.org/abs/2106.09685). **Read:** §§4.1–4.2.

Freezes a pretrained matrix and learns a low-rank additive update. The update can be merged into the matrix for inference.

**Use:** Cite the parameterization. “No additional inference latency” is tied to the merged implementation; do not assume it for our measured adapter path. Frozen base weights also do not guarantee unchanged predictions with an adapter active.

### FOCA — `foca`

[full text reviewed](https://arxiv.org/html/2606.20867v1); [ICML 2026 record](https://proceedings.mlr.press/v306/nguyen26b.html). **Read:** §3 and the data-scaling evaluation setup.

Adds explicit prediction of future interaction embeddings and implicit alignment to future observations during VLA adaptation. Action-free videos can supply auxiliary losses; at inference, learned tokens condition action prediction without supplying the realized future image.

**Use:** Recent few-shot representation-adaptation neighbor. Our next frame is an observed model input, unlike FOCA's future supervision. The bibliography uses the reviewed arXiv version's author spellings; PMLR renders several names differently. No numerical comparison is made.

## Benchmark context and screened adjacent work

### LIBERO — `libero`

[paper](https://arxiv.org/abs/2306.03310). **Read:** §§3–4/task-suite definitions.

Organizes manipulation tasks to probe transfer involving spatial relationships, objects, goals, and mixed skills. LIBERO-Goal changes goals while sharing objects and layout.

**Use:** Cite the benchmark, then describe our own six/two/two split separately. Sharing a benchmark name does not make task partitions, target data, or evaluation endpoints equivalent. Holding out tasks from our source training does not certify absence from upstream pretraining.

### LIBERO-MAX — `liberomax`

[29 September 2026 preprint](https://arxiv.org/abs/2609.36518v1). **Read:** abstract and benchmark overview; not included in the first section draft.

Introduces paired episodes with and without mid-execution events to study response to changes during control. This is deployment-time behavioral adaptation, not necessarily parameter adaptation.

**Use:** Newly surfaced since the previous list. Relevant if we later claim robustness or move to closed-loop evaluation; it is outside the recorded-transition comparison.

### SIDIL — citation deferred

[IEEE primary record](https://ieeexplore.ieee.org/document/11417181/). **Read:** indexed publisher abstract only. Exact title: *Toward Reliable Imitation Learning With Limited Expert Demonstrations via Search-Based Inverse Dynamic Learning*.

The abstract describes nearest-neighbor search, trajectory perturbation, and stitching to augment imitation data. This is an adjacent inverse-dynamics/retrieval direction. The full paper and complete bibliographic metadata remain unaudited. Retained from the old list, but excluded from the draft and BibTeX to avoid inventing details.

## Claim-to-source map for the first draft

| Draft claim | Supporting sources | Boundary |
|---|---|---|
| Frozen visual transfer and visual adaptation are both established | MAE, R3M, VC-1 | Does not predict which wins for our IDM |
| Inverse dynamics supports action labeling and video-based control | BCO, UniPi, Seer, DreamGen | Recorded futures and predicted futures are different inputs |
| Few-shot LoRA IDM adaptation exists | FADA §4.2, §5.5 | Novelty must come from the comparison and evidence |
| Retrieval is a substantive action-prediction/control alternative | VINN, REGENT, CLAW, RT-Cache | Different keys, banks, and action horizons |
| Retrieval can select training data or augment a learned policy | Behavior Retrieval, FlowRetrieval, STRAP, IWR, HAND, RAEA | Do not call these training-free action lookup |
| Lightweight updates and generated/retrieved adapters are established | LoRA, TAIL, OpenVLA/OFT, WIZARD, VLA-Pro | Cost must include the stage being compared |
| Future supervision can improve few-shot policy adaptation | FOCA | Does not evaluate our realized-frame input protocol |
| The experiment asks a restricted, controlled empirical question | `RESEARCH.md`, read alongside the above | No scientific outcome or exhaustive novelty claim yet |

## Remaining publication checks

The important method overlaps have been resolved sufficiently to write a first draft. A final submission still needs a refreshed citation search, implementation audits if any external method becomes a named reproduced baseline, complete checkpoint pretraining provenance, and results from the locked protocol. SIDIL remains an explicitly screened item. Avoid claims of “first,” universal superiority, or a literature-wide absence of matched comparisons based on this review.
