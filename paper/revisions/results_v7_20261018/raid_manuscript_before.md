# RAID: Retrieval and Low-Rank Adaptation for Few-Shot Generalization in World-Model-Based Robot Control

> Status: proposed protocol v6; no v6 experimental results. Synchronized from the updated [Google Doc](https://docs.google.com/document/d/11QDjW49qwxiZFJGYxMB5_dELf5keka6Qeh3ijXpW10s) on 11 October 2026.

*Working manuscript | Proposed protocol v6 | Revised 11 October 2026*

## Abstract

Study abstract—results pending. We investigate whether transition retrieval and low-rank parameter adaptation provide complementary benefits for few-shot generalization of robot control. A frozen visual generator proposes futures, and a separate inverse-dynamics controller maps current observations, robot state and a proposed future to action chunks. A two-by-two design varies target retrieval and LoRA plus head adaptation, yielding frozen, retrieval-only, adaptation-only and hybrid conditions from a common source checkpoint. The hybrid is trained with demonstration-level retrieval exclusions and retains a frozen reference representation for memory lookup. The initial LIBERO-Goal development pilot uses five demonstrations per task and 40 episodes across two tasks and four conditions, conditional on a source-only generator and cost gate. Recorded-transition diagnostics, paired execution outcomes and transition coverage test the complementarity hypothesis. No improved performance, new WAM architecture or first-in-the-literature claim is established; implementation and scientific evaluation remain pending.

Keywords: visual world models; inverse dynamics; retrieval; few-shot adaptation; LoRA; robot learning.

## 1. Introduction

Visual world models make possible futures available as images or video, but a visual proposal must still be translated into executable robot actions. UniPi connects video planning with inverse dynamics [2](https://arxiv.org/abs/2302.00111). More recent systems reinforce this foundation: π0.7 conditions a generalist controller on visual subgoals [30](https://arxiv.org/abs/2604.15483), while OpenWAM supports composable visual prediction and inverse-dynamics components [31](https://arxiv.org/abs/2610.07922). These precedents establish the architectural setting for our study. Our question concerns how the action-inference component should acquire task-specific knowledge when the visual predictor stays fixed.

With few target demonstrations, a controller can use explicit examples at prediction time, change its parameters, or do both. Retrieval may provide a relevant motion reference, while parameter adaptation may change how that reference is interpreted in the current scene. These mechanisms may complement one another, but can also be redundant or amplify overfitting. We therefore ask whether retrieval-conditioned LoRA adaptation improves few-shot generalization over either mechanism alone, and under which transition-coverage and cost conditions any improvement occurs.

We propose a two-by-two design with retrieval and LoRA plus action-head adaptation as factors. The four conditions are a frozen source controller, RAID with a target transition bank, LoRA plus head adaptation without retrieval, and RAID plus LoRA with retrieval active during adaptation and inference. All conditions start from identical source encoder and head weights and share a fixed-width input interface trained with present and masked memory. Head-only fine-tuning remains a smaller recorded-data diagnostic. The hypothesis concerns complementary action-inference mechanisms, not a presumption that the hybrid must win.

The pipeline generates a visual proposal from the current RGB observation and task instruction, then predicts an action chunk from that proposal, the current observation, current proprioception and the requested horizon. Cosmos3-Edge remains a candidate frozen generator [7](https://huggingface.co/nvidia/Cosmos3-Edge),[32](https://arxiv.org/abs/2606.02800). Although the foundation supports action-related modes, this experiment uses only image-to-video generation and does not consume its native actions. A source-only feasibility study must establish runtime compatibility, useful robot imagery, temporal-interface consistency and measured cost before generated-future confirmation. Matched inputs share generated clips; diverged trajectories replan from their own observations.

The initial benchmark is LIBERO-Goal [8](https://arxiv.org/abs/2306.03310), retaining six source tasks, two development tasks and two sealed final tasks. The first development pilot uses five support demonstrations, one source/support repetition and five query initial states per development task: 40 episodes across four conditions. This scope is a feasibility study, not a broad generalization result. After runtime measurement and a development freeze, a separate narrow confirmation can evaluate the two final tasks with three repetitions. LIBERO-90 expansion and larger demonstration sweeps are deferred, rather than prerequisites for the starting study.

The intended contribution is controlled evidence about when external transition memory and parametric adaptation complement one another in an established visual-planning pipeline. Generalization here means performance on held-out target-task episodes after receiving a few labeled support demonstrations. It does not mean zero-shot transfer or improved prediction by the frozen world model. ReCAP already uses retrieval in a WAM [40](https://arxiv.org/abs/2606.15631), and VLA-Pro retrieves LoRA memories [28](https://arxiv.org/abs/2605.29562v1); combining retrieval and adaptation alone is therefore not a sufficient novelty claim. The proposed contribution depends on the mechanism, fair controls, measured outcomes and their limits.

## 2. Related Work

### 2.1 Visual world models and action inference

World–action modeling connects future prediction with action generation, with different choices about how tightly the two are coupled. DreamZero models video and actions jointly [33](https://arxiv.org/abs/2602.15922). Cosmos 3 supports multiple visual and action input–output configurations [32](https://arxiv.org/abs/2606.02800). In contrast, our experiment deliberately fixes a video generator and studies a separate inverse model. MIRA motivates inspectable visual prediction in its own game domain [1](https://arxiv.org/abs/2607.05352v2); it does not establish transfer to manipulation. The availability of Cosmos3-Edge likewise does not demonstrate physically valid or useful LIBERO futures [7](https://huggingface.co/nvidia/Cosmos3-Edge).

Video planning followed by action inference has substantial precedent. UniPi uses a separately trained inverse model [2](https://arxiv.org/abs/2302.00111). π0.7 employs visual subgoal conditioning and interprets the resulting action prediction as an inverse-dynamics-like problem; it also trains with both real and generated subgoals to reduce input mismatch [30](https://arxiv.org/abs/2604.15483). OpenWAM defines local dynamics interfaces using current observations, proprioception and a supplied future, and studies reuse of an inverse model with adapted visual predictors [31](https://arxiv.org/abs/2610.07922). RAID asks the complementary adaptation question: with the visual generator fixed, should scarce target experience update external memory or the inverse model's parameters? GE-Act 2.0 further connects visual planning and inverse dynamics, including an explicit mechanism for aligning generated futures with action supervision [39](https://arxiv.org/abs/2609.05588).

Inverse dynamics supports learning from observation [9](https://arxiv.org/abs/1805.01954), predictive robot models [10](https://arxiv.org/abs/2412.15109v1), and policy supervision from synthetic video [11](https://arxiv.org/abs/2505.12705v2); its label efficiency depends on the learning setting [12](https://arxiv.org/abs/2602.02762). A generated future may depict a path different from the recorded demonstration, so it cannot automatically inherit that demonstration's action labels. OpenWAM's executed simulator branches illustrate a valid way to broaden transition supervision [31](https://arxiv.org/abs/2610.07922). Such source-data interventions are deferred here to keep the initial experiment focused and affordable; the shift from recorded to generated imagery remains an explicit limitation.

### 2.2 Visual representations for inverse dynamics

Pretrained encoders supply representations for inverse dynamics. MAE learns image features through masked reconstruction [13](https://arxiv.org/abs/2111.06377), R3M studies frozen representations for manipulation [14](https://arxiv.org/abs/2203.12601v3), and VC-1 highlights variation across embodied tasks and adaptation settings [15](https://arxiv.org/abs/2303.18240v2). We distinguish the controller representation from the retrieval reference representation. LoRA may change the controller's features in the adaptation-only and hybrid conditions, while memory keys, retrieval queries and coverage measurements always use the same frozen reference encoder. This prevents controller adaptation from invalidating the geometry of the retrieval bank.

### 2.3 Retrieval for action prediction and policy learning

VINN separates visual representation learning from behavior learning and predicts actions through locally weighted nearest-neighbor regression [3](https://arxiv.org/abs/2112.01511). REGENT combines retrieved demonstrations with a learned transformer policy to support adaptation without mandatory test-time fine-tuning [16](https://arxiv.org/abs/2412.04759). RT-Cache retrieves and replays action snippets from visual memory [17](https://arxiv.org/abs/2505.09040v3), while CLAW maps planned latent actions to executable controls through nearest-neighbor lookup in labeled reference data [18](https://arxiv.org/abs/2606.04130v1). These methods establish retrieval as a substantive action-inference mechanism. RAID's proposed bank-plus-MLP arm instead uses retrieved information as input to a learned predictor. It must therefore be distinguished from simply copying a neighbor's action or averaging retrieved actions.

Retrieval can select training data or supply context during action prediction. Behavior Retrieval [19](https://arxiv.org/abs/2304.08742), FlowRetrieval [20](https://proceedings.mlr.press/v270/lin25a.html), STRAP [21](https://arxiv.org/abs/2412.15182) and importance-weighted retrieval [22](https://arxiv.org/abs/2509.01657v1) explore data selection; HAND retrieves robot experience and fine-tunes a policy [23](https://arxiv.org/abs/2505.20455v4), while RAEA integrates experience through cross-attention [24](https://arxiv.org/abs/2404.11699v1). ReCAP adds trajectory retrieval to Cosmos Policy and extends task coverage by adding demonstrations while the policy stays frozen [40](https://arxiv.org/abs/2606.15631). RAID instead tests retrieval and target-time parameter adaptation jointly within a separate inverse controller. This is a narrower empirical question than introducing retrieval to WAMs.

### 2.4 Parameter-efficient adaptation and reusable adapters

LoRA freezes an original weight matrix and learns an additive low-rank update [6](https://arxiv.org/abs/2106.09685). TAIL evaluates LoRA and other lightweight adaptation mechanisms for continual robot imitation [4](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e2b525da536030ec1589dddea01feb-Abstract-Conference.html). OpenVLA studies efficient policy fine-tuning [25](https://arxiv.org/abs/2406.09246v3), and OpenVLA-OFT shows that action representation, chunking, decoding, and the training objective affect both performance and inference cost [26](https://arxiv.org/abs/2502.19645v2). These findings motivate reporting measured resource use in addition to trainable parameter count. Frozen base weights also do not imply that predictions remain unchanged when a target adapter is active.

FADA adapts inverse dynamics behind a frozen planner [5](https://arxiv.org/abs/2606.28476v1). WIZARD [27](https://arxiv.org/abs/2606.07217v1), VLA-Pro [28](https://arxiv.org/abs/2605.29562v1) and FOCA [29](https://arxiv.org/abs/2606.20867v1) investigate other adaptation mechanisms. In particular, VLA-Pro retrieves and fuses task-specific LoRA adapters, whereas RAID's memory stores labeled visual transitions. RoboTTT uses test-time learning for robot context [34](https://arxiv.org/abs/2607.15275), and MEM combines short visual and long linguistic memory [37](https://www.pi.website/research/memory). Retrieval/fine-tuning hybrids also appear in LM-based world modeling [41](https://arxiv.org/abs/2610.02542), a different domain. These precedents motivate complementarity tests while ruling out broad novelty claims about combining memory and parameter updates.

### 2.5 Scope of the comparison

The primary experiment isolates two factors in a common action-inference pipeline: access to target transition memory and LoRA plus head updates. RoboWM-Bench [35](https://arxiv.org/abs/2604.19092) and EVA [36](https://arxiv.org/abs/2603.17808) motivate judging proposals through execution rather than appearance alone. Fast-WAM [38](https://arxiv.org/abs/2603.16666) motivates measuring the full cost of explicit future generation. We keep unfiltered task success as the control endpoint, report the hybrid's paired differences from both individual mechanisms, and distinguish the small development pilot from any later registered confirmation. No published scores from differently trained WAMs are treated as matched baselines.

## 3. Problem Formulation

We specify protocol v6 as a proposed implementation target for a modular control pipeline. A two-by-two intervention varies target retrieval R and LoRA plus head adaptation A. The generator and source-data access stay fixed. The initial study is a five-shot LIBERO-Goal development pilot; generator feasibility, exact runtime cost and any final confirmation scope require validation.

- At time t, the controller observes RGB image oₜ and current proprioception qₜ. The generator receives oₜ and task instruction g and returns a visual proposal V. The controller receives oₜ, qₜ, a timestamped future observation from V, the requested duration, and either a retrieved transition block or its explicit null representation. Source training, target support, development queries and final evaluation have disjoint roles.

- The generator Wψ is frozen. Its allowed conditioning is exactly the current RGB frame and instruction. Recorded future observations, recorded or target action trajectories, privileged object states and native generator actions are excluded from the primary generated-future pipeline. Simulator state may be used by the evaluator for reset and scoring, never as privileged policy input.

- The proposed target is an eight-step chunk aₜ,…,aₜ₊₇ in the verified seven-dimensional LIBERO action convention. The starting interface assumes 20 Hz control and 20 fps video, with one chunk executed before replanning. These rates, observation/action indexing and selected frame offsets require replay and runtime checks. Nominal video timestamps do not prove physical alignment. Current proprioception is normalized using source-training statistics; its schema must be identical in stored data and live observations.

- Recorded-transition evaluation conditions on the actual recorded future and uses its logged action chunk as a prediction diagnostic. Generated-future evaluation conditions on a proposal and measures executed task success. Logged actions from a different trajectory are not ground-truth labels for an arbitrary generated future. Plan-quality strata are secondary analyses of generated evaluation, not additional adaptation methods.

## 4. Method

### 4.1 Shared frozen visual world model

Cosmos3-Edge remains pinned as a candidate image-to-video generator, with fixed preprocessing, prompts, runtime, sampling and seed provenance. Before generated-future confirmation, we require a source-only panel of 30 cases across the six source tasks, valid decoded outputs and frame indices, verified expert replay, qualitative assessment of viewpoint and object/robot continuity, and measured latency, memory and storage. The proposed engineering thresholds are 30/30 technically valid clips, at least 24/30 passing the predefined plausibility rubric, and at least 27/30 successful source expert replays. These thresholds are continuation rules, not proof of executability. Feature alignment to an unrelated demonstration is retained only as a heuristic diagnostic. Identical full conditioning inputs can reuse a clip; after divergence, each condition replans from its own state. The source gate and all failures are reported.

### 4.2 RAID bank with an MLP action head

One source controller receives current/future MAE features, current proprioception, duration and a fixed-width retrieval block. Source training averages retrieval-present and fully masked-memory losses with equal weight, excluding the query's entire demonstration from its episodic bank. RAID keeps this controller frozen and fills a target-support bank. Memory lookup always uses frozen reference features, including when controller features are LoRA-adapted. The default distance combines current/future cosine distances and normalized proprioceptive distance, with matching duration. Four neighbor slots contain normalized action chunks, similarity and validity flags, followed by a weighted mean. Missing neighbors are masked; an empty bank yields a finite null block.

### 4.3 LoRA adaptation and the RAID + LoRA hybrid

Both adapted conditions reset from the same source checkpoint and update rank-8, alpha-16 LoRA in MAE blocks 10–11 qkv/proj plus the common head; all remaining base weights and the generator stay fixed. LoRA-only always receives null memory. The hybrid receives support retrieval during fitting and inference, but every fitting query excludes its entire parent demonstration, including overlapping windows, from the bank. With five demonstrations it retrieves from the other four during adaptation and from all five on held-out evaluation. Controller adaptation never changes reference retrieval features. The pilot uses a fixed 500 updates with matched support queries, optimizer settings and sampling; any later tuning is development-only and equally available to both adapted conditions.

### 4.4 Training and comparison controls

The four primary conditions are source_frozen (R=0,A=0), raid_bank_mlp (1,0), lora_mlp (0,1), and raid_lora_mlp (1,1). They share source weights, support identities, normalizers, controller interface, action horizon, losses and evaluation cases. The adaptation factor includes both encoder LoRA and head updates; head-only development diagnostics help assess attribution without adding a fifth primary rollout condition. Zero-effect adapter insertion must preserve predictions before fitting. The hybrid receives no additional labeled demonstrations. Its extra reference-encoding and adaptation costs are measured. These controls test complementarity within this implementation, not universal superiority of a memory mechanism.

Table 1. Four primary conditions and one recorded-data diagnostic. The visual generator is frozen throughout.

| Condition | Target retrieval | Target updates |
| --- | --- | --- |
| Frozen baseline | No | None |
| RAID | Yes | None |
| LoRA + head | No | Encoder LoRA + head |
| RAID + LoRA | Yes | Encoder LoRA + head |
| Head-only diagnostic | No | Head only; recorded development |

## 5. Experimental Design

### 5.1 Benchmark, splits, and demonstration budgets

Use the ten-task LIBERO-Goal suite with the existing six-source/two-development/two-final split and immutable support/query membership. Target tasks retain a 30-demo support pool and 20 query demonstrations. The initial primary budget is five support demonstrations. Hash-select five query initial states for each development task and use seed 42, producing 40 pilot episodes across the four conditions. A later narrow final scope can use seeds 42/43/44 and ten starts per final task, totaling 240 episodes, or a preregistered five-start scope totaling 120. No final outcome may determine scope. LIBERO-90 and generated 2/10-shot sweeps remain optional future extensions. Held-out study tasks are not guaranteed unseen in foundation pretraining.

### 5.2 Recorded-transition diagnostics

Evaluate all four primary conditions on separate recorded query transitions at five shots. Report normalized chunk MSE, physical action errors, first-step error and gripper accuracy, averaging within demonstrations before tasks. Head-only and weighted k-NN are development diagnostics. Recorded 2/10-shot analyses may be added if their controller cost is feasible, with explicit scope labels. Source retention keeps the deployed target configuration active, including both target bank and adapter for the hybrid; resetting them is a separate reference. Coverage is measured in one frozen feature space for every condition. Recorded action error does not establish successful execution of an arbitrary generated plan.

### 5.3 Execution of generated futures

The primary control endpoint is unfiltered official task success. Begin with the 40-episode development pilot, contingent on the source-only generator gate. Pair task, support set, source seed, initial state and generation seed across all four conditions. Use the same initial clip at identical inputs; after actions diverge, each condition replans from its own observations. The proposed interface executes eight actions before replanning, up to 300 steps. Keep generated and executed videos, timeouts, errors, saturation and all denominators. Optional quality labels for shared initial proposals must be assigned independently of controller outcomes; report unknowns and full-set success. A visually plausible proposal need not be executable, and a controller failure alone does not prove impossibility.

### 5.4 Resource measurements and uncertainty

Let S_RA denote task-macro success with retrieval R and adaptation A. The co-primary hybrid contrasts are S_11−S_10 and S_11−S_01; benefit over both components requires evidence for both contrasts. The secondary interaction S_11−S_10−S_01+S_00 describes non-additivity on the success-probability scale and does not alone prove superiority to both. The single-seed development pilot supports descriptive paired counts only. A registered final analysis must preserve task, source/support and initial-state pairing, report per-task uncertainty, and address multiplicity for simultaneous claims. Two final tasks do not support broad population-level generalization. Measure all source, adaptation, retrieval, reference-encoding, generation, simulation and storage costs. At 38 plans per full episode, the pilot can request up to 1,520 clips and the 240-episode confirmation up to 9,120, before valid cache reuse. These are workload bounds, not measured runtime or approval to spend. Report cached study cost and uncached deployment latency.

### 5.5 Transition coverage and deferred extensions

Use frozen reference-space distances to describe coverage on recorded queries and shared initial generated plans. Any low/medium/high bin boundaries must be fixed on development data and retain sparse-bin counts. Later on-policy coverage is descriptive because trajectories differ. The hypothesis is that adaptation may help the controller use imperfect but relevant retrieved transitions; this is not an established result. Simulator-branch source training, visual perturbations, additional suites, multiple generators and broad budget sweeps are deferred. A future robustness test would freeze support and adapted weights before evaluating a specified perturbation distribution. It must not be inferred from ordinary held-out episodes or added after inspecting final outcomes.

## 6. Results

Section outline. Populate this section only from verified scientific artifacts. Do not insert expected wins, illustrative scores, or historical-protocol results as current evidence.

### 6.1 Recorded-transition action inference

Section outline. Present recorded errors for all four primary conditions at the registered support budget, with head-only and k-NN diagnostics clearly separated. Report deployed-configuration source retention and coverage counts. Separate development pilot evidence from final confirmation and keep older protocol measurements distinct.

### 6.2 Task execution from generated futures

Section outline. Present unfiltered success counts for frozen, RAID, LoRA and RAID + LoRA, with paired hybrid-minus-RAID and hybrid-minus-LoRA differences and uncertainty appropriate to the registered scope. Report the secondary interaction, ceilings/floors and all incomplete episodes. Show shared initial proposals and separately labeled generated and executed trajectories.

### 6.3 Accuracy, success, and computational cost

Section outline. Table or plot: adaptation and inference cost, memory, and storage alongside performance. Separate shared generation overhead from costs that differ between the two methods.

### 6.4 Failure analysis

Section outline. Examine hybrid failures and gains against frozen-reference transition coverage, support examples, ambiguous actions, generator artifacts and timing. Distinguish causal evidence from hypotheses and allow mixed or unknown causes. Include negative hybrid outcomes, with a fixed example-selection rule.

## 7. Discussion and Limitations

The study evaluates a small inverse controller downstream of a frozen visual generator. It cannot establish better world-model prediction or superiority to native Cosmos, ReCAP or other WAMs without matched experiments. The hybrid may overfit or rely too strongly on neighbors; a deterministic MLP may average action modes. Frozen lookup features avoid bank drift but add inference work and may miss useful task-specific similarity. Five-shot adaptation is not zero-shot transfer, and two final tasks provide narrow evidence. Recorded/generated appearance and motion mismatch remains unresolved. Plausibility labels do not prove reachability; nominal frame time does not ensure physical timing. A paused simulator and cached clips do not establish real-time deployment. Unknown upstream benchmark exposure limits novelty-of-task claims. Results may favor an individual mechanism or show no useful control; those outcomes must remain reportable.

## 8. Conclusion

Section outline. State whether the hybrid improves over both retrieval-only and adaptation-only under the evaluated support, task and cost conditions. Report effect sizes and uncertainty, distinguish the pilot from confirmation, and explain any negative or redundant interaction. Do not infer universal generalization, a new WAM architecture or real-world transfer.

## Acknowledgments and Declarations

Section outline. Add confirmed contributors, funding, conflicts of interest, and applicable ethics or data-use statements. Do not invent authors, affiliations, approvals, or funding. Describe the provenance and permitted use of benchmark data and model checkpoints.

## Reproducibility and Artifact Availability

Section outline. Release the v6 code and immutable scope, common source checkpoints, support/query identities, demonstration exclusions, reference/controller encoder provenance, normalizers, timing schema, generator gate records, access ledger, predictions, rollouts and reproducible reports. Preserve v3/v4 implementation evidence and the superseded v5 proposal under their original identities. This manuscript specifies intended changes; synthetic tests are not scientific results, and no v6 experiment or paid execution is established.

## References

[1] Hu et al. Multiplayer Interactive World Models with Representation Autoencoders. Technical report, 2026. [arXiv:2607.05352v2](https://arxiv.org/abs/2607.05352v2).

[2] Du et al. Learning Universal Policies via Text-Guided Video Generation. arXiv preprint, 2023. [arXiv:2302.00111](https://arxiv.org/abs/2302.00111).

[3] Pari et al. The Surprising Effectiveness of Representation Learning for Visual Imitation. Robotics: Science and Systems, 2022. [arXiv:2112.01511](https://arxiv.org/abs/2112.01511).

[4] Liu et al. TAIL: Task-specific Adapters for Imitation Learning with Large Pretrained Models. International Conference on Learning Representations, 2024. [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e2b525da536030ec1589dddea01feb-Abstract-Conference.html).

[5] Xie et al. FADA: Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control. arXiv preprint, 2026. [arXiv:2606.28476v1](https://arxiv.org/abs/2606.28476v1).

[6] Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. International Conference on Learning Representations, 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).

[7] NVIDIA. Cosmos3-Edge model card. 2026. [Official model card](https://huggingface.co/nvidia/Cosmos3-Edge). Accessed 11 October 2026.

[8] Liu et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv preprint, 2023. [arXiv:2306.03310](https://arxiv.org/abs/2306.03310).

[9] Torabi et al. Behavioral Cloning from Observation. International Joint Conference on Artificial Intelligence, 2018. [arXiv:1805.01954](https://arxiv.org/abs/1805.01954).

[10] Tian et al. Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation. arXiv preprint, 2024. [arXiv:2412.15109v1](https://arxiv.org/abs/2412.15109v1).

[11] Jang et al. DreamGen: Unlocking Generalization in Robot Learning through Video World Models. arXiv preprint, 2025. [arXiv:2505.12705v2](https://arxiv.org/abs/2505.12705v2).

[12] Morin et al. On the Sample Efficiency of Inverse Dynamics Models for Semi-Supervised Imitation Learning. Proceedings of the 43rd International Conference on Machine Learning, 2026. [arXiv:2602.02762](https://arxiv.org/abs/2602.02762).

[13] He et al. Masked Autoencoders Are Scalable Vision Learners. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377).

[14] Nair et al. R3M: A Universal Visual Representation for Robot Manipulation. arXiv preprint, 2022. [arXiv:2203.12601v3](https://arxiv.org/abs/2203.12601v3).

[15] Majumdar et al. Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence? arXiv preprint, 2023. [arXiv:2303.18240v2](https://arxiv.org/abs/2303.18240v2).

[16] Sridhar et al. REGENT: A Retrieval-Augmented Generalist Agent That Can Act In-Context in New Environments. International Conference on Learning Representations, 2025. [arXiv:2412.04759](https://arxiv.org/abs/2412.04759).

[17] Kwon et al. RT-Cache: Training-Free Retrieval for Real-Time Manipulation. arXiv preprint, 2025. [arXiv:2505.09040v3](https://arxiv.org/abs/2505.09040v3).

[18] Ayalew et al. CLAW: Learning Continuous Latent Action World Models via Adversarial Latent Regularization. arXiv preprint, 2026. [arXiv:2606.04130v1](https://arxiv.org/abs/2606.04130v1).

[19] Du et al. Behavior Retrieval: Few-Shot Imitation Learning by Querying Unlabeled Datasets. arXiv preprint, 2023. [arXiv:2304.08742](https://arxiv.org/abs/2304.08742).

[20] Lin et al. FlowRetrieval: Flow-Guided Data Retrieval for Few-Shot Imitation Learning. Proceedings of The 8th Conference on Robot Learning, 2025. [Paper](https://proceedings.mlr.press/v270/lin25a.html).

[21] Memmel et al. STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning. International Conference on Learning Representations, 2025. [arXiv:2412.15182](https://arxiv.org/abs/2412.15182).

[22] Xie et al. Data Retrieval with Importance Weights for Few-Shot Imitation Learning. Proceedings of The 9th Conference on Robot Learning, 2025. [arXiv:2509.01657v1](https://arxiv.org/abs/2509.01657v1).

[23] Hong et al. HAND Me the Data: Fast Robot Adaptation via Hand Path Retrieval. arXiv preprint, 2025. [arXiv:2505.20455v4](https://arxiv.org/abs/2505.20455v4).

[24] Zhu et al. Retrieval-Augmented Embodied Agents. arXiv preprint, 2024. [arXiv:2404.11699v1](https://arxiv.org/abs/2404.11699v1).

[25] Kim et al. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv preprint, 2024. [arXiv:2406.09246v3](https://arxiv.org/abs/2406.09246v3).

[26] Kim et al. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. arXiv preprint, 2025. [arXiv:2502.19645v2](https://arxiv.org/abs/2502.19645v2).

[27] Bianchi et al. Robotic Policy Adaptation via Weight-Space Meta-Learning. arXiv preprint, 2026. [arXiv:2606.07217v1](https://arxiv.org/abs/2606.07217v1).

[28] Si et al. VLA-Pro: Cross-Task Procedural Memory Transfer for Vision-Language-Action Models. arXiv preprint, 2026. [arXiv:2605.29562v1](https://arxiv.org/abs/2605.29562v1).

[29] Nguyen et al. FOCA: Future-Oriented Conditioning for Data-Efficient Vision-Language-Action Adaptation. arXiv preprint, 2026. [arXiv:2606.20867v1](https://arxiv.org/abs/2606.20867v1).

[30](https://arxiv.org/abs/2604.15483) Physical Intelligence. π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities. 2026. [https://arxiv.org/abs/2604.15483](https://arxiv.org/abs/2604.15483)

[31](https://arxiv.org/abs/2610.07922) Yu et al. OpenWAM: An Open Framework for Composable World-Action Models. 2026. [https://arxiv.org/abs/2610.07922](https://arxiv.org/abs/2610.07922)

[32](https://arxiv.org/abs/2606.02800) NVIDIA. Cosmos 3: Omnimodal World Models for Physical AI. 2026. [https://arxiv.org/abs/2606.02800](https://arxiv.org/abs/2606.02800)

[33](https://arxiv.org/abs/2602.15922) Ye et al. World Action Models are Zero-shot Policies (DreamZero). 2026. [https://arxiv.org/abs/2602.15922](https://arxiv.org/abs/2602.15922)

[34](https://arxiv.org/abs/2607.15275) Jiang et al. RoboTTT: Context Scaling for Robot Policies. 2026. [https://arxiv.org/abs/2607.15275](https://arxiv.org/abs/2607.15275)

[35](https://arxiv.org/abs/2604.19092) Jiang et al. RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation. 2026. [https://arxiv.org/abs/2604.19092](https://arxiv.org/abs/2604.19092)

[36](https://arxiv.org/abs/2603.17808) EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards. 2026. [https://arxiv.org/abs/2603.17808](https://arxiv.org/abs/2603.17808)

[37](https://www.pi.website/research/memory) Torne et al. MEM: Multi-Scale Embodied Memory for Vision Language Action Models. 2026. [https://www.pi.website/research/memory](https://www.pi.website/research/memory)

[38](https://arxiv.org/abs/2603.16666) Yuan et al. Fast-WAM: Do World Action Models Need Test-time Future Imagination? 2026. [https://arxiv.org/abs/2603.16666](https://arxiv.org/abs/2603.16666)

[39](https://arxiv.org/abs/2609.05588) AgiBot Research Team et al. GE-Act 2.0: Pretraining and Scaling a World-Action Model for Robotic Manipulation. 2026. [https://arxiv.org/abs/2609.05588](https://arxiv.org/abs/2609.05588)

[40](https://arxiv.org/abs/2606.15631) Park et al. Retrieve, Don't Retrain: Extending Vision Language Action Models to New Tasks at Test Time (ReCAP). 2026. [https://arxiv.org/abs/2606.15631](https://arxiv.org/abs/2606.15631)

[41](https://arxiv.org/abs/2610.02542) Ashok et al. How To Train Your World Model: Fine-tuning vs RAG for LM-based World Modeling. 2026. [https://arxiv.org/abs/2610.02542](https://arxiv.org/abs/2610.02542)

## A. Complete Implementation and Training Details

Section outline. Specify the common source objective, fixed-width memory/null interface, frozen lookup versus adapted controller features, whole-demo exclusion during hybrid fitting, head and LoRA settings, state/action timing, optimizer reset, support sampling and all measured computation costs.

## B. Full Evaluation Protocol and Results

Section outline. Publish the existing Goal split, support/query identities, 40-episode development inventory, any independently registered 120/240-episode final inventory, access history, source gate, paired hybrid contrasts, interaction and uncertainty method. Keep expanded suites and data regimes separate and optional.

## C. Visual Rollout Gallery and Failure Cases

Section outline. Provide generated-proposal and executed-rollout videos with clear labels, fixed example-selection criteria, quality rubric and annotator provenance. Include rejected, failed and unknown cases. Explain shared initial plans, own-observation replanning, and why plausible appearance or controller failure alone cannot establish physical executability.
