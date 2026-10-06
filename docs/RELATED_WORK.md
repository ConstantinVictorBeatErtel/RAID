# Related-work coverage audit

Reviewed **4 October 2026**. This audit follows the recorded-RGB, standalone MAE few-shot protocol in [RESEARCH.md](../RESEARCH.md). The previous list is preserved in [the 25 September snapshot](history/related-work-2026-09-25.md). The old snapshot's reference to further historical search records “below” did not correspond to any additional text in the file; this audit therefore explicitly tracks its named papers.

The first manuscript section is available as [readable Markdown](../paper/related_work_v1.md) and [standalone LaTeX](../paper/related_work_v1.tex). [Reading notes](../paper/literature_notes.md) provide source locations, evidence boundaries and a claim-to-source map. [BibTeX](../paper/references.bib) contains 28 verified title/author records; the draft cites 27. [Source metadata](../paper/literature_sources.json) records versions and review depth.

## Coverage decision

The old list was incomplete for the active question. It emphasized recent systems but omitted inverse-dynamics foundations, visual-representation transfer, and several major forms of training-data retrieval. The revised list contains **29 screened or reviewed works**: all 17 previously named works plus 12 additions. There are **26 focused full-text reviews**, two primary abstract/overview reviews, and one publisher-abstract-only item (SIDIL). The specifically prioritized TAIL, FADA, HAND and WIZARD now have method-level notes.

The revised coverage is sufficient for this first section: visual representation transfer; inverse-dynamics learning and adaptation; direct action retrieval; retrieval for policy training; low-rank adaptation; generated/retrieved adapter memories; and benchmark scope. This is a scoped literature audit, not a systematic review or proof that no equivalent comparison exists. SIDIL remains incompletely read and is excluded from manuscript claims. Exact baseline reproductions would require separate code audits.

## Material corrections

- **FADA explicitly performs LoRA adaptation of its IDM.** The former note left this unresolved. Neither IDM adaptation nor few-shot LoRA IDM adaptation can be claimed as new.
- **WIZARD includes within-suite held-out tasks**, not just held-out suites. Its nearest-neighbor baseline retrieves adapters, not transition actions.
- **TAIL uses 40 training and ten validation demonstrations per task** in the described experiment, not 50 training demonstrations. Its saved-adapter retention is distinct from testing the new target adapter on source data.
- **OpenVLA's efficient-fine-tuning table uses a model variant.** The last-layer condition is not equivalent to our action-head-only baseline.
- **REGENT's robotics observations are proprioceptive.** Its image experiments should not be used to imply a visual LIBERO robotics protocol.
- **HAND and RT-Cache have substantive version updates.** The notes and bibliography use HAND v4 and RT-Cache v3; earlier evaluation descriptions are not silently retained.

## Reviewed inventory

“Method/context” means focused passages from the full paper, with exact locations in the notes; it does not mean code reproduction. Publication years and arXiv dates are kept distinct in the bibliography. A preprint citation is intentional where that is the version read.

| Key / primary paper | Prior list | Priority | Reading depth | Draft |
|---|---|---|---|---|
| `tail` — [TAIL: Task-specific Adapters for Imitation Learning with Large Pretrained Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e2b525da536030ec1589dddea01feb-Abstract-Conference.html) | Retained | closest comparison | Method/context | Yes |
| `fada` — [FADA: Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control](https://arxiv.org/abs/2606.28476v1) | Retained | closest comparison | Method/context | Yes |
| `hand` — [HAND Me the Data: Fast Robot Adaptation via Hand Path Retrieval](https://arxiv.org/abs/2505.20455v4) | Retained | closest comparison | Method/context | Yes |
| `wizard` — [Robotic Policy Adaptation via Weight-Space Meta-Learning](https://arxiv.org/abs/2606.07217v1) | Retained | closest comparison | Method/context | Yes |
| `openvla` — [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246v3) | Retained | closest comparison | Method/context | Yes |
| `oft` — [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645v2) | Retained | closest comparison | Method/context | Yes |
| `regent` — [REGENT: A Retrieval-Augmented Generalist Agent That Can Act In-Context in New Environments](https://arxiv.org/abs/2412.04759) | Retained | closest comparison | Method/context | Yes |
| `vlapro` — [VLA-Pro: Cross-Task Procedural Memory Transfer for Vision-Language-Action Models](https://arxiv.org/abs/2605.29562v1) | Retained | closest comparison | Method/context | Yes |
| `vinn` — [The Surprising Effectiveness of Representation Learning for Visual Imitation](https://arxiv.org/abs/2112.01511) | Retained | closest comparison | Method/context | Yes |
| `flow` — [FlowRetrieval: Flow-Guided Data Retrieval for Few-Shot Imitation Learning](https://proceedings.mlr.press/v270/lin25a.html) | Retained | closest comparison | Method/context | Yes |
| `behavior` — [Behavior Retrieval: Few-Shot Imitation Learning by Querying Unlabeled Datasets](https://arxiv.org/abs/2304.08742) | Added | supporting context | Method/context | Yes |
| `strap` — [STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning](https://arxiv.org/abs/2412.15182) | Added | supporting context | Method/context | Yes |
| `iwr` — [Data Retrieval with Importance Weights for Few-Shot Imitation Learning](https://arxiv.org/abs/2509.01657v1) | Added | supporting context | Method/context | Yes |
| `rtcache` — [RT-Cache: Training-Free Retrieval for Real-Time Manipulation](https://arxiv.org/abs/2505.09040v3) | Retained | supporting context | Method/context | Yes |
| `raea` — [Retrieval-Augmented Embodied Agents](https://arxiv.org/abs/2404.11699v1) | Retained | supporting context | Method/context | Yes |
| `claw` — [CLAW: Learning Continuous Latent Action World Models via Adversarial Latent Regularization](https://arxiv.org/abs/2606.04130v1) | Retained | closest comparison | Method/context | Yes |
| `mae` — [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) | Added | supporting context | Method/context | Yes |
| `r3m` — [R3M: A Universal Visual Representation for Robot Manipulation](https://arxiv.org/abs/2203.12601v3) | Added | supporting context | Method/context | Yes |
| `vc1` — [Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?](https://arxiv.org/abs/2303.18240v2) | Added | screened context | Abstract/overview | Yes |
| `bco` — [Behavioral Cloning from Observation](https://arxiv.org/abs/1805.01954) | Added | supporting context | Method/context | Yes |
| `unipi` — [Learning Universal Policies via Text-Guided Video Generation](https://arxiv.org/abs/2302.00111) | Retained | supporting context | Method/context | Yes |
| `seer` — [Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation](https://arxiv.org/abs/2412.15109v1) | Retained | supporting context | Method/context | Yes |
| `dreamgen` — [DreamGen: Unlocking Generalization in Robot Learning through Video World Models](https://arxiv.org/abs/2505.12705v2) | Retained | supporting context | Method/context | Yes |
| `morin` — [On the Sample Efficiency of Inverse Dynamics Models for Semi-Supervised Imitation Learning](https://arxiv.org/abs/2602.02762) | Added | closest comparison | Method/context | Yes |
| `lora` — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | Added | supporting context | Method/context | Yes |
| `foca` — [FOCA: Future-Oriented Conditioning for Data-Efficient Vision-Language-Action Adaptation](https://arxiv.org/abs/2606.20867v1) | Added | closest comparison | Method/context | Yes |
| `libero` — [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310) | Added | supporting context | Method/context | Yes |
| `liberomax` — [LIBERO-MAX: Do Robot Policies Adapt When the World Changes?](https://arxiv.org/abs/2609.36518v1) | Added | screened context | Abstract/overview | No |
| SIDIL — [Toward Reliable Imitation Learning With Limited Expert Demonstrations via Search-Based Inverse Dynamic Learning](https://ieeexplore.ieee.org/document/11417181/) | Retained | Adjacent | Publisher abstract only | No |

## Search record and inclusion rules

Search date: 4 October 2026, using the user's local date. Started from every named work in the old list, verified identifiers at primary paper/proceedings pages, and followed relevant citations from TAIL, HAND, FlowRetrieval and STRAP. Repeated broad searches for combinations of “few-shot”, “visual inverse dynamics”, “LoRA”, “retrieval”, “robot adaptation” and “2026”; also searched September 2026 retrieval/adaptation and current LIBERO work. Primary sources were arXiv full text, ICLR proceedings, PMLR, RSS, CVF and IJCAI. Search-result summaries and third-party paper aggregators were discovery aids, not evidence for manuscript claims.

Added BCO, Morin et al., LoRA, MAE, R3M, VC-1, Behavior Retrieval, STRAP, Importance Weighted Retrieval, FOCA, the LIBERO benchmark citation and LIBERO-MAX. The last is a new 29 September 2026 benchmark and remains optional control/robustness context. Retained older RAID neighbors rather than dropping them because the experiment changed.

Included methods that directly inform the representation/adaptation/retrieval comparison, establish inverse-dynamics prior art, or define the benchmark. General VLA scaling, arbitrary world-model architectures, reinforcement-learning optimization, and unrelated humanoid systems are not individually catalogued unless they intersect those questions. This keeps the section directed at the actual experiment instead of claiming field-wide coverage.

## Supported positioning

The intended contribution is a controlled empirical comparison: given identical labeled support demonstrations and held-out recorded query transitions, does adapting the visual encoder with LoRA plus the action head improve action prediction beyond head-only tuning and transition retrieval, and what does the improvement cost? The protocol and result, including a possible null result, must carry that contribution.

This review does **not** support claims that retrieval, LoRA, their combination, IDM adaptation, or held-out LIBERO transfer are new. It also does not convert recorded-frame prediction into autonomous control, imply that upstream pretraining never saw related content, or establish broad robustness from two final-test tasks. No new scientific results are asserted in the draft.
