# Related Work

Learning from a small number of robot demonstrations raises a practical question: how should a model use the examples available for a new task? Prior work has addressed this question through transferable visual representations, inverse-dynamics learning, retrieval of past experience, and parameter-efficient adaptation. These approaches differ in both the information available at deployment and the component that changes when new demonstrations arrive. Our study examines these choices for visual inverse dynamics, where the input is a pair of recorded observations and the target is the action connecting them.

## Visual representations and inverse dynamics

Pretrained visual encoders provide a useful starting point when robot demonstrations are scarce. Masked autoencoders learn image representations through reconstruction of missing patches [1](https://arxiv.org/abs/2111.06377), while R3M uses human video to learn a representation that can remain frozen during downstream policy learning [2](https://arxiv.org/abs/2203.12601v3). The broader evaluation accompanying VC-1 shows that no pretrained representation is uniformly strongest across embodied tasks and that task or domain adaptation can substantially improve performance [3](https://arxiv.org/abs/2303.18240v2). This evidence motivates separating the benefit of learning an action predictor from the additional benefit of changing its visual representation. We make that distinction using a shared MAE encoder and source-trained inverse-dynamics head.

Inverse-dynamics models infer actions from transitions between observations. Behavioral Cloning from Observation uses an inverse model learned from an agent's interactions to supply action labels for observation-only demonstrations [4](https://arxiv.org/abs/1805.01954). UniPi instead couples a video planner with a separately trained inverse model that translates generated frames into actions [5](https://arxiv.org/abs/2302.00111). Seer jointly learns visual foresight and action prediction conditioned on predicted future representations [6](https://arxiv.org/abs/2412.15109v1), while DreamGen uses inverse or latent action models to recover supervision from synthetic videos for subsequent policy training [7](https://arxiv.org/abs/2505.12705v2). These methods establish several roles for inverse dynamics beyond direct behavioral cloning. Morin et al. analyze why inverse-model learning can require fewer action labels than policy learning, relating this advantage to differences in functional complexity and stochasticity under their semi-supervised setting [8](https://arxiv.org/abs/2602.02762). Their analysis supports studying action-label efficiency, but does not determine which adaptation strategy will work best under task shift.

FADA is particularly relevant to adaptation of the inverse model itself. It freezes a source-trained planner and fits LoRA updates to the inverse-dynamics module using a small amount of target-domain experience [9](https://arxiv.org/abs/2606.28476v1). Its experiments address changes in humanoid deployment dynamics using proprioceptive histories and future observations. Our setting concerns transfer across manipulation tasks from recorded RGB transitions, with the visual encoder as an explicit adaptation target. The comparison therefore asks whether updating that representation improves prediction beyond simpler uses of the same support data; it does not introduce inverse-model adaptation as a new mechanism.

## Retrieval as prediction and as training-data selection

Retrieval provides a way to incorporate demonstrations without fitting a new action predictor. VINN decouples visual representation learning from behavior learning and predicts actions through locally weighted nearest-neighbor regression [10](https://arxiv.org/abs/2112.01511). REGENT builds on a strong retrieve-and-play baseline, combining retrieved demonstrations with a learned transformer policy to adapt to new environments without mandatory test-time fine-tuning [11](https://arxiv.org/abs/2412.04759). RT-Cache retrieves and replays action snippets from a visual memory [12](https://arxiv.org/abs/2505.09040v3). Closest to transition-based retrieval, CLAW learns continuous latent actions from consecutive observations and maps planned latent actions to executable controls through nearest-neighbor lookup in a labeled reference set [13](https://arxiv.org/abs/2606.04130v1). These results make retrieval a substantive alternative to optimization. Our retrieval conditions use demonstrated image transitions to predict individual actions, with current-frame retrieval included to assess the value of the second observation.

A separate line of work uses retrieval to construct the data on which a policy is trained. Behavior Retrieval selects relevant state-action examples from a larger offline collection [14](https://arxiv.org/abs/2304.08742). FlowRetrieval uses motion similarity in optical-flow representations and an auxiliary flow-prediction objective [15](https://proceedings.mlr.press/v270/lin25a.html), whereas STRAP retrieves sub-trajectories through visual features and temporal alignment [16](https://arxiv.org/abs/2412.15182). Importance Weighted Retrieval further accounts for the relative density of target and prior data when selecting examples [17](https://arxiv.org/abs/2509.01657v1). HAND combines visual filtering and hand-path matching with LoRA fine-tuning of a pretrained policy [18](https://arxiv.org/abs/2505.20455v4). Retrieval can also enter the policy directly: RAEA integrates retrieved experience through cross-attention [19](https://arxiv.org/abs/2404.11699v1). Together, these methods show that retrieval and parameter adaptation are compatible. Our initial comparison isolates their effects by evaluating direct support-transition retrieval and supervised adaptation as separate conditions, without adding retrieved training data to the adaptation conditions.

## Parameter-efficient robot adaptation

LoRA represents a weight update with trainable low-rank factors while keeping the original weight matrix fixed [20](https://arxiv.org/abs/2106.09685). TAIL evaluates this approach alongside other adapters and conventional fine-tuning in continual robot imitation, demonstrating that lightweight task-specific parameters can support effective adaptation on LIBERO [21](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e2b525da536030ec1589dddea01feb-Abstract-Conference.html). OpenVLA likewise studies the performance and memory tradeoffs of efficient fine-tuning [22](https://arxiv.org/abs/2406.09246v3). OpenVLA-OFT shows that decoding, action chunking, action representation, and the training objective also affect adaptation outcomes and inference cost [23](https://arxiv.org/abs/2502.19645v2). Consequently, the number of trainable parameters alone does not establish the practical efficiency of an adaptation method.

Recent work also learns how to obtain or reuse adapters. WIZARD generates task-specific LoRA parameters from language and demonstration video, requiring neither target action labels nor gradient updates at deployment; its evaluation includes both held-out LIBERO suites and held-out tasks within suites [24](https://arxiv.org/abs/2606.07217v1). VLA-Pro retrieves and combines task-specific LoRA memories according to the current multimodal context [25](https://arxiv.org/abs/2605.29562v1). FOCA addresses limited-data adaptation by adding explicit future-latent prediction and implicit alignment with future observations to VLA training [26](https://arxiv.org/abs/2606.20867v1). These approaches use different forms of task evidence and prior training. Our comparison fixes the labeled target support demonstrations and measures the incremental effect of LoRA updates to the visual encoder over updating the same action head alone.

## Position of the present study

LIBERO was designed to study transfer across manipulation tasks with variations in spatial relationships, objects, and goals [27](https://arxiv.org/abs/2306.03310). We use its Goal suite to investigate a restricted question: with the same small support set, when does visual encoder adaptation improve inverse-dynamics prediction relative to head-only fine-tuning and retrieval, and what computational cost accompanies that improvement? The planned comparison shares support and query demonstrations, source initialization for the learned models, and action normalization, while measuring adaptation time, inference latency, memory, and source-task retention. Because the next RGB frame is observed rather than predicted, the evaluation isolates action inference from planning and execution. Its conclusions concern held-out recorded transitions; establishing benefits for autonomous task completion would require a separate control evaluation.


## References

1. He et al. (2022). [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377). IEEE/CVF Conference on Computer Vision and Pattern Recognition.

2. Nair et al. (2022). [R3M: A Universal Visual Representation for Robot Manipulation](https://arxiv.org/abs/2203.12601v3). arXiv preprint.

3. Majumdar et al. (2023). [Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?](https://arxiv.org/abs/2303.18240v2). arXiv preprint.

4. Torabi et al. (2018). [Behavioral Cloning from Observation](https://arxiv.org/abs/1805.01954). International Joint Conference on Artificial Intelligence.

5. Du et al. (2023). [Learning Universal Policies via Text-Guided Video Generation](https://arxiv.org/abs/2302.00111). arXiv preprint.

6. Tian et al. (2024). [Predictive Inverse Dynamics Models are Scalable Learners for Robotic Manipulation](https://arxiv.org/abs/2412.15109v1). arXiv preprint.

7. Jang et al. (2025). [DreamGen: Unlocking Generalization in Robot Learning through Video World Models](https://arxiv.org/abs/2505.12705v2). arXiv preprint.

8. Morin et al. (2026). [On the Sample Efficiency of Inverse Dynamics Models for Semi-Supervised Imitation Learning](https://arxiv.org/abs/2602.02762). Proceedings of the 43rd International Conference on Machine Learning.

9. Xie et al. (2026). [FADA: Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control](https://arxiv.org/abs/2606.28476v1). arXiv preprint.

10. Pari et al. (2022). [The Surprising Effectiveness of Representation Learning for Visual Imitation](https://arxiv.org/abs/2112.01511). Robotics: Science and Systems.

11. Sridhar et al. (2025). [REGENT: A Retrieval-Augmented Generalist Agent That Can Act In-Context in New Environments](https://arxiv.org/abs/2412.04759). International Conference on Learning Representations.

12. Kwon et al. (2025). [RT-Cache: Training-Free Retrieval for Real-Time Manipulation](https://arxiv.org/abs/2505.09040v3). arXiv preprint.

13. Ayalew et al. (2026). [CLAW: Learning Continuous Latent Action World Models via Adversarial Latent Regularization](https://arxiv.org/abs/2606.04130v1). arXiv preprint.

14. Du et al. (2023). [Behavior Retrieval: Few-Shot Imitation Learning by Querying Unlabeled Datasets](https://arxiv.org/abs/2304.08742). arXiv preprint.

15. Lin et al. (2025). [FlowRetrieval: Flow-Guided Data Retrieval for Few-Shot Imitation Learning](https://proceedings.mlr.press/v270/lin25a.html). Proceedings of The 8th Conference on Robot Learning.

16. Memmel et al. (2025). [STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning](https://arxiv.org/abs/2412.15182). International Conference on Learning Representations.

17. Xie et al. (2025). [Data Retrieval with Importance Weights for Few-Shot Imitation Learning](https://arxiv.org/abs/2509.01657v1). Proceedings of The 9th Conference on Robot Learning.

18. Hong et al. (2025). [HAND Me the Data: Fast Robot Adaptation via Hand Path Retrieval](https://arxiv.org/abs/2505.20455v4). arXiv preprint.

19. Zhu et al. (2024). [Retrieval-Augmented Embodied Agents](https://arxiv.org/abs/2404.11699v1). arXiv preprint.

20. Hu et al. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). International Conference on Learning Representations.

21. Liu et al. (2024). [TAIL: Task-specific Adapters for Imitation Learning with Large Pretrained Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46e2b525da536030ec1589dddea01feb-Abstract-Conference.html). International Conference on Learning Representations.

22. Kim et al. (2024). [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246v3). arXiv preprint.

23. Kim et al. (2025). [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645v2). arXiv preprint.

24. Bianchi et al. (2026). [Robotic Policy Adaptation via Weight-Space Meta-Learning](https://arxiv.org/abs/2606.07217v1). arXiv preprint.

25. Si et al. (2026). [VLA-Pro: Cross-Task Procedural Memory Transfer for Vision-Language-Action Models](https://arxiv.org/abs/2605.29562v1). arXiv preprint.

26. Nguyen et al. (2026). [FOCA: Future-Oriented Conditioning for Data-Efficient Vision-Language-Action Adaptation](https://arxiv.org/abs/2606.20867v1). arXiv preprint.

27. Liu et al. (2023). [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310). arXiv preprint.
