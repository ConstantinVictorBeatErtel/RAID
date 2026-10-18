# POST HOC: why RAID v7 fails without training on the wine-bottle task

POST HOC ANALYSIS. It was not in the frozen plan and was done after the final results were known. It changes no frozen result and selects nothing. It uses only the collected per-demo errors (6 final + 6 development settings, 20 query demos each). No model was run and no dataset was opened.
Not available: gate values, attention weights and retrieval distances were not saved in the run, and the pod storage is deleted. The statements below about the gate are therefore inferences from errors, not direct measurements.

## Finding 1: the failure is specific to one task, and retrieval itself is not the problem

| Task | Frozen source | RAID v6 | RAID v7 | k-NN | Retrieve-and-play | REGENT blend | R2 RAID v7 | R2 k-NN |
|---|---|---|---|---|---|---|---|---|
| Wine bottle on cabinet (final) | 1.453 | 0.692 | 0.682 | 0.406 | 0.501 | 0.488 | -0.51 | 0.31 |
| Bowl on cabinet (final) | 0.658 | 0.379 | 0.317 | 0.346 | 0.395 | 0.382 | 0.43 | 0.29 |
| Drawer + bowl (development) | 0.882 | 0.491 | 0.460 | 0.487 | 0.537 | 0.509 | 0.36 | 0.30 |
| Cream cheese in bowl (development) | 0.630 | 0.357 | 0.358 | 0.366 | 0.414 | 0.395 | 0.38 | 0.28 |

On the wine-bottle task, plain k-NN over the same support bank still works (0.406, similar to the other tasks). The two learned RAID heads both fail (0.692 and 0.682), and RAID v7 has a negative R2 (worse than predicting the mean action). The frozen source controller is much worse on this task (1.453) than on any other held-out task (0.63 to 0.88).

## Finding 2: on this task, RAID v7 errors follow the source controller, not retrieval

Correlation over the query demos of one task (3 seeds x 20 demos).

| Task | Demos | Pearson RAID v7 vs frozen | Pearson RAID v7 vs k-NN | Spearman RAID v7 vs frozen | Spearman RAID v7 vs k-NN | Pearson RAID v6 vs frozen | Demos where RAID v7 is worse than k-NN |
|---|---|---|---|---|---|---|---|
| Wine bottle on cabinet (final) | 60 | 0.97 | 0.31 | 0.95 | 0.26 | 0.93 | 92% |
| Bowl on cabinet (final) | 60 | 0.70 | 0.77 | 0.71 | 0.77 | 0.63 | 38% |
| Drawer + bowl (development) | 60 | 0.78 | 0.91 | 0.50 | 0.80 | 0.81 | 38% |
| Cream cheese in bowl (development) | 60 | 0.62 | 0.84 | 0.60 | 0.77 | 0.51 | 50% |

On the wine-bottle task, the demos where the source controller is bad are exactly the demos where RAID v7 is bad (r = 0.97), and the k-NN error hardly predicts it (r = 0.31). On the other three tasks the RAID v7 error follows the k-NN error more closely. Inference: on the wine-bottle task the RAID heads give too much weight to their source-trained direct path, and the retrieved support actions do not correct it enough.

## Finding 3: the failure appears only when the source controller is far outside its usual range

| Setting group | Frozen source error (range) | RAID v7 minus k-NN (range) |
|---|---|---|
| 9 other settings (3 tasks) | 0.56 to 0.96 | -0.074 to +0.016 |
| 3 wine-bottle settings | 1.38 to 1.52 | +0.208 to +0.363 |

Within the 9 other settings, a larger source error does not make RAID v7 worse than k-NN. The pattern looks like a threshold, not a smooth trend, and it rests on one task. One task cannot establish a rule.

## Finding 4: a few target updates repair it

| Task | Head-only, 25 updates | LoRA, 25 updates | RAID+LoRA v7, 25 updates |
|---|---|---|---|
| Wine bottle on cabinet (final) | 0.357 | 0.348 | 0.287 |
| Bowl on cabinet (final) | 0.283 | 0.283 | 0.256 |
| Drawer + bowl (development) | 0.416 | 0.408 | 0.352 |
| Cream cheese in bowl (development) | 0.321 | 0.317 | 0.264 |

After 25 updates, RAID+LoRA v7 is the best of these methods on every task, and its largest gain over LoRA is on the wine-bottle task. Training on five support demos quickly corrects the direct path, and the retrieved candidates then help.

## What this means

- The zero-training RAID heads depend on the source-trained controller. When that controller is very wrong for a new task, the learned gate does not fall back to retrieval. k-NN has no such dependence, so it is more robust without training.
- The cause of the poor source controller on this task is not tested. One hypothesis: the source set contains 'put the wine bottle on the rack', so the controller may map wine-bottle images to rack motions. Testing this needs the task data, which this analysis does not open.
- Possible design change for future work: make the gate depend on a retrieval-confidence or disagreement signal (for example the distance between the direct prediction and the retrieved actions), so that it can fall back to retrieval. Design it on development tasks only and test it once on new held-out tasks with a new frozen plan. The current final tasks are used.
- To measure the gate directly, a new diagnostic run must log gate values, attention weights and retrieval distances. That run reads the final tasks again, with no tuning. It needs your approval (about $1 to $2 on Runpod).

## Limits

Post hoc; 4 tasks only (one failure case); errors are correlated within task and seed; the frozen v6 controller is a proxy for the v7 direct path (same source data, different head). Development tasks were used for design.
