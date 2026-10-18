# VPP2 v5 pilot: results (development, 40 episodes)

- Run: pod kyeyj9c2t6i4tv (A100-SXM4-80GB), 16 Oct 05:56 UTC to 17 Oct about 00:35 UTC. 1144 files collected and SHA-256 verified; pod and volume deleted; Runpod empty.
- Session cost (balance difference, incl. safety checks): about $33.89. Balance after: $30.47.
- Setup: VPP2 v5 judged prompt (2 prompts + judge), 1 video per plan, 2 chunks (16 steps) per video, frames 9 and 18, 300 steps max, 5 support demos, seed 42, one shared first plan per case. User waiver of the 15-case human panel gate recorded.

## Task success (registered primary outcome)

| Condition | Successes | Plans | VPP2 hours |
|---|---|---|---|
| source_frozen | 0 / 10 | 190 | 4.3 |
| raid_bank_mlp | 0 / 10 | 190 | 4.4 |
| lora_mlp | 1 / 10 | 180 | 4.3 |
| raid_lora_mlp | 2 / 10 | 171 | 4.1 |

| Case | source_frozen | raid_bank_mlp | lora_mlp | raid_lora_mlp |
|---|---|---|---|---|
| drawer, start 14 | timeout | timeout | timeout | timeout |
| drawer, start 17 | timeout | timeout | timeout | timeout |
| drawer, start 35 | timeout | timeout | timeout | timeout |
| drawer, start 36 | timeout | timeout | timeout | success (step 193) |
| drawer, start 44 | timeout | timeout | timeout | timeout |
| cream cheese, start 17 | timeout | timeout | timeout | timeout |
| cream cheese, start 18 | timeout | timeout | timeout | success (step 84) |
| cream cheese, start 35 | timeout | timeout | timeout | timeout |
| cream cheese, start 43 | timeout | timeout | success (step 133) | timeout |
| cream cheese, start 44 | timeout | timeout | timeout | timeout |

## Registered contrasts (descriptive)

- Macro success: frozen 0.0, RAID 0.0, LoRA 0.1, RAID+LoRA 0.2.
- Hybrid over RAID +0.2; hybrid over LoRA +0.1; interaction +0.1.
- Paired hybrid minus LoRA: drawer [0.0, 0.0, 0.0, 1.0, 0.0], cream cheese [1.0, 0.0, 0.0, -1.0, 0.0].
- 3 successes in 40 episodes: no difference is statistically meaningful; the direction favours RAID+LoRA.
- Offline, LoRA alone has the lowest action error; in closed loop RAID+LoRA has the most successes. Both are small samples.

## Offline recorded-transition error (chunk MSE, lower is better)

| Condition | drawer | cream cheese |
|---|---|---|
| source_frozen | 0.815 | 0.565 |
| raid_bank_mlp | 0.476 | 0.357 |
| lora_mlp | 0.341 | 0.225 |
| raid_lora_mlp | 0.420 | 0.285 |
| support_knn_average | 0.541 | 0.410 |

## Limits

Development tasks only (final tasks sealed; held_out_claims_eligible = false). One seed, 5 starts per task. VPP2 timing evidence is weak (v3 15-case calibration). The human panel gate was waived by the user after seeing results.
