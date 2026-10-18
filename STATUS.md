## Few-shot inverse-dynamics benchmark: development, frozen plan and one final-task run — 14 to 18 October 2026

**Closed-loop pilot (14–17 Oct).** 40 VPP2 v5 episodes on the development tasks. Success: frozen 0/10, RAID 0/10, LoRA 1/10, RAID+LoRA (v6) 2/10. Cost about $33.89. [Pilot results](artifacts/raid_hybrid_libero_goal_v6/vpp2_pilot_v5_20261014/PILOT_RESULTS.md).

**Offline benchmarks on development tasks (16–17 Oct).** The v7 head (cross-attention and gate over 16 retrieved candidates) and 9 re-implemented baselines (VINN k-NN, REGENT retrieve-and-play and blend, head-only, frozen probe, LoRA without source training, full fine-tuning, STRAP-style, LoRA). Metric: chunk MSE on normalized actions against adaptation GPU-seconds. [v7 benchmark](artifacts/raid_hybrid_libero_goal_v6/compute_bench_v7_20261016/BENCH_RESULTS.md) · [Baselines](artifacts/raid_hybrid_libero_goal_v6/compute_bench_baselines_20261017/BASELINE_RESULTS.md).

**Step 1, exploratory (17 Oct, $2.23).** RAID v7 head-only: frozen encoder, source-trained v7 retrieval head fine-tuned with the head-only schedule. 25 updates: 0.312, better than all baselines in 6/6 settings. 500 updates: 0.300, worse than head-only (0.288). The source controllers are bitwise identical across pods, so the method pairs with the baseline run. [Step 1 report](artifacts/raid_hybrid_libero_goal_v6/compute_bench_v7head_20261017/V7HEAD_RESULTS.md).

**Step 2, frozen plan (17 Oct 20:15 UTC).** Method set, settings, metrics, P1/P2/P3 comparisons, cluster bootstrap over task x seed and the decision rule were frozen before any final-task file was read. Amendment digest `1b42ba59…`, file SHA-256 `acd29d3b…` (read-only). [Plan](artifacts/raid_hybrid_libero_goal_v6/compute_bench_final_20261017/FINAL_ANALYSIS_PLAN.md).

**Step 3, one final-task run (verified 18 Oct 00:03 UTC, about $6).** The 2 sealed final tasks x seeds 42/43/44, offline with real recorded futures (no VPP2). Controlled access in `hybrid_study.final_access`; the package held source and final tasks only. Results:
- P1 failed. Without training, RAID v7 (0.500) is worse than VINN k-NN (0.376). RAID v7 wins on the bowl task (3/3 seeds) and fails on the wine-bottle task (0.62–0.72 against 0.35–0.46).
- P2 at 25 updates held. RAID+LoRA v7 (0.272) is better than every baseline in 6/6 settings.
- P2 at 500 updates: no clear difference against head-only, LoRA, LoRA without source training and the probe. Worse than full fine-tuning (+0.013). Better than STRAP-style and the training-free baselines.
- P3: RAID v7 head-only reaches LoRA's 500-update error after about 0.1 GPU-seconds; LoRA needs about 190 s.

[Final results](artifacts/raid_hybrid_libero_goal_v6/compute_bench_final_20261017/FINAL_RESULTS.md). [Post hoc wine-bottle diagnosis](artifacts/raid_hybrid_libero_goal_v6/posthoc_wine_bottle_20261018/POSTHOC_WINE_BOTTLE.md): on that task RAID v7 errors follow the source controller (per-demo r = 0.97), not k-NN (r = 0.31); gate values were not collected. The final stage is used and must not run again. Runpod: 0 pods, 0 volumes; balance $6.96. Closed-loop success on the final tasks (optional step 4) is not run. `held_out_claims_eligible=false` (VPP2 saw LIBERO).

## VPP2 selected for next phase — 13 October 2026

The user selected the tested VPP2 standalone Stage-1 video generator after reviewing the four examples. This authorizes a versioned generator switch; production integration and VPP2-specific source/timing/quality/funding admission remain pending. No new run was launched for the handoff. The downloaded weights were on the now-deleted Runpod volume; only code, pinned download manifests, prompts, diagnostics and videos are retained, so the next cloud run must download weights again.

[Continuation handoff](artifacts/raid_hybrid_libero_goal_v6/explicit_prompts_vpp2_20261013/NEXT_PHASE_HANDOFF.md). Preserve the original experiment design and sealed final tasks. `held_out_claims_eligible=false`.

## Explicit prompts and VPP2 comparison — 13 October 2026, complete

Four source-only image-grounded prompts were executed with both VPP2 Stage-1 and Cosmos: eight clips generated, decoded and checksum-collected (47 manifested files). Prompt, case and seed identities passed the host audit. Source session `v6-source-20261013T111047-0f6707a9` and live acceptance are closed; compute and storage deletion are confirmed. Provider inventory is empty; current spend is zero.

The inspection page pairs each initial image, exact prompt and model output. VPP2 shows more task-directed motion in sampled frames; Cosmos still moves the wrong object in the bowl case and invents an additional burner in the stove case. These are advisory observations, not human ratings or robot success. VPP2 uses the standalone video checkpoint, not a replacement controller.

Observed account credit decreased by $2.0642715751 including acceptance and failed attempts, leaving $23.893317609; final billing may lag. Deployment `7b798f5d-d06d-444d-a138-54ed381fc411` supports explicitly priced 80/160 GB source volumes; 25 service tests and 51 focused Python checks passed. Prior registration/setup and benign Cosmos text-filter failures remain retained; runtime safety checks stayed enabled.

[Eight clips, prompts and results](artifacts/raid_hybrid_libero_goal_v6/explicit_prompts_vpp2_20261013/RESULTS.md). Scientific admission and human ratings remain false; the 40-episode pilot is unlaunched and `held_out_claims_eligible=false`.

## Revised Cosmos four-case test — 2026-10-13

The revised configuration was executed on four existing source-validation cases using their original images, instructions and seeds. All four produced 121-frame 480×480 clips; all 36 manifested files are checksum-collected. Exact-release independent acceptance passed, and both acceptance/test pods and network volumes are confirmed deleted. Provider inventory is empty and current spend is zero.

Agent advisory inspection of sampled frames still finds weak task motion and implausible interaction, including a bowl appearing near the gripper without credible pickup. This supplies no human ratings. Generation took 36–40 seconds per clip, about 3.1 times the matched historical latency. Account credit decreased by $0.4234650528 including acceptance; final resource billing is pending posting. Remaining credit is $26.0251283702.

[Four-case results and videos](artifacts/raid_hybrid_libero_goal_v6/cosmos_four_case_test_20261013/RESULTS.md). Recommendation: stop expanding this Cosmos configuration and investigate a LIBERO-specific generator. The pilot remains unlaunched, new timing and full-panel gates remain unsatisfied, and `held_out_claims_eligible=false`.

## Cosmos configuration audit — 2026-10-13

The collected source panel has 30 technical decodes and 27 expert replay passes; these do not establish generated-video quality or supply human ratings. The source session is closed with compute and storage deletion recorded. The 40-episode pilot has not started.

The exact pinned NVIDIA example confirms that 20 steps, guidance 6 and flow shift 12 were already correct. A separate source diagnostic profile now uses structured JSON prompts, square 480-pixel output, 121 frames and 24 nominal FPS. Original pins, scope and historical evidence are retained. Local structural tests pass; GPU quality, new timing calibration and the changed configuration's cost remain unmeasured. The old planning estimate cannot admit this profile. No paid resources were launched for this audit. `scientific_admission=false`; `held_out_claims_eligible=false`.

[Configuration audit](artifacts/raid_hybrid_libero_goal_v6/cosmos_configuration_audit_20261013/CONFIGURATION_AUDIT.md) · [LIBERO model comparison](artifacts/raid_hybrid_libero_goal_v6/cosmos_configuration_audit_20261013/LIBERO_WORLD_MODEL_OPTIONS.md)

Earlier entries below are historical.

## Hugging Face launcher repair — 2026-10-12T11:46:44.710446+00:00

Saved model access verified for Cosmos3-Edge, Cosmos-1.0-Guardrail and Qwen3Guard. Launcher checks access before spending and supplies the token over SSH stdin to a mode-0600 file on disposable container storage. Credentials remain outside packages, journals and result collection. Verification: 122 regression tests plus 9 focused credential/package tests passed.

Reused the exact previously verified source archive because a local LIBERO checkout read stalled. Full archive hash and embedded manifest verified; new host bindings include the credential helper. New bounded operational acceptance is running on pod `suaf4ivp702ihm`, volume `4byr5514j4`. Driver will start source session `v6-source-20261012T114315-c09fd52e` only after independent acceptance passes and fresh funding checks pass. Reservations: acceptance $1, source at most $5.50, within the existing cumulative allowance; no top-up. Evidence: `artifacts/raid_hybrid_libero_goal_v6/hf_launcher_fix_20261012/`. Scientific readiness and held-out eligibility remain false.

## Diagnostic reset — 2026-10-12T11:02:14.363529+00:00

The latest source run reached CUDA and simulator validation, then failed with `generation_failed:GatedRepoError` while fetching a generator dependency. No panel or pilot episodes were produced. All 14 collected files passed SHA-256 verification. The independently managed service confirms pod `za07u2b7c5guyb` and volume `kd3v5tbdkq` deleted; the session is closed.

Following the user's concern about repeated work and consumed usage, additional development and paid retries are stopped for this diagnostic reset. The next bounded step is local access validation for every pinned model dependency before any further GPU allocation. Prior readiness entries below are historical. The $100 cumulative source allowance and five-day window do not waive access, evidence, prepaid-credit, or cleanup gates.

Updated 2026-10-11T21:44:15.822492+00:00. **Independent acceptance passed and the repaired source Runpod allocation is running.** Pod `za07u2b7c5guyb`, volume `kd3v5tbdkq`, NVIDIA A100 80GB PCIe at $1.79/hour in EU-RO-1. The exact 787-file source release passed 108 Python tests; the deployed service passed 23 tests, type checking and a dry run. Live independent alarm termination, direct S3 sentinel recovery/checksum and storage deletion passed for this release and deployment.

The source reservation is $5.50, within refreshed prepaid credit and the existing $100 cumulative/five-day allowance. Workload cutoff 2026-10-12T00:16:56.262000+00:00, compute deadline 2026-10-12T00:31:56.262000+00:00, storage expiry 2026-10-12T00:41:56.262000+00:00. Package transfer is in progress; detached archive verification, extraction and complete local-runtime bootstrap follow. No CUDA/source scientific result or pilot is established yet. Earlier failed resources are deleted and their checksummed diagnostics/costs retained; the older uncertain volume request remains independently reconciling and reserved.

A supplementary Mac observer requests compute termination only after a closed, release-bound DONE/FAILED manifest, so large downloads need not keep the GPU rented. Independent alarms remain active. Newer local pilot proof repairs and their separate tests are outside this frozen source package. The pilot remains blocked on source/CUDA/timing evidence, 30/30 technical checks, >=27/30 expert replays, >=24/30 named human ratings, measured funded costs and managed pilot lifecycle. `launch_ready=false`; `held_out_claims_eligible=false`.

Earlier entries below are historical.

---

Updated 11 October 2026, 23:16 Berlin time. **Source setup failed before model loading; verified diagnostics are collected and compute/storage are deleted.** The failure was a wheel executable path after relocating only the virtualenv library directory. The repaired bootstrap uses a complete physical environment on local disk, both unchanged hash-locked dependency sets, dependency checks and disk reserves. Archive verification/extraction now run under the detached worker and independent deadline. Fresh release tests/acceptance are required before a bounded retry. The last clean local suites passed 108 Python and 23 service tests. No CUDA evidence, source panel or scientific result exists yet.

The authorized source allowance remains **$100 cumulative over five days**, with all earlier reservations retained. The refreshed prepaid balance is $9.2967812928; conservatively usable credit is $6.6020351830593311074 after outstanding-charge reserves. No automatic top-up. The pilot remains blocked on actual source/CUDA/timing evidence, named human ratings, a measured funded forecast and managed pilot lifecycle. `launch_ready=false`; `held_out_claims_eligible=false`. Older uncertain creation reconciliation remains active.

All entries below are historical.

---

Updated 11 October 2026, 22:20 Berlin time. **The source worker is running on Runpod.** Pod `cfr0ysj6tg1sxr`, worker PID `2127`, attached volume `cmawjxfrr1`, A100 at $1.79/hour in provider-verified `EU-RO-1`. Independent cutoff/recovery acceptance passed on deployment `b99173e6-4193-44b0-8fad-92a50783c963`, including compute deletion, S3 sentinel recovery/checksum, collection and storage deletion.

The frozen 786-file source archive and extracted input hashes passed real pod preflight. A local SSH timeout and subsequent transient SSH-discovery error were reconciled on the same resources; no replacement was created. Closed setup diagnostics were published and checksum-collected before worker startup. Runtime libraries use local container disk through the unchanged virtualenv path; pinned locks and registered stage commands remain unchanged. Locked bootstrap is running; CUDA evidence, source-panel results and scientific success are not yet established.

The authorized source allowance is **$100 cumulative over five days**, including prior spending/reservations. Prepaid credit and the $225 study spending limit remain separate constraints, with no top-up. This resource reservation is $9.50; its worst-case allocation forecast is $7.332222222222222222222222222. Independent workload cutoff: **12 October, 01:14:50.840 Berlin**; compute deadline 01:29:50.840; storage expiry 01:39:50.840. The shared allocation expires 16 October, 21:25:37.224 Berlin. The five-day window does not fund continuous rental or extend this resource's deadlines. Older uncertain creation reconciliation remains active and reserved.

Local future-release repairs cover optimizer/RNG/sampler/best-state training recovery, atomic versioned restart publication, cache containment/provenance, worker/orphan exclusion, retained interruption diagnostics, pre-command diagnostic publication, and failure publication after closed manifest verification. The latest combined suite passed 76 Python tests; 22 service tests passed separately. A synthetic CLI run produced exactly 40 condition episodes across ten matched cases, with duplicate/missing-case rejection. These newer local changes are separate from the frozen live source package and do not constitute scientific evidence or matching release acceptance.

The pilot remains blocked pending real CUDA/source evidence, 30/30 technical cases, at least 27/30 matched expert replays, at least 24/30 named human passes, valid source-only frame/action timing evidence, a fresh measured funded forecast, and completed managed pilot lifecycle/admission. Human ratings must be supplied by people. `launch_ready=false`; `scientific_result=false`; `held_out_claims_eligible=false`. [Current evidence](artifacts/raid_hybrid_libero_goal_v6/source_cap_increase_20261011T1923Z/current_progress.json).

All earlier entries below are historical observations.

---

Updated 11 October 2026, 18:44 UTC. **Source Runpod attempt failed before model loading; the missing import is fixed locally.** Live operational acceptance passed for the previous release. Source pod `pzbk90s0roe7ra` ran bootstrap/resolve, then `gpu_validate` failed because the source allowlist omitted unchanged `retrieval.py`. All 10 committed diagnostic files were checksum-verified; compute and volume deletion are confirmed. The repaired 786-file bundle and staging path pass 34 focused Python tests; the deployed independent recovery repair passes 21 service tests. No source scientific result or pilot exists.

The authorized retry window expired at **18:28:22.956 UTC**. The cumulative $10 allowance remains fully reserved, including the historical uncertain volume request. No further paid creation occurred. The new package/service require fresh matching live acceptance before source execution; pilot dispatch and interrupted-worker resume remain incomplete, followed by CUDA, source panel, timing, human-rating and measured funding gates. `launch_ready=false`; `held_out_claims_eligible=false`. Current inventory has zero pods/volumes; the older uncertain creation remains under independent reconciliation.

[Verified failure, repair and remaining gates](artifacts/raid_hybrid_libero_goal_v6/source_retry_session_20261011T1625Z/FAILURE_AND_REMEDY.md). Runpod posted charges increased by $1.7250464970711618730 since the retry snapshot; remaining prepaid balance is $12.1069637401. Account-level storage attribution and service contingencies remain explicit in the [cost report](artifacts/raid_hybrid_libero_goal_v6/source_retry_session_20261011T1625Z/final_cost_report.json).

All earlier status entries below are historical observations.

---

Updated 11 October 2026, 16:37 UTC. **Independent Runpod acceptance passed and source feasibility has started.** New acceptance `jx0ph3ivcutu5m` was terminated by the Cloudflare alarm, its S3 sentinel recovered and checksum-verified, and compute/storage deletion verified. Source pod `pzbk90s0roe7ra` and attached volume `z8t9ptb8l6` are created; SSH/setup is in progress. Source compute cutoff is `2026-10-11T18:13:21.956000+00:00`, storage expiry `2026-10-11T18:28:21.956000+00:00`. The $10 cumulative cap is fully reserved ($2 historical acceptance, $1 historical uncertain creation, $1 new acceptance, $6 source); no top-ups. Old uncertain volume reconciliation remains active under the explicit retry authorization. No CUDA/source scientific result or pilot is yet complete; launch readiness remains false and held_out_claims_eligible=false.

# RAID implementation and launch status

## Authorized retry — 2026-10-11T16:31:00.734132+00:00

The user explicitly authorized resetting approval and trying again. A new two-hour source window is registered, expiring 2026-10-11T18:28:22.956000+00:00. The cumulative $10 source cap and all prior reservations remain; no top-ups or scientific gate bypass. Historical uncertain volume creation remains unresolved with independent cleanup active.

New acceptance pod `jx0ph3ivcutu5m` (A100 80 GB, US-MD-1, $1.79/hour) and volume `40vdu69i7r` are owned and verified. Sentinel is committed; independent cutoff/recovery/collection remains pending. Source feasibility and pilot have not started. Source allowance reserves $2 earlier acceptance, $1 earlier uncertain volume, $1 current acceptance; up to $6 remains for source.

Previous observations below are historical.

## Current Runpod repair — 2026-10-11T16:09:59.835121+00:00

The placement bug is repaired and deployed as `93eb42b8-0878-4d75-b32c-7d2da207d9b6`. The selector and independent service require live standard network-volume support; US-WA-1 is rejected. Fresh read-only placement and deployed guard checks passed; 69 Python and 18 service tests passed, with TypeScript/whitespace checks passed. [Repair evidence](artifacts/raid_hybrid_libero_goal_v6/provider_repair_20261011T1603Z/REPAIR.md).

The original $10/two-hour source allowance **expired at 16:06:38.430 UTC**; deployment preserved it. No further paid creation is authorized. The original HTTP 500 volume request remains uncertain; independent cleanup/reconciliation stays active and its reservation is not cleared. Zero current pods/volumes and $0/hour spend do not prove rejection. No CUDA/source panel or pilot has run; `launch_ready=false`, `held_out_claims_eligible=false`.

The following status is retained from the earlier observation; its dates and deployment IDs are historical.

Updated 11 October 2026, 15:19 UTC. **Active protocol: v6 (`raid_hybrid_generalization_v6`). Source launch is blocked.** The current exact-release acceptance failed: Runpod returned HTTP 500 while creating its attached network volume. No new pod was requested. Current inventory shows zero pods/volumes and $0/hour spend, but the volume request remains uncertain. Independent reconciliation/cleanup stays active; cleanup is not confirmed and the $1 reservation cannot be cleared. `launch_ready=false`; `scientific_result=false`; `held_out_claims_eligible=false`.

The earlier acceptance on pod `e3edlsz49q1qek` passed independent termination, direct S3 recovery and storage deletion. It remains historical evidence for deployment `c209545e-1a41-435b-a2c1-3850820615f3`; it cannot admit the extended current deployment `c86d349a-8a56-48f2-b5f6-b5bae78e0569` and source package.

| Area | Actual current state |
| --- | --- |
| Source pipeline | Managed source-panel dispatch, detached worker, per-command closed logs/exit records, atomic result manifests, direct S3 collection and immutable release identity implemented. Resume observes existing ownership and never creates replacements. |
| Portable package | Explicit source-only allowlist now includes missing imports and only source LIBERO task assets. Hash/size/disk/archive preflight fixed. Actual Python 3.12 relocation and both CLI imports passed with original Mac paths unavailable; scientific identity preserved. Remote EGL/CUDA/model loading remain unexecuted. |
| Independent service | SQLite Durable Object source admission checks authenticated exact-release acceptance; current source-release acceptance failed before pod creation. Persisted reconciliation/recovery/expiry jobs continue independently. |
| Accounting | Earlier test approximately $0.13223 now posted. Remaining credit $13.8397879873; retain $2 prior and $1 unresolved reservations conservatively. Shared $10/two-hour allowance began 14:06:38.430 UTC and expires 16:06:38.430 UTC (18:06:38 Berlin); cannot reset. |
| Verification | Final focused rerun: 57 Python tests and 15 service lifecycle tests passed; TypeScript and whitespace checks passed. Relocation used Python 3.12; scientific GPU validation remains unexecuted. See the hash-bound verification report in the failure evidence directory. |
| Science | No source GPU feasibility, 30-case generated panel, valid generator timing, named human ratings or measured pilot forecast. No pilot episodes. Human ratings must come from people. |
| Remaining implementation | Managed pilot dispatch, complete resumable training state and CLI-driven pilot/semantic collection integration still require completion and verification. Local source benchmark weights are temporary measurement weights. No implementation or synthetic report establishes scientific success. |
| Fixed constraints | Four conditions, five-shot support, seed 42, two dev tasks/five starts, pins, GR-1 prohibition and sealed final tasks unchanged. $250 ceiling/$25 reserve and prepaid credit remain separate. No top-up, replacement compute, deadline extension, scope reduction or final evaluation. |

See [current failure and recovery](artifacts/raid_hybrid_libero_goal_v6/source_release_acceptance_20261011T1510Z/FAILURE_AND_RECOVERY.md), [current readiness](artifacts/raid_hybrid_libero_goal_v6/readiness_status_20261011T1520Z.json), [source relocation](artifacts/raid_hybrid_libero_goal_v6/source_session_20261011T1505Z/relocation_verification.json) and [historical passed acceptance](artifacts/raid_hybrid_libero_goal_v6/operational_acceptance_20261011T1405Z/live_acceptance_result.json). Obsolete GPU prices and unmeasured historical forecasts cannot authorize launch.

## Retained earlier implementation record

Everything below is an earlier v4/v3 status snapshot. Preserve evidence and constraints; historical numerical evidence does not certify v6.

---

# RAID implementation and launch status

Updated 11 October 2026. GR-1 is prohibited. **No compute spending is authorized.**

## Active protocol v4 (`raid_world_model_idm_v4`)

v4 replaces the v3 four-condition comparison with two arms behind one frozen visual world model: RAID bank + MLP versus LoRA + MLP (see [RESEARCH.md](RESEARCH.md)). **No v4 scientific, CUDA or generator measurement exists.**

| Area | State | Evidence class |
|---|---|---|
| Protocol | `wm_protocol.py`. Digest-bound record built from the v3 manifest (same splits and support/query IDs), with explicit temporal alignment, retrieval, head and generator specs. Validation includes disjoint support/query, nested budgets and the GR-1 marker scan. Built: `artifacts/raid_wm_libero_goal_v4/protocol_v4.json`. | Implementation |
| World model | `world_model.py`. Pinned Cosmos3-Edge pipeline loading, image-to-video generation, RGB decoding, MP4 output and a content-addressed clip cache. Generation rejects any conditioning beyond frame + instruction, and fails if the pipeline returns native actions. Runtime lock `requirements-wm-cosmos.lock` (1,194 hashes) is resolved but not installed. | Implementation; fake-backend tests only |
| Heads | `wm_heads.py`. Shared chunk MLP trunk; explicit RAID retrieval block (four slots plus softmax mean). The k-NN average is a control only. | Synthetic tests |
| Training / adaptation | `wm_study.py`. Per-demo MAE frame cache with no labels for query demos; matched source training of both heads with episodic banks; target RAID bank built from support only; LoRA + head adaptation with 200/500/1,000 snapshots. | Synthetic tests (tiny encoder) |
| Evaluation | Recorded diagnostic (`evaluate_recorded`) with a query-access ledger and per-stage costs. Source retention (`evaluate_retention`): LoRA with the target adapter active, RAID on an unchanged source bank. Closed loop (`wm_rollout.py`): own-observation replanning, a shared first plan, official success predicate, failure modes, rollout and side-by-side MP4s. Development freeze gate before final-test commands. | Synthetic tests (fake simulator/generator) |
| Temporal calibration | `estimate_time_scale` (monotone dynamic-programming alignment) and the `calibrate-time-scale` command (source validation only). Rollouts refuse to run while uncalibrated. | Synthetic recovery of a known scale only |
| Feasibility | `wm_feasibility.py`. With placeholder 30 s/clip: development ≈ $12.6, full confirmation ≈ $34.5, five-shot ≈ $11.7 (25% margin). At 120 s/clip: ≈ $45 / $128 / $43. | Projection from unmeasured costs |
| Tests | `tests/test_wm_v4.py`, including a synthetic end-to-end development run (`run_synthetic_experiment`) that must pass collection and must stay `scientific_result: false`. | Synthetic structural; not scientific evidence |

Open v4 gates, in order:

1. Container image is pinned: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` at `sha256:60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f` (CUDA 12.8.1, Ubuntu 24.04). It has not been booted. The lock is installed into a venv; the image's own torch is not used.
2. Simulator packages are in `requirements-wm-sim.lock` (MuJoCo 3.1.6, robosuite 1.4.0, BDDL, gym). On this Mac, `put_the_cream_cheese_in_the_bowl` constructed and stepped once under `MUJOCO_GL=glfw`, returning a 128×128 RGB frame. The pod uses `MUJOCO_GL=egl`, which is not yet run.
3. `validate-gpu` writes `cuda_evidence.json` and `costs.json`. `scripts/pod_wm_session.py launch` refuses without `--authorize-spend`. The validation plan does not start training.
4. Still needs a GPU: BF16 load, viewable LIBERO clips, measured seconds per clip, then time-scale calibration on source validation.
5. Exposure audit of Cosmos3-Edge (unresolved) and MAE.
6. Pod transport of a v4 bundle. Local collection (`wm_collect.py`) is implemented and the synthetic experiment test passes it. Renting a pod is not part of that test.
7. Separate approval of an exact package.

v3 cloud runners (`scripts/pod_session.py launch`) refuse to start unless given `--allow-superseded-v3`.

## Earlier protocol v3 (superseded)

**The v3 experiment was not ready for paid launch and is superseded.** No CUDA calibration, scientific development or confirmation result exists. Protocol v3, format-v4 evidence, standalone MAE ViT-B/16 and fixed training settings remain unchanged and preserved. The direct-pod runner gaps were fixed (key handling, FAILED marker and exit guard, collection pull, started-at cost gate, smoke check); the full local suite passed 249 tests on 11 October.

Start with [CURSOR_HANDOFF.md](CURSOR_HANDOFF.md) for the completed work and ordered path to calibration and scientific execution. [Cloud completion checklist](docs/CLOUD_COMPLETION_CHECKLIST.md) records the remaining operational gates.

## Completed implementation and retained evidence

| Area | Current state and evidence limits |
| --- | --- |
| Inputs and real CPU smoke | Earlier accepted checkpoint passed all 500 audited shards, including 80 label-free query shards. Real CPU raw/cache maximum error was 5.245208740234375e-06; zero-adapter error was zero; frozen backbone and reload checks passed. macOS ARM/Python 3.13/Torch 2.11 is diagnostic evidence, not target CUDA evidence. |
| Shared structural lifecycle | Production-generated HDF5/cache/training/adaptation/evaluation/freeze/registration/report flows produced 24/12 canonical conditions. Both scopes, three sources, separate evidence directories, relocation and label-read instrumentation passed at the retained full checkpoint. Public scientific wrappers reject the synthetic evidence. |
| Recovery and publication | Retained coverage includes 108 subprocess signal cases, four microbatch cases, all-three-source recovery/retries, and durable publication failure/concurrency checks. Immutable dependencies precede receipts; query-access history survives failures. Actual CUDA recovery is unexecuted. |
| Reporting | Complete task/seed pairs required; invalid/incomplete evidence suppresses completed aggregates and confidence intervals. Hand calculations and an independent bootstrap reference are covered. |
| Billing and forecast | Exact decimal/raw-response reconciliation; explicit spending input; full source/freeze/registry validation; total/remaining workload and validated reuse. Timing overlap is explicitly a conservative upper bound. No measured launch-eligible CUDA forecast exists. |
| Calibration | Runtime/image/lock/hardware and full collector/rehearsal graph validation implemented. Fresh-cache smoke, five warmup/twenty measured updates, eight evaluators, retention/recovery, memory and storage remain to be established on approved CUDA hardware. |
| Cloud runner | Allowlisted bundle, clean pinned LIBERO Git restore, resumable transfers, ownership checks, immutable approved paths, worker/coordinator, recovery status and pre-cleanup rehash implemented. Local bundle/transport and fake-provider checks pass; integration and stopped-compute recovery remain open. |
| Scientific schedules | First-condition, remaining-development, source-43/44 and registered-confirmation schedules implemented. First-condition worker admission/condition validation implemented. Later scientific cloud transport and all scientific stage collection certification remain incomplete. |

## Verification bindings

The [earlier complete acceptance](artifacts/fewshot_libero_goal_v3/readiness_acceptance_20261011T014454Z_9581cd23/acceptance.json) passed **183 tests**, compilation, whitespace checks, the 500/80 audit, real CPU smoke and both structural scopes. Its effective code digest was `75cfbf0843ba3c74bf01013a078aa61a351aa4f3126193a2080a0df8beb51c02`.

Current operational changes passed [83 focused checks](artifacts/fewshot_libero_goal_v3/execution_progress_20261011T031657Z_6da47e53/verification.json), a retained TLS correction, and [three final targeted checks](artifacts/fewshot_libero_goal_v3/execution_progress_20261011T031657Z_6da47e53/verification_final.json). These reruns are not additional unique tests or a new full acceptance. Current code: `25143e56f82118863260aca5386b8f84edf5760091ef016b78586ddd0ae069dd`; configuration: `acbc84b807392675a7d012973b09e6fb86070b10cd467f1a61dd757127804ba5`; protocol: `9a2f54a8ccbdecbb7c487578e0b61aff0f4fa3748cc042712975eebbca5e597d`.

The subsequent 02:50 and 02:56 UTC full acceptance attempts were interrupted/incomplete; the latter was stopped following the user's request to prioritize progress. Preserve all records. The launch readiness validator still requires passed acceptance bound to current code. A reviewed scoped-evidence mechanism or necessary current-code gate refresh remains work to do; focused results alone do not unlock it.

## Remaining launch blockers

1. **Independent termination:** installed CLI timer flags do not establish enforcement. Retained provider/schema checks cannot read back an absolute deadline; the runner rejects that unsupported contract before resource creation. Resolve a verified provider mechanism or an explicitly approved revised operational design. See the [verified findings](artifacts/fewshot_libero_goal_v3/execution_progress_20261011T031657Z_6da47e53/provider_deadline_findings_verified.json).
2. **Integrated staging, stopped-compute collection and capped volume retention:** EU-SE-1 direct S3 recovery cannot be assumed. Component checks and written retention obligations do not complete this path.
3. **Current-code readiness and exact calibration package:** bind actual evidence, inputs, lock, billing, quote, commands, ownership, storage, cap/deadline and collection; obtain separate calibration-only approval.
4. **Before science:** complete reusable canonical evidence transport, scientific collection validators and inventory/report reproduction; then pass real CUDA/runtime/recovery/cost gates and obtain development authorization. Confirmation requires its own frozen registry, forecast and approval before final access.

## Funding and authorization

The user added **$10** and disclosed no additional RAID spending outside this account. The last read-only snapshot, approximately 04:42 Europe/Berlin on 11 October, showed **$14.2378125019** credit, **$30.7621875803451984264** returned historical charges and **0 pods / 0 volumes**. The last Secure Cloud A40 48 GB EU-SE-1 quote was **$0.59/hour**; refresh quote/billing before approval and launch.

No further top-up is requested for the initial **six-hour/$10 all-in** calibration cap. The full-study top-up remains unknown until valid CUDA measurements and reconciled spending. Preserve the $250 ceiling and $25 reserve: approximately $194.24 study allowance before calibration, or $184.24 if it spends the full $10. These are ceilings, not forecasts or authorization.

Sizing proposes 30 GB container disk and 52 GB network volume. The retained $4.5713 allocation estimate includes a 25% GPU allowance and 24-hour volume retention; it is not measured CUDA cost and the older draft is not approval-ready.

`launch_ready=false`, `calibration_launch_ready=false`, `scientific_result=false`; checked-in paid execution remains disabled. No rental authorization or resource creation has occurred. Historical evidence and unrelated workspace edits remain preserved.
