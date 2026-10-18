"""Protocol v6: retrieval x adaptation hybrid for few-shot generalization.

Pipeline: current observation + task instruction -> frozen Cosmos3-Edge
image-to-video generation -> predicted RGB future -> common inverse controller
(optionally with target retrieval and/or LoRA+head adaptation) -> robot actions.

Conditions (2 x 2)
  source_frozen   R=0 A=0  no support-dependent changes
  raid_bank_mlp   R=1 A=0  support bank; predictor weights fixed
  lora_mlp        R=0 A=1  LoRA + head adaptation; null memory
  raid_lora_mlp   R=1 A=1  LoRA + head with retrieval; bank retained at inference

v3 and v4 results are historical and never mix with v6 evidence. This module
does not edit retrieval.py or models.py (they are in the v3 code snapshot).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

PROTOCOL_V6 = "raid_hybrid_generalization_v6"
PROTOCOL_V6_VERSION = 6
PARENT_V3 = "fewshot_idm_adaptation_v3"
PARENT_V4 = "raid_world_model_idm_v4"
SUPPORT_BUDGETS = (2, 5, 10)
PRIMARY_BUDGET = 5
DEVELOPMENT_SEED = 42
CONFIRMATION_SEEDS = (42, 43, 44)
ACTION_DIM = 7
FEATURE_DIM = 768
PROPRIO_DIM = 9  # 7 joint + 2 gripper
PROHIBITED_MODEL_MARKERS = ("gr1", "gr-1", "gr_1")

COSMOS_MODEL_ID = "nvidia/Cosmos3-Edge"
COSMOS_REVISION = "344d602b128d1bbdacb43b08d0a3626f46343e29"
DIFFUSERS_COMMIT = "c60830ee365d520ab52b110dda562dd26f7b4d7f"

# R = target transition retrieval; A = LoRA PLUS head adaptation.
CONDITIONS = {
    "source_frozen": {"R": 0, "A": 0, "description": "No support-dependent changes"},
    "raid_bank_mlp": {"R": 1, "A": 0, "description": "Fill support bank; keep all predictor weights fixed"},
    "lora_mlp": {"R": 0, "A": 1, "description": "Adapt LoRA and head; always null memory"},
    "raid_lora_mlp": {"R": 1, "A": 1, "description": "Adapt LoRA and head with retrieval; retain bank at inference"},
}
CONDITION_NAMES = tuple(CONDITIONS)
DIAGNOSTIC_CONTROLS = ("head_only_ft", "support_knn_average")

STAGES = {
    0: "local_synthetic_structural_checks",
    1: "source_only_generator_runtime_feasibility",
    2: "development_recorded_and_40_episode_pilot",
    3: "narrow_final_confirmation_after_freeze",
    4: "deferred_extensions",
}


@dataclass(frozen=True)
class TemporalAlignment:
    """Explicit mapping between generated video frames and simulator steps.

    timing_basis starts as nominal_unverified. A requested frame past the clip
    raises; silent clamping is forbidden. The v4 slope estimate is a diagnostic
    only and does not unlock scientific execution.
    """
    control_hz: int = 20
    generation_fps: int = 20
    num_frames: int = 57
    chunk_steps: int = 8
    chunks_per_plan: int = 1
    time_scale: float = 1.0
    time_scale_status: str = "uncalibrated"
    timing_basis: str = "nominal_unverified"
    source_protocol_digest: str | None = None

    def __post_init__(self):
        if not 12 <= self.generation_fps <= 30:
            raise ValueError("Cosmos3-Edge video generation supports 12-30 fps")
        if not 50 <= self.num_frames <= 150 or (self.num_frames - 1) % 4:
            raise ValueError("num_frames must be 4k+1 within the 50-150 Cosmos3-Edge range")
        if self.chunk_steps < 1 or self.chunks_per_plan < 1:
            raise ValueError("chunk_steps and chunks_per_plan must be positive")
        if not (0.25 <= self.time_scale <= 4.0) or not math.isfinite(self.time_scale):
            raise ValueError("time_scale outside the plausible 0.25-4.0 range")
        if self.time_scale_status not in {"uncalibrated", "calibrated_on_source_validation"}:
            raise ValueError("unknown time_scale_status")
        if self.timing_basis not in {"nominal_unverified", "verified_on_source_validation"}:
            raise ValueError("unknown timing_basis")
        if self.source_protocol_digest is not None and not re.fullmatch(
                r"[0-9a-f]{64}", self.source_protocol_digest):
            raise ValueError("source_protocol_digest must be a SHA-256 hex digest")
        if self.frame_for_offset(self.replan_steps) > self.num_frames - 1:
            raise ValueError("generated clip is shorter than one replanning interval")

    @property
    def replan_steps(self) -> int:
        return self.chunk_steps * self.chunks_per_plan

    def frame_for_offset(self, sim_steps: int) -> int:
        seconds_sim = sim_steps / self.control_hz
        seconds_video = seconds_sim / self.time_scale
        return int(round(seconds_video * self.generation_fps))

    def require_frame(self, frame: int) -> int:
        """Return ``frame`` if it is inside the clip; never clamp silently."""
        if frame < 0 or frame > self.num_frames - 1:
            raise ValueError(f"requested frame {frame} is outside clip [0, {self.num_frames - 1}]")
        return frame

    def subgoal_frames(self) -> list[int]:
        return [self.require_frame(self.frame_for_offset((c + 1) * self.chunk_steps))
                for c in range(self.chunks_per_plan)]

    def require_calibrated(self) -> None:
        if self.time_scale_status != "calibrated_on_source_validation":
            raise RuntimeError("temporal alignment is uncalibrated; measure on source validation first")
        if self.timing_basis != "verified_on_source_validation":
            raise RuntimeError("timing_basis is not verified on source validation")


VPP2_FAMILY = "vpp2_stage1"


@dataclass(frozen=True)
class VPP2TemporalAlignment:
    """Frame/step mapping for the event-level VPP2 Stage-1 generator.

    VPP2's 49 frames cover a whole predicted event, so its display fps (6) is
    not a physical rate. Steps per generated frame and the requested subgoal
    frame stay unset until the pre-registered source-only calibration stamps
    them; every frame lookup refuses before that. Requested frames past the
    clip raise and are never clamped.
    """
    generator_family: str = VPP2_FAMILY
    control_hz: int = 20
    num_frames: int = 49
    display_fps: int = 6
    chunk_steps: int = 8
    chunks_per_plan: int = 1
    steps_per_frame: float | None = None
    requested_frame: int | None = None
    time_scale_status: str = "uncalibrated"
    timing_basis: str = "nominal_unverified"
    source_protocol_digest: str | None = None

    def __post_init__(self):
        if self.generator_family != VPP2_FAMILY:
            raise ValueError("VPP2 temporal alignment requires generator_family vpp2_stage1")
        if (self.num_frames, self.display_fps) != (49, 6):
            raise ValueError("the pinned VPP2 Stage-1 checkpoint generates 49 frames shown at 6 fps")
        if self.chunk_steps < 1 or self.chunks_per_plan < 1:
            raise ValueError("chunk_steps and chunks_per_plan must be positive")
        if self.time_scale_status not in {"uncalibrated", "calibrated_on_source_validation"}:
            raise ValueError("unknown time_scale_status")
        if self.timing_basis not in {"nominal_unverified", "verified_on_source_validation"}:
            raise ValueError("unknown timing_basis")
        calibrated = self.time_scale_status == "calibrated_on_source_validation"
        if calibrated != (self.steps_per_frame is not None) or calibrated != (self.requested_frame is not None):
            raise ValueError("steps_per_frame and requested_frame are set exactly when calibrated")
        if calibrated:
            if not math.isfinite(self.steps_per_frame) or not 0.25 <= self.steps_per_frame <= 20.0:
                raise ValueError("steps_per_frame outside the plausible 0.25-20 range")
            if self.requested_frame != self._frame(self.chunk_steps):
                raise ValueError("requested_frame must equal round(chunk_steps / steps_per_frame)")
            if self.frame_for_offset(self.replan_steps) > self.num_frames - 1:
                raise ValueError("generated clip is shorter than one replanning interval")
        if self.source_protocol_digest is not None and not re.fullmatch(
                r"[0-9a-f]{64}", self.source_protocol_digest):
            raise ValueError("source_protocol_digest must be a SHA-256 hex digest")

    @property
    def generation_fps(self) -> int:
        """Playback rate for saved videos only; not physical timing."""
        return self.display_fps

    @property
    def replan_steps(self) -> int:
        return self.chunk_steps * self.chunks_per_plan

    def _frame(self, sim_steps: int) -> int:
        return max(1, int(round(sim_steps / self.steps_per_frame)))

    def frame_for_offset(self, sim_steps: int) -> int:
        if self.steps_per_frame is None:
            raise RuntimeError("VPP2 frame timing is uncalibrated; run the source-only calibration first")
        return self._frame(sim_steps)

    def require_frame(self, frame: int) -> int:
        if frame < 0 or frame > self.num_frames - 1:
            raise ValueError(f"requested frame {frame} is outside clip [0, {self.num_frames - 1}]")
        return frame

    def subgoal_frames(self) -> list[int]:
        return [self.require_frame(self.frame_for_offset((c + 1) * self.chunk_steps))
                for c in range(self.chunks_per_plan)]

    def require_calibrated(self) -> None:
        if self.time_scale_status != "calibrated_on_source_validation":
            raise RuntimeError("temporal alignment is uncalibrated; measure on source validation first")
        if self.timing_basis != "verified_on_source_validation":
            raise RuntimeError("timing_basis is not verified on source validation")


def is_vpp2(protocol: dict) -> bool:
    return protocol.get("design", {}).get("generator", {}).get("family") == VPP2_FAMILY


def temporal_alignment(protocol: dict) -> "TemporalAlignment | VPP2TemporalAlignment":
    """Generator-specific timing; Cosmos protocols keep their original class."""
    temporal = protocol["design"]["temporal"]
    if is_vpp2(protocol) or temporal.get("generator_family") == VPP2_FAMILY:
        if not is_vpp2(protocol) or temporal.get("generator_family") != VPP2_FAMILY:
            raise ValueError("VPP2 generator and temporal records must be amended together")
        return VPP2TemporalAlignment(**temporal)
    return TemporalAlignment(**temporal)


@dataclass(frozen=True)
class RetrievalSpec:
    """Frozen reference-geometry retrieval (fixed for the pilot).

    Distance:
      d = 0.5*(1-cos(cur_ref,cur_bank)) + 0.5*(1-cos(fut_ref,fut_bank))
          + 0.1*mean((q_norm-q_bank_norm)^2)
    similarity = -d. Duration must match. Invalid slots and masked-memory
    features are zero with explicit masks.
    """
    k: int = 4
    temperature: float = 0.1
    distance: str = (
        "0.5*(1-cos(current_ref,current_bank))+0.5*(1-cos(future_ref,future_bank))"
        "+0.1*mean((q_norm-q_bank_norm)^2)"
    )
    similarity: str = "-distance"
    duration_match: bool = True
    cos_weight_current: float = 0.5
    cos_weight_future: float = 0.5
    q_weight: float = 0.1
    retrieved_fields: tuple[str, ...] = ("normalized_action_chunk", "similarity", "valid_flag")
    pooling: str = "similarity_sorted_slots_plus_softmax_weighted_mean_chunk"
    source_bank: str = "episodic_same_task_m_demos_excluding_query_demo"
    episodic_bank_sizes: tuple[int, ...] = SUPPORT_BUDGETS
    target_bank: str = "target_support_set_only_excluding_query_demo_during_adaptation"
    geometry: str = "frozen_reference_mae"


@dataclass(frozen=True)
class HeadSpec:
    hidden_dim: int = 512
    bottleneck_dim: int = 256
    dropout: float = 0.1
    activation: str = "relu"
    normalization: str = "layernorm"


@dataclass(frozen=True)
class StateSchema:
    """Robot state seen by the controller. Scientific path refuses while unverified."""
    name: str = "joint7_gripper2"
    joint_dim: int = 7
    gripper_dim: int = 2
    hdf5_joint_key: str = "obs/joint_states"
    hdf5_gripper_key: str = "obs/gripper_states"
    live_joint_key: str = "robot0_joint_pos"
    live_gripper_key: str = "robot0_gripper_qpos"
    parity_status: str = "unverified"

    @property
    def dim(self) -> int:
        return self.joint_dim + self.gripper_dim

    def require_verified(self) -> None:
        if self.parity_status != "verified":
            raise RuntimeError("state schema parity is unverified; run check-state-parity first")


@dataclass(frozen=True)
class GeneratorSpec:
    model_id: str = COSMOS_MODEL_ID
    revision: str = COSMOS_REVISION
    diffusers_commit: str = DIFFUSERS_COMMIT
    pipeline: str = "Cosmos3OmniPipeline"
    mode: str = "image_to_video"
    dtype: str = "bfloat16"
    height: int = 256
    width: int = 256
    num_inference_steps: int = 20
    guidance_scale: float = 6.0
    flow_shift: float = 12.0
    scheduler: str = "UniPCMultistepScheduler"
    add_resolution_template: bool = False
    add_duration_template: bool = False
    enable_safety_check: bool = True
    prompt_template: str = (
        "A fixed third-person camera view of a Franka robot arm on a tabletop. "
        "The robot will {instruction}."
    )
    negative_prompt_source: str = "assets/negative_prompt.json@revision"
    conditioning: str = "single_current_rgb_frame_and_instruction"
    prompt_upsampling: bool = False
    base_seed: int = 20261011


@dataclass(frozen=True)
class StageScope:
    """Pilot and unselected final confirmation inventories."""
    pilot_tasks: int = 2
    pilot_conditions: int = 4
    pilot_seeds: int = 1
    pilot_starts: int = 5
    pilot_budget: int = PRIMARY_BUDGET
    pilot_episodes: int = 40  # 2 * 4 * 1 * 5
    final_option_120: int = 120  # 2 tasks x 4 cond x 3 seeds x 5 starts
    final_option_240: int = 240  # 2 tasks x 4 cond x 3 seeds x 10 starts
    final_scope_selected: str | None = None
    source_panel_cases: int = 30


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list).encode()


def _identity_view(payload: dict, *, strip_calibration: bool) -> dict:
    view = json.loads(json.dumps(payload, default=list))
    view.pop("digest", None)
    view.pop("model_digest", None)
    for spec in view.get("tasks", {}).values():
        if isinstance(spec, dict):
            spec.pop("path", None)
    if strip_calibration:
        temporal = view.get("design", {}).get("temporal", {})
        if temporal.get("generator_family") == VPP2_FAMILY:
            temporal["steps_per_frame"] = None
            temporal["requested_frame"] = None
        temporal["time_scale"] = 1.0
        temporal["time_scale_status"] = "uncalibrated"
        temporal["timing_basis"] = "nominal_unverified"
        temporal.pop("source_protocol_digest", None)
        view.get("design", {}).pop("temporal_calibration_evidence_sha256", None)
        state = view.get("design", {}).get("state_schema", {})
        if isinstance(state, dict):
            state["parity_status"] = "unverified"
        view.get("design", {}).pop("state_parity_evidence_sha256", None)
    return view


def model_digest(payload: dict) -> str:
    """Identity of the trainable study. Path rebinding and verification stamps do not change it."""
    return hashlib.sha256(_canonical(_identity_view(payload, strip_calibration=True))).hexdigest()


def digest(payload: dict) -> str:
    view = _identity_view(payload, strip_calibration=False)
    view["model_digest"] = payload.get("model_digest") or model_digest(payload)
    return hashlib.sha256(_canonical(view)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest_obj = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest_obj.update(block)
    return digest_obj.hexdigest()


def v6_code_snapshot_digest(root: str | Path | None = None) -> str:
    """Hash of the hybrid_* modules that define v6 behaviour."""
    base = Path(root) if root else Path(__file__).resolve().parent
    names = sorted(p.name for p in base.glob("hybrid_*.py"))
    digest_obj = hashlib.sha256()
    for name in names:
        digest_obj.update(name.encode())
        digest_obj.update(Path(base / name).read_bytes())
    return digest_obj.hexdigest()


def design_record(temporal: TemporalAlignment | None = None,
                  retrieval: RetrievalSpec | None = None,
                  head: HeadSpec | None = None,
                  generator: GeneratorSpec | None = None,
                  state: StateSchema | None = None,
                  scope: StageScope | None = None) -> dict:
    temporal = temporal or TemporalAlignment()
    scope = scope or StageScope()
    return {
        "conditions": {name: dict(spec) for name, spec in CONDITIONS.items()},
        "diagnostic_controls": list(DIAGNOSTIC_CONTROLS),
        "temporal": asdict(temporal),
        "retrieval": asdict(retrieval or RetrievalSpec()),
        "head": asdict(head or HeadSpec()),
        "generator": asdict(generator or GeneratorSpec()),
        "state_schema": asdict(state or StateSchema()),
        "stage_scope": asdict(scope),
        "stages": {str(k): v for k, v in STAGES.items()},
        "inverse_model": {
            "reference_encoder": "frozen MAE ViT-B/16 normalized CLS (768) for bank keys and coverage",
            "controller_encoder": "MAE ViT-B/16; LoRA-adapted when A=1",
            "input": "(controller current RGB, controller future RGB, q_t, requested duration, retrieval block or null)",
            "output": "chunk_steps x 7 source-normalized actions",
            "lora": {"blocks": [10, 11], "projections": ["qkv", "proj"], "rank": 8, "alpha": 16, "dropout": 0.0,
                     "zero_init_effect": True},
        },
        "normalizer": "source_training_actions_and_proprio_per_dimension_shared",
        "source_objective": "0.5*retrieval_present_loss + 0.5*masked_memory_loss",
        "training": {
            "source_updates": 2000, "batch_size": 64, "microbatch_size": 8,
            "head_lr": 3e-4, "weight_decay": 1e-4, "validation_every": 100,
            "target_updates": 500, "target_updates_fixed": True,
            "lora_lr": 1e-4, "grad_clip": 1.0,
            "adaptation_conditions": ["lora_mlp", "raid_lora_mlp"],
        },
        "evaluation": {
            "recorded_transition_diagnostic": {
                "inputs": "recorded o_t, q_t, o_t+H, duration H",
                "target": "recorded a_t..a_t+H-1",
                "metric": "source-normalized chunk MSE, transition->demo->task macro average",
                "primary_budget": PRIMARY_BUDGET,
            },
            "generated_future_closed_loop": {
                "initial_states": "recorded initial simulator state of each selected query start",
                "max_steps": 300, "settle_steps": 10,
                "success": "env.check_success() official predicate",
                "replanning": "every replan_steps (=chunk_steps) from the condition's own observation",
                "chunks_per_plan": 1,
                "generation_reuse": "content-addressed: identical observation, prompt, seed and settings reuse one clip",
                "first_plan_shared": "identical starting inputs share one clip across all four conditions",
                "videos": "generated plans and executed rollouts saved side by side",
                "pilot_episodes": scope.pilot_episodes,
            },
            "contrasts": {
                "hybrid_over_raid": "S_11 - S_10",
                "hybrid_over_lora": "S_11 - S_01",
                "interaction": "S_11 - S_10 - S_01 + S_00",
                "co_primary": ["hybrid_over_raid", "hybrid_over_lora"],
            },
        },
        "costs_recorded_separately": [
            "generation", "bank_construction", "retrieval", "reference_encoding",
            "adaptation", "action_inference", "simulation", "source_training",
        ],
        "source_panel_gates": {
            "technical_decodes": "30/30",
            "qualitative_rubric_pass": ">=24/30",
            "expert_replay_success": ">=27/30",
            "measured": ["bf16_load", "peak_memory", "seconds_per_clip", "storage", "next_stage_cost"],
        },
    }


def _instruction(task: str, path: str | None) -> tuple[str, str]:
    if path and Path(path).is_file():
        import h5py
        with h5py.File(path, "r") as handle:
            info = handle["data"].attrs.get("problem_info")
        if info is not None:
            text = info.decode() if isinstance(info, bytes) else str(info)
            instruction = json.loads(text).get("language_instruction")
            if isinstance(instruction, str) and instruction.strip():
                return instruction.strip(), "hdf5_problem_info"
    return task.replace("_", " "), "task_name"


def build_hybrid_protocol(v3_manifest: dict, *,
                          v4_digest: str | None = None,
                          temporal: TemporalAlignment | None = None,
                          state: StateSchema | None = None,
                          code_root: str | Path | None = None) -> dict:
    """Derive v6 from the frozen v3 splits; record parent digests."""
    if v3_manifest.get("protocol") != PARENT_V3 or not v3_manifest.get("digest"):
        raise ValueError("v6 requires the frozen fewshot_idm_adaptation_v3 manifest")
    tasks = {}
    for name, spec in v3_manifest["tasks"].items():
        instruction, source = _instruction(name, spec.get("path"))
        tasks[name] = {
            "role": spec["role"], "path": spec["path"], "source_sha256": spec["source_sha256"],
            "demo_ids": list(spec["demo_ids"]), "demo_lengths": dict(spec["demo_lengths"]),
            "source": spec.get("source", {}), "support_pool": list(spec.get("support_pool", [])),
            "support_budgets_by_seed": spec.get("support_budgets_by_seed", {}),
            "query": list(spec.get("query", [])),
            "instruction": instruction, "instruction_source": source,
        }
    roles = {spec["role"] for spec in tasks.values()}
    if roles != {"source", "development", "final_test"}:
        raise ValueError("v6 requires separate source, development and final-test tasks")
    payload = {
        "protocol": PROTOCOL_V6,
        "protocol_version": PROTOCOL_V6_VERSION,
        "parent_protocols": {
            "v3": {"protocol": PARENT_V3, "digest": v3_manifest["digest"]},
            "v4": {"protocol": PARENT_V4, "digest": v4_digest},
        },
        "source_revision": {"hybrid_modules_sha256": v6_code_snapshot_digest(code_root)},
        "benchmark": v3_manifest.get("benchmark", "libero_goal"),
        "libero_commit": v3_manifest.get("libero_commit"),
        "support_budgets": list(SUPPORT_BUDGETS),
        "primary_budget": PRIMARY_BUDGET,
        "development_seed": DEVELOPMENT_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "design": design_record(temporal, state=state),
        "tasks": tasks,
        "scientific_result": False,
        "readiness": False,
        "earlier_protocol_results": (
            "v3 recorded-frame and v4 two-arm results are historical and not v6 evidence"
        ),
    }
    # Five deterministic query starts per development task (pilot closed-loop inventory).
    payload["design"]["stage_scope"]["pilot_start_demos"] = {
        name: list(map(int, spec["query"][:5]))
        for name, spec in tasks.items() if spec["role"] == "development"
    }
    payload["model_digest"] = model_digest(payload)
    payload["digest"] = digest(payload)
    return payload


def bind_task_paths(protocol: dict, paths: dict[str, str], *,
                    task_names: list[str] | tuple[str, ...] | None = None) -> dict:
    updated = json.loads(json.dumps(protocol))
    names = list(updated["tasks"]) if task_names is None else list(dict.fromkeys(task_names))
    if set(names) - set(updated["tasks"]):
        raise ValueError("path binding contains unknown tasks")
    for task in names:
        spec = updated["tasks"][task]
        path = Path(paths.get(task) or spec.get("path") or "")
        if not path.is_file():
            raise FileNotFoundError(f"HDF5 for {task} is not a file: {path}")
        hashed = _sha256_file(path)
        if hashed != spec["source_sha256"]:
            raise ValueError(f"HDF5 hash mismatch for {task}")
        spec["path"] = str(path)
    if model_digest(updated) != protocol.get("model_digest") or digest(updated) != protocol.get("digest"):
        raise RuntimeError("binding task paths changed protocol identity")
    updated["model_digest"] = protocol["model_digest"]
    updated["digest"] = protocol["digest"]
    return updated


def find_task_paths(protocol: dict, root: str | Path, *,
                    task_names: list[str] | tuple[str, ...] | None = None) -> dict[str, str]:
    names = list(protocol["tasks"]) if task_names is None else list(dict.fromkeys(task_names))
    if set(names) - set(protocol["tasks"]):
        raise ValueError("path lookup contains unknown tasks")
    expected_hashes = {protocol["tasks"][task]["source_sha256"] for task in names}
    permitted_names = {Path(protocol['tasks'][task]['path']).name for task in names}
    index: dict[str, str] = {}
    for file in Path(root).rglob("*.hdf5"):
        if task_names is not None and file.name not in permitted_names:
            # Enforce stage access before hashing or opening a dataset.
            continue
        digest = _sha256_file(file)
        if digest in expected_hashes:
            index.setdefault(digest, str(file))
    found = {}
    for task in names:
        spec = protocol["tasks"][task]
        path = index.get(spec["source_sha256"])
        if path is None:
            raise FileNotFoundError(f"no HDF5 under {root} matches {task}; missing input or hash mismatch")
        found[task] = path
    return found


def validate_hybrid_protocol(protocol: dict) -> dict:
    if protocol.get("protocol") != PROTOCOL_V6:
        raise ValueError("v6 protocol identity is invalid")
    temporal_alignment(protocol)
    StateSchema(**protocol["design"]["state_schema"])
    for name in CONDITION_NAMES:
        if name not in protocol["design"]["conditions"]:
            raise ValueError(f"missing condition {name}")
    for name, spec in protocol["tasks"].items():
        if spec["role"] == "source":
            continue
        pool, query = set(spec["support_pool"]), set(spec["query"])
        if pool & query:
            raise ValueError(f"support and query overlap for {name}")
        for seed, budgets in spec["support_budgets_by_seed"].items():
            previous: list[int] = []
            for budget in SUPPORT_BUDGETS:
                ids = list(budgets[str(budget)])
                if len(ids) != budget or not set(ids) <= pool or ids[:len(previous)] != previous:
                    raise ValueError(f"support sets for {name} seed {seed} are not nested pool subsets")
                previous = ids
    text = json.dumps(protocol).lower()
    if any(marker in text for marker in PROHIBITED_MODEL_MARKERS):
        raise ValueError("GR-1 is prohibited in RAID protocols")
    if protocol.get("model_digest") != model_digest(protocol) or protocol.get("digest") != digest(protocol):
        raise ValueError("v6 protocol identity or digest is invalid")
    return protocol


def with_verified_state_parity(protocol: dict, evidence_sha256: str) -> dict:
    state = dict(protocol["design"]["state_schema"])
    state["parity_status"] = "verified"
    updated = json.loads(json.dumps(protocol))
    updated["design"]["state_schema"] = state
    updated["design"]["state_parity_evidence_sha256"] = evidence_sha256
    updated["model_digest"] = model_digest(updated)
    updated["digest"] = digest(updated)
    return updated


def with_verified_timing(protocol: dict, evidence_sha256: str, *,
                         time_scale: float = 1.0,
                         source_protocol_digest: str | None = None) -> dict:
    temporal = dict(protocol["design"]["temporal"])
    temporal.update(time_scale=float(time_scale), time_scale_status="calibrated_on_source_validation",
                    timing_basis="verified_on_source_validation")
    if source_protocol_digest:
        temporal["source_protocol_digest"] = source_protocol_digest
    TemporalAlignment(**temporal)
    updated = json.loads(json.dumps(protocol))
    updated["design"]["temporal"] = temporal
    updated["design"]["temporal_calibration_evidence_sha256"] = evidence_sha256
    updated["model_digest"] = model_digest(updated)
    updated["digest"] = digest(updated)
    return updated


def with_vpp2_timing(protocol: dict, evidence_sha256: str, *, steps_per_frame: float,
                     source_protocol_digest: str | None = None) -> dict:
    """Stamp the source-only VPP2 frame calibration; the model digest is unchanged."""
    if not is_vpp2(protocol):
        raise ValueError("VPP2 timing applies only to a VPP2-amended protocol")
    temporal = dict(protocol["design"]["temporal"])
    chunk = int(temporal["chunk_steps"])
    temporal.update(steps_per_frame=float(steps_per_frame),
                    requested_frame=max(1, int(round(chunk / float(steps_per_frame)))),
                    time_scale_status="calibrated_on_source_validation",
                    timing_basis="verified_on_source_validation")
    if source_protocol_digest:
        temporal["source_protocol_digest"] = source_protocol_digest
    VPP2TemporalAlignment(**temporal)
    updated = json.loads(json.dumps(protocol))
    updated["design"]["temporal"] = temporal
    updated["design"]["temporal_calibration_evidence_sha256"] = evidence_sha256
    updated["model_digest"] = model_digest(updated)
    updated["digest"] = digest(updated)
    if updated["model_digest"] != protocol["model_digest"]:
        raise RuntimeError("timing stamp changed the trainable study identity")
    return updated


def condition_flags(name: str) -> dict:
    if name not in CONDITIONS:
        raise ValueError(f"unknown condition {name}")
    return dict(CONDITIONS[name])


def assert_no_prohibited_model(identifier: str) -> None:
    lowered = identifier.lower()
    if any(marker in lowered for marker in PROHIBITED_MODEL_MARKERS):
        raise ValueError("GR-1 is prohibited in RAID")
