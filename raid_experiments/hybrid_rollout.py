"""Generated-future closed-loop evaluation for protocol v6 (four conditions).

One chunk per replan. Requested frames past the clip raise. The first plan is
shared across conditions at identical starting inputs via the content-addressed
cache. Hybrid conditions charge reference-encoding cost separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import traceback
from typing import Any, Callable

import numpy as np
import torch

from .hybrid_protocol import TemporalAlignment, condition_flags
from .hybrid_retrieval import ReferenceBank
from .world_model import FrozenWorldModel, generation_seed, resize_rgb, to_display, write_mp4

# Reuse LIBERO helpers from v4 rollout (env construction is protocol-agnostic).
from .wm_rollout import (SETTLE_ACTION, EvaluationCase, goal_bddl, make_goal_env,
                         observation_frame, postprocess_action, write_libero_config)

__all__ = ["ControllerPolicy", "EpisodeResult", "EvaluationCase", "run_episode",
           "run_matched_case", "summarize", "make_goal_env", "goal_bddl",
           "write_libero_config", "SETTLE_ACTION"]


@dataclass
class EpisodeResult:
    condition: str
    case_id: str
    success: bool
    steps: int
    outcome: str
    plans: list[dict] = field(default_factory=list)
    generation_seconds: float = 0.0
    inference_seconds: float = 0.0
    reference_encoding_seconds: float = 0.0
    simulation_seconds: float = 0.0
    error_traceback: str | None = None
    executed_actions: list[list[float]] = field(default_factory=list)
    video: str | None = None


def _frames_tensor(*frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.stack(frames)))


class ControllerPolicy:
    """One policy class for all four conditions."""

    def __init__(self, name: str, controller_encoder, reference_encoder, head,
                 bank: ReferenceBank | None, mean, std, q_mean, q_std,
                 temporal: TemporalAlignment, device: str = "cpu",
                 proprio_reader: Callable[[Any], np.ndarray] | None = None,
                 current_view: Callable[[np.ndarray], np.ndarray] | None = None):
        flags = condition_flags(name)
        self.name = name
        self.R, self.A = flags["R"], flags["A"]
        self.controller_encoder = controller_encoder
        self.reference_encoder = reference_encoder
        self.head = head.eval()
        self.bank = bank
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        self.q_mean = torch.as_tensor(q_mean)
        self.q_std = torch.as_tensor(q_std)
        self.temporal = temporal
        self.device = device
        self.proprio_reader = proprio_reader or _default_proprio
        # The live frame enters in the protocol's controller view; generated
        # subgoals arrive from the world model already in that view.
        self.current_view = current_view or (lambda frame: frame)
        self.timing: list[dict] = []

    @torch.no_grad()
    def predict_chunk(self, current_rgb: np.ndarray, subgoal_rgb: np.ndarray, obs) -> np.ndarray:
        current_rgb = self.current_view(current_rgb)
        if current_rgb.shape != subgoal_rgb.shape:
            raise ValueError("current and generated frames are in different controller views")
        t0 = time.perf_counter()
        feats = self.controller_encoder.encode_frames(_frames_tensor(current_rgb, subgoal_rgb)).to(self.device)
        t1 = time.perf_counter()
        q_raw = self.proprio_reader(obs)
        q = ((torch.as_tensor(q_raw, dtype=torch.float32) - self.q_mean) / self.q_std).unsqueeze(0).to(self.device)
        dur = torch.tensor([float(self.temporal.chunk_steps)], device=self.device)
        ref_seconds = 0.0
        if self.R:
            # Frozen geometry: retrieve with reference features; control with controller features.
            from .hybrid_retrieval import retrieval_block
            tr0 = time.perf_counter()
            ref = self.reference_encoder.encode_frames(_frames_tensor(current_rgb, subgoal_rgb)).to(self.device)
            ref_seconds = time.perf_counter() - tr0
            block, availability, info = retrieval_block(
                self.bank, ref[:1], ref[1:], q, dur, spec=self.head.retrieval,
                chunk_steps=self.head.chunk_steps)
            pred = self.head(feats[:1], feats[1:], q, dur, block, availability)
        else:
            pred, info = self.head.forward_with_memory(
                feats[:1], feats[1:], q, dur, bank=None, use_retrieval=False)
        t2 = time.perf_counter()
        self.timing.append({"encode": t1 - t0, "head": t2 - t1,
                            "reference_encoding": ref_seconds, "info_keys": list(info)})
        out = pred[0].cpu()
        return (out * self.std + self.mean).numpy()


def _default_proprio(obs) -> np.ndarray:
    if not isinstance(obs, dict):
        raise KeyError("observation is not a dict; cannot read proprioception")
    if "robot0_joint_pos" in obs and "robot0_gripper_qpos" in obs:
        return np.concatenate([np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1),
                               np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)])
    if "joint_states" in obs and "gripper_states" in obs:
        return np.concatenate([np.asarray(obs["joint_states"], dtype=np.float32).reshape(-1),
                               np.asarray(obs["gripper_states"], dtype=np.float32).reshape(-1)])
    raise KeyError("observation missing joint/gripper proprioception")


def _reset(env, initial_state, *, settle_steps: int = 10):
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
        obs = obs[0] if isinstance(obs, tuple) else obs
        for _ in range(settle_steps):
            result = env.step(SETTLE_ACTION)
            obs = result[0] if isinstance(result, tuple) else result
        obs = env.set_init_state(initial_state)
        obs = obs[0] if isinstance(obs, tuple) else obs
    return obs


def run_episode(env, policy: ControllerPolicy, world_model: FrozenWorldModel,
                case: EvaluationCase, *, temporal: TemporalAlignment,
                out_dir: str | Path, max_steps: int = 300,
                settle_steps: int = 10, initial_plan=None) -> EpisodeResult:
    out = Path(out_dir)
    result = EpisodeResult(policy.name, case.case_id, False, 0, "timeout")
    executed_frames, plan_frames = [], []
    try:
        obs = _reset(env, case.initial_state, settle_steps=settle_steps)
        step, replan = 0, 0
        subgoals = temporal.subgoal_frames()  # raises if beyond clip
        while step < max_steps and not result.success:
            current = observation_frame(obs)
            seed = generation_seed(case.case_id, replan, case.base_seed)
            if replan == 0 and initial_plan is not None:
                plan = initial_plan  # the case's shared first plan (identical for every condition)
            else:
                plan = _generate(world_model, current, case, replan)
            result.generation_seconds += plan.generation_seconds
            result.plans.append({"replan_index": replan, "start_step": step, "seed": seed,
                                 "cache_key": plan.cache_key, "cache_hit": plan.cache_hit,
                                 "generation_seconds": plan.generation_seconds,
                                 "prompt_fallback": (getattr(plan, "metadata", None) or {}).get("prompt_fallback"),
                                 "prompt_seconds": (getattr(plan, "metadata", None) or {}).get("prompt_seconds")})
            for c in range(temporal.chunks_per_plan):
                if step >= max_steps or result.success:
                    break
                frame_idx = temporal.require_frame(subgoals[c])
                t0 = time.perf_counter()
                chunk = policy.predict_chunk(observation_frame(obs), plan.subgoal(frame_idx), obs)
                result.inference_seconds += time.perf_counter() - t0
                if policy.timing:
                    result.reference_encoding_seconds += policy.timing[-1].get("reference_encoding", 0.0)
                if chunk.shape != (temporal.chunk_steps, 7) or not np.isfinite(chunk).all():
                    raise ValueError("condition returned an invalid action chunk")
                for i, action in enumerate(chunk):
                    if step >= max_steps:
                        break
                    executed = postprocess_action(action)
                    t1 = time.perf_counter()
                    obs, _, done, info = env.step(executed)[:4]
                    result.simulation_seconds += time.perf_counter() - t1
                    step += 1
                    result.executed_actions.append(executed.tolist())
                    offset = c * temporal.chunk_steps + i + 1
                    frame_for_video = temporal.require_frame(temporal.frame_for_offset(offset)) \
                        if temporal.frame_for_offset(offset) <= temporal.num_frames - 1 \
                        else temporal.num_frames - 1
                    # Display-only index for side-by-side video may use last frame; control path already validated subgoals.
                    executed_frames.append(to_display(observation_frame(obs)))
                    plan_frames.append(plan.display_frames[min(frame_for_video, len(plan.display_frames) - 1)])
                    if bool(env.check_success()):
                        result.success, result.outcome = True, "success"
                        break
            replan += 1
        result.steps = step
    except Exception as exc:
        from .hybrid_vpp2 import GeneratorFailure
        if isinstance(exc, GeneratorFailure):
            raise  # infrastructure: the case stays incomplete and is never saved as a robot failure
        result.outcome = f"exception:{type(exc).__name__}:{exc}"
        result.error_traceback = traceback.format_exc()
    if executed_frames:
        size = executed_frames[0].shape[:2]
        left = resize_rgb(np.stack(plan_frames), size)
        side = np.concatenate((left, np.stack(executed_frames)), axis=2)
        path = out / f"{case.case_id}_{policy.name}_plan_vs_rollout.mp4"
        write_mp4(side, path, temporal.control_hz)
        write_mp4(np.stack(executed_frames), out / f"{case.case_id}_{policy.name}_rollout.mp4",
                  temporal.control_hz)
        result.video = str(path)
    return result


def _generate(world_model, frame, case: EvaluationCase, replan: int):
    seed = generation_seed(case.case_id, replan, case.base_seed)
    slots = int(getattr(world_model, "prompt_slots", 0) or 0)
    return world_model.generate(frame, case.instruction, seed=seed,
                                **({"prompt_slot": replan % slots} if slots else {}))


def shared_initial_plan(env_factory, world_model, case: EvaluationCase, *, settle_steps: int = 10):
    """One first plan per case from a fresh reset of the registered initial state.

    Fresh simulator instances can render the same state with slightly different
    pixels, so per-condition first plans would differ; the matched design gives
    every condition this one plan, then each replans from its own observations.
    """
    env = env_factory(case.task)
    try:
        obs = _reset(env, case.initial_state, settle_steps=settle_steps)
        return _generate(world_model, observation_frame(obs), case, 0)
    finally:
        close = getattr(env, "close", None)
        if close:
            close()


def run_matched_case(env_factory, policies: dict[str, ControllerPolicy],
                     world_model: FrozenWorldModel, case: EvaluationCase, *,
                     temporal: TemporalAlignment, out_dir: str | Path,
                     max_steps: int = 300, settle_steps: int = 10,
                     write_record: bool = True) -> dict:
    """Run all four conditions from the identical initial state."""
    records = {}
    first_plan = shared_initial_plan(env_factory, world_model, case, settle_steps=settle_steps)
    for name, policy in policies.items():
        env = env_factory(case.task)
        try:
            records[name] = run_episode(env, policy, world_model, case, temporal=temporal,
                                        out_dir=out_dir, max_steps=max_steps,
                                        settle_steps=settle_steps, initial_plan=first_plan)
        finally:
            close = getattr(env, "close", None)
            if close:
                close()
    for record in records.values():
        for plan in record.plans:
            plan["generator_height"] = int(world_model.spec.height)
    first = {name: r.plans[0]["cache_key"] if r.plans else None for name, r in records.items()}
    summary = {"case_id": case.case_id, "task": case.task, "query_demo": case.query_demo,
               "shared_initial_plan": len(set(first.values())) == 1 and None not in first.values(),
               "initial_plan_keys": first,
               "conditions": {name: r.__dict__ for name, r in records.items()}}
    if write_record:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        tmp = Path(out_dir) / f"{case.case_id}.json.tmp"
        tmp.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        tmp.replace(Path(out_dir) / f"{case.case_id}.json")
    return summary


def summarize(episodes: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = {}
    for case in episodes:
        for cond, record in case["conditions"].items():
            by_cond.setdefault(cond, []).append(record)
    summary = {}
    for cond, rows in by_cond.items():
        outcomes: dict[str, int] = {}
        for row in rows:
            key = row["outcome"].split(":")[0] if row["outcome"].startswith("exception") else row["outcome"]
            outcomes[key] = outcomes.get(key, 0) + 1
        summary[cond] = {
            "episodes": len(rows),
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "successes": int(sum(r["success"] for r in rows)),
            "outcomes": outcomes,
            "generation_seconds": float(sum(r["generation_seconds"] for r in rows)),
            "inference_seconds": float(sum(r["inference_seconds"] for r in rows)),
            "reference_encoding_seconds": float(sum(r.get("reference_encoding_seconds", 0) for r in rows)),
            "simulation_seconds": float(sum(r["simulation_seconds"] for r in rows)),
            "generated_clips": int(sum(len(r["plans"]) for r in rows)),
            "cache_hits": int(sum(p["cache_hit"] for r in rows for p in r["plans"])),
        }
    return summary
