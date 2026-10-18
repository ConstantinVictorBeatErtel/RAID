"""Generated-future closed-loop evaluation in LIBERO.

Every ``replan_steps`` each arm asks the frozen world model for a clip
conditioned on *its own* current observation and the task instruction. For
chunk c of a plan, the arm infers an action chunk from (its current real
observation, generated frame at ``temporal.subgoal_frames()[c]``) and executes
those ``chunk_steps`` actions. At step 0 both arms present the identical
initial observation with the identical seed, so the content-addressed cache
returns one shared clip; after their trajectories diverge each arm's clips are
its own. Success is the simulator's official predicate only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np
import torch

from .wm_heads import retrieval_block
from .wm_protocol import RetrievalSpec, TemporalAlignment
from .world_model import FrozenWorldModel, generation_seed, resize_rgb, to_display, write_mp4

SETTLE_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


class ArmPolicy(Protocol):
    name: str

    def predict_chunk(self, current: np.ndarray, subgoal: np.ndarray) -> np.ndarray: ...


def _frames_tensor(*frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.stack(frames)))


class RaidPolicy:
    """Frozen encoder + retrieval-aware MLP + bank holding the target support set."""
    name = "raid_bank_mlp"

    def __init__(self, encoder, head, bank, mean, std, retrieval: RetrievalSpec, device: str = "cpu"):
        self.encoder, self.head, self.bank = encoder, head.eval(), bank
        self.mean, self.std, self.retrieval, self.device = mean, std, retrieval, device
        self.timing: list[dict] = []

    @torch.no_grad()
    def predict_chunk(self, current, subgoal):
        t0 = time.perf_counter()
        feats = self.encoder.encode_frames(_frames_tensor(current, subgoal)).to(self.device)
        t1 = time.perf_counter()
        block, info = retrieval_block(self.bank, feats[:1], feats[1:], spec=self.retrieval,
                                      chunk_steps=self.head.chunk_steps)
        t2 = time.perf_counter()
        out = self.head(feats[:1], feats[1:], block)[0].cpu()
        t3 = time.perf_counter()
        self.timing.append({"encode": t1 - t0, "retrieval": t2 - t1, "head": t3 - t2,
                            "neighbours": info["indices"][0].tolist()})
        return (out * self.std + self.mean).numpy()


class LoraPolicy:
    """LoRA-adapted encoder + adapted MLP; no retrieval."""
    name = "lora_mlp"

    def __init__(self, encoder, head, mean, std, device: str = "cpu"):
        self.encoder, self.head, self.mean, self.std, self.device = encoder, head.eval(), mean, std, device
        self.timing: list[dict] = []

    @torch.no_grad()
    def predict_chunk(self, current, subgoal):
        t0 = time.perf_counter()
        feats = self.encoder.encode_frames(_frames_tensor(current, subgoal)).to(self.device)
        t1 = time.perf_counter()
        out = self.head(feats[:1], feats[1:])[0].cpu()
        t2 = time.perf_counter()
        self.timing.append({"encode": t1 - t0, "head": t2 - t1})
        return (out * self.std + self.mean).numpy()


def postprocess_action(action: np.ndarray) -> np.ndarray:
    """Controller input range is [-1, 1]; the gripper command is binary."""
    executed = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    executed[6] = 1.0 if executed[6] > 0 else -1.0
    return executed


def observation_frame(obs) -> np.ndarray:
    for key in ("agentview_image", "agentview_rgb", "image"):
        if isinstance(obs, dict) and key in obs:
            return np.asarray(obs[key], dtype=np.uint8)
    raise KeyError("observation has no agent-view RGB image")


@dataclass
class EvaluationCase:
    case_id: str
    task: str
    instruction: str
    initial_state: np.ndarray
    query_demo: int
    base_seed: int


@dataclass
class EpisodeResult:
    arm: str
    case_id: str
    success: bool
    steps: int
    outcome: str
    plans: list[dict] = field(default_factory=list)
    generation_seconds: float = 0.0
    inference_seconds: float = 0.0
    simulation_seconds: float = 0.0
    executed_actions: list[list[float]] = field(default_factory=list)
    video: str | None = None


def _reset(env, initial_state):
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
        obs = obs[0] if isinstance(obs, tuple) else obs
    return obs


def run_episode(env, policy: ArmPolicy, world_model: FrozenWorldModel, case: EvaluationCase, *,
                temporal: TemporalAlignment, out_dir: str | Path, max_steps: int = 300,
                settle_steps: int = 10) -> EpisodeResult:
    out = Path(out_dir)
    result = EpisodeResult(policy.name, case.case_id, False, 0, "timeout")
    executed_frames, plan_frames = [], []
    try:
        obs = _reset(env, case.initial_state)
        for _ in range(settle_steps):
            obs = env.step(SETTLE_ACTION)[0]
        step, replan = 0, 0
        subgoals = temporal.subgoal_frames()
        while step < max_steps and not result.success:
            current = observation_frame(obs)
            seed = generation_seed(case.case_id, replan, case.base_seed)
            plan = world_model.generate(current, case.instruction, seed=seed)
            result.generation_seconds += plan.generation_seconds
            result.plans.append({"replan_index": replan, "start_step": step, "seed": seed,
                                 "cache_key": plan.cache_key, "cache_hit": plan.cache_hit,
                                 "generation_seconds": plan.generation_seconds})
            for c in range(temporal.chunks_per_plan):
                if step >= max_steps or result.success:
                    break
                t0 = time.perf_counter()
                chunk = policy.predict_chunk(observation_frame(obs), plan.subgoal(subgoals[c]))
                result.inference_seconds += time.perf_counter() - t0
                if chunk.shape != (temporal.chunk_steps, 7) or not np.isfinite(chunk).all():
                    raise ValueError("arm returned an invalid action chunk")
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
                    executed_frames.append(to_display(observation_frame(obs)))
                    plan_frames.append(plan.display_frames[min(temporal.frame_for_offset(offset),
                                                               len(plan.display_frames) - 1)])
                    if bool(env.check_success()):
                        result.success, result.outcome = True, "success"
                        break
            replan += 1
        result.steps = step
    except Exception as exc:  # recorded as a failure, never silently dropped
        result.outcome = f"exception:{type(exc).__name__}:{exc}"
    if executed_frames:
        size = executed_frames[0].shape[:2]
        left = resize_rgb(np.stack(plan_frames), size)
        side = np.concatenate((left, np.stack(executed_frames)), axis=2)
        path = out / f"{case.case_id}_{policy.name}_plan_vs_rollout.mp4"
        write_mp4(side, path, temporal.control_hz)
        write_mp4(np.stack(executed_frames), out / f"{case.case_id}_{policy.name}_rollout.mp4", temporal.control_hz)
        result.video = str(path)
    return result


def run_matched_case(env_factory, policies: dict[str, ArmPolicy], world_model: FrozenWorldModel,
                     case: EvaluationCase, *, temporal: TemporalAlignment, out_dir: str | Path,
                     max_steps: int = 300, settle_steps: int = 10) -> dict:
    """Run both arms from the identical initial state; each conditions on its own observations."""
    records = {}
    for name, policy in policies.items():
        env = env_factory(case.task)
        try:
            records[name] = run_episode(env, policy, world_model, case, temporal=temporal,
                                        out_dir=out_dir, max_steps=max_steps, settle_steps=settle_steps)
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
               "arms": {name: r.__dict__ for name, r in records.items()}}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"{case.case_id}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def goal_bddl(task: str, libero_root: str | Path) -> Path:
    bddl = Path(libero_root) / "libero" / "libero" / "bddl_files" / "libero_goal" / f"{task}.bddl"
    if not bddl.is_file():
        raise FileNotFoundError(f"LIBERO-Goal BDDL missing for {task}: {bddl}")
    return bddl


def write_libero_config(libero_root: str | Path, config_dir: str | Path) -> Path:
    """Point LIBERO at the pinned checkout. The package reads this file on import."""
    root = Path(libero_root)
    benchmark = root / "libero" / "libero"
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "config.yaml"
    path.write_text(
        "benchmark_root: {benchmark}\n"
        "bddl_files: {benchmark}/bddl_files\n"
        "init_states: {benchmark}/init_files\n"
        "datasets: {root}/datasets\n"
        "assets: {benchmark}/assets\n".format(benchmark=benchmark, root=root)
    )
    return path


def make_goal_env(task: str, libero_root: str | Path, *, camera_size: int = 128):
    """LIBERO-Goal environment for one task (the earlier helper loaded LIBERO-Spatial files)."""
    import os
    from libero.libero.envs import OffScreenRenderEnv
    bddl = goal_bddl(task, libero_root)
    if not os.environ.get("LIBERO_CONFIG_PATH"):
        write_libero_config(libero_root, Path.home() / ".libero")
    return OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=camera_size, camera_widths=camera_size)


def summarize(episodes: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = {}
    for case in episodes:
        for arm, record in case["arms"].items():
            by_arm.setdefault(arm, []).append(record)
    summary = {}
    for arm, rows in by_arm.items():
        outcomes: dict[str, int] = {}
        for row in rows:
            key = row["outcome"].split(":")[0] if row["outcome"].startswith("exception") else row["outcome"]
            outcomes[key] = outcomes.get(key, 0) + 1
        summary[arm] = {"episodes": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
                        "outcomes": outcomes,
                        "generation_seconds": float(sum(r["generation_seconds"] for r in rows)),
                        "inference_seconds": float(sum(r["inference_seconds"] for r in rows)),
                        "simulation_seconds": float(sum(r["simulation_seconds"] for r in rows)),
                        "generated_clips": int(sum(len(r["plans"]) for r in rows)),
                        "cache_hits": int(sum(p["cache_hit"] for r in rows for p in r["plans"]))}
    return summary
