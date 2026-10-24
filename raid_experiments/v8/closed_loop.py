"""Closed-loop test of an inverse-dynamics model with recorded future frames (no video generator).

For each query demo: reset the simulator to the demo's first recorded state. At each 8-step chunk, align the
current observation to the demo (nearest frame by frozen-encoder cosine distance, searched forward over a
24-frame window, never moving backwards), take the recorded frame 8 steps later as the target, predict an
action chunk, and execute it. Success = LIBERO's task predicate within 1.5 x demo length + 20 steps.
All query demos of a task run in parallel simulator processes; the policy runs batched on the GPU.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import data as d
from . import encoders as enc

ALIGN_WINDOW = 24
SETTLE = 10
DUMMY = [0.0] * 6 + [-1.0]


def bddl_path(libero_root: str | Path, task: d.Task) -> Path:
    path = Path(libero_root) / "libero" / "libero" / "bddl_files" / task.suite / f"{task.name}.bddl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def vector_env(libero_root, task: d.Task, n: int):
    from ..wm_rollout import write_libero_config
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH") or Path.home() / ".libero")
    if not (config_dir / "config.yaml").is_file():
        write_libero_config(libero_root, config_dir)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    import multiprocessing
    from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv, SubprocVectorEnv
    args = {"bddl_file_name": str(bddl_path(libero_root, task)), "camera_heights": 128, "camera_widths": 128}
    # Forking after CUDA is initialised breaks EGL context creation in the children (EGL_BAD_ALLOC on the pod):
    # start simulator processes with "spawn"; if they still fail, step the environments in this process.
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
    try:
        env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**args) for _ in range(n)])
        env.reset()
        return env
    except Exception as error:  # noqa: BLE001 - any simulator start failure falls back to the in-process path
        print("closed_loop: subprocess simulators failed, using in-process environments:", type(error).__name__,
              str(error)[:200], flush=True)
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
        env = DummyVectorEnv([lambda: OffScreenRenderEnv(**args) for _ in range(n)])
        env.reset()
        return env


def _proprio(obs: dict) -> np.ndarray:
    return np.concatenate([np.asarray(obs["robot0_joint_pos"], np.float32).reshape(-1),
                           np.asarray(obs["robot0_gripper_qpos"], np.float32).reshape(-1)])


def reset(env, states: list[np.ndarray]):
    env.reset()
    obs = env.set_init_state(states)
    for _ in range(SETTLE):  # let objects settle, as LIBERO's evaluation does
        obs, _, _, _ = env.step(np.array([DUMMY] * len(states)))
    return env.set_init_state(states)


def orientation(env_frame: np.ndarray, demo_frame: np.ndarray) -> bool:
    """True if the simulator image must be flipped vertically to match the recorded frames."""
    a, b = env_frame.astype(np.float32), demo_frame.astype(np.float32)
    return float(((a[::-1] - b) ** 2).mean()) < float(((a - b) ** 2).mean())


class Episode:
    def __init__(self, demo: int, rec: dict, ref_feats: torch.Tensor):
        self.demo, self.frames, self.ref = demo, rec["frames"], ref_feats  # ref: (T, 768) frozen features
        self.T = len(self.frames)
        self.progress, self.done, self.success, self.steps = 0, False, False, 0
        self.max_steps = int(1.5 * self.T) + 20


def run_task(task: d.Task, demos: list[int], ref_cache: dict, policy, frozen_encoder, norms: dict, *, libero_root,
             env=None, flip: bool | None = None, gripper: str = "raw", log=print) -> dict:
    """``policy(current_imgs uint8 (B,H,W,3), cur_ref (B,768), target_idx per episode, episodes, q (B,9)) -> (B,8,7)``
    normalized action chunks."""
    episodes = []
    for demo in demos:
        rec = d.read_demo(task.path, demo)
        start, n = ref_cache["demos"][int(demo)]
        episodes.append(Episode(demo, rec, ref_cache["features"][start:start + n].float()))
    own_env = env is None
    env = env or vector_env(libero_root, task, len(demos))
    started = time.time()
    try:
        obs = reset(env, [d.read_demo(task.path, e.demo, frames=False)["state0"] for e in episodes])
        if flip is None:
            flip = orientation(obs[0]["agentview_image"], episodes[0].frames[0])
        a_mean, a_std = norms["action_mean"].numpy(), norms["action_std"].numpy()
        q_mean, q_std = norms["proprio_mean"].numpy(), norms["proprio_std"].numpy()
        while not all(e.done for e in episodes):
            imgs = np.stack([o["agentview_image"][::-1] if flip else o["agentview_image"] for o in obs])
            with torch.no_grad():
                cur_ref = enc.encode(frozen_encoder, imgs)
            for i, e in enumerate(episodes):
                if e.done:
                    continue
                hi = min(e.T - 1, e.progress + ALIGN_WINDOW)
                window = F.normalize(e.ref[e.progress:hi + 1], dim=-1) @ F.normalize(cur_ref[i], dim=-1)
                e.progress = e.progress + int(torch.argmax(window))
            targets = [min(e.progress + d.CHUNK, e.T - 1) for e in episodes]
            q = (np.stack([_proprio(o) for o in obs]) - q_mean) / q_std
            chunk = policy(imgs, cur_ref, targets, episodes, torch.from_numpy(q).float())
            actions = np.clip(chunk.cpu().numpy() * a_std + a_mean, -1.0, 1.0)
            if gripper == "binary":  # LIBERO grippers are commanded open (-1) or closed (+1); regression gives in-between values
                actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0)
            for k in range(d.CHUNK):
                step_actions = np.array([actions[i, k] if not e.done else DUMMY for i, e in enumerate(episodes)])
                obs, _, dones, _ = env.step(step_actions)
                for i, e in enumerate(episodes):
                    if e.done:
                        continue
                    e.steps += 1
                    if bool(dones[i]):
                        e.done, e.success = True, True
                    elif e.steps >= e.max_steps:
                        e.done = True
    finally:
        if own_env:
            env.close()
    return {"task": task.key, "flip": bool(flip), "gripper": gripper, "seconds": round(time.time() - started, 1),
            "episodes": [{"demo": e.demo, "success": e.success, "steps": e.steps, "demo_length": e.T,
                          "max_steps": e.max_steps} for e in episodes],
            "success_rate": float(np.mean([e.success for e in episodes]))}


def head_policy(model, frozen_encoder, *, bank=None, use_retrieval: bool, adapted_encoder=None, device: str):
    """Policy for heads on frozen features (bank and retrieval keys always in the frozen space) or on an
    adapted encoder (LoRA / full fine-tuning; retrieval off)."""
    def act(imgs, cur_ref, targets, episodes, q):
        tgt_ref = torch.stack([e.ref[t] for e, t in zip(episodes, targets)])
        cur, fut = cur_ref, tgt_ref
        if adapted_encoder is not None:
            tgt_imgs = np.stack([e.frames[t] for e, t in zip(episodes, targets)])
            with torch.no_grad():
                both = enc.encode(adapted_encoder, np.concatenate([imgs, tgt_imgs]))
            cur, fut = both[: len(imgs)], both[len(imgs):]
        dur = torch.full((len(q),), float(d.CHUNK))
        model.eval()
        with torch.no_grad():
            out, _ = model.forward_with_memory(cur.to(device), fut.to(device), q.to(device), dur.to(device),
                                               bank=bank, use_retrieval=use_retrieval,
                                               query_current=cur_ref.to(device) if use_retrieval else None,
                                               query_future=tgt_ref.to(device) if use_retrieval else None)
        return out
    return act


def knn_policy(bank, *, device: str):
    from ..hybrid_heads import knn_average
    from .target import KNN_SPEC

    def act(imgs, cur_ref, targets, episodes, q):
        tgt_ref = torch.stack([e.ref[t] for e, t in zip(episodes, targets)])
        dur = torch.full((len(q),), float(d.CHUNK), device=device)
        with torch.no_grad():
            return knn_average(bank, cur_ref.to(device), tgt_ref.to(device), q.to(device), dur, spec=KNN_SPEC,
                               chunk_steps=d.CHUNK)
    return act
