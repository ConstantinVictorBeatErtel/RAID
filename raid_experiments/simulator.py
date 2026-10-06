"""Corrected LIBERO control adapter for predicted-future evaluation.

This module deliberately reports only the simulator's official success
predicate. It does not expose shaped reward as a success proxy.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


def make_env(task_index: int, *, libero_root: str | Path, camera_size: int = 128):
    from libero.libero.envs import OffScreenRenderEnv
    bddl_dir = Path(libero_root) / "libero" / "libero" / "bddl_files" / "libero_spatial"
    files = sorted(bddl_dir.glob("*.bddl"))
    if not files:
        raise FileNotFoundError(f"No LIBERO-Spatial BDDL files in {bddl_dir}")
    if task_index >= len(files):
        raise IndexError(f"task_index {task_index} exceeds {len(files)} task files")
    return OffScreenRenderEnv(bddl_file_name=str(files[task_index]), camera_heights=camera_size, camera_widths=camera_size)


def _frame(obs):
    for key in ("agentview_rgb", "agentview_image", "image"):
        if isinstance(obs, dict) and key in obs:
            return np.asarray(obs[key], dtype=np.uint8)
    raise KeyError("LIBERO observation has no agent-view RGB image")


def _hand_frame(obs):
    for key in ("robot0_eye_in_hand", "robot0_wristview_image", "robot0_eye_in_hand_image", "gripperview_image"):
        if isinstance(obs, dict) and key in obs:
            return np.asarray(obs[key], dtype=np.uint8)
    raise KeyError("LIBERO observation has no wrist RGB image required by this controller")


def _state(obs):
    """Map LIBERO robot state to the study action state explicitly.

    The mapping uses position, a three-dimensional rotational state, and one
    gripper value. LIBERO exposes a quaternion and two gripper joint values;
    convert the quaternion to axis-angle and use the mean joint position.
    Missing live state is an integration error, never a silent zero state.
    """
    if not isinstance(obs, dict):
        raise KeyError("LIBERO observation is not a mapping")
    pos_key = "robot0_eef_pos" if "robot0_eef_pos" in obs else "ee_pos"
    quat_key = "robot0_eef_quat" if "robot0_eef_quat" in obs else None
    grip_key = "robot0_gripper_qpos" if "robot0_gripper_qpos" in obs else "gripper_states"
    if quat_key is None or pos_key not in obs or grip_key not in obs:
        raise KeyError(f"Missing required LIBERO robot state fields; available={sorted(obs.keys())}")
    pos = np.asarray(obs[pos_key], dtype=np.float32).ravel()[:3]
    quat = np.asarray(obs[quat_key], dtype=np.float32).ravel()[:4]
    if len(pos) != 3 or len(quat) != 4:
        raise ValueError("Unexpected LIBERO end-effector state shape")
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    w = np.clip(quat[3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    scale = angle / max(float(np.sqrt(max(1.0 - w * w, 0.0))), 1e-8)
    axis_angle = quat[:3] * scale if angle > 1e-6 else np.zeros(3, dtype=np.float32)
    grip_value = float(np.mean(np.asarray(obs[grip_key], dtype=np.float32)))
    if not np.isfinite(grip_value):
        raise ValueError("LIBERO gripper state is not finite")
    grip = 1.0 if grip_value > 0.0 else -1.0
    return np.concatenate((pos, axis_angle, np.asarray([grip], dtype=np.float32))).astype(np.float32)


def run_episode(env, *, encoder, policy, memory, norm_mean, norm_std, language, device,
                initial_state=None, max_steps=600, settle_steps=5, k=3):
    """Run one deterministic episode and return per-step records plus success."""
    device = torch.device(device)
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset
    if initial_state is not None:
        if not hasattr(env, "set_init_state"):
            raise RuntimeError("Configured initial state cannot be applied by this LIBERO environment")
        reset = env.set_init_state(initial_state)
        if reset is not None:
            obs = reset[0] if isinstance(reset, tuple) else reset
        elif hasattr(env, "get_observation"):
            obs = env.get_observation()
        else:
            raise RuntimeError("LIBERO set_init_state returned no observation")
    for _ in range(settle_steps):
        settled = env.step(np.zeros(7, dtype=np.float32))
        obs = settled[0]
    records = []
    history_frames = []
    history_hand_frames = []
    history_states = []
    for _ in range(max_steps):
        frame = torch.from_numpy(_frame(obs)).unsqueeze(0)
        hand_frame = torch.from_numpy(_hand_frame(obs)).unsqueeze(0)
        state = torch.from_numpy(_state(obs)).unsqueeze(0)
        history_frames.append(frame.squeeze(0))
        history_hand_frames.append(hand_frame.squeeze(0))
        history_states.append(state.squeeze(0))
        history_frames = history_frames[-10:]
        history_hand_frames = history_hand_frames[-10:]
        history_states = history_states[-10:]
        history_image_tensor = torch.stack(history_frames).unsqueeze(0)
        history_hand_tensor = torch.stack(history_hand_frames).unsqueeze(0)
        history_state_tensor = torch.stack(history_states).unsqueeze(0).to(device)
        with torch.no_grad():
            current = encoder.encode_frames(frame)
            future = encoder.predict_next_feat(frame, language=[language], robot_state=state,
                                               history_images=history_image_tensor,
                                               history_hand_images=history_hand_tensor,
                                               history_state=history_state_tensor)
            neighbours = memory.retrieve(current, future, k=k)
            prediction, _ = policy(current, future, neighbours.actions.to(device), neighbours.valid.to(device))
            action = prediction.squeeze(0) * torch.as_tensor(norm_std, device=device) + torch.as_tensor(norm_mean, device=device)
        step = env.step(action.detach().cpu().numpy())
        obs, _, done, info = step[:4]
        success = bool(env.check_success()) if hasattr(env, "check_success") else bool(info.get("success", False))
        records.append({"action": action.detach().cpu().tolist(), "success": success})
        if done or success:
            return records, success
    return records, False
