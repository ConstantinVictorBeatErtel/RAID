"""VPP2 generator adapter: geometry, prompting, protocol amendment and timing (CPU only)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from raid_experiments import hybrid_vpp2 as v
from raid_experiments.hybrid_protocol import (VPP2TemporalAlignment, is_vpp2, temporal_alignment,
                                              validate_hybrid_protocol, with_vpp2_timing)

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "artifacts/raid_hybrid_libero_goal_v6/explicit_prompts_vpp2_20261013/vpp2_source/src/vpp2/utils/conditioning.py"
PARENT = ROOT / "artifacts/raid_hybrid_libero_goal_v6/hf_launcher_fix_20261012/collection/protocol_v6_pod.json"


def _frame(seed=0, size=128):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def _upstream():
    spec = importlib.util.spec_from_file_location("vpp2_conditioning", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canvas_matches_upstream_prepare_image_exactly():
    from PIL import Image
    upstream = _upstream()
    for size in (128, 256, 480):
        upright = _frame(size, size)
        ours, meta = v.compose_canvas(upright)
        theirs, info = upstream.prepare_image([Image.fromarray(upright)], "robot")
        assert np.array_equal(ours, np.asarray(theirs))
        assert meta["main_crop_box_xyxy"] == list(info["crop_boxes_xyxy"][0])
        assert ours.shape == (240, 416, 3)


def test_wrist_panels_are_black_and_never_reach_the_controller():
    canvas, _ = v.compose_canvas(np.full((128, 128, 3), 200, np.uint8))
    assert canvas[:, 310:].max() == 0  # right panels stay black after resize
    main = v.extract_main_view(np.repeat(canvas[None], 3, axis=0))
    assert main.shape == (3, 240, 320, 3)
    # Away from the resampled boundary, the main view is the uniform scene, not black.
    assert main[:, :, :315].min() >= 190


def test_band_transform_roundtrip_and_orientation():
    sim = _frame(3)
    band = v.controller_band(sim)
    assert band.shape == (96, 128, 3)
    upright = sim[::-1]
    assert np.array_equal(band[::-1], upright[16:112])
    canvas, _ = v.compose_canvas(upright)
    main = v.extract_main_view(canvas[None])
    back = v.generated_to_controller(main, (96, 128))[0]
    # Bicubic down/up resampling: close but not exact.
    assert np.abs(back.astype(int) - band.astype(int)).mean() < 40
    # A marked upright top band row must land at the top of the upright main view.
    marked = np.zeros((128, 128, 3), np.uint8); marked[16:24] = 255
    m = v.extract_main_view(v.compose_canvas(marked)[0][None])[0]
    assert m[:10, 50:250].mean() > 200 and m[-10:, 50:250].mean() < 20


def test_prompt_policy_parses_caches_and_falls_back(tmp_path):
    good = json.dumps({k: f"The arm does the {k} step near the bowl" for k in v.PROMPT_KEYS})
    calls = []

    def backend(image, system, user, max_new):
        calls.append(user)
        return (good, True) if len(calls) > 1 else ("not json", True)

    policy = v.ObservationPromptPolicy(backend, tmp_path)
    image = v.prompt_image(_frame(1)[::-1])
    record = policy(image, "put the bowl on the stove")
    assert not record.fallback and len(record.attempts) == 2 and "Correction" in calls[1]
    assert record.prompt.endswith(v.VIEW_SUFFIX)
    again = policy(image, "put the bowl on the stove")
    assert again.cache_hit and again.prompt == record.prompt and len(calls) == 2
    failing = v.ObservationPromptPolicy(lambda *a: ("{}", True), None)
    fb = failing(image, "turn on the stove")
    assert fb.fallback and fb.prompt == v.fallback_prompt("turn on the stove") and len(fb.attempts) == 3
    with pytest.raises(TypeError):
        policy(image, "x", future_frame=image)
    with pytest.raises(ValueError):
        policy(_frame(1), "x")


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def __call__(self, *, image, prompt, seed):
        self.calls += 1
        assert image.shape == (240, 416, 3)
        return np.repeat(image[None], 49, axis=0)


def test_world_model_returns_controller_band_and_caches(tmp_path):
    backend = FakeBackend()
    prompts = v.ObservationPromptPolicy(
        lambda *a: (json.dumps({k: f"The arm completes the {k} part" for k in v.PROMPT_KEYS}), True), tmp_path / "p")
    wm = v.VPP2WorldModel(backend, prompts, tmp_path / "c")
    obs = _frame(5)
    plan = wm.generate(obs, "open the middle drawer of the cabinet", seed=7)
    assert plan.frames.shape == (49, 96, 128, 3) and plan.display_frames.shape == (49, 240, 320, 3)
    assert np.abs(plan.frames[0].astype(int) - v.controller_band(obs).astype(int)).mean() < 40
    again = wm.generate(obs, "open the middle drawer of the cabinet", seed=7)
    assert again.cache_hit and backend.calls == 1 and again.cache_key == plan.cache_key
    other = wm.generate(obs, "open the middle drawer of the cabinet", seed=8)
    assert not other.cache_hit and backend.calls == 2
    with pytest.raises(TypeError):
        wm.generate(obs, "x", seed=1, recorded_actions=[0])
    entry = tmp_path / "c" / plan.cache_key[:2] / plan.cache_key / "frames.npz"
    np.savez_compressed(entry, canvas_frames=np.zeros((49, 240, 416, 3), np.uint8))
    with pytest.raises(ValueError, match="corrupt"):
        wm.generate(obs, "open the middle drawer of the cabinet", seed=7)


def test_rejects_wrong_generator_output_shape(tmp_path):
    wm = v.VPP2WorldModel(lambda **k: np.zeros((48, 240, 416, 3), np.uint8),
                          v.ObservationPromptPolicy(lambda *a: ("{}", True)), tmp_path)
    with pytest.raises(ValueError, match="expected"):
        wm.generate(_frame(), "turn on the stove", seed=1)


@pytest.mark.skipif(not PARENT.is_file(), reason="parent protocol missing")
def test_amendment_preserves_study_and_requires_calibration():
    parent = json.loads(PARENT.read_text())
    amended = v.amend_protocol_vpp2(parent)
    assert is_vpp2(amended) and amended["digest"] != parent["digest"]
    for key in ("conditions", "retrieval", "head", "training", "stage_scope", "state_schema"):
        assert amended["design"][key] == parent["design"][key]
    assert amended["tasks"] == parent["tasks"]
    temporal = temporal_alignment(amended)
    assert isinstance(temporal, VPP2TemporalAlignment) and temporal.chunk_steps == 8
    with pytest.raises(RuntimeError, match="uncalibrated"):
        temporal.subgoal_frames()
    stamped = with_vpp2_timing(amended, "a" * 64, steps_per_frame=2.1)
    assert stamped["model_digest"] == amended["model_digest"] and stamped["digest"] != amended["digest"]
    t = temporal_alignment(stamped)
    assert t.requested_frame == 4 and t.subgoal_frames() == [4]
    t.require_calibrated()
    with pytest.raises(ValueError):
        v.amend_protocol_vpp2(amended)
    with pytest.raises(ValueError):
        with_vpp2_timing(amended, "a" * 64, steps_per_frame=0.1)  # frame 80 is outside the clip
    broken = json.loads(json.dumps(stamped)); broken["design"]["temporal"]["requested_frame"] = 6
    with pytest.raises(ValueError):
        validate_hybrid_protocol(broken)
    assert "gr1" not in json.dumps(amended).lower()


@pytest.mark.skipif(not PARENT.is_file(), reason="parent protocol missing")
def test_localhost_service_roundtrip_matches_in_process(tmp_path):
    import socket
    import threading
    import time as _time
    protocol = v.amend_protocol_vpp2(json.loads(PARENT.read_text()))
    backend = FakeBackend()
    prompts = v.ObservationPromptPolicy(
        lambda *a: (json.dumps({k: f"The arm completes the {k} part" for k in v.PROMPT_KEYS}), True), tmp_path / "p")
    world = v.VPP2WorldModel(backend, prompts, tmp_path / "clips")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    ready = tmp_path / "ready.json"
    threading.Thread(target=v.serve, args=(world,), kwargs=dict(port=port, protocol_digest=protocol["digest"],
                                                                 ready_path=ready), daemon=True).start()
    for _ in range(100):
        if ready.exists():
            break
        _time.sleep(0.05)
    client = v.VPP2ServiceClient(f"http://127.0.0.1:{port}", tmp_path / "clips", protocol=protocol)
    client.check()
    obs = _frame(9)
    remote = client.generate(obs, "turn on the stove", seed=3)
    local = world.generate(obs, "turn on the stove", seed=3)
    assert local.cache_hit and remote.cache_key == local.cache_key and backend.calls == 1
    assert np.array_equal(remote.frames, local.frames)
    with pytest.raises(TypeError):
        client.generate(obs, "turn on the stove", seed=3, future=obs)
    wrong = json.loads(json.dumps(protocol)); wrong["digest"] = "0" * 64
    with pytest.raises(RuntimeError, match="does not serve"):
        v.VPP2ServiceClient(f"http://127.0.0.1:{port}", tmp_path / "clips", protocol=wrong).check()


def test_openrouter_requests_are_pinned_and_deterministic(tmp_path):
    image = v.prompt_image(_frame(2)[::-1])
    bodies = [v.OpenRouterPromptBackend(c, tmp_path / "key").request_body(image, v.PROMPT_SYSTEM, "u", 512)
              for c in v.PROMPT_MODELS]
    assert [b["model"] for b in bodies] == ["z-ai/glm-5.3-flashx", "deepseek/deepseek-v4.1-flash"]
    assert [b["provider"]["only"] for b in bodies] == [["z-ai"], ["alibaba"]]
    for body in bodies:
        assert body["provider"]["allow_fallbacks"] is False and body["provider"]["require_parameters"] is True
        assert body["reasoning"] == {"effort": "high"} and body["temperature"] == 0.0
        assert body["messages"][0]["content"] == v.PROMPT_SYSTEM
        assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "seed" not in bodies[0] and bodies[1]["seed"] == 42


def test_routed_prompt_split_is_deterministic(tmp_path):
    made = []
    def backend(tag):
        return lambda *a: (made.append(tag) or json.dumps({k: f"The {tag} arm does the {k} step" for k in v.PROMPT_KEYS}), True)
    routed = v.RoutedPromptPolicy([v.ObservationPromptPolicy(backend("glm"), None),
                                   v.ObservationPromptPolicy(backend("deepseek"), None)])
    wm = v.VPP2WorldModel(FakeBackend(), routed, tmp_path)
    assert wm.prompt_slots == 2
    a = wm.generate(_frame(4), "turn on the stove", seed=1, prompt_slot=0)
    b = wm.generate(_frame(4), "turn on the stove", seed=1, prompt_slot=1)
    assert "glm" in a.prompt and "deepseek" in b.prompt and a.cache_key != b.cache_key and made == ["glm", "deepseek"]
    with pytest.raises(ValueError):
        wm.generate(_frame(4), "turn on the stove", seed=1, prompt_slot=2)


def test_timing_alignment_recovers_known_stride():
    from raid_experiments.hybrid_vpp2_panel import align_steps_per_frame, local_offset
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=(140, 32)), axis=0) + 50.0  # smooth trajectory in feature space
    generated = walk[::2][:49]                                    # 2 recorded steps per generated frame
    fit = align_steps_per_frame(generated, walk)
    assert abs(fit["steps_per_frame"] - 2.0) < 0.15 and fit["plausible"]
    assert local_offset(generated[4], walk, 32) == 8


@pytest.mark.skipif(not PARENT.is_file(), reason="parent protocol missing")
def test_dependency_free_panel_selection_matches_registered_panel():
    from raid_experiments.hybrid_panel import select_panel_cases
    protocol = v.amend_protocol_vpp2(json.loads(PARENT.read_text()))
    assert v.vpp2_panel_cases(protocol) == select_panel_cases(protocol)[::2]
