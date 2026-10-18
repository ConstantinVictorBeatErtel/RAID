"""Protocol v4 validation plan, simulator step contract, and pod launch refusal."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import pod_wm_session  # noqa: E402
from raid_experiments.wm_pod import (  # noqa: E402
    launch_source_panel, launch_validation, source_panel_plan, validation_plan,
)
from raid_experiments.wm_rollout import goal_bddl  # noqa: E402
from raid_experiments.wm_validate import PINNED_IMAGE_DIGEST, validate_gpu  # noqa: E402

LIBERO = Path("data/assets/libero/upstream")


class _Env:
    def __init__(self):
        self.t = 0

    def reset(self):
        return {"agentview_image": np.zeros((16, 16, 3), dtype=np.uint8)}

    def step(self, action):
        self.t += 1
        frame = np.full((16, 16, 3), self.t, dtype=np.uint8)
        return {"agentview_image": frame}, 0.0, False, {}

    def check_success(self):
        return False

    def close(self):
        pass


class _Provider:
    def __init__(self):
        self.calls = []

    def create_pod(self, **kwargs):
        self.calls.append(("create", kwargs))
        return {"id": "pod-test"}

    def install_key(self, pod_id):
        self.calls.append(("key", pod_id))

    def start_watchdog(self, pod_id, *, mode, seconds, runner_pid_file=None):
        self.calls.append(("watchdog", mode, seconds, runner_pid_file))

    def upload(self, pod_id, uploads, study_root):
        self.calls.append(("upload", uploads))

    def remote(self, pod_id, command):
        self.calls.append(("remote", command))


def test_goal_bddl_resolves_real_task():
    path = goal_bddl("put_the_cream_cheese_in_the_bowl", LIBERO)
    assert path.is_file()
    with pytest.raises(FileNotFoundError):
        goal_bddl("not_a_task", LIBERO)


def test_validation_evidence_passes_only_with_cuda_image_and_a_timed_clip(tmp_path):
    probe = {"python": [3, 12, 3], "torch": "2.9.1+cu128", "cuda_available": True,
             "gpu_name": "A40", "capability": [8, 6], "image_digest": PINNED_IMAGE_DIGEST}

    def step(task, root):
        return {"stepped": True, "frame_shape": [16, 16, 3]}

    def generate():
        return {"num_frames": 57, "native_action_output_used": False}

    evidence = validate_gpu(tmp_path, task="put_the_cream_cheese_in_the_bowl", libero_root=LIBERO,
                            probe=lambda: probe, env_step=step, generate=generate)
    assert evidence["status"] == "passed"
    costs = json.loads((tmp_path / "costs.json").read_text())
    assert costs["measured"] is True and costs["generation_seconds_per_clip"] > 0
    cpu = dict(probe, cuda_available=False, capability=None, image_digest=None)
    blocked = validate_gpu(tmp_path / "cpu", task="put_the_cream_cheese_in_the_bowl", libero_root=LIBERO,
                           probe=lambda: cpu, env_step=step, generate=generate)
    assert blocked["status"] == "blocked"
    assert json.loads((tmp_path / "cpu" / "cuda_evidence.json").read_text())["status"] == "blocked"


def test_step_contract_uses_the_settle_action(tmp_path, monkeypatch):
    from raid_experiments import wm_validate
    seen = {}

    def factory(task):
        env = _Env()
        original = env.step

        def step(action):
            seen["action"] = list(action)
            return original(action)
        env.step = step
        return env

    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero-config"))
    report = wm_validate.step_goal_task("put_the_cream_cheese_in_the_bowl", LIBERO, steps=2, env_factory=factory)
    assert report["stepped"] is True and report["frame_shape"] == [16, 16, 3]
    assert seen["action"][-1] == -1
    assert (tmp_path / "libero-config" / "config.yaml").is_file()


def test_plan_pins_image_and_does_not_train():
    plan = validation_plan()
    assert plan["image_digest"] == PINNED_IMAGE_DIGEST
    assert plan["image"].endswith(PINNED_IMAGE_DIGEST)
    assert plan["starts_training"] is False
    assert "requirements-wm-sim.lock" in plan["bootstrap"]
    assert "requirements-wm-cosmos.lock" in plan["bootstrap"]
    assert plan["uploads"]["data/assets/libero/upstream"].endswith("/inputs/libero_repo")
    assert "PYTHONPATH" in plan["bootstrap"] and "MUJOCO_GL=egl" in plan["bootstrap"]
    assert "inputs/libero_repo" in plan["command"]
    assert "validate-gpu" in plan["command"]
    assert "train-source" not in plan["command"]
    assert "rpa_" not in json.dumps(plan)
    assert plan["watchdogs"] == ["deadline", "exit_guard"]
    pulled = plan["pull"]
    assert "collection/cuda_evidence.json" in pulled
    assert "venv" not in pulled


def test_launch_refuses_without_authorization(tmp_path):
    provider = _Provider()
    with pytest.raises(PermissionError):
        launch_validation(tmp_path / "session", provider, authorize_spend=False)
    assert provider.calls == []
    with pytest.raises(ValueError):
        launch_validation(tmp_path / "session", provider, authorize_spend=True, cap_usd=50)
    assert provider.calls == []


def test_authorized_launch_uses_the_fake_provider_and_both_watchdogs(tmp_path):
    provider = _Provider()
    record = launch_validation(tmp_path / "session", provider, authorize_spend=True)
    kinds = [call[0] for call in provider.calls]
    assert kinds[0] == "create"
    assert provider.calls[0][1]["image"].endswith(PINNED_IMAGE_DIGEST)
    assert "rpa_" not in json.dumps(provider.calls)
    assert ("watchdog", "deadline", int(1.5 * 3600)) in [(c[0], c[1], c[2]) for c in provider.calls if c[0] == "watchdog"]
    assert ("watchdog", "exit_guard", 1800) in [(c[0], c[1], c[2]) for c in provider.calls if c[0] == "watchdog"]
    assert record["plan"]["starts_training"] is False
    assert record["scientific_result"] is False


def test_session_cli_does_not_launch(monkeypatch):
    monkeypatch.setattr(pod_wm_session, "launch_validation", lambda *a, **k: pytest.fail("launch must not run"))
    assert pod_wm_session.main(["launch"]) == 2
    assert pod_wm_session.main(["plan"]) == 0


def test_source_panel_plan_has_egl_replay_panel_no_training():
    plan = source_panel_plan()
    blob = json.dumps(plan) + plan["bootstrap"] + plan["command"]
    assert "libegl1" in plan["bootstrap"]
    assert "egl_ok" in plan["bootstrap"] or "GLContext" in plan["bootstrap"]
    assert "replay-experts" in plan["command"]
    assert "source-panel" in plan["command"]
    assert "validate-gpu" in plan["command"]
    assert "train-source" not in plan["command"]
    assert "adapt" not in plan["command"]
    assert "rpa_" not in blob
    assert plan["starts_training"] is False
    assert plan["cap_usd"] == 10.0
    assert "collection/source_panel" in plan["pull"]
    assert "collection/costs.json" in plan["pull"]
    assert any("protocol_v6.json" in k for k in plan["uploads"])


def test_launch_source_panel_refuses_without_authorization_or_above_cap(tmp_path):
    provider = _Provider()
    with pytest.raises(PermissionError):
        launch_source_panel(tmp_path / "session", provider, authorize_spend=False)
    assert provider.calls == []
    with pytest.raises(ValueError, match="ceiling"):
        launch_source_panel(tmp_path / "session", provider, authorize_spend=True, cap_usd=50)
    assert provider.calls == []


def test_authorized_source_panel_launch_uses_fake_provider(tmp_path):
    provider = _Provider()
    record = launch_source_panel(tmp_path / "session", provider, authorize_spend=True)
    assert record["mode"] == "source_panel"
    assert record["plan"]["starts_training"] is False
    assert not any(c[0] in {"key", "watchdog"} for c in provider.calls)
    assert record["status"] == "running"
    assert (tmp_path / "session" / "session.json").is_file()
    remotes = [c[1] for c in provider.calls if c[0] == "remote"]
    assert any("source-panel" in r for r in remotes)
    assert "rpa_" not in json.dumps(provider.calls)


def test_source_launch_journals_intent_before_create_and_owner_immediately(tmp_path):
    session = tmp_path / "session"

    class JournalProvider(_Provider):
        def create_pod(self, **kwargs):
            assert json.loads((session / "session.json").read_text())["status"] == "launch_intent"
            return super().create_pod(**kwargs)

        def install_key(self, pod_id):
            record = json.loads((session / "session.json").read_text())
            assert record["pod_id"] == pod_id
            assert record["status"] == "created"
            return super().install_key(pod_id)

    launch_source_panel(session, JournalProvider(), authorize_spend=True)


def test_session_cli_source_panel_plan_and_refuse(monkeypatch):
    monkeypatch.setattr(pod_wm_session, "launch_source_panel",
                        lambda *a, **k: pytest.fail("launch must not run"))
    assert pod_wm_session.main(["plan", "--mode", "source-panel"]) == 0
    assert pod_wm_session.main(["launch", "--mode", "source-panel"]) == 2


def test_runpod_provider_registers_deadline_with_independent_service(monkeypatch):
    import pod_provider
    provider = pod_provider.RunpodProvider()
    seen = {}
    def cutoff(path, *, method="GET", payload=None):
        seen.update(path=path, method=method, payload=payload)
        return {"podId": "p1", "volumeId": "v1"}
    monkeypatch.setattr(provider, "_cutoff", cutoff)
    snapshot = {"status": "passed", "gpu": "NVIDIA A100 80GB PCIe",
                "gpu_hourly_usd": "2", "data_center_id": "EU-RO-1",
                "projected_source_cost_usd": "2.1", "provider_billing_digest": "b" * 64}
    monkeypatch.setattr(provider, "source_launch_admission", lambda **kwargs: snapshot)
    owned = []
    row = provider.create_pod(name="x", image="pin", gpu="NVIDIA A100 80GB PCIe", env={}, max_hours=1,
                              stage="source_panel", session_id="acceptance-123",
                              package_sha256="a" * 64, cap_usd=8, max_hourly_usd=2,
                              data_center_id="EU-RO-1",
                              on_created=lambda pod: owned.append(pod["id"]))
    assert row["id"] == "p1"
    assert row["raid_termination_verified"] is False
    assert owned == ["p1"]
    assert seen["path"] == "/sessions/acceptance-123/launch"
    assert seen["payload"]["stage"] == "source_panel"
    assert seen["payload"]["podSpec"]["dataCenterIds"] == ["EU-RO-1"]
    assert seen["payload"]["volumeSpec"]["dataCenterId"] == "EU-RO-1"
    assert seen["payload"]["spendAdmission"] == snapshot
    assert row["volume_id"] == "v1"
    assert "terminateAfter" not in seen["payload"]["podSpec"]


def test_live_runpod_launch_stays_blocked_without_strict_cutoff_and_recovery_acceptance(tmp_path):
    import pod_provider
    provider = pod_provider.RunpodProvider()
    with pytest.raises(RuntimeError, match="live independent cutoff/recovery acceptance"):
        launch_source_panel(tmp_path / "session", provider, authorize_spend=True)


def test_source_collection_requires_hash_manifest_and_done(tmp_path):
    import hashlib
    from raid_experiments.wm_pod import verify_source_panel_collection
    collection = tmp_path / "collection"
    collection.mkdir()
    item = collection / "result.json"
    item.write_text('{"ok": true}')
    digest = hashlib.sha256(item.read_bytes()).hexdigest()
    (collection / "outputs_sha256.json").write_text(json.dumps({"result.json": digest}))
    (collection / "DONE").touch()
    assert verify_source_panel_collection(collection)["status"] == "verified"
    item.write_text("changed")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_source_panel_collection(collection)
