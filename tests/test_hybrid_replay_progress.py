import json

import numpy as np
import pytest

from raid_experiments import hybrid_local_checks as checks


def test_reset_retries_only_placement_errors():
    class PlacementError(Exception):
        pass

    calls = []

    def reset():
        calls.append(1)
        if len(calls) < 3:
            raise PlacementError("placement")
        return {"ready": True}

    assert checks._bounded_reset(reset, PlacementError) == {"ready": True}
    assert len(calls) == 3
    with pytest.raises(ValueError, match="unexpected"):
        checks._bounded_reset(lambda: (_ for _ in ()).throw(ValueError("unexpected")), PlacementError)


def test_reset_exhaustion_preserves_exception():
    calls = []

    def reset():
        calls.append(1)
        raise RuntimeError("placement failed")

    with pytest.raises(RuntimeError, match="placement failed"):
        checks._bounded_reset(reset, RuntimeError, attempts=3)
    assert len(calls) == 3


@pytest.mark.parametrize("limits,expected", [((1024, 524288), (65536, 524288)),
                                           ((1024, 4096), (4096, 4096)),
                                           ((131072, 524288), (131072, 524288))])
def test_worker_file_limit_respects_hard_limit(monkeypatch, limits, expected):
    import resource
    import sys
    from raid_experiments.source_worker import configure_file_limit
    monkeypatch.setattr(sys, "platform", "linux")
    current = [limits]
    monkeypatch.setattr(resource, "getrlimit", lambda _: current[0])
    monkeypatch.setattr(resource, "setrlimit", lambda _, value: current.__setitem__(0, value))
    record = configure_file_limit()
    assert record["before"] == list(limits)
    assert tuple(record["after"]) == expected


def test_replay_records_errors_and_progress_without_marking_completion(monkeypatch, tmp_path):
    class Env:
        def reset(self):
            raise ValueError("simulator reset failed")

        def close(self):
            pass

    monkeypatch.setattr(checks, "_read_actions", lambda *args: np.zeros((2, 7)))
    monkeypatch.setattr(checks, "_read_initial_state", lambda *args: np.zeros(2))
    monkeypatch.setattr(checks, "_make_goal_env", lambda *args: Env())
    protocol = {"tasks": {"source_task": {"role": "source", "path": "fake",
                  "source": {"source_validation": [1, 2]}}}}
    progress = tmp_path / "progress.json"
    result = checks.replay_source_panel(protocol, "fake", cases=2, progress_path=progress)
    assert result["n"] == 2
    assert not result["gate_pass"]
    assert all("simulator reset failed" in row["outcome"] for row in result["cases"])
    record = json.loads(progress.read_text())
    assert record["completed_cases"] == 2
    assert record["cases"] == result["cases"]
    assert record["complete"] is False
    assert record["scientific_result"] is False
