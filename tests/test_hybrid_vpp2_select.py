import json
from pathlib import Path

import numpy as np
import pytest

from raid_experiments import hybrid_vpp2 as v
from raid_experiments import hybrid_vpp2_select as s

PROTOCOL = Path(__file__).resolve().parents[1] / "artifacts/raid_hybrid_libero_goal_v6/vpp2_panel_20261013/protocol_v6_vpp2.json"


def prompt_json(tag="A"):
    return json.dumps({k: f"The gripper {tag} moves near the stove for {k}" for k in s.PROMPT_KEYS_V5})


class FakePrompt:
    def __init__(self, tag, fail=False):
        self.tag, self.fail, self.log, self.calls = tag, fail, [], []

    def __call__(self, image, system, user, max_new_tokens):
        self.calls.append((image.shape, system, user))
        self.log.append({"usage": {"cost": 0.001}})
        return ("not json", True) if self.fail else (prompt_json(self.tag), True)


class FakeJudge:
    def __init__(self, prompt_verdict=None, video_verdict=None):
        self.prompt_verdict = prompt_verdict or {"score_A": 8, "score_B": 5, "choice": "A", "reason": "A is right."}
        self.video_verdict = video_verdict or {"score_A": 3, "score_B": 7, "acceptable_A": False,
                                               "acceptable_B": True, "choice": "B", "reason": "A blurs."}
        self.log, self.contents = [], []

    def __call__(self, system, content):
        self.contents.append((system, content))
        self.log.append({"usage": {"cost": 0.002}})
        verdict = self.prompt_verdict if system == s.PROMPT_JUDGE_SYSTEM else self.video_verdict
        return (verdict if isinstance(verdict, str) else json.dumps(verdict)), True


def fake_video(*, image, prompt, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(49, 240, 416, 3), dtype=np.uint8)


def observation():
    return np.random.default_rng(0).integers(0, 255, size=(128, 128, 3), dtype=np.uint8)


def test_parse_prompt_v5_requires_positions_and_current_state():
    assert set(s.parse_prompt_v5(prompt_json())) == set(s.PROMPT_KEYS_V5)
    bad = json.loads(prompt_json())
    bad.pop("positions")
    with pytest.raises(ValueError):
        s.parse_prompt_v5(json.dumps(bad))


def test_video_verdict_consistency():
    ok = {"score_A": 6, "score_B": 4, "acceptable_A": True, "acceptable_B": True, "choice": "A", "reason": "x"}
    assert s.parse_video_verdict(json.dumps(ok))["choice"] == "A"
    for change in ({"choice": "B"}, {"choice": "none"}, {"acceptable_A": False}, {"score_A": 11}):
        with pytest.raises(ValueError):
            s.parse_video_verdict(json.dumps({**ok, **change}))
    none = {**ok, "acceptable_A": False, "acceptable_B": False, "choice": "none"}
    assert s.parse_video_verdict(json.dumps(none))["choice"] == "none"


def test_selector_follows_judge_and_records_everything():
    prompts, judge = [FakePrompt("glm"), FakePrompt("ds")], FakeJudge()
    selector = s.JudgedSelector(prompts, judge, fake_video)
    record, videos = selector.plan(observation(), "turn on the stove", seeds=(1, 2))
    order, vorder = record["prompt_judge"]["order"], record["video_judge"]["order"]
    assert record["selected_prompt_index"] == order[0]
    assert record["selected_video_index"] == vorder[1]
    assert record["selection_complete"] and not record["none_acceptable"]
    assert len(videos) == 2 and len(record["prompt_candidates"]) == 2
    assert record["cost_usd"]["judge"] == pytest.approx(0.004)
    # Judge input: the current image, instruction, prompts and generated frames only.
    video_images = [p for p in judge.contents[1][1] if p["type"] == "image_url"]
    assert len(video_images) == 1 + 2 * len(s.JUDGE_FRAMES)
    assert v.VIEW_SUFFIX not in json.dumps(judge.contents)


def test_none_acceptable_uses_higher_score_and_judge_failure_is_technical():
    none = {"score_A": 2, "score_B": 4, "acceptable_A": False, "acceptable_B": False, "choice": "none", "reason": "x"}
    record, _ = s.JudgedSelector([FakePrompt("a"), FakePrompt("b")], FakeJudge(video_verdict=none),
                                 fake_video).plan(observation(), "x", seeds=(1, 2))
    assert record["none_acceptable"] and record["selected_video_index"] == record["video_judge"]["order"][1]
    assert record["selection_complete"]
    record, _ = s.JudgedSelector([FakePrompt("a"), FakePrompt("b")], FakeJudge(video_verdict="garbage"),
                                 fake_video).plan(observation(), "x", seeds=(1, 2))
    assert not record["selection_complete"] and record["selected_video_index"] == 0
    assert len(record["video_judge"]["attempts"]) == s.JUDGE_ATTEMPTS


def test_one_failed_prompt_model_skips_judge_and_fails_technically():
    judge = FakeJudge()
    record, _ = s.JudgedSelector([FakePrompt("a", fail=True), FakePrompt("b")], judge,
                                 fake_video).plan(observation(), "x", seeds=(1, 2))
    assert record["selected_prompt_index"] == 1 and not record["selection_complete"]
    assert record["prompt_judge"]["skipped"] == "one valid prompt"


def test_selector_rejects_privileged_inputs_and_equal_seeds():
    selector = s.JudgedSelector([FakePrompt("a"), FakePrompt("b")], FakeJudge(), fake_video)
    with pytest.raises(TypeError):
        selector.plan(observation(), "x", seeds=(1, 2), recorded_frames=np.zeros(1))
    with pytest.raises(ValueError):
        selector.plan(observation(), "x", seeds=(3, 3))


@pytest.mark.skipif(not PROTOCOL.is_file(), reason="registered VPP2 protocol not present")
def test_amendment_binds_protocol_and_policy(tmp_path):
    protocol = json.loads(PROTOCOL.read_text())
    body = s.write_amendment(protocol, tmp_path / "a.json")
    s.verify_amendment(protocol, body)
    with pytest.raises(FileExistsError):
        s.write_amendment(protocol, tmp_path / "a.json")
    with pytest.raises(ValueError):
        s.verify_amendment(protocol, {**body, "gates_unchanged": {}})
    seeds = s.candidate_seeds(protocol, "panel:turn_on_the_stove:10:57", 0)
    assert seeds[0] == v.generation_seed_for(protocol, "panel:turn_on_the_stove:10:57", 0) != seeds[1]
