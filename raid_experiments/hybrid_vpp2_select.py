"""VPP2 prompt/selection policy v5: two prompts, a judge, two videos, a judge.

User-approved on 14 October 2026. Per plan:

1. Two pinned prompt models (GLM, DeepSeek) each write a v5 prompt from the
   current main-band image and the instruction. v5 requires an explicit
   "positions" field (gripper, target and destination relative to the fixed
   scene objects) and a "current_state" field, with the fixed SCENE_OBJECTS list.
2. A pinned judge model selects the better prompt (image, instruction, prompts).
3. VPP2 generates two videos from the selected prompt with two explicit seeds.
4. The judge sees the start image, the instruction, the prompt and sampled
   frames of both videos, and selects the better video or marks "none".

The judge never sees recorded demonstrations, future frames, simulator state or
outcomes; ``JudgedSelector.plan`` accepts only the current observation and the
instruction. Every candidate, score, reason and API call is recorded. The v3
policy in ``hybrid_vpp2`` stays unchanged as history. The same procedure must
run identically for all four pilot conditions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from typing import Callable

import numpy as np

from . import hybrid_vpp2 as v

PROMPT_POLICY_V5 = "raid_vpp2_observation_prompt_v5_positions"
SELECTION_POLICY = "raid_vpp2_judged_selection_v5"
AMENDMENT_NAME = "vpp2_prompt_selection_v5"

# Unchanged from the user-reviewed v4 scene-list candidate (scene_prompt_test_20261013/pod_scene_test.py).
SCENE_OBJECTS = """The scene is the same fixed tabletop in every task; exact positions vary slightly, so check the image:
- one white-and-gray Franka robot arm with a black two-finger gripper; its black base is at the back centre;
- a black wooden cabinet on the left side with three stacked drawers; the drawer fronts and short gray handles face right;
- a tan slatted wooden wine rack at the back left, behind the cabinet;
- one dark green wine bottle with a narrow neck, normally standing upright just right of the cabinet;
- one gray-and-white patterned bowl (it can look crumpled or metallic), normally near the table centre. This is "the bowl";
- one white plate with a pink-red rim, normally in front of the bowl (lower in the image). This is "the plate";
- a white square stove at the upper right with one dark round burner; its small black control knob stands at the
  back edge of the stove (above the burner in the image). When the stove is on, the burner glows red;
- one small blue rectangular box, normally right of the bowl and below the stove.
No other objects exist: there is no lid, scale, phone, sponge, tray, foil, second plate or second bowl."""

PROMPT_SYSTEM_V5 = """You write the text prompt for an image-to-video robot model.
The image is the CURRENT frame from a fixed camera. """ + SCENE_OBJECTS + """
Name objects only with the names in this list. Use only the instruction and what is visible now.
Never use numbers, coordinates, invented objects, a second arm, tools, camera motion or scene cuts.
First study the current state carefully: where the gripper is relative to the listed objects, whether it is open or
closed, whether it already holds an object (which one), and whether the task is partly done (an object already moved,
lifted or nearly at its destination, a drawer partly open). Small objects are hard to see: look closely at the gripper,
the target and the destination before you decide. Then describe only the REMAINING motion from this exact state; never
restart or repeat steps that are already done.
Image directions: left and right as seen in the image; "back" is the top of the image, "front" is the bottom.
Return ONLY a JSON object with these string keys, in this order:
"current_state": one or two sentences stating exactly where the gripper is now, whether it is open or closed, what it
holds, and what part of the task is already done;
"positions": two or three sentences that state the position of the gripper, the position of the target object (or
control), and the position of the destination (the place or state the target must reach), each relative to the fixed
listed objects (cabinet, wine rack, stove, robot base, table centre) with left, right, back or front;
"approach": one sentence on how the gripper moves from its current position toward the contact point
(if it already holds the target, how it carries it on);
"contact": one sentence on the grasp, press or hook (if already grasped, say it keeps holding);
"manipulation": one sentence on lifting, carrying, pulling, pushing or turning;
"completion": one sentence on placing or releasing and retracting while unrelated objects stay where they are;
"final_state": one sentence stating the completed goal as it would look in the image.
Keep the seven values under 200 words in total."""

PROMPT_KEYS_V5 = ("current_state", "positions", "approach", "contact", "manipulation", "completion", "final_state")
PROMPT_ATTEMPTS = 3
# The same two pinned prompt models as v3 (providers pinned, no fallbacks, reasoning high).
PROMPT_MODELS_V5 = tuple(dict(c) for c in v.PROMPT_MODELS)

# Judge: one provider pinned, no fallbacks; text-only privacy probe before any paid run.
JUDGE_MODEL = {"model": "google/gemini-3.8-flash", "providers": ["google-ai-studio"], "reasoning_effort": "high",
               "seed": 42, "temperature": 0.0}
JUDGE_ATTEMPTS = 3
JUDGE_FRAMES = (0, 8, 16, 24, 32, 40, 48)  # main-view frames of each candidate video shown to the judge
CANDIDATE_VIDEOS = 2

PROMPT_JUDGE_SYSTEM = """You are a strict judge of text prompts for an image-to-video robot model.
The image is the CURRENT frame from a fixed camera. """ + SCENE_OBJECTS + """
You receive the task instruction and two candidate prompts, A and B, written for this exact image.
Judge each candidate only against the image and the instruction:
1. current state: it reads the gripper position, open or closed state, held object and the finished parts of the
   task correctly;
2. objects and positions: it names the correct target and destination and places them, and the gripper, correctly
   relative to the fixed objects;
3. remaining motion: it describes only the remaining motion, in a physically sensible order, without repeating
   finished steps;
4. no invention: no objects, arms, tools or events that are not in the list or the image.
Give each candidate an integer score from 0 (useless) to 10 (fully correct). Choose the candidate with the higher
score; on equal scores choose the one with fewer errors in items 1 and 2.
Return ONLY a JSON object with the keys "score_A" (integer), "score_B" (integer), "choice" ("A" or "B") and
"reason" (at most 60 words that name the decisive errors)."""

VIDEO_JUDGE_SYSTEM = """You are a strict judge of generated robot videos.
Both videos start from the same CURRENT frame of a fixed camera and come from the same text prompt. """ + SCENE_OBJECTS + """
You receive the start image, the task instruction, the prompt, then seven frames of video A in time order (first to
last), then seven frames of video B in time order.
Judge each video:
1. robot: one continuous robot arm; it does not blur into a smear, split, teleport or vanish;
2. objects: each object keeps its size, shape, color and identity; no object grows, shrinks, melts, duplicates,
   appears or vanishes (except where the gripper or arm covers it);
3. task progress: the motion goes toward completing the instruction from the start state, with the correct target
   and destination;
4. scene: the camera and the fixed objects stay still.
Give each video an integer score from 0 (useless) to 10 (perfect). A video is acceptable only if it has no severe
failure in items 1, 2 or 4 and shows clear progress in item 3. Choose the better video. If neither video is
acceptable, choose "none".
Return ONLY a JSON object with the keys "score_A" (integer), "score_B" (integer), "acceptable_A" (true or false),
"acceptable_B" (true or false), "choice" ("A", "B" or "none") and "reason" (at most 80 words that name the decisive
defects)."""

RULES = {
    "seeds": "candidate 0 uses the registered seed generation_seed(case_id, replan_index, base_seed); candidate 1 uses "
             "generation_seed(case_id + '\\x1fcandidate:1', replan_index, base_seed)",
    "order": "prompt candidates shown as A/B in model order if int(image_sha256[0],16) is even, else reversed; video "
             "candidates in seed order if int(image_sha256[1],16) is even, else reversed (position-bias control)",
    "judge_inputs": "prompt judge: current main-band image, instruction, both prompts without the fixed view suffix; "
                    "video judge: same image, instruction, selected prompt and main-view frames JUDGE_FRAMES of both "
                    "videos. Never recorded demonstrations, future frames, simulator state or outcomes",
    "prompt_failure": "if one prompt model fails all attempts, the valid prompt is used without a judge call; if both "
                      "fail, the instruction-only fallback prompt is used; either case is a technical failure",
    "judge_failure": "if the judge returns no valid JSON after JUDGE_ATTEMPTS, candidate 0 is used and the plan is a "
                     "technical failure",
    "none_acceptable": "the video with the higher judge score is still used (tie: candidate 0) and the plan is "
                       "flagged none_acceptable; this is recorded, not a technical failure",
}


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def policy_identity() -> dict:
    """Everything that defines v5; frozen into the amendment before any run."""
    return {"prompt_policy": PROMPT_POLICY_V5, "selection_policy": SELECTION_POLICY,
            "prompt_system_sha256": _sha_text(PROMPT_SYSTEM_V5), "prompt_keys": list(PROMPT_KEYS_V5),
            "scene_objects_sha256": _sha_text(SCENE_OBJECTS), "view_suffix_sha256": _sha_text(v.VIEW_SUFFIX),
            "prompt_models": [dict(c) for c in PROMPT_MODELS_V5], "prompt_attempts": PROMPT_ATTEMPTS,
            "judge_model": dict(JUDGE_MODEL), "judge_attempts": JUDGE_ATTEMPTS,
            "prompt_judge_system_sha256": _sha_text(PROMPT_JUDGE_SYSTEM),
            "video_judge_system_sha256": _sha_text(VIDEO_JUDGE_SYSTEM),
            "judge_frames": list(JUDGE_FRAMES), "candidate_videos": CANDIDATE_VIDEOS,
            "prompt_image_size": list(v.PROMPT_IMAGE_SIZE), "rules": dict(RULES)}


def policy_digest() -> str:
    return hashlib.sha256(json.dumps(policy_identity(), sort_keys=True).encode()).hexdigest()


def build_amendment(protocol: dict, *, date: str = "2026-10-14") -> dict:
    """Versioned amendment on top of the VPP2 protocol; the protocol file itself is not changed."""
    if not v.is_vpp2(protocol):
        raise ValueError("the v5 selection amendment needs a VPP2-amended protocol")
    body = {
        "name": AMENDMENT_NAME, "date": date,
        "authority": "user decisions of 14 October 2026: prompt policy v5 with positions and current_state; two "
                     "prompts (GLM, DeepSeek) and a judge; two videos from two explicit seeds and a judge that may "
                     "mark none acceptable; judge google/gemini-3.8-flash with one pinned provider",
        "parent_protocol_digest": protocol["digest"], "parent_model_digest": protocol["model_digest"],
        "replaces": "v3 prompt policy (one prompt per plan, model by slot) and one video per plan",
        "policy_identity": policy_identity(), "policy_digest": policy_digest(),
        "prompt_system": PROMPT_SYSTEM_V5, "prompt_judge_system": PROMPT_JUDGE_SYSTEM,
        "video_judge_system": VIDEO_JUDGE_SYSTEM,
        "applies_to": "the 15-case source panel now and, identically, every plan of all four pilot conditions",
        "gates_unchanged": {"technical": "15/15", "human_rubric": ">=12/15 named passes on the SELECTED videos",
                            "expert_replay": ">=14/15 reused expert replays",
                            "timing": "as registered in the parent protocol (timing_calibration_rule)"},
        "preserved": ["four conditions", "five-shot support", "seed 42", "6/2/2 split",
                      "two development tasks x five starts", "H=8", "one chunk per replan", "frozen generator",
                      "controller-only LoRA+head adaptation", "sealed final tasks",
                      "VPP2 action policy not used"],
        "invalidates": ["v3 panel human ratings for gate purposes (different policy)", "generated clip caches"],
        "cost_note": "about two prompt calls, two judge calls and two videos per plan (about twice the GPU time of v3)",
        "scientific_admission": False, "held_out_claims_eligible": False,
    }
    body["amendment_digest"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def write_amendment(protocol: dict, path: str | Path, **kw) -> dict:
    body = build_amendment(protocol, **kw)
    with Path(path).open("x") as handle:  # never overwrite a registered amendment
        handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def verify_amendment(protocol: dict, amendment: dict) -> None:
    body = {k: val for k, val in amendment.items() if k != "amendment_digest"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != amendment.get("amendment_digest"):
        raise ValueError("selection amendment was edited after registration")
    if amendment.get("name") != AMENDMENT_NAME or amendment.get("parent_protocol_digest") != protocol["digest"]:
        raise ValueError("selection amendment belongs to a different protocol")
    if amendment.get("policy_digest") != policy_digest():
        raise ValueError("selection code differs from the registered v5 policy")


def candidate_seeds(protocol: dict, case_id: str, replan_index: int) -> tuple[int, int]:
    from .world_model import generation_seed
    base = int(protocol["design"]["generator"]["base_seed"])
    seeds = (generation_seed(case_id, replan_index, base),
             generation_seed(case_id + "\x1fcandidate:1", replan_index, base))
    if seeds[0] == seeds[1]:
        raise ValueError("candidate seeds collide")
    return seeds


# ---------------------------------------------------------------- prompts

def parse_prompt_v5(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != set(PROMPT_KEYS_V5):
        raise ValueError(f"expected exactly the keys {list(PROMPT_KEYS_V5)}")
    out = {}
    for key in PROMPT_KEYS_V5:
        value = v._clean(payload[key]) if isinstance(payload[key], str) else ""
        if not value:
            raise ValueError(f"empty {key}")
        if "@@" in value or "<" in value or ">" in value:
            raise ValueError(f"{key} contains formatting characters")
        out[key] = value if value.endswith((".", "!", "?")) else value + "."
    words = sum(len(x.split()) for x in out.values())
    if not 30 <= words <= 260:
        raise ValueError(f"prompt has {words} words; expected 30-260")
    return out


def assemble_prompt_v5(parts: dict) -> str:
    return " ".join([parts[key] for key in PROMPT_KEYS_V5] + [v.VIEW_SUFFIX])


def strip_suffix(prompt: str) -> str:
    return prompt[:-len(v.VIEW_SUFFIX)].strip() if prompt.endswith(v.VIEW_SUFFIX) else prompt


class PromptPolicyV5:
    """f(current main-band image, instruction) -> PromptRecord; no cache (every plan is uncached)."""

    def __init__(self, generate_text: Callable, *, model_id: str):
        self.generate_text = generate_text
        self.model_id = model_id

    def __call__(self, image: np.ndarray, instruction: str, **forbidden) -> v.PromptRecord:
        if forbidden:
            raise TypeError(f"prompting accepts only the current image and instruction; got {sorted(forbidden)}")
        image = np.asarray(image)
        if image.shape != (v.PROMPT_IMAGE_SIZE[1], v.PROMPT_IMAGE_SIZE[0], 3) or image.dtype != np.uint8:
            raise ValueError("prompt image must be the 448x336 main-band view")
        started, attempts, correction, parts = time.perf_counter(), [], None, None
        for _ in range(PROMPT_ATTEMPTS):
            t0 = time.perf_counter()
            text, complete = self.generate_text(image, PROMPT_SYSTEM_V5, v.prompt_user_message(instruction, correction),
                                                v.PROMPT_MAX_NEW_TOKENS)
            row = {"response": text, "complete": bool(complete), "seconds": time.perf_counter() - t0}
            try:
                if not complete:
                    raise ValueError("response hit the token limit")
                parts = parse_prompt_v5(text)
                attempts.append(row)
                break
            except (ValueError, TypeError) as exc:
                row["error"] = str(exc)
                attempts.append(row)
                correction = str(exc)
        return v.PromptRecord(
            prompt=assemble_prompt_v5(parts) if parts else v.fallback_prompt(instruction), policy=PROMPT_POLICY_V5,
            model_id=self.model_id, model_revision=v.PROMPT_MODEL_REVISION,
            image_sha256=hashlib.sha256(image.tobytes()).hexdigest(), instruction=v._clean(instruction),
            fallback=parts is None, attempts=attempts, seconds=time.perf_counter() - started)


# ---------------------------------------------------------------- judge

def _png_url(image: np.ndarray) -> str:
    import base64
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _provider_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


class OpenRouterJudgeBackend:
    """Pinned OpenRouter judge; ``(system, content_parts) -> (text, complete)``; stdlib HTTP."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: dict, key_file: str | Path, *, timeout: float = 300.0, retries: int = 3):
        self.config = dict(config)
        self.key_file = Path(key_file)
        self.timeout = timeout
        self.retries = retries
        self.log: list[dict] = []

    def request_body(self, system: str, content: list) -> dict:
        c = self.config
        body = {"model": c["model"], "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": content}],
                "reasoning": {"effort": c["reasoning_effort"]}, "max_tokens": 16000,
                "provider": {"only": list(c["providers"]), "allow_fallbacks": False, "require_parameters": True},
                "response_format": {"type": "json_object"}}
        if c.get("seed") is not None:
            body["seed"] = c["seed"]
        if c.get("temperature") is not None:
            body["temperature"] = c["temperature"]
        return body

    def __call__(self, system: str, content: list):
        import urllib.error
        import urllib.request
        body = json.dumps(self.request_body(system, content)).encode()
        key = self.key_file.read_text().strip()
        allowed = [_provider_slug(p.split("/")[0]) for p in self.config["providers"]]
        last = None
        for attempt in range(self.retries):
            request = urllib.request.Request(self.URL, data=body, method="POST", headers={
                "Authorization": "Bearer " + key, "Content-Type": "application/json",
                "X-Title": "RAID VPP2 selection judge"})
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    reply = json.loads(response.read())
            except urllib.error.HTTPError as error:
                last = f"HTTP {error.code}: {error.read().decode(errors='replace')[:500]}"
            except Exception as error:  # network failure: retry with backoff
                last = f"{type(error).__name__}: {error}"
            else:
                choice = (reply.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                provider = reply.get("provider")
                self.log.append({"model": reply.get("model"), "provider": provider, "id": reply.get("id"),
                                 "finish_reason": choice.get("finish_reason"), "usage": reply.get("usage"),
                                 "reasoning": message.get("reasoning"), "seconds": time.perf_counter() - started})
                if provider and _provider_slug(provider) not in allowed:
                    raise RuntimeError(f"unpinned provider served the judge request: {provider}")
                return message.get("content") or "", choice.get("finish_reason") == "stop"
            self.log.append({"error": last, "attempt": attempt, "seconds": time.perf_counter() - started})
            time.sleep(2 ** attempt)
        return "", False


def probe_judge(key_file: str | Path) -> dict:
    """Tiny text-only request: confirms the account privacy settings allow the pinned route."""
    backend = OpenRouterJudgeBackend(JUDGE_MODEL, key_file, timeout=120, retries=1)
    text, complete = backend("Reply with a JSON object {\"ok\": true} and nothing else.",
                             [{"type": "text", "text": "Probe."}])
    row = backend.log[-1] if backend.log else {}
    return {"model": JUDGE_MODEL["model"], "pinned": JUDGE_MODEL["providers"], "complete": complete,
            "served_by": row.get("provider"), "error": row.get("error"), "reply": text[:80],
            "cost_usd": (row.get("usage") or {}).get("cost"), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _strict_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("judge reply is not a JSON object")
    return payload


def _score(payload: dict, key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
        raise ValueError(f"{key} must be an integer from 0 to 10")
    return value


def parse_prompt_verdict(text: str) -> dict:
    p = _strict_json(text)
    if set(p) != {"score_A", "score_B", "choice", "reason"}:
        raise ValueError("expected keys score_A, score_B, choice, reason")
    a, b = _score(p, "score_A"), _score(p, "score_B")
    if p["choice"] not in ("A", "B") or not isinstance(p["reason"], str) or not p["reason"].strip():
        raise ValueError("choice must be A or B and reason must be text")
    if a != b and p["choice"] != ("A" if a > b else "B"):
        raise ValueError("choice must be the candidate with the higher score")
    return {"score_A": a, "score_B": b, "choice": p["choice"], "reason": v._clean(p["reason"])}


def parse_video_verdict(text: str) -> dict:
    p = _strict_json(text)
    if set(p) != {"score_A", "score_B", "acceptable_A", "acceptable_B", "choice", "reason"}:
        raise ValueError("expected keys score_A, score_B, acceptable_A, acceptable_B, choice, reason")
    a, b = _score(p, "score_A"), _score(p, "score_B")
    ok_a, ok_b = p["acceptable_A"], p["acceptable_B"]
    if not isinstance(ok_a, bool) or not isinstance(ok_b, bool):
        raise ValueError("acceptable_A and acceptable_B must be true or false")
    if p["choice"] not in ("A", "B", "none") or not isinstance(p["reason"], str) or not p["reason"].strip():
        raise ValueError("choice must be A, B or none and reason must be text")
    if (p["choice"] == "none") != (not ok_a and not ok_b):
        raise ValueError("choose none exactly when neither video is acceptable")
    if p["choice"] == "A" and not ok_a or p["choice"] == "B" and not ok_b:
        raise ValueError("the chosen video must be acceptable")
    if ok_a and ok_b and a != b and p["choice"] != ("A" if a > b else "B"):
        raise ValueError("choice must be the acceptable video with the higher score")
    return {"score_A": a, "score_B": b, "acceptable_A": ok_a, "acceptable_B": ok_b, "choice": p["choice"],
            "reason": v._clean(p["reason"])}


def _judge(backend: Callable, system: str, content: list, parse: Callable) -> dict:
    attempts, correction = [], None
    for _ in range(JUDGE_ATTEMPTS):
        parts = content + ([{"type": "text", "text": f"Correction: {correction} Return only the required JSON."}]
                           if correction else [])
        t0 = time.perf_counter()
        text, complete = backend(system, parts)
        row = {"response": text, "complete": bool(complete), "seconds": time.perf_counter() - t0}
        try:
            if not complete:
                raise ValueError("response was not complete")
            verdict = parse(text)
            attempts.append(row)
            return {"valid": True, "verdict": verdict, "attempts": attempts}
        except (ValueError, TypeError) as exc:
            row["error"] = str(exc)
            attempts.append(row)
            correction = str(exc)
    return {"valid": False, "verdict": None, "attempts": attempts}


def _cost(log: list[dict]) -> float:
    return float(sum(float((row.get("usage") or {}).get("cost") or 0) for row in log))


# ---------------------------------------------------------------- selector

def select_prompt(policies: list, judge: Callable, image: np.ndarray, image_sha: str, instruction: str):
    """Two v5 prompts (concurrently) and the prompt judge. Returns records, judge record, chosen index, seconds."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(policies)) as pool:
        records = list(pool.map(lambda p: p(image, instruction), policies))
    prompt_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    order = [0, 1] if int(image_sha[0], 16) % 2 == 0 else [1, 0]
    valid = [i for i, r in enumerate(records) if not r.fallback]
    prompt_judge = {"order": order, "called": False}
    if len(valid) == 2:
        content = [{"type": "image_url", "image_url": {"url": _png_url(image)}},
                   {"type": "text", "text": f"Task instruction: {v._clean(instruction)}\n\n"
                                            f"Candidate A: {strip_suffix(records[order[0]].prompt)}\n\n"
                                            f"Candidate B: {strip_suffix(records[order[1]].prompt)}"}]
        result = _judge(judge, PROMPT_JUDGE_SYSTEM, content, parse_prompt_verdict)
        prompt_judge.update(result, called=True)
        chosen = order[0 if result["verdict"]["choice"] == "A" else 1] if result["valid"] else 0
    else:
        chosen = valid[0] if valid else 0
        prompt_judge.update(valid=False, verdict=None, attempts=[],
                            skipped="one valid prompt" if valid else "no valid prompt")
    return records, prompt_judge, chosen, prompt_seconds, time.perf_counter() - t0


class JudgedSelector:
    """One v5 plan: two prompts -> judge -> two videos -> judge. Inputs: current observation and instruction only."""

    def __init__(self, prompt_generators: list[Callable], judge: Callable, video_backend: Callable, *,
                 model_ids: list[str] | None = None, num_frames: int = 49):
        if len(prompt_generators) != len(PROMPT_MODELS_V5):
            raise ValueError("v5 uses exactly the two registered prompt models")
        self.model_ids = model_ids or [c["model"] for c in PROMPT_MODELS_V5]
        self.policies = [PromptPolicyV5(g, model_id=m) for g, m in zip(prompt_generators, self.model_ids)]
        self.generators = prompt_generators
        self.judge = judge
        self.video_backend = video_backend
        self.num_frames = num_frames

    def plan(self, observation: np.ndarray, instruction: str, *, seeds: tuple[int, int], **forbidden):
        """Returns (record, canvas_frames_per_candidate). observation: simulator convention HxWx3 uint8."""
        if forbidden:
            raise TypeError(f"selection accepts only the current observation and instruction; got {sorted(forbidden)}")
        observation = np.asarray(observation)
        if observation.ndim != 3 or observation.dtype != np.uint8:
            raise ValueError("selection conditions on exactly one current uint8 observation frame")
        if len(seeds) != CANDIDATE_VIDEOS or len(set(seeds)) != CANDIDATE_VIDEOS:
            raise ValueError("two distinct explicit seeds are required")
        started = time.perf_counter()
        upright = np.ascontiguousarray(observation[::-1])
        canvas, canvas_meta = v.compose_canvas(upright)
        image = v.prompt_image(upright)
        image_sha = hashlib.sha256(image.tobytes()).hexdigest()
        judge_log0 = len(getattr(self.judge, "log", []))
        prompt_log0 = [len(getattr(g, "log", [])) for g in self.generators]

        records, prompt_judge, chosen, prompt_seconds, prompt_judge_seconds = select_prompt(
            self.policies, self.judge, image, image_sha, instruction)
        prompt = records[chosen].prompt

        # 3. two videos from the selected prompt
        videos, video_seconds = [], []
        for seed in seeds:
            t0 = time.perf_counter()
            frames = np.asarray(self.video_backend(image=canvas, prompt=prompt, seed=int(seed)))
            video_seconds.append(time.perf_counter() - t0)
            if frames.shape != (self.num_frames, v.OUTPUT_SIZE[1], v.OUTPUT_SIZE[0], 3) or frames.dtype != np.uint8:
                raise ValueError(f"VPP2 returned {frames.shape} {frames.dtype}")
            videos.append(frames)

        # 4. video judge
        t0 = time.perf_counter()
        vorder = [0, 1] if int(image_sha[1], 16) % 2 == 0 else [1, 0]
        content = [{"type": "text", "text": f"Task instruction: {v._clean(instruction)}\n\nPrompt: {strip_suffix(prompt)}"
                                            "\n\nStart image:"},
                   {"type": "image_url", "image_url": {"url": _png_url(image)}}]
        for label, k in zip("AB", vorder):
            main = v.extract_main_view(videos[k])
            content.append({"type": "text", "text": f"Video {label}: frames {', '.join(map(str, JUDGE_FRAMES))} "
                                                    f"of {self.num_frames}, in time order."})
            content += [{"type": "image_url", "image_url": {"url": _png_url(main[j])}} for j in JUDGE_FRAMES]
        result = _judge(self.judge, VIDEO_JUDGE_SYSTEM, content, parse_video_verdict)
        video_judge = {"order": vorder, "called": True, **result}
        none_acceptable = False
        if result["valid"]:
            verdict = result["verdict"]
            if verdict["choice"] == "none":
                none_acceptable = True
                pick = "A" if verdict["score_A"] > verdict["score_B"] else "B" if verdict["score_B"] > verdict["score_A"] else None
                selected_video = vorder[0 if pick == "A" else 1] if pick else 0
            else:
                selected_video = vorder[0 if verdict["choice"] == "A" else 1]
        else:
            selected_video = 0
        video_judge_seconds = time.perf_counter() - t0

        judge_log = getattr(self.judge, "log", [])[judge_log0:]
        prompt_logs = [getattr(g, "log", [])[n:] for g, n in zip(self.generators, prompt_log0)]
        technical_selection = bool(prompt_judge.get("valid") and video_judge["valid"]
                                   and not records[chosen].fallback)
        record = {
            "selection_policy": SELECTION_POLICY, "policy_digest": policy_digest(),
            "instruction": v._clean(instruction), "image_sha256": image_sha,
            "conditioning_sha256": hashlib.sha256(canvas.tobytes()).hexdigest(), "canvas": canvas_meta,
            "prompt_candidates": [{**asdict(r), "api_calls": log, "cost_usd": _cost(log)}
                                  for r, log in zip(records, prompt_logs)],
            "prompt_judge": prompt_judge, "selected_prompt_index": chosen, "selected_prompt": prompt,
            "selected_prompt_model": records[chosen].model_id,
            "seeds": [int(s) for s in seeds],
            "video_candidates": [{"index": k, "seed": int(s), "seconds": sec,
                                  "frames_sha256": hashlib.sha256(f.tobytes()).hexdigest()}
                                 for k, (s, sec, f) in enumerate(zip(seeds, video_seconds, videos))],
            "video_judge": video_judge, "selected_video_index": selected_video, "none_acceptable": none_acceptable,
            "judge_api_calls": judge_log, "judge_model": dict(JUDGE_MODEL),
            "selection_complete": technical_selection,
            "seconds": {"prompts": prompt_seconds, "prompt_judge": prompt_judge_seconds,
                        "videos": video_seconds, "video_judge": video_judge_seconds,
                        "plan": time.perf_counter() - started},
            "cost_usd": {"prompts": sum(_cost(x) for x in prompt_logs), "judge": _cost(judge_log)},
            "native_action_output_used": False,
        }
        return record, videos


# ---------------------------------------------------------------- pod: panel

def _default_components(root: Path, protocol: dict, device: str):
    from . import hybrid_vpp2_panel as panel
    spec = v.VPP2Spec(**protocol["design"]["generator"])
    video = v.VPP2Backend(root / "weights", spec, device).load()
    prompts = [v.OpenRouterPromptBackend(c, panel.PROMPT_KEY) for c in PROMPT_MODELS_V5]
    return prompts, OpenRouterJudgeBackend(JUDGE_MODEL, panel.PROMPT_KEY), video


def run_panel(root: str | Path, *, device: str = "cuda", components=None) -> dict:
    """Every registered panel case, uncached, with the v5 judged selection; layout matches hybrid_vpp2_panel."""
    from PIL import Image
    from . import hybrid_vpp2_panel as panel
    from .world_model import write_mp4
    root = Path(root)
    protocol = json.loads((root / "repo/protocol.json").read_text())
    panel._require_vpp2(protocol)
    verify_amendment(protocol, json.loads((root / "repo/selection_amendment.json").read_text()))
    cases = panel.verified_inputs(root, protocol)
    subset_path = root / "repo/panel_subset.json"
    subset = json.loads(subset_path.read_text()) if subset_path.is_file() else None
    if subset is not None:  # user-reduced run: a fixed, pre-registered subset of the registered cases
        ids = [c["case_id"] for c in cases]
        if not subset["case_ids"] or any(i not in ids for i in subset["case_ids"]) \
                or subset["case_ids"] != sorted(subset["case_ids"], key=ids.index):
            raise ValueError("panel subset must list registered cases in registered order")
        cases = [c for c in cases if c["case_id"] in set(subset["case_ids"])]
    out = root / "collection" / panel.PANEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        torch, cuda = None, False
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prompts, judge, video = components or _default_components(root, protocol, device)
    load_seconds = time.perf_counter() - started
    selector = JudgedSelector(prompts, judge, video)
    temporal = v.VPP2TemporalAlignment(**protocol["design"]["temporal"])
    rows = []
    for case in cases:
        with np.load(root / "inputs" / case["input"], allow_pickle=False) as payload:
            observation = payload["observation"]
        if hashlib.sha256(observation.tobytes()).hexdigest() != case["observation_sha256"]:
            raise ValueError("observation mismatch")
        atomic_progress = {"current": case["case_id"], "completed": len(rows), "at": time.time()}
        panel.atomic(out / "progress.json", atomic_progress)
        seeds = candidate_seeds(protocol, case["case_id"], 0)
        if seeds[0] != case["seed"]:
            raise ValueError("candidate 0 must use the registered seed")
        t0 = time.perf_counter()
        record, videos = selector.plan(observation, case["instruction"], seeds=seeds)
        wall = time.perf_counter() - t0
        name = case["case_id"].replace(":", "_")
        case_dir, cand_dir = out / "clips" / name, out / "candidates" / name
        case_dir.mkdir(parents=True, exist_ok=True)
        cand_dir.mkdir(parents=True, exist_ok=True)
        mains = [v.extract_main_view(f) for f in videos]
        for k, main in enumerate(mains):
            write_mp4(main, cand_dir / f"video_{k}_main_view.mp4", temporal.display_fps)
        upright = np.ascontiguousarray(observation[::-1])
        Image.fromarray(v.prompt_image(upright)).save(cand_dir / "prompt_image.png")
        k = record["selected_video_index"]
        write_mp4(videos[k], case_dir / "generated_canvas.mp4", temporal.display_fps)
        write_mp4(mains[k], case_dir / "generated_main_view.mp4", temporal.display_fps)
        left, top, right, bottom = v.band_box((observation.shape[1], observation.shape[0]))
        band = v.generated_to_controller(mains[k], (bottom - top, right - left))
        np.savez_compressed(case_dir / "controller_band.npz", frames=band)
        canvas, _ = v.compose_canvas(upright)
        Image.fromarray(canvas).save(case_dir / "conditioning_canvas.png")
        panel.atomic(cand_dir / "selection.json", {"case_id": case["case_id"], **record})
        technical = {
            "frames": int(len(band)), "controller_shape": list(band.shape[1:]),
            "finite_uint8": bool(band.dtype == np.uint8),
            "conditioning_matches_input": record["conditioning_sha256"] == hashlib.sha256(canvas.tobytes()).hexdigest(),
            "candidate_videos": len(videos), "selection_complete": record["selection_complete"],
            "native_action_output_used": False, "cache_hit": False,
            "frame0_band_mean_abs_error": panel._frame0_error(band, observation)}
        technical["ok"] = (technical["frames"] == 49 and technical["controller_shape"] == [96, 128, 3]
                           and technical["finite_uint8"] and technical["conditioning_matches_input"]
                           and technical["candidate_videos"] == CANDIDATE_VIDEOS and technical["selection_complete"])
        selected = record["prompt_candidates"][record["selected_prompt_index"]]
        row = {**case, "prompt": record["selected_prompt"], "prompt_model": record["selected_prompt_model"],
               "prompt_fallback": selected["fallback"], "selected_video_index": k,
               "selected_seed": record["seeds"][k], "none_acceptable": record["none_acceptable"],
               "prompt_judge_verdict": record["prompt_judge"].get("verdict"),
               "video_judge_verdict": record["video_judge"].get("verdict"),
               "seconds": record["seconds"], "cost_usd": record["cost_usd"], "wall_seconds": wall,
               "frames_sha256": record["video_candidates"][k]["frames_sha256"], "technical": technical,
               "selection_policy": SELECTION_POLICY, "quality_rating": None}
        panel.atomic(case_dir / "case.json", row)
        rows.append(row)
    walls = [r["wall_seconds"] for r in rows]
    summary = {
        "n": len(rows), "protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
        "selection_policy": SELECTION_POLICY, "policy_digest": policy_digest(),
        "gpu": torch.cuda.get_device_name() if cuda else None, "torch": torch.__version__ if torch else None,
        "load_seconds": {"total": load_seconds, "video": getattr(video, "load_seconds", None)},
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if cuda else None,
        "prompt_seconds": [r["seconds"]["prompts"] for r in rows],
        "judge_seconds": [r["seconds"]["prompt_judge"] + r["seconds"]["video_judge"] for r in rows],
        "video_seconds": [s for r in rows for s in r["seconds"]["videos"]],
        "plan_wall_seconds": walls, "median_plan_wall_seconds": statistics.median(walls),
        "p90_plan_wall_seconds": float(np.percentile(walls, 90)),
        "openrouter_cost_usd": {"prompts": sum(r["cost_usd"]["prompts"] for r in rows),
                                "judge": sum(r["cost_usd"]["judge"] for r in rows)},
        "prompt_fallbacks": sum(bool(r["prompt_fallback"]) for r in rows),
        "prompt_choice_counts": {m: sum(r["prompt_model"] == m for r in rows) for m in
                                 [c["model"] for c in PROMPT_MODELS_V5]},
        "selected_video_counts": {str(i): sum(r["selected_video_index"] == i for r in rows) for i in range(2)},
        "none_acceptable": sum(r["none_acceptable"] for r in rows),
        "judge_failures": sum(not r["technical"]["selection_complete"] for r in rows),
        "technical_passes": sum(r["technical"]["ok"] for r in rows),
        "human_ratings": None, "timing_calibrated": False, "subset": subset,
        "scientific_admission": False, "held_out_claims_eligible": False,
    }
    panel.atomic(out / "panel_summary.json", summary)
    return summary


# ---------------------------------------------------------------- pilot: judged prompt, one video

PILOT_MODE = "judged_prompt_single_video_v5"
PILOT_PROMPT_POLICY = "raid_vpp2_judged_prompt_v5_single_video"


class JudgedPromptPolicy:
    """Pilot policy (user decision 14 Oct 2026): two v5 prompts + prompt judge, then ONE video per plan.

    Cached by the current image, instruction and policy digest, so identical first observations of the
    four conditions receive the identical prompt (shared initial plan). The video judge is not used.
    """

    def __init__(self, prompt_generators: list[Callable], judge: Callable, cache_root: str | Path | None = None):
        if len(prompt_generators) != len(PROMPT_MODELS_V5):
            raise ValueError("v5 uses exactly the two registered prompt models")
        self.generators, self.judge = prompt_generators, judge
        self.policies = [PromptPolicyV5(g, model_id=c["model"]) for g, c in zip(prompt_generators, PROMPT_MODELS_V5)]
        self.cache_root = Path(cache_root) if cache_root else None

    def key(self, image: np.ndarray, instruction: str) -> str:
        h = hashlib.sha256(np.ascontiguousarray(image).tobytes())
        h.update(json.dumps({"instruction": v._clean(instruction), "policy": policy_digest(), "mode": PILOT_MODE},
                            sort_keys=True).encode())
        return h.hexdigest()

    def __call__(self, image: np.ndarray, instruction: str, **forbidden) -> v.PromptRecord:
        if forbidden:
            raise TypeError(f"prompting accepts only the current image and instruction; got {sorted(forbidden)}")
        image = np.asarray(image)
        image_sha = hashlib.sha256(image.tobytes()).hexdigest()
        key = self.key(image, instruction)
        path = self.cache_root / key[:2] / f"{key}.json" if self.cache_root else None
        if path is not None and path.is_file():
            record = v.PromptRecord(**json.loads(path.read_text()))
            if record.image_sha256 != image_sha:
                raise ValueError(f"cached judged prompt {key} is corrupt")
            record.cache_hit, record.seconds = True, 0.0
            return record
        started = time.perf_counter()
        judge0 = len(getattr(self.judge, "log", []))
        logs0 = [len(getattr(g, "log", [])) for g in self.generators]
        records, prompt_judge, chosen, prompt_seconds, judge_seconds = select_prompt(
            self.policies, self.judge, image, image_sha, instruction)
        judge_log = getattr(self.judge, "log", [])[judge0:]
        prompt_logs = [getattr(g, "log", [])[n:] for g, n in zip(self.generators, logs0)]
        selection = {"mode": PILOT_MODE, "policy_digest": policy_digest(),
                     "prompt_candidates": [{**asdict(r), "api_calls": log, "cost_usd": _cost(log)}
                                           for r, log in zip(records, prompt_logs)],
                     "prompt_judge": prompt_judge, "selected_prompt_index": chosen,
                     "selection_complete": bool(prompt_judge.get("valid")) and not records[chosen].fallback,
                     "judge_api_calls": judge_log,
                     "seconds": {"prompts": prompt_seconds, "prompt_judge": judge_seconds},
                     "cost_usd": {"prompts": sum(_cost(x) for x in prompt_logs), "judge": _cost(judge_log)}}
        record = v.PromptRecord(
            prompt=records[chosen].prompt, policy=PILOT_PROMPT_POLICY, model_id=records[chosen].model_id,
            model_revision=v.PROMPT_MODEL_REVISION, image_sha256=image_sha, instruction=v._clean(instruction),
            fallback=not selection["selection_complete"], attempts=[selection],
            seconds=time.perf_counter() - started)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(record), indent=2) + "\n")
            tmp.replace(path)
        return record


def amend_protocol_pilot(stamped: dict, amendment: dict, waiver: dict, *, chunks_per_plan: int = 2,
                         date: str = "2026-10-14") -> dict:
    """Pilot protocol: timing-stamped VPP2 protocol + v5 judged prompt, one video, chunks_per_plan, gate waiver."""
    from .hybrid_protocol import VPP2TemporalAlignment, digest, model_digest, validate_hybrid_protocol
    validate_hybrid_protocol(stamped)
    if amendment.get("parent_protocol_digest") != stamped["design"]["temporal"].get("source_protocol_digest"):
        raise ValueError("selection amendment and timing stamp belong to different parent protocols")
    if amendment.get("policy_digest") != policy_digest():
        raise ValueError("selection code differs from the registered v5 policy")
    result = json.loads(json.dumps(stamped))
    design = result["design"]
    design["temporal"]["chunks_per_plan"] = int(chunks_per_plan)
    VPP2TemporalAlignment(**design["temporal"])  # the clip must cover the whole replanning interval
    design["generator"]["prompt_policy"] = PILOT_PROMPT_POLICY
    design["generator"]["prompt_model_id"] = "openrouter:" + ",".join(c["model"] for c in PROMPT_MODELS_V5) \
        + "+judge:" + JUDGE_MODEL["model"]
    waiver_sha = hashlib.sha256(json.dumps(waiver, sort_keys=True).encode()).hexdigest()
    design["prompt_selection"] = {
        "mode": PILOT_MODE, "amendment_digest": amendment["amendment_digest"], "policy_digest": policy_digest(),
        "videos_per_plan": 1, "video_judge": "not used in the pilot (user chose one video per plan for cost)",
        "chunks_per_plan": int(chunks_per_plan),
        "chunks_rationale": "user approved 2 chunks (16 steps) per video on 14 Oct 2026; 1 chunk per replan was "
                            "about $181 and exceeded the prepaid balance; applies identically to all four conditions",
        "human_gate_waiver_sha256": waiver_sha, "date": date,
        # Panel evidence (CUDA, technical) was produced under the timing-stamped parent model identity.
        "evidence_model_digest": stamped["model_digest"]}
    result["model_digest"] = model_digest(result)
    result["digest"] = digest(result)
    validate_hybrid_protocol(result)
    return result


def load_pilot_world_model(weights_root, key_file, cache_root, protocol: dict, device: str = "cuda",
                           components=None):
    """VPP2 + judged v5 prompt policy for the pilot service; refuses a protocol for another policy."""
    sel = protocol.get("design", {}).get("prompt_selection") or {}
    if sel.get("mode") != PILOT_MODE or sel.get("policy_digest") != policy_digest():
        raise ValueError("protocol does not register this judged prompt policy")
    spec = v.VPP2Spec(**protocol["design"]["generator"])
    if components is None:
        backend = v.VPP2Backend(weights_root, spec, device).load()
        prompts = [v.OpenRouterPromptBackend(c, key_file) for c in PROMPT_MODELS_V5]
        judge = OpenRouterJudgeBackend(JUDGE_MODEL, key_file)
    else:
        prompts, judge, backend = components
    policy = JudgedPromptPolicy(prompts, judge, Path(cache_root) / "prompts")
    world = v.VPP2WorldModel(backend, policy, Path(cache_root) / "clips", spec=spec,
                             temporal=v.VPP2TemporalAlignment(**protocol["design"]["temporal"]))
    world.load_seconds = {"video": getattr(backend, "load_seconds", None)}
    return world


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("root")
    s = sub.add_parser("serve", help="pilot generation service (VPP2 runtime), judged v5 prompt + one video")
    for name in ("--protocol", "--weights", "--prompt-key", "--cache", "--ready"):
        s.add_argument(name, required=True)
    s.add_argument("--port", type=int, default=8765)
    a = parser.parse_args(argv)
    if a.command == "run":
        run_panel(a.root)
        return 0
    from .hybrid_protocol import validate_hybrid_protocol
    protocol = json.loads(Path(a.protocol).read_text())
    validate_hybrid_protocol(protocol)
    world = load_pilot_world_model(a.weights, a.prompt_key, a.cache, protocol)
    v.serve(world, port=a.port, protocol_digest=protocol["digest"], ready_path=a.ready)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
