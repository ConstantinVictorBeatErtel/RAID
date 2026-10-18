"""VPP2 Stage-1 as RAID's frozen generator (user-selected 13 October 2026).

Pipeline per plan: current simulator RGB -> official VPP2 robot canvas (main
view center-cropped to 4:3, black wrist panels) -> observation-based prompt
-> pinned standalone Stage-1 video model (49 frames) -> main-view extraction
-> controller view (the same 4:3 band applied to recorded training frames).

Only the current RGB and the task instruction enter generation. The VPP2
LIBERO action policy is never loaded; RAID's inverse controller acts.

Image-coordinate transform (sizes W x H). For an upright square observation
of side S, the main band is rows [S/8, 7S/8), all columns (crop_box 4:3).
Canvas: band -> 320x240 at (0,0) of a black 440x240 canvas -> 416x240.
Inverse: generated 416x240 -> 440x240 bicubic -> columns [0,320) -> band.
Generated band pixel (u, v) in 320x240 maps to upright source pixel
(u * S / 320, S/8 + v * (3S/4) / 240). Recorded controller frames use the same
band, so the controller never sees content outside what VPP2 generated and
never sees a black wrist placeholder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np

from .hybrid_protocol import (VPP2_FAMILY, VPP2TemporalAlignment, digest, is_vpp2, model_digest,
                              v6_code_snapshot_digest, validate_hybrid_protocol)

VPP2_CODE_REPOSITORY = "https://github.com/roboterax/video-prediction-policy-2"
VPP2_CODE_COMMIT = "72c17acf022494ff63617be86079bb92e92f5bf8"
VPP2_HF_REPO = "Haodong082399/VPP2"
VPP2_HF_REVISION = "04824ddb3aae11f13db50984ed7d540c3b77dd9f"
VPP2_CHECKPOINT = "checkpoints_video/vpp2-video-stage1-49f.pth"
VPP2_WAN_DIR = "checkpoints/Wan2.1-I2V-14B-480P"
VPP2_ASSETS_SHA256 = "6abc9da3455fb717dfb3308af206571cc7a311bff7f2dc8c1968bb16dafd2431"
VPP2_LOCK_SHA256 = "5875b5f6ad473ab4a0356f2d1907261a55dcd76a244734fe69a5df7b713b3cac"

# Observation-based prompting: two user-selected OpenRouter vision models,
# each pinned to its first-party provider with fallbacks disabled, reasoning
# on, deterministic decoding where the provider supports it.
PROMPT_MODELS = (
    {"model": "z-ai/glm-5.3-flashx", "providers": ["z-ai"], "reasoning_effort": "high",
     "seed": None, "temperature": 0.0, "response_format": "json_object"},
    # DeepSeek's own endpoint trains on prompts and is blocked by the account privacy setting;
    # Alibaba serves the same model at the same price and supports a fixed seed.
    {"model": "deepseek/deepseek-v4.1-flash", "providers": ["alibaba"], "reasoning_effort": "high",
     "seed": 42, "temperature": 0.0, "response_format": "json_object"},
)
# User-selected after the six-video test (GPT-6 Luna Pro grounded the plate as "the bowl").
PROMPT_SLOT_RULE = "panel: case index mod 2; rollout: replan index mod 2 (identical for all four conditions)"
PROMPT_MODEL_ID = "openrouter:" + ",".join(m["model"] for m in PROMPT_MODELS)
PROMPT_MODEL_REVISION = "openrouter-pinned-providers-2026-10-13"
PROMPT_POLICY_VERSION = "raid_vpp2_observation_prompt_v3_glm_deepseek"

CANVAS_SIZE = (440, 240)    # W, H before the official final resize
OUTPUT_SIZE = (416, 240)    # W, H of conditioning and generated frames
MAIN_SIZE = (320, 240)      # W, H of the main view inside the canvas
MAIN_ASPECT = 4 / 3
PROMPT_IMAGE_SIZE = (448, 336)  # W, H; the main band enlarged for the prompt model

VIEW_SUFFIX = ("This is a composed robot video with two wrist-view in the right side. "
               "Both right-side wrist panels are unavailable and remain black; they contain "
               "no additional robot arms. The main camera remains static.")


@dataclass(frozen=True)
class VPP2Spec:
    family: str = VPP2_FAMILY
    code_repository: str = VPP2_CODE_REPOSITORY
    code_commit: str = VPP2_CODE_COMMIT
    model_id: str = VPP2_HF_REPO
    revision: str = VPP2_HF_REVISION
    checkpoint: str = VPP2_CHECKPOINT
    wan_model_dir: str = VPP2_WAN_DIR
    assets_manifest_sha256: str = VPP2_ASSETS_SHA256
    runtime_lock_sha256: str = VPP2_LOCK_SHA256
    mode: str = "standalone_stage1_image_to_video"
    action_policy_used: bool = False
    dtype: str = "bf16"
    height: int = 240
    width: int = 416
    num_frames: int = 49
    sampling_steps: int = 30
    guidance_scale: float = 4.0
    sigma_shift: float = 3.0
    negative_prompt: str = ""
    max_area: int = 99841
    canvas: str = "official_robot_canvas_main_view_only_black_wrist_panels"
    conditioning: str = "single_current_rgb_frame_and_instruction"
    seed_policy: str = "explicit_per_request_seed_from_generation_seed(case_id, replan, base_seed)"
    base_seed: int = 42
    prompt_policy: str = PROMPT_POLICY_VERSION
    prompt_model_id: str = PROMPT_MODEL_ID
    prompt_model_revision: str = PROMPT_MODEL_REVISION
    prompt_models: str = ",".join(m["model"] for m in PROMPT_MODELS)


CONTROLLER_VIEW = {
    "name": "vpp2_main_band_4x3",
    "transform": "upright center crop to width/height 4:3 (official VPP2 main-view crop); "
                 "recorded and generated frames both enter the controller in this band",
    "generated_inverse": "416x240 -> 440x240 bicubic, columns [0,320), resize to band size",
    "black_wrist_panels_in_controller_input": False,
}


# ---------------------------------------------------------------- geometry

def crop_box(size: tuple[int, int], aspect: float) -> tuple[float, float, float, float]:
    """Largest centered crop with width/height ``aspect`` (upstream conditioning.crop_box)."""
    width, height = size
    crop_width, crop_height = min(width, height * aspect), min(height, width / aspect)
    left, top = (width - crop_width) / 2, (height - crop_height) / 2
    return (left, top, left + crop_width, top + crop_height)


def _box_int(box) -> tuple[int, int, int, int]:
    return tuple(int(round(v)) for v in box)


def _pil(frame: np.ndarray):
    from PIL import Image
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("expected one HxWx3 uint8 RGB frame")
    return Image.fromarray(frame)


def compose_canvas(upright: np.ndarray) -> tuple[np.ndarray, dict]:
    """Official VPP2 robot canvas with only the main view (upstream prepare_image)."""
    from PIL import Image
    view = _pil(upright)
    box = crop_box(view.size, MAIN_ASPECT)
    canvas = Image.new("RGB", CANVAS_SIZE, "black")
    canvas.paste(view.crop(box).resize(MAIN_SIZE, Image.Resampling.BICUBIC), (0, 0))
    result = canvas.resize(OUTPUT_SIZE, Image.Resampling.BICUBIC)
    return np.asarray(result), {"main_crop_box_xyxy": list(box), "original_size_wh": list(view.size),
                                "missing_views": [1, 2], "output_size_wh": list(OUTPUT_SIZE)}


def extract_main_view(frames: np.ndarray) -> np.ndarray:
    """Generated (T,240,416,3) canvas frames -> upright (T,240,320,3) main view."""
    from PIL import Image
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[1:] != (OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3) or frames.dtype != np.uint8:
        raise ValueError(f"expected (T,240,416,3) uint8 VPP2 frames; got {frames.shape} {frames.dtype}")
    out = []
    for frame in frames:
        wide = Image.fromarray(frame).resize(CANVAS_SIZE, Image.Resampling.BICUBIC)
        out.append(np.asarray(wide)[:, :MAIN_SIZE[0]])
    return np.ascontiguousarray(np.stack(out))


def band_box(size_wh: tuple[int, int]) -> tuple[int, int, int, int]:
    return _box_int(crop_box(size_wh, MAIN_ASPECT))


def controller_band(frames_sim: np.ndarray) -> np.ndarray:
    """Recorded/live simulator-convention frames -> the same 4:3 band, simulator convention."""
    frames = np.asarray(frames_sim)
    single = frames.ndim == 3
    batch = frames[None] if single else frames
    if batch.ndim != 4 or batch.shape[-1] != 3:
        raise ValueError("expected (H,W,3) or (T,H,W,3) RGB frames")
    height, width = batch.shape[1:3]
    left, top, right, bottom = band_box((width, height))
    # Crop in upright coordinates; LIBERO frames are vertically flipped (opengl).
    upright = batch[:, ::-1]
    band = np.ascontiguousarray(upright[:, top:bottom, left:right][:, ::-1])
    return band[0] if single else band


def generated_to_controller(main_view_upright: np.ndarray, band_hw: tuple[int, int]) -> np.ndarray:
    """Upright main-view frames -> simulator-convention band frames at the recorded band size."""
    from .world_model import resize_rgb
    resized = resize_rgb(np.asarray(main_view_upright), band_hw)
    return np.ascontiguousarray(resized[:, ::-1] if resized.ndim == 4 else resized[::-1])


def prompt_image(upright: np.ndarray) -> np.ndarray:
    """The main band VPP2 sees, enlarged for the prompt model's patch grid."""
    from PIL import Image
    view = _pil(upright)
    band = view.crop(crop_box(view.size, MAIN_ASPECT))
    return np.asarray(band.resize(PROMPT_IMAGE_SIZE, Image.Resampling.BICUBIC))


# ---------------------------------------------------------------- prompting

PROMPT_SYSTEM = """You write the text prompt for an image-to-video robot model.
The image is the CURRENT frame from a fixed camera. It shows one white-and-gray Franka
robot arm with a two-finger gripper above a wooden tabletop with household objects.
Use only what is visible in this image and the task instruction. Never use numbers,
coordinates, invented objects, a second arm, tools, camera motion or scene cuts.
Describe only the REMAINING motion needed from the current visible state to finish the
task. If the task already appears partly done (an object held or moved, a drawer partly
open), continue from that state and do not repeat completed steps.
Return ONLY a JSON object with these string keys, in this order:
"scene": two or three sentences locating the target object and its destination or
control in image terms (left/right/top/bottom, near which visible objects), with
distinguishing colors and shapes, and where the gripper is now;
"approach": one sentence on how the gripper moves toward the contact point;
"contact": one sentence on how the fingers grasp, press or hook the target;
"manipulation": one sentence on lifting, carrying, pulling, pushing or turning it;
"completion": one sentence on placing or releasing and retracting while unrelated
objects stay where they are;
"final_state": one sentence stating the completed goal as it would look in the image.
Keep the six values under 170 words in total."""

PROMPT_KEYS = ("scene", "approach", "contact", "manipulation", "completion", "final_state")
PROMPT_MAX_NEW_TOKENS = 512
PROMPT_ATTEMPTS = 3


def _clean(text: str) -> str:
    return " ".join(str(text).split()).strip()


def parse_prompt_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != set(PROMPT_KEYS):
        raise ValueError(f"expected exactly the keys {list(PROMPT_KEYS)}")
    out = {}
    for key in PROMPT_KEYS:
        value = _clean(payload[key]) if isinstance(payload[key], str) else ""
        if not value:
            raise ValueError(f"empty {key}")
        if "@@" in value or "<" in value or ">" in value:
            raise ValueError(f"{key} contains formatting characters")
        out[key] = value if value.endswith((".", "!", "?")) else value + "."
    words = sum(len(v.split()) for v in out.values())
    if not 30 <= words <= 230:
        raise ValueError(f"prompt has {words} words; expected 30-230")
    return out


def assemble_prompt(parts: dict) -> str:
    return " ".join([parts[key] for key in PROMPT_KEYS] + [VIEW_SUFFIX])


def fallback_prompt(instruction: str) -> str:
    """Deterministic instruction-only prompt, used and logged only if the VLM fails."""
    task = _clean(instruction).rstrip(".")
    return (f"The single white-and-gray Franka robot arm on the wooden tabletop carries out this "
            f"task: {task}. It moves its open two-finger gripper to the relevant object visible in "
            f"the image, makes firm contact, performs the required motion continuously and finishes "
            f"with the task completed while unrelated objects stay where they are. {VIEW_SUFFIX}")


def prompt_user_message(instruction: str, correction: str | None = None) -> str:
    text = f"Task instruction: {_clean(instruction)}"
    return text + (f"\nCorrection: {correction} Return only the required JSON." if correction else "")


@dataclass
class PromptRecord:
    prompt: str
    policy: str
    model_id: str
    model_revision: str
    image_sha256: str
    instruction: str
    fallback: bool
    attempts: list = field(default_factory=list)
    seconds: float = 0.0
    cache_hit: bool = False


class ObservationPromptPolicy:
    """Frozen per-observation prompt policy: f(current main view, instruction) only.

    ``generate_text(image, system, user, max_new_tokens)`` returns
    ``(text, complete)`` with greedy decoding. Results are cached by the image
    bytes, instruction, policy version and pinned model revision.
    """

    def __init__(self, generate_text: Callable, cache_root: str | Path | None = None, *,
                 model_id: str = PROMPT_MODEL_ID, model_revision: str = PROMPT_MODEL_REVISION):
        self.generate_text = generate_text
        self.cache_root = Path(cache_root) if cache_root else None
        self.model_id = model_id
        self.model_revision = model_revision

    def key(self, image: np.ndarray, instruction: str) -> str:
        h = hashlib.sha256(np.ascontiguousarray(image).tobytes())
        h.update(json.dumps({"shape": list(image.shape), "instruction": _clean(instruction),
                             "policy": PROMPT_POLICY_VERSION, "system": PROMPT_SYSTEM,
                             "model": self.model_id, "revision": self.model_revision},
                            sort_keys=True).encode())
        return h.hexdigest()

    def __call__(self, image: np.ndarray, instruction: str, **forbidden) -> PromptRecord:
        if forbidden:
            raise TypeError(f"prompting accepts only the current image and instruction; got {sorted(forbidden)}")
        image = np.asarray(image)
        if image.shape != (PROMPT_IMAGE_SIZE[1], PROMPT_IMAGE_SIZE[0], 3) or image.dtype != np.uint8:
            raise ValueError("prompt image must be the 448x336 main-band view")
        key = self.key(image, instruction)
        path = self.cache_root / key[:2] / f"{key}.json" if self.cache_root else None
        if path is not None and path.is_file():
            record = PromptRecord(**json.loads(path.read_text()))
            if record.image_sha256 != hashlib.sha256(image.tobytes()).hexdigest():
                raise ValueError(f"cached prompt {key} is corrupt")
            record.cache_hit, record.seconds = True, 0.0
            return record
        started = time.perf_counter()
        attempts, correction, parts = [], None, None
        for _ in range(PROMPT_ATTEMPTS):
            t0 = time.perf_counter()
            text, complete = self.generate_text(image, PROMPT_SYSTEM,
                                                prompt_user_message(instruction, correction),
                                                PROMPT_MAX_NEW_TOKENS)
            row = {"response": text, "complete": bool(complete), "seconds": time.perf_counter() - t0}
            try:
                if not complete:
                    raise ValueError("response hit the token limit")
                parts = parse_prompt_response(text)
                attempts.append(row)
                break
            except (ValueError, TypeError) as exc:
                row["error"] = str(exc)
                attempts.append(row)
                correction = str(exc)
        record = PromptRecord(
            prompt=assemble_prompt(parts) if parts else fallback_prompt(instruction),
            policy=PROMPT_POLICY_VERSION, model_id=self.model_id, model_revision=self.model_revision,
            image_sha256=hashlib.sha256(image.tobytes()).hexdigest(), instruction=_clean(instruction),
            fallback=parts is None, attempts=attempts, seconds=time.perf_counter() - started)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(record), indent=2) + "\n")
            tmp.replace(path)
        return record


PROMPT_SCHEMA = {"type": "object", "additionalProperties": False, "required": list(PROMPT_KEYS),
                 "properties": {k: {"type": "string"} for k in PROMPT_KEYS}}


class RoutedPromptPolicy:
    """Deterministic split across prompt models; slot comes from the plan, never the outcome."""

    def __init__(self, policies: list):
        self.policies = list(policies)

    def __call__(self, image: np.ndarray, instruction: str, *, prompt_slot: int) -> "PromptRecord":
        if not isinstance(prompt_slot, int) or not 0 <= prompt_slot < len(self.policies):
            raise ValueError("prompt_slot must index a registered prompt model")
        return self.policies[prompt_slot](image, instruction)


class OpenRouterPromptBackend:
    """One pinned OpenRouter model; stdlib HTTP so it runs in the tested VPP2 runtime."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: dict, key_file: str | Path, *, timeout: float = 300.0, retries: int = 3):
        self.config = dict(config)
        self.key_file = Path(key_file)
        self.timeout = timeout
        self.retries = retries
        self.log: list[dict] = []

    def request_body(self, image: np.ndarray, system: str, user: str, max_new_tokens: int) -> dict:
        import base64
        import io
        from PIL import Image
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        c = self.config
        body = {"model": c["model"],
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                          {"type": "text", "text": user}]}],
                "reasoning": {"effort": c["reasoning_effort"]},
                "max_tokens": 16000,
                "provider": {"only": list(c["providers"]), "allow_fallbacks": False, "require_parameters": True}}
        if c["seed"] is not None:
            body["seed"] = c["seed"]
        if c["temperature"] is not None:
            body["temperature"] = c["temperature"]
        if c["response_format"] == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "vpp2_prompt", "strict": True, "schema": PROMPT_SCHEMA}}
        else:
            body["response_format"] = {"type": "json_object"}
        return body

    def __call__(self, image: np.ndarray, system: str, user: str, max_new_tokens: int):
        import urllib.error
        import urllib.request
        body = json.dumps(self.request_body(image, system, user, max_new_tokens)).encode()
        key = self.key_file.read_text().strip()
        last = None
        for attempt in range(self.retries):
            request = urllib.request.Request(self.URL, data=body, method="POST", headers={
                "Authorization": "Bearer " + key, "Content-Type": "application/json",
                "X-Title": "RAID VPP2 prompt policy"})
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
                if provider and provider.lower().replace(".", "-").split("/")[0] not in \
                        [p.split("/")[0] for p in self.config["providers"]]:
                    raise RuntimeError(f"unpinned provider served the request: {provider}")
                return message.get("content") or "", choice.get("finish_reason") == "stop"
            self.log.append({"error": last, "attempt": attempt, "seconds": time.perf_counter() - started})
            time.sleep(2 ** attempt)
        return "", False


# ---------------------------------------------------------------- generation

class VPP2Backend:
    """Pinned standalone Stage-1 video model, loaded once (upstream vpp2.inference path)."""

    def __init__(self, weights_root: str | Path, spec: VPP2Spec | None = None, device: str = "cuda"):
        self.spec = spec or VPP2Spec()
        self.weights_root = Path(weights_root)
        self.device = device
        self.model = None
        self.load_seconds = None

    def _args(self):
        from argparse import Namespace
        s = self.spec
        return Namespace(height=s.height, width=s.width, max_area=s.max_area, num_frames=s.num_frames,
                         sampling_steps=s.sampling_steps, sigma_shift=s.sigma_shift,
                         guidance_scale=s.guidance_scale, negative_prompt=s.negative_prompt)

    def load(self) -> "VPP2Backend":
        if self.model is not None:
            return self
        import torch
        from vpp2.inference import DIT_OVERRIDES, check_checkpoint
        from vpp2.models.wan21_14b.wan21 import Wan21VideoModel
        if not torch.cuda.is_available():
            raise RuntimeError("VPP2 generation requires CUDA")
        checkpoint = self.weights_root / self.spec.checkpoint
        wan = self.weights_root / self.spec.wan_model_dir
        started = time.perf_counter()
        self.checkpoint_check = check_checkpoint(checkpoint, wan)
        device = torch.device(self.device if ":" in self.device else "cuda:0")
        torch.cuda.set_device(device)
        model = Wan21VideoModel.from_wan21_14b_pretrained(
            device=str(device), torch_dtype=torch.bfloat16, model_id=str(wan),
            dit_checkpoint_path=str(checkpoint), tokenizer_max_len=512,
            tokenizer_model_id=str(wan / "google/umt5-xxl"), video_dit_config=DIT_OVERRIDES,
            load_text_encoder=True, load_clip_encoder=True,
            video_scheduler=dict(train_shift=3.0, infer_shift=self.spec.sigma_shift, num_train_timesteps=1000,
                                 train_sampling_strategy="shifted_uniform", beta_alpha=7.0, beta_beta=1.0))
        self.model = model.eval().requires_grad_(False)
        self.load_seconds = time.perf_counter() - started
        return self

    def __call__(self, *, image: np.ndarray, prompt: str, seed: int) -> np.ndarray:
        from PIL import Image
        from vpp2.inference import predict_video
        from .world_model import decode_rgb
        self.load()
        if np.asarray(image).shape != (self.spec.height, self.spec.width, 3):
            raise ValueError("VPP2 conditioning must be the composed 416x240 canvas")
        frames = predict_video(self.model, Image.fromarray(np.asarray(image)), prompt, self._args(), int(seed))
        return decode_rgb(frames)


class VPP2WorldModel:
    """Content-addressed VPP2 planning front-end returning controller-band futures."""

    def __init__(self, backend: Callable, prompt_policy: Callable, cache_root: str | Path, *,
                 spec: VPP2Spec | None = None, temporal: VPP2TemporalAlignment | None = None):
        self.backend = backend
        self.prompt_policy = prompt_policy
        self.cache_root = Path(cache_root)
        self.spec = spec or VPP2Spec()
        self.temporal = temporal or VPP2TemporalAlignment()
        self.cost_log: list[dict] = []

    def cache_key(self, canvas: np.ndarray, prompt: str, seed: int) -> str:
        h = hashlib.sha256(np.ascontiguousarray(canvas).tobytes())
        h.update(json.dumps({"shape": list(canvas.shape), "prompt": prompt, "seed": int(seed),
                             "spec": asdict(self.spec)}, sort_keys=True).encode())
        return h.hexdigest()

    @property
    def prompt_slots(self) -> int:
        return len(getattr(self.prompt_policy, "policies", [None]))

    def generate(self, observation: np.ndarray, instruction: str, *, seed: int, prompt_slot: int = 0, **forbidden):
        """observation: current frame in simulator convention (HxWx3 uint8)."""
        from .world_model import GeneratedFuture, write_mp4
        if forbidden:
            raise TypeError(f"generator accepts only the current observation and instruction; got {sorted(forbidden)}")
        observation = np.asarray(observation)
        if observation.ndim != 3 or observation.dtype != np.uint8:
            raise ValueError("generation conditions on exactly one current uint8 observation frame")
        upright = np.ascontiguousarray(observation[::-1])
        canvas, canvas_meta = compose_canvas(upright)
        prompt_record = (self.prompt_policy(prompt_image(upright), instruction, prompt_slot=prompt_slot)
                         if isinstance(self.prompt_policy, RoutedPromptPolicy)
                         else self.prompt_policy(prompt_image(upright), instruction))
        prompt = prompt_record.prompt
        key = self.cache_key(canvas, prompt, seed)
        entry = self.cache_root / key[:2] / key
        frames_path, meta_path = entry / "frames.npz", entry / "metadata.json"
        if frames_path.is_file() and meta_path.is_file():
            with np.load(frames_path) as payload:
                generated = payload["canvas_frames"]
            metadata = json.loads(meta_path.read_text())
            if metadata.get("frames_sha256") != hashlib.sha256(generated.tobytes()).hexdigest():
                raise ValueError(f"cached generated clip {key} is corrupt")
            video_seconds, hit = 0.0, True
        else:
            started = time.perf_counter()
            generated = np.asarray(self.backend(image=canvas, prompt=prompt, seed=int(seed)))
            video_seconds = time.perf_counter() - started
            expected = (self.spec.num_frames, self.spec.height, self.spec.width, 3)
            if generated.shape != expected or generated.dtype != np.uint8:
                raise ValueError(f"VPP2 returned {generated.shape} {generated.dtype}; expected {expected} uint8")
            entry.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(entry / "frames.tmp.npz", canvas_frames=generated)
            (entry / "frames.tmp.npz").replace(frames_path)
            write_mp4(generated, entry / "generated.mp4", self.temporal.display_fps)
            metadata = {"cache_key": key, "seed": int(seed), "prompt": prompt, "instruction": instruction,
                        "prompt_record": asdict(prompt_record), "spec": asdict(self.spec),
                        "canvas": canvas_meta, "conditioning_sha256": hashlib.sha256(canvas.tobytes()).hexdigest(),
                        "frames_sha256": hashlib.sha256(generated.tobytes()).hexdigest(),
                        "num_frames": int(len(generated)), "video_seconds": video_seconds,
                        "native_action_output_used": False,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            tmp = entry / "metadata.tmp.json"
            tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            tmp.replace(meta_path)
            hit = False
        main = extract_main_view(generated)
        left, top, right, bottom = band_box((observation.shape[1], observation.shape[0]))
        frames = generated_to_controller(main, (bottom - top, right - left))
        seconds = video_seconds + prompt_record.seconds
        self.cost_log.append({"cache_key": key, "cache_hit": hit, "video_seconds": video_seconds,
                              "prompt_seconds": prompt_record.seconds,
                              "prompt_cache_hit": prompt_record.cache_hit,
                              "prompt_fallback": prompt_record.fallback})
        return GeneratedFuture(cache_key=key, frames=frames, display_frames=main, seed=int(seed),
                               prompt=prompt, generation_seconds=seconds, cache_hit=hit,
                               metadata={**metadata, "prompt_seconds": prompt_record.seconds,
                                         "prompt_fallback": prompt_record.fallback})


# ---------------------------------------------------------------- protocol

TIMING_CALIBRATION_RULE = {
    "estimator": "per panel case, monotone alignment of reference-MAE features of the 49 generated "
                 "controller-band frames to the recorded source continuation (same band); "
                 "steps_per_frame = least-squares slope through the origin of recorded step vs frame",
    "plausible_case": "0.25 <= steps_per_frame <= 20 and mean aligned cosine similarity >= 0.5",
    "admission": ">= 80% plausible panel cases (12/15); steps_per_frame = median of plausible cases; "
                 "requested_frame = max(1, round(chunk_steps / steps_per_frame)) inside [1, 48]",
    "local_check": "per case, recorded offset best matching the requested generated frame; "
                   "median must lie in [chunk_steps/2, 2*chunk_steps]",
    "unchanged": "H=8, one chunk per replan, 20 Hz control, requested duration 8",
    "limitation": "feature alignment is source-only engineering evidence of frame pacing, "
                  "not a proof of physical executability",
}


VPP2_PANEL = {
    "cases": 15,
    "selection": "every second case (positions 0,2,...,28) of the registered 30-case source panel order: "
                 "3/2/3/2/3/2 cases per source task with alternating early/middle/late starts",
    "authority": "user reduced the panel to 15 cases on 13 October 2026 to save time",
    "gates": {"technical": "15/15", "human_rubric": ">=12/15 named passes",
              "expert_replay": ">=90% of the selected cases (14/15) from the reused source replays",
              "timing": ">=12/15 plausible alignments"},
}


PANEL_FRACTIONS = (0.0, 0.25, 0.5, 0.65, 0.85)  # identical to hybrid_panel.select_panel_cases
PANEL_COVERAGE = {0.0: "early", 0.25: "early_mid", 0.5: "middle", 0.65: "late_mid", 0.85: "late"}


def vpp2_panel_cases(protocol: dict) -> list[dict]:
    """Every second registered source-panel case; dependency-free so it runs in the VPP2 runtime."""
    from .world_model import generation_seed
    chunk = int(protocol["design"]["temporal"]["chunk_steps"])
    base_seed = int(protocol["design"]["generator"]["base_seed"])
    cases = []
    for task_i, task in enumerate(sorted(t for t, s in protocol["tasks"].items() if s["role"] == "source")):
        spec = protocol["tasks"][task]
        demos = list(map(int, spec["source"]["source_validation"]))
        if len(demos) < 5:
            raise ValueError(f"{task} has fewer than 5 source-validation demos")
        for demo_i, demo in enumerate(demos[:5]):
            frac = PANEL_FRACTIONS[demo_i]
            max_t = int(spec["demo_lengths"][str(demo)]) - 1 - chunk
            if max_t < 0:
                raise ValueError(f"{task}:{demo} too short for chunk {chunk}")
            t = int(round(frac * max_t))
            case_id = f"panel:{task}:{demo}:{t}"
            cases.append({"case_id": case_id, "task": task, "demo": demo, "t": t,
                          "instruction": spec["instruction"], "path": spec["path"],
                          "seed": generation_seed(case_id, 0, base_seed), "coverage": PANEL_COVERAGE[frac],
                          "task_index": task_i, "demo_index": demo_i})
    if len(cases) != 30:
        raise ValueError(f"expected 30 registered panel cases; got {len(cases)}")
    return cases[::2][:int(protocol["design"]["vpp2_panel"]["cases"])]


def amend_protocol_vpp2(parent: dict, *, amendment_date: str = "2026-10-13") -> dict:
    """Versioned generator amendment: VPP2 replaces Cosmos3-Edge; the study is unchanged."""
    validate_hybrid_protocol(parent)
    if is_vpp2(parent):
        raise ValueError("parent is already VPP2-amended")
    result = json.loads(json.dumps(parent))
    design = result["design"]
    old_temporal = design["temporal"]
    design["generator"] = asdict(VPP2Spec())
    design["temporal"] = asdict(VPP2TemporalAlignment(
        control_hz=int(old_temporal["control_hz"]), chunk_steps=int(old_temporal["chunk_steps"]),
        chunks_per_plan=int(old_temporal["chunks_per_plan"])))
    design.pop("temporal_calibration_evidence_sha256", None)
    design["controller_view"] = dict(CONTROLLER_VIEW)
    design["timing_calibration_rule"] = dict(TIMING_CALIBRATION_RULE)
    design["vpp2_panel"] = dict(VPP2_PANEL)
    result["source_revision"]["hybrid_modules_sha256"] = v6_code_snapshot_digest()
    result["generator_amendment"] = {
        "name": "vpp2_stage1_generator_v1",
        "date": amendment_date,
        "authority": "user selected VPP2 Stage-1 after reviewing four generated examples",
        "parent_protocol_digest": parent["digest"],
        "parent_model_digest": parent["model_digest"],
        "replaces": "Cosmos3-Edge generator and its uncalibrated 20/24 fps timing",
        "preserved": ["four conditions", "five-shot support", "seed 42", "6/2/2 split",
                      "two development tasks x five starts", "H=8", "one chunk per replan",
                      "frozen generator", "controller-only LoRA+head adaptation", "sealed final tasks"],
        "invalidates": ["generated clips/caches", "frame-feature caches", "source checkpoints",
                        "adapted checkpoints", "panel and pilot reports", "timing evidence"],
        "scientific_admission": False,
        "held_out_claims_eligible": False,
    }
    result["model_digest"] = model_digest(result)
    result["digest"] = digest(result)
    validate_hybrid_protocol(result)
    return result


def controller_view_fn(protocol: dict) -> Callable[[np.ndarray], np.ndarray]:
    """Frame transform for every controller/reference encoding under this protocol."""
    view = protocol.get("design", {}).get("controller_view")
    if view is None:
        return lambda frames: frames
    if view.get("name") != CONTROLLER_VIEW["name"]:
        raise ValueError(f"unknown controller view {view.get('name')}")
    return controller_band


def generation_seed_for(protocol: dict, case_id: str, replan_index: int) -> int:
    from .world_model import generation_seed
    return generation_seed(case_id, replan_index, int(protocol["design"]["generator"]["base_seed"]))


# ---------------------------------------------------------------- service
#
# VPP2 runs in its exact tested runtime; LIBERO and the controller run in the
# RAID runtime. One localhost service serializes GPU generation and owns the
# content-addressed cache; the client reads verified clips from that cache.

class GeneratorFailure(RuntimeError):
    """Infrastructure failure of the frozen generator. Never a robot outcome:
    the rollout re-raises it so the episode stays incomplete, not failed."""


class GeneratorUnavailable(GeneratorFailure):
    """Service unreachable, timed out or returned an error; retry after a restart."""


class GeneratorIntegrityError(GeneratorFailure):
    """A served clip does not match the request or the cache; stop, do not retry."""


# The client waits longer than the server's worst case: every prompt attempt
# exhausting its OpenRouter retries, plus generation and decoding.
SERVICE_TIMEOUT_SECONDS = 3600.0

def _service_identity(world_model: VPP2WorldModel, protocol_digest: str | None) -> dict:
    return {"spec": asdict(world_model.spec), "protocol_digest": protocol_digest,
            "prompt_policy": PROMPT_POLICY_VERSION, "cache_root": str(world_model.cache_root.resolve())}


def serve(world_model: VPP2WorldModel, *, host: str = "127.0.0.1", port: int = 8765,
          protocol_digest: str | None = None, ready_path: str | Path | None = None) -> None:
    import base64
    from http.server import BaseHTTPRequestHandler, HTTPServer
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("the VPP2 service listens on localhost only")
    identity = _service_identity(world_model, protocol_digest)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body, sort_keys=True, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": "unknown path"})
            self._send(200, {**identity, "ok": True})

        def do_POST(self):
            if self.path != "/generate":
                return self._send(404, {"error": "unknown path"})
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if set(request) != {"observation_b64", "shape", "instruction", "seed", "prompt_slot"}:
                    raise TypeError("request carries fields beyond the current observation, instruction, seed and slot")
                shape = tuple(int(n) for n in request["shape"])
                observation = np.frombuffer(base64.b64decode(request["observation_b64"]),
                                            dtype=np.uint8).reshape(shape)
                plan = world_model.generate(observation, str(request["instruction"]), seed=int(request["seed"]),
                                            prompt_slot=int(request["prompt_slot"]))
                self._send(200, {"cache_key": plan.cache_key, "cache_hit": plan.cache_hit,
                                 "generation_seconds": plan.generation_seconds, "prompt": plan.prompt,
                                 "metadata": plan.metadata})
            except Exception as exc:  # reported to the client, which fails the episode
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args):
            return

    server = HTTPServer((host, port), Handler)
    if ready_path:
        Path(ready_path).write_text(json.dumps({**identity, "port": port}) + "\n")
    server.serve_forever()


class VPP2ServiceClient:
    """World-model interface for the RAID runtime, backed by the localhost service."""

    def __init__(self, url: str, cache_root: str | Path, *, protocol: dict,
                 timeout: float = SERVICE_TIMEOUT_SECONDS):
        self.url = url.rstrip("/")
        self.cache_root = Path(cache_root)
        self.protocol = protocol
        self.spec = VPP2Spec(**protocol["design"]["generator"])
        self.temporal = VPP2TemporalAlignment(**protocol["design"]["temporal"])
        self.timeout = timeout

    def _request(self, path: str, body: dict | None = None) -> dict:
        import urllib.error
        import urllib.request
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, method="POST" if data else "GET",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise GeneratorUnavailable(f"VPP2 service error: {error.read().decode(errors='replace')[:2000]}") from None
        except (OSError, ValueError) as error:  # refused, reset, timeout, truncated reply
            raise GeneratorUnavailable(f"VPP2 service unreachable: {type(error).__name__}: {error}") from None

    def check(self) -> dict:
        health = self._request("/health")
        if (health.get("spec") != asdict(self.spec) or health.get("protocol_digest") != self.protocol["digest"]
                or Path(health.get("cache_root", "")).resolve() != self.cache_root.resolve()):
            raise GeneratorIntegrityError("VPP2 service does not serve this protocol, generator spec or cache")
        return health

    prompt_slots = len(PROMPT_MODELS)

    def generate(self, observation: np.ndarray, instruction: str, *, seed: int, prompt_slot: int = 0, **forbidden):
        import base64
        from .world_model import GeneratedFuture
        if forbidden:
            raise TypeError(f"generator accepts only the current observation and instruction; got {sorted(forbidden)}")
        observation = np.ascontiguousarray(observation, dtype=np.uint8)
        if observation.ndim != 3:
            raise ValueError("generation conditions on exactly one current observation frame")
        reply = self._request("/generate", {"observation_b64": base64.b64encode(observation.tobytes()).decode(),
                                            "shape": list(observation.shape), "instruction": instruction,
                                            "seed": int(seed), "prompt_slot": int(prompt_slot)})
        key = reply["cache_key"]
        entry = self.cache_root / key[:2] / key
        try:
            with np.load(entry / "frames.npz") as payload:
                generated = payload["canvas_frames"]
            metadata = json.loads((entry / "metadata.json").read_text())
        except (OSError, ValueError, KeyError) as error:
            raise GeneratorIntegrityError(f"VPP2 cache entry {key} is missing or unreadable: {error}") from None
        if (metadata.get("frames_sha256") != hashlib.sha256(generated.tobytes()).hexdigest()
                or metadata.get("seed") != int(seed) or metadata.get("instruction") != instruction
                or metadata.get("spec") != asdict(self.spec)):
            raise GeneratorIntegrityError(f"VPP2 cache entry {key} does not match the request")
        canvas, _ = compose_canvas(np.ascontiguousarray(observation[::-1]))
        if metadata.get("conditioning_sha256") != hashlib.sha256(canvas.tobytes()).hexdigest():
            raise GeneratorIntegrityError(f"VPP2 cache entry {key} was conditioned on another observation")
        main = extract_main_view(generated)
        left, top, right, bottom = band_box((observation.shape[1], observation.shape[0]))
        frames = generated_to_controller(main, (bottom - top, right - left))
        return GeneratedFuture(cache_key=key, frames=frames, display_frames=main, seed=int(seed),
                               prompt=reply["prompt"], generation_seconds=float(reply["generation_seconds"]),
                               cache_hit=bool(reply["cache_hit"]), metadata=reply["metadata"])


def load_production_world_model(weights_root: str | Path, key_file: str | Path, cache_root: str | Path,
                                protocol: dict, device: str = "cuda") -> VPP2WorldModel:
    """Pinned VPP2 in the tested runtime plus the registered OpenRouter prompt split."""
    if not is_vpp2(protocol):
        raise ValueError("protocol is not VPP2-amended")
    spec = VPP2Spec(**protocol["design"]["generator"])
    if spec.prompt_models != ",".join(m["model"] for m in PROMPT_MODELS) or spec.prompt_policy != PROMPT_POLICY_VERSION:
        raise ValueError("protocol prompt policy differs from this implementation")
    backend = VPP2Backend(weights_root, spec, device).load()
    policies = [ObservationPromptPolicy(OpenRouterPromptBackend(c, key_file), Path(cache_root) / "prompts",
                                        model_id=c["model"], model_revision=PROMPT_MODEL_REVISION)
                for c in PROMPT_MODELS]
    world = VPP2WorldModel(backend, RoutedPromptPolicy(policies), Path(cache_root) / "clips", spec=spec,
                           temporal=VPP2TemporalAlignment(**protocol["design"]["temporal"]))
    world.load_seconds = {"video": backend.load_seconds}
    return world


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="VPP2 generation service (runs in the VPP2 runtime)")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--protocol", required=True)
    s.add_argument("--weights", required=True)
    s.add_argument("--prompt-key", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--ready", required=True)
    a = parser.parse_args(argv)
    protocol = json.loads(Path(a.protocol).read_text())
    validate_hybrid_protocol(protocol)
    world = load_production_world_model(a.weights, a.prompt_key, a.cache, protocol)
    serve(world, port=a.port, protocol_digest=protocol["digest"], ready_path=a.ready)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
