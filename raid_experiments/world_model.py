"""Frozen visual world model: Cosmos3-Edge image-to-video generation.

The generator conditions only on the current RGB observation and the task
instruction. It never receives held-out future frames, recorded actions or an
action trajectory, and its native action output is never used by either arm.

Generated clips are content-addressed: the cache key covers the conditioning
image bytes, prompt, negative prompt, seed, every generation setting and the
pinned checkpoint revision. Two arms that present the same observation with
the same seed therefore reuse one clip, and each stored clip keeps a viewable
MP4 next to its decoded uint8 frames and metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import numpy as np

from .wm_protocol import GeneratorSpec, TemporalAlignment, assert_no_prohibited_model
from .hybrid_generator_profile import render_cosmos_prompt

LIBERO_IMAGE_CONVENTION = "opengl"  # LIBERO HDF5 frames and renders are vertically flipped


def to_display(frame: np.ndarray, convention: str = LIBERO_IMAGE_CONVENTION) -> np.ndarray:
    """Simulator/HDF5 convention -> upright RGB used for generation and videos."""
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("expected one HxWx3 uint8 RGB frame")
    return np.ascontiguousarray(frame[::-1]) if convention == "opengl" else frame


def from_display(frame: np.ndarray, convention: str = LIBERO_IMAGE_CONVENTION) -> np.ndarray:
    """Upright RGB -> simulator/HDF5 convention consumed by the inverse model."""
    return to_display(frame, convention)


def resize_rgb(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    frames = np.asarray(frames)
    single = frames.ndim == 3
    batch = frames[None] if single else frames
    if tuple(batch.shape[1:3]) == (size[0], size[1]):
        out = np.ascontiguousarray(batch)
        return out[0] if single else out
    out = np.stack([np.asarray(Image.fromarray(f).resize((size[1], size[0]), Image.BICUBIC)) for f in batch])
    return out[0] if single else out


@dataclass(frozen=True)
class GeneratedFuture:
    """Decoded generated clip in simulator convention at the inverse-model resolution."""
    cache_key: str
    frames: np.ndarray            # (num_frames, H, W, 3) uint8, simulator convention
    display_frames: np.ndarray    # (num_frames, h, w, 3) uint8 upright generator output
    seed: int
    prompt: str
    generation_seconds: float
    cache_hit: bool
    metadata: dict

    def subgoal(self, frame_index: int) -> np.ndarray:
        if not 0 <= frame_index < len(self.frames):
            raise IndexError(f"generated frame {frame_index} outside clip of {len(self.frames)}")
        return self.frames[frame_index]


class VideoBackend(Protocol):
    def __call__(self, *, image: np.ndarray, prompt: str, negative_prompt: str, seed: int,
                 spec: GeneratorSpec, temporal: TemporalAlignment) -> np.ndarray: ...


class Cosmos3EdgeBackend:
    """Loads the pinned Cosmos3OmniPipeline and runs image-to-video only."""

    def __init__(self, spec: GeneratorSpec | None = None, *, device: str = "cuda",
                 cache_dir: str | None = None, local_files_only: bool = False):
        self.spec = spec or GeneratorSpec()
        assert_no_prohibited_model(self.spec.model_id)
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._pipe = None
        self.negative_prompt = None
        self.load_seconds = None

    def _check_runtime(self) -> None:
        import diffusers
        import torch
        version = getattr(diffusers, "__version__", "")
        if not torch.cuda.is_available():
            raise RuntimeError("Cosmos3-Edge generation requires a CUDA GPU (Ampere or newer, BF16)")
        if torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError("Cosmos3-Edge requires Ampere (sm80) or newer for BF16")
        if not hasattr(diffusers, "Cosmos3OmniPipeline"):
            raise RuntimeError(f"diffusers {version} lacks Cosmos3OmniPipeline; install commit {self.spec.diffusers_commit}")

    def load(self) -> "Cosmos3EdgeBackend":
        if self._pipe is not None:
            return self
        self._check_runtime()
        import torch
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from huggingface_hub import hf_hub_download
        started = time.perf_counter()
        pipe = Cosmos3OmniPipeline.from_pretrained(
            self.spec.model_id, revision=self.spec.revision, torch_dtype=torch.bfloat16,
            enable_safety_checker=self.spec.enable_safety_check, cache_dir=self.cache_dir,
            local_files_only=self.local_files_only)
        pipe.to(self.device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=self.spec.flow_shift, use_karras_sigmas=False)
        negative = hf_hub_download(self.spec.model_id, "assets/negative_prompt.json",
                                   revision=self.spec.revision, cache_dir=self.cache_dir,
                                   local_files_only=self.local_files_only)
        self.negative_prompt = json.dumps(json.loads(Path(negative).read_text()))
        self._pipe = pipe
        self.load_seconds = time.perf_counter() - started
        return self

    def __call__(self, *, image, prompt, negative_prompt, seed, spec, temporal):
        import torch
        self.load()
        from PIL import Image
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        result = self._pipe(
            prompt=prompt, negative_prompt=negative_prompt, image=Image.fromarray(image),
            num_frames=temporal.num_frames, height=spec.height, width=spec.width,
            fps=float(temporal.generation_fps), num_inference_steps=spec.num_inference_steps,
            guidance_scale=spec.guidance_scale, generator=generator, output_type="np",
            add_resolution_template=spec.add_resolution_template,
            add_duration_template=spec.add_duration_template,
            enable_safety_check=spec.enable_safety_check, use_system_prompt=None)
        if getattr(result, "action", None) is not None:
            raise RuntimeError("image-to-video run returned native actions; refusing generator action output")
        return decode_rgb(result.video)


def decode_rgb(video: Any) -> np.ndarray:
    """Pipeline video output (np [T,H,W,C] float 0-1, uint8, or PIL list) -> uint8."""
    if isinstance(video, list) and video and hasattr(video[0], "size"):
        array = np.stack([np.asarray(frame.convert("RGB")) for frame in video])
    else:
        array = np.asarray(video)
        if array.ndim == 5 and array.shape[0] == 1:
            array = array[0]
        if array.dtype != np.uint8:
            if not np.isfinite(array).all():
                raise ValueError("generated video contains non-finite values")
            array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"generated video must be (T,H,W,3); got {array.shape}")
    return array


def write_mp4(frames: np.ndarray, path: Path, fps: int) -> None:
    import imageio.v3 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    iio.imwrite(tmp, np.asarray(frames, dtype=np.uint8), fps=fps, codec="libx264",
                macro_block_size=1)
    tmp.replace(path)


def generation_seed(case_id: str, replan_index: int, base_seed: int) -> int:
    """Seed shared by both arms for the same case and replanning index."""
    text = f"{base_seed}\x1f{case_id}\x1f{int(replan_index)}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


class FrozenWorldModel:
    """Generation front-end with a content-addressed clip cache."""

    def __init__(self, backend: VideoBackend | Callable, cache_root: str | Path, *,
                 spec: GeneratorSpec | None = None, temporal: TemporalAlignment | None = None,
                 negative_prompt: str | None = None, idm_resolution: tuple[int, int] = (128, 128)):
        self.backend = backend
        self.cache_root = Path(cache_root)
        self.spec = spec or GeneratorSpec()
        self.temporal = temporal or TemporalAlignment()
        self.negative_prompt = negative_prompt
        self.idm_resolution = idm_resolution
        self.cost_log: list[dict] = []

    def _negative(self) -> str:
        if self.negative_prompt is not None:
            return self.negative_prompt
        loaded = getattr(self.backend, "negative_prompt", None)
        if loaded is None and hasattr(self.backend, "load"):
            self.backend.load()
            loaded = getattr(self.backend, "negative_prompt", None)
        if loaded is None:
            raise RuntimeError("negative prompt unavailable from the pinned model revision")
        return loaded

    def prompt(self, instruction: str) -> str:
        return render_cosmos_prompt(self.spec.prompt_template, instruction)

    def cache_key(self, display_image: np.ndarray, prompt: str, negative: str, seed: int) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(display_image).tobytes())
        digest.update(json.dumps({"shape": list(display_image.shape), "prompt": prompt,
                                  "negative_sha256": hashlib.sha256(negative.encode()).hexdigest(),
                                  "seed": int(seed), "spec": asdict(self.spec),
                                  "temporal": {k: v for k, v in asdict(self.temporal).items()
                                               if k in {"num_frames", "generation_fps"}}},
                                 sort_keys=True).encode())
        return digest.hexdigest()

    def generate(self, observation: np.ndarray, instruction: str, *, seed: int, **forbidden) -> GeneratedFuture:
        """observation: current frame in simulator convention (HxWx3 uint8)."""
        if forbidden:
            raise TypeError(f"generator accepts only the current observation and instruction; got {sorted(forbidden)}")
        observation = np.asarray(observation)
        if observation.ndim != 3:
            raise ValueError("generation conditions on exactly one current observation frame")
        display = resize_rgb(to_display(observation), (self.spec.height, self.spec.width))
        prompt = self.prompt(instruction)
        negative = self._negative()
        key = self.cache_key(display, prompt, negative, seed)
        entry = self.cache_root / key[:2] / key
        frames_path, meta_path = entry / "frames.npz", entry / "metadata.json"
        if frames_path.is_file() and meta_path.is_file():
            with np.load(frames_path) as payload:
                generated = payload["display_frames"]
            metadata = json.loads(meta_path.read_text())
            if metadata.get("frames_sha256") != hashlib.sha256(generated.tobytes()).hexdigest():
                raise ValueError(f"cached generated clip {key} is corrupt")
            seconds, hit = 0.0, True
        else:
            started = time.perf_counter()
            generated = self.backend(image=display, prompt=prompt, negative_prompt=negative, seed=int(seed),
                                     spec=self.spec, temporal=self.temporal)
            seconds = time.perf_counter() - started
            generated = decode_rgb(generated)
            if len(generated) != self.temporal.num_frames:
                raise ValueError(f"generator returned {len(generated)} frames; expected {self.temporal.num_frames}")
            entry.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(entry / "frames.tmp.npz", display_frames=generated)
            (entry / "frames.tmp.npz").replace(frames_path)
            write_mp4(generated, entry / "generated.mp4", self.temporal.generation_fps)
            metadata = {"cache_key": key, "seed": int(seed), "prompt": prompt, "instruction": instruction,
                        "negative_prompt_sha256": hashlib.sha256(negative.encode()).hexdigest(),
                        "spec": asdict(self.spec), "num_frames": int(len(generated)),
                        "fps": self.temporal.generation_fps,
                        "conditioning_sha256": hashlib.sha256(display.tobytes()).hexdigest(),
                        "frames_sha256": hashlib.sha256(generated.tobytes()).hexdigest(),
                        "generation_seconds": seconds, "native_action_output_used": False,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            tmp = entry / "metadata.tmp.json"
            tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            tmp.replace(meta_path)
            hit = False
        sim_frames = np.stack([from_display(f) for f in resize_rgb(generated, self.idm_resolution)])
        self.cost_log.append({"cache_key": key, "seconds": seconds, "cache_hit": hit})
        return GeneratedFuture(cache_key=key, frames=sim_frames, display_frames=generated, seed=int(seed),
                               prompt=prompt, generation_seconds=seconds, cache_hit=hit, metadata=metadata)
