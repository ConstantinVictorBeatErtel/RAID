"""Experimental predictor adapters retained for separate future-model work.

They are not part of the active recorded-frame study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class FuturePrediction:
    feature: torch.Tensor
    timestamp: float
    representation: str
    action_conditioned: bool
    metadata: dict[str, Any]


class FuturePredictor(Protocol):
    def predict(self, images: torch.Tensor, *, language: list[str] | None = None,
                robot_state: torch.Tensor | None = None) -> FuturePrediction: ...


class Cosmos3EdgeFuturePredictor:
    """Adapter boundary for the pinned Cosmos 3 Edge runtime.

    Cosmos releases multiple inference modes and runtime packages. Requiring a
    supplied callable prevents accidentally substituting a native policy or an
    action-conditioned forward model while presenting it as RAID's predictor.
    """
    def __init__(self, generate_future, *, horizon: int = 3, fps: int = 20, generation_frames: int = 50):
        if not callable(generate_future):
            raise TypeError("Cosmos3EdgeFuturePredictor requires a pinned, tested generation callable")
        self.generate_future = generate_future
        self.horizon = int(horizon)
        self.fps = int(fps)
        self.generation_frames = int(generation_frames)
        if not 12 <= self.fps <= 30:
            raise ValueError("Cosmos 3 Edge video generation supports 12-30 fps")
        if not 50 <= self.generation_frames <= 150:
            raise ValueError("Cosmos 3 Edge video generation supports 50-150 frames")

    def predict(self, images, *, language=None, robot_state=None):
        result = self.generate_future(images=images, language=language, robot_state=robot_state,
                                      # Cosmos Edge requires 50-150 generated
                                      # frames; the decoder consumes the frame
                                      # matched to the requested timestamp.
                                      num_frames=self.generation_frames, fps=self.fps)
        if isinstance(result, FuturePrediction):
            if result.action_conditioned:
                raise ValueError("Cosmos action-conditioned futures are excluded from the main RAID comparison")
            return result
        if not isinstance(result, torch.Tensor):
            raise TypeError("Cosmos generation callable must return a feature tensor or FuturePrediction")
        return FuturePrediction(feature=result, timestamp=self.horizon / self.fps,
                                representation="cosmos3_edge_rgb", action_conditioned=False,
                                metadata={"horizon": self.horizon, "fps": self.fps, "num_frames": self.generation_frames})


def build_predictor(profile: str, *, encoder=None, generate_future=None, horizon: int = 3, fps: int = 20):
    if profile == "cosmos3_edge":
        return Cosmos3EdgeFuturePredictor(generate_future, horizon=horizon, fps=fps)
    raise ValueError(f"unsupported predictor profile: {profile}")
