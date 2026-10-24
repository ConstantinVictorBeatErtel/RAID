"""Frozen image encoders for RAID v8: MAE ViT-B/16 (primary), DINOv2 ViT-B/14, SigLIP ViT-B/16. All 768-d."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..hybrid_protocol import assert_no_prohibited_model

ENCODERS = {
    "mae": {"source": "data/assets/mae/mae_pretrain_vit_base.pth", "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)},
    "dinov2": {"source": "facebook/dinov2-base", "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "siglip": {"source": "google/siglip-base-patch16-224", "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
}
FEATURE_DIM = 768
IMAGE_SIZE = 224


class FrozenEncoder(nn.Module):
    """uint8 NHWC frames -> (N, 768) features. ``model`` is the backbone (LoRA attaches to MAE's ``model``)."""

    def __init__(self, name: str, model: nn.Module, device: str):
        super().__init__()
        self.name, self.model, self.device = name, model.to(device).eval(), torch.device(device)
        spec = ENCODERS[name]
        self.register_buffer("mean", torch.tensor(spec["mean"], device=device).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(spec["std"], device=device).view(1, 3, 1, 1), persistent=False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.feat_dim = FEATURE_DIM

    @property
    def mae(self) -> nn.Module:  # name used by the LoRA helpers of the v6 code
        return self.model

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        images = images.to(self.device)
        images = images.float().div_(255.0) if images.dtype == torch.uint8 else images.float()
        images = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=False,
                               antialias=True)
        return (images - self.mean) / self.std

    def forward_features(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.name == "mae":
            out = self.model(pixels)
            out = out[0] if isinstance(out, (tuple, list)) else out
            return out[:, 0] if out.ndim == 3 else out
        return self.model(pixel_values=pixels).pooler_output

    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(self.preprocess(images)).float()


def load(name: str, device: str, *, root: str | Path = ".") -> tuple[FrozenEncoder, dict]:
    """Return the encoder and an identity record (source, revision)."""
    spec = ENCODERS[name]
    assert_no_prohibited_model(spec["source"])
    if name == "mae":
        from src.mae_encoder import MAEImageEncoder
        path = Path(root) / spec["source"]
        model = MAEImageEncoder.from_checkpoint(path, device="cpu").mae
        from .data import sha256_file
        identity = {"source": spec["source"], "sha256": sha256_file(path)}
    else:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(spec["source"]).sha
        if name == "dinov2":
            from transformers import Dinov2Model
            model = Dinov2Model.from_pretrained(spec["source"], revision=revision)
        else:
            from transformers import SiglipVisionModel
            model = SiglipVisionModel.from_pretrained(spec["source"], revision=revision)
        identity = {"source": spec["source"], "revision": revision}
    return FrozenEncoder(name, model, device), identity


@torch.no_grad()
def encode(encoder: FrozenEncoder, frames: np.ndarray, batch: int = 256) -> torch.Tensor:
    out = []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if encoder.device.type == "cuda" else torch.autocast(
        "cpu", enabled=False)
    for start in range(0, len(frames), batch):
        chunk = torch.from_numpy(np.ascontiguousarray(frames[start:start + batch]))
        with autocast:
            out.append(encoder.encode_frames(chunk).float().cpu())
    return torch.cat(out)
