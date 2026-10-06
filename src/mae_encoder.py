"""Standalone MAE ViT-B/16 image encoder used by the recorded-frame study."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize
import timm


FEATURE_DIM = 768
PREPROCESSING_VERSION = "mae_vit_b16_resize224_bicubic_imagenet_v1"
MAE_IMPLEMENTATION = f"timm=={timm.__version__}:vit_base_patch16_224:normalized_cls_v1"
MAE_PREPROCESSING = {"image_size": 224, "patch_size": 16, "interpolation": "bicubic",
    "align_corners": False, "antialias": True, "input_range": "uint8_0_255_or_float_0_1",
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class MAEImageEncoder(nn.Module):
    """MAE ViT-B/16 normalized CLS features with shared RGB preprocessing."""

    def __init__(self, mae: nn.Module, device: str | torch.device = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.mae = mae.to(self.device).eval()
        self.feat_dim = FEATURE_DIM
        for parameter in self.mae.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path,
                        device: str | torch.device = "cuda") -> "MAEImageEncoder":
        model = timm.models.vision_transformer.VisionTransformer(
            img_size=224, patch_size=16, embed_dim=FEATURE_DIM, depth=12,
            num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=0,
            global_pool="", norm_layer=partial(nn.LayerNorm, eps=1e-6))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("state_dict", payload))
        if not isinstance(state, dict):
            raise ValueError("MAE checkpoint does not contain a model state dictionary")
        model_keys = set(model.state_dict())
        candidates = [state]
        for prefix in ("module.", "model."):
            candidates.append({k[len(prefix):] if k.startswith(prefix) else k: v
                               for k, v in state.items()})
        state = max(candidates, key=lambda item: len(set(item) & model_keys))
        encoder_state = {key: value for key, value in state.items() if key in model_keys}
        missing = model_keys - set(encoder_state)
        unexpected_encoder = {key for key in state if key not in model_keys
                              and not key.startswith(("decoder_", "decoder."))
                              and key not in {"mask_token"}}
        if missing or unexpected_encoder:
            raise RuntimeError(f"MAE encoder checkpoint mismatch: missing={sorted(missing)[:8]} "
                               f"unexpected={sorted(unexpected_encoder)[:8]}")
        for key, value in encoder_state.items():
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ValueError(f"MAE checkpoint contains invalid encoder tensor {key}")
        model.load_state_dict(encoder_state, strict=True)
        return cls(model, device)

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError("RGB images must have rank 3 or 4")
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        if images.shape[1] != 3:
            raise ValueError("RGB images must have three channels")
        images = images.to(self.device)
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        else:
            images = images.float()
        images = F.interpolate(images, size=(224, 224), mode="bicubic",
                              align_corners=False, antialias=True)
        return normalize(images, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        output = self.mae(self._preprocess_images(images))
        output = output[0] if isinstance(output, (tuple, list)) else output
        if output.ndim == 3:
            output = output[:, 0]
        if output.ndim != 2 or output.shape[-1] != FEATURE_DIM:
            raise ValueError(f"MAE normalized CLS output must have shape (batch, {FEATURE_DIM})")
        return output.float()
