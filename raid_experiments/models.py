"""Pilot decoder models and a small, auditable LoRA implementation."""

from __future__ import annotations

import math
from typing import Iterable
import torch
from torch import nn


class LowRankLinear(nn.Module):
    """Frozen linear layer plus a zero-initialized LoRA update."""
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        # The adapter is installed after the pretrained backbone has already
        # been moved to its execution device (and may be in reduced precision).
        # Construct it alongside the base layer so the first forward pass does
        # not mix CPU/GPU tensors or silently upcast the model.
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale
        return self.base(x) + update


class DirectDecoder(nn.Module):
    def __init__(self, feature_dim=384, action_dim=7, hidden_dim=512, bottleneck_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, action_dim),
        )

    def forward(self, current, future, retrieved=None, valid=None):
        return self.net(torch.cat((current, future), dim=-1)), None


class RAIDDecoder(nn.Module):
    def __init__(self, feature_dim=384, action_dim=7, hidden_dim=512, bottleneck_dim=256, dropout=0.1, prior_dropout=0.4, prior_noise=0.05):
        super().__init__()
        dim = feature_dim * 2
        self.q_proj = nn.Linear(dim, 64)
        self.k_proj = nn.Linear(action_dim, 64)
        self.gate = nn.Linear(dim, action_dim)
        self.direct = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, action_dim),
        )
        self.prior_dropout = nn.Dropout(prior_dropout)
        self.prior_noise = prior_noise

    def forward(self, current, future, retrieved, valid=None):
        if retrieved is None:
            return self.direct(torch.cat((current, future), dim=-1)), None
        x = torch.cat((current, future), dim=-1)
        direct = self.direct(x)
        scores = (self.q_proj(x).unsqueeze(1) @ self.k_proj(retrieved).transpose(-1, -2)) / 8.0
        if valid is not None:
            scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
            scores = scores.masked_fill(~torch.isfinite(scores), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        if valid is not None:
            weights = weights.masked_fill(~valid.unsqueeze(1), 0.0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        prior = (weights @ retrieved).squeeze(1)
        if self.training:
            prior = self.prior_dropout(prior) + torch.randn_like(prior) * self.prior_noise
        gate = torch.sigmoid(self.gate(x))
        out = gate * direct + (1 - gate) * prior
        if valid is not None:
            out = torch.where(valid.any(dim=-1, keepdim=True), out, direct)
        return out, weights.squeeze(1)


def attach_feature_lora(module: nn.Module, rank: int, alpha: int, dropout: float) -> list[str]:
    """Attach LoRA to a decoder's first feature projection for cached-feature smoke tests.

    This helper is limited to synthetic cached-feature checks. Scientific
    image adaptation uses ``attach_mae_lora`` in the standalone-MAE study.
    """
    names = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.in_features >= 700:
            setattr(module, name, LowRankLinear(child, rank, alpha, dropout))
            names.append(name)
            break
        names.extend(f"{name}.{x}" for x in attach_feature_lora(child, rank, alpha, dropout))
    return names


def attach_mae_lora(mae: nn.Module, blocks: Iterable[int], rank: int, alpha: int, dropout: float) -> list[str]:
    """Attach LoRA to final MAE ViT attention qkv/proj projections."""
    for parameter in mae.parameters():
        parameter.requires_grad_(False)
    stack = getattr(mae, "blocks", None)
    if stack is None:
        raise AttributeError("MAE encoder has no ViT blocks")
    found = []
    for block_idx in blocks:
        block = stack[block_idx]
        attn = getattr(block, "attn", None)
        if attn is None:
            raise AttributeError(f"MAE block {block_idx} has no attention module")
        for target in ("qkv", "proj"):
            candidate = getattr(attn, target, None)
            if not isinstance(candidate, nn.Linear):
                raise AttributeError(f"MAE block {block_idx} missing linear attention target {target}")
            setattr(attn, target, LowRankLinear(candidate, rank, alpha, dropout))
            found.append(f"blocks.{block_idx}.attn.{target}")
    return found


def trainable_parameter_names(module: nn.Module) -> list[str]:
    return [name for name, p in module.named_parameters() if p.requires_grad]
