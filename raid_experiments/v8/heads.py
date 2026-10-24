"""RAID v8 head: the v7 attention head with a disagreement-aware gate.

    direct = MLP(s)                                   s = [cur, fut, q, duration]
    prior  = sum_i alpha_i a_i                        alpha from cross-attention over k = 16 retrieved candidates
    c      = [top-1 sim, mean valid sim, attention entropy, |direct - prior| (detached, per element)]
    g      = sigmoid(W_g [s, c])                      per action element; g = 1 without valid candidates
    action = g * direct + (1 - g) * prior

Training (source and adaptation): with probability ``corrupt`` a row's direct prediction is replaced, inside the
mix only, by the direct prediction of another row of the batch (detached). The gate sees a large disagreement
on such rows and is trained to take the prior, which is the behaviour needed when the source model is wrong on a
new task.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from ..hybrid_heads import _trunk
from ..hybrid_protocol import ACTION_DIM, HeadSpec
from ..hybrid_v7 import ATTENTION_DIM, CANDIDATE_JITTER, AttentionGateController, v7_retrieval
from ..hybrid_protocol import RetrievalSpec

HEAD_KIND = "disagreement_gate_v8"
CONFIDENCE_SCALARS = 3


class DisagreementGateController(AttentionGateController):
    head_kind = HEAD_KIND

    def __init__(self, chunk_steps: int, feature_dim: int, proprio_dim: int, *, corrupt: float = 0.25,
                 spec: HeadSpec | None = None, retrieval: RetrievalSpec | None = None):
        super().__init__(chunk_steps, feature_dim, proprio_dim, spec=spec, retrieval=retrieval or v7_retrieval(RetrievalSpec()))
        out = chunk_steps * ACTION_DIM
        state_dim = 2 * feature_dim + proprio_dim + 1
        self.corrupt = float(corrupt)
        self.gate = nn.Sequential(nn.Linear(state_dim + CONFIDENCE_SCALARS + out, 256), nn.ReLU(), nn.Linear(256, out))

    def forward_with_memory(self, current, future, q, duration, *, bank=None, use_retrieval: bool,
                            exclude_groups=None, query_current=None, query_future=None):
        state = self._state(current, future, q, duration)
        direct = self.trunk(state)
        if not use_retrieval:
            return direct.view(-1, self.chunk_steps, ACTION_DIM), {"null_memory": True}
        result = bank.retrieve(current if query_current is None else query_current,
                               future if query_future is None else query_future, q, duration,
                               spec=self.retrieval, exclude_groups=exclude_groups)
        valid = result.valid
        idx = result.indices.clamp_min(0)
        sims = torch.where(valid, result.similarities, torch.zeros_like(result.similarities))
        keys = self.key(torch.cat((F.normalize(bank.current[idx], dim=-1), F.normalize(bank.future[idx], dim=-1),
                                   sims.unsqueeze(-1)), dim=-1))
        logits = (self.query(state).unsqueeze(1) * keys).sum(-1) / math.sqrt(ATTENTION_DIM)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1) * valid.float()
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        actions = result.actions
        if self.training and CANDIDATE_JITTER > 0:
            actions = actions + CANDIDATE_JITTER * torch.randn_like(actions) * valid.unsqueeze(-1).float()
        prior = (attention.unsqueeze(-1) * actions).sum(1)
        mixed_direct = direct
        corrupted = None
        if self.training and self.corrupt > 0 and len(direct) > 1:
            corrupted = torch.rand(len(direct), device=direct.device) < self.corrupt
            other = direct.detach()[torch.randperm(len(direct), device=direct.device)]
            mixed_direct = torch.where(corrupted.unsqueeze(-1), other, direct)
        nvalid = valid.float().sum(-1).clamp_min(1.0)
        top1 = sims[:, 0]
        mean_sim = sims.sum(-1) / nvalid
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1) / math.log(max(2, valid.shape[-1]))
        disagreement = (mixed_direct.detach() - prior.detach()).abs()
        conf = torch.cat((top1.unsqueeze(-1), mean_sim.unsqueeze(-1), entropy.unsqueeze(-1), disagreement), -1)
        available = valid.any(-1, keepdim=True).float()
        gate = torch.sigmoid(self.gate(torch.cat((state, conf), -1)))
        gate = available * gate + (1.0 - available)
        out = gate * mixed_direct + (1.0 - gate) * prior
        info = {"null_memory": False, "attention": attention, "candidate_actions": result.actions, "valid": valid,
                "indices": result.indices, "gate": gate.detach(), "gate_mean": float(gate.mean().detach()),
                "corrupted": corrupted}
        return out.view(-1, self.chunk_steps, ACTION_DIM), info


def build(kind: str, chunk: int, feature_dim: int, proprio_dim: int, *, corrupt: float = 0.25) -> nn.Module:
    if kind == "v7":
        return AttentionGateController(chunk, feature_dim, proprio_dim, retrieval=v7_retrieval(RetrievalSpec()))
    if kind == "v8":
        return DisagreementGateController(chunk, feature_dim, proprio_dim, corrupt=corrupt)
    raise ValueError(kind)
