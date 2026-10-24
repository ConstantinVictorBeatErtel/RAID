"""Target-task methods for RAID v8: training-free baselines, frozen-feature head updates, LoRA and full fine-tuning.

Every method sees the same 5 support demos, normalizers and query windows. Quality = chunk MSE on normalized
actions, averaged per query demo, then over the 20 query demos. Adaptation GPU-seconds exclude evaluation.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from ..hybrid_heads import chunk_mse, knn_average, make_bank
from ..hybrid_protocol import ACTION_DIM, RetrievalSpec
from ..hybrid_v7 import SELECTION_WEIGHT, selection_loss
from . import data as d
from .cache import Rows

BATCH = 64
HEAD_CHECKPOINTS = (0, 10, 25, 50, 100, 200)
ENCODER_CHECKPOINTS = (0, 25, 50, 100, 200)
KNN_SPEC = RetrievalSpec(k=4, temperature=0.1)
REGENT_L = 10.0
WD, CLIP = 1e-4, 1.0
LORA = {"blocks": (10, 11), "rank": 8, "alpha": 16, "dropout": 0.0}


def sync(device) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def support_bank(rows: Rows, device: str):
    return make_bank(rows.current, rows.future, rows.q, rows.target.reshape(len(rows), -1), rows.duration,
                     rows.group, device=device)


def r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    p, t = pred.reshape(-1, ACTION_DIM).double(), target.reshape(-1, ACTION_DIM).double()
    return float((1 - ((p - t) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0).clamp_min(1e-12)).mean())


def score(pred: torch.Tensor, query: Rows) -> dict:
    errors = chunk_mse(pred.cpu(), query.target.cpu())
    per = {}
    for e, demo in zip(errors.tolist(), query.demo):
        per.setdefault(demo, []).append(e)
    by_demo = {str(k): float(np.mean(v)) for k, v in per.items()}
    return {"mse": float(np.mean(list(by_demo.values()))), "r2": r2(pred, query.target), "mse_by_demo": by_demo}


@torch.no_grad()
def predict(model, query: Rows, *, bank=None, use_retrieval: bool, current=None, future=None) -> torch.Tensor:
    model.eval()
    out, _ = model.forward_with_memory(query.current if current is None else current,
                                       query.future if future is None else future, query.q, query.duration,
                                       bank=bank if use_retrieval else None, use_retrieval=use_retrieval,
                                       query_current=query.current if (use_retrieval and current is not None) else None,
                                       query_future=query.future if (use_retrieval and future is not None) else None)
    return out


@torch.no_grad()
def knn(bank, query: Rows, chunk: int = d.CHUNK) -> torch.Tensor:
    return knn_average(bank, query.current, query.future, query.q, query.duration, spec=KNN_SPEC, chunk_steps=chunk)


@torch.no_grad()
def regent_blend(base, bank, query: Rows, chunk: int = d.CHUNK) -> torch.Tensor:
    from dataclasses import replace
    r = bank.retrieve(query.current, query.future, query.q, query.duration, spec=replace(KNN_SPEC, k=1))
    model = predict(base, query, use_retrieval=False).reshape(len(query), -1)
    w = (torch.exp(-REGENT_L * r.distances[:, 0].clamp_min(0)) * r.valid[:, 0].float()).unsqueeze(-1)
    return (w * r.actions[:, 0] + (1 - w) * model).view(-1, chunk, ACTION_DIM)


def adapt_head(source_model, support: Rows, query: Rows, *, use_retrieval: bool, lr: float, seed: int,
               device: str, checkpoints=HEAD_CHECKPOINTS) -> dict:
    """Frozen features: update the whole head (trunk, attention, gate). Retrieval on: bank = support set, each
    support window excludes its own demo; evaluation uses all five support demos."""
    model = copy.deepcopy(source_model).to(device).train()
    bank = support_bank(support, device) if use_retrieval else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    gen = torch.Generator().manual_seed(seed)
    curve, gpu_s, step = [], 0.0, 0
    for target in sorted(checkpoints):
        sync(device); t0 = time.perf_counter()
        while step < target:
            idx = torch.randint(len(support), (BATCH,), generator=gen)
            b = support.select(idx)
            out, info = model.forward_with_memory(b.current, b.future, b.q, b.duration, bank=bank,
                                                  use_retrieval=use_retrieval, exclude_groups=b.group if use_retrieval else None)
            loss = ((out - b.target) ** 2).mean()
            if use_retrieval:
                loss = loss + SELECTION_WEIGHT * selection_loss(info, b.target.reshape(len(b), -1))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
            step += 1
        sync(device); gpu_s += time.perf_counter() - t0
        pred = predict(model, query, bank=bank, use_retrieval=use_retrieval)
        curve.append({"updates": target, "gpu_seconds": gpu_s, **score(pred, query)})
        model.train()
    return {"curve": curve, "model": model, "bank": bank}


def _encode_rows(encoder, frames: dict, rows: Rows, chunk: int, *, grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Live controller features for the windows of ``rows`` (frames: demo -> uint8 array)."""
    imgs_c = np.stack([frames[dm][t] for dm, t in zip(rows.demo, rows.step)])
    imgs_f = np.stack([frames[dm][t + chunk] for dm, t in zip(rows.demo, rows.step)])
    batch = torch.from_numpy(np.concatenate([imgs_c, imgs_f]))
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=encoder.device.type == "cuda"):
        f = encoder.encode_frames(batch).float()
    return f[: len(rows)], f[len(rows):]


@torch.no_grad()
def _encode_query(encoder, frames: dict, query: Rows, chunk: int, batch: int = 512):
    curs, futs = [], []
    for s in range(0, len(query), batch):
        idx = torch.arange(s, min(len(query), s + batch))
        c, f = _encode_rows(encoder, frames, query.select(idx), chunk, grad=False)
        curs.append(c); futs.append(f)
    return torch.cat(curs), torch.cat(futs)


def adapt_encoder(source_model, encoder_factory, support: Rows, support_frames: dict, query: Rows,
                  query_frames: dict, *, mode: str, lr: float, head_lr: float, seed: int, device: str,
                  checkpoints=ENCODER_CHECKPOINTS, chunk: int = d.CHUNK) -> dict:
    """MAE LoRA (blocks 10-11, rank 8) or full fine-tuning, plus head updates; retrieval off."""
    from ..models import attach_mae_lora
    encoder = encoder_factory()
    if mode == "lora":
        attach_mae_lora(encoder.model, LORA["blocks"], LORA["rank"], LORA["alpha"], LORA["dropout"])
    elif mode == "full":
        for p in encoder.model.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(mode)
    encoder.model.to(device).train()
    model = copy.deepcopy(source_model).to(device).train()
    enc_params = [p for p in encoder.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": head_lr}, {"params": enc_params, "lr": lr}],
                            weight_decay=WD)
    gen = torch.Generator().manual_seed(seed)
    curve, gpu_s, step = [], 0.0, 0
    for target in sorted(checkpoints):
        sync(device); t0 = time.perf_counter()
        while step < target:
            b = support.select(torch.randint(len(support), (BATCH,), generator=gen))
            cur, fut = _encode_rows(encoder, support_frames, b, chunk, grad=True)
            out, _ = model.forward_with_memory(cur, fut, b.q, b.duration, use_retrieval=False)
            loss = ((out - b.target) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + enc_params, CLIP); opt.step()
            step += 1
        sync(device); gpu_s += time.perf_counter() - t0
        encoder.model.eval()
        cur, fut = _encode_query(encoder, query_frames, query, chunk)
        pred = predict(model, query, use_retrieval=False, current=cur, future=fut)
        curve.append({"updates": target, "gpu_seconds": gpu_s, **score(pred, query)})
        encoder.model.train(); model.train()
    trainable = sum(p.numel() for p in enc_params) + sum(p.numel() for p in model.parameters())
    return {"curve": curve, "trainable_params": trainable, "encoder": encoder, "model": model}


def flops_per_update(fn) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return int(counter.get_total_flops())
