"""Re-implemented few-shot inverse-dynamics baselines for the compute benchmark (user-approved 17 Oct 2026).

All baselines use the same five support demos, query demos, normalizers, update count, batch, sampler seed,
gradient clip and checkpoints as the registered LoRA adaptation, so every method gets a quality-vs-compute curve:

  head_only      frozen MAE, source-trained head fine-tuned              (protocol head-only diagnostic)
  probe_fresh    frozen MAE, freshly initialized head                    (frozen-encoder probe protocol)
  scratch_lora   LoRA MAE, freshly initialized head (no source training) (from-scratch inverse model)
  full_finetune  all MAE weights + source head (encoder lr 1e-5)         (full fine-tuning)
  strap_style    frozen MAE, source head; batches = half support, half nearest source-task transitions
                                                                         (retrieval for training data)
  retrieve_and_play  nearest support transition's action chunk          (REGENT retrieve-and-play)
  regent_blend   w * nearest action + (1 - w) * source head, w = exp(-L d) (REGENT-style interpolation)

These are faithful re-implementations of each method family in our setting, not reproductions of the papers.

Exploratory (development only, user-approved 17 Oct 2026; not part of TRAINED, so suite "full" is unchanged):

  raid_v7_head   frozen MAE (cached features), source-trained v7 head with retrieval (k=16) over the support bank,
                 selection loss; same schedule and sample stream as head_only
"""
from __future__ import annotations

from dataclasses import replace
import math
import time

import numpy as np
import torch

from .hybrid_heads import chunk_mse, make_bank
from .hybrid_protocol import ACTION_DIM, PRIMARY_BUDGET, HeadSpec
from . import hybrid_study as hs

FULL_ENCODER_LR = 1e-5
STRAP_NEIGHBOURS = 4
REGENT_L = 10.0
TRAINED = ("head_only", "probe_fresh", "scratch_lora", "full_finetune", "strap_style")
SPECS = {"head_only": ("frozen", "source", False), "probe_fresh": ("frozen", "fresh", False),
         "scratch_lora": ("lora", "fresh", False), "full_finetune": ("full", "source", False),
         "strap_style": ("frozen", "source", True)}
EXPLORATORY = {"raid_v7_head": ("frozen", "source", False)}  # retrieval head; needs the v7 source and its bank


def describe() -> dict:
    return {"trained": {k: {"encoder": v[0], "head_init": v[1], "retrieved_training_data": v[2]} for k, v in SPECS.items()},
            "full_encoder_lr": FULL_ENCODER_LR, "strap_neighbours_per_support_chunk": STRAP_NEIGHBOURS,
            "strap_pool": "source_train demos of every source task (frozen reference features)",
            "strap_batches": "half support rows, half retrieved rows", "regent_L": REGENT_L,
            "regent_weight": "exp(-L * retrieval distance of the nearest support transition)",
            "zero_training": ["retrieve_and_play", "regent_blend"]}


class RetrieveAndPlay(torch.nn.Module):
    head_kind = "retrieve_and_play"

    def __init__(self, base, retrieval, chunk_steps: int):
        super().__init__()
        self.base, self.retrieval, self.chunk_steps = base, replace(retrieval, k=1), chunk_steps

    def _nearest(self, bank, current, future, q, duration, query_current, query_future):
        r = bank.retrieve(current if query_current is None else query_current,
                          future if query_future is None else query_future, q, duration, spec=self.retrieval)
        return r.actions[:, 0], r.distances[:, 0], r.valid[:, 0]

    def forward_with_memory(self, current, future, q, duration, *, bank=None, use_retrieval=True, exclude_groups=None,
                            query_current=None, query_future=None):
        action, _, _ = self._nearest(bank, current, future, q, duration, query_current, query_future)
        return action.view(-1, self.chunk_steps, ACTION_DIM), {"null_memory": False}


class RegentBlend(RetrieveAndPlay):
    head_kind = "regent_blend"

    def forward_with_memory(self, current, future, q, duration, *, bank=None, use_retrieval=True, exclude_groups=None,
                            query_current=None, query_future=None):
        action, distance, valid = self._nearest(bank, current, future, q, duration, query_current, query_future)
        model, _ = self.base.forward_with_memory(current, future, q, duration, bank=None, use_retrieval=False)
        w = (torch.exp(-REGENT_L * distance.clamp_min(0)) * valid.float()).unsqueeze(-1)
        out = w * action + (1 - w) * model.reshape(len(model), -1)
        return out.view(-1, self.chunk_steps, ACTION_DIM), {"null_memory": False, "weight_mean": float(w.mean())}


def strap_rows(protocol: dict, cache: dict, source: dict, support, device: str) -> hs.ChunkRows:
    """Nearest source-task transitions to each support chunk (frozen reference features); unique rows."""
    retrieval = replace(hs._as_retrieval(source["retrieval_spec"]), k=STRAP_NEIGHBOURS)
    pools = [hs.chunk_rows(cache["shards"][t], list(map(int, protocol["tasks"][t]["source"]["source_train"])), t,
                           source["chunk_steps"], source["normalizers"]) for t in hs._source_tasks(protocol)]
    cat = lambda name: torch.cat([getattr(p, name) for p in pools])
    pool = hs.ChunkRows(cat("current"), cat("future"), cat("q"), cat("target"), cat("duration"), cat("group_id"),
                        sum((p.demo for p in pools), []), sum((p.step for p in pools), []), "source_pool")
    bank = make_bank(pool.current, pool.future, pool.q, pool.target, pool.duration, pool.group_id, device)
    found = bank.retrieve(support.current.to(device), support.future.to(device), support.q.to(device),
                          support.duration.to(device), spec=retrieval)
    idx = torch.unique(found.indices[found.valid]).cpu()
    return hs.ChunkRows(pool.current[idx], pool.future[idx], pool.q[idx], pool.target[idx], pool.duration[idx],
                        pool.group_id[idx], [pool.demo[i] for i in idx], [pool.step[i] for i in idx], "strap_retrieved")


def train_variant(name: str, protocol: dict, source: dict, encoder_factory, task: str, seed: int, *, cache: dict,
                  device: str, steps: tuple[int, ...], callback, bank=None) -> dict:
    """Same schedule as hybrid_study.adapt_target; callback(step, controller, encoder|None, seconds) at checkpoints.

    ``bank`` (raid_v7_head only): the support bank; each row excludes its own demo from retrieval, as in
    raid_lora_mlp adaptation."""
    from .models import attach_mae_lora
    use_bank = name in EXPLORATORY
    encoder_mode, head_init, use_strap = EXPLORATORY[name] if use_bank else SPECS[name]
    if use_bank and (bank is None or source.get("head_kind") != "attention_gate_v7"):
        raise ValueError("raid_v7_head needs the v7 source controller and its support bank")
    design, cfg = protocol["design"], protocol["design"]["training"]
    lora = design["inverse_model"]["lora"]
    updates, batch, micro = int(cfg["target_updates"]), int(cfg["batch_size"]), int(cfg["microbatch_size"])
    dev = torch.device(device)
    torch.manual_seed(seed)
    chunk, norms = source["chunk_steps"], source["normalizers"]
    if head_init == "source":
        controller = hs.build_controller(source, device)
    else:
        controller = hs._new_controller("mlp_v6", chunk, source["feature_dim"], source["proprio_dim"],
                                        HeadSpec(**source["head_spec"]), hs._as_retrieval(source["retrieval_spec"])).to(dev)
    controller.train()
    groups = [{"params": list(controller.parameters()), "lr": cfg["head_lr"]}]
    encoder = None
    if encoder_mode != "frozen":
        encoder = encoder_factory()
        if encoder_mode == "lora":
            attach_mae_lora(encoder.mae, lora["blocks"], lora["rank"], lora["alpha"], lora["dropout"])
            enc_params = [p for p in encoder.mae.parameters() if p.requires_grad]
            groups.append({"params": enc_params, "lr": cfg["lora_lr"]})
        else:
            for p in encoder.mae.parameters():
                p.requires_grad_(True)
            enc_params = list(encoder.mae.parameters())
            groups.append({"params": enc_params, "lr": FULL_ENCODER_LR})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    ids = hs.support_ids(protocol, task, PRIMARY_BUDGET, seed)
    gen = torch.Generator().manual_seed(int(hs._sha_bytes(f"v6adapt:{task}:{PRIMARY_BUDGET}:{seed}".encode())[:8], 16))
    support = hs.chunk_rows(cache["shards"][task], ids, task, chunk, norms)
    retrieved = strap_rows(protocol, cache, source, support, device) if use_strap else None
    if encoder_mode != "frozen":
        path = protocol["tasks"][task]["path"]
        live = hs._SupportLive({d: hs.read_view_frames(protocol, path, d) for d in ids},
                               {d: hs.read_actions(path, d) for d in ids},
                               {d: hs.read_proprio(path, d, protocol) for d in ids}, chunk, norms)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    if 0 in steps:  # untrained starting point (fresh heads have no zero-adaptation row elsewhere)
        callback(0, controller, encoder, 0.0)
    elapsed, started = 0.0, time.perf_counter()
    for step in range(1, updates + 1):
        opt.zero_grad(set_to_none=True)
        if use_bank:
            idx = torch.randint(0, len(support), (batch,), generator=gen)
            tgt = support.target[idx].to(dev)
            pred, info = controller.forward_with_memory(support.current[idx].to(dev), support.future[idx].to(dev),
                                                        support.q[idx].to(dev), support.duration[idx].to(dev),
                                                        bank=bank, use_retrieval=True,
                                                        exclude_groups=support.group_id[idx].to(dev))
            loss = chunk_mse(pred, tgt).mean() + hs._selection_term(controller, info, tgt.reshape(batch, -1))
            loss.backward()
        elif encoder_mode == "frozen":
            if retrieved is None:
                rows_idx = [(support, torch.randint(0, len(support), (batch,), generator=gen))]
            else:
                half = batch // 2
                rows_idx = [(support, torch.randint(0, len(support), (batch - half,), generator=gen)),
                            (retrieved, torch.randint(0, len(retrieved), (half,), generator=gen))]
            loss = 0.0
            for rows, idx in rows_idx:
                pred, _ = controller.forward_with_memory(rows.current[idx].to(dev), rows.future[idx].to(dev),
                                                         rows.q[idx].to(dev), rows.duration[idx].to(dev),
                                                         bank=None, use_retrieval=False)
                loss = loss + chunk_mse(pred, rows.target[idx].to(dev)).sum() / batch
            loss.backward()
        else:
            indices = torch.randint(0, len(live), (batch,), generator=gen)
            for part in indices.split(micro):
                cur_img, fut_img, q, tgt, _, _, _ = live.get(part)
                feats = encoder.encode_frames(torch.cat((cur_img, fut_img)))
                n = len(part)
                pred, _ = controller.forward_with_memory(feats[:n], feats[n:], q.to(dev),
                                                         torch.full((n,), float(chunk), device=dev),
                                                         bank=None, use_retrieval=False)
                (chunk_mse(pred, tgt.to(dev)).mean() * (n / batch)).backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        if step in steps:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            paused = time.perf_counter()
            callback(step, controller, encoder, elapsed + paused - started)
            started += time.perf_counter() - paused
    seconds = elapsed + time.perf_counter() - started
    return {"adaptation_seconds": seconds, "trainable_parameters": int(sum(p.numel() for p in params)),
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None,
            "retrieved_rows": len(retrieved) if retrieved is not None else 0, "encoder_mode": encoder_mode,
            "head_init": head_init, "retrieval_bank_entries": len(bank) if use_bank else 0}


def frozen_update_flops(protocol: dict, source: dict, controller, cache: dict, task: str, seed: int, device: str,
                        bank=None) -> int:
    """FLOPs of one frozen-encoder head update (batch 64, forward + backward); with ``bank``, retrieval included."""
    from torch.utils.flop_counter import FlopCounterMode
    batch = int(protocol["design"]["training"]["batch_size"])
    rows = hs.chunk_rows(cache["shards"][task], hs.support_ids(protocol, task, PRIMARY_BUDGET, seed), task,
                         source["chunk_steps"], source["normalizers"])
    idx = torch.arange(min(batch, len(rows)))
    counter = FlopCounterMode(display=False)
    with counter:
        tgt = rows.target[idx].to(device)
        pred, info = controller.forward_with_memory(rows.current[idx].to(device), rows.future[idx].to(device),
                                                    rows.q[idx].to(device), rows.duration[idx].to(device),
                                                    bank=bank, use_retrieval=bank is not None,
                                                    exclude_groups=rows.group_id[idx].to(device) if bank is not None else None)
        loss = chunk_mse(pred, tgt).mean()
        if bank is not None:
            loss = loss + hs._selection_term(controller, info, tgt.reshape(len(idx), -1))
        loss.backward()
    for p in controller.parameters():
        p.grad = None
    return int(counter.get_total_flops()) * math.ceil(batch / len(idx))


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean over the 7 action dimensions of 1 - SSE/SST (all query chunk elements)."""
    p = pred.reshape(-1, ACTION_DIM).double()
    t = target.reshape(-1, ACTION_DIM).double()
    sse = ((p - t) ** 2).sum(0)
    sst = ((t - t.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    return float((1 - sse / sst).mean())
