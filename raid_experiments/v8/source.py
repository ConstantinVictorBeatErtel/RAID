"""Source training of the RAID v7 / v8 heads on cached LIBERO-90 features.

Each update draws G tasks; for each task an episodic bank of m in {2, 5, 10} training demos and Q query windows
from the other training demos of that task. One batched forward uses a bank that masks other tasks and the
query's own demo. Loss = 0.5 * retrieval-present MSE + 0.5 * null-memory MSE + 0.1 * selection loss.
Validation every ``val_every`` updates on the 5 held-back demos per task (bank: 5 training demos); the best
checkpoint is kept.
"""
from __future__ import annotations

import copy
import time

import torch

from ..hybrid_protocol import RetrievalSpec
from ..hybrid_retrieval import ReferenceBank, RetrievalResult
from ..hybrid_v7 import SELECTION_WEIGHT, selection_loss
from . import data as d
from .heads import build

BANK_SIZES = (2, 5, 10)
GROUPS, PER_GROUP = 8, 16
LR, WD, CLIP = 3e-4, 1e-4, 1.0


class EpisodicBank(ReferenceBank):
    """Bank whose entries are only visible to queries of the same task; the query's own demo is excluded.

    ``exclude_groups`` carries the query's global demo id (task_index * 1000 + demo)."""

    def retrieve(self, current, future, q, duration, *, spec: RetrievalSpec, exclude_groups=None) -> RetrievalResult:
        n, k = len(current), int(spec.k)
        dist = self.distance(current, future, q, spec=spec)
        own = torch.as_tensor(exclude_groups, device=self.device).reshape(-1, 1)
        mask = (self.group_ids.unsqueeze(0) // 1000 != own // 1000) | (self.group_ids.unsqueeze(0) == own)
        dist = dist.masked_fill(mask, float("inf"))
        kk = min(k, len(self))
        sims, idx = torch.topk(-dist, kk, dim=-1)
        valid = torch.isfinite(sims)
        if kk < k:
            pad = k - kk
            sims = torch.cat((sims, sims.new_full((n, pad), float("-inf"))), 1)
            idx = torch.cat((idx, idx.new_full((n, pad), -1)), 1)
            valid = torch.cat((valid, valid.new_zeros(n, pad)), 1)
        actions = self.actions[idx.clamp_min(0)] * valid.unsqueeze(-1).float()
        groups = torch.where(valid, self.group_ids[idx.clamp_min(0)], idx.new_full(idx.shape, -1))
        return RetrievalResult(actions, idx, sims, valid, torch.where(valid, -sims, torch.full_like(sims, float("inf"))),
                               groups)


class SourceData:
    """All source tasks' training/validation windows on the device, indexed by (task, demo)."""

    def __init__(self, caches: list[dict], splits: list[dict], norms: dict, device: str, chunk: int = d.CHUNK):
        self.device, self.chunk, self.tasks = device, chunk, []
        for ti, (tc, split) in enumerate(zip(caches, splits)):
            feats = tc["features"].to(device)  # fp16
            acts = ((tc["actions"] - norms["action_mean"]) / norms["action_std"]).to(device)
            prop = ((tc["proprio"] - norms["proprio_mean"]) / norms["proprio_std"]).to(device)
            demos = {}
            for role in ("train", "val"):
                for demo in split[role]:
                    start, n = tc["demos"][int(demo)]
                    if n > chunk:
                        demos[int(demo)] = (start, n, role)
            self.tasks.append({"feats": feats, "acts": acts, "prop": prop, "demos": demos,
                               "train": [x for x, v in demos.items() if v[2] == "train"],
                               "val": [x for x, v in demos.items() if v[2] == "val"], "index": ti})

    def windows(self, ti: int, demo: int, steps: torch.Tensor | None = None):
        t = self.tasks[ti]
        start, n, _ = t["demos"][demo]
        steps = torch.arange(n - self.chunk, device=self.device) if steps is None else steps.to(self.device)
        pos = start + steps
        offs = torch.arange(self.chunk, device=self.device)
        return {"cur": t["feats"][pos].float(), "fut": t["feats"][pos + self.chunk].float(), "q": t["prop"][pos],
                "tgt": t["acts"][pos.unsqueeze(-1) + offs], "group": torch.full((len(pos),), ti * 1000 + demo,
                                                                               device=self.device)}


def _cat(parts: list[dict]) -> dict:
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def _bank(src: SourceData, items: list[tuple[int, list[int]]]) -> EpisodicBank:
    parts = [src.windows(ti, demo) for ti, demos in items for demo in demos]
    w = _cat(parts)
    bank = EpisodicBank(w["cur"].shape[-1], w["q"].shape[-1], w["tgt"][0].numel(), src.device)
    bank.extend(w["cur"], w["fut"], w["q"], w["tgt"].reshape(len(w["tgt"]), -1),
                torch.full((len(w["cur"]),), float(src.chunk), device=src.device), w["group"])
    return bank


def _batch(src: SourceData, gen: torch.Generator):
    tasks = torch.randperm(len(src.tasks), generator=gen)[:GROUPS].tolist()
    bank_items, queries = [], []
    for ti in tasks:
        train = src.tasks[ti]["train"]
        m = BANK_SIZES[int(torch.randint(len(BANK_SIZES), (1,), generator=gen))]
        order = torch.randperm(len(train), generator=gen).tolist()
        bank_demos = [train[i] for i in order[:m]]
        bank_items.append((ti, bank_demos))
        for _ in range(PER_GROUP):
            demo = train[order[m + int(torch.randint(len(train) - m, (1,), generator=gen))]] if len(train) > m else train[0]
            start, n, _ = src.tasks[ti]["demos"][demo]
            step = torch.randint(n - src.chunk, (1,), generator=gen)
            queries.append(src.windows(ti, demo, step))
    return _bank(src, bank_items), _cat(queries)


def _loss(model, batch: dict, bank, *, train: bool = True):
    dur = torch.full((len(batch["cur"]),), 8.0, device=batch["cur"].device)
    out_r, info = model.forward_with_memory(batch["cur"], batch["fut"], batch["q"], dur, bank=bank, use_retrieval=True,
                                            exclude_groups=batch["group"])
    out_n, _ = model.forward_with_memory(batch["cur"], batch["fut"], batch["q"], dur, use_retrieval=False)
    mse_r = ((out_r - batch["tgt"]) ** 2).mean()
    mse_n = ((out_n - batch["tgt"]) ** 2).mean()
    sel = selection_loss(info, batch["tgt"].reshape(len(batch["tgt"]), -1)) if train else mse_r.new_zeros(())
    return 0.5 * mse_r + 0.5 * mse_n + SELECTION_WEIGHT * sel, mse_r.item(), mse_n.item()


@torch.no_grad()
def validate(model, src: SourceData, seed: int) -> dict:
    """Retrieval-present and null-memory chunk MSE on held-back demos (bank: 5 training demos per task)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed + 7)
    r, nm, count = 0.0, 0.0, 0
    for ti, task in enumerate(src.tasks):
        bank_demos = [task["train"][i] for i in torch.randperm(len(task["train"]), generator=gen)[:5].tolist()]
        bank = _bank(src, [(ti, bank_demos)])
        for demo in task["val"]:
            w = src.windows(ti, demo)
            dur = torch.full((len(w["cur"]),), 8.0, device=src.device)
            out_r, _ = model.forward_with_memory(w["cur"], w["fut"], w["q"], dur, bank=bank, use_retrieval=True,
                                                 exclude_groups=w["group"])
            out_n, _ = model.forward_with_memory(w["cur"], w["fut"], w["q"], dur, use_retrieval=False)
            r += float(((out_r - w["tgt"]) ** 2).mean()); nm += float(((out_n - w["tgt"]) ** 2).mean()); count += 1
    model.train()
    return {"retrieval_mse": r / count, "null_mse": nm / count}


def train(kind: str, src: SourceData, *, seed: int, updates: int, corrupt: float = 0.25, val_every: int = 250,
          log=print) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    feat_dim = src.tasks[0]["feats"].shape[-1]
    model = build(kind, src.chunk, feat_dim, src.tasks[0]["prop"].shape[-1], corrupt=corrupt).to(src.device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, curve = float("inf"), None, []
    started = time.time()
    for step in range(1, updates + 1):
        bank, batch = _batch(src, gen)
        loss, mse_r, mse_n = _loss(model, batch, bank)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        if step % val_every == 0 or step == updates:
            val = validate(model, src, seed)
            score = 0.5 * val["retrieval_mse"] + 0.5 * val["null_mse"]
            curve.append({"step": step, **val, "train_retrieval_mse": mse_r, "train_null_mse": mse_n,
                          "seconds": round(time.time() - started, 1)})
            log(f"source {kind} seed {seed} step {step} val_r {val['retrieval_mse']:.4f} val_null {val['null_mse']:.4f}")
            if score < best:
                best, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, {"kind": kind, "seed": seed, "updates": updates, "corrupt": corrupt, "curve": curve,
                   "best_score": best, "seconds": round(time.time() - started, 1)}
