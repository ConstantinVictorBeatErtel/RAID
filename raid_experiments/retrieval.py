"""Vectorized retrieval with explicit provenance and leave-one-demo-out support."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class RetrievalResult:
    actions: torch.Tensor
    indices: torch.Tensor
    similarities: torch.Tensor
    valid: torch.Tensor


class TransitionMemory:
    def __init__(self, feature_dim: int, action_dim: int = 7, device: str | torch.device = "cpu"):
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.current = torch.empty((0, feature_dim), dtype=torch.float32, device=self.device)
        self.future = torch.empty((0, feature_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((0, action_dim), dtype=torch.float32, device=self.device)
        self.demo_ids = torch.empty((0,), dtype=torch.long, device=self.device)
        self.keys = torch.empty((0, feature_dim * 2), dtype=torch.float32, device=self.device)

    def add(self, current: torch.Tensor, future: torch.Tensor, action: torch.Tensor, demo_id: torch.Tensor | int) -> None:
        current = current.detach().reshape(-1, self.feature_dim).to(self.device, dtype=torch.float32)
        future = future.detach().reshape(-1, self.feature_dim).to(self.device, dtype=torch.float32)
        action = action.detach().reshape(-1, self.action_dim).to(self.device, dtype=torch.float32)
        ids = torch.as_tensor(demo_id, device=self.device, dtype=torch.long).reshape(-1)
        if not (len(current) == len(future) == len(action) == len(ids)):
            raise ValueError("memory fields have different lengths")
        self.extend(current, future, action, ids)

    def extend(self, current: torch.Tensor, future: torch.Tensor, action: torch.Tensor,
               demo_id: torch.Tensor | int) -> None:
        """Append a batch and build the normalized keys once per batch."""
        current = torch.as_tensor(current, device=self.device, dtype=torch.float32).reshape(-1, self.feature_dim)
        future = torch.as_tensor(future, device=self.device, dtype=torch.float32).reshape(-1, self.feature_dim)
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(-1, self.action_dim)
        ids = torch.as_tensor(demo_id, device=self.device, dtype=torch.long).reshape(-1)
        if not (len(current) == len(future) == len(action) == len(ids)):
            raise ValueError("memory fields have different lengths")
        self.current = torch.cat((self.current, current))
        self.future = torch.cat((self.future, future))
        self.actions = torch.cat((self.actions, action))
        self.demo_ids = torch.cat((self.demo_ids, ids))
        self.keys = F.normalize(torch.cat((self.current, self.future), dim=-1), dim=-1, eps=1e-8)

    def retrieve(
        self,
        current: torch.Tensor,
        future: torch.Tensor,
        *,
        k: int = 3,
        query_demo_ids: torch.Tensor | None = None,
        current_only: bool = False,
        temperature: float | None = None,
        query_chunk: int = 256,
        bank_chunk: int = 8192,
    ) -> RetrievalResult:
        current = torch.as_tensor(current)
        future = torch.as_tensor(future)
        if current.ndim == 1:
            current = current.unsqueeze(0)
            future = future.unsqueeze(0)
        if query_chunk < 1 or bank_chunk < 1:
            raise ValueError("retrieval chunk sizes must be positive")
        if not len(self.actions):
            shape = (len(current), max(1, k), self.action_dim)
            z = torch.zeros(shape, device=self.device)
            return RetrievalResult(z, torch.full(shape[:2], -1, dtype=torch.long, device=self.device), torch.full(shape[:2], float("-inf"), device=self.device), torch.zeros(shape[:2], dtype=torch.bool, device=self.device))
        bank_key = F.normalize(self.current, dim=-1, eps=1e-8) if current_only else self.keys
        kk = min(max(1, int(k)), len(self.actions))
        sim_chunks, index_chunks = [], []
        for qstart in range(0, len(current), query_chunk):
            qend = min(len(current), qstart + query_chunk)
            qcur = current[qstart:qend].to(self.device, dtype=torch.float32)
            qfut = future[qstart:qend].to(self.device, dtype=torch.float32)
            query = qcur if current_only else torch.cat((qcur, qfut), dim=-1)
            key = F.normalize(query, dim=-1, eps=1e-8)
            best_scores = torch.full((len(key), 0), float("-inf"), device=self.device)
            best_indices = torch.empty((len(key), 0), dtype=torch.long, device=self.device)
            qids = query_demo_ids[qstart:qend].to(self.device).reshape(-1) if query_demo_ids is not None else None
            for bstart in range(0, len(self.actions), bank_chunk):
                bend = min(len(self.actions), bstart + bank_chunk)
                scores = key @ bank_key[bstart:bend].T
                if qids is not None:
                    scores = scores.masked_fill(self.demo_ids[bstart:bend].unsqueeze(0).eq(qids.unsqueeze(1)), float("-inf"))
                indices = torch.arange(bstart, bend, device=self.device).expand(len(key), -1)
                merged_scores = torch.cat((best_scores, scores), dim=1)
                merged_indices = torch.cat((best_indices, indices), dim=1)
                keep = min(kk, merged_scores.shape[1])
                best_scores, order = torch.topk(merged_scores, keep, dim=-1)
                best_indices = merged_indices.gather(1, order)
            sim_chunks.append(best_scores); index_chunks.append(best_indices)
        similarities, indices = torch.cat(sim_chunks), torch.cat(index_chunks)
        valid = torch.isfinite(similarities)
        actions = self.actions[indices.clamp_min(0)]
        if temperature is not None:
            logits = similarities / max(temperature, 1e-6)
            # Avoid softmax(-inf) rows becoming NaN when a whole query
            # demonstration is excluded from a tiny bank.
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=-1).masked_fill(~valid, 0.0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            actions = actions * weights.unsqueeze(-1)
        return RetrievalResult(actions, indices, similarities, valid)

    def clear(self) -> None:
        self.__init__(self.feature_dim, self.action_dim, self.device)
