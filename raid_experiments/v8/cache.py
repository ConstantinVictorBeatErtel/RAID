"""Per-task feature caches and chunk rows for RAID v8.

Cache file ``<cache>/<encoder>/<suite>/<task>.pt``: features (N, 768) fp16 of every frame of every demo, the
actions (N, 7) and proprio (N, 9), and ``demos`` = {demo: (start, length)}. Rows are stride-1 windows
(current t, future t + 8, proprio t, action chunk t..t+7), normalized with source-training statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch

from . import data as d
from . import encoders as enc


def cache_path(cache: str | Path, encoder: str, task: d.Task) -> Path:
    return Path(cache) / encoder / task.suite / f"{task.name}.pt"


def build_task_cache(task: d.Task, encoders: dict, cache: str | Path) -> dict:
    """Encode every frame of every demo once per encoder (one HDF5 read per demo)."""
    todo = {n: e for n, e in encoders.items() if not cache_path(cache, n, task).is_file()}
    if not todo:
        return {"task": task.key, "skipped": True}
    started = time.time()
    feats = {n: [] for n in todo}
    actions, proprio, demos, pos = [], [], {}, 0
    for demo in d.demo_ids(task.path):
        rec = d.read_demo(task.path, demo)
        n = len(rec["actions"])
        demos[int(demo)] = (pos, n)
        pos += n
        actions.append(torch.from_numpy(rec["actions"]))
        proprio.append(torch.from_numpy(rec["proprio"]))
        for name, encoder in todo.items():
            feats[name].append(enc.encode(encoder, rec["frames"]).half())
    for name in todo:
        out = cache_path(cache, name, task)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        torch.save({"task": task.key, "encoder": name, "features": torch.cat(feats[name]),
                    "actions": torch.cat(actions), "proprio": torch.cat(proprio), "demos": demos}, tmp)
        tmp.replace(out)
    return {"task": task.key, "frames": pos, "demos": len(demos), "encoders": sorted(todo),
            "seconds": round(time.time() - started, 2)}


def load_task(cache: str | Path, encoder: str, task: d.Task) -> dict:
    return torch.load(cache_path(cache, encoder, task), map_location="cpu", weights_only=False)


@dataclass
class Rows:
    current: torch.Tensor
    future: torch.Tensor
    q: torch.Tensor
    target: torch.Tensor
    duration: torch.Tensor
    group: torch.Tensor      # global demo id (task index * 1000 + demo)
    demo: list
    step: list

    def __len__(self) -> int:
        return len(self.current)

    def to(self, device) -> "Rows":
        return Rows(*(x.to(device) if torch.is_tensor(x) else x for x in
                      (self.current, self.future, self.q, self.target, self.duration, self.group, self.demo, self.step)))

    def select(self, idx: torch.Tensor) -> "Rows":
        il = idx.tolist()
        return Rows(self.current[idx], self.future[idx], self.q[idx], self.target[idx], self.duration[idx],
                    self.group[idx], [self.demo[i] for i in il], [self.step[i] for i in il])


def rows(task_cache: dict, demos: list[int], norms: dict, *, task_index: int, chunk: int = d.CHUNK) -> Rows:
    feats = task_cache["features"].float()
    acts = (task_cache["actions"] - norms["action_mean"]) / norms["action_std"]
    prop = (task_cache["proprio"] - norms["proprio_mean"]) / norms["proprio_std"]
    cur, fut, q, tgt, group, dd, ss = [], [], [], [], [], [], []
    for demo in demos:
        start, n = task_cache["demos"][int(demo)]
        if n <= chunk:
            continue
        t = torch.arange(n - chunk)
        cur.append(feats[start + t]); fut.append(feats[start + t + chunk]); q.append(prop[start + t])
        tgt.append(torch.stack([acts[start + i: start + i + chunk] for i in range(n - chunk)]))
        group.append(torch.full((n - chunk,), task_index * 1000 + int(demo), dtype=torch.long))
        dd += [int(demo)] * (n - chunk); ss += t.tolist()
    if not cur:
        raise ValueError(f"no complete chunks for {task_cache['task']} {demos}")
    cur = torch.cat(cur)
    return Rows(cur, torch.cat(fut), torch.cat(q), torch.cat(tgt), torch.full((len(cur),), float(chunk)),
                torch.cat(group), dd, ss)


def normalizers(caches: list[tuple[dict, list[int]]]) -> dict:
    """Action/proprio mean and std over the source-training demos only."""
    a, q = [], []
    for task_cache, demos in caches:
        for demo in demos:
            start, n = task_cache["demos"][int(demo)]
            a.append(task_cache["actions"][start:start + n]); q.append(task_cache["proprio"][start:start + n])
    a, q = torch.cat(a).float(), torch.cat(q).float()
    fix = lambda s: torch.where(s < 1e-6, torch.ones_like(s), s)
    return {"action_mean": a.mean(0), "action_std": fix(a.std(0, unbiased=False)),
            "proprio_mean": q.mean(0), "proprio_std": fix(q.std(0, unbiased=False))}


def write_json(path: str | Path, body) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
