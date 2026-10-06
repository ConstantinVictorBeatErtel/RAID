"""Inexpensive lookup controls used beside learned Direct and RAID heads."""

from __future__ import annotations

import json
from pathlib import Path
import torch

from .config import ExperimentConfig
from .data import FeatureTransitions
from .models import DirectDecoder
from .retrieval import TransitionMemory
from .training import _memory, seed_everything
from .evaluation import action_metrics


def run_baselines(cfg: ExperimentConfig, *, output_dir: str | Path | None = None) -> dict:
    seed_everything(cfg.data.train_seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    train = FeatureTransitions(cfg.data.feature_dir, cfg.data.split_path, "train", train_per_task=cfg.data.train_per_task, train_seed=cfg.data.train_seed, horizon=cfg.horizon)
    validation = FeatureTransitions(cfg.data.feature_dir, cfg.data.split_path, "validation", train_per_task=cfg.data.train_per_task, train_seed=cfg.data.train_seed, horizon=cfg.horizon, norm_mean=train.norm_mean, norm_std=train.norm_std)
    memory = _memory(train, device)
    mean_action = torch.stack([x.action for x in train.items]).mean(0).to(device)
    methods = {"mean_action": [], "current_1nn": [], "transition_1nn": [], "transition_weighted_3nn": []}
    target = []
    task_ids = []
    demo_ids = []
    direct_predictions = []
    direct_path = Path(output_dir or cfg.output_dir) / "direct.pt"
    direct = DirectDecoder(cfg.model.feature_dim, cfg.model.action_dim, cfg.model.hidden_dim, cfg.model.bottleneck_dim, cfg.model.decoder_dropout).to(device)
    if direct_path.exists():
        direct.load_state_dict(torch.load(direct_path, map_location=device, weights_only=False)["state_dict"])
        direct.eval()
    with torch.no_grad():
        for row in validation.items:
            current, future = row.current.unsqueeze(0).to(device), row.future.unsqueeze(0).to(device)
            result_transition = memory.retrieve(current, future, k=3, query_demo_ids=torch.tensor([row.demo_id], device=device))
            result_current = memory.retrieve(current, future, k=1, query_demo_ids=torch.tensor([row.demo_id], device=device), current_only=True)
            methods["mean_action"].append(mean_action)
            methods["current_1nn"].append(result_current.actions[0, 0])
            methods["transition_1nn"].append(result_transition.actions[0, 0])
            weighted = memory.retrieve(current, future, k=3, query_demo_ids=torch.tensor([row.demo_id], device=device), temperature=cfg.model.retrieval_temperature)
            methods["transition_weighted_3nn"].append(weighted.actions[0].sum(0))
            if direct_path.exists():
                direct_predictions.append(direct(current, future)[0].squeeze(0))
            target.append(row.action.to(device))
            task_ids.append(row.task); demo_ids.append(row.demo_id)
    target = torch.stack(target)
    result = {"seed": cfg.data.train_seed, "methods": {}}
    for name, values in methods.items():
        pred = torch.stack(values)
        result["methods"][name] = action_metrics(pred.cpu().numpy(), target.cpu().numpy(), task_ids=task_ids, demo_ids=demo_ids)
    if direct_predictions:
        direct_pred = torch.stack(direct_predictions)
        # Similarity is the cosine score of the nearest transition key.
        blend = []
        for row, dp in zip(validation.items, direct_pred):
            q = torch.cat((row.current, row.future)).to(device)
            key = torch.nn.functional.normalize(q, dim=0)
            score = torch.nn.functional.cosine_similarity(key.unsqueeze(0), memory.keys, dim=1).max()
            weight = torch.exp(-10.0 * (1.0 - score)).clamp(0.0, 1.0)
            nn_action = memory.retrieve(row.current.to(device), row.future.to(device), k=1, query_demo_ids=torch.tensor([row.demo_id], device=device)).actions[0, 0]
            blend.append((1 - weight) * dp + weight * nn_action)
        result["methods"]["distance_blend"] = action_metrics(torch.stack(blend).cpu().numpy(), target.cpu().numpy(), task_ids=task_ids, demo_ids=demo_ids)
    out = Path(output_dir or cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
