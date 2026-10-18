"""Feasibility projection for protocol v4 (planning only; authorizes no spending).

Generation dominates closed-loop cost. Case identity is (task, query demo), so
the first plan of a case is one clip shared by both arms and every budget and
seed; later plans are per episode because trajectories diverge. Counts are
upper bounds (every episode runs to ``max_steps``); successes end earlier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import math

from .wm_protocol import TemporalAlignment


@dataclass(frozen=True)
class UnitCosts:
    generation_seconds_per_clip: float = 30.0
    simulation_steps_per_second: float = 60.0
    inference_seconds_per_chunk: float = 0.03
    lora_seconds_per_update: float = 0.6
    source_training_seconds: float = 1800.0
    frame_cache_seconds: float = 1500.0
    model_load_seconds: float = 600.0
    hourly_usd: float = 0.59
    margin: float = 1.25
    measured: bool = False


SCOPES = {
    "development": {"tasks": 2, "queries": 20, "budgets": 3, "seeds": 1},
    "confirmation_all": {"tasks": 2, "queries": 20, "budgets": 3, "seeds": 3},
    "confirmation_five_shot": {"tasks": 2, "queries": 20, "budgets": 1, "seeds": 3},
}


def project(scope: str, costs: UnitCosts, temporal: TemporalAlignment | None = None, *,
            max_steps: int = 300, arms: int = 2, calibration_clips: int = 30) -> dict:
    temporal = temporal or TemporalAlignment()
    s = SCOPES[scope]
    cases = s["tasks"] * s["queries"]
    episodes = cases * s["budgets"] * s["seeds"] * arms
    plans_per_episode = math.ceil(max_steps / temporal.replan_steps)
    clips = cases + episodes * (plans_per_episode - 1)
    chunks = episodes * math.ceil(max_steps / temporal.chunk_steps)
    adaptations = s["tasks"] * s["budgets"] * s["seeds"]
    seconds = {
        "generation": clips * costs.generation_seconds_per_clip,
        "simulation": episodes * max_steps / costs.simulation_steps_per_second,
        "action_inference": chunks * costs.inference_seconds_per_chunk,
        "lora_adaptation": adaptations * 1000 * costs.lora_seconds_per_update,
        "raid_bank_construction": adaptations * 1.0,
        "setup": costs.model_load_seconds,
    }
    if scope == "development":
        seconds["temporal_calibration"] = calibration_clips * costs.generation_seconds_per_clip
        seconds["source_training"] = costs.source_training_seconds
        seconds["frame_cache"] = costs.frame_cache_seconds
    total = sum(seconds.values())
    return {"scope": scope, "cases": cases, "episodes": episodes, "plans_per_episode_max": plans_per_episode,
            "generated_clips_upper_bound": clips, "seconds": seconds, "gpu_hours": total / 3600,
            "projected_usd_with_margin": total / 3600 * costs.hourly_usd * costs.margin,
            "inputs_measured": costs.measured}


def feasibility(costs: UnitCosts | None = None, temporal: TemporalAlignment | None = None) -> dict:
    costs = costs or UnitCosts()
    scopes = {name: project(name, costs, temporal) for name in SCOPES}
    sensitivity = {f"{g:g}s_per_clip": {name: project(name, UnitCosts(**{**asdict(costs), "generation_seconds_per_clip": g}),
                                                     temporal)["projected_usd_with_margin"] for name in SCOPES}
                   for g in (10.0, 30.0, 60.0, 120.0)}
    return {"unit_costs": asdict(costs), "scopes": scopes, "generation_sensitivity_usd": sensitivity,
            "status": "planning_only_no_spending_authorized",
            "note": ("Unit costs are placeholders until measured on the pinned runtime and GPU."
                     if not costs.measured else "Unit costs supplied as measured.")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in asdict(UnitCosts()).items():
        if isinstance(value, bool):
            parser.add_argument(f"--{name.replace('_', '-')}", action="store_true")
        else:
            parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value)
    args = parser.parse_args(argv)
    costs = UnitCosts(**{k: getattr(args, k) for k in asdict(UnitCosts())})
    print(json.dumps(feasibility(costs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
