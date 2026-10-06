"""Generate a compact result table from condition JSON files."""

from __future__ import annotations

import json
from pathlib import Path


def build_report(root: str | Path) -> dict:
    root = Path(root)
    rows = []
    expected = ["direct", "raid", "lora", "lora_raid"]
    run_meta = root / "run.json"
    if run_meta.exists():
        expected = json.loads(run_meta.read_text()).get("method_registry", expected)
    seen = set()
    for path in sorted(root.glob("*.json")):
        if path.name in {"manifest.json", "run.json", "summary.json", "baselines.json"}:
            continue
        payload = json.loads(path.read_text())
        if "condition" in payload:
            seen.add(payload["condition"])
            rows.append({k: payload.get(k) for k in ("condition", "status", "seed", "best_validation_mse", "best_step", "seconds", "trainable_parameters", "feature_cache_lora")})
    for condition in expected:
        if condition not in seen:
            rows.append({"condition": condition, "status": "unrun"})
    baseline_path = root / "baselines.json"
    baselines = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    result = {"format": 1, "root": str(root), "conditions": rows, "baselines": baselines.get("methods", {})}
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["condition,status,seed,best_validation_mse,best_step,seconds,trainable_parameters,feature_cache_lora"]
    lines.extend(",".join(str(row.get(k, "")) for k in ("condition", "status", "seed", "best_validation_mse", "best_step", "seconds", "trainable_parameters", "feature_cache_lora")) for row in rows)
    (root / "summary.csv").write_text("\n".join(lines) + "\n")
    return result
