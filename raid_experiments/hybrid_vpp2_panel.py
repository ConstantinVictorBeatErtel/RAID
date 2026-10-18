"""VPP2 15-case source panel, source-only timing calibration and review materials.

Host side (no model weights): prepare hash-verified source inputs, analyze the
collected clips with the frozen reference MAE, rebind generator-independent
expert-replay evidence, and write rating materials. Pod side (VPP2 runtime):
download pinned assets, then generate one uncached plan per case with the
production prompt policy and measure costs. Nothing here rates clips; human
rubric decisions must come from named people.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import time

import numpy as np

from .hybrid_protocol import is_vpp2, temporal_alignment, validate_hybrid_protocol, with_vpp2_timing
from . import hybrid_vpp2 as v

PANEL_DIR = "vpp2_panel"
PROMPT_KEY = "/root/raid-model-cache/openrouter-key"


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, body) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _require_vpp2(protocol: dict) -> None:
    validate_hybrid_protocol(protocol)
    if not is_vpp2(protocol):
        raise ValueError("the VPP2 panel requires a VPP2-amended protocol")


# ---------------------------------------------------------------- host: inputs

def prepare_inputs(protocol: dict, data_root: str | Path, out_dir: str | Path) -> dict:
    """Write the registered panel observations (frame t only) from source-validation demos."""
    import h5py
    _require_vpp2(protocol)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    cases = v.vpp2_panel_cases(protocol)
    verified = {}
    rows = []
    for case in cases:
        spec = protocol["tasks"][case["task"]]
        # Stage access precedes any dataset hashing or read: source tasks only.
        if spec["role"] != "source" or int(case["demo"]) not in set(map(int, spec["source"]["source_validation"])):
            raise ValueError("panel inputs must be source-validation demos of source tasks")
        path = Path(data_root) / Path(spec["path"]).name
        if case["task"] not in verified:
            if sha(path) != spec["source_sha256"]:
                raise ValueError(f"source dataset changed: {case['task']}")
            verified[case["task"]] = str(path)
        with h5py.File(path, "r") as handle:
            frames = np.asarray(handle[f"data/demo_{int(case['demo'])}/obs/agentview_rgb"][int(case["t"]):],
                                dtype=np.uint8)
        observation = frames[0].copy()
        name = case["case_id"].replace(":", "_") + ".npz"
        recorded = case["case_id"].replace(":", "_") + ".recorded.npz"
        np.savez_compressed(out / name, observation=observation)
        # Source continuation for timing calibration and review only; generation never opens it.
        np.savez_compressed(out / recorded, frames=frames)
        rows.append({k: val for k, val in case.items() if k != "path"} | {
            "input": name, "input_sha256": sha(out / name),
            "recorded": recorded, "recorded_sha256": sha(out / recorded),
            "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
            "dataset_sha256": spec["source_sha256"]})
    manifest = {"cases": rows, "n": len(rows), "protocol_digest": protocol["digest"],
                "model_digest": protocol["model_digest"], "inputs": "current RGB frame and instruction only",
                "scientific_admission": False, "held_out_claims_eligible": False}
    atomic(out / "cases.json", manifest)
    return manifest


def verified_inputs(root: Path, protocol: dict) -> list[dict]:
    manifest = json.loads((root / "inputs/cases.json").read_text())
    n = int(protocol["design"]["vpp2_panel"]["cases"])
    if (manifest["protocol_digest"] != protocol["digest"] or manifest["n"] != n or len(manifest["cases"]) != n
            or [c["case_id"] for c in manifest["cases"]] != [c["case_id"] for c in v.vpp2_panel_cases(protocol)]):
        raise ValueError("panel inputs do not belong to this protocol")
    seen = set()
    for case in manifest["cases"]:
        path = (root / "inputs" / case["input"]).resolve()
        recorded = (root / "inputs" / case["recorded"]).resolve()
        if (not path.is_relative_to((root / "inputs").resolve()) or sha(path) != case["input_sha256"]
                or not recorded.is_relative_to((root / "inputs").resolve()) or sha(recorded) != case["recorded_sha256"]):
            raise ValueError(f"panel input changed: {case['case_id']}")
        if case["case_id"] in seen or case["seed"] != v.generation_seed_for(protocol, case["case_id"], 0):
            raise ValueError("duplicate case or unregistered seed")
        seen.add(case["case_id"])
    return manifest["cases"]


# ---------------------------------------------------------------- pod: assets

def download(root: str | Path) -> dict:
    """Fetch the pinned VPP2 files directly on Runpod and verify each one."""
    from huggingface_hub import hf_hub_download
    root = Path(root)
    registry = json.loads((root / "repo/vpp2_assets.json").read_text())
    dest = root / "weights"
    if shutil.disk_usage(root).free < sum(f["size"] for f in registry["files"]) + 12 * 1024 ** 3:
        raise RuntimeError("insufficient volume space for pinned assets")
    rows, started_all = [], time.time()
    for entry in registry["files"]:
        started = time.time()
        atomic(root / "collection/download_progress.json", {"file": entry["path"], "completed": len(rows), "at": started})
        path = Path(hf_hub_download(registry["model_id"], entry["path"], revision=registry["revision"], local_dir=str(dest)))
        if path.stat().st_size != entry["size"]:
            raise RuntimeError(f"size mismatch: {entry['path']}")
        got = sha(path)
        if entry.get("sha256") and got != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch: {entry['path']}")
        if entry.get("git_blob_sha1"):
            blob = hashlib.sha1(b"blob " + str(entry["size"]).encode() + b"\0" + path.read_bytes()).hexdigest()
            if blob != entry["git_blob_sha1"]:
                raise RuntimeError(f"git blob mismatch: {entry['path']}")
        rows.append({"path": entry["path"], "sha256": got, "size_bytes": entry["size"], "seconds": time.time() - started})
    report = {"model_id": registry["model_id"], "revision": registry["revision"], "files": rows, "complete": True,
              "seconds": time.time() - started_all, "bytes": sum(r["size_bytes"] for r in rows)}
    atomic(root / "collection/vpp2_download.json", report)
    return report


MAE_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
MAE_SHA256 = "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"


def download_mae(root: str | Path) -> dict:
    """Frozen reference MAE for timing features, verified against the study checksum."""
    import urllib.request
    root = Path(root)
    target = root / "weights/mae_pretrain_vit_base.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not target.is_file() or sha(target) != MAE_SHA256:
        tmp = target.with_suffix(".tmp")
        urllib.request.urlretrieve(MAE_URL, tmp)
        if sha(tmp) != MAE_SHA256:
            tmp.unlink()
            raise RuntimeError("MAE checksum mismatch")
        tmp.replace(target)
    report = {"url": MAE_URL, "sha256": MAE_SHA256, "seconds": time.time() - started}
    atomic(root / "collection/mae_download.json", report)
    return report


# ---------------------------------------------------------------- pod: panel

def _frame0_error(plan_frames: np.ndarray, observation: np.ndarray) -> float:
    band = v.controller_band(observation).astype(np.float32)
    return float(np.abs(plan_frames[0].astype(np.float32) - band).mean())


def run_panel(root: str | Path, *, device: str = "cuda") -> dict:
    """Generate every registered uncached plan with the production adapter; record costs."""
    import torch
    from .world_model import write_mp4
    root = Path(root)
    protocol = json.loads((root / "repo/protocol.json").read_text())
    _require_vpp2(protocol)
    cases = verified_inputs(root, protocol)
    out = root / "collection" / PANEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    world = v.load_production_world_model(root / "weights", PROMPT_KEY, root / "cache", protocol, device)
    load_seconds = time.perf_counter() - started
    rows = []
    for index, case in enumerate(cases):
        with np.load(root / "inputs" / case["input"], allow_pickle=False) as payload:
            observation = payload["observation"]
        if hashlib.sha256(observation.tobytes()).hexdigest() != case["observation_sha256"]:
            raise ValueError("observation mismatch")
        atomic(out / "progress.json", {"current": case["case_id"], "completed": len(rows), "at": time.time()})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        plan = world.generate(observation, case["instruction"], seed=case["seed"], prompt_slot=index % world.prompt_slots)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        case_dir = out / "clips" / case["case_id"].replace(":", "_")
        case_dir.mkdir(parents=True, exist_ok=True)
        entry = world.cache_root / plan.cache_key[:2] / plan.cache_key
        shutil.copyfile(entry / "generated.mp4", case_dir / "generated_canvas.mp4")
        write_mp4(plan.display_frames, case_dir / "generated_main_view.mp4", world.temporal.display_fps)
        np.savez_compressed(case_dir / "controller_band.npz", frames=plan.frames)
        canvas, _ = v.compose_canvas(np.ascontiguousarray(observation[::-1]))
        from PIL import Image
        Image.fromarray(canvas).save(case_dir / "conditioning_canvas.png")
        meta = plan.metadata
        technical = {
            "frames": int(len(plan.frames)), "controller_shape": list(plan.frames.shape[1:]),
            "finite_uint8": bool(plan.frames.dtype == np.uint8),
            "conditioning_matches_input": meta["conditioning_sha256"] == hashlib.sha256(canvas.tobytes()).hexdigest(),
            "native_action_output_used": False, "cache_hit": plan.cache_hit,
            "frame0_band_mean_abs_error": _frame0_error(plan.frames, observation),
        }
        technical["ok"] = (technical["frames"] == 49 and technical["controller_shape"] == [96, 128, 3]
                           and technical["finite_uint8"] and technical["conditioning_matches_input"]
                           and not plan.cache_hit)
        row = {**case, "cache_key": plan.cache_key, "prompt": plan.prompt,
               "prompt_model": meta["prompt_record"]["model_id"], "prompt_slot": index % world.prompt_slots,
               "prompt_record": meta["prompt_record"], "prompt_fallback": meta["prompt_fallback"],
               "prompt_seconds": meta["prompt_seconds"], "video_seconds": meta["video_seconds"],
               "wall_seconds": wall, "frames_sha256": meta["frames_sha256"], "technical": technical,
               "quality_rating": None}
        atomic(case_dir / "case.json", row)
        rows.append(row)
    sizes = [sum(p.stat().st_size for p in (out / "clips" / r["case_id"].replace(":", "_")).iterdir()) for r in rows]
    cache_sizes = [sum(p.stat().st_size for p in (world.cache_root / r["cache_key"][:2] / r["cache_key"]).iterdir())
                   for r in rows]
    summary = {
        "n": len(rows), "protocol_digest": protocol["digest"], "model_digest": protocol["model_digest"],
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "load_seconds": {"total": load_seconds, **world.load_seconds},
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "prompt_seconds": [r["prompt_seconds"] for r in rows],
        "video_seconds": [r["video_seconds"] for r in rows],
        "plan_wall_seconds": [r["wall_seconds"] for r in rows],
        "median_plan_wall_seconds": statistics.median(r["wall_seconds"] for r in rows),
        "p90_plan_wall_seconds": float(np.percentile([r["wall_seconds"] for r in rows], 90)),
        "prompt_fallbacks": sum(bool(r["prompt_fallback"]) for r in rows),
        "technical_passes": sum(r["technical"]["ok"] for r in rows),
        "collected_bytes_per_case_mean": float(np.mean(sizes)),
        "cache_bytes_per_plan_mean": float(np.mean(cache_sizes)),
        "human_ratings": None, "timing_calibrated": False,
        "scientific_admission": False, "held_out_claims_eligible": False,
    }
    atomic(out / "panel_summary.json", summary)
    return summary


def check_collection(root: str | Path) -> dict:
    import imageio.v3 as iio
    root = Path(root)
    out = root / "collection" / PANEL_DIR
    summary = json.loads((out / "panel_summary.json").read_text())
    rows = []
    for case_dir in sorted((out / "clips").iterdir()):
        canvas = iio.imread(case_dir / "generated_canvas.mp4")
        main = iio.imread(case_dir / "generated_main_view.mp4")
        with np.load(case_dir / "controller_band.npz") as payload:
            band = payload["frames"]
        ok = canvas.shape == (49, 240, 416, 3) and main.shape == (49, 240, 320, 3) and band.shape == (49, 96, 128, 3)
        rows.append({"case": case_dir.name, "decoded": bool(ok)})
    report = {"cases": rows, "decoded": sum(r["decoded"] for r in rows), "n": len(rows),
              "summary_technical_passes": summary["technical_passes"]}
    if report["n"] != summary["n"] or report["decoded"] != summary["n"]:
        atomic(out / "decode_check.json", report)
        raise RuntimeError("panel clips are missing or failed to decode")
    atomic(out / "decode_check.json", report)
    return report


# ---------------------------------------------------------------- host: timing

def align_steps_per_frame(generated: np.ndarray, recorded: np.ndarray) -> dict:
    """Monotone DP alignment of generated frames to recorded steps (free end)."""
    import torch
    g = torch.nn.functional.normalize(torch.as_tensor(generated, dtype=torch.float32), dim=-1)
    r = torch.nn.functional.normalize(torch.as_tensor(recorded, dtype=torch.float32), dim=-1)
    sim = g @ r.T
    n_g, n_r = sim.shape
    score = torch.full((n_g, n_r), float("-inf"))
    back = torch.zeros((n_g, n_r), dtype=torch.long)
    score[0, 0] = sim[0, 0]
    for j in range(1, n_g):
        best = torch.cummax(score[j - 1], dim=0)
        score[j] = best.values + sim[j]
        back[j] = best.indices
    path = [int(torch.argmax(score[-1]))]
    for j in range(n_g - 1, 0, -1):
        path.append(int(back[j, path[-1]]))
    path.reverse()
    frames = np.arange(n_g, dtype=np.float64)
    steps = np.asarray(path, dtype=np.float64)
    slope = float((frames * steps).sum() / (frames * frames).sum())
    mean_sim = float(sim[torch.arange(n_g), torch.tensor(path)].mean())
    return {"steps_per_frame": slope, "alignment": path, "mean_similarity": mean_sim,
            "rmse_steps": float(np.sqrt(((steps - slope * frames) ** 2).mean())),
            "plausible": bool(math.isfinite(slope) and 0.25 <= slope <= 20.0 and mean_sim >= 0.5)}


def local_offset(generated_frame: np.ndarray, recorded: np.ndarray, max_offset: int) -> int:
    """Recorded offset (1..max_offset) best matching one generated frame."""
    import torch
    g = torch.nn.functional.normalize(torch.as_tensor(generated_frame, dtype=torch.float32), dim=-1)
    r = torch.nn.functional.normalize(torch.as_tensor(recorded[1:max_offset + 1], dtype=torch.float32), dim=-1)
    return int(torch.argmax(r @ g)) + 1


def _recorded(root: Path, case: dict) -> np.ndarray:
    with np.load(root / "inputs" / case["recorded"], allow_pickle=False) as payload:
        return payload["frames"]


def calibrate_timing(root: str | Path, mae_path: str | Path | None = None, *, device: str = "cuda") -> dict:
    """Apply the pre-registered rule in protocol['design']['timing_calibration_rule'] (on the pod)."""
    import torch
    from src.mae_encoder import MAEImageEncoder
    root = Path(root)
    protocol = json.loads((root / "repo/protocol.json").read_text())
    _require_vpp2(protocol)
    cases = {c["case_id"]: c for c in verified_inputs(root, protocol)}
    temporal = temporal_alignment(protocol)
    chunk = temporal.chunk_steps
    mae_path = Path(mae_path or root / "weights/mae_pretrain_vit_base.pth")
    if sha(mae_path) != MAE_SHA256:
        raise ValueError("reference MAE identity mismatch")
    encoder = MAEImageEncoder.from_checkpoint(str(mae_path), device if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def encode(frames):
        return torch.cat([encoder.encode_frames(torch.from_numpy(np.ascontiguousarray(frames[i:i + 64]))).float().cpu()
                          for i in range(0, len(frames), 64)])

    rows, feats = [], {}
    for case_dir in sorted((root / "collection" / PANEL_DIR / "clips").iterdir()):
        row = json.loads((case_dir / "case.json").read_text())
        case = cases[row["case_id"]]
        if protocol["tasks"][case["task"]]["role"] != "source":
            raise ValueError("timing calibration may only read source tasks")
        recorded = v.controller_band(_recorded(root, case))
        with np.load(case_dir / "controller_band.npz") as payload:
            generated = payload["frames"]
        g, r = encode(generated), encode(recorded)
        fit = align_steps_per_frame(g.numpy(), r.numpy())
        feats[case["case_id"]] = (g, r)
        rows.append({"case_id": case["case_id"], "task": case["task"], "coverage": case["coverage"],
                     "prompt_model": row.get("prompt_model"), "recorded_steps_available": int(len(recorded)), **fit})
    plausible = [row["steps_per_frame"] for row in rows if row["plausible"]]
    spf = float(statistics.median(plausible)) if plausible else None
    requested = max(1, int(round(chunk / spf))) if spf else None
    offsets = []
    if requested is not None and requested <= temporal.num_frames - 1:
        for row in rows:
            g, r = feats[row["case_id"]]
            limit = min(4 * chunk, len(r) - 1)
            if limit >= 1:
                row["requested_frame_local_offset"] = local_offset(g[requested].numpy(), r.numpy(), limit)
                offsets.append(row["requested_frame_local_offset"])
    median_offset = float(statistics.median(offsets)) if offsets else None
    reasons = []
    needed = math.ceil(0.8 * len(rows))
    if len(plausible) < needed:
        reasons.append(f"plausible_cases:{len(plausible)}<{needed}")
    if requested is None or not 1 <= requested <= temporal.num_frames - 1:
        reasons.append("requested_frame_outside_clip")
    if median_offset is None or not chunk / 2 <= median_offset <= 2 * chunk:
        reasons.append(f"median_local_offset_outside_[{chunk / 2},{2 * chunk}]:{median_offset}")
    evidence = {"rule": protocol["design"]["timing_calibration_rule"], "protocol_digest": protocol["digest"],
                "cases": rows, "n": len(rows), "plausible_cases": len(plausible),
                "steps_per_frame_median": spf, "requested_frame": requested,
                "requested_frame_local_offsets": offsets, "median_local_offset": median_offset,
                "passed": not reasons, "reasons": reasons, "encoder_sha256": MAE_SHA256,
                "data_scope": "source_validation_only", "scientific_result": False,
                "label": "source-only engineering timing calibration; not a physical executability proof"}
    atomic(root / "collection" / PANEL_DIR / "timing_calibration.json", evidence)
    return evidence


def stamp_calibrated_protocol(protocol: dict, evidence_path: str | Path, out_path: str | Path) -> dict:
    evidence = json.loads(Path(evidence_path).read_text())
    if not evidence.get("passed") or evidence.get("protocol_digest") != protocol["digest"]:
        raise ValueError("timing evidence did not pass for this protocol")
    stamped = with_vpp2_timing(protocol, sha(evidence_path), steps_per_frame=evidence["steps_per_frame_median"],
                               source_protocol_digest=protocol["digest"])
    out = Path(out_path)
    with out.open("x") as handle:
        handle.write(json.dumps(stamped, indent=2, sort_keys=True) + "\n")
    return stamped


# ---------------------------------------------------------------- host: evidence reuse

def rebind_expert_replays(protocol: dict, parent_dir: str | Path, out_path: str | Path) -> dict:
    """Reuse generator-independent expert replays only with identical case/source identity."""
    from .hybrid_panel import select_panel_cases
    _require_vpp2(protocol)
    parent_dir = Path(parent_dir)
    manifest = json.loads((parent_dir / "outputs_sha256.json").read_text())
    for rel in ("source_panel/expert_replays.json", "source_panel/panel_cases.json", "protocol_v6_pod.json"):
        if sha(parent_dir / rel) != manifest[rel]:
            raise ValueError(f"parent evidence changed: {rel}")
    parent = json.loads((parent_dir / "protocol_v6_pod.json").read_text())
    if protocol["generator_amendment"]["parent_protocol_digest"] != parent["digest"]:
        raise ValueError("expert replays belong to a different parent protocol")
    replays = json.loads((parent_dir / "source_panel/expert_replays.json").read_text())
    old_cases = json.loads((parent_dir / "source_panel/panel_cases.json").read_text())["cases"]
    new_cases = select_panel_cases(protocol)
    key = lambda c: (c["case_id"], c["task"], int(c["demo"]), int(c["t"]))
    if [key(c) for c in new_cases] != [key(c) for c in old_cases]:
        raise ValueError("panel case identities differ from the replayed cases")
    for task in {c["task"] for c in new_cases}:
        if protocol["tasks"][task]["source_sha256"] != parent["tasks"][task]["source_sha256"]:
            raise ValueError("source dataset identity changed")
    if protocol["design"]["state_schema"] != parent["design"]["state_schema"]:
        raise ValueError("robot state schema changed")
    record = {"source": str(parent_dir / "source_panel/expert_replays.json"),
              "source_sha256": manifest["source_panel/expert_replays.json"],
              "parent_protocol_digest": parent["digest"], "protocol_digest": protocol["digest"],
              "expert_replay_successes": replays["expert_replay_successes"], "n": replays["n"],
              "gate_threshold": 27, "gate_pass": replays["expert_replay_successes"] >= 27,
              "reuse_basis": "replays execute recorded source actions from registered starts; they do not "
                             "depend on the generator, prompt or controller view. Case, demo, start, task "
                             "dataset hashes and state schema verified identical.",
              "scientific_result": False}
    atomic(out_path, record)
    return record


# ---------------------------------------------------------------- host: review

def write_review_materials(root: str | Path) -> dict:
    """Side-by-side generated vs recorded videos, an index page and an empty rating sheet."""
    from .hybrid_collect import RUBRIC_TEMPLATE
    from .world_model import resize_rgb, write_mp4
    root = Path(root)
    protocol = json.loads((root / "repo/protocol.json").read_text())
    _require_vpp2(protocol)
    cases = {c["case_id"]: c for c in verified_inputs(root, protocol)}
    timing = root / "collection" / PANEL_DIR / "timing_calibration.json"
    steps_per_frame = json.loads(timing.read_text()).get("steps_per_frame_median") if timing.is_file() else None
    collection, out = root / "collection", root / "collection" / PANEL_DIR / "review"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_dir in sorted((collection / PANEL_DIR / "clips").iterdir()):
        case = json.loads((case_dir / "case.json").read_text())
        recorded = _recorded(root, cases[case["case_id"]])
        import imageio.v3 as iio
        generated = iio.imread(case_dir / "generated_main_view.mp4")
        if steps_per_frame:
            index = [min(len(recorded) - 1, int(round(j * steps_per_frame))) for j in range(len(generated))]
        else:
            index = np.linspace(0, len(recorded) - 1, len(generated)).round().astype(int).tolist()
        reference = resize_rgb(np.stack([v.controller_band(recorded[i])[::-1] for i in index]),
                               generated.shape[1:3])
        side = np.concatenate([generated, reference], axis=2)
        name = case_dir.name + "_generated_vs_recorded.mp4"
        write_mp4(side, out / name, 6)
        rows.append({"case_id": case["case_id"], "task": case["task"], "coverage": case["coverage"],
                     "video": name, "prompt": case["prompt"], "prompt_fallback": case["prompt_fallback"],
                     "prompt_model": case.get("prompt_model")})
    sheet = {"instructions": "Rate each generated clip (left) against the rubric. The recorded demonstration "
                             "(right) is context, not a target. Enter your name; leave unknown if unsure.",
             "rubric": RUBRIC_TEMPLATE, "annotator": None,
             "cases": [{"case_id": r["case_id"], "video": r["video"], "pass": None, "notes": ""} for r in rows]}
    atomic(out / "rating_sheet_empty.json", sheet)
    cards = "\n".join(
        f"<section><h2>{i + 1}. {r['task'].replace('_', ' ')} <small>({r['coverage']})</small></h2>"
        f"<video src='{r['video']}' controls loop muted playsinline></video>"
        f"<p><b>Case</b> {r['case_id']} · <b>prompt model</b> {r['prompt_model']}{' · <b>fallback prompt</b>' if r['prompt_fallback'] else ''}</p>"
        f"<details><summary>Prompt</summary><p>{r['prompt']}</p></details></section>" for i, r in enumerate(rows))
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>VPP2 source panel</title><style>body{font:15px system-ui;max-width:900px;margin:auto;padding:16px}"
        "video{width:100%;background:#000}section{margin:28px 0}</style>"
        "<h1>VPP2 source panel: 15 cases</h1><p>Left: VPP2 main view. Right: recorded source demonstration "
        "(same crop). Not a robot success measure.</p>" + cards)
    return {"cases": len(rows), "index": str(out / "index.html")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "download-mae", "run", "check", "calibrate", "review"])
    parser.add_argument("root")
    a = parser.parse_args(argv)
    {"download": download, "download-mae": download_mae, "run": run_panel, "check": check_collection,
     "calibrate": calibrate_timing, "review": write_review_materials}[a.command](a.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
