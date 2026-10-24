"""Rehearse the exact v8 development archive on this Mac (CPU, truncated real LIBERO files; no final suites).

    python3 rehearse.py SCRATCH MINI_DATA FULL_GOAL_DATA SIM_PYTHON

1. Extract, verify every file hash and the manifest, `bash -n` every stage command.
2. With only the package's repo on sys.path: tiny development run (cache, source, offline, encoder), selection
   and freeze; the final stage is refused without a frozen amendment.
3. Closed loop with the packaged LIBERO source in the simulator venv (3 episodes, real full-length demos).
"""
from pathlib import Path
import hashlib, json, os, subprocess, sys, tarfile

P = Path(__file__).resolve().parent
TINY = {"encoders": ["mae"], "seeds": [42], "source_updates": 20, "source_val_every": 10,
        "source_heads": [["v7", 0.0], ["v8_c0", 0.0], ["v8_c25", 0.25]], "base_head": "v8_c25",
        "head_methods": [["head_only", "v8_c25", False], ["raid_v7", "v7", True], ["raid_v8_c0", "v8_c0", True],
                         ["raid_v8_c25", "v8_c25", True]],
        "head_lrs": {"head_only": [3e-4], "raid_v7": [3e-4], "raid_v8_c0": [3e-4], "raid_v8_c25": [3e-4]},
        "head_checkpoints": [0, 5], "encoder_seed": 42, "encoder_tasks": ["libero_goal/turn_on_the_stove"],
        "encoder_lrs": {"lora": [1e-4], "full": [1e-5]}, "encoder_head_lr": 3e-4, "encoder_checkpoints": [0, 2],
        "select_head_budget": 5, "select_encoder_budget": 2,
        "closed_loop": {"seed": 42, "episodes": 3, "adapted": [["raid_v8_c25", "v8_c25", True, 3e-4, 5]],
                        "encoder": [["lora", 1e-4, 2]]}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, env=None, cwd=None):
    done = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    tail = (done.stdout[-1500:] + "\n" + done.stderr[-2500:])
    if done.returncode:
        raise RuntimeError(f"{cmd[:4]} failed:\n{tail}")
    return done.stdout


def main(scratch, mini, full_goal, sim_python):
    scratch = Path(scratch).resolve(); root = scratch / "extracted"; root.mkdir(parents=True)
    release = json.loads((P / "release.json").read_text())
    assert sha(release["archive"]) == release["archive_sha256"]
    with tarfile.open(release["archive"]) as t:
        t.extractall(root, filter="data")
    rows = json.loads((root / "package_manifest.json").read_text())["files"]
    assert {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == {r["path"] for r in rows} | {"package_manifest.json"}
    assert all(sha(root / r["path"]) == r["sha256"] for r in rows)
    for stage in release["stages"]:
        assert subprocess.run(["bash", "-n"], input=stage["command"], text=True).returncode == 0, stage["name"]
    repo = root / "repo"
    (scratch / "tiny.json").write_text(json.dumps(TINY))
    env = {**os.environ, "PYTHONPATH": str(repo), "PYTHONNOUSERSITE": "1"}
    bench = [sys.executable, "-m", "raid_experiments.v8.bench", "--stage", "development", "--config",
             str(scratch / "tiny.json"), "--data", str(mini), "--cache", str(scratch / "cache"), "--out",
             str(scratch / "out"), "--root", str(repo), "--device", "cpu"]
    run(bench + ["--steps", "cache,source,offline,encoder"], env=env, cwd=scratch)
    run([sys.executable, "-m", "raid_experiments.v8.analysis", "select", "--dev", str(scratch / "out"), "--config",
         str(scratch / "tiny.json"), "--write", str(scratch / "amend.json")], env=env, cwd=scratch)
    amend = json.loads((scratch / "amend.json").read_text())
    assert amend["status"] == "frozen" and amend["code_sha256"]["bench.py"] == sha(repo / "raid_experiments/v8/bench.py")
    refused = subprocess.run(bench[:3] + ["--stage", "final", "--config", str(scratch / "tiny.json")] + bench[7:] +
                             ["--steps", "offline"], capture_output=True, text=True, env=env, cwd=scratch)
    assert refused.returncode != 0 and "frozen amendment" in (refused.stdout + refused.stderr), "final stage not refused"
    import torch
    sources = scratch / "out" / "sources"
    # closed loop with the packaged LIBERO source, full-length development demos
    sim_env = {**env, "PYTHONPATH": f"{root / 'inputs/libero_repo'}:{repo}", "MUJOCO_GL": "glfw",
               "LIBERO_CONFIG_PATH": str(scratch / "libero-config")}
    (scratch / "run_cl.py").write_text("import sys\nfrom raid_experiments.v8 import bench\n"
                                       "if __name__ == '__main__':\n    bench.main(sys.argv[1:])\n")
    out = run([sim_python, "-W", "ignore", str(scratch / "run_cl.py"), "--stage", "development", "--config",
               str(scratch / "tiny.json"), "--data", str(full_goal), "--cache", str(scratch / "cache_full"),
               "--out", str(scratch / "out_cl"), "--sources", str(sources), "--libero-root",
               str(root / "inputs/libero_repo"), "--root", str(repo), "--device", "cpu",
               "--steps", "cache,closed_loop"], env=sim_env, cwd=scratch)
    cl = [json.loads(p.read_text()) for p in (scratch / "out_cl" / "closed_loop").glob("*.json")]
    assert cl and all(len(m["episodes"]) == 3 for u in cl for m in u["methods"].values())
    knn = cl[0]["methods"]["knn"]["success_rate"]
    result = {"status": "passed", "archive_sha256": release["archive_sha256"], "stages": [s["name"] for s in release["stages"]],
              "tiny_amendment_digest": amend["digest"], "final_stage_refused": True,
              "closed_loop_methods": sorted(cl[0]["methods"]), "closed_loop_knn_success": knn,
              "closed_loop_flip": cl[0]["methods"]["knn"]["flip"], "final_suites_read": False}
    (P / "rehearsal_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:5]))
