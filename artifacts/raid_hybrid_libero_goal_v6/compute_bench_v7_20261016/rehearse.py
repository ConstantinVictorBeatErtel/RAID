"""Rehearse the frozen bench archive: extract into a new dir, verify, then python3 -I with only the package on sys.path."""
from pathlib import Path
import hashlib, json, subprocess, sys, tarfile, textwrap
P = Path(__file__).resolve().parent
POD = textwrap.dedent(r'''
    import importlib, json, runpy, sys
    from pathlib import Path
    root = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(root / "repo"))
    import raid_experiments
    assert Path(raid_experiments.__file__).resolve().is_relative_to(root)
    for p in sorted((root / "repo/raid_experiments").glob("*.py")):
        if p.stem != "__init__":
            m = importlib.import_module("raid_experiments." + p.stem)
            assert Path(m.__file__).resolve().is_relative_to(root), p
    (root / "collection").mkdir(exist_ok=True)
    sys.argv = ["hybrid_cli", "--protocol", str(root / "repo/protocol.json"), "resolve-paths", "--data-root",
                str(root / "inputs/dataset"), "--stage", "pilot", "--output", str(root / "collection/protocol_bench_pod.json")]
    try:
        runpy.run_module("raid_experiments.hybrid_cli", run_name="__main__")
    except SystemExit as e:
        assert not e.code, e.code
    protocol = json.loads((root / "collection/protocol_bench_pod.json").read_text())
    from raid_experiments import hybrid_study as hs, hybrid_compute_bench as bench
    dev = sorted(t for t, s in protocol["tasks"].items() if s["role"] == "development")
    finals = [t for t, s in protocol["tasks"].items() if s["role"] == "final_test"]
    assert not any((root / "inputs/dataset" / Path(protocol["tasks"][t]["path"]).name).exists() for t in finals), "sealed task packaged"
    for task in dev:
        spec = protocol["tasks"][task]
        frames = hs.read_view_frames(protocol, spec["path"], int(spec["query"][0]))
        for seed in bench.SEEDS:
            assert len(hs.support_ids(protocol, task, 5, seed)) == 5
    amendment = json.loads((root / "repo/bench_amendment_v7.json").read_text())
    assert amendment["parent_protocol_digest"] == protocol["digest"] or amendment["parent_protocol_digest"]
    print("BENCH_REHEARSAL_OK", json.dumps({"dev_tasks": dev, "frames": list(frames.shape)}))
''')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(scratch):
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
    (scratch / "pod.py").write_text(POD)
    done = subprocess.run([sys.executable, "-I", str(scratch / "pod.py"), str(root)], capture_output=True, text=True)
    print(done.stdout[-800:], done.stderr[-2500:])
    ok = done.returncode == 0 and "BENCH_REHEARSAL_OK" in done.stdout
    (P / "rehearsal_result.json").write_text(json.dumps({"status": "passed" if ok else "failed",
        "archive_sha256": release["archive_sha256"], "code_sha256": {r["path"]: r["sha256"] for r in rows if r["path"].endswith(".py")},
        "stages": [s["name"] for s in release["stages"]], "stdout_tail": done.stdout[-400:]}, indent=2) + "\n")
    print("passed" if ok else "failed"); return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
