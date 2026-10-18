"""Rehearse the frozen FINAL archive without reading final-task demo contents.

Extract into a new dir, verify every file hash, then run python3 -I with only the package on sys.path: imports
come from the package; resolve-paths --stage final is refused without the freeze file and accepted with it (this
hashes the HDF5 files, it does not open them); the benchmark refuses the final stage without the freeze file; no
development dataset is packaged. No frames, actions or labels of final tasks are loaded on the Mac."""
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
    freeze = str(root / "repo/final_analysis_amendment.json")
    base = ["hybrid_cli", "--protocol", str(root / "repo/protocol.json"), "resolve-paths", "--data-root",
            str(root / "inputs/dataset"), "--stage", "final", "--output", str(root / "collection/protocol_bench_pod.json")]
    def cli(argv):
        sys.argv = argv
        try:
            runpy.run_module("raid_experiments.hybrid_cli", run_name="__main__")
        except SystemExit as e:
            return e.code
        return 0
    refused = cli(base)
    assert isinstance(refused, str) and "final-stage access refused" in refused, refused
    assert not (root / "collection/protocol_bench_pod.json").exists()
    assert not cli(base + ["--freeze-file", freeze])
    protocol = json.loads((root / "collection/protocol_bench_pod.json").read_text())
    from raid_experiments import hybrid_compute_bench as bench, hybrid_study as hs, hybrid_baselines as hb
    final = hs.final_access(protocol, freeze)
    dev = [t for t, s in protocol["tasks"].items() if s["role"] == "development"]
    fin = sorted(t for t, s in protocol["tasks"].items() if s["role"] == "final_test")
    packaged = {p.name for p in (root / "inputs/dataset").iterdir()}
    assert not any(Path(protocol["tasks"][t]["path"]).name in packaged for t in dev), "development task packaged"
    assert all(Path(protocol["tasks"][t]["path"]).name in packaged for t in fin) and len(fin) == 2
    for task in fin:
        for seed in bench.SEEDS:
            assert len(hs.support_ids(protocol, task, 5, seed)) == 5
    try:
        bench.run(root, device="cpu", suite="final", stage="final")
        raise AssertionError("final stage ran without a freeze file")
    except ValueError as e:
        assert "freeze file" in str(e)
    assert not (root / "collection/bench/final_access.json").exists() and not (root / "work").exists()
    assert bench.SUITES["final"] == (True, True, ("raid_v7_head",)) and "raid_v7_head" in hb.EXPLORATORY
    amendment = json.loads(Path(freeze).read_text())
    assert amendment["parent_protocol_digest"] == protocol["digest"] and amendment["held_out_claims_eligible"] is False
    print("FINAL_REHEARSAL_OK", json.dumps({"final_tasks": fin, "freeze_sha256": final["freeze_sha256"]}))
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
    ok = done.returncode == 0 and "FINAL_REHEARSAL_OK" in done.stdout
    (P / "rehearsal_result.json").write_text(json.dumps({"status": "passed" if ok else "failed",
        "archive_sha256": release["archive_sha256"], "code_sha256": {r["path"]: r["sha256"] for r in rows if r["path"].endswith(".py")},
        "stages": [s["name"] for s in release["stages"]], "stdout_tail": done.stdout[-400:],
        "final_demo_contents_read": False}, indent=2) + "\n")
    print("passed" if ok else "failed"); return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
