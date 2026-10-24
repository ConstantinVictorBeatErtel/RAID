"""Rehearse the exact v8 FINAL archive on this Mac without any final-suite data.

Extract, verify every hash, `bash -n` every stage; with only the package on sys.path: the amendment is frozen and binds
the packaged code; all bound source checkpoints load; the final stage refuses a tampered amendment; offline and
encoder steps run end to end on a stand-in dataset (development files placed under final-suite folder names, CPU).
    python3 rehearse.py SCRATCH STANDIN_DATA
"""
from pathlib import Path
import hashlib, json, os, subprocess, sys, tarfile
P = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(scratch, standin):
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
    repo = root / "repo"; env = {**os.environ, "PYTHONPATH": str(repo), "PYTHONNOUSERSITE": "1"}
    amend = json.loads((repo / "amendment.json").read_text())
    check = ("import json,hashlib,pathlib\nfrom raid_experiments.v8.bench import config_digest, load_source\n"
             f"a=json.loads(pathlib.Path({str(repo / 'amendment.json')!r}).read_text())\n"
             "assert a['status']=='frozen' and a['digest']==config_digest(a)\n"
             "code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pathlib.Path('raid_experiments/v8').glob('*.py'))}\n"
             "assert code==a['code_sha256'], 'packaged code differs from the frozen amendment'\n"
             f"src=pathlib.Path({str(repo / 'sources')!r})\n"
             "n=0\nfor e in a['encoders']:\n for h,_ in a['source_heads']:\n  for s in a['seeds']:\n   load_source(src,e,h,s,'cpu');n+=1\nprint('SOURCES_OK',n)\n")
    out = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, env=env, cwd=repo)
    assert "SOURCES_OK" in out.stdout, out.stderr[-2000:]
    tampered = scratch / "tampered.json"; t = dict(amend); t["seeds"] = [42]; tampered.write_text(json.dumps(t))
    bench = [sys.executable, "-m", "raid_experiments.v8.bench", "--stage", "final", "--data", str(standin),
             "--cache", str(scratch / "cache"), "--sources", str(repo / "sources"), "--root", str(repo), "--device", "cpu"]
    refused = subprocess.run(bench + ["--config", str(tampered), "--out", str(scratch / "t"), "--steps", "offline"],
                             capture_output=True, text=True, env=env, cwd=repo)
    assert refused.returncode != 0 and "frozen amendment" in refused.stdout + refused.stderr
    tiny = dict(amend); tiny.update(encoders=["mae"], seeds=[42], head_checkpoints=[0, 5], encoder_checkpoints=[0, 2])
    from_digest = subprocess.run([sys.executable, "-c", "import json,sys\nfrom raid_experiments.v8.bench import config_digest\n"
                                  "a=json.load(open(sys.argv[1]));a['digest']=config_digest(a);json.dump(a,open(sys.argv[1],'w'))",
                                  str(scratch / "tiny.json")], env=env, cwd=repo) if (scratch / "tiny.json").write_text(json.dumps(tiny)) else None
    run = subprocess.run(bench + ["--config", str(scratch / "tiny.json"), "--out", str(scratch / "out"), "--steps", "cache,offline,encoder"],
                         capture_output=True, text=True, env=env, cwd=repo)
    assert run.returncode == 0, run.stderr[-3000:]
    n_off = len(list((scratch / "out/offline/mae").glob("*.json"))); n_enc = len(list((scratch / "out/encoder").glob("*.json")))
    assert n_off == 3 and n_enc == 3, (n_off, n_enc)
    result = {"status": "passed", "archive_sha256": release["archive_sha256"], "amendment_digest": amend["digest"],
              "sources_loaded": int(out.stdout.split("SOURCES_OK")[1]), "tampered_refused": True, "standin_offline_units": n_off,
              "final_suites_read": False, "stages": [s["name"] for s in release["stages"]]}
    (P / "rehearsal_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
