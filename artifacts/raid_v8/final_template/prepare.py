"""Freeze the ONE RAID v8 FINAL package: code, MAE checkpoint, pinned LIBERO source, the frozen amendment and the
development source checkpoints it selects. The pod downloads only LIBERO-10/Spatial/Object (bench.py refuses the
final stage unless the amendment is frozen and its digest matches).
Copy this folder to artifacts/raid_v8/final_<date>/, put the frozen amendment there as amendment.json, then:
    python3 prepare.py /path/to/LIBERO-checkout-at-8f1084e /path/to/dev/collection/v8_dev/sources
"""
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, tarfile, uuid
from datetime import datetime, timezone
P = Path(__file__).resolve().parent; ROOT = P.parents[2]; sys.path.insert(0, str(ROOT))
from raid_experiments import operational_acceptance as o, managed_source as m, hybrid_vpp2_pilot as pilot
from raid_experiments.wm_pod import bootstrap_script, runtime_exports, runtime_python
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CONFIG = P / "amendment.json"
if (P / "release.json").exists():
    raise RuntimeError("immutable release exists")
libero, sources = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
amend = json.loads(CONFIG.read_text())
from raid_experiments.v8.bench import config_digest
assert amend["status"] == "frozen" and amend["digest"] == config_digest(amend), "amendment not frozen"
code_now = {p.name: o.sha(p) for p in sorted((ROOT / "raid_experiments/v8").glob("*.py"))}
assert code_now == amend["code_sha256"], "v8 code changed after the freeze"
head = subprocess.run(["git", "-C", str(libero), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
assert head == LIBERO_COMMIT, f"LIBERO checkout is {head}"
assert not subprocess.run(["git", "-C", str(libero), "status", "--porcelain"], capture_output=True, text=True).stdout.strip(), "LIBERO checkout modified"
st = P / "bundle"
if st.exists():
    shutil.rmtree(st)
files = {"repo/" + rel: ROOT / rel for rel in pilot.CODE_FILES}
files.update({"repo/raid_experiments/v8/" + p.name: p for p in sorted((ROOT / "raid_experiments/v8").glob("*.py"))})
files.update({"repo/" + n: ROOT / n for n in ("requirements-wm-cosmos.lock", "requirements-wm-sim.lock")})
files["repo/src/__init__.py"] = ROOT / "src/__init__.py" if (ROOT / "src/__init__.py").is_file() else None
files["repo/data/assets/mae/mae_pretrain_vit_base.pth"] = ROOT / "data/assets/mae/mae_pretrain_vit_base.pth"
files["repo/amendment.json"] = CONFIG
files["repo/sources/normalizers.pt"] = sources / "normalizers.pt"
for enc in amend["encoders"]:
    for label, _ in amend["source_heads"]:
        for seed in amend["seeds"]:
            files[f"repo/sources/{enc}/{label}_s{seed}.pt"] = sources / enc / f"{label}_s{seed}.pt"
files = {k: v for k, v in files.items() if v is not None}
for p in sorted((libero / "libero").rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts:
        files["inputs/libero_repo/" + p.relative_to(libero).as_posix()] = p
for remote, local in files.items():
    assert local.is_file() and not local.is_symlink(), f"missing input {local}"
    assert not pilot.PROHIBITED_NAME.search(Path(remote).name), "prohibited model file in package"
    target = st / remote; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(local, target)
rows = [{"path": p.relative_to(st).as_posix(), "sha256": o.sha(p), "size_bytes": p.stat().st_size}
        for p in sorted(st.rglob("*")) if p.is_file()]
manifest = {"files": rows, "stage": "raid_v8_final", "file_count": len(rows),
            "unpacked_bytes": sum(x["size_bytes"] for x in rows), "libero_commit": LIBERO_COMMIT}
o.atomic(st / "package_manifest.json", manifest)
archive = P / "v8_final_package.tar"
with tarfile.open(archive, "w") as t:
    for row in rows:
        t.add(st / row["path"], arcname=row["path"], recursive=False)
    t.add(st / "package_manifest.json", arcname="package_manifest.json", recursive=False)
manifest.update(archive_sha256=o.sha(archive), archive_bytes=archive.stat().st_size)
o.atomic(P / "package_release.json", manifest)
shutil.rmtree(st)
ident = "v8-final-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
root = "/workspace/results/" + ident
data = "/workspace/v8data"  # on the 160 GB network volume, outside the collected results root
code = {r["path"]: r["sha256"] for r in rows if r["path"].endswith(".py") and r["path"].startswith("repo/")}
for name in m.HOST_FILES:
    code["host/" + name] = o.sha(ROOT / name)
bindings = {"code": hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
            "protocol": hashlib.sha256((o.sha(CONFIG) + o.sha(ROOT / "docs/RAID_V8_PLAN.md")).encode()).hexdigest(),
            "runtime": hashlib.sha256((o.IMAGE_REFERENCE + o.sha(ROOT / "requirements-wm-cosmos.lock")
                                       + o.sha(ROOT / "requirements-wm-sim.lock")).encode()).hexdigest()}
py = runtime_python(root)
env = (runtime_exports(root) + f"export PYTHONPATH={root}/repo:$PYTHONPATH\nexport HF_HOME=/root/raid-model-cache\n"
       "export HF_HUB_ENABLE_HF_TRANSFER=0\nexport OMP_NUM_THREADS=16\nexport MKL_NUM_THREADS=16\n" + f"cd {root}/repo\n")
out = f"{root}/collection/v8_final"
bench = (f"{py} -m raid_experiments.v8.bench --stage final --config {root}/repo/amendment.json --sources {root}/repo/sources "
         f"--data {data}/libero --cache {data}/cache --out {out} --libero-root {root}/inputs/libero_repo --root {root}/repo")
stages = [{"name": "archive_verify", "command": m.archive_verification_command(root, manifest["archive_sha256"]), "resume_safe": True},
          # data (about 73 GB) and feature caches live on the volume: require unpacked package + 110 GB
          {"name": "package_preflight", "command": pilot.preflight_command(
              root, manifest["archive_sha256"], manifest["unpacked_bytes"] - pilot.WEIGHTS_FREE_BYTES + 45 * 1024 ** 3)},
          {"name": "bootstrap", "command": bootstrap_script(root), "resume_safe": True},
          {"name": "v8_download_cache", "command": env + bench + " --steps download,cache", "resume_safe": True},
          {"name": "v8_offline", "command": env + bench + " --steps offline", "resume_safe": True},
          {"name": "v8_encoder", "command": env + bench + " --steps encoder", "resume_safe": True},
          {"name": "v8_closed_loop", "command": env + bench + " --steps closed_loop", "resume_safe": True},
          {"name": "v8_timing", "command": env + bench + " --steps timing", "resume_safe": True},
          {"name": "v8_analysis", "command": env + f"{py} -m raid_experiments.v8.analysis final --final {out} "
                                                  f"--amendment {root}/repo/amendment.json --write {out}/final_results.json",
           "resume_safe": True}]
identity = {"session_id": ident, "stage": "raid_v8_final", "amendment_digest": amend["digest"], "package_sha256": manifest["archive_sha256"],
            "bindings": bindings, "config_sha256": o.sha(CONFIG), "scientific_result": False,
            "held_out_claims_eligible": False,
            "stage_commands": {s["name"]: hashlib.sha256(s["command"].encode()).hexdigest() for s in stages}}
o.atomic(P / "release_identity.json", identity); o.atomic(P / "code_binding_files.json", code)
release = {"session_id": ident, "root": root, "archive": str(archive), "archive_sha256": manifest["archive_sha256"],
           "bindings": bindings, "identity_sha256": o.sha(P / "release_identity.json"), "stages": stages}
o.atomic(P / "release.json", release)
print(json.dumps({"session_id": ident, "files": len(rows), "archive_gb": round(archive.stat().st_size / 1e9, 3)}))
