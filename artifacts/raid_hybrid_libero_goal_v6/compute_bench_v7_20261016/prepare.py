"""Freeze the offline v7 compute benchmark package (no VPP2, no simulator; source + development data only).
Run with system python3 (no network): python3 prepare.py"""
from pathlib import Path
import sys, json, shutil, hashlib, tarfile, uuid
from datetime import datetime, timezone
P = Path(__file__).resolve().parent; ROOT = P.parents[2]; sys.path.insert(0, str(ROOT))
from raid_experiments import operational_acceptance as o, managed_source as m, hybrid_vpp2_pilot as pilot
from raid_experiments.wm_pod import bootstrap_script, runtime_exports, runtime_python
PROTOCOL = P.parent / "vpp2_pilot_v5_20261014/admission/protocol.json"
DATA = ROOT / "data/assets/libero/libero_goal"
if (P / "release.json").exists():
    raise RuntimeError("immutable release exists")
protocol = json.loads(PROTOCOL.read_text())
amendment = json.loads((P / "bench_amendment_v7.json").read_text())
assert amendment["parent_protocol_digest"] == protocol["digest"], "amendment belongs to another protocol"
st = P / "bundle"
if st.exists():
    shutil.rmtree(st)
files = {"repo/" + rel: ROOT / rel for rel in pilot.CODE_FILES}
files["repo/raid_experiments/hybrid_compute_bench.py"] = ROOT / "raid_experiments/hybrid_compute_bench.py"
files.update({"repo/" + n: ROOT / n for n in ("requirements-wm-cosmos.lock", "requirements-wm-sim.lock")})
files["repo/protocol.json"] = PROTOCOL
files["repo/bench_amendment_v7.json"] = P / "bench_amendment_v7.json"
for name, spec in protocol["tasks"].items():
    if spec["role"] in ("source", "development"):  # final tasks never enter the package
        local = DATA / Path(spec["path"]).name
        assert o.sha(local) == spec["source_sha256"], f"dataset differs from protocol: {local.name}"
        files["inputs/dataset/" + local.name] = local
for remote, local in files.items():
    assert local.is_file() and not local.is_symlink(), f"missing input {local}"
    assert not pilot.PROHIBITED_NAME.search(Path(remote).name), "prohibited model file in package"
    target = st / remote; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(local, target)
rows = [{"path": p.relative_to(st).as_posix(), "sha256": o.sha(p), "size_bytes": p.stat().st_size}
        for p in sorted(st.rglob("*")) if p.is_file()]
manifest = {"files": rows, "stage": "v7_compute_benchmark", "file_count": len(rows),
            "unpacked_bytes": sum(x["size_bytes"] for x in rows)}
o.atomic(st / "package_manifest.json", manifest)
archive = P / "bench_package.tar"
with tarfile.open(archive, "w") as t:
    for row in rows:
        t.add(st / row["path"], arcname=row["path"], recursive=False)
    t.add(st / "package_manifest.json", arcname="package_manifest.json", recursive=False)
manifest.update(archive_sha256=o.sha(archive), archive_bytes=archive.stat().st_size)
o.atomic(P / "package_release.json", manifest)
shutil.rmtree(st)  # the archive is the frozen copy; keep disk free
ident = "v6-source-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
root = "/workspace/results/" + ident
code = {r["path"]: r["sha256"] for r in rows if r["path"].endswith(".py")}
for name in m.HOST_FILES:
    code["host/" + name] = o.sha(ROOT / name)
bindings = {"code": hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
            "protocol": hashlib.sha256((o.sha(PROTOCOL) + o.sha(P / "bench_amendment_v7.json")).encode()).hexdigest(),
            "runtime": hashlib.sha256((o.IMAGE_REFERENCE + o.sha(ROOT / "requirements-wm-cosmos.lock")
                                       + o.sha(ROOT / "requirements-wm-sim.lock")).encode()).hexdigest()}
py = runtime_python(root)
raid = runtime_exports(root) + f"export PYTHONPATH={root}/repo:$PYTHONPATH\nexport HF_HOME=/root/raid-model-cache\ncd {root}/repo\n"
stages = [{"name": "archive_verify", "command": m.archive_verification_command(root, manifest["archive_sha256"]), "resume_safe": True},
          # No VPP2 weights here: require the unpacked package plus 20 GB (cache, checkpoints), not the pilot's 95 GB.
          {"name": "package_preflight", "command": pilot.preflight_command(
              root, manifest["archive_sha256"], manifest["unpacked_bytes"] - pilot.WEIGHTS_FREE_BYTES + 20 * 1024 ** 3)},
          {"name": "bootstrap", "command": bootstrap_script(root), "resume_safe": True},
          {"name": "mae_download", "command": raid + f"{py} -m raid_experiments.hybrid_vpp2_panel download-mae {root}", "resume_safe": True},
          {"name": "resolve_paths", "command": raid + f"{py} -m raid_experiments.hybrid_cli --protocol {root}/repo/protocol.json "
                                                      f"resolve-paths --data-root {root}/inputs/dataset --stage pilot "
                                                      f"--output {root}/collection/protocol_bench_pod.json", "resume_safe": True},
          {"name": "bench_run", "command": raid + f"{py} -m raid_experiments.hybrid_compute_bench run {root}", "resume_safe": True},
          {"name": "bench_report", "command": raid + f"{py} -m raid_experiments.hybrid_compute_bench report {root}", "resume_safe": True}]
identity = {"session_id": ident, "stage": "v7_compute_benchmark", "package_sha256": manifest["archive_sha256"],
            "bindings": bindings, "protocol_digest": protocol["digest"], "amendment_digest": amendment["digest"],
            "expected_units": 6, "scientific_result": False, "held_out_claims_eligible": False,
            "stage_commands": {s["name"]: hashlib.sha256(s["command"].encode()).hexdigest() for s in stages}}
o.atomic(P / "release_identity.json", identity); o.atomic(P / "code_binding_files.json", code)
release = {"session_id": ident, "root": root, "archive": str(archive), "archive_sha256": manifest["archive_sha256"],
           "bindings": bindings, "identity_sha256": o.sha(P / "release_identity.json"), "stages": stages}
o.atomic(P / "release.json", release)
print(json.dumps({"session_id": ident, "files": len(rows), "archive_gb": archive.stat().st_size / 1e9}))
