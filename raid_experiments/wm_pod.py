"""Pod plan for the protocol v4 GPU validation run.

``launch_validation`` rents nothing until a provider is passed and
``authorize_spend`` is true. The default command is validation only: it does
not start source training or open development queries. The rendered scripts
never contain an API key; the watchdog reads ``/root/.raid/runpod_key``.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shlex
import subprocess
import tarfile
import uuid
from decimal import Decimal, InvalidOperation

from .wm_protocol import COSMOS_MODEL_ID, COSMOS_REVISION, assert_no_prohibited_model
from .wm_runtime import PINNED_IMAGE_DIGEST

IMAGE_REFERENCE = "runpod/pytorch@" + PINNED_IMAGE_DIGEST
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
UPLOADS = {
    "requirements-wm-cosmos.lock": "repo/requirements-wm-cosmos.lock",
    "requirements-wm-sim.lock": "repo/requirements-wm-sim.lock",
    "raid_experiments": "repo/raid_experiments",
    "src": "repo/src",
    "configs/cosmos3_edge_asset_notes.json": "repo/configs/cosmos3_edge_asset_notes.json",
    "data/assets/libero/libero_goal": "inputs/dataset",
    "data/assets/libero/upstream": "inputs/libero_repo",
    "data/assets/mae/mae_pretrain_vit_base.pth": "inputs/files/mae_checkpoint",
}
VALIDATION_CAP_USD = 8.0
VALIDATION_MAX_HOURS = 1.5
GPU = "NVIDIA A100 80GB PCIe"
VALIDATION_GPU = "NVIDIA A40"
DATA_CENTER_ID = "EU-RO-1"
GPU_SECURE_PRICE_PER_HOUR_USD = Decimal("1.79")


def runtime_exports(study_root: str) -> str:
    """Environment for every remote command. LIBERO's setup metadata does not install the nested package."""
    return (
        f"export PYTHONPATH={shlex.quote(study_root + '/inputs/libero_repo')}:${{PYTHONPATH:-}}\n"
        f"export LIBERO_CONFIG_PATH={shlex.quote(study_root + '/libero-config')}\n"
        "export MUJOCO_GL=egl\n"
    )


def egl_apt_script() -> str:
    """Install headless EGL libraries on Ubuntu images before any MuJoCo render."""
    return """if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libglvnd0 libglib2.0-0 rsync
fi
"""


def egl_check_script(python: str) -> str:
    return f"""{python} - <<'PY'
import mujoco
ctx = mujoco.GLContext(64, 64)
ctx.free()
print('egl_ok')
PY
"""


def _collection_finalize_script(root: str) -> str:
    """Hash stable outputs after stages close; never hash the still-open worker log."""
    return f"""{root}/venv/bin/python - {root}/collection >{root}/collection/logs/finalize.log 2>&1 <<'PY'
import hashlib, json, pathlib, sys
r = pathlib.Path(sys.argv[1])
d = {{}}
for path in sorted(r.rglob('*')):
    if not path.is_file() or path.name in {{'outputs_sha256.json', 'outputs_sha256.json.tmp', 'DONE', 'FAILED', 'worker.log'}} or path.as_posix().endswith('logs/finalize.exit'):
        continue
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    d[path.relative_to(r).as_posix()] = digest.hexdigest()
tmp = r / 'outputs_sha256.json.tmp'
tmp.write_text(json.dumps(d, sort_keys=True, indent=2) + '\\n')
tmp.replace(r / 'outputs_sha256.json')
PY
code=$?; printf '%s\\n' "$code" >{root}/collection/logs/finalize.exit
if [ "$code" -ne 0 ]; then printf 'finalize failed\\n' >{root}/collection/FAILED.tmp; mv {root}/collection/FAILED.tmp {root}/collection/FAILED; exit "$code"; fi
"""


def runtime_python(study_root: str) -> str:
    """Keep the complete environment on local disk, including wheel scripts.

    Relocating only ``lib`` breaks wheel paths such as ../../../bin/jp.py.
    Results stay on recoverable storage; the disposable runtime does not.
    """
    suffix = hashlib.sha256(study_root.encode()).hexdigest()[:32]
    return '/root/raid-runtime-' + suffix + '/bin/python'


def bootstrap_script(study_root: str) -> str:
    root = shlex.quote(study_root)
    runtime = shlex.quote(str(Path(runtime_python(study_root)).parent.parent))
    return f"""set -eu
python3 --version | grep -q 'Python 3.12'
{egl_apt_script()}
python3 -c 'import shutil; assert shutil.disk_usage("/root").free >= 45*1024**3, "insufficient local dependency/model-cache space"'
test ! -L {runtime}
python3 -m venv {runtime}
PY={runtime}/bin/python
"$PY" -c 'import sys; assert sys.prefix.startswith("/root/raid-runtime-")'
# Compatibility for legacy stage renderers; installation always uses local PY.
if test -e {root}/venv || test -L {root}/venv; then
  test -L {root}/venv && test "$(readlink {root}/venv)" = {runtime}
else
  ln -s {runtime} {root}/venv
fi
"$PY" -m pip install --no-cache-dir --force-reinstall --require-hashes -r {root}/repo/requirements-wm-cosmos.lock \\
  --extra-index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --no-cache-dir --force-reinstall --require-hashes -r {root}/repo/requirements-wm-sim.lock
"$PY" -m pip check
"$PY" -c 'import shutil; assert shutil.disk_usage("/root").free >= 30*1024**3, "insufficient local model-cache space"'
test "$("$PY" -c 'import torch; print(torch.__version__)')" = "2.9.1+cu128"
{runtime_exports(study_root)}
{egl_check_script('"$PY"')}
"""


def validation_command(study_root: str, task: str) -> str:
    root = shlex.quote(study_root)
    return (
        runtime_exports(study_root)
        + f"cd {root}/repo && {root}/venv/bin/python -m raid_experiments.wm_cli validate-gpu "
        f"--task {shlex.quote(task)} --libero-root {root}/inputs/libero_repo "
        f"--output {root}/collection --load-model --hf-revision {shlex.quote(COSMOS_REVISION)}"
    )


def validation_plan(study_root: str = "/workspace/raid-wm/validation",
                    task: str = "put_the_cream_cheese_in_the_bowl") -> dict:
    assert_no_prohibited_model(COSMOS_MODEL_ID)
    script = bootstrap_script(study_root) + "\n" + validation_command(study_root, task) + "\n"
    if "rpa_" in script or "gr-1" in script.lower() or "gr1" in script.lower():
        raise RuntimeError("validation plan contains a prohibited token")
    return {
        "image": IMAGE_REFERENCE,
        "image_digest": PINNED_IMAGE_DIGEST,
        "gpu": VALIDATION_GPU,
        "cap_usd": VALIDATION_CAP_USD,
        "max_hours": VALIDATION_MAX_HOURS,
        "study_root": study_root,
        "task": task,
        "libero_commit": LIBERO_COMMIT,
        "cosmos_revision": COSMOS_REVISION,
        "uploads": {rel: f"{study_root}/{dest}" for rel, dest in UPLOADS.items()},
        "bootstrap": bootstrap_script(study_root),
        "command": validation_command(study_root, task),
        "watchdogs": ["deadline", "exit_guard"],
        "pull": ["collection/cuda_evidence.json", "collection/costs.json", "collection/validation_clips",
                 "DONE", "FAILED"],
        "starts_training": False,
        "scientific_result": False,
    }


def pull_plan(study_root: str, local: str | Path) -> dict:
    local = Path(local)
    return {"local": str(local), "sources": [f"{study_root}/{rel}" for rel in validation_plan(study_root)["pull"]],
            "excludes": ["/root/.raid", "venv", ".ssh"]}


SOURCE_PANEL_CAP_USD = 10.0
SOURCE_PANEL_MAX_HOURS = 2.0

# Source feasibility needs only this v6 scientific path. Keeping module names
# here makes the archive allowlist reviewable and keeps legacy/final-workflow
# code out of the paid bundle.
SOURCE_PANEL_PYTHON_ALLOWLIST = {
    "__init__.py", "hybrid_admission.py", "hybrid_live_acceptance.py", "hybrid_analysis.py", "hybrid_cli.py",
    "hybrid_collect.py", "hybrid_feasibility.py", "hybrid_generator_profile.py", "hybrid_heads.py",
    "hybrid_local_checks.py", "hybrid_panel.py", "hybrid_protocol.py",
    "hybrid_retrieval.py", "hybrid_rollout.py", "hybrid_runner.py", "hybrid_study.py",
    "predictors.py", "simulator.py", "wm_cli.py", "wm_collect.py", "wm_experiment.py",
    "wm_heads.py", "wm_protocol.py", "wm_rollout.py", "wm_study.py", "wm_validate.py",
    "world_model.py", "models.py", "retrieval.py", "wm_feasibility.py", "source_worker.py", "hybrid_measure.py", "wm_runtime.py",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_panel_command(study_root: str) -> str:
    """Stage-1 worker: resolve staged inputs, run checks, and publish atomic DONE/FAILED."""
    root = shlex.quote(study_root)
    protocol = f"{root}/repo/artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json"
    mae = f"{root}/inputs/files/mae_checkpoint"
    worker = (
        runtime_exports(study_root)
        + f"cd {root}/repo\n"
        "set -u -o pipefail\n"
        f"mkdir -p {root}/collection/logs\n"
        "run_stage() { name=$1; shift; if \"$@\" >\"$STUDY_ROOT/collection/logs/$name.log\" 2>&1; then code=0; else code=$?; fi; "
        "printf '%s\\n' \"$code\" >\"$STUDY_ROOT/collection/logs/$name.exit\"; "
        "if [ \"$code\" -ne 0 ]; then printf '%s\\n' \"$name failed with exit $code\" >\"$STUDY_ROOT/collection/FAILED.tmp\"; "
        "mv \"$STUDY_ROOT/collection/FAILED.tmp\" \"$STUDY_ROOT/collection/FAILED\"; exit \"$code\"; fi; }\n"
        f"export STUDY_ROOT={root}\n"
        f"run_stage resolve {root}/venv/bin/python -m raid_experiments.hybrid_cli --protocol "
        f"{root}/repo/artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json resolve-paths "
        f"--data-root {root}/inputs/dataset --stage source-panel "
        f"--output {root}/collection/protocol_v6_pod.json\n"
        f"run_stage gpu_validate {root}/venv/bin/python -m raid_experiments.wm_cli validate-gpu "
        f"--task put_the_cream_cheese_in_the_bowl --libero-root {root}/inputs/libero_repo "
        f"--protocol {root}/collection/protocol_v6_pod.json "
        f"--output {root}/collection --load-model --hf-revision {shlex.quote(COSMOS_REVISION)}\n"
        f"run_stage expert_replays {root}/venv/bin/python -m raid_experiments.hybrid_cli "
        f"--protocol {root}/collection/protocol_v6_pod.json replay-experts --cases 30 "
        f"--libero-root {root}/inputs/libero_repo "
        f"--output {root}/collection/source_panel/expert_replays.json\n"
        f"run_stage source_panel {root}/venv/bin/python -m raid_experiments.hybrid_cli "
        f"--protocol {root}/collection/protocol_v6_pod.json source-panel --load-model "
        f"--mae {mae} --device cuda --output {root}/collection "
        f"--expert-replays {root}/collection/source_panel/expert_replays.json\n"
        + _collection_finalize_script(root)
        + f"printf 'complete\\n' > {root}/collection/DONE.tmp && mv {root}/collection/DONE.tmp {root}/collection/DONE\n"
    )
    # Detach the worker so a lost SSH connection or local app restart does not
    # silently terminate the experiment. The session journal owns runner.pid.
    return ("set -eu\n"
            f"mkdir -p {root}/collection\n"
            f"rm -f {root}/collection/DONE {root}/collection/FAILED\n"
            f"nohup bash -c {shlex.quote(worker)} >{root}/collection/worker.log 2>&1 </dev/null &\n"
            f"echo $! > {root}/runner.pid\n"
            f"test -s {root}/runner.pid\n")


def source_panel_plan(study_root: str = "/workspace/raid-wm/source_panel", *,
                      data_center_id: str = DATA_CENTER_ID,
                      gpu_hourly_price_usd: Decimal = GPU_SECURE_PRICE_PER_HOUR_USD,
                      gpu_name: str = GPU) -> dict:
    """Render the stage-1 source-only feasibility package. Does not launch."""
    assert_no_prohibited_model(COSMOS_MODEL_ID)
    uploads = dict(UPLOADS)
    uploads["configs/raid_hybrid_libero_goal_v6.yaml"] = "repo/configs/raid_hybrid_libero_goal_v6.yaml"
    uploads["artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json"] = (
        "repo/artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json")
    script = bootstrap_script(study_root) + "\n" + source_panel_command(study_root) + "\n"
    if "rpa_" in script or "gr-1" in script.lower() or "gr1" in script.lower():
        raise RuntimeError("source panel plan contains a prohibited token")
    return {
        "mode": "source_panel",
        "image": IMAGE_REFERENCE,
        "image_digest": PINNED_IMAGE_DIGEST,
        "gpu": gpu_name,
        "data_center_id": data_center_id,
        "gpu_hourly_price_usd": str(gpu_hourly_price_usd),
        "cap_usd": SOURCE_PANEL_CAP_USD,
        "max_hours": SOURCE_PANEL_MAX_HOURS,
        "study_root": study_root,
        "libero_commit": LIBERO_COMMIT,
        "cosmos_revision": COSMOS_REVISION,
        "uploads": {rel: f"{study_root}/{dest}" for rel, dest in uploads.items()},
        "bootstrap": bootstrap_script(study_root),
        "command": source_panel_command(study_root),
        "watchdogs": ["deadline", "exit_guard"],
        "pull": [
            "collection/cuda_evidence.json", "collection/costs.json",
            "collection/validation_clips", "collection/source_panel",
            "collection/outputs_sha256.json", "collection/DONE", "collection/FAILED",
        ],
        "starts_training": False,
        "panel_cases": 30,
        "rubric_required": True,
        "scientific_result": False,
        "note": "Stage-1 source-only feasibility. Launch requires separate --authorize-spend.",
    }


def _source_panel_package(session: Path) -> tuple[Path, dict]:
    """Build a content-addressed tar containing only the declared upload inputs."""
    repo = Path(__file__).resolve().parents[1]
    git = Path("/Users/ConstiX/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git")
    git_cmd = str(git) if git.is_file() else "git"
    revision = subprocess.run([git_cmd, "-C", str(repo / "data/assets/libero/upstream"),
                               "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    dirty = subprocess.run([git_cmd, "-C", str(repo / "data/assets/libero/upstream"),
                            "status", "--porcelain"], check=True, capture_output=True,
                           text=True, timeout=30).stdout.strip()
    if revision != LIBERO_COMMIT or dirty:
        raise ValueError("source-panel package requires the clean pinned LIBERO revision")
    uploads = dict(UPLOADS)
    uploads.update({
        "configs/raid_hybrid_libero_goal_v6.yaml": "repo/configs/raid_hybrid_libero_goal_v6.yaml",
        "artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json":
            "repo/artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json",
    })
    files: dict[str, Path] = {}
    protocol_path = repo / "artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json"
    protocol = json.loads(protocol_path.read_text())
    source_inputs = {Path(spec["path"]).resolve(): spec["source_sha256"]
                     for spec in protocol["tasks"].values() if spec.get("role") == "source"}
    excluded_parts = {".git", ".ssh", "__pycache__", "node_modules", ".venv", "venv"}
    excluded_markers = ("credential", "secret", "token", "api_key", "apikey")
    for local_rel, remote_root in uploads.items():
        source = repo / local_rel
        if not source.exists():
            raise FileNotFoundError(f"required source-panel package input missing: {source}")
        candidates = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
        if local_rel == "raid_experiments":
            candidates = [path for path in candidates
                          if path.relative_to(source).as_posix() in SOURCE_PANEL_PYTHON_ALLOWLIST]
        elif local_rel == "src":
            candidates = [path for path in candidates
                          if path.relative_to(source).as_posix() == "mae_encoder.py"]
        elif local_rel == "data/assets/libero/upstream":
            # The clean, pinned upstream commit defines the exact source allowlist;
            # record every tracked source member from that commit in the manifest.
            tracked = set(subprocess.run(
                [git_cmd, "-C", str(source), "ls-files"], check=True,
                capture_output=True, text=True, timeout=30).stdout.splitlines())
            candidates = [path for path in candidates
                          if path.relative_to(source).as_posix() in tracked]
            source_names={name for name,spec in protocol['tasks'].items() if spec['role']=='source'}
            candidates=[path for path in candidates if not ({'bddl_files','init_files'} & set(path.relative_to(source).parts))
                        or ('libero_goal' in path.relative_to(source).parts and any(path.name.startswith(name+'.') for name in source_names))]
        if local_rel == "data/assets/libero/libero_goal":
            candidates = [path for path in candidates if path.resolve() in source_inputs]
        for path in candidates:
            relative = path.relative_to(source) if source.is_dir() else Path()
            parts = path.relative_to(repo).parts
            if (path.is_symlink() or any(part in excluded_parts or part.startswith(".") for part in parts)
                    or any(marker in path.name.lower() for marker in excluded_markers)
                    or path.suffix in {".pyc", ".pyo"}):
                continue
            if path.resolve() in source_inputs and _sha256_file(path) != source_inputs[path.resolve()]:
                raise ValueError(f"source HDF5 does not match protocol digest: {path.name}")
            remote = (Path(remote_root) / relative).as_posix()
            files[remote] = path
    if not files:
        raise RuntimeError("source-panel package allowlist is empty")
    manifest = {
        "format": 1,
        "libero_commit": LIBERO_COMMIT,
        "files": [{"path": remote, "sha256": _sha256_file(path),
                   "size_bytes": path.stat().st_size}
                  for remote, path in sorted(files.items())],
        "file_count": len(files),
        "unpacked_bytes": sum(path.stat().st_size for path in files.values()),
        "excluded": ["credentials", "hidden/local configuration", "Git metadata", "bytecode", "unlisted artifacts"],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
    manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n"
    session.mkdir(parents=True, exist_ok=True)
    archive = session / "source_panel_package.tar"
    with tarfile.open(archive, "w") as tar:
        for remote, path in sorted(files.items()):
            tar.add(path, arcname=remote, recursive=False)
        import io
        tarinfo = tarfile.TarInfo("package_manifest.json")
        tarinfo.size = len(manifest_bytes)
        tar.addfile(tarinfo, io.BytesIO(manifest_bytes))
    manifest["archive_sha256"] = _sha256_file(archive)
    manifest["archive_bytes"] = archive.stat().st_size
    return archive, manifest


def source_panel_package_preflight(study_root: str, expected_archive_sha256: str, *, model_cache_reserve_bytes: int = 30 * 1024**3) -> str:
    """Extract and hash-check the explicit package before bootstrap or model loading."""
    root = shlex.quote(study_root)
    return f'''set -eu
python3 - <<'PY'
import hashlib, json, os, pathlib, shutil, tarfile
root = pathlib.Path({study_root!r})
archive = root / "source_panel_package.tar"
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
if sha(archive) != {expected_archive_sha256!r}:
    raise SystemExit("uploaded package archive digest differs from launch journal")
with tarfile.open(archive, "r:") as tar:
    members = tar.getmembers()
    if any(not m.isfile() or pathlib.PurePosixPath(m.name).is_absolute() or ".." in pathlib.PurePosixPath(m.name).parts for m in members):
        raise SystemExit("unsafe package member")
    if len({{m.name for m in members}}) != len(members):
        raise SystemExit("duplicate package member")
    manifest_member = tar.extractfile("package_manifest.json")
    if manifest_member is None:
        raise SystemExit("package manifest missing")
    manifest = json.loads(manifest_member.read())
manifest_body = dict(manifest)
manifest_digest = manifest_body.pop("manifest_sha256", None)
manifest_bytes = json.dumps(manifest_body, sort_keys=True, indent=2).encode() + b"\\n"
if hashlib.sha256(manifest_bytes).hexdigest() != manifest_digest:
    raise SystemExit("package manifest digest mismatch")
expected_names = {{row["path"] for row in manifest.get("files", [])}} | {{"package_manifest.json"}}
if {{m.name for m in members}} != expected_names:
    raise SystemExit("archive membership differs from the signed file allowlist")
if sum(m.size for m in members) != manifest.get("unpacked_bytes", 0) + next(m.size for m in members if m.name == "package_manifest.json"):
    raise SystemExit("archive declared size is inconsistent")
declared = {{m.name: m.size for m in members}}
for row in manifest.get("files", []):
    if declared.get(row.get("path")) != row.get("size_bytes"):
        raise SystemExit("archive member size differs from allowlist: " + str(row.get("path")))
free = shutil.disk_usage(root).free
required = int(manifest["unpacked_bytes"] * 1.20) + {model_cache_reserve_bytes}
if free < required:
    raise SystemExit(f"insufficient package disk reserve: free={{free}} required={{required}}")
with tarfile.open(archive, "r:") as tar:
    tar.extractall(root, filter="data")
for row in manifest["files"]:
    path = root / row["path"]
    if not path.is_file() or path.stat().st_size != row["size_bytes"]:
        raise SystemExit("missing or truncated package input: " + row["path"])
    if sha(path) != row["sha256"]:
        raise SystemExit("package checksum mismatch: " + row["path"])
required_paths = ["repo/requirements-wm-cosmos.lock", "repo/requirements-wm-sim.lock",
 "repo/configs/raid_hybrid_libero_goal_v6.yaml",
 "repo/artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json",
 "inputs/files/mae_checkpoint"]
for name in required_paths:
    if not (root / name).is_file():
        raise SystemExit("required file missing: " + name)
if not (root / "inputs/dataset").is_dir() or not any((root / "inputs/dataset").rglob("*.hdf5")):
    raise SystemExit("staged HDF5 dataset missing")
if not (root / "inputs/libero_repo/libero").is_dir():
    raise SystemExit("pinned LIBERO source missing")
if manifest.get("libero_commit") != {LIBERO_COMMIT!r}:
    raise SystemExit("pinned LIBERO source revision mismatch")
if tuple(__import__("sys").version_info[:2]) != (3, 12):
    raise SystemExit("Python 3.12 is required")
os.environ["PYTHONPATH"] = str(root / "repo")
import sys
sys.path.insert(0, str(root / "repo"))
from raid_experiments.hybrid_protocol import validate_hybrid_protocol
validate_hybrid_protocol(json.loads((root / required_paths[3]).read_text()))
archive.unlink()
print(json.dumps({{"status":"package_verified", "files":manifest["file_count"], "manifest_sha256":manifest["manifest_sha256"]}}))
PY
'''


def launch_source_panel(session: Path, provider, *, authorize_spend: bool = False,
                        cap_usd: float = SOURCE_PANEL_CAP_USD,
                        max_hours: float = SOURCE_PANEL_MAX_HOURS) -> dict:
    """Create one source-panel pod. Refuses unless spending was explicitly authorized."""
    if not authorize_spend:
        raise PermissionError("source-panel launch refused: spending is not authorized")
    try:
        cap = Decimal(str(cap_usd))
        hours = Decimal(str(max_hours))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("source-panel cap and runtime must be finite positive decimals") from exc
    if not cap.is_finite() or not hours.is_finite() or cap <= 0 or hours <= 0:
        raise ValueError("source-panel cap and runtime must be finite positive decimals")
    if cap > Decimal(str(SOURCE_PANEL_CAP_USD)) or hours > Decimal(str(SOURCE_PANEL_MAX_HOURS)):
        raise ValueError("source-panel launch exceeds the 2 hour / $10 ceiling")
    if hasattr(provider, "assert_launch_ready"):
        provider.assert_launch_ready()
    session = Path(session)
    session.mkdir(parents=True, exist_ok=True)
    session_path = session / "session.json"
    if session_path.exists():
        raise RuntimeError("source-panel session already exists; use status/resume/teardown instead of launching a duplicate")
    admission = None
    if hasattr(provider, "source_launch_admission"):
        admission = provider.source_launch_admission(
            cap_usd=float(cap), max_hours=float(hours), network_volume_size_gb=80)
        plan = source_panel_plan(
            data_center_id=admission["data_center_id"],
            gpu_hourly_price_usd=Decimal(admission["gpu_hourly_usd"]),
            gpu_name=admission["gpu"])
    else:
        plan = source_panel_plan()
    package = None
    package_manifest = None
    if hasattr(provider, "upload_package"):
        package, package_manifest = _source_panel_package(session)
    session_id = uuid.uuid4().hex
    intent = {"mode": "source_panel", "status": "launch_intent",
              "cutoff_session_id": session_id,
              "launch_request_at": __import__("time").time(),
              "local_collection": str(session / "collection"),
              "plan": {k: plan[k] for k in ("image", "cap_usd", "max_hours", "starts_training", "panel_cases")},
              "package_manifest": package_manifest, "scientific_result": False}
    _write_session(session, intent)
    def journal_owner(created: dict) -> None:
        intent.update(status="created", pod_id=str(created["id"]),
                      volume_id=created.get("volume_id"),
                      data_center_id=created.get("data_center_id"),
                      cutoff_session_id=created.get("cutoff_session_id"),
                      ownership_recorded_at=__import__("time").time())
        _write_session(session, intent)
    def journal_admission(snapshot: dict) -> None:
        intent["spend_admission"] = snapshot
        _write_session(session, intent)
    try:
        import inspect
        create_args = {"name": "raid-wm-source-panel", "image": plan["image"], "gpu": plan["gpu"],
                       "max_hours": float(hours), "stage": "source_panel",
                       "session_id": session_id,
                       "package_sha256": package_manifest["archive_sha256"] if package_manifest else "",
                       "cap_usd": float(cap),
                       "max_hourly_usd": float(Decimal(plan["gpu_hourly_price_usd"])),
                       "data_center_id": plan["data_center_id"],
                       "network_volume_size_gb": 80,
                       "collection_reserve_seconds": 600,
                       "storage_retention_seconds": 0,
                       "env": {"RAID_CONTAINER_IMAGE_DIGEST": plan["image_digest"],
                               "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}}
        if "on_created" in inspect.signature(provider.create_pod).parameters:
            create_args["on_created"] = journal_owner
        if "on_admission" in inspect.signature(provider.create_pod).parameters:
            create_args["on_admission"] = journal_admission
        pod = provider.create_pod(**create_args)
    except Exception as exc:
        intent.update(status="create_failed", launch_error=f"{type(exc).__name__}: {exc}")
        _write_session(session, intent)
        raise
    pod_id = str(pod["id"])
    journal_owner(pod)
    record = {"pod_id": pod_id, "mode": "source_panel", "status": "created",
              "volume_id": pod.get("volume_id"), "data_center_id": pod.get("data_center_id"),
              "cutoff_session_id": session_id,
              "study_root": plan["study_root"], "local_collection": str(session / "collection"),
              "started_at": __import__("time").time(),
              "termination_deadline": pod.get("raid_termination_deadline"),
              "termination_verified": pod.get("raid_termination_verified") is True,
              "launch_request_at": intent["launch_request_at"],
              "package_manifest": package_manifest,
              "plan": {k: plan[k] for k in ("image", "cap_usd", "max_hours", "starts_training", "panel_cases")},
              "scientific_result": False}
    _write_session(session, record)
    if hasattr(provider, "wait_ready"):
        try:
            provider.wait_ready(pod_id)
        except Exception as exc:
            record["status"] = "created_ssh_unavailable"
            record["launch_error"] = f"{type(exc).__name__}: {exc}"
            _write_session(session, record)
            raise
    # Provider credentials stay in the independent cutoff service. The DO alarm
    # owns termination; pod-side Runpod API keys/watchdogs are not installed.
    if package is not None and hasattr(provider, "upload_package"):
        provider.upload_package(pod_id, package, f"{plan['study_root']}/source_panel_package.tar")
        provider.remote(pod_id, source_panel_package_preflight(
            plan["study_root"], package_manifest["archive_sha256"]))
    else:
        provider.upload(pod_id, plan["uploads"], plan["study_root"])
    provider.remote(pod_id, plan["bootstrap"])
    provider.remote(pod_id, plan["command"])
    record["status"] = "running"
    record["worker_pid_file"] = f"{plan['study_root']}/runner.pid"
    _write_session(session, record)
    return record


def _write_session(session: Path, record: dict) -> None:
    session.mkdir(parents=True, exist_ok=True)
    path = session / "session.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _load_session(session: Path) -> dict:
    path = Path(session) / "session.json"
    if not path.is_file():
        raise FileNotFoundError(f"source-panel session missing: {path}")
    return json.loads(path.read_text())


def verify_source_panel_collection(collection: Path) -> dict:
    """Verify complete committed files against the worker's SHA-256 manifest."""
    manifest_path = collection / "outputs_sha256.json"
    if not manifest_path.is_file() or not (collection / "DONE").is_file():
        raise RuntimeError("completed source-panel marker or output manifest is missing")
    expected = json.loads(manifest_path.read_text())
    if not expected:
        raise RuntimeError("source-panel output manifest is empty")
    observed = {}
    ignored = {"outputs_sha256.json", "DONE", "FAILED", "worker.log"}
    actual_names = {path.relative_to(collection).as_posix() for path in collection.rglob("*")
                    if path.is_file() and path.name not in ignored
                    and path.relative_to(collection).as_posix() != "logs/finalize.exit"}
    if actual_names != set(expected):
        raise RuntimeError("collection contains missing or uncommitted output files")
    for rel, sha in expected.items():
        path = collection / rel
        if not path.is_file():
            raise RuntimeError(f"collected source-panel file is missing: {rel}")
        observed[rel] = _sha256_file(path)
        if observed[rel] != sha:
            raise RuntimeError(f"source-panel collection checksum mismatch: {rel}")
    return {"status": "verified", "files": len(observed),
            "manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()}


def source_panel_status(session: Path, provider) -> dict:
    record = _load_session(session)
    status = provider.status(record.get("pod_id"), record["study_root"],
                             cutoff_session_id=record.get("cutoff_session_id"))
    cutoff = status.get("cutoff", {})
    if not record.get("pod_id") and cutoff.get("podId"):
        record["pod_id"] = str(cutoff["podId"])
    if not record.get("volume_id") and cutoff.get("volumeId"):
        record["volume_id"] = str(cutoff["volumeId"])
    record["last_provider_status"] = status
    if status.get("worker") == "DONE":
        record["status"] = "done_pending_collection"
    elif status.get("worker") == "FAILED":
        record["status"] = "failed_pending_collection"
    elif not status.get("pod_present"):
        record["status"] = "pod_absent_incomplete" if not record.get("collection_verified") else "collected"
    else:
        record["status"] = status.get("worker", "unknown").lower()
    _write_session(Path(session), record)
    return record


def collect_source_panel(session: Path, provider) -> dict:
    record = _load_session(session)
    local = Path(record["local_collection"])
    status = provider.status(record.get("pod_id"), record["study_root"],
                             cutoff_session_id=record.get("cutoff_session_id"))
    cutoff = status.get("cutoff", {})
    record["pod_id"] = record.get("pod_id") or cutoff.get("podId")
    record["volume_id"] = record.get("volume_id") or cutoff.get("volumeId")
    record["data_center_id"] = record.get("data_center_id") or cutoff.get("volumeSpec", {}).get("dataCenterId")
    _write_session(Path(session), record)
    if cutoff.get("status") != "terminated":
        raise RuntimeError("collection requires independent service verification that compute is terminated")
    import inspect
    collect_args = {"volume_id": record.get("volume_id"),
                    "data_center_id": record.get("data_center_id")}
    if "volume_id" in inspect.signature(provider.collect).parameters:
        provider.collect(record["pod_id"], record["study_root"], local, **collect_args)
    else:
        provider.collect(record["pod_id"], record["study_root"], local)
    try:
        verified = verify_source_panel_collection(local / "collection")
    except Exception as exc:
        record["collection_verified"] = False
        record["collection_error"] = f"{type(exc).__name__}: {exc}"
        _write_session(Path(session), record)
        raise
    if hasattr(provider, "commit_collection"):
        receipt = provider.commit_collection(
            record["cutoff_session_id"], sha256=verified["manifest_sha256"],
            files=verified["files"])
        record["cutoff_collection_receipt"] = receipt
    record["collection_verified"] = True
    record["collection_verification"] = verified
    record["status"] = "collected"
    _write_session(Path(session), record)
    return record


def watch_source_panel(session: Path, provider, *, interval_seconds: int = 10,
                       max_wait_seconds: int = 7200) -> dict:
    """Watch a detached source worker, continuously collecting logs and outputs."""
    import time
    if interval_seconds < 1 or max_wait_seconds < 1:
        raise ValueError("watch intervals and deadlines must be positive")
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        record = source_panel_status(session, provider)
        status = record["last_provider_status"]
        if status.get("pod_present"):
            try:
                Path(record["local_collection"]).mkdir(parents=True, exist_ok=True)
                import inspect
                collect_args = {"volume_id": record.get("volume_id"),
                                "data_center_id": record.get("data_center_id")}
                if "volume_id" in inspect.signature(provider.collect).parameters:
                    provider.collect(record["pod_id"], record["study_root"],
                                     Path(record["local_collection"]), **collect_args)
                else:
                    provider.collect(record["pod_id"], record["study_root"],
                                     Path(record["local_collection"]))
            except Exception as exc:
                record["last_collect_error"] = f"{type(exc).__name__}: {exc}"
                _write_session(Path(session), record)
        if status.get("worker") == "DONE":
            provider.terminate(record["pod_id"], cutoff_session_id=record.get("cutoff_session_id"))
            record = collect_source_panel(session, provider)
            record["status"] = "complete"
            record["terminated"] = True
            _write_session(Path(session), record)
            return record
        if status.get("worker") == "FAILED" or not status.get("pod_present"):
            record["status"] = "failed_incomplete" if status.get("worker") == "FAILED" else "pod_lost_incomplete"
            _write_session(Path(session), record)
            return record
        time.sleep(interval_seconds)
    record = _load_session(session)
    record["status"] = "watch_timeout_running"
    _write_session(Path(session), record)
    return record


def resume_source_panel(session: Path, provider, *, interval_seconds: int = 10,
                        max_wait_seconds: int = 60) -> dict:
    """Reconcile a journaled run; never creates replacement resources."""
    import time
    if interval_seconds < 1 or max_wait_seconds < 1:
        raise ValueError("resume intervals and deadlines must be positive")
    deadline = time.monotonic() + max_wait_seconds
    while True:
        record = source_panel_status(session, provider)
        remote = record.get("last_provider_status", {})
        cutoff = remote.get("cutoff", {})
        worker = remote.get("worker")
        if worker in {"DONE", "FAILED"}:
            if record.get("pod_id") and cutoff.get("status") != "terminated":
                provider.terminate(record["pod_id"], cutoff_session_id=record["cutoff_session_id"])
                continue
            if cutoff.get("status") == "terminated":
                return collect_source_panel(session, provider)
        if cutoff.get("status") == "terminated":
            # Collect committed partial diagnostics when present. Missing DONE
            # remains explicitly incomplete and cannot become a scientific result.
            try:
                record = collect_source_panel(session, provider)
            except (FileNotFoundError, RuntimeError) as exc:
                record["status"] = "terminated_incomplete_recovery_required"
                record["resume_error"] = f"{type(exc).__name__}: {exc}"
                _write_session(Path(session), record)
            return record
        if worker in {"RUNNING", "DONE", "FAILED"} and record.get("pod_id"):
            remaining = max(1, int(deadline - time.monotonic()))
            return watch_source_panel(session, provider, interval_seconds=interval_seconds,
                                      max_wait_seconds=remaining)
        if time.monotonic() >= deadline:
            record["status"] = "resume_reconciled_no_replacement"
            _write_session(Path(session), record)
            return record
        time.sleep(min(interval_seconds, max(1, deadline - time.monotonic())))


def teardown_source_panel(session: Path, provider) -> dict:
    record = _load_session(session)
    if not record.get("collection_verified"):
        raise RuntimeError("refusing normal teardown before local collection is checksum verified")
    provider.terminate(record.get("pod_id"), cutoff_session_id=record.get("cutoff_session_id"))
    verified = provider.status(record.get("pod_id"), record["study_root"],
                              cutoff_session_id=record.get("cutoff_session_id")).get("cutoff", {})
    if verified.get("status") != "terminated" or not any(
            event.get("type") == "storage_cleanup_verified" for event in verified.get("events", [])):
        raise RuntimeError("teardown requested; provider termination and storage cleanup are not verified")
    record["terminated"] = True
    record["status"] = "complete"
    _write_session(Path(session), record)
    return record


def launch_validation(session: Path, provider, *, authorize_spend: bool = False,
                      cap_usd: float = VALIDATION_CAP_USD, max_hours: float = VALIDATION_MAX_HOURS,
                      task: str = "put_the_cream_cheese_in_the_bowl") -> dict:
    """Create one validation pod. Refuses unless spending was explicitly authorized."""
    if not authorize_spend:
        raise PermissionError("validation launch refused: spending is not authorized")
    if cap_usd > VALIDATION_CAP_USD or max_hours > VALIDATION_MAX_HOURS:
        raise ValueError("validation launch exceeds the 1.5 hour / $8 ceiling")
    plan = validation_plan(task=task)
    pod = provider.create_pod(name="raid-wm-validate", image=plan["image"], gpu=plan["gpu"],
                              env={"RAID_CONTAINER_IMAGE_DIGEST": plan["image_digest"],
                                   "PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    pod_id = str(pod["id"])
    provider.install_key(pod_id)
    provider.start_watchdog(pod_id, mode="deadline", seconds=int(max_hours * 3600))
    provider.start_watchdog(pod_id, mode="exit_guard", seconds=1800)
    provider.upload(pod_id, plan["uploads"], plan["study_root"])
    provider.remote(pod_id, plan["bootstrap"])
    provider.remote(pod_id, plan["command"])
    record = {"pod_id": pod_id, "plan": {k: plan[k] for k in ("image", "cap_usd", "max_hours", "starts_training")},
              "scientific_result": False}
    session.mkdir(parents=True, exist_ok=True)
    (session / "session.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
