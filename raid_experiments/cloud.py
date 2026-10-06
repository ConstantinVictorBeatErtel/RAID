"""Runpod quoting and lifecycle helpers.

All mutating provider calls live here so ``preflight`` and ``estimate`` stay
read-only. The runner tags resources by experiment name and refuses to delete
resources with a different tag.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import hashlib
from pathlib import Path
import subprocess
import shlex
import time
import tarfile
import tempfile
from typing import Any

from .config import ExperimentConfig


def _call(*args: str, timeout: int = 120) -> str:
    proc = subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout)
    return proc.stdout


def account() -> dict[str, Any]:
    return json.loads(_call("runpodctl", "user"))


def gpu_quotes() -> list[dict[str, Any]]:
    payload = json.loads(_call("runpodctl", "gpu", "list"))
    return payload if isinstance(payload, list) else payload.get("data", payload.get("gpus", []))


def choose_gpu(cfg: ExperimentConfig) -> dict[str, Any]:
    candidates = {cfg.cloud.preferred_gpu, cfg.cloud.fallback_gpu}
    def matches(row: dict[str, Any]) -> bool:
        names = {str(row.get("displayName", "")), str(row.get("gpuId", ""))}
        return bool(names & candidates)
    rows = [row for row in gpu_quotes() if matches(row) and row.get("available")]
    rows = [row for row in rows if (row.get("securePricePerHr") or float("inf")) <= cfg.cloud.max_gpu_hourly_usd]
    if not rows:
        raise RuntimeError("No permitted Runpod GPU is currently available below the configured rate")
    rows.sort(key=lambda row: (0 if row.get("displayName") == cfg.cloud.preferred_gpu else 1, row.get("securePricePerHr", 999)))
    return rows[0]


def estimate(cfg: ExperimentConfig, *, hours: float = 2.0) -> dict[str, Any]:
    gpu = choose_gpu(cfg)
    hourly = float(gpu.get("securePricePerHr", 0.0))
    compute = hourly * hours
    # Runpod lists network-volume storage at $0.07/GB/month billed hourly;
    # container disk is $0.10/GB/month billed per second.
    storage_hour = cfg.cloud.volume_gb * 0.07 / (30.0 * 24.0)
    container_hour = cfg.cloud.container_disk_gb * 0.10 / (30.0 * 24.0 * 3600.0)
    reserve = float(cfg.cloud.reserve_usd)
    workload = compute + storage_hour * hours + container_hour * hours
    total = workload + reserve
    margin_total = workload * 1.25 + reserve
    balance = float(account().get("clientBalance") or 0.0)
    return {
        "gpu_hourly": hourly,
        "gpu": gpu.get("displayName", gpu.get("gpuId", "unknown")),
        "hours": hours,
        "compute": compute,
        "network_storage_hourly": storage_hour,
        "container_storage_hourly": container_hour,
        "storage_for_run": (storage_hour + container_hour) * hours,
        "reserve": reserve,
        "reserved_total": total,
        "pilot_ceiling": cfg.cloud.budget_usd,
        "with_25_percent_margin": margin_total,
        "within_ceiling": margin_total <= cfg.cloud.budget_usd,
        "client_balance": balance,
        "stage_zero_launch_ready": balance >= 10.0,
    }


def list_network_volumes() -> list[dict[str, Any]]:
    payload = json.loads(_call("runpodctl", "network-volume", "list"))
    return payload if isinstance(payload, list) else payload.get("volumes", payload.get("data", []))


def ensure_network_volume(cfg: ExperimentConfig, gpu: dict[str, Any]) -> dict[str, Any]:
    existing = [v for v in list_network_volumes() if v.get("name") == cfg.name]
    if existing:
        volume = existing[0]
        if int(volume.get("size", volume.get("sizeInGb", 0))) < cfg.cloud.volume_gb:
            raise RuntimeError(f"Existing volume {volume.get('id')} is smaller than the configured {cfg.cloud.volume_gb} GB")
        return volume
    choices = [x for x in gpu.get("dataCenterAvailability", []) if x.get("stockStatus") != "none"]
    if not choices:
        raise RuntimeError("GPU has no usable data center for the experiment network volume")
    created = json.loads(_call("runpodctl", "network-volume", "create", "--name", cfg.name, "--size", str(cfg.cloud.volume_gb), "--data-center-id", choices[0]["dataCenterId"]))
    created["raid_created"] = True
    return created


def _write_ledger(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _verify_capacity(pod_id: str, cfg: ExperimentConfig) -> dict[str, Any]:
    details = json.loads(_call("runpodctl", "pod", "get", str(pod_id), "--include-machine"))
    machine = details.get("machine", details)
    vcpus = machine.get("vcpuCount", machine.get("vcpus", machine.get("cpuCount")))
    ram = machine.get("memoryInGb", machine.get("ramInGb", machine.get("memoryGb")))
    if vcpus is None or ram is None:
        raise RuntimeError(f"pod capacity was not reported; refusing unverified resource: {details}")
    if int(vcpus) < cfg.cloud.min_vcpus or float(ram) < cfg.cloud.min_host_ram_gb:
        raise RuntimeError(f"pod capacity {vcpus} vCPU/{ram} GB does not meet configured minimum")
    return {"vcpus": int(vcpus), "host_ram_gb": float(ram)}


def create_pod(cfg: ExperimentConfig, *, minutes: int | None = None, ledger_path: str | Path | None = None) -> dict[str, Any]:
    """Create exactly one tagged pod after an auditable budget check."""
    if not cfg.cloud.launch_enabled:
        raise RuntimeError("paid Runpod provisioning is disabled by this configuration")
    # Derive the provider deadline from the remaining budget after storage and
    # reserve, while respecting the configured maximum wall time.
    gpu_quote = choose_gpu(cfg)
    hourly = float(gpu_quote.get("securePricePerHr", 0.0))
    storage_per_hour = cfg.cloud.volume_gb * 0.07 / (30.0 * 24.0) + cfg.cloud.container_disk_gb * 0.10 / (30.0 * 24.0 * 3600.0)
    # Keep the reserve outside the margin calculation. This deliberately
    # under-allocates time, which is safer when provider billing arrives late.
    available_compute = max(0.0, cfg.cloud.budget_usd / 1.25 - cfg.cloud.reserve_usd)
    budget_minutes = max(1, int(available_compute / max(hourly + storage_per_hour, 1e-9) * 60.0))
    duration = min(int(minutes or cfg.cloud.terminate_after_minutes), budget_minutes)
    if duration < 1:
        raise RuntimeError("configured budget leaves no billable compute time after reserve and storage")
    quote = estimate(cfg, hours=duration / 60.0)
    if not quote["within_ceiling"]:
        raise RuntimeError(f"Launch exceeds configured budget after margin: {quote}")
    funds = account()
    balance = float(funds.get("clientBalance") or 0.0)
    # Stage 0 is capped at $10.  Refuse to create a pod until that minimum
    # launch reserve exists; this avoids consuming a nearly empty account while
    # Luna is still preparing the run.
    if balance < 10.0:
        raise RuntimeError(f"Runpod launch blocked: client balance ${balance:.2f} is below the $10 stage-0 reserve")
    gpu = choose_gpu(cfg)
    volume = ensure_network_volume(cfg, gpu)
    stop_at = datetime.now(timezone.utc) + timedelta(minutes=duration)
    args = ["runpodctl", "pod", "create", "--image", "runpod/pytorch:2.1.2-py3.10-cuda11.8.0-devel-ubuntu22.04", "--gpu-id", gpu["gpuId"], "--gpu-count", "1", "--cloud-type", "SECURE", "--container-disk-in-gb", str(cfg.cloud.container_disk_gb), "--network-volume-id", str(volume["id"]), "--ports", "22/tcp", "--name", cfg.name, "--terminate-after", stop_at.strftime("%Y-%m-%dT%H:%M:%SZ")]
    # Persist the volume ownership before the provider call. If the process is
    # interrupted, the run can still reconcile the volume by name and ID.
    ledger = Path(ledger_path) if ledger_path else None
    _write_ledger(ledger, {"experiment_name": cfg.name, "network_volume_id": volume.get("id"), "gpu": gpu, "quote": quote, "status": "provisioning", "created_at": datetime.now(timezone.utc).isoformat()})
    try:
        pod = json.loads(_call(*args, timeout=120))
    except Exception:
        # The provider may have created a pod before the client timed out.
        # Keep the volume and reconcile by the experiment tag before retrying.
        _write_ledger(ledger, {"experiment_name": cfg.name, "network_volume_id": volume.get("id"), "gpu": gpu, "quote": quote, "created_at": datetime.now(timezone.utc).isoformat(), "status": "creation_uncertain"})
        raise
    if not pod.get("id"):
        raise RuntimeError(f"Runpod did not return a pod ID: {pod}")
    pod["raid_network_volume_id"] = volume["id"]
    _write_ledger(ledger, {"experiment_name": cfg.name, "pod_id": pod.get("id"), "network_volume_id": volume.get("id"), "gpu": gpu, "quote": quote, "created_at": datetime.now(timezone.utc).isoformat(), "status": "created"})
    try:
        pod["capacity"] = _verify_capacity(str(pod["id"]), cfg)
    except Exception:
        terminated = False
        try:
            # Re-check provider ownership before cleanup. If the create response
            # or provider listing is inconsistent, preserve the resource for
            # manual reconciliation rather than deleting an unverified pod.
            delete_pod(str(pod["id"]), cfg.name)
            terminated = True
        finally:
            _write_ledger(ledger, {"experiment_name": cfg.name, "pod_id": pod.get("id"), "network_volume_id": volume.get("id"), "gpu": gpu, "quote": quote, "created_at": datetime.now(timezone.utc).isoformat(), "status": "terminated" if terminated else "termination_uncertain"})
            if terminated and volume.get("raid_created"):
                try:
                    _call("runpodctl", "network-volume", "delete", str(volume["id"]))
                except Exception:
                    pass
        raise
    return pod


def _ssh_info(pod_id: str) -> dict[str, Any]:
    return json.loads(_call("runpodctl", "ssh", "info", pod_id))


def _ssh_target(info: dict[str, Any]) -> tuple[str, str, int]:
    host = info.get("host") or info.get("ip") or info.get("publicIp")
    port = int(info.get("port") or info.get("publicPort") or 22)
    user = info.get("user") or info.get("username") or "root"
    if not host:
        raise RuntimeError(f"runpodctl ssh info did not expose a host: {info}")
    return str(user), str(host), port


def remote_command(pod_id: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    user, host, port = _ssh_target(_ssh_info(pod_id))
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), f"{user}@{host}", command], text=True, capture_output=True, check=check, timeout=60)


def upload_repository(pod_id: str, local_root: str | Path, remote_root: str = "/workspace/raid-main") -> None:
    user, host, port = _ssh_target(_ssh_info(pod_id))
    root = Path(local_root).resolve()
    excluded = {".git", ".venv", "__pycache__", "artifacts", "data", "checkpoints", "models"}
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as archive:
        with tarfile.open(archive.name, "w:gz") as tf:
            for path in root.rglob("*"):
                rel = path.relative_to(root)
                if any(part in excluded for part in rel.parts) or path.name == ".env" or path.name.startswith(".env.") or path.name in {"credentials.json", "service-account.json"} or path.is_symlink() or path.suffix in {".pt", ".pth", ".hdf5", ".h5"}:
                    continue
                tf.add(path, arcname=str(rel), recursive=False)
        remote_archive = f"/tmp/raid-source-{int(time.time())}.tar.gz"
        subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new", "-P", str(port), archive.name, f"{user}@{host}:{remote_archive}"], check=True, timeout=600)
        command = f"mkdir -p {shlex.quote(remote_root)} && tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(remote_root)} && rm -f {shlex.quote(remote_archive)}"
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), f"{user}@{host}", command], check=True, timeout=120)


def upload_file(pod_id: str, local_path: str | Path, remote_path: str) -> None:
    user, host, port = _ssh_target(_ssh_info(pod_id))
    remote_parent = str(Path(remote_path).parent)
    subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new", "-p", str(port), f"{user}@{host}", f"mkdir -p {shlex.quote(remote_parent)}"], check=True, timeout=60)
    subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new", "-P", str(port), str(local_path), f"{user}@{host}:{remote_path}"], check=True, timeout=600)


def prepare_remote_environment(pod_id: str, *, remote_root: str, dataset_dir: str, feature_dir: str, split_path: str) -> None:
    """Install the provisional lock and verify data staged on the volume."""
    command = (
        f"cd {shlex.quote(remote_root)} && "
        "python -m pip install --disable-pip-version-check --no-input -r requirements-pilot.txt && "
        f"test -d {shlex.quote(dataset_dir)} && test -d {shlex.quote(feature_dir)} && test -f {shlex.quote(split_path)}"
    )
    remote_command(pod_id, command)


def start_remote_job(*args, **kwargs) -> None:
    raise RuntimeError("legacy remote training is disabled; paid execution stays off for the active study")


def wait_remote_job(pod_id: str, *, timeout_seconds: int, poll_seconds: int = 30) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = remote_command(pod_id, "test -f /workspace/raid-main/raid_job.exit && cat /workspace/raid-main/raid_job.exit || true", check=False)
        code = result.stdout.strip()
        if code:
            return {"complete": True, "exit_code": int(code)}
        disk = remote_command(pod_id, "df -P /workspace | awk 'NR==2 {print $5}'", check=False).stdout.strip().rstrip("%")
        if disk.isdigit() and int(disk) >= 95:
            return {"complete": False, "reason": "remote disk usage exceeded 95 percent", "disk_percent": int(disk)}
        time.sleep(min(poll_seconds, max(1, int(deadline - time.time()))))
    return {"complete": False, "reason": "deadline exceeded"}


def collect_remote_results(pod_id: str, remote_root: str, local_root: str | Path, artifact_rel: str | None = None, *, expected_experiment_id: str | None = None) -> None:
    user, host, port = _ssh_target(_ssh_info(pod_id))
    Path(local_root).mkdir(parents=True, exist_ok=True)
    staging = Path(local_root) / f".collection-staging-{int(time.time() * 1000)}"
    staging.mkdir(parents=True, exist_ok=False)
    remote_artifacts = f"{remote_root}/artifacts/{artifact_rel}" if artifact_rel else f"{remote_root}/artifacts"
    remote_source = f"{user}@{host}:{remote_artifacts}"
    subprocess.run(["scp", "-r", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new", "-P", str(port), remote_source, str(staging)], check=True, timeout=600)
    staged_root = staging / artifact_rel if artifact_rel else staging / "artifacts"
    manifest = staged_root / "collection_manifest.json"
    if not manifest.exists():
        raise RuntimeError("remote collection lacks an experiment collection_manifest.json; retaining resources")
    payload = json.loads(manifest.read_text())
    expected_id = expected_experiment_id or artifact_rel
    if payload.get("format") != 1 or not payload.get("files") or (expected_id and payload.get("experiment_id") != expected_id):
        raise RuntimeError("collection manifest is empty, malformed, or belongs to another experiment")
    required = {"run.json", "config.json", "provenance.json"}
    listed = {str(entry.get("path", "")) for entry in payload["files"]}
    if not required.issubset(listed):
        raise RuntimeError(f"collection manifest lacks required files: {sorted(required - listed)}")
    for entry in payload["files"]:
        rel = Path(str(entry.get("path", "")))
        if rel.is_absolute() or ".." in rel.parts or not entry.get("sha256") or "size" not in entry:
            raise RuntimeError("collection manifest contains an unsafe or incomplete file entry")
        path = staged_root / rel
        if not path.exists() or path.stat().st_size != int(entry["size"]):
            raise RuntimeError(f"collected artifact failed verification: {entry['path']}")
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"collected artifact hash mismatch: {entry['path']}")
    final = Path(local_root) / (artifact_rel or "artifacts")
    if final.exists():
        receipt_path = Path(local_root) / "collection_receipt.json"
        if receipt_path.is_file():
            existing = json.loads(receipt_path.read_text())
            manifest_path = Path(existing.get("manifest", ""))
            if (existing.get("verified") is True and existing.get("experiment_id") == (expected_experiment_id or artifact_rel)
                    and manifest_path.is_file()
                    and hashlib.sha256(manifest_path.read_bytes()).hexdigest() == existing.get("manifest_sha256")):
                return
        raise RuntimeError(f"refusing to overwrite existing unverified artifact directory: {final}")
    staged_root.replace(final)
    manifest_hash = hashlib.sha256((final / "collection_manifest.json").read_bytes()).hexdigest()
    (Path(local_root) / "collection_receipt.json").write_text(json.dumps({"experiment_id": payload["experiment_id"], "manifest": str(final / "collection_manifest.json"), "manifest_sha256": manifest_hash, "verified": True}, indent=2) + "\n")


def list_pods() -> list[dict[str, Any]]:
    payload = json.loads(_call("runpodctl", "pod", "list"))
    return payload if isinstance(payload, list) else payload.get("pods", payload.get("data", []))


def status(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    return [p for p in list_pods() if p.get("name") == cfg.name]


def delete_pod(pod_id: str, experiment_name: str) -> None:
    pods = {str(p.get("id")): p for p in list_pods()}
    pod = pods.get(str(pod_id))
    if pod is None:
        # Runpod may have already applied the scheduled termination.
        return
    if pod.get("name") != experiment_name:
        raise ValueError("Refusing to delete a pod belonging to another experiment")
    _call("runpodctl", "pod", "delete", str(pod_id))


def delete_network_volume(volume_id: str, experiment_name: str, *, receipt_path: str | Path | None = None, receipt_experiment_id: str | None = None) -> None:
    if receipt_path is None or not Path(receipt_path).is_file():
        raise RuntimeError("refusing volume deletion without a verified collection receipt")
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("verified") is not True or receipt.get("experiment_id") != (receipt_experiment_id or experiment_name):
        raise RuntimeError("collection receipt is missing, invalid, or belongs to another experiment")
    manifest_path = Path(receipt.get("manifest", ""))
    if not manifest_path.is_file() or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != receipt.get("manifest_sha256"):
        raise RuntimeError("collection receipt manifest no longer matches its recorded hash")
    volumes = {str(v.get("id")): v for v in list_network_volumes()}
    volume = volumes.get(str(volume_id))
    if volume is None:
        # Idempotent cleanup after a previously verified deletion.
        return
    if volume.get("name") != experiment_name:
        raise ValueError("Refusing to delete a network volume belonging to another experiment")
    _call("runpodctl", "network-volume", "delete", str(volume_id))


def render_runbook(cfg: ExperimentConfig, pod: dict[str, Any]) -> str:
    return f"""# RAID Runpod job\n\nExperiment: {cfg.name}\nPod: {pod.get('id', '<unknown>')}\nNetwork volume: {pod.get('raid_network_volume_id', '<unknown>')}\n\nThe automated runner uploads the repository and checkpoints, starts the four-condition job, waits for its exit marker, collects a result summary, and terminates resources. This file is a recovery record if the client process is interrupted. Resume with:\n\n```bash\npython -m raid_experiments resume --config configs/pilot_100.yaml --experiment-dir EXPERIMENT_DIR\n```\n\nNever remove a pod or volume whose name does not match this experiment.\n"""
