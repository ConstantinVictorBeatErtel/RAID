"""v7 compute benchmark: acceptance -> managed launch (fixed 9 h window) -> setup on the same pod -> watch -> collect
-> verified cleanup. Run with the ops venv; never twice (it refuses an existing intent)."""
from pathlib import Path
import json, sys, time, traceback, urllib.error
from decimal import Decimal
P = Path(__file__).resolve().parent; ROOT = P.parents[2]; sys.path.insert(0, str(ROOT))
from raid_experiments import operational_acceptance as o, managed_source as m
DEPOSITS = "135"
# Same reasoned tolerance as the pilot drivers (billing over-states the balance drop; conservative direction).
o.OVERSTATED_BILLING_TOLERANCE_USD = Decimal("0.50")  # grows with rounded storage hours; user: ignore money checks
CAP, COMPUTE, STORAGE, VOLUME, GPU_GB = "12", 21600, 25200, 80, 80  # same 80 GB GPU class as the pilot (A100)
a = P / "acceptance"
TRANSIENT = ("not funded", "stage reservation exceeds", "no available compatible GPU", "Bad Gateway", "ListBuckets",
             "timed out", "Connection reset")
SETUP_TEMPORARY = ("SSH not ready", "credential delivery failed", "Connection reset", "Connection was closed",
                   "ConnectionClosedError", "(524)", "(502)", "(503)", "(500)", "Bad Gateway", "timed out", "Read timeout",
                   "EndpointConnectionError", "Could not connect")


def retrying(fn, what, minutes=90):
    end = time.monotonic() + minutes * 60
    while True:
        try:
            return fn()
        except Exception as error:
            temporary = any(t in str(error) for t in TRANSIENT) or isinstance(error, (TimeoutError, OSError, urllib.error.URLError))
            if not temporary or time.monotonic() > end:
                raise
            print("WAIT_RETRY", what, type(error).__name__, str(error)[:140], flush=True)
            time.sleep(180)


def session_cap() -> str:
    """User (16 Oct 2026): no dollar cap; use almost the whole prepaid balance (Runpod stops at zero credit)."""
    balance = Decimal(str(o.cli("user")["clientBalance"]))
    return str(max(Decimal(CAP), (balance - Decimal("2")).quantize(Decimal("0.01"))))


def admit_launch():
    global CAP
    CAP = session_cap()
    print("SESSION_CAP_USD", CAP, flush=True)
    admission = o.quote(P, observed_deposits_usd=DEPOSITS, cap_usd=CAP, compute_seconds=COMPUTE,
                        storage_seconds=STORAGE, volume_gb=VOLUME, min_gpu_memory_gb=GPU_GB)
    o.atomic(P / "s3_preflight.json", o.verify_recovery_access(admission["data_center_id"]))
    return admission, int(time.time() * 1000)


def launch(acceptance_id):
    release = json.loads((P / "release.json").read_text())
    admission, now = retrying(admit_launch, "launch_admission")
    health = o.service("/v2/health")
    body = {"sessionId": release["session_id"], "stage": "source_panel", "quoteStartAt": now,
            "collectBy": now + (COMPUTE - 600) * 1000, "deadlineAt": now + COMPUTE * 1000, "deleteStorageAt": now + STORAGE * 1000,
            "capUsd": CAP, "maxHourlyUsd": admission["gpu_hourly_usd"], "packageSha256": release["archive_sha256"],
            "bindings": {**release["bindings"], "deployment": health["deployment"]}, "acceptanceSessionId": acceptance_id,
            "sentinel": {"key": f"results/{release['session_id']}/collection/outputs_sha256.json", "sha256": release["identity_sha256"]},
            "volumeSpec": {"size": VOLUME, "dataCenterId": admission["data_center_id"]},
            "podSpec": {"imageName": o.IMAGE_REFERENCE, "gpuTypeIds": [admission["gpu"]], "gpuCount": 1, "cloudType": "SECURE",
                        "computeType": "GPU", "dataCenterIds": [admission["data_center_id"]], "containerDiskInGb": 60,
                        "volumeInGb": 0, "volumeMountPath": "/workspace", "ports": ["22/tcp"], "supportPublicIp": True,
                        "env": {"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                "RAID_CONTAINER_IMAGE_DIGEST": o.IMAGE_REFERENCE.split("@")[1]}},
            "spendAdmission": admission}
    o.atomic(P / "intent.json", body)
    registered = o.service("/v2/sessions/" + body["sessionId"] + "/register", body); o.atomic(P / "registered.json", registered)
    assert all(registered["request"][k] == body[k] for k in ("collectBy", "deadlineAt", "deleteStorageAt"))
    o.atomic(P / "launch_intent.json", {"fingerprint": registered["fingerprint"], "at": time.time()})
    o.atomic(P / "launch_response.json", o.service("/v2/sessions/" + body["sessionId"] + "/launch", {"fingerprint": registered["fingerprint"]}))
    print("GPU", admission["gpu"], admission["gpu_hourly_usd"], flush=True)
    for attempt in range(12):  # same allocation; the package upload resumes from its journal
        try:
            return m.resume(P, setup=True)
        except Exception as error:
            if not any(t in f"{type(error).__name__} {error}" for t in SETUP_TEMPORARY):
                raise
            print("SETUP_RETRY", attempt, type(error).__name__, str(error)[:120], flush=True)
            time.sleep(30 if "SSH not ready" in str(error) else 60)
    raise RuntimeError("pod setup did not complete within the reconciliation allowance")


try:
    r = json.loads((P / "release.json").read_text()); assert o.sha(r["archive"]) == r["archive_sha256"]
    b = json.loads((P / "code_binding_files.json").read_text())
    assert all(o.sha(ROOT / n.removeprefix("host/")) == h for n, h in b.items() if n.startswith("host/")), "host code changed"
    rehearsal = json.loads((P / "rehearsal_result.json").read_text())
    assert rehearsal["status"] == "passed" and rehearsal["archive_sha256"] == r["archive_sha256"], "rehearse this exact archive"
    assert not (P / "intent.json").exists() and not (a / "intent.json").exists(), "use reconciliation, never duplicate launch"
    state = retrying(lambda: o.launch(a, scientific_release=m.acceptance_release(r), timing={"cutoff": 180, "compute": 480, "storage": 600},
                                      cap_usd="1", observed_deposits_usd=DEPOSITS), "acceptance_launch")
    print("ACCEPTANCE_STARTED", state["request"]["sessionId"], flush=True)
    for watch_try in range(20):
        try:
            state = o.dispatch("watch", a, max_wait=1500); break
        except (TimeoutError, OSError, urllib.error.URLError) as error:
            print("ACCEPTANCE_WATCH_RETRY", watch_try, type(error).__name__, flush=True); time.sleep(20)
    else:
        raise RuntimeError("acceptance watch did not complete")
    assert state.get("closed") and state.get("podDeleted") and state.get("volumeDeleted") and state.get("terminationReason") == "independent_alarm" and not state.get("failure")
    o.atomic(a / "live_acceptance_result.json", {"status": "passed", "authenticated_service_status": state, "scientific_result": False})
    print("ACCEPTANCE_PASSED", flush=True)
    state = launch(state["request"]["sessionId"])
    print("BENCH_WORKER_STARTED", state.get("podId"), flush=True)
    end = time.monotonic() + STORAGE + 900
    while True:
        try:
            state = m.watch(P, max_wait=max(60, int(end - time.monotonic()))); break
        except (TimeoutError, OSError, urllib.error.URLError) as error:
            print("WATCH_RETRY", type(error).__name__, flush=True)
            if time.monotonic() > end:
                raise
            time.sleep(30)
    o.atomic(P / "source_final_result.json", state)
    assert state.get("closed") and state.get("podDeleted") and state.get("volumeDeleted") and not state.get("failure")
    manifest = json.loads((P / "collection/outputs_sha256.json").read_text())
    assert all(o.sha(P / "collection" / n) == v for n, v in manifest.items())
    assert (P / "collection/DONE").exists() and not (P / "collection/FAILED").exists()
    summary = json.loads((P / "collection/bench/bench_summary.json").read_text())
    assert summary["units"] == 6
    o.atomic(P / "verified_result.json", {"status": "bench_collected", "files": len(manifest), "pod_deleted": True,
                                         "volume_deleted": True, "units": summary["units"], "scientific_result": False})
    print("BENCH_VERIFIED_AND_CLEANED", flush=True)
except BaseException as e:
    o.atomic(P / "execution_error.json", {"type": type(e).__name__, "error": str(e), "traceback": traceback.format_exc()})
    for folder in (P, a):
        if (folder / "intent.json").exists():
            try:
                ident = json.loads((folder / "intent.json").read_text())["sessionId"]; s = o.service("/v2/sessions/" + ident)
                if not s.get("closed"): o.service("/v2/sessions/" + ident + "/terminate", {})
            except Exception: pass
    print("STOPPED", type(e).__name__, str(e), flush=True); raise
