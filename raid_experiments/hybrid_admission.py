"""Single fail-closed admission check for scientific v6 execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from .hybrid_panel import validate_source_panel_gate


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"scientific admission evidence is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"scientific admission evidence must be an object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_scientific_admission(protocol: dict, evidence_dir: str | Path, *,
                                  stage: str) -> dict:
    """Validate the exact live operational, CUDA, panel and funded-forecast gates.

    Evidence files are written only after their own collection/validation steps.
    This function is shared by both ``pilot`` and individual ``rollout`` so neither
    command can bypass one of the scientific execution gates.
    """
    if stage not in {"pilot", "rollout"}:
        raise ValueError("scientific admission stage must be pilot or rollout")
    from .hybrid_protocol import is_vpp2
    if is_vpp2(protocol):
        # VPP2 has its own 15-case gate. Funding freshness is checked by the
        # host at launch; on the pod the independent service cap bounds money.
        from .hybrid_vpp2_admission import validate_vpp2_admission
        return validate_vpp2_admission(protocol, evidence_dir, stage=stage, require_fresh_funding=False)
    root = Path(evidence_dir)
    panel_root = root / "source_panel" if (root / "source_panel" / "panel_summary.json").is_file() else root
    evidence_root = root.parent if panel_root == root and root.name == "source_panel" else root
    source_protocol_digest = (protocol.get("design", {}).get("temporal", {}).get("source_protocol_digest")
                              or protocol.get("digest"))
    errors: list[str] = []

    try:
        panel = validate_source_panel_gate(protocol, panel_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        panel = None
        errors.append(f"source_panel_gate:{exc}")

    cuda_path = evidence_root / "cuda_evidence.json"
    cuda = _read(cuda_path) if cuda_path.is_file() else {}
    if (cuda.get("status") != "passed"
            or cuda.get("protocol_digest") != source_protocol_digest
            or cuda.get("model_digest") != protocol.get("model_digest")
            or not isinstance(cuda.get("cuda_device"), str)
            or not cuda.get("cuda_device")
            or not isinstance(cuda.get("peak_memory_bytes"), int)
            or cuda.get("peak_memory_bytes", 0) <= 0):
        errors.append("cuda_measurement_missing_or_not_bound_to_protocol")

    operational_path = evidence_root / "operational_acceptance.json"
    operational = _read(operational_path) if operational_path.is_file() else {}
    def valid_digest(value: object) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(
            character in "0123456789abcdef" for character in value)

    try:
        completed_at = datetime.fromisoformat(str(operational["completed_at"]).replace("Z", "+00:00"))
        acceptance_age = (datetime.now(timezone.utc) - completed_at.astimezone(timezone.utc)).total_seconds()
        acceptance_fresh = completed_at.tzinfo is not None and 0 <= acceptance_age <= 24 * 60 * 60
    except (KeyError, TypeError, ValueError):
        acceptance_fresh = False
    required_operational = (
        operational.get("status") == "passed"
        and operational.get("mode") == "live_independent_cutoff_and_recovery"
        and operational.get("protocol_digest") == source_protocol_digest
        and operational.get("termination_verified") is True
        and operational.get("recovery_verified") is True
        and operational.get("storage_cleanup_verified") is True
        and operational.get("local_supervisor_disabled") is True
        and operational.get("pod_watchdog_disabled") is True
        and operational.get("service_deployment_sha256")
        and operational.get("package_sha256")
        and operational.get("runtime_sha256")
        and operational.get("recovered_artifact_sha256")
        and operational.get("recovered_artifact_sha256") == operational.get("expected_artifact_sha256")
        and acceptance_fresh
        and valid_digest(operational.get("service_deployment_sha256"))
        and valid_digest(operational.get("package_sha256"))
        and valid_digest(operational.get("runtime_sha256"))
        and valid_digest(operational.get("recovered_artifact_sha256"))
        and valid_digest(operational.get("source_code_sha256"))
    )
    if not required_operational:
        errors.append("live_independent_cutoff_and_recovery_acceptance_missing_or_unbound")
    else:
        from .hybrid_live_acceptance import verify_live_acceptance
        try:
            authority = verify_live_acceptance(protocol, evidence_root, operational)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            errors.append(f"live_acceptance_authority_or_release_verification_failed:{exc}")

    forecast_path = evidence_root / "pilot_forecast.json"
    forecast = _read(forecast_path) if forecast_path.is_file() else {}
    def amount(key: str) -> Decimal:
        value = Decimal(str(forecast[key]))
        if not value.is_finite() or value < 0:
            raise ValueError(key)
        return value

    def fresh(key: str) -> bool:
        stamp = datetime.fromisoformat(str(forecast[key]).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            return False
        age = (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
        return 0 <= age <= 300

    try:
        projected_total = amount("projected_total_usd")
        prepaid_available = amount("prepaid_available_usd")
        cumulative_spend = amount("cumulative_study_spend_usd")
        project_ceiling = amount("project_ceiling_usd")
        reserve = amount("reserved_usd")
        remaining_project_allowance = amount("remaining_project_allowance_usd")
        margin = Decimal(str(forecast["margin_fraction"]))
        forecast_ok = (
            forecast.get("status") == "passed"
            and forecast.get("protocol_digest") == protocol.get("digest")
            and fresh("billing_refreshed_at")
            and fresh("quote_refreshed_at")
            and fresh("provider_inventory_refreshed_at")
            and forecast.get("all_liabilities_known") is True
            and forecast.get("outstanding_charges_known") is True
            and forecast.get("service_costs_known") is True
            and forecast.get("provider_resource_inventory_checked") is True
            and project_ceiling == Decimal("250")
            and reserve >= Decimal("25")
            and remaining_project_allowance == project_ceiling - reserve - cumulative_spend
            and margin >= 0.25
            and projected_total > 0
            and prepaid_available > 0
            and remaining_project_allowance > 0
            and projected_total <= prepaid_available
            and projected_total <= remaining_project_allowance
            and forecast.get("storage_recovery_and_cleanup_included") is True
        )
    except (KeyError, TypeError, ValueError, InvalidOperation):
        forecast_ok = False
    if not forecast_ok:
        errors.append("current_funded_pilot_forecast_missing_or_over_budget")

    if errors:
        raise ValueError("scientific execution blocked: " + "; ".join(errors))
    return {
        "status": "passed",
        "stage": stage,
        "protocol_digest": protocol["digest"],
        "model_digest": protocol["model_digest"],
        "source_panel": panel,
        "authoritative_acceptance": authority,
        "evidence_sha256": {
            "cuda_evidence": _sha(cuda_path),
            "operational_acceptance": _sha(operational_path),
            "pilot_forecast": _sha(forecast_path),
        },
        "scientific_result": False,
    }
