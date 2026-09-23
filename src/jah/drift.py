"""Drift monitoring against exact Clopper-Pearson bounds per ADR-08 / claude_gap.md Step 3B.

Tracks realized coverage and realized error rate against certified binomial confidence
bounds per task/workload, alerting when realized error approaches the bound or when
coverage falls, and signaling when recalibration is required.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jah.calibration import CalibrationProfile, compute_binomial_upper_bound, load_profile
from jah.workload import sha256_file


@dataclass(frozen=True)
class DriftAlert:
    severity: str  # "WARNING" | "CRITICAL"
    code: str
    message: str


@dataclass(frozen=True)
class DriftReport:
    workload_id: str
    profile_id: str
    evaluated_at: str
    total_decisions: int
    accepted_count: int
    realized_coverage: float
    target_coverage: float
    coverage_delta: float
    errors_in_accepted: int
    realized_error_rate: float
    realized_error_95_upper_bound: float
    target_error_upper_bound: float
    alerts: list[DriftAlert] = field(default_factory=list)
    trigger_recalibration: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "profile_id": self.profile_id,
            "evaluated_at": self.evaluated_at,
            "total_decisions": self.total_decisions,
            "accepted_count": self.accepted_count,
            "realized_coverage": self.realized_coverage,
            "target_coverage": self.target_coverage,
            "coverage_delta": self.coverage_delta,
            "errors_in_accepted": self.errors_in_accepted,
            "realized_error_rate": self.realized_error_rate,
            "realized_error_95_upper_bound": self.realized_error_95_upper_bound,
            "target_error_upper_bound": self.target_error_upper_bound,
            "alerts": [
                {"severity": a.severity, "code": a.code, "message": a.message}
                for a in self.alerts
            ],
            "trigger_recalibration": self.trigger_recalibration,
        }


def check_prediction_drift(
    records: list[dict[str, Any]],
    profile: CalibrationProfile,
    *,
    error_warning_threshold_ratio: float = 0.80,
    min_acceptable_coverage: float = 0.50,
) -> DriftReport:
    """Analyze decision predictions against a certified calibration profile."""
    if not records:
        raise ValueError("drift analysis requires at least one evaluation record")

    total_decisions = len(records)
    accepted_records = [r for r in records if r.get("disposition") == "accept"]
    accepted_count = len(accepted_records)
    realized_coverage = accepted_count / total_decisions if total_decisions > 0 else 0.0

    target_coverage = profile.metrics.coverage if profile.metrics else min_acceptable_coverage
    target_error_bound = (
        profile.metrics.accepted_error_95_upper_bound if profile.metrics else 0.05
    )

    errors_in_accepted = 0
    for r in accepted_records:
        ref = r.get("reference")
        pred = r.get("selected") or r.get("prediction") or r.get("value")
        if ref is not None and pred is not None and str(ref) != str(pred):
            errors_in_accepted += 1

    realized_error_rate = errors_in_accepted / accepted_count if accepted_count > 0 else 0.0
    realized_upper_bound = (
        compute_binomial_upper_bound(errors_in_accepted, accepted_count)
        if accepted_count > 0
        else 1.0
    )

    alerts: list[DriftAlert] = []

    # Check 1: Realized error upper bound breach
    if accepted_count > 0 and realized_upper_bound > target_error_bound:
        alerts.append(
            DriftAlert(
                severity="CRITICAL",
                code="ERROR_BOUND_BREACH",
                message=(
                    f"Realized 95% error upper bound {realized_upper_bound:.4f} exceeds target "
                    f"bound {target_error_bound:.4f} ({errors_in_accepted}/{accepted_count} accepted errors)"
                ),
            )
        )
    elif accepted_count > 0 and realized_upper_bound >= (target_error_bound * error_warning_threshold_ratio):
        alerts.append(
            DriftAlert(
                severity="WARNING",
                code="ERROR_BOUND_WARNING",
                message=(
                    f"Realized 95% error upper bound {realized_upper_bound:.4f} is approaching "
                    f"target threshold {target_error_bound:.4f} (>= {error_warning_threshold_ratio*100:.0f}%)"
                ),
            )
        )

    # Check 2: Coverage drop
    coverage_delta = realized_coverage - target_coverage
    if realized_coverage < min_acceptable_coverage:
        alerts.append(
            DriftAlert(
                severity="CRITICAL",
                code="COVERAGE_BELOW_MINIMUM",
                message=(
                    f"Realized coverage {realized_coverage:.4f} is below minimum requirement "
                    f"{min_acceptable_coverage:.4f}"
                ),
            )
        )
    elif coverage_delta < -0.05:
        alerts.append(
            DriftAlert(
                severity="WARNING",
                code="COVERAGE_DROP",
                message=(
                    f"Realized coverage {realized_coverage:.4f} dropped {abs(coverage_delta):.4f} "
                    f"below baseline target {target_coverage:.4f}"
                ),
            )
        )

    trigger_recalibration = any(a.severity == "CRITICAL" for a in alerts) or len(alerts) >= 2

    return DriftReport(
        workload_id=profile.workload_id,
        profile_id=profile.profile_id,
        evaluated_at=datetime.now(UTC).isoformat(),
        total_decisions=total_decisions,
        accepted_count=accepted_count,
        realized_coverage=realized_coverage,
        target_coverage=target_coverage,
        coverage_delta=coverage_delta,
        errors_in_accepted=errors_in_accepted,
        realized_error_rate=realized_error_rate,
        realized_error_95_upper_bound=realized_upper_bound,
        target_error_upper_bound=target_error_bound,
        alerts=alerts,
        trigger_recalibration=trigger_recalibration,
    )


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSON or JSONL prediction file."""
    if not path.exists():
        raise FileNotFoundError(f"prediction file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "rows" in data and isinstance(data["rows"], list):
            return data["rows"]
        if "decisions" in data and isinstance(data["decisions"], list):
            return data["decisions"]
    raise ValueError(f"unrecognized prediction format in {path}")


def run_drift_check(args: argparse.Namespace) -> int:
    """CLI entrypoint for drift checking."""
    pred_path = Path(args.predictions).resolve()
    profile_path = Path(args.profile).resolve()

    records = load_prediction_rows(pred_path)
    profile = load_profile(profile_path)

    report = check_prediction_drift(
        records,
        profile,
        error_warning_threshold_ratio=getattr(args, "warning_ratio", 0.80),
        min_acceptable_coverage=getattr(args, "min_coverage", 0.50),
    )

    report_dict = report.to_dict()
    print(json.dumps(report_dict, indent=2))

    if getattr(args, "output", None):
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report_dict, indent=2) + "\n", encoding="utf-8")
        print(f"Drift report written to {out_path}")

    return 1 if report.trigger_recalibration else 0
