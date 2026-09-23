"""Unit tests for Step 3B drift monitoring against Clopper-Pearson bounds."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from jah.calibration import CalibrationMetrics, CalibrationProfile
from jah.drift import (
    DriftAlert,
    check_prediction_drift,
    load_prediction_rows,
)
from jah.policy import AcceptancePolicy


@pytest.fixture
def sample_profile() -> CalibrationProfile:
    metrics = CalibrationMetrics(
        brier_score=0.20,
        prevalence_brier_score=0.80,
        nll=0.50,
        ece=0.03,
        coverage=0.60,
        accepted_error_rate=0.02,
        accepted_error_95_upper_bound=0.045,
        sample_count=200,
        accepted_count=120,
        errors_in_accepted=2,
        gate_ece_passed=True,
        gate_brier_passed=True,
        gate_selective_passed=True,
        samples_sufficient_for_gate=True,
    )
    return CalibrationProfile(
        profile_id="test-profile-v1",
        workload_id="test-workload-v1",
        primitive="choice",
        artifact_id="test-artifact",
        model_id="Qwen/Qwen3.5-4B",
        revision="test-rev",
        precision="bfloat16",
        prompt_version="decision-prompt-v2",
        label_version="latin-uppercase-bare-v2",
        cardinality_range=(2, 16),
        metrics=metrics,
    )


def test_drift_healthy_predictions(sample_profile) -> None:
    # 120 accepted, 0 errors -> upper bound is 0.0247 < 0.036 (no warnings)
    records = []
    for i in range(120):
        records.append({
            "question_id": f"q_{i}",
            "reference": "a",
            "selected": "a",
            "disposition": "accept",
        })
    for i in range(80):
        records.append({
            "question_id": f"rev_{i}",
            "reference": "a",
            "selected": "b",
            "disposition": "review",
        })

    report = check_prediction_drift(records, sample_profile)
    assert report.total_decisions == 200
    assert report.accepted_count == 120
    assert report.realized_coverage == 0.60
    assert report.errors_in_accepted == 0
    assert report.realized_error_95_upper_bound < 0.03
    assert len(report.alerts) == 0
    assert report.trigger_recalibration is False


def test_drift_warning_when_approaching_bound(sample_profile) -> None:
    # 120 accepted, 1 error -> upper bound is 0.0389 (>= 80% of 0.045 = 0.036)
    records = []
    for i in range(120):
        records.append({
            "question_id": f"q_{i}",
            "reference": "a",
            "selected": "a" if i > 0 else "b",
            "disposition": "accept",
        })
    report = check_prediction_drift(records, sample_profile)
    assert any(a.code == "ERROR_BOUND_WARNING" for a in report.alerts)
    assert report.trigger_recalibration is False


def test_drift_error_bound_breach_triggers_recalibration(sample_profile) -> None:
    # 100 accepted, 8 errors -> upper bound is ~0.13 > 0.045
    records = []
    for i in range(100):
        records.append({
            "question_id": f"q_{i}",
            "reference": "a",
            "selected": "a" if i >= 8 else "b",
            "disposition": "accept",
        })
    for i in range(50):
        records.append({
            "question_id": f"rev_{i}",
            "reference": "a",
            "selected": "b",
            "disposition": "review",
        })

    report = check_prediction_drift(records, sample_profile)
    assert report.errors_in_accepted == 8
    assert report.realized_error_95_upper_bound > 0.05
    assert any(a.code == "ERROR_BOUND_BREACH" for a in report.alerts)
    assert report.trigger_recalibration is True


def test_drift_coverage_below_minimum_triggers_alert(sample_profile) -> None:
    # Coverage is 30/100 = 0.30 < 0.50
    records = []
    for i in range(30):
        records.append({
            "question_id": f"q_{i}",
            "reference": "a",
            "selected": "a",
            "disposition": "accept",
        })
    for i in range(70):
        records.append({
            "question_id": f"rev_{i}",
            "reference": "a",
            "selected": "b",
            "disposition": "review",
        })

    report = check_prediction_drift(records, sample_profile)
    assert report.realized_coverage == 0.30
    assert any(a.code == "COVERAGE_BELOW_MINIMUM" for a in report.alerts)
    assert report.trigger_recalibration is True


def test_load_prediction_rows(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "preds.jsonl"
    jsonl_path.write_text(
        '{"question_id": "q1", "reference": "a", "selected": "a", "disposition": "accept"}\n'
        '{"question_id": "q2", "reference": "b", "selected": "b", "disposition": "review"}\n',
        encoding="utf-8",
    )
    rows = load_prediction_rows(jsonl_path)
    assert len(rows) == 2
    assert rows[0]["question_id"] == "q1"
