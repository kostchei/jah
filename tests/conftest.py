import pytest

from jah.calibration import CalibrationMetrics


@pytest.fixture
def validated_metrics() -> CalibrationMetrics:
    """Evidence for 100 accepted, error-free observations out of 100."""
    return CalibrationMetrics(
        brier_score=0.01,
        prevalence_brier_score=0.5,
        nll=0.01,
        ece=0.01,
        coverage=1.0,
        accepted_error_rate=0.0,
        accepted_error_95_upper_bound=0.029514,
        sample_count=100,
        accepted_count=100,
        errors_in_accepted=0,
        gate_ece_passed=True,
        gate_brier_passed=True,
        gate_selective_passed=True,
        samples_sufficient_for_gate=True,
    )
