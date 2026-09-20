import pytest
from pydantic import ValidationError

from jah.policy import AcceptancePolicy, evaluate_acceptance, fit_acceptance_threshold


def test_acceptance_policy_validation() -> None:
    policy = AcceptancePolicy(type="threshold", threshold=0.85)
    assert policy.threshold == 0.85
    assert policy.type == "threshold"

    with pytest.raises(ValidationError):
        AcceptancePolicy(threshold=1.5)

    with pytest.raises(ValidationError):
        AcceptancePolicy(threshold=-0.1)


def test_evaluate_acceptance_review_only() -> None:
    policy = AcceptancePolicy(type="review_only")
    disposition = evaluate_acceptance(
        policy,
        {"a": 0.05, "b": 0.95},
        primitive="choice",
    )
    assert disposition == "review"


def test_evaluate_acceptance_threshold() -> None:
    policy = AcceptancePolicy(type="threshold", threshold=0.90)

    # Above threshold -> accept
    disp1 = evaluate_acceptance(policy, {"billing": 0.92, "technical": 0.08}, primitive="choice")
    assert disp1 == "accept"

    # Below threshold -> review
    disp2 = evaluate_acceptance(policy, {"billing": 0.85, "technical": 0.15}, primitive="choice")
    assert disp2 == "review"


def test_evaluate_acceptance_margin_metric() -> None:
    policy = AcceptancePolicy(type="threshold", threshold=0.50, metric="margin")

    # Margin 0.80 - 0.20 = 0.60 >= 0.50 -> accept
    assert evaluate_acceptance(policy, {"a": 0.80, "b": 0.20}, "choice") == "accept"

    # Margin 0.60 - 0.40 = 0.20 < 0.50 -> review
    assert evaluate_acceptance(policy, {"a": 0.60, "b": 0.40}, "choice") == "review"


def test_fit_acceptance_threshold() -> None:
    # 10 samples: confidences and accuracies
    confs = [0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.98, 0.99]
    # At conf <= 0.8, mistakes occur; at conf >= 0.85, all correct
    accs = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    threshold, summary = fit_acceptance_threshold(
        confs, accs, target_error=0.05, min_coverage=0.50
    )
    assert threshold >= 0.85
    assert summary["accepted_error"] == 0.0
    assert summary["coverage"] >= 0.50
