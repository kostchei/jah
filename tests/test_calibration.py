import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from jah.calibration import (
    CalibrationProfile,
    check_profile_compatibility,
    compute_binomial_upper_bound,
    compute_brier_score,
    compute_ece,
    compute_nll,
    compute_prevalence_brier,
    compute_selective_metrics,
    fit_temperature_scaling,
    has_validated_evidence,
    load_profile,
    load_profile_registry,
    save_profile,
)
from jah.policy import AcceptancePolicy


def test_temperature_validation() -> None:
    with pytest.raises(ValidationError):
        CalibrationProfile(
            profile_id="test",
            workload_id="test",
            primitive="choice",
            artifact_id="art",
            model_id="mod",
            revision="rev",
            precision="bf16",
            prompt_version="v1",
            label_version="v1",
            cardinality_range=(2, 4),
            temperature=-0.5,
        )


def test_fit_temperature_scaling_softens_overconfidence() -> None:
    # Overconfident predictions that are 50% wrong
    logits_list = [
        [5.0, -5.0],
        [-5.0, 5.0],
        [5.0, -5.0],
        [-5.0, 5.0],
    ]
    # Targets: only 2 correct
    targets = [0, 1, 1, 0]  # half match, half mismatch
    t = fit_temperature_scaling(logits_list, targets)
    assert t > 1.5, f"expected T > 1.5 to soften overconfident logits; got {t}"


def test_fit_temperature_scaling_sharpens_underconfidence() -> None:
    # Weak logits that are 100% correct
    logits_list = [
        [0.2, -0.2],
        [-0.2, 0.2],
        [0.3, -0.3],
        [-0.3, 0.3],
    ]
    targets = [0, 1, 0, 1]  # all correct
    t = fit_temperature_scaling(logits_list, targets)
    assert t < 1.0, f"expected T < 1.0 to sharpen underconfident logits; got {t}"


def test_compute_binomial_upper_bound() -> None:
    # Analytical known case: k=0, n=59 -> 1 - 0.05^(1/59) ≈ 0.049507... <= 0.05
    bound_59 = compute_binomial_upper_bound(0, 59, confidence=0.95)
    assert math.isclose(bound_59, 1.0 - 0.05 ** (1 / 59), rel_tol=1e-5)
    assert bound_59 <= 0.05

    # k=0, n=1 -> 0.95
    assert math.isclose(compute_binomial_upper_bound(0, 1, confidence=0.95), 0.95, rel_tol=1e-5)

    # Edge cases
    assert compute_binomial_upper_bound(0, 0) == 1.0
    assert compute_binomial_upper_bound(5, 5) == 1.0


def test_calibration_metrics() -> None:
    # Perfect predictions
    probs = [[1.0, 0.0], [0.0, 1.0]]
    targets = [0, 1]
    assert compute_brier_score(probs, targets) == 0.0
    assert compute_nll(probs, targets) < 1e-6

    # ECE with perfect predictions
    ece, bins = compute_ece([1.0, 1.0], [1, 1], num_bins=2)
    assert ece == 0.0
    assert len(bins) >= 1

    # Prevalence Brier score for balanced binary classes
    prev_brier = compute_prevalence_brier([0, 1], num_classes=2)
    assert prev_brier == 0.5


def test_compute_selective_metrics() -> None:
    confs = [0.95, 0.92, 0.88, 0.70]
    accs = [1, 1, 1, 0]
    # At threshold 0.85 -> 3 accepted, 0 errors
    metrics = compute_selective_metrics(confs, accs, threshold=0.85)
    assert metrics["accepted_count"] == 3
    assert metrics["coverage"] == 0.75
    assert metrics["accepted_error_rate"] == 0.0
    assert metrics["accepted_error_95_upper_bound"] > 0.0


def test_check_profile_compatibility() -> None:
    profile = CalibrationProfile(
        profile_id="doc-rel-v1",
        workload_id="document-relevance-v1",
        primitive="boolean",
        artifact_id="qwen3.5-4b-m0",
        model_id="Qwen/Qwen3.5-4B",
        revision="851bf",
        precision="bfloat16",
        prompt_version="decision-prompt-v2",
        label_version="latin-uppercase-bare-v2",
        cardinality_range=(2, 2),
        temperature=1.2,
    )

    # Compatible match
    assert check_profile_compatibility(
        profile,
        primitive="boolean",
        option_count=2,
        artifact_id="qwen3.5-4b-m0",
        model_id="Qwen/Qwen3.5-4B",
        revision="851bf",
        precision="bfloat16",
        prompt_version="decision-prompt-v2",
        label_version="latin-uppercase-bare-v2",
    ) is True

    # Primitive mismatch
    assert check_profile_compatibility(
        profile,
        primitive="choice",
        option_count=2,
    ) is False

    # Cardinality mismatch
    assert check_profile_compatibility(
        profile,
        primitive="boolean",
        option_count=4,
    ) is False

    # Prompt version mismatch
    assert check_profile_compatibility(
        profile,
        primitive="boolean",
        option_count=2,
        prompt_version="decision-prompt-v1",
    ) is False


def test_profile_yaml_roundtrip(tmp_path: Path) -> None:
    profile = CalibrationProfile(
        profile_id="routing-v1",
        workload_id="support-routing-v1",
        primitive="choice",
        artifact_id="art-1",
        model_id="mod-1",
        revision="rev-1",
        precision="bfloat16",
        prompt_version="prompt-v2",
        label_version="label-v2",
        cardinality_range=(2, 16),
        temperature=1.05,
        policy=AcceptancePolicy(type="threshold", threshold=0.88),
    )
    profile_path = tmp_path / "routing-v1.yaml"
    save_profile(profile, profile_path)

    loaded = load_profile(profile_path)
    assert loaded == profile

    registry = load_profile_registry(tmp_path)
    assert "routing-v1" in registry
    assert registry["routing-v1"].temperature == 1.05


def test_profile_validation_status() -> None:
    registry = load_profile_registry(
        Path(__file__).resolve().parents[1] / "configs/profiles/public"
    )
    assert registry
    assert has_validated_evidence(registry["banking77-16-intent-v1"])
    assert has_validated_evidence(registry["wikiqa-answer-relevance-v1"])
    assert not has_validated_evidence(registry["asap2-source-essay-v1"])


def test_retired_profiles_are_outside_every_default_registry() -> None:
    """The degenerate M2 profiles must not be reachable from a default profiles directory."""
    root = Path(__file__).resolve().parents[1]
    retired = load_profile_registry(root / "configs/profiles/retired")
    assert set(retired) == {
        "document-relevance-v1",
        "rubric-assessment-v1",
        "support-routing-v1",
    }
    for directory in ("configs/profiles", "configs/profiles/public"):
        loaded = load_profile_registry(root / directory)
        assert not set(loaded) & set(retired), f"{directory} still exposes a retired profile"
