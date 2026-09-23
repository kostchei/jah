"""CI determinism and optimization equivalence suite per ADR-03 / ADR-07.

Verifies:
1. Step 2B margin-aware equivalence gate logic:
   - Hard decision equivalence (zero flips) on decisions with top-1 logit margin > 2 ULPs (0.125).
   - Bounded probability equivalence (max Δp <= 0.01) outside the tie band.
   - Tie-band population inside margin <= 0.125 is reported and counted, not gated as a failure.
   - Policy flips are gated at zero across all decisions.
2. GPU determinism across batch shapes (batch sizes 1, 8, 16) and prefix KV cache reuse.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from jah.equivalence import (
    DEFAULT_MARGIN_THRESHOLD,
    MAXIMUM_PROBABILITY_DEVIATION_GATE,
    compare_responses,
    summarize,
)
from jah.schemas import BooleanAnswer, ChoiceAnswer


class FakeResponse:
    def __init__(self, answers: dict) -> None:
        self.answers = answers


class FakeDecision:
    def __init__(self, logits: dict[str, float]) -> None:
        self.logits = logits


class FakeMeasurement:
    def __init__(self, logits: dict[str, float]) -> None:
        self.decision = FakeDecision(logits)


def _choice(selected: str, probs: dict[str, float], disposition: str = "review") -> ChoiceAnswer:
    return ChoiceAnswer(
        type="choice",
        value=selected,
        probabilities=probs,
        calibration_status="uncalibrated",
        disposition=disposition,
    )


def _boolean(p_true: float, disposition: str = "review") -> BooleanAnswer:
    return BooleanAnswer(
        type="boolean",
        value=p_true >= 0.5,
        p_true=p_true,
        probabilities={"true": p_true, "false": 1.0 - p_true},
        calibration_status="uncalibrated",
        disposition=disposition,
    )


def test_tie_band_flip_is_diagnostic_but_still_fails_gate() -> None:
    """Tie-band classification does not exempt a disagreement from release gates."""
    # Near-tied decision: reference selects "a", optimized selects "b".
    ref = FakeResponse({"q_tie": _choice("a", {"a": 0.51, "b": 0.49})})
    opt = FakeResponse({"q_tie": _choice("b", {"a": 0.48, "b": 0.52})})
    ref_meas = {"q_tie": FakeMeasurement({"a": 10.05, "b": 10.00})}

    rows = compare_responses(
        ref, opt, request_name="test_tie",
        reference_measurements=ref_meas,
        margin_threshold=0.125,
    )
    assert len(rows) == 1
    assert rows[0]["in_tie_band"] is True
    assert rows[0]["argmax_agrees"] is False

    summary = summarize(rows, margin_threshold=0.125)
    assert summary["tie_band_count"] == 1
    assert summary["tie_band_fraction"] == 1.0
    assert summary["gated_decisions"] == 1
    assert summary["gate_argmax_passed"] is False
    assert summary["gate_deviation_passed"] is False
    assert summary["gate_passed"] is False


def test_margin_aware_gate_fails_flip_outside_tie_band() -> None:
    """Decisions outside the tie band (margin > 0.125) must have 100% argmax agreement."""
    # Clear decision with margin = 1.0 > 0.125: flip is a hard failure
    ref = FakeResponse({"q_clear": _choice("a", {"a": 0.73, "b": 0.27})})
    opt = FakeResponse({"q_clear": _choice("b", {"a": 0.45, "b": 0.55})})
    ref_meas = {"q_clear": FakeMeasurement({"a": 11.0, "b": 10.0})}

    rows = compare_responses(
        ref, opt, request_name="test_clear",
        reference_measurements=ref_meas,
        margin_threshold=0.125,
    )
    assert len(rows) == 1
    assert rows[0]["in_tie_band"] is False
    assert rows[0]["argmax_agrees"] is False

    summary = summarize(rows, margin_threshold=0.125)
    assert summary["tie_band_count"] == 0
    assert summary["gated_decisions"] == 1
    assert summary["gate_argmax_passed"] is False
    assert summary["gate_passed"] is False


def test_margin_aware_gate_enforces_bounded_probability_deviation() -> None:
    """Decisions outside tie band must satisfy max Δp <= 0.01."""
    ref = FakeResponse({"q1": _boolean(0.80)})
    opt = FakeResponse({"q1": _boolean(0.82)})  # Δp = 0.02 > 0.01
    ref_meas = {"q1": FakeMeasurement({"true": 12.0, "false": 10.0})}

    rows = compare_responses(
        ref, opt, request_name="test_dev",
        reference_measurements=ref_meas,
        margin_threshold=0.125,
    )
    assert rows[0]["in_tie_band"] is False
    assert rows[0]["argmax_agrees"] is True

    summary = summarize(rows, margin_threshold=0.125)
    assert summary["gate_argmax_passed"] is True
    assert summary["gate_deviation_passed"] is False
    assert summary["gate_passed"] is False


def test_policy_flip_always_fails_the_gate() -> None:
    """A policy flip (accept <-> review) fails even if inside tie band."""
    ref = FakeResponse({"q1": _choice("a", {"a": 0.51, "b": 0.49}, disposition="accept")})
    opt = FakeResponse({"q1": _choice("b", {"a": 0.48, "b": 0.52}, disposition="review")})
    ref_meas = {"q1": FakeMeasurement({"a": 10.02, "b": 10.00})}

    rows = compare_responses(
        ref, opt, request_name="test_policy",
        reference_measurements=ref_meas,
        margin_threshold=0.125,
    )
    summary = summarize(rows, margin_threshold=0.125)
    assert summary["policy_flips"] == 1
    assert summary["gate_policy_passed"] is False
    assert summary["gate_passed"] is False


@pytest.fixture(scope="module")
def model_config() -> dict:
    from jah.workload import load_yaml
    root = Path(__file__).resolve().parents[1]
    return load_yaml(root / "configs" / "models" / "qwen3.5-4b.yaml")


@pytest.fixture(scope="module")
def backend(model_config: dict):
    from jah.evaluation import load_backend
    return load_backend("direct", model_config)


# ---------------------------------------------------------------------------
# Real Backbone GPU Determinism Sweep (Step 2C)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("microbatch_size", [1, 8, 16])
@pytest.mark.parametrize("use_prefix_cache", [False, True])
def test_gpu_determinism_sweep(backend, model_config, microbatch_size: int, use_prefix_cache: bool) -> None:
    """ADR-03 / ADR-07 determinism sweep on real backbone across batch shapes and prefix reuse."""
    import torch
    from jah.engine import DecisionEngine, EngineConfig
    from jah.schemas import EvaluateRequest

    assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, (
        "FP32 reduction accumulation is required per ADR-07"
    )
    assert not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, (
        "FP32 reduction accumulation is required per ADR-07"
    )

    root = Path(__file__).resolve().parents[1]
    fixture_path = root / "evals" / "fixtures" / "equivalence" / "wikiqa-boolean-08.json"
    request = EvaluateRequest.model_validate_json(fixture_path.read_text(encoding="utf-8"))

    base_config = {
        "artifact_id": model_config["artifact_id"],
        "maximum_input_tokens": model_config["maximum_input_tokens"],
        "model_metadata": {
            "model_id": model_config["model_id"],
            "revision": model_config["revision"],
            "precision": model_config["precision"],
            "prompt_version": model_config["prompt_version"],
            "label_version": model_config["label_version"],
        },
    }

    ref_engine = DecisionEngine(backend, EngineConfig(**base_config, force_sequential=True, enable_prefix_cache=False))
    opt_engine = DecisionEngine(
        backend,
        EngineConfig(**base_config, force_sequential=False, enable_prefix_cache=use_prefix_cache, microbatch_size=microbatch_size),
    )

    ref_resp, ref_meas = ref_engine.evaluate_detailed(request, request_id="gpu_ref")
    opt_resp, opt_meas = opt_engine.evaluate_detailed(request, request_id="gpu_opt")

    rows = compare_responses(
        ref_resp, opt_resp,
        request_name="wikiqa-boolean-08",
        reference_measurements=ref_meas,
        optimized_measurements=opt_meas,
        margin_threshold=DEFAULT_MARGIN_THRESHOLD,
    )
    summary = summarize(rows, margin_threshold=DEFAULT_MARGIN_THRESHOLD)

    assert summary["gate_passed"] is True, f"Determinism gate failed under batch={microbatch_size}, prefix={use_prefix_cache}: {summary}"
