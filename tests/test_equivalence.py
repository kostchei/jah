from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from jah.equivalence import (
    compare_responses,
    load_regression_requests,
    summarize,
)
from jah.schemas import BooleanAnswer, ChoiceAnswer


@dataclass
class FakeResponse:
    answers: dict


def _boolean(p_true: float, *, disposition: str = "review") -> BooleanAnswer:
    return BooleanAnswer(
        type="boolean",
        value=p_true >= 0.5,
        p_true=p_true,
        probabilities={"true": p_true, "false": 1.0 - p_true},
        calibration_status="uncalibrated",
        disposition=disposition,
    )


def _choice(selected: str, probabilities: dict[str, float], *, disposition: str = "review"):
    return ChoiceAnswer(
        type="choice",
        value=selected,
        probabilities=probabilities,
        calibration_status="uncalibrated",
        disposition=disposition,
    )


def test_compare_responses_reports_deviation_and_agreement() -> None:
    reference = FakeResponse({"q1": _boolean(0.80)})
    optimized = FakeResponse({"q1": _boolean(0.803)})
    rows = compare_responses(reference, optimized, request_name="fixture")
    assert rows[0]["argmax_agrees"] is True
    assert rows[0]["maximum_probability_deviation"] == pytest.approx(0.003)
    assert rows[0]["policy_flip"] is False


def test_compare_responses_detects_argmax_divergence() -> None:
    reference = FakeResponse({"q1": _choice("a", {"a": 0.6, "b": 0.4})})
    optimized = FakeResponse({"q1": _choice("b", {"a": 0.4, "b": 0.6})})
    rows = compare_responses(reference, optimized, request_name="fixture")
    assert rows[0]["argmax_agrees"] is False
    assert rows[0]["maximum_probability_deviation"] == pytest.approx(0.2)


def test_compare_responses_detects_policy_flip() -> None:
    reference = FakeResponse({"q1": _boolean(0.9, disposition="accept")})
    optimized = FakeResponse({"q1": _boolean(0.9, disposition="review")})
    rows = compare_responses(reference, optimized, request_name="fixture")
    assert rows[0]["policy_flip"] is True


def test_compare_responses_rejects_divergent_answer_keys() -> None:
    reference = FakeResponse({"q1": _boolean(0.5)})
    optimized = FakeResponse({"q2": _boolean(0.5)})
    with pytest.raises(ValueError, match="answer keys diverged"):
        compare_responses(reference, optimized, request_name="fixture")


def test_compare_responses_rejects_divergent_label_sets() -> None:
    reference = FakeResponse({"q1": _choice("a", {"a": 0.6, "b": 0.4})})
    optimized = FakeResponse({"q1": _choice("a", {"a": 0.6, "c": 0.4})})
    with pytest.raises(ValueError, match="label sets diverged"):
        compare_responses(reference, optimized, request_name="fixture")


def test_summarize_applies_the_specification_gates() -> None:
    rows = [
        {"argmax_agrees": True, "maximum_probability_deviation": 0.001, "policy_flip": False}
        for _ in range(200)
    ]
    summary = summarize(rows)
    assert summary["argmax_agreement"] == 1.0
    assert summary["gate_passed"] is True

    rows[0]["argmax_agrees"] = False
    rows[1]["argmax_agrees"] = False
    degraded = summarize(rows)
    assert degraded["argmax_agreement"] == pytest.approx(0.99)
    assert degraded["gate_argmax_passed"] is False
    assert degraded["gate_passed"] is False


def test_summarize_fails_on_excess_probability_deviation() -> None:
    rows = [{"argmax_agrees": True, "maximum_probability_deviation": 0.02, "policy_flip": False}]
    summary = summarize(rows)
    assert summary["gate_deviation_passed"] is False
    assert summary["gate_passed"] is False


def test_summarize_rejects_an_empty_comparison() -> None:
    with pytest.raises(ValueError, match="no compared decisions"):
        summarize([])


def test_regression_set_rejects_single_question_requests(tmp_path: Path) -> None:
    path = tmp_path / "single.json"
    path.write_text(
        json.dumps(
            {
                "state": "example",
                "questions": {
                    "q1": {"type": "boolean", "instructions": "Judge it.", "proposition": "True."}
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multi-question requests"):
        load_regression_requests(tmp_path)


def test_regression_set_requires_at_least_one_request(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no regression requests"):
        load_regression_requests(tmp_path)


def test_committed_regression_set_is_multi_question_and_covers_every_primitive() -> None:
    root = Path(__file__).resolve().parents[1]
    loaded = load_regression_requests(root / "evals" / "fixtures" / "equivalence")
    assert len(loaded) >= 5
    primitives = {
        question.type for _, _, request in loaded for question in request.questions.values()
    }
    assert primitives == {"boolean", "choice", "score"}
    assert max(len(request.questions) for _, _, request in loaded) >= 16
