import math

import pytest
from pydantic import ValidationError

from jah.schemas import BooleanAnswer, EvaluateRequest


def test_choice_request_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate option IDs"):
        EvaluateRequest.model_validate(
            {
                "state": "hello",
                "questions": {
                    "route": {
                        "type": "choice",
                        "instructions": "Choose.",
                        "options": [
                            {"id": "same", "description": "One"},
                            {"id": "same", "description": "Two"},
                        ],
                    }
                },
            }
        )


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_request_rejects_non_finite_state(bad_value: float) -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        EvaluateRequest.model_validate(
            {
                "state": {"value": bad_value},
                "questions": {
                    "answer": {
                        "type": "boolean",
                        "instructions": "Decide.",
                        "proposition": "The value is present.",
                    }
                },
            }
        )


def test_score_levels_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="ascending"):
        EvaluateRequest.model_validate(
            {
                "state": "x",
                "questions": {
                    "quality": {
                        "type": "score",
                        "instructions": "Score quality.",
                        "levels": [
                            {"id": "high", "description": "High", "value": 1},
                            {"id": "low", "description": "Low", "value": 0},
                        ],
                    }
                },
            }
        )


def test_boolean_answer_enforces_complement_and_threshold() -> None:
    answer = BooleanAnswer(
        type="boolean",
        value=True,
        p_true=0.7,
        probabilities={"false": 0.3, "true": 0.7},
        calibration_status="uncalibrated",
        disposition="review",
    )
    assert answer.value is True

    with pytest.raises(ValidationError, match="0.5 threshold"):
        BooleanAnswer(
            type="boolean",
            value=False,
            p_true=0.7,
            probabilities={"false": 0.3, "true": 0.7},
            calibration_status="uncalibrated",
            disposition="review",
        )
