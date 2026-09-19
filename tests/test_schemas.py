import math

import pytest
from pydantic import ValidationError

from jah.schemas import EvaluateRequest


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
