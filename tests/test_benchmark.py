import pytest

from jah.benchmark import validate_benchmark_shape
from jah.schemas import EvaluateRequest


def test_latency_protocol_requires_boolean_shape() -> None:
    request = EvaluateRequest.model_validate(
        {
            "state": "x",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "options": [
                        {"id": "a", "description": "A"},
                        {"id": "b", "description": "B"},
                    ],
                }
            },
        }
    )
    with pytest.raises(ValueError, match="Boolean"):
        validate_benchmark_shape(request, expected_questions=1)


def test_latency_protocol_requires_exact_question_count() -> None:
    request = EvaluateRequest.model_validate(
        {
            "state": "x",
            "questions": {
                "check": {
                    "type": "boolean",
                    "instructions": "Check.",
                    "proposition": "X.",
                }
            },
        }
    )
    with pytest.raises(ValueError, match="requires 16"):
        validate_benchmark_shape(request, expected_questions=16)
