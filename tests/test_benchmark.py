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


def test_benchmark_fixture_file_has_valid_shape() -> None:
    from pathlib import Path

    fixture_path = Path("evals/fixtures/benchmark-2048-16b.json")
    assert fixture_path.exists(), "Benchmark fixture file does not exist"
    request = EvaluateRequest.model_validate_json(fixture_path.read_text(encoding="utf-8"))
    validate_benchmark_shape(request, expected_questions=16)
    assert len(request.questions) == 16
