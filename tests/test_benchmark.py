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


def test_protocol_minimums_are_enforced_without_the_smoke_flag() -> None:
    from jah.benchmark import parse_args, run

    args = parse_args(["--request", "evals/fixtures/benchmark-2048-16b.json"])
    args.warmup_requests = 5
    args.measured_requests = 15
    with pytest.raises(ValueError, match="at least 20 warmup requests"):
        run(args)

    args.warmup_requests = 20
    with pytest.raises(ValueError, match="at least 200 measured requests"):
        run(args)


def test_default_protocol_matches_the_specification_minimums() -> None:
    from jah.benchmark import (
        REQUIRED_MEASURED_REQUESTS,
        REQUIRED_WARMUP_REQUESTS,
        parse_args,
    )

    args = parse_args(["--request", "evals/fixtures/benchmark-2048-16b.json"])
    assert args.warmup_requests == REQUIRED_WARMUP_REQUESTS == 20
    assert args.measured_requests == REQUIRED_MEASURED_REQUESTS == 200
    assert args.smoke is False


def test_recorded_benchmark_artifact_declares_protocol_compliance() -> None:
    """A smoke-sized run must never be published as a passing latency gate."""
    import json
    from pathlib import Path

    for path in sorted(Path("artifacts").rglob("http-latency.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        protocol = document["protocol"]
        assert "protocol_compliant" in protocol, f"{path} predates protocol compliance recording"
        if document.get("gate_passed"):
            assert protocol["protocol_compliant"] is True
            assert protocol["warmup_requests"] >= 20
            assert protocol["measured_requests"] >= 200
