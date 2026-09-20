from jah.backends.huggingface import InferenceMeasurement
from jah.engine import DecisionEngine, EngineConfig
from jah.errors import InferenceUnavailableError
from jah.schemas import EvaluateRequest
from jah.scoring import ScoredDecision


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        labels = {label: 100 + index for index, label in enumerate("ABCDEFGHIJKLMNOP")}
        if text and text[-1] in labels:
            return [ord(char) for char in text[:-1]] + [labels[text[-1]]]
        return [ord(char) for char in text]


class FakeBackend:
    tokenizer = FakeTokenizer()

    def score(self, question):
        if question.primitive == "boolean":
            decision = ScoredDecision("true", {"false": 0.25, "true": 0.75})
        elif question.primitive == "choice":
            decision = ScoredDecision("b", {"a": 0.1, "b": 0.9})
        else:
            decision = ScoredDecision(
                "high", {"low": 0.2, "high": 0.8}, expected_value=8.0
            )
        return InferenceMeasurement(
            decision=decision,
            label_mass=0.9,
            inference_ms=2.0,
            input_tokens=10,
            peak_vram_bytes=0,
        )


def make_request() -> EvaluateRequest:
    return EvaluateRequest.model_validate(
        {
            "state": {"message": "hello"},
            "questions": {
                "relevant": {
                    "type": "boolean",
                    "instructions": "Decide relevance.",
                    "proposition": "The message is relevant.",
                },
                "route": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "options": [
                        {"id": "a", "description": "First"},
                        {"id": "b", "description": "Second"},
                    ],
                },
                "quality": {
                    "type": "score",
                    "instructions": "Score.",
                    "levels": [
                        {"id": "low", "description": "Low", "value": 0},
                        {"id": "high", "description": "High", "value": 10},
                    ],
                },
            },
            "profile": "unfitted-profile",
        }
    )


def test_engine_returns_all_typed_answers_as_uncalibrated_review() -> None:
    engine = DecisionEngine(FakeBackend(), EngineConfig(artifact_id="test-artifact"))
    response = engine.evaluate(make_request(), request_id="req_test")

    assert response.answers["relevant"].value is True
    assert response.answers["relevant"].p_true == 0.75
    assert response.answers["route"].value == "b"
    assert response.answers["quality"].level_id == "high"
    assert response.answers["quality"].expected_value == 8.0
    assert {answer.calibration_status for answer in response.answers.values()} == {"uncalibrated"}
    assert {answer.disposition for answer in response.answers.values()} == {"review"}
    assert response.usage.decisions == 3
    assert response.usage.processed_input_tokens == 30
    assert response.profile_id == "unfitted-profile"


def test_engine_is_atomic_when_a_question_fails() -> None:
    class FailingBackend(FakeBackend):
        calls = 0

        def score(self, question):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("inference failed")
            return super().score(question)

    engine = DecisionEngine(FailingBackend(), EngineConfig(artifact_id="test-artifact"))
    try:
        engine.evaluate(make_request())
    except InferenceUnavailableError as exc:
        assert str(exc) == "inference failed before the atomic response"
    else:
        raise AssertionError("expected the request to fail atomically")
