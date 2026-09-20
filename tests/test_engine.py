import pytest

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
            decision = ScoredDecision(
                "true",
                {"false": 0.25, "true": 0.75},
                logits={"false": 0.0, "true": 1.0986},
            )
        elif question.primitive == "choice":
            decision = ScoredDecision(
                "b",
                {"a": 0.1, "b": 0.9},
                logits={"a": 0.0, "b": 2.1972},
            )
        else:
            decision = ScoredDecision(
                "high",
                {"low": 0.2, "high": 0.8},
                expected_value=8.0,
                logits={"low": 0.0, "high": 1.3863},
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


def test_engine_with_validated_profile_and_acceptance(validated_metrics) -> None:
    from jah.calibration import CalibrationProfile
    from jah.policy import AcceptancePolicy

    profile_accept = CalibrationProfile(
        profile_id="test-bool-accept",
        workload_id="test-bool",
        primitive="boolean",
        artifact_id="test-artifact",
        model_id="test-model",
        revision="test-rev",
        precision="bf16",
        prompt_version="decision-prompt-v2",
        label_version="latin-uppercase-bare-v2",
        cardinality_range=(2, 2),
        temperature=1.0,
        policy=AcceptancePolicy(type="threshold", threshold=0.70),
        metrics=validated_metrics,
    )
    profile_review = CalibrationProfile(
        profile_id="test-bool-review",
        workload_id="test-bool",
        primitive="boolean",
        artifact_id="test-artifact",
        model_id="test-model",
        revision="test-rev",
        precision="bf16",
        prompt_version="decision-prompt-v2",
        label_version="latin-uppercase-bare-v2",
        cardinality_range=(2, 2),
        temperature=1.0,
        policy=AcceptancePolicy(type="threshold", threshold=0.90),
        metrics=validated_metrics,
    )
    model_metadata = {
        "model_id": "test-model",
        "revision": "test-rev",
        "precision": "bf16",
        "prompt_version": "decision-prompt-v2",
        "label_version": "latin-uppercase-bare-v2",
    }
    config = EngineConfig(
        artifact_id="test-artifact",
        profiles={
            "test-bool-accept": profile_accept,
            "test-bool-review": profile_review,
        },
        model_metadata=model_metadata,
    )
    engine = DecisionEngine(FakeBackend(), config)

    # 1. Evaluate with threshold 0.70 (confidence 0.75 >= 0.70 -> accept)
    req_accept = EvaluateRequest.model_validate(
        {
            "state": "test",
            "questions": {
                "q": {
                    "type": "boolean",
                    "instructions": "test",
                    "proposition": "test prop",
                }
            },
            "profile": "test-bool-accept",
        }
    )
    res_accept = engine.evaluate(req_accept)
    assert res_accept.answers["q"].calibration_status == "validated"
    assert res_accept.answers["q"].disposition == "accept"

    # 2. Evaluate with threshold 0.90 (confidence 0.75 < 0.90 -> review)
    req_review = req_accept.model_copy(update={"profile": "test-bool-review"})
    res_review = engine.evaluate(req_review)
    assert res_review.answers["q"].calibration_status == "validated"
    assert res_review.answers["q"].disposition == "review"

    # 3. Mismatched profile (e.g. primitive mismatch: choice profile for boolean question)
    choice_profile = CalibrationProfile(
        profile_id="test-choice",
        workload_id="test-choice",
        primitive="choice",
        artifact_id="test-artifact",
        model_id="test-model",
        revision="test-rev",
        precision="bf16",
        prompt_version="decision-prompt-v2",
        label_version="latin-uppercase-bare-v2",
        cardinality_range=(2, 4),
        temperature=1.0,
        metrics=validated_metrics,
    )
    config_mismatch = EngineConfig(
        artifact_id="test-artifact",
        profiles={"test-choice": choice_profile},
        model_metadata=model_metadata,
    )
    engine_mismatch = DecisionEngine(FakeBackend(), config_mismatch)
    req_mismatch = req_accept.model_copy(update={"profile": "test-choice"})
    res_mismatch = engine_mismatch.evaluate(req_mismatch)
    assert res_mismatch.answers["q"].calibration_status == "uncalibrated"
    assert res_mismatch.answers["q"].disposition == "review"


@pytest.mark.parametrize("primitive", ["boolean", "choice", "score"])
@pytest.mark.parametrize("evidence_update", [
    None,
    {"gate_ece_passed": False},
    {"gate_brier_passed": False},
    {"gate_selective_passed": False},
    {"samples_sufficient_for_gate": False},
    {"accepted_count": 6, "sample_count": 6},
    {"accepted_error_95_upper_bound": 0.393},
    {"errors_in_accepted": 10},
    {"ece": float("nan")},
    {"brier_score": 0.6},
    {"coverage": 0.49},
    {"sample_count": 300},
])
def test_unvalidated_evidence_forces_review(
    validated_metrics, primitive, evidence_update,
) -> None:
    from jah.calibration import CalibrationProfile
    from jah.policy import AcceptancePolicy

    metadata = {
        "model_id": "test-model",
        "revision": "test-rev",
        "precision": "bf16",
        "prompt_version": "decision-prompt-v2",
        "label_version": "latin-uppercase-bare-v2",
    }
    profile = CalibrationProfile(
        profile_id="unfitted-profile",
        workload_id="test-workload",
        primitive=primitive,
        artifact_id="test-artifact",
        cardinality_range=(2, 2),
        temperature=0.1,
        policy=AcceptancePolicy(threshold=0.5),
        metrics=(
            None if evidence_update is None
            else validated_metrics.model_copy(update=evidence_update)
        ),
        **metadata,
    )
    engine = DecisionEngine(FakeBackend(), EngineConfig(
        artifact_id="test-artifact",
        profiles={profile.profile_id: profile},
        model_metadata=metadata,
    ))
    request = make_request()
    request = request.model_copy(update={"questions": {
        key: question for key, question in request.questions.items() if question.type == primitive
    }})
    response = engine.evaluate(request)
    reference = DecisionEngine(FakeBackend(), EngineConfig(artifact_id="test-artifact"))
    raw_response = reference.evaluate(request)
    assert response.answers == raw_response.answers
    assert all(answer.disposition == "review" for answer in response.answers.values())
    assert all(answer.calibration_status == "uncalibrated" for answer in response.answers.values())
