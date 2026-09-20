"""Reusable request-to-response decision engine."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from jah.backends.huggingface import InferenceMeasurement
from jah.compiler import CompiledQuestion, canonicalize_state, compile_request
from jah.errors import InferenceUnavailableError
from jah.schemas import (
    BooleanAnswer,
    ChoiceAnswer,
    EvaluateRequest,
    EvaluateResponse,
    ScoreAnswer,
    Timing,
    Usage,
)


class ScoringBackend(Protocol):
    tokenizer: object

    def score(self, question: CompiledQuestion) -> InferenceMeasurement: ...


@dataclass(frozen=True)
class EngineConfig:
    artifact_id: str
    maximum_input_tokens: int = 8_192


class DecisionEngine:
    """Compile and score an entire request atomically on the reference path."""

    def __init__(self, backend: ScoringBackend, config: EngineConfig) -> None:
        self.backend = backend
        self.config = config

    def evaluate(self, request: EvaluateRequest, *, request_id: str | None = None) -> EvaluateResponse:
        started = time.perf_counter()
        compiled = compile_request(
            request,
            tokenizer=self.backend.tokenizer,
            max_input_tokens=self.config.maximum_input_tokens,
        )

        # Score sequentially for the M1 correctness reference. The current backbone showed
        # prediction changes under naive padding/batching during M0 diagnostics.
        measurements = []
        try:
            for question in compiled:
                measurements.append(self.backend.score(question))
        except InferenceUnavailableError:
            raise
        except Exception as exc:
            raise InferenceUnavailableError("inference failed before the atomic response") from exc
        answers = {}
        for question, measurement in zip(compiled, measurements, strict=True):
            common = {"calibration_status": "uncalibrated", "disposition": "review"}
            decision = measurement.decision
            if question.primitive == "boolean":
                p_true = decision.probabilities["true"]
                answers[question.question_id] = BooleanAnswer(
                    type="boolean",
                    value=p_true >= 0.5,
                    p_true=p_true,
                    probabilities=decision.probabilities,
                    **common,
                )
            elif question.primitive == "choice":
                answers[question.question_id] = ChoiceAnswer(
                    type="choice",
                    value=decision.selected_id,
                    probabilities=decision.probabilities,
                    **common,
                )
            elif question.primitive == "score":
                if decision.expected_value is None:
                    raise RuntimeError("score backend did not return an expected value")
                answers[question.question_id] = ScoreAnswer(
                    type="score",
                    level_id=decision.selected_id,
                    expected_value=decision.expected_value,
                    probabilities=decision.probabilities,
                    **common,
                )
            else:  # pragma: no cover - compiler owns the primitive set
                raise RuntimeError(f"unsupported compiled primitive: {question.primitive}")

        tokenizer = self.backend.tokenizer
        state_tokens = len(
            tokenizer.encode(canonicalize_state(request.state), add_special_tokens=False)
        )
        inference_ms = sum(item.inference_ms for item in measurements)
        total_ms = (time.perf_counter() - started) * 1_000
        return EvaluateResponse(
            request_id=request_id or f"req_{uuid.uuid4().hex}",
            answers=answers,
            artifact_id=self.config.artifact_id,
            profile_id=request.profile,
            usage=Usage(
                unique_state_tokens=state_tokens,
                processed_input_tokens=sum(item.input_tokens for item in measurements),
                decisions=len(measurements),
            ),
            timing_ms=Timing(queue=0.0, inference=inference_ms, total=total_ms),
        )
