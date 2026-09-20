"""Strict request contracts shared by the M0 harness and future HTTP API."""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChoiceOption(StrictModel):
    id: Identifier
    description: Annotated[str, Field(min_length=1, max_length=2_000)]


class ScoreLevel(ChoiceOption):
    value: float

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score level values must be finite")
        return value


class QuestionBase(StrictModel):
    instructions: Annotated[str, Field(min_length=1, max_length=4_000)]


class BooleanQuestion(QuestionBase):
    type: Literal["boolean"]
    proposition: Annotated[str, Field(min_length=1, max_length=2_000)]


class ChoiceQuestion(QuestionBase):
    type: Literal["choice"]
    options: Annotated[list[ChoiceOption], Field(min_length=2, max_length=16)]

    @model_validator(mode="after")
    def unique_option_ids(self) -> ChoiceQuestion:
        _require_unique_ids(self.options, "option")
        return self


class ScoreQuestion(QuestionBase):
    type: Literal["score"]
    levels: Annotated[list[ScoreLevel], Field(min_length=2, max_length=16)]

    @model_validator(mode="after")
    def valid_levels(self) -> ScoreQuestion:
        _require_unique_ids(self.levels, "level")
        values = [level.value for level in self.levels]
        if len(set(values)) != len(values):
            raise ValueError("score level values must be unique")
        if values != sorted(values):
            raise ValueError("score levels must be ordered by ascending value")
        return self


Question = Annotated[
    BooleanQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]


class EvaluateRequest(StrictModel):
    state: str | dict[str, Any] | list[Any]
    questions: Annotated[dict[Identifier, Question], Field(min_length=1, max_length=32)]
    profile: Identifier | None = None

    @field_validator("state")
    @classmethod
    def finite_json_state(cls, state: Any) -> Any:
        _reject_non_finite(state)
        return state


Probability = Annotated[float, Field(ge=0.0, le=1.0)]
CalibrationStatus = Literal["uncalibrated", "validated"]
Disposition = Literal["review", "accept"]


class AnswerBase(StrictModel):
    calibration_status: CalibrationStatus
    disposition: Disposition


class BooleanAnswer(AnswerBase):
    type: Literal["boolean"]
    value: bool
    p_true: Probability
    probabilities: dict[Literal["false", "true"], Probability]

    @model_validator(mode="after")
    def valid_boolean_distribution(self) -> BooleanAnswer:
        if set(self.probabilities) != {"false", "true"}:
            raise ValueError("boolean probabilities must contain false and true")
        if not math.isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-5):
            raise ValueError("boolean probabilities must sum to one")
        if not math.isclose(self.p_true, self.probabilities["true"], abs_tol=1e-8):
            raise ValueError("p_true must equal the true probability")
        if self.value != (self.p_true >= 0.5):
            raise ValueError("boolean value must use the 0.5 threshold")
        return self


class ChoiceAnswer(AnswerBase):
    type: Literal["choice"]
    value: Identifier
    probabilities: dict[Identifier, Probability]

    @model_validator(mode="after")
    def valid_choice_distribution(self) -> ChoiceAnswer:
        _validate_distribution(self.value, self.probabilities)
        return self


class ScoreAnswer(AnswerBase):
    type: Literal["score"]
    level_id: Identifier
    expected_value: float
    probabilities: dict[Identifier, Probability]

    @field_validator("expected_value")
    @classmethod
    def finite_expected_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("expected value must be finite")
        return value

    @model_validator(mode="after")
    def valid_score_distribution(self) -> ScoreAnswer:
        _validate_distribution(self.level_id, self.probabilities)
        return self


Answer = Annotated[BooleanAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(StrictModel):
    unique_state_tokens: Annotated[int, Field(ge=0)]
    processed_input_tokens: Annotated[int, Field(ge=0)]
    decisions: Annotated[int, Field(ge=1)]


class Timing(StrictModel):
    queue: Annotated[float, Field(ge=0.0)]
    inference: Annotated[float, Field(ge=0.0)]
    total: Annotated[float, Field(ge=0.0)]


class EvaluateResponse(StrictModel):
    request_id: Identifier
    answers: dict[Identifier, Answer]
    artifact_id: Identifier
    profile_id: Identifier | None
    usage: Usage
    timing_ms: Timing


class HealthResponse(StrictModel):
    status: Literal["live", "ready", "not_ready"]
    artifact_id: Identifier | None = None


def _validate_distribution(selected_id: str, probabilities: dict[str, float]) -> None:
    if not probabilities:
        raise ValueError("probabilities cannot be empty")
    if selected_id not in probabilities:
        raise ValueError("selected ID must appear in probabilities")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
        raise ValueError("probabilities must sum to one")
    maximum = max(probabilities.values())
    if not math.isclose(probabilities[selected_id], maximum, abs_tol=1e-8):
        raise ValueError("selected ID must have maximum probability")


def _require_unique_ids(items: list[ChoiceOption], kind: str) -> None:
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {kind} IDs are not allowed")


def _reject_non_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("state cannot contain non-finite numbers")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _reject_non_finite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_non_finite(item)


def validate_identifier(value: str) -> str:
    """Validate IDs loaded outside Pydantic, such as dataset metadata."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"invalid identifier: {value!r}")
    return value
