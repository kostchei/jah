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
