"""Frozen workload and M0 dataset loading/validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from jah.schemas import ChoiceOption, ChoiceQuestion, EvaluateRequest, Identifier, StrictModel


class Annotation(StrictModel):
    annotator_id: Identifier
    label: Identifier


class Provenance(StrictModel):
    kind: Literal["synthetic", "production", "challenge"]
    author: Identifier


class M0Example(StrictModel):
    example_id: Identifier
    source_group_id: Identifier
    state: Annotated[str, Field(min_length=1)]
    question: Annotated[str, Field(min_length=1)]
    options: Annotated[list[ChoiceOption], Field(min_length=2, max_length=16)]
    reference_answer: Identifier
    rubric: Annotated[str, Field(min_length=1)]
    task_id: Identifier
    template_id: Identifier
    provenance: Provenance
    annotations: Annotated[list[Annotation], Field(min_length=2)]
    annotation_status: Literal["adjudicated-agreement", "adjudicated-resolution"]

    @model_validator(mode="after")
    def valid_adjudication(self) -> M0Example:
        option_ids = {option.id for option in self.options}
        if self.reference_answer not in option_ids:
            raise ValueError("reference answer must name a supplied option")
        if any(annotation.label not in option_ids for annotation in self.annotations):
            raise ValueError("annotation label must name a supplied option")
        if self.annotation_status == "adjudicated-agreement" and any(
            annotation.label != self.reference_answer for annotation in self.annotations
        ):
            raise ValueError("agreement annotations must match the reference answer")
        return self

    def as_request(self, instructions: str) -> EvaluateRequest:
        return EvaluateRequest(
            state=self.state,
            questions={
                "route": ChoiceQuestion(
                    type="choice",
                    instructions=instructions,
                    options=self.options,
                )
            },
            profile=None,
        )


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a YAML mapping in {path}")
    return value


def load_dataset(path: Path) -> list[M0Example]:
    examples: list[M0Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    examples.append(M0Example.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"invalid dataset row at {path}:{line_number}: {exc}") from exc
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset example IDs must be unique")
    return examples


def validate_dataset_against_workload(examples: list[M0Example], workload: dict) -> None:
    if len(examples) != 50:
        raise ValueError(f"M0 dataset must contain exactly 50 examples; found {len(examples)}")
    frozen_options = workload["options"]
    for example in examples:
        if example.template_id != workload["workload_id"]:
            raise ValueError(f"{example.example_id}: template does not match frozen workload")
        if [option.model_dump() for option in example.options] != frozen_options:
            raise ValueError(f"{example.example_id}: options differ from frozen workload")
    counts = Counter(example.reference_answer for example in examples)
    expected = {option["id"] for option in frozen_options}
    if set(counts) != expected:
        raise ValueError(f"dataset label set {set(counts)} does not match workload {expected}")
    if min(counts.values()) < 10:
        raise ValueError(f"every M0 label needs at least 10 examples; observed {dict(counts)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
