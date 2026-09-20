"""M1 dataset contracts, leakage-safe deterministic splits, and readiness gates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from jah.schemas import BooleanQuestion, ChoiceQuestion, EvaluateRequest, Identifier, ScoreQuestion
from jah.workload import Provenance, StrictModel, sha256_file

SplitName = Literal["train", "development", "calibration", "locked_test", "task_holdout"]


class DecisionAnnotation(StrictModel):
    annotator_id: Identifier
    annotator_kind: Literal["human", "model", "unknown"] = "unknown"
    answers: dict[str, str]
    independent: bool = False


class M1Example(StrictModel):
    example_id: Identifier
    source_group_id: Identifier
    near_duplicate_cluster_id: Identifier
    workload_id: Identifier
    task_id: Identifier
    template_id: Identifier
    request: EvaluateRequest
    reference_answers: dict[str, str]
    rubric: Annotated[str, Field(min_length=1)]
    provenance: Provenance
    annotations: list[DecisionAnnotation]
    annotation_status: Literal[
        "pending-independent-review",
        "model-adjudicated",
        "adjudicated-agreement",
        "adjudicated-resolution",
        "excluded",
    ]
    adjudicator_id: Identifier | None = None
    task_template_holdout: bool = False

    @model_validator(mode="after")
    def valid_references_and_review(self) -> M1Example:
        question_ids = set(self.request.questions)
        if set(self.reference_answers) != question_ids:
            raise ValueError("reference answers must match request question IDs exactly")
        for question_id, question in self.request.questions.items():
            answer = self.reference_answers[question_id]
            if isinstance(question, BooleanQuestion):
                allowed = {"false", "true"}
            elif isinstance(question, ChoiceQuestion):
                allowed = {option.id for option in question.options}
            elif isinstance(question, ScoreQuestion):
                allowed = {level.id for level in question.levels}
            else:  # pragma: no cover
                raise TypeError(f"unsupported question type: {type(question)!r}")
            if answer not in allowed:
                raise ValueError(f"reference answer for {question_id!r} is not an allowed option")

        if self.annotation_status.startswith("adjudicated-"):
            independent_humans = {
                annotation.annotator_id
                for annotation in self.annotations
                if annotation.annotator_kind == "human" and annotation.independent
            }
            if len(independent_humans) < 2:
                raise ValueError(
                    "adjudicated rows require two distinct independent human annotators"
                )
            for annotation in self.annotations:
                if set(annotation.answers) != question_ids:
                    raise ValueError("annotation answers must match request question IDs exactly")
            if self.annotation_status == "adjudicated-agreement" and any(
                annotation.answers != self.reference_answers
                for annotation in self.annotations
                if annotation.annotator_kind == "human" and annotation.independent
            ):
                raise ValueError("agreement annotations must match the reference answers")
            if self.annotation_status == "adjudicated-resolution" and self.adjudicator_id is None:
                raise ValueError("resolved disagreements require an adjudicator ID")
        elif self.annotation_status == "model-adjudicated":
            if not self.annotations:
                raise ValueError("model-adjudicated rows require at least one annotation")
            for annotation in self.annotations:
                if set(annotation.answers) != question_ids:
                    raise ValueError("annotation answers must match request question IDs exactly")
        return self

    @property
    def decisions(self) -> int:
        return len(self.request.questions)


def load_m1_dataset(path: Path) -> list[M1Example]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(M1Example.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid dataset row at {path}:{line_number}: {exc}") from exc
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset example IDs must be unique")
    return examples


def _ordinary_split(identity: str, *, seed: str) -> SplitName:
    identity = f"{seed}\0{identity}"
    bucket = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big") % 100
    if bucket < 60:
        return "train"
    if bucket < 75:
        return "development"
    if bucket < 85:
        return "calibration"
    return "locked_test"


def split_assignments(examples: list[M1Example], *, seed: str) -> dict[str, SplitName]:
    """Keep every source and near-duplicate connected component in one split."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for example in examples:
        union(f"source:{example.source_group_id}", f"cluster:{example.near_duplicate_cluster_id}")

    components: dict[str, list[M1Example]] = {}
    for example in examples:
        root = find(f"source:{example.source_group_id}")
        components.setdefault(root, []).append(example)

    assignments: dict[str, SplitName] = {}
    for component in components.values():
        holdout_values = {example.task_template_holdout for example in component}
        if len(holdout_values) != 1:
            raise ValueError("a connected source/cluster component mixes holdout and ordinary rows")
        if True in holdout_values:
            split: SplitName = "task_holdout"
        else:
            component_identity = "\0".join(
                sorted(
                    {f"source:{item.source_group_id}" for item in component}
                    | {f"cluster:{item.near_duplicate_cluster_id}" for item in component}
                )
            )
            split = _ordinary_split(component_identity, seed=seed)
        assignments.update({example.example_id: split for example in component})
    return assignments


def validate_m1_suite(
    examples: list[M1Example],
    *,
    minimum_decisions: int = 1_000,
    allowed_statuses: tuple[str, ...] | None = None,
) -> dict:
    if allowed_statuses is None:
        allowed_statuses = (
            "adjudicated-agreement",
            "adjudicated-resolution",
            "model-adjudicated",
        )
    included = [example for example in examples if example.annotation_status != "excluded"]
    decisions = sum(example.decisions for example in included)
    primitive_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for example in included:
        status_counts[example.annotation_status] += example.decisions
        primitive_counts.update(question.type for question in example.request.questions.values())
    failures = []
    if decisions < minimum_decisions:
        failures.append(f"needs at least {minimum_decisions} decisions; found {decisions}")
    for primitive in ("choice", "boolean", "score"):
        if primitive_counts[primitive] == 0:
            failures.append(f"suite contains no {primitive} decisions")
    disallowed = sum(
        example.decisions
        for example in included
        if example.annotation_status not in allowed_statuses
    )
    if disallowed:
        failures.append(
            f"{disallowed} decisions do not have an approved annotation status: "
            f"allowed {sorted(allowed_statuses)}"
        )
    return {
        "ready": not failures,
        "examples": len(included),
        "decisions": decisions,
        "primitive_counts": dict(sorted(primitive_counts.items())),
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "failures": failures,
    }


def build_split_manifest(
    examples: list[M1Example],
    *,
    dataset_path: Path,
    seed: str,
    dataset_label: str | None = None,
) -> dict:
    assignments = split_assignments(examples, seed=seed)
    split_decisions: Counter[str] = Counter()
    split_examples: Counter[str] = Counter(assignments.values())
    group_splits: dict[tuple[str, str], set[str]] = {}
    for example in examples:
        split = assignments[example.example_id]
        split_decisions[split] += example.decisions
        group = (example.source_group_id, example.near_duplicate_cluster_id)
        group_splits.setdefault(group, set()).add(split)
    leaking = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"source/cluster leakage detected: {leaking[:5]}")
    assignment_hash = hashlib.sha256(
        json.dumps(assignments, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "dataset": dataset_label or str(dataset_path).replace("\\", "/"),
        "dataset_sha256": sha256_file(dataset_path),
        "split_seed": seed,
        "split_policy": {
            "group_keys": ["source_group_id", "near_duplicate_cluster_id"],
            "percentages": {
                "train": 60,
                "development": 15,
                "calibration": 10,
                "locked_test": 15,
            },
            "task_template_holdout_is_separate": True,
        },
        "example_counts": dict(sorted(split_examples.items())),
        "decision_counts": dict(sorted(split_decisions.items())),
        "assignment_sha256": assignment_hash,
        "assignments": dict(sorted(assignments.items())),
    }
