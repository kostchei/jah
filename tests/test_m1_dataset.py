import json

import pytest
from pydantic import ValidationError

from jah.m1_dataset import (
    M1Example,
    build_split_manifest,
    split_assignments,
    validate_m1_suite,
)


def example_payload(index: int = 0, *, source: str = "source-1", cluster: str = "cluster-1"):
    answers = {"route": "billing"}
    return {
        "example_id": f"example-{index}",
        "source_group_id": source,
        "near_duplicate_cluster_id": cluster,
        "workload_id": "support-routing-v1",
        "task_id": "routing",
        "template_id": "routing-v1",
        "request": {
            "state": "A duplicate charge.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "options": [
                        {"id": "billing", "description": "Billing"},
                        {"id": "other", "description": "Other"},
                    ],
                }
            },
        },
        "reference_answers": answers,
        "rubric": "Duplicate charges are billing.",
        "provenance": {"kind": "synthetic", "author": "dataset-owner"},
        "annotations": [
            {
                "annotator_id": "human-a",
                "annotator_kind": "human",
                "answers": answers,
                "independent": True,
            },
            {
                "annotator_id": "human-b",
                "annotator_kind": "human",
                "answers": answers,
                "independent": True,
            },
        ],
        "annotation_status": "adjudicated-agreement",
    }


def test_adjudication_requires_two_independent_humans() -> None:
    payload = example_payload()
    payload["annotations"] = payload["annotations"][:1]
    with pytest.raises(ValidationError, match="two distinct independent human"):
        M1Example.model_validate(payload)


def test_suite_gate_reports_missing_size_and_primitives() -> None:
    status = validate_m1_suite([M1Example.model_validate(example_payload())])
    assert status["ready"] is False
    assert status["decisions"] == 1
    assert any("at least 1000" in failure for failure in status["failures"])
    assert any("no boolean" in failure for failure in status["failures"])
    assert any("no score" in failure for failure in status["failures"])


def test_source_and_near_duplicate_components_never_cross_splits() -> None:
    first = M1Example.model_validate(example_payload(1, source="source-a", cluster="cluster-x"))
    second = M1Example.model_validate(example_payload(2, source="source-b", cluster="cluster-x"))
    third = M1Example.model_validate(example_payload(3, source="source-b", cluster="cluster-y"))
    assignments = split_assignments([first, second, third], seed="fixed")
    assert len(set(assignments.values())) == 1


def test_manifest_is_reproducible_and_hashed(tmp_path) -> None:
    examples = [M1Example.model_validate(example_payload())]
    dataset = tmp_path / "suite.jsonl"
    dataset.write_text(json.dumps(examples[0].model_dump(mode="json")) + "\n", encoding="utf-8")
    first = build_split_manifest(examples, dataset_path=dataset, seed="fixed")
    second = build_split_manifest(examples, dataset_path=dataset, seed="fixed")
    assert first == second
    assert len(first["assignment_sha256"]) == 64
    assert len(first["dataset_sha256"]) == 64
