"""Unit tests for the LLM review harness (audit, slice, generate, and control arm)."""

import json
from pathlib import Path

import pytest

from jah.review import (
    CORRELATED_FAMILY_LIMITATION,
    parse_args,
    run_audit,
    run_generate,
    run_slice,
)


@pytest.fixture
def mock_predictions(tmp_path: Path) -> Path:
    preds_file = tmp_path / "mock_predictions.jsonl"
    rows = [
        # 4 correct predictions (agreement rows)
        {"example_id": "ex-1", "question_id": "q1", "workload_id": "test-w", "reference": "A", "prediction": "A"},
        {"example_id": "ex-2", "question_id": "q1", "workload_id": "test-w", "reference": "B", "prediction": "B"},
        {"example_id": "ex-3", "question_id": "q1", "workload_id": "test-w", "reference": "C", "prediction": "C"},
        {"example_id": "ex-4", "question_id": "q1", "workload_id": "test-w", "reference": "D", "prediction": "D"},
        # 2 incorrect predictions (disagreement rows)
        {"example_id": "ex-5", "question_id": "q1", "workload_id": "test-w", "reference": "A", "prediction": "B"},
        {"example_id": "ex-6", "question_id": "q1", "workload_id": "test-w", "reference": "C", "prediction": "D"},
    ]
    with preds_file.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return preds_file


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    from jah.m1_dataset import M1Example
    from jah.schemas import ChoiceOption, ChoiceQuestion, EvaluateRequest

    dataset_file = tmp_path / "mock_suite.jsonl"
    examples = []
    for i in range(1, 7):
        from jah.m1_dataset import PublicSource

        ps = PublicSource(
            dataset="test-benchmark",
            revision="1.0",
            file_sha256="a" * 64,
            source_url="https://example.com",
            license="apache-2.0",
            annotation_documentation="Upstream docs",
            upstream_id=f"up-{i}",
            upstream_split="dev",
            original_label="A",
        )
        ex = M1Example(
            example_id=f"ex-{i}",
            source_group_id="sg-1",
            near_duplicate_cluster_id="cl-1",
            workload_id="test-w",
            task_id="t-1",
            template_id="tpl-1",
            request=EvaluateRequest(
                state="sample state text",
                questions={
                    "q1": ChoiceQuestion(
                        type="choice",
                        instructions="Select option",
                        options=[ChoiceOption(id=opt, description=f"Desc {opt}") for opt in ["A", "B", "C", "D"]],
                    )
                },
            ),
            reference_answers={"q1": "A" if i in (1, 5) else ("B" if i == 2 else ("C" if i in (3, 6) else "D"))},
            rubric="Valid rubric",
            provenance={"kind": "public-benchmark", "author": "upstream-author"},
            annotations=[],
            annotation_status="upstream-human",
            public_source=ps,
            evaluation_split="development",
        )
        examples.append(ex)

    with dataset_file.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.model_dump(mode="json")) + "\n")
    return dataset_file


def test_audit_subcommand_with_control_arm(mock_predictions: Path, mock_dataset: Path, tmp_path: Path):
    output_file = tmp_path / "audit_result.json"
    args = parse_args([
        "audit",
        "--predictions", str(mock_predictions),
        "--dataset", str(mock_dataset),
        "--output", str(output_file),
        "--mock",
        "--control-arm-ratio", "1.0",
        "--max-control-dispute-rate", "0.15",
    ])
    result = run_audit(args)

    assert output_file.exists()
    assert result["command"] == "audit"
    assert result["provenance"]["ground_truth"] is False
    assert result["provenance"]["limitation"] == CORRELATED_FAMILY_LIMITATION
    assert "control_arm" in result
    assert result["control_arm"]["sample_size"] > 0
    # In mock, agreements are not disputed -> dispute rate is 0.0 -> control arm passes
    assert result["control_arm"]["passed"] is True
    assert len(result["quarantined_examples"]) == 2
    assert "ex-5:q1" in result["quarantined_examples"]
    assert "ex-6:q1" in result["quarantined_examples"]


def test_audit_control_arm_failure_discards_quarantine(mock_predictions: Path, mock_dataset: Path, tmp_path: Path):
    output_file = tmp_path / "audit_failed.json"
    # Setting max_control_dispute_rate to negative guarantees failure if any dispute,
    # but in mock dispute rate is 0.0. If max_control_dispute_rate < 0, it fails.
    args = parse_args([
        "audit",
        "--predictions", str(mock_predictions),
        "--dataset", str(mock_dataset),
        "--output", str(output_file),
        "--mock",
        "--max-control-dispute-rate", "-0.01",
    ])
    result = run_audit(args)

    assert result["control_arm"]["passed"] is False
    # When control arm fails, quarantine recommendations must be discarded
    assert len(result["quarantined_examples"]) == 0


def test_slice_subcommand(mock_predictions: Path, mock_dataset: Path, tmp_path: Path):
    output_file = tmp_path / "slices.json"
    args = parse_args([
        "slice",
        "--predictions", str(mock_predictions),
        "--dataset", str(mock_dataset),
        "--output", str(output_file),
        "--mock",
    ])
    result = run_slice(args)

    assert output_file.exists()
    assert result["command"] == "slice"
    assert "test-w" in result["slices"]
    assert result["slices"]["test-w"]["error_count"] == 2


def test_generate_subcommand(tmp_path: Path):
    output_file = tmp_path / "generated.json"
    args = parse_args([
        "generate",
        "--output", str(output_file),
        "--mock",
    ])
    result = run_generate(args)

    assert output_file.exists()
    assert result["command"] == "generate"
    assert len(result["generated_cases"]) > 0
    assert result["generated_cases"][0]["verifiable_by_construction"] is True


def test_quarantine_exclusion_in_evaluation(mock_predictions: Path, mock_dataset: Path, tmp_path: Path):
    from jah.m1_dataset import load_m1_dataset

    # 1. Produce audit with quarantined rows
    audit_file = tmp_path / "audit_result.json"
    audit_args = parse_args([
        "audit",
        "--predictions", str(mock_predictions),
        "--dataset", str(mock_dataset),
        "--output", str(audit_file),
        "--mock",
    ])
    run_audit(audit_args)

    # 2. Verify quarantine loading logic
    with audit_file.open("r", encoding="utf-8") as f:
        audit_data = json.load(f)
    quarantined = {item.split(":")[0] for item in audit_data["quarantined_examples"]}
    assert quarantined == {"ex-5", "ex-6"}

    examples = load_m1_dataset(mock_dataset)
    filtered = [e for e in examples if e.example_id not in quarantined]
    assert len(filtered) == 4
    assert {e.example_id for e in filtered} == {"ex-1", "ex-2", "ex-3", "ex-4"}

