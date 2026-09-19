from pathlib import Path

import pytest

from jah.workload import (
    M0Example,
    load_dataset,
    load_yaml,
    sha256_file,
    validate_dataset_against_workload,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_m0_dataset_is_complete_and_honestly_labeled() -> None:
    workload_path = ROOT / "configs/workloads/support-routing-v1.yaml"
    workload = load_yaml(workload_path)
    dataset_path = ROOT / workload["dataset"]
    examples = load_dataset(dataset_path)
    validate_dataset_against_workload(examples, workload)
    assert len(examples) == 50
    assert all(len(example.annotations) >= 2 for example in examples)
    assert all(example.annotation_status == "single-author-two-pass" for example in examples)
    assert sha256_file(dataset_path) == workload["dataset_sha256"]


def test_adjudicated_status_requires_two_distinct_human_annotators() -> None:
    workload = load_yaml(ROOT / "configs/workloads/support-routing-v1.yaml")
    example = load_dataset(ROOT / workload["dataset"])[0].model_dump()
    example["annotation_status"] = "adjudicated-agreement"

    with pytest.raises(ValueError, match="two distinct human annotator IDs"):
        M0Example.model_validate(example)

    for annotation in example["annotations"]:
        annotation["annotator_kind"] = "human"
    validated = M0Example.model_validate(example)
    assert validated.annotation_status == "adjudicated-agreement"
