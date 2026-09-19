from pathlib import Path

from jah.workload import (
    load_dataset,
    load_yaml,
    sha256_file,
    validate_dataset_against_workload,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_m0_dataset_is_complete_and_adjudicated() -> None:
    workload_path = ROOT / "configs/workloads/support-routing-v1.yaml"
    workload = load_yaml(workload_path)
    dataset_path = ROOT / workload["dataset"]
    examples = load_dataset(dataset_path)
    validate_dataset_against_workload(examples, workload)
    assert len(examples) == 50
    assert all(len(example.annotations) >= 2 for example in examples)
    assert all(example.annotation_status.startswith("adjudicated-") for example in examples)
    assert sha256_file(dataset_path) == workload["dataset_sha256"]
