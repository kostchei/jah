import argparse
import json

from jah.evaluation import _classification_metrics, _quadratic_weighted_kappa, compare_reports


def test_classification_and_ordinal_metrics() -> None:
    metrics = _classification_metrics(["a", "a", "b"], ["a", "b", "b"])
    assert metrics["accuracy"] == 2 / 3
    assert metrics["macro_f1"] == 2 / 3
    assert _quadratic_weighted_kappa([(0, 0, 3), (1, 1, 3), (2, 2, 3)]) == 1.0


def test_paired_comparison_checks_quality_and_latency_gates(tmp_path) -> None:
    common = {
        "dataset_sha256": "dataset",
        "split": "development",
        "split_assignment_sha256": "assignment",
        "artifact_id": "artifact",
    }
    direct = {
        **common,
        "backend": "direct",
        "metrics": {"routing/choice": {"macro_f1": 0.90}},
        "latency_ms": {"median_request": 50.0, "p95_request": 100.0},
    }
    generative = {
        **common,
        "backend": "generative",
        "metrics": {"routing/choice": {"macro_f1": 0.91}},
        "latency_ms": {"median_request": 200.0, "p95_request": 300.0},
    }
    (tmp_path / "direct.json").write_text(json.dumps(direct), encoding="utf-8")
    (tmp_path / "generative.json").write_text(json.dumps(generative), encoding="utf-8")
    row = {"example_id": "one", "question_id": "route", "prediction": "billing"}
    for name in ("direct.jsonl", "generative.jsonl"):
        (tmp_path / name).write_text(json.dumps(row) + "\n", encoding="utf-8")

    args = argparse.Namespace(
        root=tmp_path,
        direct_report="direct.json",
        generative_report="generative.json",
        direct_predictions="direct.jsonl",
        generative_predictions="generative.jsonl",
        output="comparison.json",
    )
    assert compare_reports(args) == 0
    result = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert result["all_provisional_gates_passed"] is True
    assert result["latency"]["generative_over_direct_speedup"] == 4.0


def test_paired_comparison_with_calibration_gate(tmp_path) -> None:
    common = {
        "dataset_sha256": "dataset",
        "split": "development",
        "split_assignment_sha256": "assignment",
        "artifact_id": "artifact",
    }
    direct = {
        **common,
        "backend": "direct",
        "metrics": {
            "routing/choice": {
                "macro_f1": 0.90,
                "calibration": {
                    "ece": 0.03,
                    "brier_score": 0.10,
                    "prevalence_brier": 0.25,
                    "nll": 0.20,
                    "gate_passed": True,
                },
                "selective": {
                    "coverage": 0.80,
                    "accepted_error_rate": 0.02,
                    "accepted_error_95_upper_bound": 0.04,
                    "gate_passed": True,
                },
            }
        },
        "latency_ms": {"median_request": 50.0, "p95_request": 100.0},
    }
    generative = {
        **common,
        "backend": "generative",
        "metrics": {"routing/choice": {"macro_f1": 0.90}},
        "latency_ms": {"median_request": 200.0, "p95_request": 300.0},
    }
    (tmp_path / "direct.json").write_text(json.dumps(direct), encoding="utf-8")
    (tmp_path / "generative.json").write_text(json.dumps(generative), encoding="utf-8")
    row = {"example_id": "one", "question_id": "route", "prediction": "billing"}
    for name in ("direct.jsonl", "generative.jsonl"):
        (tmp_path / name).write_text(json.dumps(row) + "\n", encoding="utf-8")

    args = argparse.Namespace(
        root=tmp_path,
        direct_report="direct.json",
        generative_report="generative.json",
        direct_predictions="direct.jsonl",
        generative_predictions="generative.jsonl",
        output="comparison.json",
    )
    assert compare_reports(args) == 0
    result = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert result["all_provisional_gates_passed"] is True
    assert result["metrics"]["routing/choice"]["calibration"]["gate_passed"] is True
