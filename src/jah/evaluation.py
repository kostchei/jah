"""Reproducible M1 suite validation and single-backend evaluation runner."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from jah.engine import DecisionEngine, EngineConfig
from jah.m1_dataset import build_split_manifest, load_m1_dataset, validate_m1_suite
from jah.schemas import ScoreAnswer
from jah.workload import git_commit_sha, load_yaml, sha256_file, stable_json_sha256


def _classification_metrics(references: list[str], predictions: list[str]) -> dict:
    labels = sorted(set(references) | set(predictions))
    recalls = []
    f1s = []
    for label in labels:
        true_positive = sum(r == label and p == label for r, p in zip(references, predictions))
        false_positive = sum(r != label and p == label for r, p in zip(references, predictions))
        false_negative = sum(r == label and p != label for r, p in zip(references, predictions))
        recall_denominator = true_positive + false_negative
        recalls.append(true_positive / recall_denominator if recall_denominator else 0.0)
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1s.append(2 * true_positive / f1_denominator if f1_denominator else 0.0)
    return {
        "decisions": len(references),
        "accuracy": sum(r == p for r, p in zip(references, predictions)) / len(references),
        "balanced_accuracy": statistics.fmean(recalls),
        "macro_f1": statistics.fmean(f1s),
        "labels": labels,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    return sorted(values)[int(fraction * (len(values) - 1))]


def _quadratic_weighted_kappa(pairs: list[tuple[int, int, int]]) -> float:
    cardinalities = {cardinality for _, _, cardinality in pairs}
    if len(cardinalities) != 1:
        raise ValueError("weighted kappa requires stable score cardinality within a workload")
    cardinality = cardinalities.pop()
    actual = Counter(reference for reference, _, _ in pairs)
    predicted = Counter(prediction for _, prediction, _ in pairs)
    observed = Counter((reference, prediction) for reference, prediction, _ in pairs)
    count = len(pairs)
    weighted_observed = 0.0
    weighted_expected = 0.0
    for reference in range(cardinality):
        for prediction in range(cardinality):
            weight = ((reference - prediction) / (cardinality - 1)) ** 2
            weighted_observed += weight * observed[(reference, prediction)]
            weighted_expected += weight * actual[reference] * predicted[prediction] / count
    if weighted_expected == 0:
        return 1.0 if weighted_observed == 0 else 0.0
    return 1.0 - weighted_observed / weighted_expected


def load_backend(name: str, model_config: dict):
    if name == "direct":
        from jah.backends.huggingface import HuggingFaceDirectLogitBackend

        backend_type = HuggingFaceDirectLogitBackend
    elif name == "generative":
        from jah.backends.generative import HuggingFaceGenerativeBackend

        backend_type = HuggingFaceGenerativeBackend
    else:
        raise ValueError(f"unknown backend: {name}")
    return backend_type(
        model_config["model_id"],
        model_config["revision"],
        device=model_config["device"],
    )


def run_evaluation(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    dataset_path = root / args.dataset
    examples = load_m1_dataset(dataset_path)
    readiness = validate_m1_suite(examples)
    if not readiness["ready"]:
        raise ValueError("M1 suite is not ready: " + "; ".join(readiness["failures"]))
    manifest = build_split_manifest(
        examples,
        dataset_path=dataset_path,
        seed=args.split_seed,
        dataset_label=Path(args.dataset).as_posix(),
    )
    selected = [
        example
        for example in examples
        if manifest["assignments"][example.example_id] == args.split
    ]
    if not selected:
        raise ValueError(f"split {args.split!r} contains no examples")

    model_path = root / args.model_config
    model_config = load_yaml(model_path)
    backend = load_backend(args.backend, model_config)
    engine = DecisionEngine(
        backend,
        EngineConfig(
            artifact_id=model_config["artifact_id"],
            maximum_input_tokens=model_config["maximum_input_tokens"],
        ),
    )

    rows = []
    request_latencies = []
    grouped_references: dict[tuple[str, str], list[str]] = defaultdict(list)
    grouped_predictions: dict[tuple[str, str], list[str]] = defaultdict(list)
    score_errors: dict[str, list[float]] = defaultdict(list)
    score_pairs: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for example in selected:
        started = time.perf_counter()
        response = engine.evaluate(example.request, request_id=f"eval_{example.example_id}")
        request_latencies.append((time.perf_counter() - started) * 1_000)
        for question_id, question in example.request.questions.items():
            answer = response.answers[question_id]
            reference = example.reference_answers[question_id]
            if isinstance(answer, ScoreAnswer):
                reference_level = next(level for level in question.levels if level.id == reference)
                value_range = question.levels[-1].value - question.levels[0].value
                if value_range <= 0:
                    raise ValueError(f"{example.example_id}/{question_id}: invalid score range")
                score_errors[example.workload_id].append(
                    abs(answer.expected_value - reference_level.value) / value_range
                )
                prediction = answer.level_id
                reference_index = next(
                    index for index, level in enumerate(question.levels) if level.id == reference
                )
                prediction_index = next(
                    index
                    for index, level in enumerate(question.levels)
                    if level.id == prediction
                )
                score_pairs[example.workload_id].append(
                    (reference_index, prediction_index, len(question.levels))
                )
            else:
                prediction = "true" if getattr(answer, "value", None) is True else (
                    "false" if getattr(answer, "value", None) is False else answer.value
                )
                group = (example.workload_id, question.type)
                grouped_references[group].append(reference)
                grouped_predictions[group].append(prediction)
            rows.append(
                {
                    "example_id": example.example_id,
                    "question_id": question_id,
                    "workload_id": example.workload_id,
                    "primitive": question.type,
                    "reference": reference,
                    "prediction": prediction,
                    "answer": answer.model_dump(mode="json"),
                    "request_total_ms": response.timing_ms.total,
                    "inference_ms": response.timing_ms.inference,
                }
            )

    metrics = {
        f"{workload}/{primitive}": _classification_metrics(
            grouped_references[(workload, primitive)], grouped_predictions[(workload, primitive)]
        )
        for workload, primitive in sorted(grouped_references)
    }
    for workload, errors in sorted(score_errors.items()):
        metrics[f"{workload}/score"] = {
            "decisions": len(errors),
            "normalized_mae": statistics.fmean(errors),
            "quadratic_weighted_kappa": _quadratic_weighted_kappa(score_pairs[workload]),
        }

    predictions_path = root / args.predictions
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit_sha": git_commit_sha(root),
        "backend": args.backend,
        "artifact_id": model_config["artifact_id"],
        "model_config_sha256": sha256_file(model_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "split": args.split,
        "split_assignment_sha256": manifest["assignment_sha256"],
        "examples": len(selected),
        "decisions": len(rows),
        "metrics": metrics,
        "latency_ms": {
            "median_request": statistics.median(request_latencies),
            "p95_request": _percentile(request_latencies, 0.95),
        },
        "predictions_sha256": sha256_file(predictions_path),
    }
    report["evidence_sha256"] = stable_json_sha256(
        {key: value for key, value in report.items() if key not in {"created_at", "evidence_sha256"}}
    )
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def validate_dataset(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    dataset_path = root / args.dataset
    examples = load_m1_dataset(dataset_path)
    readiness = validate_m1_suite(examples)
    manifest = build_split_manifest(
        examples,
        dataset_path=dataset_path,
        seed=args.split_seed,
        dataset_label=Path(args.dataset).as_posix(),
    )
    result = {"readiness": readiness, "split_manifest": manifest}
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if readiness["ready"] else 2


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _load_prediction_index(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["example_id"], row["question_id"])
            if key in rows:
                raise ValueError(f"duplicate prediction key at {path}:{line_number}: {key}")
            rows[key] = row
    return rows


def compare_reports(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    direct = _load_json(root / args.direct_report)
    generative = _load_json(root / args.generative_report)
    for field in ("dataset_sha256", "split", "split_assignment_sha256", "artifact_id"):
        if direct.get(field) != generative.get(field):
            raise ValueError(f"reports are not paired: {field} differs")
    if direct.get("backend") != "direct" or generative.get("backend") != "generative":
        raise ValueError("comparison requires direct and generative reports")

    direct_rows = _load_prediction_index(root / args.direct_predictions)
    generative_rows = _load_prediction_index(root / args.generative_predictions)
    if set(direct_rows) != set(generative_rows):
        raise ValueError("prediction files do not contain identical decision keys")

    paired_metrics = {}
    gates = []
    for name in sorted(set(direct["metrics"]) | set(generative["metrics"])):
        direct_metric = direct["metrics"].get(name)
        generated_metric = generative["metrics"].get(name)
        if direct_metric is None or generated_metric is None:
            raise ValueError(f"metric {name!r} is missing from one report")
        if "macro_f1" in direct_metric:
            difference = direct_metric["macro_f1"] - generated_metric["macro_f1"]
            metric_gate = direct_metric["macro_f1"] >= 0.85 and difference >= -0.02
            paired_metrics[name] = {
                "direct_macro_f1": direct_metric["macro_f1"],
                "generative_macro_f1": generated_metric["macro_f1"],
                "direct_minus_generative": difference,
                "gate_passed": metric_gate,
            }
        else:
            metric_gate = direct_metric["normalized_mae"] <= 0.15
            paired_metrics[name] = {
                "direct_normalized_mae": direct_metric["normalized_mae"],
                "generative_normalized_mae": generated_metric["normalized_mae"],
                "gate_passed": metric_gate,
            }
        gates.append(metric_gate)

    direct_median = direct["latency_ms"]["median_request"]
    generative_median = generative["latency_ms"]["median_request"]
    speedup = generative_median / direct_median if direct_median else float("inf")
    latency_gate = speedup >= 2.0 and direct["latency_ms"]["p95_request"] <= 2_000
    gates.append(latency_gate)
    disagreements = sum(
        direct_rows[key]["prediction"] != generative_rows[key]["prediction"]
        for key in direct_rows
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit_sha": git_commit_sha(root),
        "dataset_sha256": direct["dataset_sha256"],
        "split": direct["split"],
        "split_assignment_sha256": direct["split_assignment_sha256"],
        "artifact_id": direct["artifact_id"],
        "paired_decisions": len(direct_rows),
        "prediction_disagreements": disagreements,
        "metrics": paired_metrics,
        "latency": {
            "direct_median_request_ms": direct_median,
            "generative_median_request_ms": generative_median,
            "generative_over_direct_speedup": speedup,
            "direct_p95_request_ms": direct["latency_ms"]["p95_request"],
            "gate_passed": latency_gate,
        },
        "all_provisional_gates_passed": all(gates),
    }
    result["evidence_sha256"] = stable_json_sha256(
        {key: value for key, value in result.items() if key not in {"created_at", "evidence_sha256"}}
    )
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_provisional_gates_passed"] else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset", default="evals/data/m1-suite.jsonl")
    validate.add_argument("--split-seed", default="jah-m1-split-v1")
    validate.add_argument("--output", default="artifacts/m1/suite-validation.json")
    validate.set_defaults(function=validate_dataset)

    run = subparsers.add_parser("run")
    run.add_argument("--dataset", default="evals/data/m1-suite.jsonl")
    run.add_argument("--split-seed", default="jah-m1-split-v1")
    run.add_argument("--model-config", default="configs/models/qwen3.5-4b.yaml")
    run.add_argument("--backend", choices=("direct", "generative"), required=True)
    run.add_argument(
        "--split",
        choices=("train", "development", "calibration", "locked_test", "task_holdout"),
        default="development",
    )
    run.add_argument("--output", required=True)
    run.add_argument("--predictions", required=True)
    run.set_defaults(function=run_evaluation)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--direct-report", required=True)
    compare.add_argument("--generative-report", required=True)
    compare.add_argument("--direct-predictions", required=True)
    compare.add_argument("--generative-predictions", required=True)
    compare.add_argument("--output", default="artifacts/m1/paired-comparison.json")
    compare.set_defaults(function=compare_reports)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
