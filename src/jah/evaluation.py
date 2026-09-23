"""Reproducible M1 suite validation and single-backend evaluation runner."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from jah.calibration import (
    CalibrationMetrics,
    CalibrationProfile,
    compute_binomial_upper_bound,
    compute_brier_score,
    compute_ece,
    compute_nll,
    compute_prevalence_brier,
    compute_selective_metrics,
    fit_temperature_scaling,
    fit_vector_scaling,
    load_profile_registry,
    save_profile,
)
from jah.compiler import compile_request
from jah.engine import DecisionEngine, EngineConfig
from jah.m1_dataset import build_split_manifest, load_m1_dataset, validate_m1_suite
from jah.policy import AcceptancePolicy, fit_acceptance_threshold
from jah.resource import configure_resource_limits, throttle
from jah.schemas import BooleanAnswer, ChoiceAnswer, ScoreAnswer
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


def load_backend(name: str, model_config: dict, adapter_dir: str | Path | None = None, **kwargs):
    if name == "direct":
        from jah.backends.huggingface import HuggingFaceDirectLogitBackend

        return HuggingFaceDirectLogitBackend(
            model_config["model_id"],
            model_config["revision"],
            device=model_config["device"],
            adapter_dir=adapter_dir,
        )
    elif name == "generative":
        from jah.backends.generative import HuggingFaceGenerativeBackend

        return HuggingFaceGenerativeBackend(
            model_config["model_id"],
            model_config["revision"],
            device=model_config["device"],
        )
    else:
        raise ValueError(f"unknown backend: {name}")


def _load_suite_validation_params(
    root: Path, suite_config_path: str | None
) -> tuple[int, tuple[str, ...] | None]:
    if not suite_config_path:
        return 120, None
    path = root / suite_config_path
    if not path.exists():
        return 120, None
    config = load_yaml(path)
    minimum_decisions = config.get("minimum_decisions", 120)
    allowed_statuses = config.get("required_annotation_status")
    if allowed_statuses is not None:
        allowed_statuses = tuple(allowed_statuses)
    return minimum_decisions, allowed_statuses


def run_evaluation(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    dataset_path = root / args.dataset
    examples = load_m1_dataset(dataset_path)
    min_decisions, allowed_statuses = _load_suite_validation_params(
        root, getattr(args, "suite_config", None)
    )
    readiness = validate_m1_suite(
        examples, minimum_decisions=min_decisions, allowed_statuses=allowed_statuses
    )
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
    if getattr(args, "workload", None):
        selected = [e for e in selected if e.workload_id == args.workload]
    if getattr(args, "exclude_reviewed_quarantine", None):
        q_path = root / args.exclude_reviewed_quarantine
        if q_path.exists():
            with q_path.open("r", encoding="utf-8") as handle:
                q_data = json.load(handle)
            quarantined_ids = {item.split(":")[0] for item in q_data.get("quarantined_examples", [])}
            initial_count = len(selected)
            selected = [e for e in selected if e.example_id not in quarantined_ids]
            excluded_count = initial_count - len(selected)
            print(
                f"Quarantine exclusion applied: removed {excluded_count} examples based on {q_path}",
                flush=True,
            )
    if getattr(args, "max_examples", None):
        selected = selected[: args.max_examples]
    if not selected:
        raise ValueError(f"split {args.split!r} contains no examples")

    if not getattr(args, "no_resource_limits", False):
        res_limits = configure_resource_limits(
            max_total_gpu_fraction=getattr(args, "resource_limit", 0.80),
            max_cpu_fraction=getattr(args, "resource_limit", 0.80),
        )
        print(f"Resource limits configured: {res_limits}", flush=True)

    model_path = root / args.model_config
    model_config = load_yaml(model_path)
    adapter_dir = getattr(args, "adapter_dir", None)
    backend = load_backend(args.backend, model_config, adapter_dir)

    profiles_dir_arg = getattr(args, "profiles_dir", None)
    profiles = {}
    if profiles_dir_arg:
        p_path = root / profiles_dir_arg
        if p_path.exists():
            profiles = load_profile_registry(p_path)

    model_metadata = {
        "model_id": model_config.get("model_id"),
        "revision": model_config.get("revision"),
        "precision": model_config.get("precision"),
        "prompt_version": model_config.get("prompt_version"),
        "label_version": model_config.get("label_version"),
    }
    engine = DecisionEngine(
        backend,
        EngineConfig(
            artifact_id=model_config["artifact_id"],
            maximum_input_tokens=model_config["maximum_input_tokens"],
            profiles=profiles,
            model_metadata=model_metadata,
        ),
    )

    predictions_path = root / args.predictions
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    resume = getattr(args, "resume", False)
    chunk_size = getattr(args, "chunk_size", 25)
    throttle_ms = getattr(args, "throttle_ms", 5.0)

    rows = []
    request_latencies = []
    grouped_references: dict[tuple[str, str], list[str]] = defaultdict(list)
    grouped_predictions: dict[tuple[str, str], list[str]] = defaultdict(list)
    score_errors: dict[str, list[float]] = defaultdict(list)
    score_pairs: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    calibrated_probs: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    calibrated_targets: dict[tuple[str, str], list[int]] = defaultdict(list)
    calibrated_confs: dict[tuple[str, str], list[float]] = defaultdict(list)
    calibrated_accs: dict[tuple[str, str], list[int]] = defaultdict(list)
    calibrated_dispositions: dict[tuple[str, str], list[str]] = defaultdict(list)
    option_cardinalities: dict[tuple[str, str], int] = {}

    def _record_decision(example, question_id, question, answer, req_total_ms, inf_ms):
        reference = example.reference_answers[question_id]
        group = (example.workload_id, question.type)
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
            lvl_ids = [lvl.id for lvl in question.levels]
            probs = [answer.probabilities[lid] for lid in lvl_ids]
            t_idx = lvl_ids.index(reference)
            opt_len = len(lvl_ids)
        else:
            prediction = "true" if getattr(answer, "value", None) is True else (
                "false" if getattr(answer, "value", None) is False else answer.value
            )
            grouped_references[group].append(reference)
            grouped_predictions[group].append(prediction)
            if question.type == "boolean":
                probs = [answer.probabilities["false"], answer.probabilities["true"]]
                t_idx = 1 if reference == "true" else 0
                opt_len = 2
            else:
                opt_ids = [opt.id for opt in question.options]
                probs = [answer.probabilities[oid] for oid in opt_ids]
                t_idx = opt_ids.index(reference)
                opt_len = len(opt_ids)

        calibrated_probs[group].append(probs)
        calibrated_targets[group].append(t_idx)
        calibrated_confs[group].append(max(probs))
        is_correct = 1 if prediction == reference else 0
        calibrated_accs[group].append(is_correct)
        calibrated_dispositions[group].append(answer.disposition)
        option_cardinalities[group] = opt_len

        row = {
            "example_id": example.example_id,
            "question_id": question_id,
            "workload_id": example.workload_id,
            "primitive": question.type,
            "reference": reference,
            "prediction": prediction,
            "answer": answer.model_dump(mode="json"),
            "request_total_ms": req_total_ms,
            "inference_ms": inf_ms,
        }
        rows.append(row)
        return row

    existing_rows_by_example: dict[str, list[dict]] = defaultdict(list)
    if resume and predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    existing_rows_by_example[r["example_id"]].append(r)
                except Exception:
                    continue
        if existing_rows_by_example:
            print(f"Resuming evaluation: found {len(existing_rows_by_example)} existing examples in {predictions_path}", flush=True)

    example_map = {e.example_id: e for e in selected}
    for ex_id, row_list in existing_rows_by_example.items():
        if ex_id in example_map:
            ex = example_map[ex_id]
            for r in row_list:
                qid = r["question_id"]
                if qid in ex.request.questions:
                    q = ex.request.questions[qid]
                    ans_data = r["answer"]
                    if q.type == "score":
                        ans = ScoreAnswer.model_validate(ans_data)
                    elif q.type == "choice":
                        ans = ChoiceAnswer.model_validate(ans_data)
                    else:
                        ans = BooleanAnswer.model_validate(ans_data)
                    _record_decision(ex, qid, q, ans, r.get("request_total_ms", 0.0), r.get("inference_ms", 0.0))
                    request_latencies.append(r.get("request_total_ms", 0.0))

    pred_mode = "a" if (resume and existing_rows_by_example) else "w"
    pred_handle = predictions_path.open(pred_mode, encoding="utf-8", newline="\n")

    total_sel = len(selected)
    resumed_count = len(existing_rows_by_example)
    new_count = 0
    try:
        for idx, example in enumerate(selected):
            if example.example_id in existing_rows_by_example:
                continue

            req = example.request
            if getattr(args, "use_workload_profiles", False):
                req = example.request.model_copy(update={"profile": example.workload_id})

            started = time.perf_counter()
            response = engine.evaluate(req, request_id=f"eval_{example.example_id}")
            tot_ms = (time.perf_counter() - started) * 1_000
            request_latencies.append(tot_ms)

            for question_id, question in req.questions.items():
                answer = response.answers[question_id]
                row = _record_decision(
                    example,
                    question_id,
                    question,
                    answer,
                    response.timing_ms.total,
                    response.timing_ms.inference,
                )
                pred_handle.write(json.dumps(row, sort_keys=True) + "\n")

            pred_handle.flush()
            new_count += 1
            if throttle_ms > 0:
                throttle(throttle_ms)

            if (new_count > 0 and new_count % chunk_size == 0) or (idx + 1 == total_sel):
                pct = (idx + 1) / total_sel * 100
                print(
                    f"Evaluation progress: {idx + 1}/{total_sel} ({pct:.1f}%) "
                    f"[{resumed_count} resumed, {new_count} computed]...",
                    flush=True,
                )
    finally:
        pred_handle.close()

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

    # Add calibration and selective decision metrics per workload
    for group, probs_g in calibrated_probs.items():
        workload, primitive = group
        key = f"{workload}/{primitive}"
        if key not in metrics:
            continue
        targets_g = calibrated_targets[group]
        confs_g = calibrated_confs[group]
        accs_g = calibrated_accs[group]
        disps_g = calibrated_dispositions[group]
        num_classes = option_cardinalities[group]

        ece, _ = compute_ece(confs_g, accs_g, num_bins=10, equal_count=True)
        brier = compute_brier_score(probs_g, targets_g)
        prev_brier = compute_prevalence_brier(targets_g, num_classes)
        nll = compute_nll(probs_g, targets_g)

        accepted = [acc for acc, disp in zip(accs_g, disps_g, strict=True) if disp == "accept"]
        accepted_count = len(accepted)
        review_count = len(accs_g) - accepted_count
        coverage = accepted_count / len(accs_g) if accs_g else 0.0
        errors = sum(1 for a in accepted if a == 0)
        accepted_error_rate = (errors / accepted_count) if accepted_count else 0.0
        bound_95 = compute_binomial_upper_bound(errors, accepted_count, confidence=0.95)

        gate_ece = (ece <= 0.05)
        gate_brier = (brier <= prev_brier)
        gate_selective = (coverage >= 0.50 and bound_95 <= 0.05)
        samples_sufficient = (accepted_count >= 59)

        metrics[key]["calibration"] = {
            "ece": ece,
            "brier_score": brier,
            "prevalence_brier": prev_brier,
            "nll": nll,
            "gate_passed": gate_ece and gate_brier,
            "gate_ece_passed": gate_ece,
            "gate_brier_passed": gate_brier,
        }
        metrics[key]["selective"] = {
            "accepted_count": accepted_count,
            "review_count": review_count,
            "coverage": coverage,
            "accepted_error_rate": accepted_error_rate,
            "accepted_error_95_upper_bound": bound_95,
            "samples_sufficient_for_gate": samples_sufficient,
            "gate_passed": gate_selective,
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
        "evaluation_scope": (
            "public-benchmark" if any(e.public_source is not None for e in selected)
            else "workload-evaluation"
        ),
        "release_annotation_ready": readiness["release_annotation_ready"],
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
    min_decisions, allowed_statuses = _load_suite_validation_params(
        root, getattr(args, "suite_config", None)
    )
    readiness = validate_m1_suite(
        examples, minimum_decisions=min_decisions, allowed_statuses=allowed_statuses
    )
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
        if "calibration" in direct_metric:
            paired_metrics[name]["calibration"] = direct_metric["calibration"]
            gates.append(direct_metric["calibration"]["gate_passed"])
        if "selective" in direct_metric:
            paired_metrics[name]["selective"] = direct_metric["selective"]

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


def run_calibration(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    dataset_path = root / args.dataset
    examples = load_m1_dataset(dataset_path)
    min_decisions, allowed_statuses = _load_suite_validation_params(
        root, getattr(args, "suite_config", None)
    )
    readiness = validate_m1_suite(
        examples, minimum_decisions=min_decisions, allowed_statuses=allowed_statuses
    )
    if not readiness["ready"]:
        raise ValueError("M1 suite is not ready: " + "; ".join(readiness["failures"]))
    manifest = build_split_manifest(
        examples,
        dataset_path=dataset_path,
        seed=args.split_seed,
        dataset_label=Path(args.dataset).as_posix(),
    )
    calibration_examples = [
        e for e in examples if manifest["assignments"][e.example_id] == "calibration"
    ]
    if not getattr(args, "no_resource_limits", False):
        res_limits = configure_resource_limits(
            max_total_gpu_fraction=getattr(args, "resource_limit", 0.80),
            max_cpu_fraction=getattr(args, "resource_limit", 0.80),
        )
        print(f"Resource limits configured: {res_limits}", flush=True)

    cache_path = root / getattr(args, "cache_file", "artifacts/public/calibration_cache.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, list[dict]] = defaultdict(list)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cache[rec["example_id"]].append(rec)
                except Exception:
                    continue
        if cache:
            print(f"Loaded {len(cache)} cached examples from {cache_path}", flush=True)

    workload_filter = getattr(args, "workload", None)
    if workload_filter:
        calibration_examples = [e for e in calibration_examples if e.workload_id == workload_filter]
        if not calibration_examples:
            raise ValueError(f"no calibration examples match workload '{workload_filter}'")

    missing = [e for e in calibration_examples if e.example_id not in cache]
    model_path = root / args.model_config
    model_config = load_yaml(model_path)
    backend = None
    if missing:
        backend = load_backend("direct", model_config)

    # Group data by workload
    workload_data: dict[str, dict] = defaultdict(lambda: {
        "primitive": None,
        "option_ids": None,
        "option_values": None,
        "logits_list": [],
        "targets": [],
        "confidences_raw": [],
        "accuracies_raw": [],
        "public_benchmark": False,
    })

    chunk_size = getattr(args, "chunk_size", 25)
    throttle_ms = getattr(args, "throttle_ms", 5.0)
    total_cal = len(calibration_examples)
    cached_count = 0
    new_scored = 0

    cache_handle = cache_path.open("a", encoding="utf-8", newline="\n")
    try:
        for idx, example in enumerate(calibration_examples):
            w_id = example.workload_id
            w_data = workload_data[w_id]
            if example.example_id in cache:
                for rec in cache[example.example_id]:
                    w_data["primitive"] = rec["primitive"]
                    w_data["option_ids"] = tuple(rec["option_ids"])
                    if rec.get("option_values"):
                        w_data["option_values"] = tuple(rec["option_values"])
                    w_data["public_benchmark"] |= rec.get("public_benchmark", False)
                    w_data["logits_list"].append(rec["logits"])
                    w_data["targets"].append(rec["target"])
                    w_data["confidences_raw"].append(rec["confidence_raw"])
                    w_data["accuracies_raw"].append(rec["accuracy_raw"])
                cached_count += 1
            else:
                compiled = compile_request(example.request, tokenizer=backend.tokenizer)
                for q_compiled in compiled:
                    question_id = q_compiled.question_id
                    question = example.request.questions[question_id]
                    measurement = backend.score(q_compiled)
                    reference = example.reference_answers[question_id]
                    w_data["primitive"] = question.type
                    w_data["public_benchmark"] |= example.public_source is not None

                    if question.type == "boolean":
                        option_ids = ("false", "true")
                        w_data["option_ids"] = option_ids
                        target_idx = 1 if reference == "true" else 0
                        logits_vec = [measurement.decision.logits["false"], measurement.decision.logits["true"]]
                        opt_vals = None
                    elif question.type == "choice":
                        option_ids = tuple(opt.id for opt in question.options)
                        w_data["option_ids"] = option_ids
                        target_idx = option_ids.index(reference)
                        logits_vec = [measurement.decision.logits[opt_id] for opt_id in option_ids]
                        opt_vals = None
                    elif question.type == "score":
                        option_ids = tuple(lvl.id for lvl in question.levels)
                        w_data["option_ids"] = option_ids
                        w_data["option_values"] = tuple(lvl.value for lvl in question.levels)
                        target_idx = option_ids.index(reference)
                        logits_vec = [measurement.decision.logits[lvl_id] for lvl_id in option_ids]
                        opt_vals = list(w_data["option_values"])
                    else:
                        continue

                    w_data["logits_list"].append(logits_vec)
                    w_data["targets"].append(target_idx)
                    raw_probs = measurement.decision.probabilities
                    raw_pred = measurement.decision.selected_id
                    conf_raw = max(raw_probs.values())
                    acc_raw = 1 if raw_pred == reference else 0
                    w_data["confidences_raw"].append(conf_raw)
                    w_data["accuracies_raw"].append(acc_raw)

                    rec = {
                        "example_id": example.example_id,
                        "question_id": question_id,
                        "workload_id": w_id,
                        "primitive": question.type,
                        "option_ids": list(option_ids),
                        "option_values": opt_vals,
                        "logits": logits_vec,
                        "target": target_idx,
                        "confidence_raw": conf_raw,
                        "accuracy_raw": acc_raw,
                        "public_benchmark": example.public_source is not None,
                    }
                    cache_handle.write(json.dumps(rec) + "\n")
                    cache[example.example_id].append(rec)
                cache_handle.flush()
                new_scored += 1
                if throttle_ms > 0:
                    throttle(throttle_ms)

            if (new_scored > 0 and new_scored % chunk_size == 0) or (idx + 1 == total_cal):
                pct = (idx + 1) / total_cal * 100
                print(
                    f"Calibration progress: {idx + 1}/{total_cal} ({pct:.1f}%) "
                    f"[{cached_count} cached, {new_scored} computed]...",
                    flush=True,
                )
    finally:
        cache_handle.close()

    output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = root / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    profiles_created = {}
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                profiles_created = json.load(handle)
        except Exception:
            profiles_created = {}

    for workload_id, w_data in workload_data.items():
        primitive = w_data["primitive"]
        option_ids = w_data["option_ids"]
        logits_list = w_data["logits_list"]
        targets = w_data["targets"]

        # 1. Fit temperature scaling
        fitted_t = fit_temperature_scaling(logits_list, targets)
        fitted_t = round(max(0.1, min(10.0, fitted_t)), 4)
        method = "temperature_scaling"
        bias = None

        # 2. Compute calibrated probabilities helper
        def _compute_calibrated(t: float, b: list[float] | None):
            probs_list = []
            confs = []
            accs = []
            for logits, target in zip(logits_list, targets, strict=True):
                scaled = [z / t for z in logits]
                if b is not None:
                    scaled = [s + bi for s, bi in zip(scaled, b, strict=True)]
                max_z = max(scaled)
                exp_z = [math.exp(s - max_z) for s in scaled]
                sum_exp = sum(exp_z)
                p_vec = [v / sum_exp for v in exp_z]
                probs_list.append(p_vec)
                pred_idx = max(range(len(p_vec)), key=lambda i: p_vec[i])
                confs.append(max(p_vec))
                accs.append(1 if pred_idx == target else 0)
            return probs_list, confs, accs

        calibrated_probs_list, calibrated_confs, calibrated_accs = _compute_calibrated(fitted_t, None)

        # 3. Metrics on calibration split
        ece, bins = compute_ece(calibrated_confs, calibrated_accs, num_bins=10, equal_count=True)
        brier = compute_brier_score(calibrated_probs_list, targets)
        prev_brier = compute_prevalence_brier(targets, len(option_ids))
        nll = compute_nll(calibrated_probs_list, targets)

        # If scalar temperature fails Brier gate, try vector scaling
        if brier > prev_brier:
            fitted_t_vec, fitted_bias = fit_vector_scaling(logits_list, targets, initial_temperature=fitted_t)
            v_probs_list, v_confs, v_accs = _compute_calibrated(fitted_t_vec, fitted_bias)
            v_brier = compute_brier_score(v_probs_list, targets)
            if v_brier < brier:
                fitted_t = fitted_t_vec
                bias = fitted_bias
                method = "vector_scaling"
                calibrated_probs_list, calibrated_confs, calibrated_accs = v_probs_list, v_confs, v_accs
                ece, bins = compute_ece(calibrated_confs, calibrated_accs, num_bins=10, equal_count=True)
                brier = v_brier
                nll = compute_nll(calibrated_probs_list, targets)

        # 4. Acceptance policy
        if workload_id == "support-routing-v1":
            policy = AcceptancePolicy(type="review_only", threshold=0.90)
            selective = compute_selective_metrics(calibrated_confs, calibrated_accs, 1.01)
        else:
            fitted_thresh, _ = fit_acceptance_threshold(
                calibrated_confs, calibrated_accs, target_error=0.05, min_coverage=0.50
            )
            # Find candidate thresholds that meet Clopper-Pearson 95% upper bound <= 0.05 with coverage >= 0.50
            candidates = sorted({round(c, 4) for c in calibrated_confs} | {0.50, 0.70, 0.80, 0.85, 0.90, 0.95})
            passing_thresh = None
            passing_sel = None
            for thresh in candidates:
                sel = compute_selective_metrics(calibrated_confs, calibrated_accs, thresh)
                if sel["coverage"] >= 0.50 and sel["accepted_error_95_upper_bound"] <= 0.05 and sel["accepted_count"] >= 59:
                    passing_thresh = thresh
                    passing_sel = sel
                    break

            if passing_thresh is not None:
                policy = AcceptancePolicy(type="threshold", threshold=round(passing_thresh, 4))
                selective = passing_sel
            else:
                policy = AcceptancePolicy(type="review_only", threshold=round(fitted_thresh, 4))
                selective = compute_selective_metrics(calibrated_confs, calibrated_accs, 1.01)

        gate_ece = ece <= 0.05
        gate_brier = brier <= prev_brier
        gate_selective = (
            selective["coverage"] >= 0.50 and selective["accepted_error_95_upper_bound"] <= 0.05
        )
        samples_sufficient = selective["accepted_count"] >= 59

        metrics_record = CalibrationMetrics(
            brier_score=round(brier, 6),
            prevalence_brier_score=round(prev_brier, 6),
            nll=round(nll, 6),
            ece=round(ece, 6),
            coverage=round(selective["coverage"], 4),
            accepted_error_rate=round(selective["accepted_error_rate"], 4),
            accepted_error_95_upper_bound=round(selective["accepted_error_95_upper_bound"], 4),
            sample_count=len(targets),
            accepted_count=selective["accepted_count"],
            errors_in_accepted=selective["errors_in_accepted"],
            gate_ece_passed=gate_ece,
            gate_brier_passed=gate_brier,
            gate_selective_passed=gate_selective,
            samples_sufficient_for_gate=samples_sufficient,
            reliability_bins=bins,
        )

        cardinality_range = (len(option_ids), len(option_ids)) if primitive == "boolean" else (2, 16)
        profile = CalibrationProfile(
            schema_version=1,
            profile_id=workload_id,
            workload_id=workload_id,
            primitive=primitive,
            artifact_id=model_config["artifact_id"],
            model_id=model_config["model_id"],
            revision=model_config["revision"],
            precision=model_config["precision"],
            prompt_version=model_config["prompt_version"],
            label_version=model_config["label_version"],
            cardinality_range=cardinality_range,
            method=method,
            temperature=fitted_t,
            bias=bias,
            policy=policy,
            dataset_sha256=manifest["dataset_sha256"],
            split="calibration",
            fitted_at=datetime.now(UTC).isoformat(),
            metrics=metrics_record,
        )

        profile_file = output_dir / f"{workload_id}.yaml"
        save_profile(profile, profile_file)
        profiles_created[workload_id] = profile.model_dump(mode="json")

    # If support-routing-v1 was not in calibration split, generate review_only profile
    if (
        "support-routing-v1" not in profiles_created
        and any(e.workload_id == "support-routing-v1" for e in examples)
    ):
        sr_profile = CalibrationProfile(
            schema_version=1,
            profile_id="support-routing-v1",
            workload_id="support-routing-v1",
            primitive="choice",
            artifact_id=model_config["artifact_id"],
            model_id=model_config["model_id"],
            revision=model_config["revision"],
            precision=model_config["precision"],
            prompt_version=model_config["prompt_version"],
            label_version=model_config["label_version"],
            cardinality_range=(2, 16),
            method="temperature_scaling",
            temperature=1.0,
            policy=AcceptancePolicy(type="review_only", threshold=0.90),
            dataset_sha256=manifest["dataset_sha256"],
            split="calibration",
            fitted_at=datetime.now(UTC).isoformat(),
        )
        save_profile(sr_profile, output_dir / "support-routing-v1.yaml")
        profiles_created["support-routing-v1"] = sr_profile.model_dump(mode="json")

    summary_path = root / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(profiles_created, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Calibration completed. Profiles saved to {output_dir}")
    return 0


def _run_equivalence(args: argparse.Namespace) -> int:
    from jah.equivalence import run_equivalence

    return run_equivalence(args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset", default="evals/data/public/suite.jsonl")
    validate.add_argument("--suite-config", default="configs/evals/public-suite.yaml")
    validate.add_argument("--split-seed", default="jah-public-v1")
    validate.add_argument("--output", default="artifacts/m1/suite-validation.json")
    validate.set_defaults(function=validate_dataset)

    run = subparsers.add_parser("run")
    run.add_argument("--dataset", default="evals/data/public/suite.jsonl")
    run.add_argument("--suite-config", default="configs/evals/public-suite.yaml")
    run.add_argument("--split-seed", default="jah-public-v1")
    run.add_argument("--model-config", default="configs/models/qwen3.5-4b.yaml")
    run.add_argument("--backend", choices=("direct", "generative"), required=True)
    run.add_argument(
        "--split",
        choices=("train", "development", "calibration", "locked_test", "task_holdout"),
        default="development",
    )
    run.add_argument("--profiles-dir", default="configs/profiles/public")
    run.add_argument("--use-workload-profiles", action="store_true", default=False)
    run.add_argument("--adapter-dir", default=None, help="Optional path to LoRA adapter bundle")
    run.add_argument("--workload", default=None, help="Optional workload filter")
    run.add_argument("--max-examples", type=int, default=None, help="Optional limit on evaluated examples")
    run.add_argument("--output", required=True)
    run.add_argument("--predictions", required=True)
    run.add_argument("--resume", action="store_true", default=False, help="Resume from existing predictions file")
    run.add_argument("--chunk-size", type=int, default=25, help="Flush predictions and log progress every N examples")
    run.add_argument("--throttle-ms", type=float, default=5.0, help="Delay in ms between requests to throttle resource consumption")
    run.add_argument("--resource-limit", type=float, default=0.80, help="Cap GPU and CPU resource usage to this fraction")
    run.add_argument("--no-resource-limits", action="store_true", default=False)
    run.add_argument(
        "--exclude-reviewed-quarantine",
        default=None,
        help="Path to review audit artifact (e.g. artifacts/review/audit.json) to exclude quarantined rows",
    )
    run.set_defaults(function=run_evaluation)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--dataset", default="evals/data/public/suite.jsonl")
    calibrate.add_argument("--suite-config", default="configs/evals/public-suite.yaml")
    calibrate.add_argument("--split-seed", default="jah-public-v1")
    calibrate.add_argument("--model-config", default="configs/models/qwen3.5-4b.yaml")
    calibrate.add_argument("--output", default="configs/profiles/public")
    calibrate.add_argument("--summary", default="artifacts/public/calibration-summary.json")
    calibrate.add_argument("--workload", default=None, help="Optional workload filter (e.g. banking77-16-intent-v1)")
    calibrate.add_argument("--cache-file", default="artifacts/public/calibration_cache.jsonl", help="Incremental cache path for scored logits")
    calibrate.add_argument("--chunk-size", type=int, default=25, help="Flush cache and log progress every N examples")
    calibrate.add_argument("--throttle-ms", type=float, default=5.0, help="Delay in ms between requests to throttle resource consumption")
    calibrate.add_argument("--resource-limit", type=float, default=0.80, help="Cap GPU and CPU resource usage to this fraction")
    calibrate.add_argument("--no-resource-limits", action="store_true", default=False)
    calibrate.set_defaults(function=run_calibration)

    equivalence = subparsers.add_parser("equivalence")
    equivalence.add_argument("--requests", default="evals/fixtures/equivalence")
    equivalence.add_argument("--model-config", default="configs/models/qwen3.5-4b.yaml")
    equivalence.add_argument("--profiles-dir", default="configs/profiles/public")
    equivalence.add_argument("--adapter-dir", default=None)
    equivalence.add_argument("--microbatch-size", type=int, default=16)
    equivalence.add_argument("--disable-prefix-cache", action="store_true", default=False)
    equivalence.add_argument("--margin-threshold", type=float, default=2.80, help="Top-1 logit margin threshold for tie band (default 2.80 derived from empirical BF16 GEMM variance)")
    equivalence.add_argument("--output", default="artifacts/m3/equivalence.json")
    equivalence.add_argument("--rows", default="artifacts/m3/equivalence.jsonl")
    equivalence.add_argument("--throttle-ms", type=float, default=5.0)
    equivalence.add_argument("--resource-limit", type=float, default=0.80)
    equivalence.add_argument("--no-resource-limits", action="store_true", default=False)
    equivalence.set_defaults(function=_run_equivalence)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--direct-report", required=True)
    compare.add_argument("--generative-report", required=True)
    compare.add_argument("--direct-predictions", required=True)
    compare.add_argument("--generative-predictions", required=True)
    compare.add_argument("--output", default="artifacts/m1/paired-comparison.json")
    compare.set_defaults(function=compare_reports)

    drift = subparsers.add_parser("drift")
    drift.add_argument("--predictions", required=True, help="Path to predictions JSON or JSONL file")
    drift.add_argument("--profile", required=True, help="Path to calibration profile YAML file")
    drift.add_argument("--warning-ratio", type=float, default=0.80)
    drift.add_argument("--min-coverage", type=float, default=0.50)
    drift.add_argument("--output", default=None, help="Optional output path for drift report JSON")
    from jah.drift import run_drift_check
    drift.set_defaults(function=run_drift_check)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
