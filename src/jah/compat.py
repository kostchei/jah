"""M0 compatibility and feasibility runner."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from jah.compiler import compile_request
from jah.workload import (
    load_dataset,
    load_yaml,
    sha256_file,
    stable_json_sha256,
    validate_dataset_against_workload,
)


def _gpu_inventory() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    inventory = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, memory_mib, driver, compute_capability = [item.strip() for item in line.split(",")]
        inventory.append(
            {
                "name": name,
                "memory_mib": memory_mib,
                "driver": driver,
                "compute_capability": compute_capability,
            }
        )
    return inventory


def _macro_f1(references: list[str], predictions: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        true_positive = sum(r == label and p == label for r, p in zip(references, predictions))
        false_positive = sum(r != label and p == label for r, p in zip(references, predictions))
        false_negative = sum(r == label and p != label for r, p in zip(references, predictions))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return statistics.fmean(scores)


def _per_label_metrics(
    references: list[str], predictions: list[str], labels: list[str]
) -> dict[str, dict[str, float | int]]:
    metrics = {}
    for label in labels:
        true_positive = sum(r == label and p == label for r, p in zip(references, predictions))
        false_positive = sum(r != label and p == label for r, p in zip(references, predictions))
        support = sum(r == label for r in references)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {
            "support": support,
            "correct": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    model_path = root / args.model_config
    workload_path = root / args.workload_config
    model_config = load_yaml(model_path)
    workload = load_yaml(workload_path)
    dataset_path = root / workload["dataset"]
    examples = load_dataset(dataset_path)
    validate_dataset_against_workload(examples, workload)
    dataset_sha256 = sha256_file(dataset_path)
    if dataset_sha256 != workload["dataset_sha256"]:
        raise ValueError(
            "dataset hash differs from the frozen workload definition: "
            f"expected {workload['dataset_sha256']}, observed {dataset_sha256}"
        )

    import psutil
    import torch
    import transformers
    from transformers import AutoTokenizer

    report: dict = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "host_ram_bytes": psutil.virtual_memory().total,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpus": _gpu_inventory(),
        },
        "model": model_config,
        "model_config_sha256": sha256_file(model_path),
        "workload_config_sha256": sha256_file(workload_path),
        "dataset": {
            "path": str(dataset_path.relative_to(root)).replace("\\", "/"),
            "sha256": dataset_sha256,
            "examples": len(examples),
            "label_counts": dict(Counter(example.reference_answer for example in examples)),
            "annotation_status_counts": dict(
                Counter(example.annotation_status for example in examples)
            ),
        },
    }

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("M0 requires CUDA, but the active PyTorch build cannot access a GPU")

    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"], revision=model_config["revision"]
    )
    compiled = [
        compile_request(
            example.as_request(workload["instructions"]),
            tokenizer=tokenizer,
            max_input_tokens=model_config["maximum_input_tokens"],
        )[0]
        for example in examples
    ]
    report["tokenizer"] = {
        "class": tokenizer.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
        "load_ms": (time.perf_counter() - tokenizer_started) * 1_000,
        "verified_label_token_ids": list(compiled[0].label_token_ids or ()),
        "maximum_compiled_tokens": max(item.input_tokens or 0 for item in compiled),
        "all_label_mappings_identical": len({item.label_token_ids for item in compiled}) == 1,
    }

    if args.tokenizer_only:
        report["status"] = "tokenizer-pass"
    else:
        from jah.backends.huggingface import HuggingFaceDirectLogitBackend

        load_started = time.perf_counter()
        backend = HuggingFaceDirectLogitBackend(
            model_config["model_id"],
            model_config["revision"],
            device=model_config["device"],
        )
        model_load_ms = (time.perf_counter() - load_started) * 1_000

        predictions = []
        references = []
        prediction_rows = []
        latencies = []
        peak_vram_bytes = 0
        for example, question in zip(examples, compiled, strict=True):
            measurement = backend.score(question)
            predictions.append(measurement.decision.selected_id)
            references.append(example.reference_answer)
            latencies.append(measurement.inference_ms)
            peak_vram_bytes = max(peak_vram_bytes, measurement.peak_vram_bytes)
            prediction_rows.append(
                {
                    "example_id": example.example_id,
                    "reference": example.reference_answer,
                    "prediction": measurement.decision.selected_id,
                    "probabilities": measurement.decision.probabilities,
                    "input_tokens": measurement.input_tokens,
                    "inference_ms": measurement.inference_ms,
                }
            )

        labels = [option["id"] for option in workload["options"]]
        accuracy = sum(r == p for r, p in zip(references, predictions)) / len(references)
        majority = max(Counter(references).values()) / len(references)
        macro_f1 = _macro_f1(references, predictions, labels)
        gate = workload["provisional_feasibility_gate"]
        feasible = (
            accuracy >= gate["minimum_accuracy"]
            and accuracy - majority >= gate["minimum_margin_over_majority"]
        )
        report["inference"] = {
            "model_load_ms": model_load_ms,
            "decisions": len(predictions),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "majority_accuracy": majority,
            "margin_over_majority": accuracy - majority,
            "per_label": _per_label_metrics(references, predictions, labels),
            "median_inference_ms": statistics.median(latencies),
            "p95_inference_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
            "peak_vram_bytes": peak_vram_bytes,
            "feasibility_gate_passed": feasible,
        }
        prediction_path = root / args.predictions
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        report["predictions_sha256"] = sha256_file(prediction_path)
        report["status"] = "pass" if feasible else "feasibility-gate-failed"

    report["evidence_sha256"] = stable_json_sha256(
        {
            key: value
            for key, value in report.items()
            if key not in {"created_at", "evidence_sha256"}
        }
    )
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"pass", "tokenizer-pass"} else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path.cwd())
    parser.add_argument("--model-config", default="configs/models/qwen3.5-4b.yaml")
    parser.add_argument("--workload-config", default="configs/workloads/support-routing-v1.yaml")
    parser.add_argument("--output", default="artifacts/m0/compatibility.json")
    parser.add_argument("--predictions", default="artifacts/m0/predictions.jsonl")
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
