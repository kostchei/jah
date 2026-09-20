"""Optimization-equivalence measurement for the ADR-03 / section 7 gate.

The reference path is single-item, full-prompt scoring (ADR-02). The optimized path is
shared-prefix reuse plus microbatching (ADR-02, ADR-03). This module runs both over an
identical frozen regression set on the real backbone and records argmax agreement, maximum
probability deviation, and acceptance/review policy flips.

Mocked unit tests cannot discharge this gate: the quantities it measures only exist once the
pinned weights are resident on the device.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jah.calibration import load_profile_registry
from jah.engine import DecisionEngine, EngineConfig
from jah.schemas import BooleanAnswer, ChoiceAnswer, EvaluateRequest, ScoreAnswer
from jah.workload import git_commit_sha, load_yaml, sha256_file, stable_json_sha256

ARGMAX_AGREEMENT_GATE = 0.995
MAXIMUM_PROBABILITY_DEVIATION_GATE = 0.01


def _selected_id(answer: BooleanAnswer | ChoiceAnswer | ScoreAnswer) -> str:
    if isinstance(answer, BooleanAnswer):
        return "true" if answer.value else "false"
    if isinstance(answer, ChoiceAnswer):
        return answer.value
    if isinstance(answer, ScoreAnswer):
        return answer.level_id
    raise TypeError(f"unsupported answer type: {type(answer).__name__}")


def load_regression_requests(directory: Path) -> list[tuple[str, Path, EvaluateRequest]]:
    """Load every checked-in regression request, sorted by file name."""
    paths = [path for path in sorted(directory.glob("*.json")) if path.name != "manifest.json"]
    if not paths:
        raise ValueError(f"no regression requests found in {directory}")
    loaded = []
    for path in paths:
        request = EvaluateRequest.model_validate_json(path.read_text(encoding="utf-8"))
        if len(request.questions) < 2:
            raise ValueError(
                f"{path.name}: the equivalence regression set requires multi-question requests; "
                "single-question requests never reach the optimized path"
            )
        loaded.append((path.stem, path, request))
    return loaded


def compare_responses(
    reference: Any,
    optimized: Any,
    *,
    request_name: str,
) -> list[dict[str, Any]]:
    """Compare two responses for the same request, one row per decision."""
    if set(reference.answers) != set(optimized.answers):
        raise ValueError(f"{request_name}: response answer keys diverged between paths")

    rows = []
    for question_id in sorted(reference.answers):
        reference_answer = reference.answers[question_id]
        optimized_answer = optimized.answers[question_id]
        if set(reference_answer.probabilities) != set(optimized_answer.probabilities):
            raise ValueError(f"{request_name}/{question_id}: label sets diverged between paths")

        deviation = max(
            abs(reference_answer.probabilities[label] - optimized_answer.probabilities[label])
            for label in reference_answer.probabilities
        )
        reference_selected = _selected_id(reference_answer)
        optimized_selected = _selected_id(optimized_answer)
        rows.append(
            {
                "request": request_name,
                "question_id": question_id,
                "primitive": reference_answer.type,
                "reference_selected": reference_selected,
                "optimized_selected": optimized_selected,
                "argmax_agrees": reference_selected == optimized_selected,
                "maximum_probability_deviation": deviation,
                "reference_disposition": reference_answer.disposition,
                "optimized_disposition": optimized_answer.disposition,
                "policy_flip": reference_answer.disposition != optimized_answer.disposition,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("equivalence produced no compared decisions")
    agreements = sum(1 for row in rows if row["argmax_agrees"])
    agreement = agreements / len(rows)
    deviation = max(row["maximum_probability_deviation"] for row in rows)
    flips = sum(1 for row in rows if row["policy_flip"])
    gate_argmax_passed = agreement >= ARGMAX_AGREEMENT_GATE
    gate_deviation_passed = deviation <= MAXIMUM_PROBABILITY_DEVIATION_GATE
    gate_policy_passed = flips == 0
    return {
        "compared_decisions": len(rows),
        "argmax_agreements": agreements,
        "argmax_agreement": agreement,
        "maximum_probability_deviation": deviation,
        "policy_flips": flips,
        "argmax_agreement_gate": ARGMAX_AGREEMENT_GATE,
        "maximum_probability_deviation_gate": MAXIMUM_PROBABILITY_DEVIATION_GATE,
        "gate_argmax_passed": gate_argmax_passed,
        "gate_deviation_passed": gate_deviation_passed,
        "gate_policy_passed": gate_policy_passed,
        "gate_passed": gate_argmax_passed and gate_deviation_passed and gate_policy_passed,
    }


def run_equivalence(args: argparse.Namespace) -> int:
    from jah.evaluation import load_backend

    root = Path(args.root).resolve()
    requests_dir = root / args.requests
    regression = load_regression_requests(requests_dir)

    if not getattr(args, "no_resource_limits", False):
        from jah.resource import configure_resource_limits

        limits = configure_resource_limits(
            max_total_gpu_fraction=getattr(args, "resource_limit", 0.80),
            max_cpu_fraction=getattr(args, "resource_limit", 0.80),
        )
        print(f"Resource limits configured: {limits}", flush=True)

    model_path = root / args.model_config
    model_config = load_yaml(model_path)
    backend = load_backend("direct", model_config, getattr(args, "adapter_dir", None))
    if not hasattr(backend, "score_batch"):
        raise RuntimeError("the loaded backend exposes no optimized path to compare against")

    profiles = {}
    profiles_dir = root / args.profiles_dir
    if profiles_dir.exists():
        profiles = load_profile_registry(profiles_dir)

    model_metadata = {
        "model_id": model_config["model_id"],
        "revision": model_config["revision"],
        "precision": model_config["precision"],
        "prompt_version": model_config["prompt_version"],
        "label_version": model_config["label_version"],
    }
    base_config = {
        "artifact_id": model_config["artifact_id"],
        "maximum_input_tokens": model_config["maximum_input_tokens"],
        "profiles": profiles,
        "model_metadata": model_metadata,
    }
    reference_engine = DecisionEngine(
        backend, EngineConfig(**base_config, force_sequential=True, enable_prefix_cache=False)
    )
    optimized_engine = DecisionEngine(
        backend,
        EngineConfig(
            **base_config,
            force_sequential=False,
            enable_prefix_cache=not args.disable_prefix_cache,
            microbatch_size=args.microbatch_size,
        ),
    )

    rows: list[dict[str, Any]] = []
    per_request = []
    for index, (name, path, request) in enumerate(regression, start=1):
        print(f"Equivalence: {index}/{len(regression)} {name}...", flush=True)
        started = time.perf_counter()
        reference_response = reference_engine.evaluate(request, request_id=f"ref_{name}")
        reference_ms = (time.perf_counter() - started) * 1_000

        started = time.perf_counter()
        optimized_response = optimized_engine.evaluate(request, request_id=f"opt_{name}")
        optimized_ms = (time.perf_counter() - started) * 1_000

        request_rows = compare_responses(
            reference_response, optimized_response, request_name=name
        )
        rows.extend(request_rows)
        if getattr(args, "throttle_ms", 5.0) > 0:
            from jah.resource import throttle

            throttle(getattr(args, "throttle_ms", 5.0))
        per_request.append(
            {
                "request": name,
                "request_sha256": sha256_file(path),
                "decisions": len(request_rows),
                "state_tokens": reference_response.usage.unique_state_tokens,
                "reference_total_ms": reference_ms,
                "optimized_total_ms": optimized_ms,
                "speedup": reference_ms / optimized_ms,
                **summarize(request_rows),
            }
        )

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit_sha": git_commit_sha(root),
        "artifact_id": model_config["artifact_id"],
        "model": model_metadata,
        "reference_path": "sequential full-prompt single-item scoring (ADR-02)",
        "optimized_path": (
            "shared-prefix reuse"
            if not args.disable_prefix_cache
            else "microbatching only (prefix reuse disabled)"
        ),
        "microbatch_size": args.microbatch_size,
        "regression_set": {
            "directory": Path(args.requests).as_posix(),
            "requests": len(regression),
        },
        "per_request": per_request,
        "summary": summarize(rows),
    }
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

    rows_path = root / args.rows
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["gate_passed"] else 2
