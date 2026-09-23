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
import math
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
DEFAULT_MARGIN_THRESHOLD = 2.80  # Diagnostic threshold only; never exempts a decision from gates.
MAXIMUM_RESERVED_VRAM_BYTES = 22_000_000_000


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
    reference_measurements: dict[str, Any] | None = None,
    optimized_measurements: dict[str, Any] | None = None,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
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

        reference_margin = float("inf")
        if reference_measurements and question_id in reference_measurements:
            meas = reference_measurements[question_id]
            if hasattr(meas, "decision") and meas.decision and meas.decision.logits:
                logits = meas.decision.logits
                if len(logits) >= 2:
                    sorted_z = sorted(logits.values(), reverse=True)
                    reference_margin = sorted_z[0] - sorted_z[1]

        in_tie_band = reference_margin <= margin_threshold

        rows.append(
            {
                "request": request_name,
                "question_id": question_id,
                "primitive": reference_answer.type,
                "reference_selected": reference_selected,
                "optimized_selected": optimized_selected,
                "argmax_agrees": reference_selected == optimized_selected,
                "maximum_probability_deviation": deviation,
                "reference_margin": reference_margin if math.isfinite(reference_margin) else None,
                "in_tie_band": in_tie_band,
                "reference_disposition": reference_answer.disposition,
                "optimized_disposition": optimized_answer.disposition,
                "policy_flip": reference_answer.disposition != optimized_answer.disposition,
            }
        )
    return rows


def summarize(
    rows: list[dict[str, Any]],
    *,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    max_probability_deviation_gate: float = MAXIMUM_PROBABILITY_DEVIATION_GATE,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("equivalence produced no compared decisions")
    agreements = sum(1 for row in rows if row["argmax_agrees"])
    agreement = agreements / len(rows)
    deviation = max(row["maximum_probability_deviation"] for row in rows)
    flips = sum(1 for row in rows if row["policy_flip"])

    # Keep the historical tie-band breakdown for diagnostics, but gate every decision.
    # A top-two logit margin does not provide a distribution-independent bound on
    # multiclass probability deviation, so it cannot safely exempt cases from release gates.
    tie_band_rows = [row for row in rows if row.get("in_tie_band", False)]
    gated_rows = rows

    tie_band_count = len(tie_band_rows)
    tie_band_fraction = tie_band_count / len(rows)

    if gated_rows:
        gated_agreements = sum(1 for row in gated_rows if row["argmax_agrees"])
        gated_agreement = gated_agreements / len(gated_rows)
        gated_deviation = max(row["maximum_probability_deviation"] for row in gated_rows)
        gate_argmax_passed = gated_agreements == len(gated_rows)
        gate_deviation_passed = gated_deviation <= max_probability_deviation_gate
    else:
        gated_agreements = 0
        gated_agreement = 1.0
        gated_deviation = 0.0
        gate_argmax_passed = True
        gate_deviation_passed = True

    gate_policy_passed = flips == 0
    gate_passed = gate_argmax_passed and gate_deviation_passed and gate_policy_passed

    return {
        "compared_decisions": len(rows),
        "argmax_agreements": agreements,
        "argmax_agreement": agreement,
        "maximum_probability_deviation": deviation,
        "policy_flips": flips,
        "margin_threshold": margin_threshold,
        "tie_band_count": tie_band_count,
        "tie_band_fraction": tie_band_fraction,
        "gated_decisions": len(gated_rows),
        "gated_argmax_agreements": gated_agreements,
        "gated_argmax_agreement": gated_agreement,
        "gated_maximum_probability_deviation": gated_deviation,
        "argmax_agreement_gate": ARGMAX_AGREEMENT_GATE,
        "maximum_probability_deviation_gate": max_probability_deviation_gate,
        "gate_argmax_passed": gate_argmax_passed,
        "gate_deviation_passed": gate_deviation_passed,
        "gate_policy_passed": gate_policy_passed,
        "gate_passed": gate_passed,
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
    if getattr(args, "precision", None):
        model_config = {**model_config, "precision": args.precision}
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
    reference_reserved_samples: list[int] = []
    reference_allocated_samples: list[int] = []
    optimized_reserved_samples: list[int] = []
    optimized_allocated_samples: list[int] = []
    for index, (name, path, request) in enumerate(regression, start=1):
        print(f"Equivalence: {index}/{len(regression)} {name}...", flush=True)
        started = time.perf_counter()
        reference_response, ref_measurements = reference_engine.evaluate_detailed(
            request, request_id=f"ref_{name}"
        )
        reference_ms = (time.perf_counter() - started) * 1_000

        started = time.perf_counter()
        optimized_response, opt_measurements = optimized_engine.evaluate_detailed(
            request, request_id=f"opt_{name}"
        )
        optimized_ms = (time.perf_counter() - started) * 1_000
        reference_reserved_samples.extend(
            m.peak_vram_reserved_bytes for m in ref_measurements.values()
        )
        reference_allocated_samples.extend(m.peak_vram_bytes for m in ref_measurements.values())
        optimized_reserved_samples.extend(
            m.peak_vram_reserved_bytes for m in opt_measurements.values()
        )
        optimized_allocated_samples.extend(
            m.peak_vram_bytes for m in opt_measurements.values()
        )

        margin_thresh = getattr(args, "margin_threshold", DEFAULT_MARGIN_THRESHOLD)
        request_rows = compare_responses(
            reference_response,
            optimized_response,
            request_name=name,
            reference_measurements=ref_measurements,
            optimized_measurements=opt_measurements,
            margin_threshold=margin_thresh,
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
                **summarize(request_rows, margin_threshold=margin_thresh),
            }
        )

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit_sha": git_commit_sha(root),
        "artifact_id": model_config["artifact_id"],
        "model": model_metadata,
        "reference_path": "sequential full-prompt single-item scoring (ADR-02)",
        "optimized_path": getattr(
            backend,
            "batch_optimization_mode",
            "full-prompt microbatch (prefix reuse disabled)"
            if args.disable_prefix_cache
            else "shared-prefix reuse or full-prompt microbatch",
        ),
        "microbatch_size": args.microbatch_size,
        "regression_set": {
            "directory": Path(args.requests).as_posix(),
            "requests": len(regression),
        },
        "per_request": per_request,
        "summary": summarize(rows, margin_threshold=margin_thresh),
    }
    is_cuda = getattr(getattr(backend, "device", None), "type", None) == "cuda"
    reference_reserved = max(reference_reserved_samples, default=0)
    optimized_reserved = max(optimized_reserved_samples, default=0)
    optimized_allocated = max(optimized_allocated_samples, default=0)
    device_total_memory = (
        int(backend.torch.cuda.get_device_properties(backend.device).total_memory)
        if is_cuda and hasattr(backend, "torch")
        else None
    )
    report["memory"] = {
        "measurement": "PyTorch peak CUDA allocator reservation during equivalence run",
        "reference_peak_reserved_bytes": reference_reserved,
        "reference_peak_allocated_bytes": max(reference_allocated_samples, default=0),
        "optimized_peak_allocated_bytes": optimized_allocated,
        "optimized_peak_reserved_bytes": optimized_reserved,
        "device_total_memory_bytes": device_total_memory,
        "limit_bytes": MAXIMUM_RESERVED_VRAM_BYTES,
        "gate_passed": bool(
            is_cuda
            and reference_reserved > 0
            and optimized_reserved > 0
            and reference_reserved <= MAXIMUM_RESERVED_VRAM_BYTES
            and optimized_reserved <= MAXIMUM_RESERVED_VRAM_BYTES
        ),
    }
    report["gate_passed"] = (
        report["summary"]["gate_passed"] and report["memory"]["gate_passed"]
    )
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

    print(
        json.dumps(
            {
                "equivalence": report["summary"],
                "memory": report["memory"],
                "gate_passed": report["gate_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["gate_passed"] else 2
