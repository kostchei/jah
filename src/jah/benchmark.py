"""Fixed-shape warm HTTP latency protocol for the M1 reference service."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from jah.schemas import BooleanQuestion, EvaluateRequest, EvaluateResponse
from jah.workload import sha256_file, stable_json_sha256


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[int(fraction * (len(values) - 1))]


def validate_benchmark_shape(
    request: EvaluateRequest,
    *,
    expected_questions: int,
) -> None:
    if len(request.questions) != expected_questions:
        raise ValueError(
            f"benchmark requires {expected_questions} questions; found {len(request.questions)}"
        )
    if any(not isinstance(question, BooleanQuestion) for question in request.questions.values()):
        raise ValueError("the M1 latency benchmark requires Boolean questions only")


def run(args: argparse.Namespace) -> int:
    if not getattr(args, "smoke", False):
        if args.warmup_requests < 20:
            raise ValueError("the M1 protocol requires at least 20 warmup requests")
        if args.measured_requests < 200:
            raise ValueError("the M1 protocol requires at least 200 measured requests")

    request_path = Path(args.request).resolve()
    request = EvaluateRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    validate_benchmark_shape(request, expected_questions=args.expected_questions)
    payload = request.model_dump(mode="json")
    client_latencies = []
    server_totals = []
    failures = []
    artifact_ids = set()
    observed_state_tokens = set()

    with httpx.Client(base_url=args.base_url, timeout=args.timeout_seconds) as client:
        ready = client.get("/health/ready")
        ready.raise_for_status()
        for _ in range(args.warmup_requests):
            response = client.post("/v1/evaluate", json=payload)
            response.raise_for_status()

        for index in range(args.measured_requests):
            started = time.perf_counter()
            try:
                raw_response = client.post("/v1/evaluate", json=payload)
                elapsed_ms = (time.perf_counter() - started) * 1_000
                raw_response.raise_for_status()
                response = EvaluateResponse.model_validate(raw_response.json())
            except Exception as exc:  # noqa: BLE001 - failures belong in benchmark evidence
                failures.append({"request_index": index, "error": type(exc).__name__})
                continue
            client_latencies.append(elapsed_ms)
            server_totals.append(response.timing_ms.total)
            artifact_ids.add(response.artifact_id)
            observed_state_tokens.add(response.usage.unique_state_tokens)
            if response.usage.decisions != args.expected_questions:
                raise ValueError("server decision count does not match the benchmark shape")

    if failures:
        raise RuntimeError(f"{len(failures)} measured benchmark requests failed")
    if observed_state_tokens != {args.expected_state_tokens}:
        raise ValueError(
            "benchmark state token count mismatch: "
            f"expected {args.expected_state_tokens}, observed {sorted(observed_state_tokens)}"
        )
    if len(artifact_ids) != 1:
        raise ValueError(f"benchmark observed multiple artifacts: {sorted(artifact_ids)}")

    result = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "warmup_requests": args.warmup_requests,
            "measured_requests": args.measured_requests,
            "concurrency": 1,
            "state_tokens": args.expected_state_tokens,
            "questions_per_request": args.expected_questions,
            "primitive": "boolean",
        },
        "request_sha256": sha256_file(request_path),
        "artifact_id": next(iter(artifact_ids)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "base_url": args.base_url,
        },
        "latency_ms": {
            "client_median": statistics.median(client_latencies),
            "client_p95": _percentile(client_latencies, 0.95),
            "server_median": statistics.median(server_totals),
            "server_p95": _percentile(server_totals, 0.95),
        },
        "failures": failures,
        "p95_two_second_gate_passed": _percentile(client_latencies, 0.95) <= 2_000,
    }
    result["evidence_sha256"] = stable_json_sha256(
        {key: value for key, value in result.items() if key not in {"created_at", "evidence_sha256"}}
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["p95_two_second_gate_passed"] else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="artifacts/m1/http-latency.json")
    parser.add_argument("--warmup-requests", type=int, default=20)
    parser.add_argument("--measured-requests", type=int, default=200)
    parser.add_argument("--expected-state-tokens", type=int, default=2_048)
    parser.add_argument("--expected-questions", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--smoke", action="store_true", help="Allow smaller warmup and measurement counts for smoke testing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
