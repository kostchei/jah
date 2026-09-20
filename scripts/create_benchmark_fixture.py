"""Generate an exact 2,048-token, 16-Boolean benchmark fixture for jah-benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from jah.schemas import BooleanQuestion, EvaluateRequest
from jah.workload import load_yaml


def generate_benchmark_request(
    tokenizer,
    *,
    target_tokens: int = 2_048,
    num_questions: int = 16,
) -> EvaluateRequest:
    # Build a realistic structured system event log
    base_paragraph = (
        "INCIDENT REPORT - CLUSTER US-EAST-4B\n"
        "Timestamp: 2026-09-20T08:14:22Z | Severity: SEV-2 | Status: MITIGATED\n"
        "Component: ingress-controller-envoy-v2 | Service: billing-transaction-pipeline\n"
        "Description: High HTTP 504 gateway timeout rate observed following canary deployment 4.18.2.\n"
        "Root Cause: Database connection pool exhaustion caused by misconfigured keepalive idle timeout.\n"
        "Action Taken: Traffic shifted back to stable deployment 4.18.1; connection pool max_size doubled to 256.\n"
        "Impact: 1,420 checkout requests failed between 08:02Z and 08:14Z. No data corruption detected.\n"
        "Follow-up: Audit all service keepalive defaults and implement circuit breaker backpressure.\n"
        "Telemetry snapshot: CPU utilization peaked at 78.4%, memory steady at 14.2 GB of 32 GB allocatable.\n"
        "Network egress rate: 420 Mbps peak, packet drop rate 0.002% on eth0.\n\n"
    )

    # Repeat paragraph until we exceed target tokens
    text = base_paragraph * 25
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < target_tokens:
        text = base_paragraph * 40
        tokens = tokenizer.encode(text, add_special_tokens=False)

    # Truncate tokens to target and decode
    truncated_tokens = tokens[:target_tokens]
    state = tokenizer.decode(truncated_tokens)

    # Check for boundary issues where re-encoding might shift token count by 1 or 2
    encoded = tokenizer.encode(state, add_special_tokens=False)
    while len(encoded) > target_tokens:
        truncated_tokens = truncated_tokens[:-1]
        state = tokenizer.decode(truncated_tokens)
        encoded = tokenizer.encode(state, add_special_tokens=False)

    while len(encoded) < target_tokens:
        # Append single token
        truncated_tokens.append(tokens[len(truncated_tokens)])
        state = tokenizer.decode(truncated_tokens)
        encoded = tokenizer.encode(state, add_special_tokens=False)

    assert len(encoded) == target_tokens, f"Expected {target_tokens} tokens, got {len(encoded)}"

    questions = {}
    propositions = [
        "The incident severity was SEV-2.",
        "The affected cluster was US-WEST-2A.",
        "The canary deployment version was 4.18.2.",
        "Traffic was shifted back to stable deployment 4.18.1.",
        "Database connection pool exhaustion caused the timeouts.",
        "Over 5,000 checkout requests failed during the incident.",
        "No data corruption was detected during the incident.",
        "The incident status is currently OPEN and unmitigated.",
        "Memory utilization exceeded 95% of allocatable capacity.",
        "CPU utilization peaked at 78.4%.",
        "The connection pool max_size was increased to 256.",
        "Packet drop rate exceeded 10% on interface eth0.",
        "The incident occurred on September 20, 2026.",
        "The follow-up action includes auditing service keepalive defaults.",
        "The service affected was the billing-transaction-pipeline.",
        "Network egress rate reached 10 Gbps peak.",
    ]

    for index, prop in enumerate(propositions[:num_questions]):
        q_id = f"q_{index:02d}"
        questions[q_id] = BooleanQuestion(
            type="boolean",
            instructions="Determine whether the proposition is supported by the incident log.",
            proposition=prop,
        )

    return EvaluateRequest(
        state=state,
        questions=questions,
        profile="support-routing-v1",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/models/qwen3.5-4b.yaml")
    parser.add_argument("--output", default="evals/fixtures/benchmark-2048-16b.json")
    parser.add_argument("--tokens", type=int, default=2_048)
    parser.add_argument("--questions", type=int, default=16)
    args = parser.parse_args()

    config = load_yaml(Path(args.model_config))
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"])

    request = generate_benchmark_request(
        tokenizer, target_tokens=args.tokens, num_questions=args.questions
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(request.model_dump(mode="json"), handle, indent=2)

    print(f"Generated benchmark request at {out_path} with {len(request.questions)} questions.")
    verified_tokens = len(tokenizer.encode(request.state, add_special_tokens=False))
    print(f"Verified state token count: {verified_tokens}")


if __name__ == "__main__":
    main()
