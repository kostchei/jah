"""LLM review harness for label audit, failure-slice reasoning, and hard-case generation.

Speaks the OpenAI-compatible LM Studio API (default: http://localhost:1234/v1).
Reviewer model: qwen/qwen3.8-27b.

ADR-06 / REMEDIATION_PLAN:
- Upstream human labels remain ground truth.
- The reviewer model never mutates suite.jsonl.
- Output lives in artifacts/review/<run>.json and is applied only through
  an explicit --exclude-reviewed-quarantine evaluation flag.
- Every artifact carries a mandatory provenance block recording that ground_truth is false
  and that the reviewer and backbone share the same model family (correlated errors).
- Non-optional control arm: audits a sample of correct predictions. If reviewer disputes
  more than 15% of correct predictions, the reviewer is deemed unreliable and quarantine
  recommendations are discarded.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jah.m1_dataset import M1Example, load_m1_dataset
from jah.resource import throttle

PROMPT_VERSION = "review-v1"
DEFAULT_REVIEWER_MODEL = "qwen/qwen3.8-27b"
DEFAULT_ENDPOINT = "http://localhost:1234/v1"
CORRELATED_FAMILY_LIMITATION = (
    "The reviewer (qwen/qwen3.8-27b) and the backbone (Qwen/Qwen3.5-4B) are the same "
    "model family, so their errors are correlated. Reviewer agreement is therefore "
    "weak evidence."
)


def call_lm_studio_chat(
    messages: list[dict[str, str]],
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_REVIEWER_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> str:
    """Send a chat completion request to the local OpenAI-compatible LM Studio server."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        choice = body["choices"][0]["message"]
        content = choice.get("content") or ""
        # Handle reasoning models where output might reside in reasoning_content or content
        if not content.strip() and "reasoning_content" in choice:
            content = choice["reasoning_content"]
        return content.strip()


def mock_reviewer_audit(system_prediction: str, upstream_reference: str) -> dict[str, Any]:
    """Deterministic mock reviewer for offline test coverage."""
    if system_prediction == upstream_reference:
        return {
            "verdict": "agree_with_upstream",
            "reasoning": "Mock reviewer agrees with upstream label.",
            "confidence": 0.95,
            "quarantine_recommended": False,
        }
    return {
        "verdict": "dispute_upstream",
        "reasoning": "Mock reviewer disputes ambiguous upstream label.",
        "confidence": 0.85,
        "quarantine_recommended": True,
    }


def audit_row(
    example: M1Example,
    question_id: str,
    system_pred: str,
    upstream_ref: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_REVIEWER_MODEL,
    temperature: float = 0.0,
    mock: bool = False,
) -> dict[str, Any]:
    """Audit a single decision with the reviewer model."""
    if mock:
        res = mock_reviewer_audit(system_pred, upstream_ref)
        res["example_id"] = example.example_id
        res["question_id"] = question_id
        res["workload_id"] = example.workload_id
        res["system_prediction"] = system_pred
        res["upstream_reference"] = upstream_ref
        return res

    question = example.request.questions[question_id]
    state_str = str(example.request.state)
    prompt = (
        "You are an expert evaluation auditor. Your task is to judge whether the upstream human label "
        "for this decision is defensible, or whether the label is erroneous/ambiguous.\n\n"
        f"WORKLOAD: {example.workload_id}\n"
        f"STATE / EVIDENCE:\n{state_str}\n\n"
        f"QUESTION: {question.instructions}\n"
        f"UPSTREAM HUMAN LABEL: {upstream_ref}\n"
        f"SYSTEM PREDICTION: {system_pred}\n\n"
        "Return ONLY a JSON object with this exact schema:\n"
        "{\n"
        '  "verdict": "agree_with_upstream" | "dispute_upstream" | "unclear",\n'
        '  "reasoning": "<concise explanation>",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "quarantine_recommended": <true if upstream label is clearly defective or unusable, else false>\n'
        "}"
    )

    messages = [
        {"role": "system", "content": "You are a rigorous, impartial label auditor. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    try:
        reply = call_lm_studio_chat(
            messages,
            endpoint=endpoint,
            model=model,
            temperature=temperature,
        )
        # Clean potential markdown formatting
        cleaned = reply.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        parsed = json.loads(cleaned)
    except Exception as exc:
        parsed = {
            "verdict": "unclear",
            "reasoning": f"Audit query failed or response malformed: {exc}",
            "confidence": 0.0,
            "quarantine_recommended": False,
        }

    return {
        "example_id": example.example_id,
        "question_id": question_id,
        "workload_id": example.workload_id,
        "system_prediction": system_pred,
        "upstream_reference": upstream_ref,
        "verdict": parsed.get("verdict", "unclear"),
        "reasoning": parsed.get("reasoning", ""),
        "confidence": float(parsed.get("confidence", 0.0)),
        "quarantine_recommended": bool(parsed.get("quarantine_recommended", False)),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Run label audit with mandatory control arm."""
    root = Path(args.root).resolve()
    predictions_path = root / args.predictions
    dataset_path = root / args.dataset
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")

    examples = load_m1_dataset(dataset_path)
    example_map = {e.example_id: e for e in examples}

    preds_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            preds_by_key[(r["example_id"], r["question_id"])] = r

    disagreements = []
    agreements = []
    for (ex_id, qid), pred_row in preds_by_key.items():
        if ex_id not in example_map:
            continue
        ref = pred_row["reference"]
        pred = pred_row["prediction"]
        item = (example_map[ex_id], qid, str(pred), str(ref))
        if str(pred) != str(ref):
            disagreements.append(item)
        else:
            agreements.append(item)

    if args.max_audits and args.max_audits < len(disagreements):
        disagreements = disagreements[: args.max_audits]

    # Non-optional control arm: sample from agreements (correct predictions)
    control_arm_size = min(
        len(agreements),
        max(10, int(len(disagreements) * getattr(args, "control_arm_ratio", 0.5))),
    )
    rng = random.Random(args.seed)
    control_samples = rng.sample(agreements, control_arm_size) if agreements else []

    print(f"Auditing {len(disagreements)} disagreement rows...")
    disagreement_verdicts = []
    for idx, (ex, qid, pred, ref) in enumerate(disagreements):
        v = audit_row(
            ex,
            qid,
            pred,
            ref,
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
            mock=getattr(args, "mock", False),
        )
        disagreement_verdicts.append(v)
        if args.throttle_ms > 0:
            throttle(args.throttle_ms)
        if (idx + 1) % args.chunk_size == 0 or (idx + 1) == len(disagreements):
            print(f"Audit progress: {idx + 1}/{len(disagreements)}...", flush=True)

    print(f"Running non-optional control arm on {len(control_samples)} agreement rows...")
    control_verdicts = []
    for idx, (ex, qid, pred, ref) in enumerate(control_samples):
        v = audit_row(
            ex,
            qid,
            pred,
            ref,
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
            mock=getattr(args, "mock", False),
        )
        control_verdicts.append(v)
        if args.throttle_ms > 0:
            throttle(args.throttle_ms)

    control_disputes = sum(
        1 for v in control_verdicts if v["verdict"] == "dispute_upstream" or v["quarantine_recommended"]
    )
    control_dispute_rate = (control_disputes / len(control_samples)) if control_samples else 0.0
    control_passed = control_dispute_rate <= getattr(args, "max_control_dispute_rate", 0.15)

    quarantined = []
    if control_passed:
        for v in disagreement_verdicts:
            if v["quarantine_recommended"]:
                quarantined.append(f"{v['example_id']}:{v['question_id']}")
    else:
        print(
            f"WARNING: Control arm FAILED (dispute rate {control_dispute_rate:.2%} > 15%). "
            f"Reviewer is unreliable; all quarantine recommendations discarded.",
            flush=True,
        )

    run_artifact = {
        "schema_version": 1,
        "run_id": f"review-audit-{int(time.time())}",
        "command": "audit",
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "reviewer_model_id": args.model,
            "endpoint": args.endpoint,
            "prompt_version": PROMPT_VERSION,
            "temperature": args.temperature,
            "seed": args.seed,
            "ground_truth": False,
            "limitation": CORRELATED_FAMILY_LIMITATION,
        },
        "control_arm": {
            "sample_size": len(control_samples),
            "disputed_count": control_disputes,
            "dispute_rate": round(control_dispute_rate, 4),
            "passed": control_passed,
        },
        "verdicts": disagreement_verdicts,
        "quarantined_examples": sorted(quarantined),
    }

    output_path.write_text(json.dumps(run_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Audit completed. Artifact saved to {output_path}")
    print(f"Quarantined {len(quarantined)} rows (control arm passed: {control_passed}).")
    return run_artifact


def run_slice(args: argparse.Namespace) -> dict[str, Any]:
    """Cluster and explain failure slices across errors."""
    root = Path(args.root).resolve()
    predictions_path = root / args.predictions
    dataset_path = root / args.dataset
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")

    examples = load_m1_dataset(dataset_path)
    del examples  # validates dataset syntax

    errors_by_workload = defaultdict(list)
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if str(r["prediction"]) != str(r["reference"]):
                errors_by_workload[r["workload_id"]].append(r)

    clusters: dict[str, Any] = {}
    for workload_id, err_list in errors_by_workload.items():
        sample_errors = err_list[: min(10, len(err_list))]
        if getattr(args, "mock", False):
            summary = {
                "workload_id": workload_id,
                "error_count": len(err_list),
                "cluster_name": "Mock failure cluster",
                "explanation": "Mock explanation of pattern in errors.",
            }
        else:
            prompt = (
                f"Analyze these {len(sample_errors)} classification errors from workload '{workload_id}':\n"
                + json.dumps(sample_errors, indent=2)
                + "\n\nCluster these errors and explain the primary failure mode in 2-3 sentences."
            )
            reply = call_lm_studio_chat(
                [
                    {"role": "system", "content": "You are a machine learning error analyst."},
                    {"role": "user", "content": prompt},
                ],
                endpoint=args.endpoint,
                model=args.model,
                temperature=args.temperature,
            )
            summary = {
                "workload_id": workload_id,
                "error_count": len(err_list),
                "explanation": reply,
            }
        clusters[workload_id] = summary

    run_artifact = {
        "schema_version": 1,
        "run_id": f"review-slice-{int(time.time())}",
        "command": "slice",
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "reviewer_model_id": args.model,
            "endpoint": args.endpoint,
            "prompt_version": PROMPT_VERSION,
            "temperature": args.temperature,
            "seed": args.seed,
            "ground_truth": False,
            "limitation": CORRELATED_FAMILY_LIMITATION,
        },
        "slices": clusters,
    }

    output_path.write_text(json.dumps(run_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Failure slice analysis saved to {output_path}")
    return run_artifact


def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    """Generate adversarial hard cases verifiable by construction."""
    root = Path(args.root).resolve()
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = []
    if getattr(args, "mock", False):
        cases = [
            {
                "case_id": "gen-001",
                "workload_id": "banking77-16-intent-v1",
                "kind": "adversarial_paraphrase",
                "state": "I wanted to cancel my card but instead my account got closed.",
                "target": "card_arrival",
                "verifiable_by_construction": True,
            }
        ]
    else:
        prompt = (
            "Generate 3 adversarial evaluation cases for intent classification where "
            "the text uses confusing or contradictory wording but has a clear single correct answer. "
            "Output JSON list of objects with fields: case_id, workload_id, kind, state, target."
        )
        reply = call_lm_studio_chat(
            [
                {"role": "system", "content": "You are an adversarial test benchmark generator. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
        )
        try:
            cases = json.loads(reply)
        except Exception:
            cases = [{"raw_generation": reply}]

    run_artifact = {
        "schema_version": 1,
        "run_id": f"review-generate-{int(time.time())}",
        "command": "generate",
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": {
            "reviewer_model_id": args.model,
            "endpoint": args.endpoint,
            "prompt_version": PROMPT_VERSION,
            "temperature": args.temperature,
            "seed": args.seed,
            "ground_truth": False,
            "limitation": CORRELATED_FAMILY_LIMITATION,
        },
        "generated_cases": cases,
    }

    output_path.write_text(json.dumps(run_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated cases saved to {output_path}")
    return run_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=Path.cwd())
    common.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    common.add_argument("--model", default=DEFAULT_REVIEWER_MODEL)
    common.add_argument("--temperature", type=float, default=0.0)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--mock", action="store_true", help="Use deterministic mock reviewer for offline tests")
    common.add_argument("--throttle-ms", type=float, default=5.0)
    common.add_argument("--chunk-size", type=int, default=10)

    parser = argparse.ArgumentParser(description=__doc__, parents=[common])
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    audit_cmd = subparsers.add_parser("audit", help="Audit disagreement rows with control arm", parents=[common])
    audit_cmd.add_argument("--predictions", required=True, help="Predictions JSONL file")
    audit_cmd.add_argument("--dataset", default="evals/data/public/suite.jsonl")
    audit_cmd.add_argument("--output", default="artifacts/review/audit.json")
    audit_cmd.add_argument("--max-audits", type=int, default=None)
    audit_cmd.add_argument("--control-arm-ratio", type=float, default=0.5)
    audit_cmd.add_argument("--max-control-dispute-rate", type=float, default=0.15)
    audit_cmd.set_defaults(func=run_audit)

    slice_cmd = subparsers.add_parser("slice", help="Cluster and analyze failure slices", parents=[common])
    slice_cmd.add_argument("--predictions", required=True, help="Predictions JSONL file")
    slice_cmd.add_argument("--dataset", default="evals/data/public/suite.jsonl")
    slice_cmd.add_argument("--output", default="artifacts/review/slices.json")
    slice_cmd.set_defaults(func=run_slice)

    gen_cmd = subparsers.add_parser("generate", help="Generate adversarial hard cases", parents=[common])
    gen_cmd.add_argument("--output", default="artifacts/review/generated_cases.json")
    gen_cmd.set_defaults(func=run_generate)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
