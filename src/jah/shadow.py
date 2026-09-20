"""Shadow evaluation engine for recording proposed decisions without applying them."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jah.schemas import EvaluateRequest, EvaluateResponse


class ShadowEvaluator:
    """Asynchronous/non-blocking decision recorder for shadow deployments."""

    def __init__(self, log_path: Path | str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        request: EvaluateRequest,
        response: EvaluateResponse,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log a shadow evaluation event in JSONL format."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": response.request_id,
            "artifact_id": response.artifact_id,
            "profile_id": response.profile_id,
            "metadata": metadata or {},
            "decisions": {
                qid: {
                    "type": answer.type,
                    "value": answer.value if hasattr(answer, "value") else answer.level_id,
                    "probabilities": answer.probabilities,
                    "calibration_status": answer.calibration_status,
                    "disposition": answer.disposition,
                }
                for qid, answer in response.answers.items()
            },
            "timing_ms": response.timing_ms.model_dump(),
        }
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry


def analyze_shadow_traffic(
    shadow_log_path: Path | str,
    *,
    ground_truth_labels: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate aggregate shadow metrics: review rate, coverage, confidence distributions, and agreement."""
    log_file = Path(shadow_log_path)
    if not log_file.exists():
        raise FileNotFoundError(f"shadow log file not found: {shadow_log_path}")

    total_requests = 0
    total_decisions = 0
    disposition_counts = Counter()
    calibration_status_counts = Counter()
    agreements = 0
    labeled_count = 0

    with log_file.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            total_requests += 1
            req_id = entry.get("request_id")
            truth_for_req = ground_truth_labels.get(req_id) if ground_truth_labels else None

            for qid, decision in entry["decisions"].items():
                total_decisions += 1
                disposition_counts[decision["disposition"]] += 1
                calibration_status_counts[decision["calibration_status"]] += 1

                if truth_for_req and qid in truth_for_req:
                    labeled_count += 1
                    if str(decision["value"]).lower() == str(truth_for_req[qid]).lower():
                        agreements += 1

    coverage = disposition_counts["accept"] / max(1, total_decisions)
    review_rate = disposition_counts["review"] / max(1, total_decisions)
    agreement_rate = (agreements / labeled_count) if labeled_count > 0 else None

    return {
        "total_requests": total_requests,
        "total_decisions": total_decisions,
        "coverage": round(coverage, 4),
        "review_rate": round(review_rate, 4),
        "disposition_counts": dict(disposition_counts),
        "calibration_status_counts": dict(calibration_status_counts),
        "evaluated_against_ground_truth": labeled_count > 0,
        "ground_truth_samples": labeled_count,
        "agreement_rate": round(agreement_rate, 4) if agreement_rate is not None else None,
    }
