"""Workload acceptance policy rules and threshold evaluation."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, field_validator

from jah.schemas import Disposition, StrictModel


class AcceptancePolicy(StrictModel):
    """Specification of when a decision is automatically accepted versus reviewed."""

    type: Literal["threshold", "review_only"] = "threshold"
    threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    metric: Literal["max_probability", "margin"] = "max_probability"
    max_order_discrepancy: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    @field_validator("threshold")
    @classmethod
    def finite_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("policy threshold must be finite")
        return value


def evaluate_acceptance(
    policy: AcceptancePolicy,
    probabilities: dict[str, float],
    primitive: str,
    *,
    order_discrepancy: float | None = None,
) -> Disposition:
    """Evaluate whether calibrated probabilities meet the acceptance policy."""
    if policy.type == "review_only":
        return "review"

    if not probabilities:
        return "review"

    if (
        order_discrepancy is not None
        and policy.max_order_discrepancy is not None
        and order_discrepancy > policy.max_order_discrepancy
    ):
        return "review"

    if policy.metric == "max_probability":
        confidence = max(probabilities.values())
    elif policy.metric == "margin":
        sorted_probs = sorted(probabilities.values(), reverse=True)
        confidence = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)
    else:
        raise ValueError(f"unsupported policy metric: {policy.metric}")

    if confidence >= policy.threshold:
        return "accept"
    return "review"


def fit_acceptance_threshold(
    confidences: list[float],
    accuracies: list[int],
    *,
    target_error: float = 0.05,
    min_coverage: float = 0.50,
) -> tuple[float, dict]:
    """Fit a confidence threshold that minimizes review while bounding accepted error.

    Selects the lowest threshold achieving accepted error <= target_error
    and coverage >= min_coverage. If no threshold meets both, selects the threshold
    meeting target_error with the highest coverage.
    """
    if len(confidences) != len(accuracies):
        raise ValueError("confidences and accuracies must have equal length")
    if not confidences:
        return 0.90, {
            "threshold": 0.90,
            "coverage": 0.0,
            "accepted_error": 0.0,
            "accepted_count": 0,
            "total": 0,
        }

    total = len(confidences)
    candidates = sorted({round(c, 4) for c in confidences} | {0.50, 0.70, 0.80, 0.85, 0.90, 0.95})
    best_threshold = 0.90
    best_summary: dict | None = None
    fallback_threshold = 0.90
    fallback_summary: dict | None = None

    for threshold in candidates:
        accepted = [acc for conf, acc in zip(confidences, accuracies, strict=True) if conf >= threshold]
        accepted_count = len(accepted)
        coverage = accepted_count / total
        errors = sum(1 for acc in accepted if acc == 0)
        error_rate = (errors / accepted_count) if accepted_count > 0 else 0.0

        summary = {
            "threshold": threshold,
            "coverage": coverage,
            "accepted_error": error_rate,
            "accepted_count": accepted_count,
            "total": total,
            "errors": errors,
        }

        if error_rate <= target_error:
            if coverage >= min_coverage:
                # Lowest threshold meeting both
                best_threshold = threshold
                best_summary = summary
                break
            if fallback_summary is None or coverage > fallback_summary["coverage"]:
                fallback_threshold = threshold
                fallback_summary = summary

    chosen_threshold = best_threshold if best_summary is not None else fallback_threshold
    chosen_summary = best_summary if best_summary is not None else (fallback_summary or {
        "threshold": chosen_threshold,
        "coverage": 0.0,
        "accepted_error": 0.0,
        "accepted_count": 0,
        "total": total,
    })
    return chosen_threshold, chosen_summary
