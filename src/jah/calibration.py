"""Calibration profile schema, fitting, metrics, and compatibility validation."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, field_validator

from jah.policy import AcceptancePolicy
from jah.schemas import Identifier, StrictModel


class CalibrationMetrics(StrictModel):
    """Evaluation evidence recorded for a validated calibration profile."""

    brier_score: float
    prevalence_brier_score: float
    nll: float
    ece: float
    coverage: float
    accepted_error_rate: float
    accepted_error_95_upper_bound: float
    sample_count: int
    accepted_count: int
    errors_in_accepted: int
    gate_ece_passed: bool
    gate_brier_passed: bool
    gate_selective_passed: bool
    samples_sufficient_for_gate: bool
    reliability_bins: list[dict[str, Any]] = Field(default_factory=list)


class CalibrationProfile(StrictModel):
    """Versioned calibration and acceptance policy artifact per ADR-04."""

    schema_version: int = 1
    profile_id: Identifier
    workload_id: Identifier
    primitive: Literal["boolean", "choice", "score"]

    # Model and prompt compatibility constraints
    artifact_id: str
    model_id: str
    revision: str
    precision: str
    prompt_version: str
    label_version: str
    cardinality_range: tuple[int, int]

    # Calibrator configuration
    method: Literal["temperature_scaling", "vector_scaling", "identity"] = "temperature_scaling"
    temperature: Annotated[float, Field(gt=0.0)] = 1.0
    bias: list[float] | None = None

    # Acceptance policy
    policy: AcceptancePolicy = Field(default_factory=AcceptancePolicy)

    # Provenance and evidence metadata
    dataset_sha256: str | None = None
    split: str | None = None
    fitted_at: str | None = None
    metrics: CalibrationMetrics | None = None

    @field_validator("temperature")
    @classmethod
    def finite_temperature(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("temperature must be finite and positive")
        return value


def has_validated_evidence(profile: CalibrationProfile) -> bool:
    """Require passing calibration and selective-error evidence before promotion.

    Compatibility alone is not validation. Missing evidence and failed gates must
    leave the profile in review mode, even when its confidence exceeds policy.
    """
    metrics = profile.metrics
    if metrics is None:
        return False
    return (
        metrics.gate_ece_passed
        and metrics.gate_brier_passed
        and metrics.gate_selective_passed
        and metrics.samples_sufficient_for_gate
        and 0.0 <= metrics.ece <= 0.05
        and 0.0 <= metrics.brier_score <= metrics.prevalence_brier_score
        and math.isfinite(metrics.prevalence_brier_score)
        and 0.50 <= metrics.coverage <= 1.0
        and 0.0 <= metrics.accepted_error_95_upper_bound <= 0.05
        and 0 <= metrics.errors_in_accepted <= metrics.accepted_count
        and 59 <= metrics.accepted_count <= metrics.sample_count
        and metrics.accepted_count / metrics.sample_count >= 0.50
        and compute_binomial_upper_bound(
            metrics.errors_in_accepted, metrics.accepted_count
        ) <= 0.05
    )


def check_profile_compatibility(
    profile: CalibrationProfile,
    *,
    primitive: str,
    option_count: int,
    artifact_id: str | None = None,
    model_id: str | None = None,
    revision: str | None = None,
    precision: str | None = None,
    prompt_version: str | None = None,
    label_version: str | None = None,
) -> bool:
    """Validate whether a profile is compatible with a question and runtime backbone."""
    if profile.primitive != primitive:
        return False
    min_cardinality, max_cardinality = profile.cardinality_range
    if not (min_cardinality <= option_count <= max_cardinality):
        return False
    if artifact_id is not None and profile.artifact_id != artifact_id:
        return False
    if model_id is not None and profile.model_id != model_id:
        return False
    if revision is not None and profile.revision != revision:
        return False
    if precision is not None and profile.precision != precision:
        return False
    if prompt_version is not None and profile.prompt_version != prompt_version:
        return False
    return not (label_version is not None and profile.label_version != label_version)


def fit_temperature_scaling(
    logits_list: list[list[float]],
    targets: list[int],
    *,
    bounds: tuple[float, float] = (0.05, 10.0),
    tolerance: float = 1e-4,
) -> float:
    """Fit a single scalar temperature T minimizing NLL via golden-section search."""
    if len(logits_list) != len(targets):
        raise ValueError("logits and targets must have equal length")
    if not logits_list:
        return 1.0

    def nll_at(temp: float) -> float:
        total_nll = 0.0
        for logits, target in zip(logits_list, targets, strict=True):
            max_logit = max(logits)
            exp_logits = [math.exp((z - max_logit) / temp) for z in logits]
            sum_exp = sum(exp_logits)
            p_target = exp_logits[target] / sum_exp
            total_nll -= math.log(max(p_target, 1e-15))
        return total_nll / len(logits_list)

    # Golden section search for 1D convex optimization
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    inv_phi2 = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382

    a, b = bounds
    h = b - a
    if h <= tolerance:
        return (a + b) / 2.0

    c = a + inv_phi2 * h
    d = a + inv_phi * h
    fc = nll_at(c)
    fd = nll_at(d)

    for _ in range(60):
        if (b - a) < tolerance:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            h = b - a
            c = a + inv_phi2 * h
            fc = nll_at(c)
        else:
            a = c
            c = d
            fc = fd
            h = b - a
            d = a + inv_phi * h
            fd = nll_at(d)

    return (a + b) / 2.0


def fit_vector_scaling(
    logits_list: list[list[float]],
    targets: list[int],
    *,
    initial_temperature: float = 1.0,
    max_iter: int = 100,
) -> tuple[float, list[float]]:
    """Fit scalar temperature T and bias vector b minimizing NLL via L-BFGS."""
    import torch
    import torch.nn.functional as F

    if len(logits_list) != len(targets):
        raise ValueError("logits and targets must have equal length")
    if not logits_list:
        return 1.0, []

    K = len(logits_list[0])
    z = torch.tensor(logits_list, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.long)

    init_w = math.log(max(0.05, initial_temperature))
    log_t = torch.tensor([init_w], dtype=torch.float32, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [log_t, b], lr=0.5, max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=1e-9
    )

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_t) + 1e-4
        scaled = z / t + b
        loss = F.cross_entropy(scaled, y)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        final_t = float((torch.exp(log_t) + 1e-4).item())
        b_centered = b - b.mean()
        final_b = [round(float(val.item()), 6) for val in b_centered]

    return round(max(0.05, min(10.0, final_t)), 4), final_b


def compute_prior_correction(
    targets: list[int],
    num_classes: int,
    uncalibrated_probs: list[list[float]],
) -> list[float]:
    """Compute per-class logit bias based on target empirical prior vs average model probability."""
    counts = Counter(targets)
    total = len(targets)
    target_priors = [max(counts[k], 1) / total for k in range(num_classes)]
    avg_model_priors = [
        max(sum(p[k] for p in uncalibrated_probs) / len(uncalibrated_probs), 1e-6)
        for k in range(num_classes)
    ]
    shifts = [math.log(tp) - math.log(mp) for tp, mp in zip(target_priors, avg_model_priors, strict=True)]
    mean_shift = sum(shifts) / len(shifts)
    return [round(s - mean_shift, 6) for s in shifts]


def compute_nll(
    probabilities: list[list[float]],
    targets: list[int],
    *,
    eps: float = 1e-15,
) -> float:
    """Compute average negative log-likelihood."""
    if len(probabilities) != len(targets):
        raise ValueError("probabilities and targets must have equal length")
    if not probabilities:
        return 0.0
    total = 0.0
    for probs, target in zip(probabilities, targets, strict=True):
        p_target = probs[target]
        total -= math.log(max(p_target, eps))
    return total / len(probabilities)


def compute_brier_score(
    probabilities: list[list[float]],
    targets: list[int],
) -> float:
    """Compute multi-class Brier score."""
    if len(probabilities) != len(targets):
        raise ValueError("probabilities and targets must have equal length")
    if not probabilities:
        return 0.0
    total = 0.0
    for probs, target in zip(probabilities, targets, strict=True):
        for k, p in enumerate(probs):
            y_k = 1.0 if k == target else 0.0
            total += (p - y_k) ** 2
    return total / len(probabilities)


def compute_prevalence_brier(
    targets: list[int],
    num_classes: int,
) -> float:
    """Compute Brier score of a constant prevalence baseline."""
    if not targets:
        return 0.0
    counts = Counter(targets)
    total = len(targets)
    prevalences = [counts[k] / total for k in range(num_classes)]
    brier = 0.0
    for target in targets:
        for k in range(num_classes):
            y_k = 1.0 if k == target else 0.0
            brier += (prevalences[k] - y_k) ** 2
    return brier / total


def compute_ece(
    confidences: list[float],
    accuracies: list[int],
    *,
    num_bins: int = 10,
    equal_count: bool = True,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute 10-bin equal-count or equal-width Expected Calibration Error."""
    if len(confidences) != len(accuracies):
        raise ValueError("confidences and accuracies must have equal length")
    if not confidences:
        return 0.0, []

    total = len(confidences)
    items = sorted(zip(confidences, accuracies, strict=True), key=lambda pair: pair[0])

    bins: list[list[tuple[float, int]]] = []
    if equal_count:
        bin_size = max(1, math.ceil(total / num_bins))
        for i in range(0, total, bin_size):
            bins.append(items[i : i + bin_size])
    else:
        width = 1.0 / num_bins
        binned: list[list[tuple[float, int]]] = [[] for _ in range(num_bins)]
        for conf, acc in items:
            index = min(int(conf / width), num_bins - 1)
            binned[index].append((conf, acc))
        bins = [b for b in binned if b]

    ece = 0.0
    bin_details = []
    for b in bins:
        bin_count = len(b)
        avg_conf = sum(conf for conf, _ in b) / bin_count
        avg_acc = sum(acc for _, acc in b) / bin_count
        weight = bin_count / total
        diff = abs(avg_acc - avg_conf)
        ece += weight * diff
        bin_details.append({
            "count": bin_count,
            "mean_confidence": avg_conf,
            "accuracy": avg_acc,
            "difference": diff,
            "min_confidence": b[0][0],
            "max_confidence": b[-1][0],
        })

    return ece, bin_details


def compute_binomial_upper_bound(
    k: int,
    n: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Exact one-sided Clopper-Pearson binomial upper bound on error rate."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    if k == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / n)

    target = 1.0 - confidence
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        # CDF of Binomial(n, mid) at k
        prob = sum(math.comb(n, j) * (mid**j) * ((1.0 - mid) ** (n - j)) for j in range(k + 1))
        if prob > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def compute_selective_metrics(
    confidences: list[float],
    accuracies: list[int],
    threshold: float,
) -> dict[str, Any]:
    """Calculate coverage, empirical error, and 95% Clopper-Pearson bound at threshold."""
    if len(confidences) != len(accuracies):
        raise ValueError("confidences and accuracies must have equal length")
    total = len(confidences)
    if total == 0:
        return {
            "coverage": 0.0,
            "accepted_error_rate": 0.0,
            "accepted_error_95_upper_bound": 1.0,
            "accepted_count": 0,
            "errors_in_accepted": 0,
            "total": 0,
        }

    accepted = [(conf, acc) for conf, acc in zip(confidences, accuracies, strict=True) if conf >= threshold]
    accepted_count = len(accepted)
    coverage = accepted_count / total
    errors = sum(1 for _, acc in accepted if acc == 0)
    error_rate = (errors / accepted_count) if accepted_count > 0 else 0.0
    upper_bound = compute_binomial_upper_bound(errors, accepted_count, confidence=0.95)

    return {
        "coverage": coverage,
        "accepted_error_rate": error_rate,
        "accepted_error_95_upper_bound": upper_bound,
        "accepted_count": accepted_count,
        "errors_in_accepted": errors,
        "total": total,
    }


def save_profile(profile: CalibrationProfile, path: Path) -> None:
    """Save a profile to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = profile.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_profile(path: Path) -> CalibrationProfile:
    """Load a profile from YAML."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"expected YAML mapping in {path}")
    return CalibrationProfile.model_validate(data)


def load_profile_registry(directory: Path) -> dict[str, CalibrationProfile]:
    """Load all valid YAML calibration profiles in a directory into a dict by profile_id."""
    from pydantic import ValidationError

    registry: dict[str, CalibrationProfile] = {}
    if not directory.exists():
        return registry
    for path in sorted(directory.glob("*.yaml")):
        try:
            profile = load_profile(path)
            registry[profile.profile_id] = profile
        except (ValidationError, yaml.YAMLError, TypeError, ValueError, OSError):
            continue
    return registry
