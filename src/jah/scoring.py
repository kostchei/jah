"""Reference direct-logit probability calculations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredDecision:
    selected_id: str
    probabilities: dict[str, float]
    expected_value: float | None = None
    logits: dict[str, float] | None = None


def last_unpadded_logits(logits, attention_mask):
    """Gather [batch, vocab] logits at each row's final unpadded position."""
    import torch

    if logits.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("expected logits [batch, sequence, vocab] and mask [batch, sequence]")
    if logits.shape[:2] != attention_mask.shape:
        raise ValueError("logits and attention mask sequence dimensions must match")
    positions = attention_mask.to(dtype=torch.long).sum(dim=1) - 1
    if torch.any(positions < 0):
        raise ValueError("every sequence must contain at least one unmasked token")
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, positions]


def score_option_logits(
    final_logits,
    *,
    option_ids: tuple[str, ...],
    label_token_ids: tuple[int, ...],
    temperature: float = 1.0,
    option_values: tuple[float, ...] | None = None,
) -> ScoredDecision:
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(option_ids) != len(label_token_ids):
        raise ValueError("option IDs and label token IDs must have equal length")
    if option_values is not None and len(option_values) != len(option_ids):
        raise ValueError("option values and option IDs must have equal length")

    indices = torch.tensor(label_token_ids, dtype=torch.long, device=final_logits.device)
    selected_logits = final_logits.to(dtype=torch.float32).index_select(-1, indices)
    probabilities_tensor = torch.softmax(selected_logits / temperature, dim=-1)
    probabilities_list = probabilities_tensor.detach().cpu().tolist()
    probabilities = dict(zip(option_ids, probabilities_list, strict=True))
    logits_list = selected_logits.detach().cpu().tolist()
    logits_dict = dict(zip(option_ids, logits_list, strict=True))
    selected_index = int(torch.argmax(probabilities_tensor).item())
    expected_value = None
    if option_values is not None:
        expected_value = sum(
            probability * value
            for probability, value in zip(probabilities_list, option_values, strict=True)
        )
    return ScoredDecision(
        selected_id=option_ids[selected_index],
        probabilities=probabilities,
        expected_value=expected_value,
        logits=logits_dict,
    )


def rescale_logits(
    logits: dict[str, float],
    temperature: float,
    *,
    option_values: tuple[float, ...] | None = None,
) -> ScoredDecision:
    """Compute temperature-scaled FP32 probabilities from unnormalized logits."""
    import math

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not logits:
        raise ValueError("logits cannot be empty")

    max_logit = max(logits.values())
    exp_logits = {k: math.exp((v - max_logit) / temperature) for k, v in logits.items()}
    total_exp = sum(exp_logits.values())
    probabilities = {k: v / total_exp for k, v in exp_logits.items()}

    # Guarantee probabilities sum to 1.0 within 1e-5
    sum_prob = sum(probabilities.values())
    if not math.isclose(sum_prob, 1.0, abs_tol=1e-5):
        probabilities = {k: v / sum_prob for k, v in probabilities.items()}

    selected_id = max(probabilities, key=lambda k: (probabilities[k], k))
    expected_value = None
    if option_values is not None:
        if len(option_values) != len(logits):
            raise ValueError("option values and logits must have equal length")
        expected_value = sum(
            probabilities[option_id] * value
            for option_id, value in zip(logits.keys(), option_values, strict=True)
        )
    return ScoredDecision(
        selected_id=selected_id,
        probabilities=probabilities,
        expected_value=expected_value,
        logits=logits,
    )


def label_probability_mass(final_logits, label_token_ids: tuple[int, ...]) -> float:
    """Return the labels' combined probability under the full vocabulary softmax."""
    import torch

    if final_logits.ndim != 1:
        raise ValueError("expected final logits [vocab]")
    if not label_token_ids:
        raise ValueError("at least one label token ID is required")
    indices = torch.tensor(label_token_ids, dtype=torch.long, device=final_logits.device)
    probabilities = torch.softmax(final_logits.to(dtype=torch.float32), dim=-1)
    return float(probabilities.index_select(-1, indices).sum().item())
