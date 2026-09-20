"""Shared-prefix KV cache manager for direct option-logit scoring.

Extracts invariant prompt prefixes (e.g. system instructions and state context),
precomputes past_key_values on the prefix once, and evaluates question suffixes
against the isolated cached representation.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence

from jah.compiler import CompiledQuestion
from jah.scoring import (
    ScoredDecision,
    label_probability_mass,
    score_option_logits,
)

SPLIT_DELIMITER = "</STATE>\n\n"


def can_extract_prefix(questions: Sequence[CompiledQuestion]) -> bool:
    """Check if all questions in the batch share an identical prompt prefix."""
    if not questions or len(questions) < 2:
        return False
    first_prompt = questions[0].prompt
    idx = first_prompt.find(SPLIT_DELIMITER)
    if idx == -1:
        return False
    prefix = first_prompt[: idx + len(SPLIT_DELIMITER)]
    return all(q.prompt.startswith(prefix) for q in questions[1:])


def extract_prefix_and_suffixes(
    questions: Sequence[CompiledQuestion],
) -> tuple[str, list[str]]:
    """Split prompts into invariant prefix and question-specific suffixes."""
    first_prompt = questions[0].prompt
    idx = first_prompt.find(SPLIT_DELIMITER)
    if idx == -1:
        raise ValueError(f"prefix delimiter {SPLIT_DELIMITER!r} not found in prompt")
    prefix = first_prompt[: idx + len(SPLIT_DELIMITER)]
    suffixes = [q.prompt[len(prefix) :] for q in questions]
    return prefix, suffixes


def score_with_prefix_cache(
    model,
    tokenizer,
    questions: Sequence[CompiledQuestion],
    *,
    device,
    temperature: float = 1.0,
) -> list:
    """Evaluate questions using precomputed KV cache on the shared prefix."""
    import torch

    from jah.backends.huggingface import InferenceMeasurement

    if not questions:
        return []

    prefix, suffixes = extract_prefix_and_suffixes(questions)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    prefix_tensor = torch.tensor([prefix_ids], device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started_prefix = time.perf_counter()
    with torch.inference_mode():
        prefix_out = model(prefix_tensor, use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefix_ms = (time.perf_counter() - started_prefix) * 1_000

    cached_kv = prefix_out.past_key_values
    amortized_prefix_ms = prefix_ms / len(questions)

    measurements: list[InferenceMeasurement] = []
    for q, suffix in zip(questions, suffixes, strict=True):
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        suffix_tensor = torch.tensor([suffix_ids], device=device)

        # Clone cache state to ensure branch isolation across questions
        branch_cache = copy.deepcopy(cached_kv)

        started_suffix = time.perf_counter()
        with torch.inference_mode():
            suffix_out = model(suffix_tensor, past_key_values=branch_cache)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        suffix_ms = (time.perf_counter() - started_suffix) * 1_000

        final_logit = suffix_out.logits[0, -1]
        decision: ScoredDecision = score_option_logits(
            final_logit,
            option_ids=q.option_ids,
            label_token_ids=q.label_token_ids,
            temperature=temperature,
            option_values=q.option_values,
        )
        label_mass = label_probability_mass(final_logit, q.label_token_ids)
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

        measurements.append(
            InferenceMeasurement(
                decision=decision,
                label_mass=label_mass,
                inference_ms=amortized_prefix_ms + suffix_ms,
                input_tokens=len(prefix_ids) + len(suffix_ids),
                peak_vram_bytes=int(peak),
            )
        )

    return measurements
