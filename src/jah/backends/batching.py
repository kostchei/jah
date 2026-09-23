"""Full-prompt microbatching for direct option-logit scoring."""

from __future__ import annotations

import time
from collections.abc import Sequence

from jah.compiler import CompiledQuestion
from jah.scoring import (
    ScoredDecision,
    label_probability_mass,
    last_unpadded_logits,
    score_option_logits,
)


def batch_score_full_prompt(
    model,
    tokenizer,
    questions: Sequence[CompiledQuestion],
    *,
    device,
    temperature: float = 1.0,
    microbatch_size: int = 16,
) -> list:
    """Score compiled questions in length-bucketed full-prompt microbatches."""
    import torch

    from jah.backends.huggingface import InferenceMeasurement

    if not questions:
        return []

    for q in questions:
        if q.label_token_ids is None:
            raise ValueError(f"compiled question {q.question_id!r} is missing label_token_ids")

    measurements: list[InferenceMeasurement] = []

    for i in range(0, len(questions), microbatch_size):
        chunk = questions[i : i + microbatch_size]
        prompts = [q.prompt for q in chunk]

        encoded = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**encoded, use_cache=False, return_dict=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1_000

        finals = last_unpadded_logits(outputs.logits, encoded["attention_mask"])
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        per_q_ms = elapsed_ms / len(chunk)

        for j, q in enumerate(chunk):
            final = finals[j]
            decision: ScoredDecision = score_option_logits(
                final,
                option_ids=q.option_ids,
                label_token_ids=q.label_token_ids,
                temperature=temperature,
                option_values=q.option_values,
            )
            label_mass = label_probability_mass(final, q.label_token_ids)
            input_toks = int(encoded["attention_mask"][j].sum().item())
            measurements.append(
                InferenceMeasurement(
                    decision=decision,
                    label_mass=label_mass,
                    inference_ms=per_q_ms,
                    input_tokens=input_toks,
                    peak_vram_bytes=int(peak),
                    peak_vram_reserved_bytes=int(reserved),
                )
            )

    return measurements
