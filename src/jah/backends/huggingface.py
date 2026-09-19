"""Hugging Face reference backend for direct option-logit scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass

from jah.compiler import CompiledQuestion
from jah.scoring import (
    ScoredDecision,
    label_probability_mass,
    last_unpadded_logits,
    score_option_logits,
)


@dataclass(frozen=True)
class InferenceMeasurement:
    decision: ScoredDecision
    label_mass: float
    inference_ms: float
    input_tokens: int
    peak_vram_bytes: int


class HuggingFaceDirectLogitBackend:
    def __init__(self, model_id: str, revision: str, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def score(self, question: CompiledQuestion) -> InferenceMeasurement:
        torch = self.torch
        if question.label_token_ids is None:
            raise ValueError("compiled question is missing tokenizer-verified label token IDs")
        encoded = self.tokenizer(
            question.prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**encoded, use_cache=False, return_dict=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1_000
        final = last_unpadded_logits(outputs.logits, encoded["attention_mask"])[0]
        decision = score_option_logits(
            final,
            option_ids=question.option_ids,
            label_token_ids=question.label_token_ids,
            option_values=question.option_values,
        )
        label_mass = label_probability_mass(final, question.label_token_ids)
        peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        return InferenceMeasurement(
            decision=decision,
            label_mass=label_mass,
            inference_ms=inference_ms,
            input_tokens=int(encoded["input_ids"].shape[-1]),
            peak_vram_bytes=int(peak),
        )

    def score_batch(self, questions: list[CompiledQuestion]) -> tuple[InferenceMeasurement, ...]:
        if not questions:
            return ()
        if any(question.label_token_ids is None for question in questions):
            raise ValueError("compiled question is missing tokenizer-verified label token IDs")

        torch = self.torch
        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            encoded = self.tokenizer(
                [question.prompt for question in questions],
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
        finally:
            self.tokenizer.padding_side = previous_padding_side
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**encoded, use_cache=False, return_dict=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        batch_inference_ms = (time.perf_counter() - started) * 1_000
        final_logits = last_unpadded_logits(outputs.logits, encoded["attention_mask"])
        peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        per_decision_ms = batch_inference_ms / len(questions)
        measurements = []
        for index, question in enumerate(questions):
            label_token_ids = question.label_token_ids
            if label_token_ids is None:  # pragma: no cover - guarded above
                raise AssertionError("label token IDs unexpectedly missing")
            final = final_logits[index]
            measurements.append(
                InferenceMeasurement(
                    decision=score_option_logits(
                        final,
                        option_ids=question.option_ids,
                        label_token_ids=label_token_ids,
                        option_values=question.option_values,
                    ),
                    label_mass=label_probability_mass(final, label_token_ids),
                    inference_ms=per_decision_ms,
                    input_tokens=int(encoded["attention_mask"][index].sum().item()),
                    peak_vram_bytes=int(peak),
                )
            )
        return tuple(measurements)
