"""Hugging Face reference backend for direct option-logit scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass

from jah.compiler import CompiledQuestion
from jah.scoring import ScoredDecision, last_unpadded_logits, score_option_logits


@dataclass(frozen=True)
class InferenceMeasurement:
    decision: ScoredDecision
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
        peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        return InferenceMeasurement(
            decision=decision,
            inference_ms=inference_ms,
            input_tokens=int(encoded["input_ids"].shape[-1]),
            peak_vram_bytes=int(peak),
        )
