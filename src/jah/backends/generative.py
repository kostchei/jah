"""Compact constrained-generation baseline for paired M1 comparisons."""

from __future__ import annotations

import time

from jah.backends.huggingface import InferenceMeasurement
from jah.compiler import CompiledQuestion
from jah.scoring import ScoredDecision


class HuggingFaceGenerativeBackend:
    """Generate exactly one allowed answer-label token with the same backbone."""

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
        encoded = self.tokenizer(question.prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        allowed = list(question.label_token_ids)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=1,
                min_new_tokens=1,
                prefix_allowed_tokens_fn=lambda _batch_id, _input_ids: allowed,
                use_cache=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1_000

        input_length = int(encoded["input_ids"].shape[-1])
        generated_token_id = int(generated[0, input_length].item())
        if generated_token_id not in question.label_token_ids:
            raise RuntimeError("generator returned a token outside the allowed label set")
        selected_index = question.label_token_ids.index(generated_token_id)
        selected_id = question.option_ids[selected_index]
        probabilities = {
            option_id: float(index == selected_index)
            for index, option_id in enumerate(question.option_ids)
        }
        expected_value = None
        if question.option_values is not None:
            expected_value = question.option_values[selected_index]
        peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        return InferenceMeasurement(
            decision=ScoredDecision(
                selected_id=selected_id,
                probabilities=probabilities,
                expected_value=expected_value,
            ),
            label_mass=1.0,
            inference_ms=inference_ms,
            input_tokens=input_length,
            peak_vram_bytes=int(peak),
        )
