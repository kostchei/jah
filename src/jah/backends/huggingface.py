"""Hugging Face reference backend for direct option-logit scoring."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str = "cuda",
        adapter_dir: str | Path | None = None,
    ) -> None:
        import json
        from pathlib import Path

        import torch
        from transformers import AutoModelForMultimodalLM, AutoTokenizer

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.adapter_dir = Path(adapter_dir).resolve() if adapter_dir else None
        self.lora_modules = None
        if self.adapter_dir:
            from jah.training.adapter import LoRAConfig, inject_lora, load_adapter_into_model

            manifest_path = self.adapter_dir / "adapter_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"adapter manifest missing in {self.adapter_dir}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            lora_config = LoRAConfig(**manifest["config"])
            self.lora_modules = inject_lora(self.model, lora_config)
            load_adapter_into_model(self.lora_modules, self.adapter_dir)

        self.model.eval()

    def score(
        self, question: CompiledQuestion, *, temperature: float = 1.0
    ) -> InferenceMeasurement:
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
            temperature=temperature,
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

    def score_batch(
        self,
        questions: Sequence[CompiledQuestion],
        *,
        temperature: float = 1.0,
        microbatch_size: int = 16,
        use_prefix_cache: bool = True,
    ) -> list[InferenceMeasurement]:
        from jah.backends.batching import batch_score_full_prompt
        from jah.backends.prefix_cache import can_extract_prefix, score_with_prefix_cache

        if not questions:
            return []
        if len(questions) == 1:
            return [self.score(questions[0], temperature=temperature)]

        if use_prefix_cache and can_extract_prefix(questions):
            return score_with_prefix_cache(
                self.model,
                self.tokenizer,
                questions,
                device=self.device,
                temperature=temperature,
            )

        return batch_score_full_prompt(
            self.model,
            self.tokenizer,
            questions,
            device=self.device,
            temperature=temperature,
            microbatch_size=microbatch_size,
        )

