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
    label_mass: float | None
    inference_ms: float
    input_tokens: int
    peak_vram_bytes: int
    peak_vram_reserved_bytes: int = 0

class HuggingFaceDirectLogitBackend:
    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str = "cuda",
        precision: str = "bfloat16",
        adapter_dir: str | Path | None = None,
    ) -> None:
        import json
        from pathlib import Path

        import torch
        from transformers import AutoModelForMultimodalLM, AutoTokenizer

        precision_dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if precision not in precision_dtypes:
            raise ValueError(f"unsupported inference precision: {precision}")
        dtype = precision_dtypes[precision] if device == "cuda" else torch.float32

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
            if precision == "bfloat16":
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            if precision == "float16":
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            if precision == "float32":
                torch.backends.cuda.matmul.allow_tf32 = False
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        layer_types = getattr(getattr(self.model, "config", None), "layer_types", ())
        self.batch_optimization_mode = (
            "sequential full-prompt fallback for recurrent/linear-attention model"
            if any("linear_attention" in str(layer_type) for layer_type in layer_types)
            else "prefix-cache or full-prompt microbatch"
        )

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
        reserved = torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0
        return InferenceMeasurement(
            decision=decision,
            label_mass=label_mass,
            inference_ms=inference_ms,
            input_tokens=int(encoded["input_ids"].shape[-1]),
            peak_vram_bytes=int(peak),
            peak_vram_reserved_bytes=int(reserved),
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

        # Recurrent/linear-attention layers carry state across sequence chunks. Splitting
        # a full prompt into a cached prefix and question suffix changes the numerical
        # execution path (and can materially change option probabilities) for these
        # architectures. Keep their batch API exact by scoring each complete prompt,
        # matching the reference path. This is an architecture-specific safety fallback;
        # ordinary attention models may still use prefix reuse and microbatching below.
        if "sequential full-prompt fallback" in self.batch_optimization_mode:
            return [self.score(question, temperature=temperature) for question in questions]

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
