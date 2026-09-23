"""Learned scalar readout head backend for candidate scoring (ADR-07).

Decouples scoring from vocabulary token unigram frequency and single-token label
constraints by projecting the backbone's last-token hidden representation into an
unconstrained 1-D scalar logit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from jah.backends.huggingface import InferenceMeasurement
from jah.compiler import CompiledQuestion
from jah.scoring import (
    rescale_logits,
)


class ScalarReadoutHead:
    """1-D linear projection head over hidden representations."""

    def __init__(self, hidden_size: int, device: str = "cpu", dtype: Any = None) -> None:
        import torch
        from torch import nn

        self.torch = torch
        self.linear = nn.Linear(hidden_size, 1, bias=True, device=device, dtype=dtype or torch.float32)

    def initialize_from_contrast(self, yes_embedding: Any, no_embedding: Any) -> None:
        """Initialize weights as the contrast direction between positive and negative embeddings."""
        with self.torch.no_grad():
            direction = (yes_embedding - no_embedding).detach().float().clone()
            self.linear.weight.copy_(direction.unsqueeze(0))
            self.linear.bias.zero_()

    def forward(self, hidden_states: Any) -> Any:
        """Project [batch, hidden_size] -> [batch]."""
        return self.linear(hidden_states.float()).squeeze(-1)

    def state_dict(self) -> dict[str, Any]:
        return self.linear.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.linear.load_state_dict(state)


class HuggingFaceScalarHeadBackend:
    """Evaluates candidate options via a learned scalar head instead of vocabulary logits."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str = "cuda",
        adapter_dir: str | Path | None = None,
        head_path: str | Path | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.model.config, "d_model"):
            hidden_size = self.model.config.d_model

        self.head = ScalarReadoutHead(hidden_size, device=str(self.device))

        # Check for explicit head or adapter
        self.adapter_dir = Path(adapter_dir).resolve() if adapter_dir else None
        effective_head_path = Path(head_path) if head_path else (self.adapter_dir / "head.pt" if self.adapter_dir else None)

        if effective_head_path and effective_head_path.exists():
            state = torch.load(effective_head_path, map_location=self.device)
            self.head.load_state_dict(state)
        else:
            # Contrastive zero-shot initialization from Yes/No tokens
            yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)
            no_id = self.tokenizer.encode("No", add_special_tokens=False)
            if len(yes_id) == 1 and len(no_id) == 1:
                emb = self.model.get_input_embeddings().weight
                self.head.initialize_from_contrast(emb[yes_id[0]], emb[no_id[0]])

        # Optional LoRA adapter injection
        if self.adapter_dir:
            from jah.training.adapter import LoRAConfig, inject_lora, load_adapter_into_model

            manifest_path = self.adapter_dir / "adapter_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                lora_config = LoRAConfig(**manifest["config"])
                self.lora_modules = inject_lora(self.model, lora_config)
                load_adapter_into_model(self.lora_modules, self.adapter_dir)

        self.model.eval()

    def _candidate_prompts(self, question: CompiledQuestion) -> list[str]:
        """Construct candidate-specific evaluation prompts from question context."""
        # For each option, append the option description as the completion hypothesis
        prompts = []
        base_prompt = question.prompt
        for opt_id, label in zip(question.option_ids, question.labels, strict=True):
            prompts.append(f"{base_prompt} [{label}] {opt_id}")
        return prompts

    def score(
        self, question: CompiledQuestion, *, temperature: float = 1.0
    ) -> InferenceMeasurement:
        torch = self.torch
        prompts = self._candidate_prompts(question)

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=question.input_tokens or 8192,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**encoded, output_hidden_states=True, return_dict=True)
            # Extract last unpadded hidden state for each candidate
            last_hidden = outputs.hidden_states[-1]
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(len(prompts), device=self.device)
            candidate_features = last_hidden[batch_idx, lengths]
            scalar_scores = self.head.forward(candidate_features)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1_000

        scores_list = scalar_scores.detach().cpu().tolist()
        if not isinstance(scores_list, list):
            scores_list = [scores_list]
        logits_dict = dict(zip(question.option_ids, scores_list, strict=True))

        decision = rescale_logits(
            logits_dict,
            temperature=temperature,
            option_values=question.option_values,
        )

        peak_vram = (
            torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        )
        total_tokens = int(encoded["attention_mask"].sum().item())

        return InferenceMeasurement(
            decision=decision,
            label_mass=1.0,  # Scalar head has full normalized mass over options
            inference_ms=inference_ms,
            input_tokens=total_tokens,
            peak_vram_bytes=peak_vram,
        )
