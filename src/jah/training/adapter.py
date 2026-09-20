"""LoRA adaptation module and training objective for causal decision scoring."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from jah.workload import sha256_file


class LoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    rank: int = Field(default=8, ge=1, le=128)
    alpha: float = Field(default=16.0, gt=0.0)
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    learning_rate: float = Field(default=1e-4, gt=0.0)
    batch_size: int = Field(default=4, ge=1)
    epochs: int = Field(default=3, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0.0)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around a frozen PyTorch linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Initialize A with Kaiming uniform, B with zeros (so ΔW = 0 at start)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_in = self.dropout(x.to(self.lora_A.dtype))
        lora_out = (lora_in @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out.to(base_out.dtype)


def _get_parent_module(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    current = root
    for part in parts[:-1]:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    leaf = parts[-1]
    return current, leaf


def inject_lora(model: nn.Module, config: LoRAConfig) -> dict[str, LoRALinear]:
    """Replace target nn.Linear layers matching config.target_modules with LoRALinear wrappers."""
    lora_modules: dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            matches = any(
                name.endswith(f".{target}") or name == target
                for target in config.target_modules
            )
            if matches:
                parent, leaf = _get_parent_module(model, name)
                lora_layer = LoRALinear(
                    module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                if leaf.isdigit():
                    parent[int(leaf)] = lora_layer
                else:
                    setattr(parent, leaf, lora_layer)
                lora_modules[name] = lora_layer
    return lora_modules


def remove_lora(model: nn.Module, lora_modules: dict[str, LoRALinear]) -> None:
    """Restore the original base layers from LoRALinear wrappers."""
    for name, lora_layer in lora_modules.items():
        parent, leaf = _get_parent_module(model, name)
        if leaf.isdigit():
            parent[int(leaf)] = lora_layer.base_layer
        else:
            setattr(parent, leaf, lora_layer.base_layer)


def compute_decision_loss(
    logits: torch.Tensor,
    targets: torch.Tensor | Sequence[int] | Sequence[dict[str, float]],
    label_token_ids: Sequence[int],
) -> torch.Tensor:
    """Compute cross-entropy loss restricted to verified candidate answer labels.

    Supports hard targets (class indices) or soft targets (label probability distributions).
    """
    label_indices = torch.tensor(label_token_ids, device=logits.device, dtype=torch.long)
    # Gather logits for candidate answer tokens: shape [batch, num_candidates]
    candidate_logits = logits.index_select(dim=-1, index=label_indices)

    if isinstance(targets, torch.Tensor) and targets.dtype == torch.long:
        return F.cross_entropy(candidate_logits, targets)
    if isinstance(targets, (list, tuple)) and targets and isinstance(targets[0], int):
        target_tensor = torch.tensor(targets, device=logits.device, dtype=torch.long)
        return F.cross_entropy(candidate_logits, target_tensor)

    # Soft label probability distribution targets
    if isinstance(targets, torch.Tensor) and targets.dtype in (torch.float32, torch.float16, torch.bfloat16):
        target_probs = targets
    else:
        # Sequence of probability vectors
        target_probs = torch.tensor(targets, device=logits.device, dtype=torch.float32)

    log_probs = F.log_softmax(candidate_logits, dim=-1)
    return -torch.sum(target_probs * log_probs, dim=-1).mean()


def export_adapter(
    lora_modules: dict[str, LoRALinear],
    config: LoRAConfig,
    output_dir: Path | str,
    *,
    base_model: str,
    base_revision: str,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Export LoRA weights, configuration, and reproducibility manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    state_dict = {}
    for name, module in lora_modules.items():
        state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu()
        state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu()

    weights_path = out / "adapter_weights.pt"
    torch.save(state_dict, weights_path)

    manifest = {
        "schema_version": 1,
        "adapter_type": "lora",
        "created_at": datetime.now(UTC).isoformat(),
        "base_model": base_model,
        "base_revision": base_revision,
        "dataset_sha256": dataset_sha256,
        "weights_sha256": sha256_file(weights_path),
        "config": config.model_dump(),
    }
    manifest_path = out / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_adapter_into_model(
    lora_modules: dict[str, LoRALinear],
    adapter_dir: Path | str,
) -> dict[str, Any]:
    """Load exported adapter weights into an existing set of LoRALinear layers."""
    adapter_path = Path(adapter_dir)
    weights_path = adapter_path / "adapter_weights.pt"
    manifest_path = adapter_path / "adapter_manifest.json"

    if not weights_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"invalid adapter bundle in {adapter_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_dict = torch.load(weights_path, map_location="cpu")

    for name, module in lora_modules.items():
        a_key = f"{name}.lora_A"
        b_key = f"{name}.lora_B"
        if a_key in state_dict and b_key in state_dict:
            module.lora_A.data.copy_(state_dict[a_key].to(module.lora_A.device))
            module.lora_B.data.copy_(state_dict[b_key].to(module.lora_B.device))

    return manifest
