import pytest
import torch
from pydantic import ValidationError
from torch import nn

from jah.training.adapter import (
    LoRAConfig,
    LoRALinear,
    compute_decision_loss,
    export_adapter,
    load_adapter_into_model,
)


def test_lora_config_validation():
    cfg = LoRAConfig(rank=16, alpha=32.0, dropout=0.1)
    assert cfg.rank == 16
    assert cfg.alpha == 32.0

    with pytest.raises(ValidationError):
        LoRAConfig(rank=0)

    with pytest.raises(ValidationError):
        LoRAConfig(alpha=-1.0)

    with pytest.raises(ValidationError):
        LoRAConfig(dropout=0.9)


def test_lora_linear_forward_and_gradients():
    base = nn.Linear(32, 16)
    lora = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)

    x = torch.randn(2, 32)
    # At initialization, lora_B is zero, so output exactly equals base_layer(x)
    y_base = base(x)
    y_lora = lora(x)
    assert torch.allclose(y_base, y_lora)

    # Compute loss and check gradients: base layer must be frozen, lora parameters have grad
    loss = y_lora.sum()
    loss.backward()

    assert base.weight.grad is None
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None


def test_compute_decision_loss_hard_targets():
    # Batch of 2, vocab size 10, candidate tokens at indices [3, 7]
    logits = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
    ], requires_grad=True)
    label_token_ids = [3, 7]  # index 0 is token 3, index 1 is token 7

    # Target 0 for row 0, target 1 for row 1 (both match the highest logits)
    targets = [0, 1]
    loss = compute_decision_loss(logits, targets, label_token_ids)
    assert loss.item() < 0.1  # Low loss because predictions align with targets


def test_compute_decision_loss_soft_targets():
    logits = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ], requires_grad=True)
    label_token_ids = [3, 7]
    soft_targets = [[0.5, 0.5]]

    loss = compute_decision_loss(logits, soft_targets, label_token_ids)
    # Log softmax of [1.0, 1.0] is [-log(2), -log(2)], CE loss is log(2) ~ 0.6931
    assert abs(loss.item() - 0.6931) < 0.01


def test_export_and_load_adapter(tmp_path):
    base1 = nn.Linear(16, 8)
    base2 = nn.Linear(8, 4)
    modules = {
        "layer1": LoRALinear(base1, rank=2),
        "layer2": LoRALinear(base2, rank=2),
    }

    # Perturb weights so they are non-zero
    with torch.no_grad():
        modules["layer1"].lora_B.fill_(0.5)

    config = LoRAConfig(rank=2)
    manifest = export_adapter(
        modules,
        config,
        tmp_path / "test_adapter",
        base_model="test-model",
        base_revision="rev1",
        dataset_sha256="abc123sha",
    )
    assert manifest["adapter_type"] == "lora"
    assert (tmp_path / "test_adapter" / "adapter_weights.pt").exists()

    # Create fresh modules and load
    new_modules = {
        "layer1": LoRALinear(nn.Linear(16, 8), rank=2),
        "layer2": LoRALinear(nn.Linear(8, 4), rank=2),
    }
    loaded_manifest = load_adapter_into_model(new_modules, tmp_path / "test_adapter")
    assert loaded_manifest["base_model"] == "test-model"
    assert torch.allclose(new_modules["layer1"].lora_B, modules["layer1"].lora_B)
