import json
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from jah.training.adapter import LoRALinear
from jah.training.train import parse_args, train_adapter


def test_parse_args():
    args = parse_args([
        "--dataset", "evals/data/public/suite.jsonl",
        "--epochs", "2",
        "--batch-size", "8",
        "--learning-rate", "5e-5",
        "--max-samples", "100",
        "--workload", "banking77-16-intent-v1",
        "--output-dir", "artifacts/adapters/test-run",
    ])
    assert args.epochs == 2
    assert args.batch_size == 8
    assert args.learning_rate == 5e-5
    assert args.max_samples == 100
    assert args.workload == "banking77-16-intent-v1"
    assert args.output_dir == "artifacts/adapters/test-run"


class MockCausalModel(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        h = self.q_proj(h) + self.v_proj(h)
        logits = self.lm_head(h)
        output = MagicMock()
        output.logits = logits
        return output


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        # Simple ASCII mapping capped at 255
        return [ord(c) % 256 for c in text]

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        del add_special_tokens
        tokens = self.encode(text)
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
        }


def test_train_adapter_mock(tmp_path):
    # Set up mock configs and output
    out_dir = tmp_path / "adapter_out"
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text("model_id: mock-model\nrevision: rev1\ndevice: cpu\n", encoding="utf-8")

    lora_cfg = tmp_path / "lora.yaml"
    lora_cfg.write_text(
        "schema_version: 1\n"
        "rank: 2\n"
        "alpha: 4.0\n"
        "dropout: 0.0\n"
        "target_modules: ['q_proj', 'v_proj', 'lm_head']\n"
        "learning_rate: 0.01\n"
        "batch_size: 1\n"
        "epochs: 1\n"
        "max_grad_norm: 1.0\n",
        encoding="utf-8",
    )

    mock_model = MockCausalModel()
    mock_tokenizer = MockTokenizer()

    with (
        patch("jah.training.train.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("jah.training.train.AutoModelForMultimodalLM.from_pretrained", return_value=mock_model),
    ):
        report = train_adapter(
            dataset_path="evals/data/m1-suite.jsonl",
            output_dir=out_dir,
            model_config_path=model_cfg,
            training_config_path=lora_cfg,
            split_seed="jah-m1-split-v1",
            max_samples=10,
            epochs=1,
            device="cpu",
            log_interval=5,
        )

    assert report["epochs"] == 1
    assert report["train_samples"] <= 10
    assert report["total_optimizer_steps"] > 0
    assert (out_dir / "adapter_weights.pt").exists()
    assert (out_dir / "adapter_manifest.json").exists()
    assert (out_dir / "training-report.json").exists()

    manifest = json.loads((out_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter_type"] == "lora"
    assert manifest["base_model"] == "mock-model"

    # Verify loading into HuggingFaceDirectLogitBackend
    from jah.backends.huggingface import HuggingFaceDirectLogitBackend

    fresh_model = MockCausalModel()
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("transformers.AutoModelForMultimodalLM.from_pretrained", return_value=fresh_model),
    ):
        backend = HuggingFaceDirectLogitBackend(
            "mock-model",
            "rev1",
            device="cpu",
            adapter_dir=out_dir,
        )
        assert backend.lora_modules is not None
        assert isinstance(backend.model.lm_head, LoRALinear)
