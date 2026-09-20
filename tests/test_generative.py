import torch

from jah.backends.generative import HuggingFaceGenerativeBackend
from jah.compiler import CompiledQuestion


class FakeTokenizer:
    def __call__(self, prompt, *, return_tensors, add_special_tokens):
        del prompt, return_tensors, add_special_tokens
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class FakeModel:
    def generate(self, **kwargs):
        allowed = kwargs["prefix_allowed_tokens_fn"](0, kwargs["input_ids"][0])
        assert allowed == [101, 102]
        return torch.tensor([[1, 2, 3, 102]])


def test_generative_baseline_emits_one_constrained_label() -> None:
    backend = HuggingFaceGenerativeBackend.__new__(HuggingFaceGenerativeBackend)
    backend.torch = torch
    backend.device = torch.device("cpu")
    backend.tokenizer = FakeTokenizer()
    backend.model = FakeModel()
    question = CompiledQuestion(
        question_id="route",
        primitive="choice",
        prompt="prompt",
        option_ids=("a", "b"),
        option_values=None,
        labels=("A", "B"),
        label_token_ids=(101, 102),
        input_tokens=3,
    )
    result = backend.score(question)
    assert result.decision.selected_id == "b"
    assert result.decision.probabilities == {"a": 0.0, "b": 1.0}
    assert result.input_tokens == 3
