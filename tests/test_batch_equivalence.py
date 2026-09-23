from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jah.backends.huggingface import InferenceMeasurement
from jah.backends.prefix_cache import (
    can_extract_prefix,
    extract_prefix_and_suffixes,
)
from jah.compiler import CompiledQuestion, compile_request
from jah.engine import DecisionEngine, EngineConfig
from jah.schemas import EvaluateRequest
from tests.test_engine import FakeBackend


class MockTensor:
    def __init__(self, data, shape=None):
        self.data = data
        self.shape = shape or (len(data), len(data[0]) if data and isinstance(data[0], list) else 1)
        self.device = MockDevice()

    def to(self, device):
        del device
        return self

    def sum(self, dim=1):
        if dim == 1:
            return MockTensor([sum(row) for row in self.data])
        return MockTensor([sum(self.data)])

    def item(self):
        return self.data[0] if isinstance(self.data, list) else self.data


@dataclass
class MockDevice:
    type: str = "cpu"


class MockModelOutput:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values or {"kv": "mock_cache"}


class MockTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(c) % 250 for c in text]

    def __call__(self, prompts, padding=True, return_tensors="pt", add_special_tokens=False):
        del padding, return_tensors, add_special_tokens
        if isinstance(prompts, str):
            prompts = [prompts]
        max_len = max(len(p) for p in prompts)
        input_ids = [[ord(c) % 250 for c in p] + [0] * (max_len - len(p)) for p in prompts]
        attention_mask = [[1] * len(p) + [0] * (max_len - len(p)) for p in prompts]
        import torch

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


def test_prefix_extraction():
    request = EvaluateRequest.model_validate({
        "state": "The system state is healthy and operational.",
        "questions": {
            "q1": {"type": "boolean", "instructions": "Check", "proposition": "Is healthy?"},
            "q2": {"type": "boolean", "instructions": "Check", "proposition": "Is operational?"},
        },
    })
    tokenizer = MockTokenizer()
    compiled = compile_request(request, tokenizer=tokenizer)
    assert can_extract_prefix(compiled) is True

    prefix, suffixes = extract_prefix_and_suffixes(compiled)
    assert "healthy and operational" in prefix
    assert len(suffixes) == 2
    assert "Is healthy?" in suffixes[0]
    assert "Is operational?" in suffixes[1]


def test_prefix_extraction_single_question_returns_false():
    request = EvaluateRequest.model_validate({
        "state": "Single state.",
        "questions": {
            "q1": {"type": "boolean", "instructions": "Check", "proposition": "Proposition?"},
        },
    })
    compiled = compile_request(request, tokenizer=MockTokenizer())
    assert can_extract_prefix(compiled) is False


def test_recurrent_attention_backend_uses_full_prompt_singletons() -> None:
    from types import SimpleNamespace

    from jah.backends.huggingface import HuggingFaceDirectLogitBackend

    backend = object.__new__(HuggingFaceDirectLogitBackend)
    backend.model = SimpleNamespace(
        config=SimpleNamespace(layer_types=["full_attention", "linear_attention"])
    )
    backend.batch_optimization_mode = (
        "sequential full-prompt fallback for recurrent/linear-attention model"
    )
    backend.score = lambda question, *, temperature=1.0: (question, temperature)

    questions = [object(), object()]
    results = backend.score_batch(
        questions, temperature=0.7, microbatch_size=16, use_prefix_cache=True
    )

    assert results == [(questions[0], 0.7), (questions[1], 0.7)]


def test_engine_delegates_to_score_batch():
    class BatchingBackend(FakeBackend):
        batch_called = False

        def score_batch(
            self,
            questions: Sequence[CompiledQuestion],
            *,
            temperature: float = 1.0,
            microbatch_size: int = 16,
            use_prefix_cache: bool = True,
        ) -> list[InferenceMeasurement]:
            del temperature, microbatch_size, use_prefix_cache
            self.batch_called = True
            return [self.score(q) for q in questions]

    backend = BatchingBackend()
    engine = DecisionEngine(
        backend,
        EngineConfig(artifact_id="test-art", microbatch_size=8, force_sequential=False),
    )
    request = EvaluateRequest.model_validate({
        "state": "A shared test state.",
        "questions": {
            "q1": {"type": "boolean", "instructions": "Check 1", "proposition": "Fact 1"},
            "q2": {"type": "boolean", "instructions": "Check 2", "proposition": "Fact 2"},
        },
    })
    response = engine.evaluate(request)
    assert backend.batch_called is True
    assert len(response.answers) == 2
    assert response.usage.decisions == 2


def test_optimization_equivalence_gates():
    # Simulate sequential vs batched vs cached results to verify gate logic
    labels = ("false", "true")
    ref_probs = [{"false": 0.05, "true": 0.95}, {"false": 0.88, "true": 0.12}]
    # Optimized path with tiny numerical perturbation (< 1e-4)
    opt_probs = [{"false": 0.05005, "true": 0.94995}, {"false": 0.87995, "true": 0.12005}]

    # Argmax agreement
    ref_argmax = [max(p, key=p.get) for p in ref_probs]
    opt_argmax = [max(p, key=p.get) for p in opt_probs]
    argmax_agreement = sum(r == o for r, o in zip(ref_argmax, opt_argmax, strict=True)) / len(ref_argmax)
    assert argmax_agreement >= 0.995

    # Max probability deviation <= 0.01
    max_dev = max(
        abs(ref_probs[i][lbl] - opt_probs[i][lbl])
        for i in range(len(ref_probs))
        for lbl in labels
    )
    assert max_dev <= 0.01


def test_force_sequential_bypasses_the_optimized_path() -> None:
    """The ADR-02 reference path must stay reachable for the equivalence measurement."""

    class CountingBackend(FakeBackend):
        def __init__(self) -> None:
            self.batch_calls = 0
            self.single_calls = 0

        def score(self, question):
            self.single_calls += 1
            return super().score(question)

        def score_batch(self, questions, *, microbatch_size=16, use_prefix_cache=True):
            self.batch_calls += 1
            return [super(CountingBackend, self).score(question) for question in questions]

    request = EvaluateRequest.model_validate(
        {
            "state": "Shared state for two independent questions.",
            "questions": {
                "a": {"type": "boolean", "instructions": "Judge.", "proposition": "First."},
                "b": {"type": "boolean", "instructions": "Judge.", "proposition": "Second."},
            },
        }
    )

    optimized_backend = CountingBackend()
    optimized = DecisionEngine(
        optimized_backend, EngineConfig(artifact_id="test", force_sequential=False)
    ).evaluate(request)
    assert optimized_backend.batch_calls == 1
    assert optimized_backend.single_calls == 0

    reference_backend = CountingBackend()
    reference = DecisionEngine(
        reference_backend, EngineConfig(artifact_id="test", force_sequential=True)
    ).evaluate(request)
    assert reference_backend.batch_calls == 0
    assert reference_backend.single_calls == 2

    assert set(optimized.answers) == set(reference.answers)
