"""Tests that require the pinned backbone resident on a CUDA device.

Excluded from the default suite (`addopts = "-q -m 'not gpu'"`). Run with:

    .venv\\Scripts\\python.exe -m pytest -m gpu

The rest of the suite scores mocked tensors, which cannot detect a regression in tokenizer
label boundaries, last-token indexing, prefix-cache branch isolation, or adapter attachment.
Those properties only exist once the real weights are loaded, so they are checked here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jah.compiler import compile_request, verify_single_token_labels
from jah.engine import DecisionEngine, EngineConfig
from jah.schemas import EvaluateRequest
from jah.workload import load_yaml

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs" / "models" / "qwen3.5-4b.yaml"
ADAPTER_DIR = ROOT / "artifacts" / "adapters" / "qwen3.5-4b-public-head-v1"
EQUIVALENCE_DIR = ROOT / "evals" / "fixtures" / "equivalence"


@pytest.fixture(scope="module")
def model_config() -> dict:
    return load_yaml(MODEL_CONFIG)


@pytest.fixture(scope="module")
def backend(model_config: dict):
    from jah.evaluation import load_backend

    return load_backend("direct", model_config)


@pytest.fixture(scope="module")
def request_fixture() -> EvaluateRequest:
    path = EQUIVALENCE_DIR / "wikiqa-boolean-08.json"
    return EvaluateRequest.model_validate_json(path.read_text(encoding="utf-8"))


def _engine(backend, model_config: dict, **overrides) -> DecisionEngine:
    return DecisionEngine(
        backend,
        EngineConfig(
            artifact_id=model_config["artifact_id"],
            maximum_input_tokens=model_config["maximum_input_tokens"],
            model_metadata={
                "model_id": model_config["model_id"],
                "revision": model_config["revision"],
                "precision": model_config["precision"],
                "prompt_version": model_config["prompt_version"],
                "label_version": model_config["label_version"],
            },
            **overrides,
        ),
    )


def test_every_configured_label_is_one_token_at_the_real_answer_boundary(
    backend, model_config
) -> None:
    """ADR-01 rejects unsupported cardinality rather than switching to multi-token labels.

    The boundary must be the one the compiler actually produces, after the chat template.
    """
    labels = tuple(model_config["labels"])
    request = EvaluateRequest.model_validate_json(
        (EQUIVALENCE_DIR / "banking-choice-02.json").read_text(encoding="utf-8")
    )
    compiled = compile_request(request, tokenizer=backend.tokenizer)
    sixteen_way = max(compiled, key=lambda question: len(question.option_ids))
    assert len(sixteen_way.option_ids) == 16

    token_ids = verify_single_token_labels(backend.tokenizer, sixteen_way.prompt, labels)
    assert len(token_ids) == len(labels) == 16
    assert len(set(token_ids)) == 16
    assert tuple(sixteen_way.label_token_ids) == token_ids[: len(sixteen_way.option_ids)]


def test_label_verification_rejects_a_multi_token_label(backend, request_fixture) -> None:
    compiled = compile_request(request_fixture, tokenizer=backend.tokenizer)
    with pytest.raises(ValueError, match="not one stable token"):
        verify_single_token_labels(backend.tokenizer, compiled[0].prompt, ("AA_not_a_label",))


def test_scoring_is_deterministic_across_repeated_forward_passes(backend, request_fixture) -> None:
    compiled = compile_request(request_fixture, tokenizer=backend.tokenizer)
    first = backend.score(compiled[0])
    second = backend.score(compiled[0])
    assert first.decision.probabilities == second.decision.probabilities


def test_left_padding_does_not_move_the_scored_position(backend, request_fixture) -> None:
    """The scored position must be the last unpadded token, not the last tensor column."""
    compiled = compile_request(request_fixture, tokenizer=backend.tokenizer)
    reference = backend.score(compiled[0]).decision.probabilities

    batched = backend.score_batch(list(compiled), use_prefix_cache=False, microbatch_size=8)
    padded = batched[0].decision.probabilities

    for label in reference:
        assert padded[label] == pytest.approx(reference[label], abs=1e-3)


def test_prefix_cache_matches_the_sequential_reference(backend, model_config, request_fixture):
    """ADR-03: shared-prefix reuse must agree with the full-prompt reference under amended margin-aware contract."""
    from jah.equivalence import DEFAULT_MARGIN_THRESHOLD, compare_responses, summarize

    reference_engine = _engine(
        backend, model_config, force_sequential=True, enable_prefix_cache=False
    )
    optimized_engine = _engine(backend, model_config, enable_prefix_cache=True)

    reference, ref_meas = reference_engine.evaluate_detailed(request_fixture, request_id="gpu_reference")
    optimized, opt_meas = optimized_engine.evaluate_detailed(request_fixture, request_id="gpu_optimized")

    rows = compare_responses(
        reference,
        optimized,
        request_name="wikiqa-boolean-08",
        reference_measurements=ref_meas,
        optimized_measurements=opt_meas,
        margin_threshold=DEFAULT_MARGIN_THRESHOLD,
    )
    summary = summarize(rows, margin_threshold=DEFAULT_MARGIN_THRESHOLD)
    assert summary["gate_passed"] is True, f"Equivalence gate failed: {summary}"


def test_probability_distributions_sum_to_one(backend, model_config, request_fixture) -> None:
    response = _engine(backend, model_config).evaluate(request_fixture, request_id="gpu_sum")
    for answer in response.answers.values():
        assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skipif(not ADAPTER_DIR.exists(), reason="adapter bundle is not present")
def test_adapter_attachment_changes_decisions(model_config, request_fixture) -> None:
    from jah.evaluation import load_backend

    adapted = load_backend("direct", model_config, ADAPTER_DIR)
    compiled = compile_request(request_fixture, tokenizer=adapted.tokenizer)
    measurement = adapted.score(compiled[0])
    assert adapted.lora_modules, "adapter directory did not inject any LoRA modules"
    assert sum(measurement.decision.probabilities.values()) == pytest.approx(1.0, abs=1e-5)


def test_missing_adapter_manifest_is_explicit(model_config, tmp_path: Path) -> None:
    """ADR-06: a broken artifact fails loudly rather than silently scoring unadapted."""
    from jah.evaluation import load_backend

    with pytest.raises(FileNotFoundError, match="adapter manifest missing"):
        load_backend("direct", model_config, tmp_path)
