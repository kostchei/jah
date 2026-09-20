"""Isolate why batched scoring diverges from the single-item reference.

`jah-eval equivalence` showed that every configuration which batches diverges from the reference
(37 of 69 decisions, up to 0.0897) while microbatch size 1 is exact. That experiment cannot
distinguish two candidate causes, because a microbatch of one has neither padding nor a batch
dimension greater than one:

  A. padding — pad positions perturb the real positions' outputs
  B. batching — identical inputs produce different numerics at batch size > 1

This script separates them:

  Experiment 1 (batching, no padding): replicate one prompt N times. Every row is identical, so
    no padding is emitted. Any difference from the single-item result is cause B.
  Experiment 2 (padding): batch prompts of genuinely different lengths and compare each row
    against its own single-item result. Any additional difference is cause A.
  Experiment 3 (position sensitivity): move the same prompt to different rows of a padded batch
    and check whether its result depends on the row it occupies.

Everything is reported in logit space as well as probability space, because a probability
deviation is only interpretable next to the top-two margin it moved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jah.compiler import CompiledQuestion, compile_request
from jah.schemas import EvaluateRequest
from jah.workload import load_yaml


def _label_logits(final, question: CompiledQuestion) -> dict[str, float]:
    return {
        option_id: float(final[token_id].item())
        for option_id, token_id in zip(question.option_ids, question.label_token_ids, strict=True)
    }


def _softmax(values: dict[str, float]) -> dict[str, float]:
    import math

    peak = max(values.values())
    exponentials = {key: math.exp(value - peak) for key, value in values.items()}
    total = sum(exponentials.values())
    return {key: value / total for key, value in exponentials.items()}


def _top_two_margin(probabilities: dict[str, float]) -> float:
    ordered = sorted(probabilities.values(), reverse=True)
    return ordered[0] - ordered[1]


def _forward(backend, prompts: list[str]) -> Any:
    """One forward pass over a list of prompts, returning final-position logits per row."""
    import torch

    from jah.scoring import last_unpadded_logits

    encoded = backend.tokenizer(
        prompts, padding=True, return_tensors="pt", add_special_tokens=False
    )
    encoded = {key: value.to(backend.device) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = backend.model(**encoded, use_cache=False, return_dict=True)
    return last_unpadded_logits(outputs.logits, encoded["attention_mask"]), encoded


def _compare(reference: dict[str, float], other: dict[str, float]) -> dict[str, float]:
    reference_probabilities = _softmax(reference)
    other_probabilities = _softmax(other)
    return {
        "max_logit_deviation": max(
            abs(reference[key] - other[key]) for key in reference
        ),
        "max_probability_deviation": max(
            abs(reference_probabilities[key] - other_probabilities[key]) for key in reference
        ),
        "reference_top_two_margin": _top_two_margin(reference_probabilities),
        "argmax_agrees": max(reference_probabilities, key=reference_probabilities.get)
        == max(other_probabilities, key=other_probabilities.get),
    }


def run(args: argparse.Namespace) -> int:
    from jah.evaluation import load_backend

    root = Path(args.root).resolve()
    model_config = load_yaml(root / args.model_config)
    backend = load_backend("direct", model_config)

    fixture_dir = root / "evals" / "fixtures" / "equivalence"
    findings: dict[str, Any] = {"model": model_config["model_id"]}

    # --- Experiment 1: batching with no padding at all -------------------------------------
    request = EvaluateRequest.model_validate_json(
        (fixture_dir / "banking-choice-02.json").read_text(encoding="utf-8")
    )
    compiled = compile_request(request, tokenizer=backend.tokenizer)
    probe = max(compiled, key=lambda question: len(question.option_ids))

    single_final, _ = _forward(backend, [probe.prompt])
    single = _label_logits(single_final[0], probe)

    experiment_one = {}
    for batch_size in (2, 4, 8, 16):
        finals, encoded = _forward(backend, [probe.prompt] * batch_size)
        widths = encoded["attention_mask"].sum(dim=1).tolist()
        assert len(set(widths)) == 1, "replicated prompts must not be padded"
        rows = [_label_logits(finals[index], probe) for index in range(batch_size)]
        experiment_one[f"batch_{batch_size}"] = {
            "padded": False,
            "against_single_item": _compare(single, rows[0]),
            "row_to_row_spread": max(
                max(abs(rows[0][key] - row[key]) for key in rows[0]) for row in rows
            ),
        }
    findings["experiment_1_batching_without_padding"] = experiment_one

    # --- Experiment 2: a padded batch of genuinely different lengths -----------------------
    wikiqa = EvaluateRequest.model_validate_json(
        (fixture_dir / "wikiqa-boolean-08.json").read_text(encoding="utf-8")
    )
    wikiqa_compiled = list(compile_request(wikiqa, tokenizer=backend.tokenizer))
    lengths = [
        len(backend.tokenizer.encode(question.prompt, add_special_tokens=False))
        for question in wikiqa_compiled
    ]

    singles = []
    for question in wikiqa_compiled:
        final, _ = _forward(backend, [question.prompt])
        singles.append(_label_logits(final[0], question))

    finals, encoded = _forward(backend, [question.prompt for question in wikiqa_compiled])
    padded_width = int(encoded["input_ids"].shape[1])
    experiment_two = []
    for index, question in enumerate(wikiqa_compiled):
        batched = _label_logits(finals[index], question)
        experiment_two.append(
            {
                "question_id": question.question_id,
                "prompt_tokens": lengths[index],
                "pad_positions": padded_width - lengths[index],
                **_compare(singles[index], batched),
            }
        )
    findings["experiment_2_padded_batch"] = {
        "padded_width": padded_width,
        "prompt_token_lengths": lengths,
        "rows": experiment_two,
    }

    # --- Experiment 3: does a row's result depend on where it sits in the batch? -----------
    short = min(wikiqa_compiled, key=lambda question: len(question.prompt))
    long = max(wikiqa_compiled, key=lambda question: len(question.prompt))
    single_short, _ = _forward(backend, [short.prompt])
    short_alone = _label_logits(single_short[0], short)

    experiment_three = {}
    for name, prompts, position in (
        ("first_of_two", [short.prompt, long.prompt], 0),
        ("second_of_two", [long.prompt, short.prompt], 1),
    ):
        finals, _ = _forward(backend, prompts)
        experiment_three[name] = _compare(short_alone, _label_logits(finals[position], short))
    findings["experiment_3_row_position"] = experiment_three

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(findings, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(findings, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path.cwd())
    parser.add_argument("--model-config", default="configs/models/qwen3.5-4b.yaml")
    parser.add_argument("--output", default="artifacts/m3/batch-divergence-diagnosis.json")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
