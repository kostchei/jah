"""Measure option-position and catch-all wording bias on the M0 workload."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from jah.compat import _macro_f1, require_minimum_label_mass
from jah.compiler import LABELS, compile_request
from jah.schemas import ChoiceOption, ChoiceQuestion, EvaluateRequest
from jah.workload import (
    git_commit_sha,
    load_dataset,
    load_yaml,
    sha256_file,
    stable_json_sha256,
    validate_dataset_against_workload,
)


def _ordered_options(
    options: list[ChoiceOption],
    *,
    rotation: int = 0,
    other_first: bool = False,
    shorten_other: bool = False,
) -> list[ChoiceOption]:
    values = [option.model_dump() for option in options]
    if shorten_other:
        for value in values:
            if value["id"] == "other":
                value["description"] = "None of the listed teams applies."
    if other_first:
        values = [value for value in values if value["id"] == "other"] + [
            value for value in values if value["id"] != "other"
        ]
    if rotation:
        offset = rotation % len(values)
        values = values[offset:] + values[:offset]
    return [ChoiceOption.model_validate(value) for value in values]


def _compile(
    example,
    *,
    instructions: str,
    tokenizer,
    maximum_input_tokens: int,
    state: str | None = None,
    rotation: int = 0,
    other_first: bool = False,
    shorten_other: bool = False,
):
    options = _ordered_options(
        example.options,
        rotation=rotation,
        other_first=other_first,
        shorten_other=shorten_other,
    )
    request = EvaluateRequest(
        state=example.state if state is None else state,
        questions={
            "route": ChoiceQuestion(
                type="choice",
                instructions=instructions,
                options=options,
            )
        },
    )
    return compile_request(
        request,
        tokenizer=tokenizer,
        max_input_tokens=maximum_input_tokens,
    )[0]


def _run_variants(
    backend, variants, baseline_predictions, minimum_label_mass: float, batch_size: int = 8
) -> dict:
    rows = []
    label_masses = []
    probability_by_letter: dict[str, list[float]] = defaultdict(list)
    other_wins_by_position: dict[str, list[bool]] = defaultdict(list)
    predicted_letters = Counter()
    cache = {}

    unique_questions = {}
    for variant in variants:
        question = variant["question"]
        unique_questions.setdefault(question.prompt, question)
    unique_items = list(unique_questions.items())
    for start in range(0, len(unique_items), batch_size):
        batch = unique_items[start : start + batch_size]
        measurements = backend.score_batch([question for _, question in batch])
        cache.update(
            (prompt, measurement)
            for (prompt, _), measurement in zip(batch, measurements, strict=True)
        )

    for variant in variants:
        question = variant["question"]
        measurement = cache[question.prompt]
        label_masses.append((variant["variant_id"], measurement.label_mass))
        letter_probabilities = {}
        predicted_letter = None
        other_position = None
        for label, option_id in zip(question.labels, question.option_ids, strict=True):
            probability = measurement.decision.probabilities[option_id]
            letter_probabilities[label] = probability
            probability_by_letter[label].append(probability)
            if option_id == measurement.decision.selected_id:
                predicted_letter = label
            if option_id == "other":
                other_position = label
        if predicted_letter is None or other_position is None:
            raise RuntimeError("diagnostic option-to-letter mapping is incomplete")
        predicted_letters[predicted_letter] += 1
        other_wins_by_position[other_position].append(measurement.decision.selected_id == "other")
        rows.append(
            {
                "example_id": variant["example_id"],
                "variant_id": variant["variant_id"],
                "rotation": variant["rotation"],
                "reference": variant["reference"],
                "prediction": measurement.decision.selected_id,
                "predicted_letter": predicted_letter,
                "other_position": other_position,
                "probabilities": measurement.decision.probabilities,
                "letter_probabilities": letter_probabilities,
                "label_mass": measurement.label_mass,
                "inference_ms": measurement.inference_ms,
            }
        )

    require_minimum_label_mass(label_masses, minimum_label_mass)
    references = [row["reference"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    labels = list(rows[0]["probabilities"])
    return {
        "cases": len(rows),
        "unique_forward_passes": len(cache),
        "forward_batches": (len(cache) + batch_size - 1) // batch_size,
        "microbatch_size": batch_size,
        "accuracy": sum(r == p for r, p in zip(references, predictions, strict=True)) / len(rows),
        "macro_f1": _macro_f1(references, predictions, labels),
        "predicted_class_counts": dict(sorted(Counter(predictions).items())),
        "predicted_letter_counts": dict(sorted(predicted_letters.items())),
        "label_aligned_flip_rate_vs_r0": sum(
            row["prediction"] != baseline_predictions[row["example_id"]] for row in rows
        )
        / len(rows),
        "minimum_label_mass": min(mass for _, mass in label_masses),
        "median_label_mass": statistics.median(mass for _, mass in label_masses),
        "mean_probability_by_letter": {
            label: statistics.fmean(probabilities)
            for label, probabilities in sorted(probability_by_letter.items())
        },
        "other_prediction_share_by_position": {
            label: sum(wins) / len(wins) for label, wins in sorted(other_wins_by_position.items())
        },
        "rows": rows,
    }


def _variants(examples, compile_variant, run_name: str) -> list[dict]:
    variants = []
    for example in examples:
        rotations = range(4) if run_name in {"R1", "R4"} else range(1)
        for rotation in rotations:
            variants.append(
                {
                    "example_id": example.example_id,
                    "variant_id": f"{run_name}:{example.example_id}:r{rotation}",
                    "rotation": rotation,
                    "reference": example.reference_answer,
                    "question": compile_variant(example, rotation),
                }
            )
    return variants


def _finding(runs: dict) -> dict[str, object]:
    position_shares = runs["R1"]["other_prediction_share_by_position"]
    d_share = position_shares["D"]
    other_position_max = max(value for label, value in position_shares.items() if label != "D")
    position_bias = d_share - other_position_max >= 0.10
    r0_other_share = runs["R0"]["predicted_class_counts"].get("other", 0) / runs["R0"]["cases"]
    r3_other_share = runs["R3"]["predicted_class_counts"].get("other", 0) / runs["R3"]["cases"]
    wording_bias = r0_other_share - r3_other_share >= 0.10
    if position_bias and wording_bias:
        branch = "both"
        action = (
            "Use order permutation/contextual calibration and rewrite the catch-all description."
        )
    elif position_bias:
        branch = "position-bias"
        action = "Use order permutation or contextual calibration and add permutation tests."
    elif wording_bias:
        branch = "wording-bias"
        action = "Rewrite the catch-all description and tune it on the development split only."
    else:
        branch = "neither"
        action = "Audit ambiguous labels and whether the M0 references are too lenient."
    return {
        "branch": branch,
        "m1_prompt_action": action,
        "heuristic": {
            "minimum_absolute_share_difference": 0.10,
            "position_bias": position_bias,
            "wording_bias": wording_bias,
            "r1_other_share_when_position_d": d_share,
            "r1_max_other_share_other_positions": other_position_max,
            "r0_other_share": r0_other_share,
            "r3_other_share": r3_other_share,
        },
    }


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    model_path = root / args.model_config
    workload_path = root / args.workload_config
    model_config = load_yaml(model_path)
    workload = load_yaml(workload_path)
    dataset_path = root / workload["dataset"]
    examples = load_dataset(dataset_path)
    validate_dataset_against_workload(examples, workload)
    dataset_sha256 = sha256_file(dataset_path)
    if dataset_sha256 != workload["dataset_sha256"]:
        raise ValueError(
            "dataset hash differs from the frozen workload definition: "
            f"expected {workload['dataset_sha256']}, observed {dataset_sha256}"
        )
    if model_config["labels"] != list(LABELS):
        raise ValueError("model-config labels do not match compiler labels")

    from jah.backends.huggingface import HuggingFaceDirectLogitBackend

    backend = HuggingFaceDirectLogitBackend(
        model_config["model_id"], model_config["revision"], device=model_config["device"]
    )
    maximum_input_tokens = model_config["maximum_input_tokens"]
    instructions = workload["instructions"]

    def compile_r0(example, rotation):
        return _compile(
            example,
            instructions=instructions,
            tokenizer=backend.tokenizer,
            maximum_input_tokens=maximum_input_tokens,
            rotation=rotation,
        )

    baseline_variants = _variants(examples, compile_r0, "R0")
    for warmup_index in range(5):
        backend.score(baseline_variants[warmup_index]["question"])
    minimum_label_mass = workload["provisional_feasibility_gate"]["minimum_label_mass"]
    empty_baseline = {example.example_id: "" for example in examples}
    r0 = _run_variants(backend, baseline_variants, empty_baseline, minimum_label_mass)
    baseline_predictions = {row["example_id"]: row["prediction"] for row in r0["rows"]}
    r0["label_aligned_flip_rate_vs_r0"] = 0.0

    compilers = {
        "R1": compile_r0,
        "R2": lambda example, rotation: _compile(
            example,
            instructions=instructions,
            tokenizer=backend.tokenizer,
            maximum_input_tokens=maximum_input_tokens,
            other_first=True,
        ),
        "R3": lambda example, rotation: _compile(
            example,
            instructions=instructions,
            tokenizer=backend.tokenizer,
            maximum_input_tokens=maximum_input_tokens,
            shorten_other=True,
        ),
        "R4": lambda example, rotation: _compile(
            example,
            instructions=instructions,
            tokenizer=backend.tokenizer,
            maximum_input_tokens=maximum_input_tokens,
            state="N/A",
            rotation=rotation,
        ),
    }
    runs = {"R0": r0}
    for run_name, compiler in compilers.items():
        runs[run_name] = _run_variants(
            backend,
            _variants(examples, compiler, run_name),
            baseline_predictions,
            minimum_label_mass,
        )

    report = {
        "schema_version": 1,
        "source_commit_sha": git_commit_sha(root),
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": dataset_sha256,
        "model": {
            "model_id": model_config["model_id"],
            "revision": model_config["revision"],
            "prompt_version": model_config["prompt_version"],
            "label_version": model_config["label_version"],
        },
        "run_definitions": {
            "R0": "Unmodified baseline.",
            "R1": "All four cyclic option rotations.",
            "R2": "Other moved to position A; remaining options retain their order.",
            "R3": 'Other description shortened to "None of the listed teams applies."',
            "R4": 'State replaced by "N/A" across all four cyclic rotations.',
        },
        "runs": runs,
    }
    report["finding"] = _finding(runs)
    report["evidence_sha256"] = stable_json_sha256(
        {
            key: value
            for key, value in report.items()
            if key not in {"created_at", "evidence_sha256"}
        }
    )
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "finding": report["finding"],
                "runs": {
                    k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in runs.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path.cwd())
    parser.add_argument("--model-config", default="configs/models/qwen3.5-4b.yaml")
    parser.add_argument("--workload-config", default="configs/workloads/support-routing-v1.yaml")
    parser.add_argument("--output", default="artifacts/m0/order_bias.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
