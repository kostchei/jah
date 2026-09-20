"""Build the frozen optimization-equivalence regression set from public suite content.

The equivalence gate compares the single-item reference path against shared-prefix reuse and
microbatching, so every regression request must carry more than one question over a shared
state. Public suite rows are one question each, so this script composes them into multi-question
requests using the same real text. Labels are irrelevant here: the gate measures agreement
between two execution paths, not correctness.

Shapes cover the section 7 benchmark axes that the optimized path can reach: 2 / 4 / 8 / 16 / 32
questions, short and long states, and all three primitives including a mixed-primitive request.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from jah.schemas import EvaluateRequest
from jah.workload import sha256_file, stable_json_sha256

BANKING_COARSE_OPTIONS = [
    {"id": "card_lifecycle", "description": "Ordering, delivery, activation, or expiry of a card."},
    {"id": "payments", "description": "Transfers, top-ups, fees, or payment acceptance."},
    {"id": "access", "description": "Eligibility, identity, linking, or account access."},
    {"id": "other", "description": "No listed group applies or the evidence is insufficient."},
]

ASAP_LEVELS = [
    {"id": "score-1", "description": "Little or no command of source-based writing.", "value": 1.0},
    {"id": "score-2", "description": "Limited command; weak use of the source.", "value": 2.0},
    {"id": "score-3", "description": "Partial command; uneven use of the source.", "value": 3.0},
    {"id": "score-4", "description": "Adequate command; generally sound use of the source.", "value": 4.0},
    {"id": "score-5", "description": "Strong command; effective use of the source.", "value": 5.0},
    {"id": "score-6", "description": "Thorough command; skillful use of the source.", "value": 6.0},
]

ASAP_TRAIT_PROMPTS = [
    ("claim", "the clarity and defensibility of the central claim"),
    ("evidence", "the use of specific evidence drawn from the source text"),
    ("organization", "the organization and logical progression of ideas"),
    ("language", "the control of sentence structure and word choice"),
    ("conventions", "the control of spelling, punctuation, and grammar"),
    ("audience", "the appropriateness of tone and register for the stated audience"),
    ("reasoning", "the quality of reasoning connecting evidence to the claim"),
    ("conclusion", "the effectiveness of the concluding position"),
]

ASAP_BOOLEAN_PROMPTS = [
    ("cites_source", "The essay refers to material from the supplied source text."),
    ("takes_position", "The essay states a clear position on the prompt question."),
    ("addresses_counter", "The essay acknowledges at least one opposing consideration."),
    ("names_audience", "The essay addresses the audience named in the prompt."),
]


def _load_rows(suite_path: Path) -> dict[str, list[dict[str, Any]]]:
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with suite_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_workload[row["workload_id"]].append(row)
    return by_workload


def _wikiqa_request(rows: list[dict[str, Any]], *, question_count: int) -> EvaluateRequest:
    """One WikiQA document becomes one shared state with a proposition per candidate sentence."""
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["source_group_id"]].append(row)

    for group_id in sorted(by_group):
        group = sorted(by_group[group_id], key=lambda item: item["example_id"])
        if len(group) < question_count:
            continue
        group = group[:question_count]
        first = group[0]["request"]["state"]
        sentences = [item["request"]["state"]["candidate_sentence"] for item in group]
        state = {
            "question": first["question"],
            "document_title": first["document_title"],
            "candidate_sentences": {
                f"s{index:02d}": sentence for index, sentence in enumerate(sentences, start=1)
            },
        }
        questions = {
            f"s{index:02d}": {
                "type": "boolean",
                "instructions": (
                    "Judge whether the identified candidate sentence answers the supplied "
                    "question. Topical similarity alone is insufficient."
                ),
                "proposition": f"Candidate sentence s{index:02d} answers the supplied question.",
            }
            for index in range(1, len(sentences) + 1)
        }
        return EvaluateRequest.model_validate({"state": state, "questions": questions})

    raise ValueError(f"no WikiQA source group supplies {question_count} candidate sentences")


def _asap_rows_sorted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: item["example_id"])


def _asap_state(row: dict[str, Any]) -> dict[str, Any]:
    state = dict(row["request"]["state"])
    return state


def _asap_score_request(rows: list[dict[str, Any]], *, question_count: int) -> EvaluateRequest:
    """One essay becomes one shared state with an independent ordinal question per trait."""
    if question_count > len(ASAP_TRAIT_PROMPTS):
        raise ValueError(f"only {len(ASAP_TRAIT_PROMPTS)} ordinal traits are defined")
    row = _asap_rows_sorted(rows)[0]
    questions = {
        trait_id: {
            "type": "score",
            "instructions": (
                f"Using the six-level rubric, rate {description} in the supplied essay."
            ),
            "levels": ASAP_LEVELS,
        }
        for trait_id, description in ASAP_TRAIT_PROMPTS[:question_count]
    }
    return EvaluateRequest.model_validate({"state": _asap_state(row), "questions": questions})


def _asap_mixed_request(
    rows: list[dict[str, Any]],
    *,
    score_count: int,
    boolean_count: int,
    row_index: int = 1,
) -> EvaluateRequest:
    """A mixed-primitive request: ordinal traits, Boolean checks, and one Choice question."""
    row = _asap_rows_sorted(rows)[row_index]
    questions: dict[str, Any] = {}
    for trait_id, description in ASAP_TRAIT_PROMPTS[:score_count]:
        questions[trait_id] = {
            "type": "score",
            "instructions": (
                f"Using the six-level rubric, rate {description} in the supplied essay."
            ),
            "levels": ASAP_LEVELS,
        }
    for proposition_id, proposition in ASAP_BOOLEAN_PROMPTS[:boolean_count]:
        questions[proposition_id] = {
            "type": "boolean",
            "instructions": "Judge the proposition against the supplied essay only.",
            "proposition": proposition,
        }
    questions["overall_band"] = {
        "type": "choice",
        "instructions": "Select the band that best describes the essay as a whole.",
        "options": [
            {"id": "weak", "description": "Little or limited command of source-based writing."},
            {"id": "developing", "description": "Partial command with uneven source use."},
            {"id": "proficient", "description": "Adequate to strong command with sound source use."},
            {"id": "advanced", "description": "Thorough command with skillful source use."},
        ],
    }
    return EvaluateRequest.model_validate({"state": _asap_state(row), "questions": questions})


def _banking_request(rows: list[dict[str, Any]]) -> EvaluateRequest:
    """One customer query with two independent Choice questions of different cardinality."""
    row = min(rows, key=lambda item: item["example_id"])
    intent_question = dict(row["request"]["questions"]["decision"])
    return EvaluateRequest.model_validate(
        {
            "state": row["request"]["state"],
            "questions": {
                "intent": intent_question,
                "coarse_group": {
                    "type": "choice",
                    "instructions": "Select the coarse handling group for this customer query.",
                    "options": BANKING_COARSE_OPTIONS,
                },
            },
        }
    )


def build(suite_path: Path, output_dir: Path, benchmark_fixture: Path | None) -> dict[str, Any]:
    by_workload = _load_rows(suite_path)
    wikiqa = by_workload["wikiqa-answer-relevance-v1"]
    asap = by_workload["asap2-source-essay-v1"]
    banking = by_workload["banking77-16-intent-v1"]

    fixtures: dict[str, EvaluateRequest] = {
        "wikiqa-boolean-02": _wikiqa_request(wikiqa, question_count=2),
        "wikiqa-boolean-08": _wikiqa_request(wikiqa, question_count=8),
        "wikiqa-boolean-16": _wikiqa_request(wikiqa, question_count=16),
        "banking-choice-02": _banking_request(banking),
        "asap-score-08": _asap_score_request(asap, question_count=8),
        "asap-mixed-04": _asap_mixed_request(asap, score_count=2, boolean_count=1),
        "asap-mixed-13": _asap_mixed_request(asap, score_count=8, boolean_count=4, row_index=2),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(output_dir.glob("*.json")):
        path.unlink()

    manifest_entries = {}
    for name, request in sorted(fixtures.items()):
        path = output_dir / f"{name}.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(request.model_dump(mode="json"), handle, indent=2, sort_keys=True)
            handle.write("\n")
        manifest_entries[name] = {
            "questions": len(request.questions),
            "primitives": sorted({question.type for question in request.questions.values()}),
            "sha256": sha256_file(path),
        }

    if benchmark_fixture is not None:
        target = output_dir / benchmark_fixture.name
        target.write_text(benchmark_fixture.read_text(encoding="utf-8"), encoding="utf-8")
        request = EvaluateRequest.model_validate_json(target.read_text(encoding="utf-8"))
        manifest_entries[target.stem] = {
            "questions": len(request.questions),
            "primitives": sorted({question.type for question in request.questions.values()}),
            "sha256": sha256_file(target),
            "note": "Copied from the frozen 2,048-token, 16-Boolean latency benchmark request.",
        }

    manifest = {
        "schema_version": 1,
        "source_suite": suite_path.as_posix(),
        "source_suite_sha256": sha256_file(suite_path),
        "purpose": (
            "Optimization-equivalence regression set. Content is real public suite text; "
            "reference labels are deliberately absent because this set measures agreement "
            "between execution paths, not decision quality."
        ),
        "fixtures": manifest_entries,
    }
    manifest["manifest_sha256"] = stable_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="evals/data/public/suite.jsonl")
    parser.add_argument("--output-dir", default="evals/fixtures/equivalence")
    parser.add_argument("--benchmark-fixture", default="evals/fixtures/benchmark-2048-16b.json")
    args = parser.parse_args(argv)

    benchmark = Path(args.benchmark_fixture).resolve() if args.benchmark_fixture else None
    if benchmark is not None and not benchmark.exists():
        raise FileNotFoundError(f"benchmark fixture not found: {benchmark}")
    manifest = build(Path(args.suite).resolve(), Path(args.output_dir).resolve(), benchmark)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
