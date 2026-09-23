"""Prepare CLINC150's untouched test rows for a JAH travel-intent evaluation.

Only ``test`` and ``oos_test`` partitions are read. The archive, derived examples, and
request fixtures belong under evals/cache and are intentionally not committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

TRAVEL_INTENTS = {
    "book_flight": "Arrange or purchase an airline flight.",
    "book_hotel": "Find or book lodging for a trip.",
    "car_rental": "Rent a car for travel.",
    "carry_on": "Ask about cabin baggage or carry-on rules.",
    "exchange_rate": "Ask about currency exchange rates for travel.",
    "flight_status": "Check the status or timing of a flight.",
    "international_visa": "Ask about visa or entry requirements for a trip.",
    "lost_luggage": "Report or ask about lost travel baggage.",
    "plug_type": "Ask which electrical plug or adapter is used at a destination.",
    "timezone": "Ask about the local time or time zone at a destination.",
    "translate": "Ask how to translate a phrase for travel.",
    "travel_alert": "Ask about travel warnings or alerts for a destination.",
    "travel_notification": "Ask about notifications for a trip or itinerary.",
    "travel_suggestion": "Ask for destination or travel-activity suggestions.",
    "vaccines": "Ask which vaccinations are needed before a trip.",
}
SMALL_TALK_INTENTS = {
    "are_you_a_bot",
    "do_you_have_pets",
    "fun_fact",
    "goodbye",
    "greeting",
    "how_old_are_you",
    "meaning_of_life",
    "tell_joke",
    "thank_you",
    "what_are_your_hobbies",
    "what_can_i_ask_you",
    "what_is_your_name",
    "where_are_you_from",
    "who_do_you_work_for",
    "who_made_you",
}
OTHER_DESCRIPTION = "The utterance is not one of the listed travel intents or is out of scope."
SOURCE_URL = "https://archive.ics.uci.edu/static/public/570/clinc150.zip"
ANNOTATION_DOC = "CLINC150 authors' crowdsourced intent labels; UCI dataset record and archive metadata"
EXPECTED_DATA_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    source: Path,
    output: Path,
    request_dir: Path,
    request_pairs: int,
    seed: str,
    expected_source_sha256: str | None,
) -> dict:
    source = source.resolve()
    source_file_sha256 = sha256_file(source)
    if expected_source_sha256 and source_file_sha256 != expected_source_sha256:
        raise ValueError(
            "CLINC150 source data hash changed; review the source and update the pinned "
            f"hash deliberately (expected {expected_source_sha256}, got {source_file_sha256})"
        )
    data = json.loads(source.read_text(encoding="utf-8"))
    for split in ("train", "val", "oos_train", "oos_val"):
        if split not in data:
            raise ValueError(f"CLINC150 archive lacks required partition {split!r}")

    seen_before_test = {
        normalized_text(utterance)
        for split in ("train", "val", "oos_train", "oos_val")
        for utterance, _label in data[split]
    }
    selected: list[tuple[str, str, str, str, int]] = []
    exact_train_dev_overlaps_excluded = 0
    for split in ("test", "oos_test"):
        if split not in data:
            raise ValueError(f"CLINC150 archive lacks required held-out partition {split!r}")
        for index, row in enumerate(data[split]):
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError(f"unexpected record in {split}[{index}]")
            utterance, original_label = row
            if normalized_text(utterance) in seen_before_test:
                exact_train_dev_overlaps_excluded += 1
                continue
            if original_label in TRAVEL_INTENTS:
                selected.append((split, utterance, original_label, original_label, index))
            elif original_label in SMALL_TALK_INTENTS or original_label == "oos":
                selected.append((split, utterance, "other", original_label, index))

    required_intents = set(TRAVEL_INTENTS)
    observed_intents = {label for _split, _utterance, label, _raw, _index in selected}
    if not required_intents <= observed_intents or "other" not in observed_intents:
        raise ValueError("held-out data is missing one or more travel intents or the other class")
    if len(selected) < 1_000:
        raise ValueError(f"prepared test set is too small: {len(selected)} decisions")

    source_digest = source_file_sha256
    options = [
        {"id": intent, "description": description}
        for intent, description in sorted(TRAVEL_INTENTS.items())
    ] + [{"id": "other", "description": OTHER_DESCRIPTION}]
    examples = []
    for split, utterance, label, original_label, source_index in selected:
        example_id = f"clinc150-{split}-{original_label}-{source_index:05d}"
        text_digest = hashlib.sha256(normalized_text(utterance).encode("utf-8")).hexdigest()[:20]
        examples.append(
            {
                "example_id": example_id,
                "source_group_id": example_id,
                "near_duplicate_cluster_id": f"text-{text_digest}",
                "workload_id": "clinc150-travel-intent-v1",
                "task_id": "travel-intent",
                "template_id": "single-utterance-choice-v1",
                "request": {
                    "state": utterance,
                    "questions": {
                        "intent": {
                            "type": "choice",
                            "instructions": (
                                "Select the user's travel intent. Choose other when the utterance "
                                "is outside the listed travel intents or is out of scope."
                            ),
                            "options": options,
                        }
                    },
                    "profile": None,
                },
                "reference_answers": {"intent": label},
                "rubric": (
                    "Map the CLINC150 held-out label to its matching travel intent. CLINC150 "
                    "small-talk and out-of-scope examples map to other."
                ),
                "provenance": {"kind": "public-benchmark", "author": "clinc150-uci"},
                "annotations": [],
                "annotation_status": "upstream-human",
                "public_source": {
                    "dataset": "clinc150",
                    "revision": f"data_full.json-sha256-{source_digest}",
                    "file_sha256": source_digest,
                    "source_url": SOURCE_URL,
                    "license": "CC BY 3.0 (license included in downloaded archive)",
                    "annotation_documentation": ANNOTATION_DOC,
                    "upstream_id": f"{split}-{source_index}",
                    "upstream_split": "test",
                    "original_label": original_label,
                },
                "evaluation_split": "locked_test",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")

    request_dir.mkdir(parents=True, exist_ok=True)
    chosen = random.Random(seed).sample(selected, min(len(selected), request_pairs * 2))
    request_options = [
        {"id": option["id"], "description": option["description"]} for option in options
    ]
    for pair_index in range(0, len(chosen) - 1, 2):
        pair = chosen[pair_index : pair_index + 2]
        request = {
            "state": f"Utterance 1: {pair[0][1]}\nUtterance 2: {pair[1][1]}",
            "profile": None,
            "questions": {
                f"utterance_{number + 1}": {
                    "type": "choice",
                    "instructions": f"Classify utterance {number + 1} in the supplied state.",
                    "options": request_options,
                }
                for number in range(2)
            },
        }
        request_path = request_dir / f"clinc150-pair-{pair_index // 2:04d}.json"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    return {
        "source_file_sha256": source_file_sha256,
        "prepared_decisions": len(examples),
        "prepared_examples": len(examples),
        "exact_train_dev_overlaps_excluded": exact_train_dev_overlaps_excluded,
        "travel_test_decisions": sum(
            label in TRAVEL_INTENTS for _, _, label, _raw, _index in selected
        ),
        "small_talk_test_decisions": sum(
            raw_label in SMALL_TALK_INTENTS
            for _split, _utterance, _mapped, raw_label, _index in selected
        ),
        "oos_test_decisions": sum(
            raw_label == "oos" for _split, _utterance, _mapped, raw_label, _index in selected
        ),
        "equivalence_requests": len(list(request_dir.glob("clinc150-pair-*.json"))),
        "output": output.as_posix(),
        "request_directory": request_dir.as_posix(),
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("evals/cache/clinc150/clinc150_uci/data_full.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evals/cache/clinc150/clinc150-travel-test.jsonl")
    )
    parser.add_argument(
        "--request-dir", type=Path, default=Path("evals/cache/clinc150/equivalence-requests")
    )
    parser.add_argument("--equivalence-pairs", type=int, default=100)
    parser.add_argument("--seed", default="clinc150-fresh-equivalence-v1")
    parser.add_argument("--expected-source-sha256", default=EXPECTED_DATA_SHA256)
    args = parser.parse_args()
    if args.equivalence_pairs < 1:
        parser.error("--equivalence-pairs must be at least one")
    print(
        json.dumps(
            prepare(
                args.source,
                args.output,
                args.request_dir,
                args.equivalence_pairs,
                args.seed,
                args.expected_source_sha256,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
