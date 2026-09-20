"""Pinned public benchmark importers; no model-generated labels or automatic approval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from jah.m1_dataset import (
    M1Example,
    build_split_manifest,
    source_components,
    split_assignments,
    validate_m1_suite,
)
from jah.workload import sha256_file, stable_json_sha256

DEFAULT_CONFIG = "configs/evals/public-sources.json"
DEFAULT_CACHE = "evals/cache/public"
DEFAULT_OUTPUT = "evals/data/public"


def fetch_sources(config: dict, cache: Path, *, offline: bool = False) -> None:
    """Download only pinned inputs, and verify cached files on every invocation."""
    cache.mkdir(parents=True, exist_ok=True)
    for name, source in config["files"].items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError(f"source filename must be a basename: {name}")
        destination = cache / name
        if not destination.exists():
            if offline:
                raise FileNotFoundError(f"missing pinned input: {destination}")
            if not source["url"].startswith("https://"):
                raise ValueError("source URLs must use HTTPS")
            request = urllib.request.Request(source["url"], headers={"User-Agent": "jah-data/1"})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"download checksum mismatch: {name}")
            destination.write_bytes(data)
        if sha256_file(destination) != source["sha256"]:
            raise ValueError(f"cached input checksum mismatch: {name}")


def _identity(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _example(
    config: dict, *, source: str, filename: str, upstream_id: str, split: str,
    workload: str, group: str, template: str, state: str | dict, question: dict,
    answer: str, original_label: str, rubric: str,
) -> M1Example:
    metadata = config["sources"][source]
    return M1Example.model_validate({
        "example_id": f"{metadata['dataset']}-{_identity(upstream_id)}",
        "source_group_id": f"{metadata['dataset']}-{_identity(group)}",
        "near_duplicate_cluster_id": f"{metadata['dataset']}-{_identity(upstream_id)}",
        "workload_id": workload,
        "task_id": workload,
        "template_id": template,
        "request": {"state": state, "questions": {"decision": question}},
        "reference_answers": {"decision": answer},
        "rubric": rubric,
        "provenance": {"kind": "public-benchmark", "author": metadata["dataset"]},
        "annotations": [],
        "annotation_status": "upstream-human",
        "public_source": {
            **{key: metadata[key] for key in (
                "dataset", "revision", "license", "source_url", "annotation_documentation"
            )},
            "file_sha256": config["files"][filename]["sha256"],
            "upstream_id": upstream_id,
            "upstream_split": split,
            "original_label": original_label,
        },
        "evaluation_split": {"dev": "development", "test": "locked_test"}.get(split),
    })


def import_banking(config: dict, cache: Path) -> tuple[list[M1Example], dict]:
    intents = config["banking_intents"]
    if not 2 <= len(intents) <= 16 or len(set(intents)) != len(intents):
        raise ValueError("BANKING77 requires 2-16 unique frozen intents")
    options = [{"id": intent, "description": intent.replace("_", " ")} for intent in intents]
    examples = []
    excluded = Counter()
    for split in ("train", "test"):
        filename = f"banking-{split}.csv"
        seen = set()
        with (cache / filename).open(encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if not row["text"].strip() or not row["category"].strip():
                    raise ValueError("BANKING77 contains an empty query/label")
                seen.add(row["category"])
                if row["category"] not in intents:
                    excluded["outside_frozen_16_intents"] += 1
                    continue
                identity = f"{split}:{index}"
                examples.append(_example(
                    config, source="banking", filename=filename, upstream_id=identity,
                    split=split, workload="banking77-16-intent-v1", group=identity,
                    template="banking77-choice-v1", state=row["text"],
                    question={"type": "choice", "options": options, "instructions": (
                        "Select the banking intent expressed by this customer query. "
                        "Choose exactly one of the supplied intents."
                    )},
                    answer=row["category"], original_label=row["category"],
                    rubric="Original BANKING77 intent within the frozen 16-intent closed set.",
                ))
        if not set(intents) <= seen:
            raise ValueError(f"BANKING77 {split} missing configured intents")
    return examples, dict(excluded)


def import_wikiqa(config: dict, cache: Path) -> tuple[list[M1Example], dict]:
    examples = []
    with zipfile.ZipFile(cache / "wikiqa.zip") as archive:
        for split in ("train", "dev", "test"):
            text = archive.read(f"WikiQACorpus/WikiQA-{split}.tsv").decode("utf-8-sig")
            for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
                if row["Label"] not in {"0", "1"}:
                    raise ValueError("WikiQA requires explicit binary judgments")
                if not row["Question"].strip() or not row["Sentence"].strip():
                    raise ValueError("WikiQA contains an empty question/sentence")
                identity = f"{row['QuestionID']}:{row['SentenceID']}"
                examples.append(_example(
                    config, source="wikiqa", filename="wikiqa.zip", upstream_id=identity,
                    split=split, workload="wikiqa-answer-relevance-v1",
                    group=row["DocumentID"], template="wikiqa-boolean-v1",
                    state={"question": row["Question"], "candidate_sentence": row["Sentence"],
                           "document_title": row["DocumentTitle"]},
                    question={"type": "boolean", "instructions": (
                        "Judge whether the candidate sentence answers the question. "
                        "Topical similarity alone is insufficient."
                    ), "proposition": "The candidate sentence answers the supplied question."},
                    answer="true" if row["Label"] == "1" else "false",
                    original_label=row["Label"],
                    rubric="Original WikiQA answer-sentence judgment: 1 answers; 0 does not.",
                ))
    return examples, {}


def read_asap_rubric(path: Path) -> list[dict]:
    """Preserve all rubric paragraphs, including the split score-1 paragraph."""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    levels: dict[int, str] = {}
    current = None
    for paragraph in root.findall(".//w:p", ns):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", ns)).strip()
        match = re.match(r"SCORE OF ([1-6]):", text)
        if match:
            current = int(match[1])
            if current in levels:
                raise ValueError("duplicate ASAP rubric level")
            levels[current] = text
        elif current is not None and text:
            levels[current] += " " + text
    if set(levels) != set(range(1, 7)):
        raise ValueError("ASAP rubric must contain all six original levels")
    return [{"id": f"score-{value}", "value": value, "description": levels[value]}
            for value in range(1, 7)]


def read_asap_sources(config: dict, cache: Path) -> dict[str, str]:
    from pypdf import PdfReader

    contexts = {}
    with zipfile.ZipFile(cache / "asap-sources.zip") as archive:
        for prompt, member in config["asap_prompts"].items():
            reader = PdfReader(io.BytesIO(archive.read(member)))
            contexts[prompt] = "\n".join(page.extract_text() or "" for page in reader.pages)
            if len(contexts[prompt].split()) < 300:
                raise ValueError(f"ASAP source is not text-extractable: {prompt}; OCR required")
    return contexts


def import_asap(config: dict, cache: Path) -> tuple[list[M1Example], dict]:
    levels = read_asap_rubric(cache / "asap-rubric.docx")
    contexts = read_asap_sources(config, cache)
    examples = []
    excluded = Counter()
    with zipfile.ZipFile(cache / "asap-train.zip") as archive:
        text = archive.read("ASAP_2_Final_github_train.csv").decode("utf-8-sig")
        for row in csv.DictReader(io.StringIO(text)):
            if row["set"] != "train":
                raise ValueError("unexpected held-out essay in ASAP training source")
            if row["prompt_name"] not in contexts:
                excluded["prompt_source_requires_ocr"] += 1
                continue
            if row["score"] not in {str(i) for i in range(1, 7)}:
                raise ValueError("ASAP requires an original integer score from 1 through 6")
            if not row["full_text"].strip() or not row["assignment"].strip():
                raise ValueError("ASAP essay/assignment must not be empty")
            examples.append(_example(
                config, source="asap", filename="asap-train.zip", upstream_id=row["essay_id"],
                split="train", workload="asap2-source-essay-v1", group=row["essay_id"],
                template=f"asap2-{_identity(row['prompt_name'])}",
                state={"essay": row["full_text"], "assignment": row["assignment"],
                       "source_texts": contexts[row["prompt_name"]]},
                question={"type": "score", "levels": levels, "instructions": (
                    "Assign the essay a holistic score from 1 to 6 using the supplied original "
                    "ASAP rubric, assignment, and source texts. Treat score intervals as equal."
                )},
                answer=f"score-{row['score']}", original_label=row["score"],
                rubric="Original ASAP 2.0 holistic 1-6 rubric; original released human score.",
            ))
    return examples, dict(excluded)


def _decision_text(example: M1Example) -> str:
    state = example.request.state
    if isinstance(state, str):
        return state
    if "essay" in state:
        return state["essay"]  # Shared assignment/source/rubric are fixed task context.
    return state["question"] + " " + state["candidate_sentence"]


def cluster_duplicates(examples: list[M1Example]) -> list[M1Example]:
    """Exact normalized text and >=90% Jaccard similarity of five-word shingles.

    Prefix filtering considers every pair meeting the threshold, without an
    approximate nearest-neighbor dependency. Cluster IDs are order independent.
    """
    examples = sorted(examples, key=lambda item: item.example_id)
    parent = list(range(len(examples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)

    exact: dict[str, int] = {}
    shingles: list[set[bytes]] = []
    index: dict[bytes, list[int]] = defaultdict(list)
    for i, example in enumerate(examples):
        words = re.findall(r"\w+", unicodedata.normalize("NFKC", _decision_text(example)).casefold())
        normalized = " ".join(words)
        if not normalized:
            raise ValueError(f"empty normalized decision text: {example.example_id}")
        duplicate = exact.get(normalized)
        if duplicate is not None:
            union(i, duplicate)
            shingles.append(set())
            continue
        exact[normalized] = i
        grams = {
            hashlib.sha256(" ".join(words[j:j + 5]).encode()).digest()[:16]
            for j in range(max(0, len(words) - 4))
        }
        shingles.append(grams)
        if not grams:
            continue
        prefix = sorted(grams)[:len(grams) - math.ceil(0.9 * len(grams)) + 1]
        candidates = {other for token in prefix for other in index[token]}
        for other in candidates:
            other_grams = shingles[other]
            if min(len(grams), len(other_grams)) < 0.9 * max(len(grams), len(other_grams)):
                continue
            if len(grams & other_grams) >= 0.9 * len(grams | other_grams):
                union(i, other)
        for token in prefix:
            index[token].append(i)
    return [item.model_copy(update={
        "near_duplicate_cluster_id": f"dup-{_identity(examples[find(i)].example_id)}"
    }) for i, item in enumerate(examples)]


def quarantine_boundary_conflicts(
    examples: list[M1Example],
) -> tuple[list[M1Example], list[str]]:
    retained = []
    quarantined = []
    for component in source_components(examples):
        boundaries = {(item.public_source.upstream_split, item.evaluation_split)
                      for item in component}
        if len(boundaries) > 1:
            quarantined.extend(item.example_id for item in component)
        else:
            retained.extend(component)
    return sorted(retained, key=lambda item: item.example_id), sorted(quarantined)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(config_path: Path, cache: Path, output: Path, *, offline: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != 1:
        raise ValueError("unsupported source manifest version")
    fetch_sources(config, cache, offline=offline)
    examples = []
    exclusions = {}
    for name, importer in (("banking", import_banking), ("wikiqa", import_wikiqa),
                           ("asap", import_asap)):
        imported, reasons = importer(config, cache)
        examples.extend(imported)
        exclusions[name] = reasons
    ids = [item.example_id for item in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate upstream example IDs")
    examples, quarantined = quarantine_boundary_conflicts(cluster_duplicates(examples))
    assignments = split_assignments(examples, seed=config["split_seed"])
    examples = [item.model_copy(update={"evaluation_split": assignments[item.example_id]})
                for item in examples]
    readiness = validate_m1_suite(examples, allowed_statuses=("upstream-human",))
    if not readiness["ready"]:
        raise ValueError("public suite is not ready: " + "; ".join(readiness["failures"]))
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "suite.jsonl"
    with dataset_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in examples:
            handle.write(item.model_dump_json() + "\n")
    manifest = build_split_manifest(examples, dataset_path=dataset_path,
                                    seed=config["split_seed"], dataset_label="suite.jsonl")
    write_json(output / "split-manifest.json", manifest)
    workloads = {}
    for workload in sorted({item.workload_id for item in examples}):
        selected = [item for item in examples if item.workload_id == workload]
        split_counts = Counter(assignments[item.example_id] for item in selected)
        workloads[workload] = {
            "decisions": len(selected),
            "split_counts": dict(sorted(split_counts.items())),
            "upstream_split_counts": dict(sorted(Counter(
                item.public_source.upstream_split for item in selected
            ).items())),
            "labels_by_split": {
                split: dict(sorted(Counter(item.reference_answers["decision"] for item in selected
                                           if assignments[item.example_id] == split).items()))
                for split in sorted(split_counts)
            },
            "calibration_source_groups": len({item.source_group_id for item in selected
                                              if assignments[item.example_id] == "calibration"}),
            "selective_error_certified": False,
            "reason": "No model evaluation or independent policy certification performed.",
        }
    report = {
        "schema_version": 1, "suite_id": config["suite_id"],
        "source_config_sha256": sha256_file(config_path),
        "dataset_sha256": manifest["dataset_sha256"],
        "assignment_sha256": manifest["assignment_sha256"],
        "readiness": readiness, "workloads": workloads,
        "excluded_by_adapter": exclusions, "quarantined_count": len(quarantined),
        "quarantined_ids_sha256": stable_json_sha256(quarantined),
        "deduplication": "NFKC/casefold word normalization; exact or 5-word Jaccard >=0.90",
        "source_grouping": {
            "banking": "query row; original customer identifiers are not provided",
            "wikiqa": "document ID; all question/sentence pairs for a document stay together",
            "asap": "essay ID plus duplicate cluster; shared prompt context is not an essay",
        },
        "release_approved": False,
        "limitations": [
            "Public benchmarks may occur in model pretraining; contamination is unknown.",
            "Upstream human labels do not establish local independent adjudication.",
            "Near-duplicate detection does not detect all semantic paraphrases.",
            "Sources can contain correlated decisions; binomial independence is not established.",
            "ASAP includes two prompts only; no task/template holdout is claimed.",
            "Token budgets have not been checked by import; evaluation rejects overlong inputs.",
        ],
    }
    write_json(output / "import-report.json", report)
    write_json(output / "quarantined.json", quarantined)
    write_json(output / "sources.json", config)
    (output / "NOTICE.md").write_text(
        "# Derived public benchmark\n\n"
        "Modified from original datasets by jah public-data importer v1, 2026-09-20.\n"
        "Labels retained; schemas, subsets, grouping, and splits adapted.\n"
        "Research/technology-development benchmark; not an approved production dataset.\n\n"
        + "\n\n".join(f"{s['attribution']}\n{s['source_url']}\nLicense: {s['license']}\n"
                        f"{s['limitations']}" for s in config["sources"].values()) + "\n",
        encoding="utf-8",
    )
    # Retain original terms with the derived data, including the full WikiQA agreement.
    (output / "BANKING77-LICENSE.txt").write_bytes((cache / "banking-license.txt").read_bytes())
    (output / "ASAP2-README.md").write_bytes((cache / "asap-readme.md").read_bytes())
    with zipfile.ZipFile(cache / "wikiqa.zip") as archive:
        (output / "WikiQA-LICENSE.pdf").write_bytes(archive.read("WikiQACorpus/LICENSE.pdf"))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--cache", type=Path, default=Path(DEFAULT_CACHE))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    report = build(args.config, args.cache, args.output, offline=args.offline)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
