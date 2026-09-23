import csv
import io
import json
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from jah import public_data
from jah.m1_dataset import M1Example, split_assignments, validate_m1_suite
from jah.public_data import (
    cluster_duplicates,
    fetch_sources,
    import_asap,
    import_banking,
    import_wikiqa,
    quarantine_boundary_conflicts,
    read_asap_rubric,
)
from jah.workload import sha256_file


@pytest.fixture
def source_config():
    return json.loads((Path(__file__).resolve().parents[1]
                       / "configs/evals/public-sources.json").read_text(encoding="utf-8"))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_public(source_config, *, identity="one", split="train", group=None, state="A query"):
    return public_data._example(
        source_config, source="banking", filename="banking-train.csv",
        upstream_id=identity, split=split, workload="test-public",
        group=group or identity, template="test-template", state=state,
        question={"type": "boolean", "instructions": "Check.", "proposition": "A fact."},
        answer="true", original_label="1", rubric="Original public judgment.",
    )


def test_pinned_cache_rejects_corruption_and_missing_files(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("known data", encoding="utf-8")
    config = {"files": {"data.csv": {"url": "https://example.com/data.csv",
                                     "sha256": sha256_file(path)}}}
    fetch_sources(config, tmp_path, offline=True)
    path.write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        fetch_sources(config, tmp_path, offline=True)
    with pytest.raises(FileNotFoundError, match="missing pinned input"):
        fetch_sources(config, tmp_path / "empty", offline=True)


def test_banking_preserves_intents_without_four_team_mapping(source_config, tmp_path):
    source_config["banking_intents"] = ["card_arrival", "card_not_working"]
    rows = [{"text": "Card still missing", "category": "card_arrival"},
            {"text": "Card is broken", "category": "card_not_working"},
            {"text": "A different intent", "category": "cancel_transfer"}]
    for split in ("train", "test"):
        write_csv(tmp_path / f"banking-{split}.csv", rows, ["text", "category"])
    imported, excluded = import_banking(source_config, tmp_path)
    assert len(imported) == 4
    assert excluded == {"outside_frozen_16_intents": 2}
    assert imported[0].reference_answers == {"decision": "card_arrival"}
    assert imported[0].public_source.original_label == "card_arrival"
    assert imported[0].annotations == []
    assert imported[2].evaluation_split == "locked_test"
    assert imported[0].request.profile is None
    assert [o.id for o in imported[0].request.questions["decision"].options] == (
        source_config["banking_intents"]
    )


def test_wikiqa_uses_explicit_negatives_and_groups_by_document(source_config, tmp_path):
    fields = ["QuestionID", "Question", "DocumentID", "DocumentTitle", "SentenceID",
              "Sentence", "Label"]
    with zipfile.ZipFile(tmp_path / "wikiqa.zip", "w") as archive:
        for split in ("train", "dev", "test"):
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for index in range(2):
                writer.writerow({"QuestionID": f"{split}-q", "Question": "When was it built?",
                                 "DocumentID": f"{split}-doc", "DocumentTitle": "Building",
                                 "SentenceID": f"{split}-{index}", "Sentence": "In 1920.",
                                 "Label": str(index)})
            archive.writestr(f"WikiQACorpus/WikiQA-{split}.tsv", buffer.getvalue())
    imported, _ = import_wikiqa(source_config, tmp_path)
    assert [item.reference_answers["decision"] for item in imported[:2]] == ["false", "true"]
    assert imported[0].source_group_id == imported[1].source_group_id
    assert imported[2].evaluation_split == "development"
    assert imported[4].evaluation_split == "locked_test"
    assert "Label" not in imported[0].request.state


def rubric_file(path):
    paragraphs = [f"SCORE OF {score}: Original descriptor {score}." for score in range(6, 0, -1)]
    paragraphs.append("Continuation of score one.")
    xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    xml += "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    xml += "</w:document>"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)


def test_asap_retains_score_rubric_context_without_demographics(
    source_config, tmp_path, monkeypatch,
):
    rubric_file(tmp_path / "asap-rubric.docx")
    levels = read_asap_rubric(tmp_path / "asap-rubric.docx")
    assert levels[0]["value"] == 1
    assert levels[-1]["value"] == 6
    assert levels[0]["description"].endswith("Continuation of score one.")
    monkeypatch.setattr(public_data, "read_asap_sources", lambda *_: {"Car-free cities": "Source"})
    row = {"essay_id": "essay1", "set": "train", "score": "3", "full_text": "An essay",
           "assignment": "Discuss the source.", "prompt_name": "Car-free cities", "gender": "F"}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    writer.writerow({**row, "essay_id": "essay2", "prompt_name": "OCR-needed"})
    with zipfile.ZipFile(tmp_path / "asap-train.zip", "w") as archive:
        archive.writestr("ASAP_2_Final_github_train.csv", buffer.getvalue())
    imported, excluded = import_asap(source_config, tmp_path)
    assert excluded == {"prompt_source_requires_ocr": 1}
    assert imported[0].reference_answers == {"decision": "score-3"}
    assert imported[0].public_source.original_label == "3"
    assert imported[0].request.state == {
        "essay": "An essay", "assignment": "Discuss the source.", "source_texts": "Source",
    }


def test_upstream_labels_are_not_local_adjudication(source_config):
    example = make_public(source_config)
    readiness = validate_m1_suite([example], minimum_decisions=1)
    assert any("approved annotation status" in failure for failure in readiness["failures"])
    assert readiness["release_annotation_ready"] is False
    payload = example.model_dump(mode="json")
    payload["annotation_status"] = "adjudicated-agreement"
    with pytest.raises(ValidationError, match="two distinct independent human"):
        M1Example.model_validate(payload)
    payload["annotation_status"] = "upstream-human"
    payload["public_source"] = None
    with pytest.raises(ValidationError, match="public source provenance"):
        M1Example.model_validate(payload)


@pytest.mark.parametrize("split,fixed", [("dev", "development"), ("test", "locked_test")])
def test_upstream_split_cannot_be_reassigned(source_config, split, fixed):
    example = make_public(source_config, split=split)
    assert split_assignments([example], seed="anything")[example.example_id] == fixed
    payload = example.model_dump(mode="json")
    payload["evaluation_split"] = "train"
    with pytest.raises(ValidationError, match="boundaries must be preserved"):
        M1Example.model_validate(payload)


def test_duplicate_and_source_components_are_quarantined_across_boundaries(source_config):
    examples = [
        make_public(source_config, identity="1", group="a", state="The exact text"),
        make_public(source_config, identity="2", group="b", state="THE exact text!", split="test"),
        make_public(source_config, identity="3", group="a", state="Other text entirely"),
        make_public(source_config, identity="4", state="Unique independent item"),
    ]
    clustered = cluster_duplicates(examples)
    with pytest.raises(ValueError, match="split boundaries"):
        split_assignments(clustered, seed="fixed")
    retained, quarantine = quarantine_boundary_conflicts(clustered)
    assert [item.example_id for item in retained] == [examples[3].example_id]
    assert set(quarantine) == {item.example_id for item in examples[:3]}


def test_near_duplicate_detection_is_order_independent(source_config):
    text = " ".join(f"word{index}" for index in range(100))
    examples = [make_public(source_config, identity="a", state=text),
                make_public(source_config, identity="b", state=text + " extra"),
                make_public(source_config, identity="c", state="Completely unrelated sentence")]
    first = {item.example_id: item.near_duplicate_cluster_id for item in cluster_duplicates(examples)}
    second = {item.example_id: item.near_duplicate_cluster_id
              for item in cluster_duplicates(list(reversed(examples)))}
    assert first == second
    assert first[examples[0].example_id] == first[examples[1].example_id]
    assert first[examples[0].example_id] != first[examples[2].example_id]


def test_public_suite_runs_and_calibrates_without_promoting_profiles(
    source_config, tmp_path, monkeypatch,
):
    from jah import evaluation
    from jah.calibration import load_profile_registry
    from tests.test_engine import FakeBackend, make_request

    examples = []
    for i, question in enumerate(make_request().questions.values()):
        answer = {"boolean": "true", "choice": "b", "score": "high"}[question.type]
        example = make_public(source_config, identity=str(i))
        payload = example.model_dump(mode="json")
        payload.update({
            "workload_id": f"public-{question.type}",
            "request": {"state": "A short test.", "questions": {"decision": question.model_dump()}},
            "reference_answers": {"decision": answer},
            "evaluation_split": "calibration",
        })
        examples.append(M1Example.model_validate(payload))
    dataset = tmp_path / "suite.jsonl"
    dataset.write_text("\n".join(e.model_dump_json() for e in examples) + "\n", encoding="utf-8")
    (tmp_path / "suite.yaml").write_text(
        "minimum_decisions: 3\nrequired_annotation_status: [upstream-human]\n", encoding="utf-8",
    )
    (tmp_path / "model.yaml").write_text(
        "artifact_id: test-artifact\nmodel_id: test-model\nrevision: test-rev\n"
        "precision: bf16\nprompt_version: decision-prompt-v2\n"
        "label_version: latin-uppercase-bare-v2\nmaximum_input_tokens: 8192\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation, "load_backend", lambda *_, **__: FakeBackend())
    base = ["--root", str(tmp_path)]
    common = ["--dataset", "suite.jsonl", "--suite-config", "suite.yaml",
              "--model-config", "model.yaml"]
    assert evaluation.main(base + ["run"] + common + [
        "--backend", "direct", "--split", "calibration", "--output", "report.json",
        "--predictions", "predictions.jsonl",
    ]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["evaluation_scope"] == "public-benchmark"
    assert report["release_annotation_ready"] is False
    assert report["decisions"] == 3
    assert evaluation.main(base + ["calibrate"] + common + [
        "--output", "profiles", "--summary", "calibration.json",
    ]) == 0
    registry = load_profile_registry(tmp_path / "profiles")
    assert set(registry) == {"public-boolean", "public-choice", "public-score"}
    assert all(p.policy.type == "review_only" for p in registry.values())
