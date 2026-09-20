from __future__ import annotations

import json
from pathlib import Path

import pytest

from jah.report import check, collect_gates, collect_metrics, generate, render


class Args:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifacts = "artifacts"
        self.output = "EVIDENCE.md"


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)


@pytest.fixture
def evidence_root(tmp_path: Path) -> Path:
    _write(
        tmp_path / "artifacts" / "public" / "report.json",
        {
            "evidence_sha256": "abc123",
            "metrics": {
                "workload/choice": {
                    "macro_f1": 0.8126,
                    "calibration": {
                        "ece": 0.0541,
                        "brier_score": 0.2646,
                        "gate_ece_passed": False,
                        "gate_brier_passed": True,
                        "gate_passed": False,
                    },
                }
            },
        },
    )
    return tmp_path


def test_collect_gates_finds_nested_gate_booleans(evidence_root: Path) -> None:
    document = json.loads(
        (evidence_root / "artifacts" / "public" / "report.json").read_text(encoding="utf-8")
    )
    gates = dict(collect_gates(document))
    assert gates["metrics.workload/choice.calibration.gate_ece_passed"] is False
    assert gates["metrics.workload/choice.calibration.gate_brier_passed"] is True
    assert len(gates) == 3


def test_collect_metrics_ignores_gate_booleans(evidence_root: Path) -> None:
    document = json.loads(
        (evidence_root / "artifacts" / "public" / "report.json").read_text(encoding="utf-8")
    )
    metrics = dict(collect_metrics(document))
    assert metrics["metrics.workload/choice.macro_f1"] == pytest.approx(0.8126)
    assert all(not isinstance(value, bool) for value in metrics.values())


def test_render_states_failed_gates_verbatim(evidence_root: Path) -> None:
    content = render(evidence_root, evidence_root / "artifacts")
    assert "**1 of 3 recorded gates pass.**" in content
    assert "| `metrics.workload/choice.calibration.gate_ece_passed` | **FAIL** |" in content


def test_render_is_deterministic(evidence_root: Path) -> None:
    first = render(evidence_root, evidence_root / "artifacts")
    second = render(evidence_root, evidence_root / "artifacts")
    assert first == second


def test_check_fails_when_documentation_drifts(evidence_root: Path) -> None:
    args = Args(evidence_root)
    assert generate(args) == 0
    assert check(args) == 0

    _write(
        evidence_root / "artifacts" / "public" / "report.json",
        {"metrics": {"workload/choice": {"calibration": {"gate_ece_passed": True}}}},
    )
    with pytest.raises(ValueError, match="stale relative to the artifacts"):
        check(args)


def test_check_requires_the_file_to_exist(evidence_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check(Args(evidence_root))


def test_repository_evidence_file_matches_artifacts() -> None:
    """The committed EVIDENCE.md must never drift from the recorded artifacts."""
    root = Path(__file__).resolve().parents[1]
    assert check(Args(root)) == 0
