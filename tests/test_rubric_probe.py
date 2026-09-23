"""Unit tests for Phase 1A rubric diagnosis probe harness."""

from __future__ import annotations

from pathlib import Path
import pytest

from jah.diagnostics.rubric_probe import (
    ProbeTask,
    build_probe_request,
    evaluate_probes,
    load_probe_tasks,
)
from jah.schemas import ChoiceAnswer, EvaluateResponse, Timing, Usage


class MockProbeEngine:
    def __init__(self, mode_answers: dict[str, dict[str, str]]) -> None:
        self.mode_answers = mode_answers

    def evaluate(self, request, request_id: str | None = None) -> EvaluateResponse:
        clean = (request_id or "probe_mode_task").removeprefix("probe_")
        mode, _, task_id = clean.rpartition("_")
        pred = self.mode_answers.get(mode, {}).get(task_id, "default")
        return EvaluateResponse(
            request_id=request_id or "mock",
            artifact_id="mock",
            profile_id=None,
            answers={
                "judgment": ChoiceAnswer(
                    type="choice",
                    value=pred,
                    probabilities={pred: 0.90, "other": 0.10},
                    calibration_status="uncalibrated",
                    disposition="review",
                )
            },
            usage=Usage(unique_state_tokens=10, processed_input_tokens=10, decisions=1),
            timing_ms=Timing(queue=0.0, inference=1.0, total=1.0),
        )


def test_load_probe_tasks() -> None:
    root = Path(__file__).resolve().parents[1]
    probes_path = root / "evals" / "fixtures" / "rubric_probes" / "probes.json"
    tasks = load_probe_tasks(probes_path)
    assert len(tasks) == 10
    categories = {t.category for t in tasks}
    assert categories == {"nominal", "ordinal", "free_form"}


def test_build_probe_request_modes() -> None:
    task = ProbeTask(
        task_id="t1",
        category="nominal",
        state="Customer issue text.",
        instructions="Route ticket.",
        candidates=[{"id": "a", "description": "Desc A"}, {"id": "b", "description": "Desc B"}],
        ground_truth="a",
        decision_tree="If money -> a, else -> b.",
        exemplars_8shot=[{"state": "Ex 1", "selected": "a"}],
    )
    req_zero = build_probe_request(task, "bare_zero_shot")
    assert req_zero.state == "Customer issue text."
    assert req_zero.questions["judgment"].instructions == "Route ticket."

    req_tree = build_probe_request(task, "decision_tree")
    assert "DECISION TREE CRITERIA" in req_tree.questions["judgment"].instructions

    req_few = build_probe_request(task, "in_context_8shot")
    assert "REFERENCE ADJUDICATED EXAMPLES" in req_few.state


def test_evaluate_probes_calculates_recovery() -> None:
    tasks = [
        ProbeTask(
            task_id="t1", category="nominal", state="s1", instructions="i1",
            candidates=[{"id": "a", "description": "d"}, {"id": "other", "description": "d2"}], ground_truth="a",
            decision_tree="dt", exemplars_8shot=[],
        ),
        ProbeTask(
            task_id="t2", category="ordinal", state="s2", instructions="i2",
            candidates=[{"id": "b", "description": "d"}, {"id": "other", "description": "d2"}], ground_truth="b",
            decision_tree="dt", exemplars_8shot=[],
        ),
    ]

    # In zero shot: only t1 correct (acc = 0.5)
    # In decision tree: both correct (acc = 1.0)
    # In 8-shot: both correct (acc = 1.0)
    mode_answers = {
        "bare_zero_shot": {"t1": "a", "t2": "wrong"},
        "decision_tree": {"t1": "a", "t2": "b"},
        "in_context_8shot": {"t1": "a", "t2": "b"},
    }
    engine = MockProbeEngine(mode_answers)
    summary = evaluate_probes(tasks, engine)

    assert summary["modes"]["bare_zero_shot"]["overall_accuracy"] == 0.5
    assert summary["modes"]["decision_tree"]["overall_accuracy"] == 1.0
    assert summary["gap_recovery"]["recovery_decision_tree"] == 0.5
    assert summary["gap_recovery"]["recovery_in_context_8shot"] == 0.5
