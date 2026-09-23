"""Rubric diagnosis harness per claude_gap.md Phase 1A.

Separates "cannot read the rubric" from "can read the rubric, cannot map it to an ordinal scale".
Scores a probe set of tasks spanning:
- Nominal labels
- Ordinal rubrics
- Free-form candidate descriptions

Across three distinct modes:
(a) bare zero-shot
(b) rubric restated as an explicit decision tree
(c) 8-shot in-context demonstration
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jah.compiler import LABELS, _apply_chat_template
from jah.schemas import ChoiceOption, ChoiceQuestion, EvaluateRequest


@dataclass(frozen=True)
class ProbeTask:
    task_id: str
    category: str  # "nominal" | "ordinal" | "free_form"
    state: str
    instructions: str
    candidates: list[dict[str, str]]
    ground_truth: str
    decision_tree: str
    exemplars_8shot: list[dict[str, str]]


def load_probe_tasks(path: Path) -> list[ProbeTask]:
    """Load probe tasks from JSON fixture."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        ProbeTask(
            task_id=item["task_id"],
            category=item["category"],
            state=item["state"],
            instructions=item["instructions"],
            candidates=item["candidates"],
            ground_truth=item["ground_truth"],
            decision_tree=item["decision_tree"],
            exemplars_8shot=item["exemplars_8shot"],
        )
        for item in data["tasks"]
    ]


def build_probe_request(task: ProbeTask, mode: str) -> EvaluateRequest:
    """Compile a probe task into an EvaluateRequest according to the mode."""
    options = [
        ChoiceOption(id=c["id"], description=c["description"])
        for c in task.candidates
    ]

    if mode == "bare_zero_shot":
        instructions = task.instructions
        state = task.state
    elif mode == "decision_tree":
        instructions = (
            f"{task.instructions}\n\n"
            f"DECISION TREE CRITERIA:\n{task.decision_tree}\n"
            f"Evaluate against the decision tree criteria step by step."
        )
        state = task.state
    elif mode == "in_context_8shot":
        exemplars_text = "\n\n".join(
            f"Example {idx + 1}:\nState: {ex['state']}\nSelected: {ex['selected']}"
            for idx, ex in enumerate(task.exemplars_8shot)
        )
        state = (
            f"REFERENCE ADJUDICATED EXAMPLES:\n{exemplars_text}\n\n"
            f"CURRENT TASK STATE:\n{task.state}"
        )
        instructions = task.instructions
    else:
        raise ValueError(f"unknown probe mode: {mode}")

    return EvaluateRequest(
        state=state,
        questions={
            "judgment": ChoiceQuestion(
                type="choice",
                instructions=instructions,
                options=options,
            )
        },
    )


def evaluate_probes(
    tasks: list[ProbeTask],
    engine: Any,
    *,
    modes: tuple[str, ...] = ("bare_zero_shot", "decision_tree", "in_context_8shot"),
) -> dict[str, Any]:
    """Run all probe tasks across specified modes and compute recovery metrics."""
    results_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}

    for mode in modes:
        for task in tasks:
            req = build_probe_request(task, mode)
            resp = engine.evaluate(req, request_id=f"probe_{mode}_{task.task_id}")
            answer = resp.answers["judgment"]
            selected_id = answer.value
            is_correct = selected_id == task.ground_truth

            results_by_mode[mode].append({
                "task_id": task.task_id,
                "category": task.category,
                "ground_truth": task.ground_truth,
                "selected": selected_id,
                "is_correct": is_correct,
                "probabilities": answer.probabilities,
            })

    # Summarize per mode and per category
    summary: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "task_count": len(tasks),
        "modes": {},
    }

    for mode, rows in results_by_mode.items():
        total_correct = sum(1 for r in rows if r["is_correct"])
        acc = total_correct / len(rows) if rows else 0.0

        categories = sorted(set(r["category"] for r in rows))
        cat_acc = {}
        for cat in categories:
            cat_rows = [r for r in rows if r["category"] == cat]
            cat_correct = sum(1 for r in cat_rows if r["is_correct"])
            cat_acc[cat] = cat_correct / len(cat_rows) if cat_rows else 0.0

        summary["modes"][mode] = {
            "overall_accuracy": acc,
            "category_accuracy": cat_acc,
            "results": rows,
        }

    # Recovery metrics relative to bare zero-shot
    if "bare_zero_shot" in summary["modes"]:
        base_acc = summary["modes"]["bare_zero_shot"]["overall_accuracy"]
        recovery = {}
        for mode in modes:
            if mode != "bare_zero_shot":
                acc = summary["modes"][mode]["overall_accuracy"]
                recovery[f"recovery_{mode}"] = acc - base_acc
        summary["gap_recovery"] = recovery

    return summary


def run_rubric_probes(args: argparse.Namespace) -> int:
    """CLI runner for rubric probes."""
    from jah.engine import DecisionEngine, EngineConfig
    from jah.evaluation import load_backend
    from jah.workload import load_yaml

    root = Path(args.root).resolve()
    probes_path = root / args.probes
    tasks = load_probe_tasks(probes_path)

    model_config = load_yaml(root / args.model_config)
    backend = load_backend("direct", model_config)

    engine = DecisionEngine(
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
            force_sequential=True,
        ),
    )

    report = evaluate_probes(tasks, engine)
    output = json.dumps(
        {k: v for k, v in report.items() if k != "modes"} | {
            "modes": {
                m: {k: v for k, v in data.items() if k != "results"}
                for m, data in report["modes"].items()
            }
        },
        indent=2,
    )
    print(output)

    if getattr(args, "output", None):
        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Probe report saved to {out_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path.cwd())
    parser.add_argument("--probes", default="evals/fixtures/rubric_probes/probes.json")
    parser.add_argument("--model-config", default="configs/models/qwen3.5-4b.yaml")
    parser.add_argument("--output", default="artifacts/diagnostics/rubric_probe_report.json")
    args = parser.parse_args(argv)
    return run_rubric_probes(args)


if __name__ == "__main__":
    raise SystemExit(main())
