"""Rubric exemplar store and in-context retrieval per claude_gap.md Phase 1B.

Provides per-task adjudicated exemplar storage and retrieval to condition the decision
backbone on runtime rubrics without fine-tuning weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Exemplar:
    task_id: str
    state: str
    selected_id: str
    reasoning: str | None = None


class ExemplarStore:
    """In-memory and file-backed store for adjudicated task exemplars."""

    def __init__(self) -> None:
        self._exemplars_by_task: dict[str, list[Exemplar]] = {}

    def add(self, exemplar: Exemplar) -> None:
        self._exemplars_by_task.setdefault(exemplar.task_id, []).append(exemplar)

    def count(self, task_id: str | None = None) -> int:
        if task_id is None:
            return sum(len(exs) for exs in self._exemplars_by_task.values())
        return len(self._exemplars_by_task.get(task_id, []))

    def retrieve(
        self,
        task_id: str,
        query_state: str | dict[str, Any] | list[Any],
        k: int = 3,
    ) -> list[Exemplar]:
        """Retrieve k nearest/most relevant exemplars for a given task using token overlap."""
        pool = self._exemplars_by_task.get(task_id, [])
        if not pool:
            return []
        if len(pool) <= k:
            return list(pool)

        query_tokens = set(str(query_state).lower().split())

        def score(ex: Exemplar) -> float:
            ex_tokens = set(ex.state.lower().split())
            if not ex_tokens or not query_tokens:
                return 0.0
            overlap = len(query_tokens & ex_tokens)
            union = len(query_tokens | ex_tokens)
            return overlap / union if union > 0 else 0.0

        ranked = sorted(pool, key=score, reverse=True)
        return ranked[:k]

    def format_exemplars_block(self, exemplars: list[Exemplar]) -> str:
        """Format exemplars into the canonical prompt demonstration block."""
        if not exemplars:
            return ""
        blocks = []
        for idx, ex in enumerate(exemplars, start=1):
            line = f"Example {idx}:\nState: {ex.state.strip()}\nSelected: {ex.selected_id}"
            if ex.reasoning:
                line += f"\nReasoning: {ex.reasoning.strip()}"
            blocks.append(line)
        return "<EXEMPLARS>\n" + "\n\n".join(blocks) + "\n</EXEMPLARS>\n\n"

    def save(self, path: Path) -> None:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1,
            "exemplars": [
                asdict(ex)
                for exs in self._exemplars_by_task.values()
                for ex in exs
            ],
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ExemplarStore:
        path = Path(path).resolve()
        store = cls()
        if not path.exists():
            return store
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("exemplars", []):
            store.add(
                Exemplar(
                    task_id=item["task_id"],
                    state=item["state"],
                    selected_id=item["selected_id"],
                    reasoning=item.get("reasoning"),
                )
            )
        return store
