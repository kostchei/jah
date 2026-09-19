import pytest
import torch

from jah.compiler import compile_request
from jah.schemas import EvaluateRequest
from jah.scoring import last_unpadded_logits, score_option_logits


class BoundaryTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        labels = {"A": 101, "B": 102, "C": 103}
        for label, token_id in labels.items():
            if text.endswith(label):
                return [ord(char) for char in text[: -len(label)]] + [token_id]
        return [ord(char) for char in text]


def test_last_unpadded_logits_uses_each_rows_mask() -> None:
    logits = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    selected = last_unpadded_logits(logits, mask)
    torch.testing.assert_close(selected[0], logits[0, 1])
    torch.testing.assert_close(selected[1], logits[1, 2])


def test_scoring_normalizes_only_allowed_labels_in_fp32() -> None:
    logits = torch.tensor([0.0, 1.0, 3.0, 100.0], dtype=torch.bfloat16)
    decision = score_option_logits(
        logits,
        option_ids=("low", "high"),
        label_token_ids=(1, 2),
        option_values=(0.0, 1.0),
    )
    assert decision.selected_id == "high"
    assert sum(decision.probabilities.values()) == pytest.approx(1.0)
    assert decision.expected_value == pytest.approx(decision.probabilities["high"])


def test_boolean_probabilities_are_complements_end_to_end() -> None:
    request = EvaluateRequest.model_validate(
        {
            "state": "The service is available.",
            "questions": {
                "available": {
                    "type": "boolean",
                    "instructions": "Assess the proposition.",
                    "proposition": "The service is available.",
                }
            },
        }
    )
    compiled = compile_request(request, tokenizer=BoundaryTokenizer())[0]
    logits = torch.zeros(104)
    logits[101] = -0.5
    logits[102] = 1.25
    decision = score_option_logits(
        logits,
        option_ids=compiled.option_ids,
        label_token_ids=compiled.label_token_ids or (),
    )
    assert decision.probabilities["false"] + decision.probabilities["true"] == pytest.approx(
        1.0, abs=1e-6
    )


def test_score_expected_value_includes_negative_levels_end_to_end() -> None:
    request = EvaluateRequest.model_validate(
        {
            "state": "A mixed outcome.",
            "questions": {
                "severity": {
                    "type": "score",
                    "instructions": "Score severity.",
                    "levels": [
                        {"id": "low", "description": "Low", "value": -2.0},
                        {"id": "medium", "description": "Medium", "value": 1.0},
                        {"id": "high", "description": "High", "value": 4.0},
                    ],
                }
            },
        }
    )
    compiled = compile_request(request, tokenizer=BoundaryTokenizer())[0]
    logits = torch.zeros(104)
    logits[101:104] = torch.tensor([-1.0, 0.5, 2.0])
    decision = score_option_logits(
        logits,
        option_ids=compiled.option_ids,
        label_token_ids=compiled.label_token_ids or (),
        option_values=compiled.option_values,
    )
    probabilities = torch.softmax(torch.tensor([-1.0, 0.5, 2.0]), dim=0)
    expected = sum(
        probability.item() * value
        for probability, value in zip(probabilities, (-2.0, 1.0, 4.0), strict=True)
    )
    assert decision.expected_value == pytest.approx(expected)
