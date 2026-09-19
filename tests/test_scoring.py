import pytest
import torch

from jah.scoring import last_unpadded_logits, score_option_logits


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
