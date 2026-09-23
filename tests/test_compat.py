import pytest
import torch

from jah.compat import require_minimum_label_mass
from jah.scoring import label_probability_mass


def test_low_label_mass_raises_with_example_context() -> None:
    logits = torch.zeros(8)
    logits[7] = 10.0
    mass = label_probability_mass(logits, (0, 1))

    with pytest.raises(RuntimeError, match=r"sr-001=.*below minimum|below minimum.*sr-001"):
        require_minimum_label_mass([("sr-001", mass)], 0.5)


def test_unmeasured_label_mass_cannot_pass_full_vocabulary_gate() -> None:
    with pytest.raises(ValueError, match="full-vocabulary label mass is not measured"):
        require_minimum_label_mass([("scalar-001", None)], 0.5)
