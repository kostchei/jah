import pytest
from pydantic import ValidationError

from jah.policy import AcceptancePolicy, evaluate_acceptance
from jah.schemas import EvaluateRequest
from jah.scoring import ScoredDecision, debias_null_prior, ensemble_cyclic_logits


def test_evaluate_request_order_debias_passes_validation() -> None:
    payload = {
        "state": "Sample customer email.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Select routing team.",
                "options": [
                    {"id": "billing", "description": "Billing inquiries"},
                    {"id": "technical", "description": "Technical errors"},
                ],
            }
        },
    }
    # Default is 1
    req1 = EvaluateRequest.model_validate(payload)
    assert req1.order_debias_passes == 1

    # Explicit 2 is allowed
    payload["order_debias_passes"] = 2
    req2 = EvaluateRequest.model_validate(payload)
    assert req2.order_debias_passes == 2

    # 3 is out of range [1, 2]
    payload["order_debias_passes"] = 3
    with pytest.raises(ValidationError):
        EvaluateRequest.model_validate(payload)


def test_ensemble_cyclic_logits() -> None:
    # Simulated decision rotation 0: model is biased towards option "billing" (which was in slot A)
    decision_r0 = ScoredDecision(
        selected_id="billing",
        probabilities={"billing": 0.80, "technical": 0.20},
        logits={"billing": 2.0, "technical": 0.6137},
    )

    # Simulated decision rotation 1: option "technical" was in slot A, so it got higher relative logit
    decision_r1 = ScoredDecision(
        selected_id="technical",
        probabilities={"billing": 0.40, "technical": 0.60},
        logits={"billing": 1.0, "technical": 1.4055},
    )

    ensembled, delta_order = ensemble_cyclic_logits(decision_r0, decision_r1)

    # Delta order = max(|0.80 - 0.40|, |0.20 - 0.60|) = 0.40
    assert pytest.approx(delta_order, abs=1e-4) == 0.40

    # Averaged logits:
    # billing: (2.0 + 1.0) / 2 = 1.5
    # technical: (0.6137 + 1.4055) / 2 = 1.0096
    assert "billing" in ensembled.probabilities
    assert "technical" in ensembled.probabilities
    assert pytest.approx(sum(ensembled.probabilities.values()), abs=1e-5) == 1.0
    # Option billing has higher average logit
    assert ensembled.selected_id == "billing"


def test_debias_null_prior() -> None:
    # Model has unigram prior preferring option A by 1.5 logit units
    raw_logits = {"opt_a": 3.0, "opt_b": 2.0}
    null_prior_logits = {"opt_a": 1.5, "opt_b": 0.2}

    debiased = debias_null_prior(raw_logits, null_prior_logits, beta=1.0)
    assert pytest.approx(debiased["opt_a"], abs=1e-5) == 1.5
    assert pytest.approx(debiased["opt_b"], abs=1e-5) == 1.8


def test_policy_enforces_max_order_discrepancy() -> None:
    policy = AcceptancePolicy(
        type="threshold",
        threshold=0.70,
        metric="max_probability",
        max_order_discrepancy=0.15,
    )
    # High confidence (0.85 >= 0.70), and low discrepancy (0.05 <= 0.15) -> accept
    disp1 = evaluate_acceptance(
        policy,
        {"billing": 0.85, "technical": 0.15},
        primitive="choice",
        order_discrepancy=0.05,
    )
    assert disp1 == "accept"

    # High confidence (0.85 >= 0.70), but HIGH discrepancy (0.25 > 0.15) -> forced to review!
    disp2 = evaluate_acceptance(
        policy,
        {"billing": 0.85, "technical": 0.15},
        primitive="choice",
        order_discrepancy=0.25,
    )
    assert disp2 == "review"
