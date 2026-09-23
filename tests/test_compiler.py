from jah.compiler import (
    PROMPT_VERSION,
    canonicalize_state,
    compile_request,
    verify_single_token_labels,
)
from jah.schemas import EvaluateRequest


class BoundaryTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        labels = {"A": 101, "B": 102, "C": 103}
        for label, token_id in labels.items():
            if text.endswith(label):
                return [ord(char) for char in text[: -len(label)]] + [token_id]
        return [ord(char) for char in text]


def test_canonical_json_is_stable() -> None:
    assert canonicalize_state({"b": 2, "a": [True, None]}) == '{"a":[true,null],"b":2}'


def test_choice_compiles_stable_single_token_labels() -> None:
    request = EvaluateRequest.model_validate(
        {
            "state": "A customer message.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose a route.",
                    "options": [
                        {"id": "one", "description": "First"},
                        {"id": "two", "description": "Second"},
                    ],
                }
            },
        }
    )
    compiled = compile_request(request, tokenizer=BoundaryTokenizer())[0]
    assert compiled.option_ids == ("one", "two")
    assert compiled.label_token_ids == (101, 102)
    assert compiled.prompt.endswith("ANSWER:")
    assert PROMPT_VERSION == "decision-prompt-v2"
    assert "PROMPT_VERSION: decision-prompt-v2" in compiled.prompt
    assert compiled.labels == ("A", "B")


def test_unstable_boundary_is_rejected() -> None:
    class BadTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [len(text)]

    try:
        verify_single_token_labels(BadTokenizer(), "prompt", ("A",))
    except ValueError as exc:
        assert "not one stable token" in str(exc)
    else:
        raise AssertionError("expected unstable label boundary to fail")


def test_canonical_rubric_with_exemplars_and_tie_break() -> None:
    from jah.exemplars import Exemplar, ExemplarStore

    request = EvaluateRequest.model_validate(
        {
            "state": "Customer asks for refund after 45 days.",
            "questions": {
                "policy_check": {
                    "type": "choice",
                    "instructions": "Determine if refund is eligible under policy.",
                    "options": [
                        {"id": "eligible", "description": "Within 30-day window or exceptional approval."},
                        {"id": "ineligible", "description": "Past 30-day window without exception."},
                    ],
                }
            },
        }
    )
    exemplars = [
        Exemplar(
            task_id="policy_check",
            state="Customer asks for refund after 10 days.",
            selected_id="eligible",
            reasoning="10 days is within the 30-day return policy window.",
        ),
        Exemplar(
            task_id="policy_check",
            state="Customer asks for refund after 60 days.",
            selected_id="ineligible",
            reasoning="60 days exceeds 30-day window with no documented exception.",
        ),
    ]
    tie_break = "If evidence is ambiguous, default to ineligible."

    compiled = compile_request(
        request,
        tokenizer=BoundaryTokenizer(),
        exemplars=exemplars,
        tie_break_rule=tie_break,
    )[0]

    assert "<EXEMPLARS>" in compiled.prompt
    assert "Customer asks for refund after 10 days." in compiled.prompt
    assert "Selected: eligible" in compiled.prompt
    assert "Reasoning: 10 days is within the 30-day return policy window." in compiled.prompt
    assert "</EXEMPLARS>" in compiled.prompt
    assert "TIE_BREAK_RULE: If evidence is ambiguous, default to ineligible." in compiled.prompt

    # Verify canonical ordering: PROMPT_VERSION -> <EXEMPLARS> -> <STATE> -> QUESTION -> CANDIDATES -> TIE_BREAK_RULE -> ANSWER:
    idx_version = compiled.prompt.index("PROMPT_VERSION:")
    idx_exemplars = compiled.prompt.index("<EXEMPLARS>")
    idx_state = compiled.prompt.index("<STATE>")
    idx_candidates = compiled.prompt.index("CANDIDATES:")
    idx_tie_break = compiled.prompt.index("TIE_BREAK_RULE:")
    idx_answer = compiled.prompt.index("ANSWER:")
    assert idx_version < idx_exemplars < idx_state < idx_candidates < idx_tie_break < idx_answer


def test_canonical_rubric_with_exemplar_store() -> None:
    from jah.exemplars import Exemplar, ExemplarStore

    store = ExemplarStore()
    store.add(
        Exemplar(
            task_id="route",
            state="User asks how to transfer funds to external checking account.",
            selected_id="transfer",
        )
    )
    store.add(
        Exemplar(
            task_id="route",
            state="User reports lost credit card while traveling.",
            selected_id="card_lost",
        )
    )

    request = EvaluateRequest.model_validate(
        {
            "state": "I need to move funds from savings to checking account.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Route request to department.",
                    "options": [
                        {"id": "transfer", "description": "Money transfer queries."},
                        {"id": "card_lost", "description": "Lost or stolen cards."},
                    ],
                }
            },
        }
    )

    compiled = compile_request(request, tokenizer=BoundaryTokenizer(), exemplars=store)[0]
    assert "<EXEMPLARS>" in compiled.prompt
    assert "transfer funds to external checking account" in compiled.prompt
    assert "TIE_BREAK_RULE" not in compiled.prompt

