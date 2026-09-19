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
