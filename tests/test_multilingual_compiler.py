import unicodedata

from jah.compiler import (
    canonicalize_state,
    compile_request,
    normalize_nfc,
)
from jah.schemas import EvaluateRequest


class MockBoundaryTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        labels = {"A": 101, "B": 102, "C": 103, "D": 104}
        for label, token_id in labels.items():
            if text.endswith(label):
                return [ord(char) for char in text[: -len(label)]] + [token_id]
        return [ord(char) for char in text]


def test_nfc_normalization_vietnamese_and_thai() -> None:
    # Decomposed NFD Vietnamese string: 'ti' + 'e' + combining circumflex + combining acute
    vietnamese_nfd = "ti\u0065\u0302\u0301ng Vi\u0065\u0323t"
    vietnamese_nfc = normalize_nfc(vietnamese_nfd)
    assert unicodedata.is_normalized("NFC", vietnamese_nfc)
    assert len(vietnamese_nfc) < len(vietnamese_nfd)

    # Thai text with tone markers
    thai_text = "ขอยกเลิกคำสั่งซื้อ"
    assert normalize_nfc(thai_text) == thai_text


def test_canonicalize_state_normalizes_nfc() -> None:
    nfd_state = {"message": "ti\u0065\u0302\u0301ng Vi\u0065\u0323t", "lang": "vi"}
    canonical = canonicalize_state(nfd_state)
    assert unicodedata.is_normalized("NFC", canonical)


def test_compile_request_multilingual_rotation() -> None:
    request = EvaluateRequest.model_validate(
        {
            "state": "Tôi muốn hoàn lại tiền cho hóa đơn #1234 (Thai: ขอยกเลิกคำสั่งซื้อ)",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Chọn bộ phận xử lý yêu cầu này.",
                    "options": [
                        {"id": "billing", "description": "Thanh toán và hoàn tiền"},
                        {"id": "technical", "description": "Lỗi kỹ thuật"},
                        {"id": "other", "description": "Khác"},
                    ],
                }
            },
        }
    )
    # Rotation 0 (baseline)
    compiled_r0 = compile_request(request, tokenizer=MockBoundaryTokenizer(), rotation=0)[0]
    assert compiled_r0.option_ids == ("billing", "technical", "other")
    assert compiled_r0.labels == ("A", "B", "C")
    assert compiled_r0.label_token_ids == (101, 102, 103)

    # Rotation 1 (cyclic shift by 1)
    compiled_r1 = compile_request(request, tokenizer=MockBoundaryTokenizer(), rotation=1)[0]
    assert compiled_r1.option_ids == ("technical", "other", "billing")
    assert compiled_r1.labels == ("A", "B", "C")
    assert compiled_r1.label_token_ids == (101, 102, 103)

    # State in prompt is strictly NFC normalized
    assert unicodedata.is_normalized("NFC", compiled_r0.prompt)
