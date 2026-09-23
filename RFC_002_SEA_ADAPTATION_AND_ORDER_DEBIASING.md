# RFC 002: Southeast Asian (SEA) Adaptation, Tokenization Resilience & Order Debiasing

**Status:** Code paths implemented; task-level validation pending · **Base Specification:** `JEV_AT_HOME_SPEC.md` (RFC 001)
**Date:** 2026-09-23  

---

## 1. Overview & Objectives

This document records the architectural enhancements integrated into `jev-at-home` (`jah`) based on learnings from **Laya** (`NandhaKishorM/laya`) and **Open-Jev** (`Zefan-Cai/Open-Jev`), with specific focus on **Southeast Asian (SEA) language adaptation (Thai, Vietnamese, Indonesian, Malay)**, **tokenization resilience**, and **order bias / unigram prior debiasing**.

### Core Enhancements Delivered:
1. **Option A (Dual-Mode Execution):**
   - Preserves `direct_logit` gathering on base causal LMs for zero-shot workloads without data cold-starts.
   - Adds an experimental [`HuggingFaceScalarHeadBackend`](src/jah/backends/scalar_head.py) projecting candidate-specific last-token representations through a 1-D linear head. It avoids selecting class logits from vocabulary label tokens, but its zero-shot contrast initialization is not task-trained and has not been validated for decision quality.
2. **Tokenization Resilience for Southeast Asia:**
   - **Unicode Normalization (NFC):** Canonicalizes state contexts, question instructions, and candidate descriptions to NFC in [`compiler.py`](src/jah/compiler.py). This handles canonically equivalent encodings; it does not demonstrate SEA language competence or guarantee tokenizer resilience.
   - **Isolated Answer Boundaries:** Ensures clean delimiter separation (`"\nANSWER: "`) across chat-templated models.
   - **Candidate Option Rotation:** Adds deterministic cyclic rotation (`rotation: int`) in `compile_request` to support cyclic debiasing without re-tokenizing state.
3. **Inference-Time Order Debiasing Flag:**
   - Adds `order_debias_passes: int = Field(default=1, ge=1, le=2)` in [`schemas.py`](src/jah/schemas.py) and [`engine.py`](src/jah/engine.py).
   - `order_debias_passes = 1` (default): Ultra-low latency single forward pass.
   - `order_debias_passes = 2`: Runs rotation 0 and cyclic rotation 1 ($M=2$), averages aligned unnormalized logits in [`scoring.py`](src/jah/scoring.py) as a limited two-order ensemble, and calculates order discrepancy. This may reduce order sensitivity; it does not guarantee that positional bias is eliminated.
     $$\Delta_{\text{order}} = \max_k |p_{\text{r0}}(k) - p_{\text{r1}}(k)|$$
4. **Order Discrepancy Policy Gating:**
   - Added `max_order_discrepancy` to [`AcceptancePolicy`](src/jah/policy.py). If $\Delta_{\text{order}} > \tau_{\text{order}}$, the decision is automatically demoted to `disposition: review` to prevent overconfident mistakes caused by option permutation.
5. **Context-Free Prior Subtraction:**
   - Implemented `debias_null_prior(logits, prior_logits, beta=1.0)` as an isolated helper in [`scoring.py`](src/jah/scoring.py). It has no engine call site and is not currently part of inference behavior; any future integration requires task-level evaluation.
6. **SEA-LION Model Configurations:**
   - Added [`configs/models/sealion-qwen-8b.yaml`](configs/models/sealion-qwen-8b.yaml) targeting `aisingapore/Qwen-SEA-LION-v4-8B-IT` with native Southeast Asian vocabulary.
   - Added [`configs/models/qwen3.5-4b-sea.yaml`](configs/models/qwen3.5-4b-sea.yaml) with NFC tokenizer settings.

---

## 2. API & Schema Contract Changes

### `EvaluateRequest`
```json
{
  "state": "Tôi muốn hoàn lại tiền cho hóa đơn #1234 (Thai: ขอยกเลิกคำสั่งซื้อ)",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Chọn bộ phận phụ trách.",
      "options": [
        {"id": "billing", "description": "Thanh toán và hoàn tiền"},
        {"id": "technical", "description": "Lỗi kỹ thuật"},
        {"id": "other", "description": "Khác"}
      ]
    }
  },
  "order_debias_passes": 2
}
```

### `EvaluateResponse`
```json
{
  "request_id": "req_example",
  "answers": {
    "route": {
      "type": "choice",
      "value": "billing",
      "probabilities": {"billing": 0.88, "technical": 0.08, "other": 0.04},
      "calibration_status": "uncalibrated",
      "disposition": "review",
      "order_discrepancy": 0.04
    }
  },
  "usage": {
    "unique_state_tokens": 28,
    "processed_input_tokens": 124,
    "decisions": 1
  }
}
```

---

## 3. Verification & Test Suite

All changes are covered by automated unit tests running against Python 3.12 without external network dependencies:
* [`tests/test_multilingual_compiler.py`](tests/test_multilingual_compiler.py): Asserts Vietnamese NFD $\to$ NFC normalization, Thai unspaced script stability, and cyclic question rotation.
* [`tests/test_debiasing.py`](tests/test_debiasing.py): Asserts `order_debias_passes` schema bounds, cyclic logit averaging, null prior subtraction, and policy gating on $\Delta_{\text{order}}$.
* [`tests/test_engine.py`](tests/test_engine.py): Tests end-to-end `DecisionEngine` execution with $M=2$ debiasing passes and usage token accounting.
