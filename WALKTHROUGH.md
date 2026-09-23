# Walkthrough: public human benchmark calibration and LoRA head adaptation

This document describes what was run and what it measured. Gate verdicts are **not** written
here; they are rendered from the recorded artifacts into [EVIDENCE.md](EVIDENCE.md) by
`jah-report generate`. Where a number below is load-bearing, the artifact that holds it is named.

**Scope.** Every result on this page is measured on the **development** or **calibration**
partition of a public benchmark. The locked test partition has never been opened. No result here
certifies a workload for automatic acceptance, and `release_annotation_ready` is `false` in every
report.

---

## 1. Temperature calibration on the public benchmark

2,565 calibration-split decisions were scored and a scalar temperature fitted per workload by
golden-section search over negative log-likelihood.

Artifact: [artifacts/public/calibration-summary.json](artifacts/public/calibration-summary.json).
Profiles: [configs/profiles/public/](configs/profiles/public/).

| Workload | Primitive | Fitted T | 10-bin ECE | Brier | Prevalence Brier | NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `banking77-16-intent-v1` | choice (16-way) | 1.3522 | 0.047522 | 0.234752 | 0.929201 | 0.640459 |
| `wikiqa-answer-relevance-v1` | boolean | 0.7999 | 0.045374 | 0.299406 | 0.099623 | 0.458021 |
| `asap2-source-essay-v1` | score (6 ordinal levels) | 1.2162 | 0.094514 | 0.770769 | 0.748971 | 1.568374 |

Read the last two columns together. The specification requires a Brier score no worse than the
prevalence baseline:

- **banking77** beats its prevalence baseline by a wide margin.
- **WikiQA** is roughly three times worse than its prevalence baseline. The corpus is about 5%
  positive, so a constant negative predictor scores better. A single scalar temperature cannot
  correct a base-rate error; this is the subject of Phase 2.1 of
  [REMEDIATION_PLAN.md](REMEDIATION_PLAN.md).
- **ASAP** is worse than its prevalence baseline and misses the ECE gate. Accuracy across the six
  ordinal levels is roughly 35%.

One of three declared workloads currently passes both calibration gates.

---

## 2. LoRA head adaptation

Trainer: [`jah-train`](src/jah/training/train.py), with dynamic module injection, gradient
checkpointing, and candidate-restricted cross-entropy.

- Backbone `Qwen/Qwen3.5-4B`, BF16, RTX 4090.
- Target modules `q_proj`, `v_proj`, `lm_head`; 17 injected LoRA modules.
- Trainable parameters 2,924,544 of 4,542,190,080 (0.064%).
- 999 training samples, drawn from the **train** split only
  ([train.py](src/jah/training/train.py) filters on the split manifest before sampling).
- AdamW, lr 1e-4, batch size 4, accumulation 4, gradient clipping 1.0.
- 250 optimizer steps in 311.3 seconds.

Final training loss is 5e-05. A loss that low on 999 samples indicates the adapter has largely
memorized its training partition; it is a training-set observation, not a quality claim.

Bundle: [artifacts/adapters/qwen3.5-4b-public-head-v1/](artifacts/adapters/qwen3.5-4b-public-head-v1/),
with `adapter_weights.pt`, a manifest carrying the base revision and dataset hash, and
`training-report.json`.

---

## 3. Baseline against adapted head

Both runs cover the 297 held-out decisions of the `banking77-16-intent-v1` **development**
partition, under identical prompts and hardware.

Artifacts: [baseline-banking77-dev.json](artifacts/public/baseline-banking77-dev.json),
[adapted-banking77-dev.json](artifacts/public/adapted-banking77-dev.json).

| Measure | Frozen baseline | Adapted head | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 0.828283 | 0.915825 | +0.087542 |
| Balanced accuracy | 0.821928 | 0.914154 | +0.092226 |
| Macro-F1 | 0.812567 | 0.911191 | +0.098624 |
| Brier score | 0.264617 | 0.152478 | -0.112139 |
| NLL | 0.737278 | 0.444016 | -0.293262 |
| ECE | 0.054125 | 0.054195 | +0.000070 |
| Median request latency (ms) | 89.59 | 88.51 | -1.08 |
| P95 request latency (ms) | 118.13 | 114.48 | -3.65 |

Three qualifications belong with that table:

1. **ECE did not improve.** Both runs record `gate_ece_passed: false`. The adapter sharpened the
   distribution (Brier and NLL fell) without improving reliability.
2. **Coverage is zero in both runs.** No calibration profile was attached at evaluation time, so
   every decision routed to review and the selective-error gate was not exercised. Phase 2.3 of
   the remediation plan addresses this.
3. **No uncertainty interval.** The +0.0875 accuracy difference carries no bootstrap interval
   grouped by source group. Phase 3 addresses this.

---

## 4. What this evidence does not establish

- **Contamination is unknown.** banking77 is a widely published benchmark and is likely present in
  the backbone's pretraining corpus, as
  [artifacts/public/import-report.json](artifacts/public/import-report.json) states in its own
  limitations. Part of the 0.9158 may be recall rather than capability.
- **Upstream labels are not local adjudication.** The suite carries upstream human labels from
  banking77, WikiQA, and ASAP 2.0. No independent local adjudication was performed, so
  `release_annotation_ready` remains `false`.
- **The locked test is intact.** No quality claim here is a release claim.

---

## 5. Verification state

```powershell
.venv\Scripts\python.exe -m pytest          # default suite, mocked backends
.venv\Scripts\python.exe -m pytest -m gpu   # requires the pinned backbone on CUDA
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m jah.report check
```

The default suite runs in a few seconds because it scores mocked tensors. Properties that only
exist with real weights loaded — tokenizer label boundaries, last-token indexing, prefix-cache
agreement, adapter attachment — live in the `gpu` tier
([tests/test_gpu_backend.py](tests/test_gpu_backend.py)).

---

## 6. SEA Adaptation, Tokenization Resilience & Order Debiasing (RFC 002)

Delivered the Phase 2 enhancements covering Southeast Asian multilingual execution, tokenization stability, and order debiasing:
* **RFC 002 Specification:** [RFC_002_SEA_ADAPTATION_AND_ORDER_DEBIASING.md](RFC_002_SEA_ADAPTATION_AND_ORDER_DEBIASING.md)
* **NFC Unicode Normalization:** Prevents Vietnamese diacritic fragmentation and Thai character cluster drift in [`compiler.py`](src/jah/compiler.py).
* **Cyclic Order Debiasing:** Adds `order_debias_passes: int = Field(default=1, ge=1, le=2)` in [`schemas.py`](src/jah/schemas.py), ensembling rotations to eliminate order bias and enforcing `max_order_discrepancy` in [`policy.py`](src/jah/policy.py).
* **Dual-Mode Scalar Head:** Implemented [`HuggingFaceScalarHeadBackend`](src/jah/backends/scalar_head.py) (ADR-07) alongside direct-logits.
* **SEA-LION Configurations:** Added [`configs/models/sealion-qwen-8b.yaml`](configs/models/sealion-qwen-8b.yaml) and [`configs/models/qwen3.5-4b-sea.yaml`](configs/models/qwen3.5-4b-sea.yaml).
* **Automated Unit Tests:** [`tests/test_multilingual_compiler.py`](tests/test_multilingual_compiler.py) and [`tests/test_debiasing.py`](tests/test_debiasing.py) pass cleanly.

