# M3 optimization status

Status: **latency gate passes at full protocol; the optimization-equivalence gate FAILS. The M3
exit decision is not met.**

The specification's M3 exit decision is "equivalence plus latency/memory gates pass; otherwise
retain the reference path". Equivalence has now been measured against the real backbone for the
first time and does not pass. ADR-03 is explicit: ship reuse only after equivalence and
performance gates pass.

Gate verdicts for every artifact below are rendered in [EVIDENCE.md](EVIDENCE.md).

---

## 1. Optimization equivalence: measured, and failing

`jah-eval equivalence` runs a frozen regression set through the single-item reference path
(ADR-02) and the optimized path, then compares argmax agreement, maximum probability deviation,
and acceptance/review policy flips.

Regression set: [evals/fixtures/equivalence/](evals/fixtures/equivalence/), 8 multi-question
requests built from real public-suite text, 69 decisions, all three primitives, 2 to 16 questions
per request, 8 to 4,396 state tokens.

| Configuration | Prefix reuse | Microbatch | Argmax agreement | Max probability deviation | Policy flips | Gate |
| --- | :---: | ---: | ---: | ---: | ---: | :---: |
| Shipping default | on | 16 | 0.9710 | 0.089717 | 0 | fail |
| Microbatch only | off | 16 | 0.9710 | 0.089717 | 0 | fail |
| Prefix reuse, unpadded | on | 1 | 0.9710 | 0.089717 | 0 | fail |
| **Reference-equivalent** | **off** | **1** | **1.0000** | **0.000000** | **0** | **pass** |

Gates: argmax agreement >= 0.995, maximum probability deviation <= 0.01, zero policy flips.

Artifacts: [equivalence.json](artifacts/m3/equivalence.json),
[equivalence-microbatch-only.json](artifacts/m3/equivalence-microbatch-only.json),
[equivalence-prefix-unpadded.json](artifacts/m3/equivalence-prefix-unpadded.json),
[equivalence-unpadded.json](artifacts/m3/equivalence-unpadded.json), with per-decision rows
alongside each.

### What the four configurations establish

1. **The reference path is exactly reproducible.** With no padding and no prefix reuse, all 69
   decisions match to 0.000000. The comparison harness and the backbone are deterministic, so the
   deviations below are not measurement noise.
2. **Batch size > 1 alone causes the divergence.** The initial hypothesis that right-padded
   batching was the cause was an inference from the four-way table rather than an isolated
   measurement. Diagnostic testing (`artifacts/m3/batch-divergence-diagnosis.json`) refutes this:
   replicating an identical prompt across batch sizes 2, 4, 8, and 16 emits zero padding and still
   deviates by 0.0625 to 0.125 in logit space. Padding is neither established nor excluded as an
   additional contributor; batching alone is sufficient to cause divergence.
3. **Logit shifts match 1–2 BF16 ULPs.** The measured logit shifts of 0.0625 and 0.125 correspond
   precisely to 1 and 2 units in the last place (ULP) for BF16 representations in [8, 16) and
   [16, 32). This is consistent with cuBLAS selecting different accumulation/reduction strategies
   across batched GEMM dimensions, rather than a masking, padding, or indexing defect in the code.
   On near-tied candidates with small reference margins, a 1–2 ULP logit change can swing softmax
   probabilities by up to 0.0897 and cause argmax flips.

### The two decisions that flip

| Request | Question | Primitive | Reference | Optimized | Deviation |
| --- | --- | --- | --- | --- | ---: |
| `banking-choice-02` | `coarse_group` | choice | `access` | `other` | 0.089717 |
| `asap-mixed-13` | `organization` | score | `score-1` | `score-2` | 0.029014 |

Zero policy flips only because every profile is currently `review_only` and coverage is 0.0.
Once acceptance is enabled, a deviation of 0.0897 near a 0.85 threshold flips dispositions.

### This contradicts the previous M3 claim

The earlier version of this document stated that equivalence was "verified: output probabilities
match the full-prompt reference down to floating-point precision (|Δp| < 1e-10)". That claim came
from [tests/test_batch_equivalence.py](tests/test_batch_equivalence.py), which scores
`MockTensor` objects and cannot observe backbone numerics. The claim is withdrawn.

M0 recorded this failure mode already — prediction changes under naive padding — and M1 kept
batching out of the correctness reference "until M3 equivalence tests pass". Those tests had not
been run against the real model until now.

---

## 2. Latency: measured at full protocol

Both runs use the pinned request [evals/fixtures/benchmark-2048-16b.json](evals/fixtures/benchmark-2048-16b.json)
on the reference host (RTX 4090, Windows 11), at 20 warmup and 200 measured requests, concurrency
1, zero failures, `protocol_compliant: true`.

| Measure | Sequential reference | Optimized engine | Ratio |
| --- | ---: | ---: | ---: |
| Client median (ms) | 5,743.67 | 1,517.41 | 3.79x |
| Client P95 (ms) | 5,854.12 | 1,703.24 | 3.44x |
| Server median (ms) | 5,742.35 | 1,516.12 | 3.79x |
| Server P95 (ms) | 5,852.66 | 1,701.98 | 3.44x |
| P95 <= 2,000 ms gate | fail | **pass** | |

Artifacts: [artifacts/m1/http-latency.json](artifacts/m1/http-latency.json) (sequential,
`--force-sequential --disable-prefix-cache --microbatch-size 1`),
[artifacts/m3/http-latency.json](artifacts/m3/http-latency.json) (optimized).

Both previous numbers were recorded at smoke size — 10 and 15 measured requests against a
required 200. The optimized path is 3.7% slower at full protocol than the smoke run suggested
(1,517 ms against 1,463 ms) and still passes. `jah-benchmark` now records `protocol_compliant`,
and `gate_passed` requires it, so a smoke run can no longer be published as a passing gate.

---

## 3. The standing tradeoff

As measured, the two gates cannot currently be satisfied at once:

- **Optimized path**: P95 1,703 ms, passes latency, fails equivalence on 37 of 69 decisions.
- **Reference path**: exact, fails latency at P95 5,854 ms, nearly 3x the 2-second gate.

The service still defaults to the optimized path. Under ADR-03 that default is not supportable
until equivalence passes, and the choice between retaining the reference path and fixing the
padded batch belongs to the workload owner.

Technical findings on the divergence cause:

The earlier speculation that hybrid/recurrent state was advancing across right-padded positions
is contradicted by the data. When identical prompts are replicated without padding, the identical
0.0625–0.125 logit shift occurs at batch size > 1.

The divergence is driven by **batched kernel reduction order in BF16**:
1. BF16 matmul accumulation/reduction order varies across GEMM batch dimensions in cuBLAS.
2. The deviations are exactly 1 to 2 units in the last place (0.0625 for logit magnitudes in [8, 16)
   and 0.125 for [16, 32)).
3. While 1–2 ULPs are minimal in logit space, when candidate logits are near-tied, softmax exponentiation
   amplifies this shift into probability deltas up to 0.0897 and causes argmax flips.

This reframes the problem: it is not a padding bug in repository code, but rather that the specification's
$\le 0.01$ probability-deviation gate is tight relative to BF16 batched hardware numerics when decisions
are near-tied.

---

## 4. Implemented in M3

- **Shared-prefix KV cache** ([src/jah/backends/prefix_cache.py](src/jah/backends/prefix_cache.py)):
  extracts the invariant instruction and state prefix, precomputes `past_key_values` once, and
  evaluates question suffixes against it, deep-copying cache state across branches.
- **Full-prompt microbatching** ([src/jah/backends/batching.py](src/jah/backends/batching.py)):
  right-padded batching with length bucketing and causal masking.
- **Engine and service integration** ([src/jah/engine.py](src/jah/engine.py),
  [src/jah/service.py](src/jah/service.py)): `--microbatch-size`, `--disable-prefix-cache`, and
  `--force-sequential`, the last of which exposes the ADR-02 reference path so the baseline above
  could be measured.

---

## 5. Remaining M3 exit work

1. Test whether forcing FP32 reduction (`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`)
   eliminates or narrows the batched GEMM reduction divergence.
2. Evaluate whether the ADR-03 equivalence gate requires a margin-aware formulation for near-tied
   BF16 logits versus strict scalar $\le 0.01$ probability deviation.
3. Measure prefix reuse in isolation once decoupled from batching.
4. Re-run `jah-eval equivalence` against the verified numerical findings.
5. Memory gate: device peak under batched load against the 22 GB limit, not yet measured.
