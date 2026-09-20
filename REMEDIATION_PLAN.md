# Remediation plan

Status: active (Phase 0 complete; Phases 1–5 proposed) · Date: 2026-09-20 · Supersedes the verdict columns in `M2_STATUS.md`,
`M3_STATUS.md`, `M4_STATUS.md`, and `WALKTHROUGH.md`.

This plan closes the gap between what the repository's artifacts measured and what its
human-facing documents claimed, and then closes the measurement gaps themselves. It assumes
**no human annotation**. A local LM Studio model performs reasoned review.

## 0. Reviewer role and its limits

`JEV_AT_HOME_SPEC.md` ADR-06 states that teacher-model answers are not evaluation ground truth.
M1 demonstrated the failure mode directly: 141 rows labeled by `qwen/qwen3.8-27b` and then scored
by the backbone produced macro-F1 `1.000` and a Brier score of `3.15e-16` on 22 development
decisions. That is a model agreeing with itself, not a measurement.

Therefore:

- **Ground truth remains the upstream human labels** already imported from banking77, WikiQA, and
  ASAP 2.0 (`evals/data/public/suite.jsonl`, 33,126 decisions, `annotation_status:
  upstream-human`).
- **The reviewer model is used for three tasks only:**
  1. **Label audit.** On rows where the system disagrees with the upstream label, judge whether
     the upstream label is defensible. Output is a quarantine *recommendation*.
  2. **Failure-slice reasoning.** Cluster and name error modes so fixes can be targeted.
  3. **Hard-case generation.** Produce adversarial cases (missing information, conflicting
     evidence, paraphrase pairs) whose correctness is verifiable by construction.
- **The reviewer never writes a label into the evaluation suite.** Its output lives in a separate
  artifact and is applied only through an explicit exclusion flag.

**Recorded limitation.** The reviewer (`qwen/qwen3.8-27b`) and the backbone
(`Qwen/Qwen3.5-4B`) are the same model family, so their errors are correlated. Reviewer agreement
is therefore weak evidence. This is the ceiling of a no-human-annotation setup and must appear in
every artifact the reviewer touches.

**Consequence for release.** Without independent adjudication, `release_annotation_ready` stays
`false` permanently. The system may be honestly described as validated against public benchmarks.
It may not be described as validated for production acceptance on unseen local traffic.

---

## Phase 0 — Evidence integrity (Completed)

No new model capability. This phase made the human-facing documents match the recorded artifacts verbatim.

| # | Work | Status | Outcome / Reason |
| --- | --- | :---: | --- |
| 0.1 | `jah-report`: regenerate status tables **from** artifact JSON, rendering `gate_*_passed` booleans verbatim. Remove hand-written verdict columns. | **Done** | Implemented in `src/jah/report.py`; generated `EVIDENCE.md` tracking all 216 gates (125 pass). Status documents now link directly to verified artifacts. |
| 0.2 | Re-run `jah-benchmark` at the full protocol: 20 warmup, 200 measured, no `--smoke`. | **Done** | `artifacts/m3/http-latency.json` records 20 warmup, 200 measured requests, `protocol_compliant: true`. P95 is 1,703 ms (passes $\le 2.0$s gate). |
| 0.3 | `jah-eval equivalence`: real backbone, optimized path against sequential reference over frozen regression set. | **Done** | Emitted `artifacts/m3/equivalence.json`. Revealed that optimization equivalence fails on 37/69 decisions ($\max \Delta p = 0.0897$, 2 flips) due to batched GEMM reduction numerics. |
| 0.4 | `pytest -m gpu` tier covering label verification, last-token indexing, prefix-cache equivalence, and adapter attachment against the real model. | **Done** | Implemented in `tests/test_gpu_backend.py`. Fast suite stays mocked; live GPU path is guarded. |
| 0.5 | Demote `evals/data/m1-suite.jsonl` to a smoke fixture and retire the `T = 0.1` M2 profiles. | **Done** | `configs/evals/m1-suite.yaml` updated; degenerate M2 profiles moved to `configs/profiles/retired/`. |

**Exit criteria met.** Every number in status documents traces to a hashed artifact, and no gate verdict is written by hand.

---

## Phase 1 — LLM review harness (Completed)

Implemented `src/jah/review.py` and a `jah-review` CLI speaking the OpenAI-compatible LM Studio API at
`http://localhost:1234/v1`. Registered entrypoint `jah-review` in `pyproject.toml`.

- **Subcommands:** `audit`, `slice`, `generate`.
- **Contract:** reads a predictions JSONL plus suite rows; writes `artifacts/review/<run>.json`
  holding a per-row verdict, its reasoning, and a confidence value. Never mutates `suite.jsonl`.
  Quarantine is applied through an explicit `--exclude-reviewed-quarantine` evaluation flag in
  `jah-eval run` so every metric is reproducible with and without it.
- **Mandatory provenance block:** reviewer model ID, LM Studio build, prompt version, temperature,
  seed, `ground_truth: false`, and the correlated-family limitation string verbatim.
- **Schema:** added `llm-reviewed` to `annotation_status` in `src/jah/m1_dataset.py`. Rejected by
  `required_annotation_status: [upstream-human]` in `configs/evals/public-suite.yaml`.
- **Control arm, non-optional:** runs `audit` over an agreement sample of correct predictions.
  If the reviewer disputes $> 15\%$ of correct rows, the reviewer is flagged unreliable and
  all quarantine recommendations are discarded.
- **Tests:** `tests/test_review.py` passes (5/5 tests).

---

## Phase 2 — Real quality & calibration failures

### 2.1 WikiQA Brier worse than the prevalence baseline (Completed)

Measured: Brier `0.299406` against prevalence `0.099623`. WikiQA is roughly 5% positive.
- Implemented **vector scaling** `p = softmax(z / T + b)` in `src/jah/calibration.py`, fitting
  a per-label bias under NLL on the calibration partition.
- Fitted parameters: $T = 0.7897$, $b = [1.409963, -1.409963]$.
- Brier score dropped to `0.086645` (passing prevalence `0.099623`), ECE `0.016399`.
- Gate passed and published to `configs/profiles/public/wikiqa-answer-relevance-v1.yaml` and
  `artifacts/public/calibration-summary.json`.

### 2.2 ASAP ordinal assessment (Active)

Measured: roughly 35% accuracy across six levels, Brier `0.770769` against prevalence `0.748971`.

Three remediation attempts:
1. **Rubric & prompt conditioning.** Refine question instructions and scoring criteria structure.
2. **Distance-weighted ordinal cross-entropy.** (Done) Implemented distance-weighted ordinal cross-entropy
   loss in `src/jah/training/adapter.py`: $\mathcal{L} = \mathcal{L}_{\text{CE}} + \alpha \sum_i p_i \frac{(i - y)^2}{(K - 1)^2}$.
   Exposed `--loss-type {cross_entropy, ordinal}` in `jah-train`. Tested in `tests/test_ordinal_loss.py`.
3. **Score-specific LoRA head.** Train an adapter dedicated solely to ordinal essay scoring rather
   than sharing capacity in a multi-task head across intent classification and boolean relevance.

Report **normalized MAE and quadratic weighted kappa** via `jah-eval` on the development partition.

**Stop condition.** If normalized MAE remains above 0.15 after all three attempts, declare ASAP
out of scope for v0.1 with the evidence attached. Two of three workloads shipping is a legitimate
result. Three claimed workloads with one silently failing is not.

### 2.3 Selective decisions (Completed)

- Fit acceptance thresholds using `fit_acceptance_threshold()` on the **calibration** split.
- Swept confidence thresholds and computed exact Clopper-Pearson 95% upper bounds.
- `banking77-16-intent-v1`: threshold `0.8877`, coverage `0.5667` ($\ge 50\%$), accepted error 95% upper bound `0.0456` ($\le 5\%$).
- `wikiqa-answer-relevance-v1`: threshold `0.9404`, coverage `0.9501` ($\ge 50\%$), accepted error 95% upper bound `0.0457` ($\le 5\%$).
- Both profiles now satisfy `has_validated_evidence()` and allow autonomous acceptance.

Coverage is `0.0` in current public reports because profiles defaulted to `policy.type: review_only`,
and `has_validated_evidence()` requires validated selective performance on calibration data before
authorizing automatic acceptance.

- Fit acceptance thresholds using `fit_acceptance_threshold()` on the **calibration** split.
- Sweep confidence thresholds and publish complete risk-coverage curves rather than a single selected operating point.
- Compute the exact Clopper-Pearson bound at the $\ge 50\%$ coverage point. With $n = 297$ on the
  banking77 development partition the $n \ge 59$ sufficiency requirement is met.
- Wire `--use-workload-profiles --profiles-dir configs/profiles/public` into evaluation to verify
  accepted error bounds on held-out development data.

### 2.4 Optimization numerical equivalence (M3 remediation)

`artifacts/m3/equivalence.json` revealed that the optimized path diverges from the reference path on
37 of 69 regression decisions ($\max \Delta p = 0.0897$, 2 flips).

Diagnostic experiments (`artifacts/m3/batch-divergence-diagnosis.json`) isolated the cause:
- Replicating identical prompts across batch sizes 2, 4, 8, 16 produces zero padding and identical
  $0.0625$–$0.125$ logit shifts. Batch size $> 1$ alone is sufficient to cause divergence.
- The shifts are exactly 1–2 BF16 ULPs, resulting from non-associative GEMM accumulation / reduction
  variations in cuBLAS across batch shapes, not a code defect in padding or recurrent state.
- For near-tied decisions (small reference margin), a 1–2 ULP logit shift is amplified by softmax
  into $\Delta p$ up to $0.0897$.

Remediation steps:
- Probe whether forcing FP32 reduction (`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`)
  closes or narrows the batched GEMM reduction gap.
- Evaluate whether the ADR-03 equivalence gate ($\max \Delta p \le 0.01$) requires a margin-aware
  formulation for near-tied BF16 decisions or strict FP32 reference alignment.
- Measure prefix-cache reuse in isolation once fully decoupled from batching.

---

## Phase 3 — Unmeasured gates

- **Generative baseline on public data.** The >= 2x median latency and <= 2 percentage point
  quality-drop gates have only been exercised on 22 rows, at 1.25x
  (`artifacts/m1/paired-comparison.json`).
- **Capacity.** 1 request/second sustained for 10 minutes, >= 16 decisions/second, no OOM and no
  unbounded queue growth.
- **Memory.** Device peak under batched load against the <= 22 GB gate. M0 measured 9.32 GB on the
  single-item path; batched load is the risk.
- **Latency matrix.** 256/2,048/8,192 state tokens, 1/8/16/32 questions, 2/4/8/16 options.
- **Order and paraphrase robustness.** <= 5% label-aligned flips, using the paired inputs generated
  in Phase 1. M0 already identified a real wording bias in the catch-all option description.
- **Bootstrap 95% intervals** grouped by source group for every quality delta, including the
  +8.75% M4 lift, which currently carries no interval.

---

## Phase 4 — Locked test

Freeze prompts, model revision, adapter, calibration profiles, and thresholds. Open the locked
test partition **once**, run the full gate battery, and publish.

A reviewed test failure becomes development data only by replacing the affected locked rows for
the next release, per the specification. No quiet re-tuning and re-opening.

This is the only phase that yields a defensible quality claim. Everything before it is
development-split evidence.

---

## Phase 5 — Release candidate

`src/jah/release.py` implements the manifest and verification contracts; offline verification tests are in `tests/test_release.py`.

- Build a real immutable bundle and verify offline installation on a network-disabled run.
- Exercise rollback to the previous bundle.
- Produce a shadow-run report over recorded traffic, using the reviewer for deferred agreement
  labels and marking them reviewer-derived rather than ground truth.

---

## Sequencing and expectations

Phases run in order. Phase 0 has landed; Phase 1 and Phase 2 can proceed in parallel.

Two limits this plan does not remove:

- **Contamination is unknown.** banking77 is a widely published benchmark and is likely present in
  the backbone's pretraining data, as `artifacts/public/import-report.json` already states. The
  91.58% adapted accuracy may partly reflect recall. The contamination-free slice generated in
  Phase 1, not the banking77 figure, is the honest capability estimate.
- **No independent adjudication.** See section 0.
