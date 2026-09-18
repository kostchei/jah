# jev-at-home

Machine Learning Architecture & Implementation Specification — RFC 001 + Evaluation Spec

Status: proposed implementation baseline · Version: 0.1 · Date: 2026-09-18

## 1. Product and scope

Build a local decision service that accepts text or JSON plus runtime-defined questions and returns typed decisions and probability distributions. Initial applications are support routing, document relevance, and rubric-based assessment. Applications own subsequent actions; this service only evaluates supplied context.

The MVP reuses an open-weight language model and reads option logits directly, without an autoregressive output-generation loop. It retains the pretrained backbone and its causal computation. Independent questions can be batched; this does not mean arbitrary interdependent decisions are solved in one shared forward pass.

This is an independent implementation of the decision-service pattern. It does not reproduce TypeSafe's undisclosed architecture or RLCD training method. No dependency on TypeSafe's service is required.

**Success:** on a declared workload, deliver useful decisions with validated calibration and a measured latency advantage over generating equivalent structured answers locally.

**Out of scope for v0.1:** free-form text generation, repository discovery, tool execution, autonomous agents, multimodal input, arbitrary numeric regression, training a foundation model from scratch, and general factual verification without supplied evidence.

## 2. Planning assumptions and constraints

These are engineering defaults to validate during the first milestone, not measured capabilities.

| Item | v0.1 default |
| --- | --- |
| Host | Linux or WSL2; one NVIDIA GPU with 24 GB VRAM; 64 GB host RAM |
| Runtime | Python, PyTorch, Hugging Face Transformers; HTTP service with FastAPI/Pydantic |
| Backbone | Candidate: Qwen/Qwen3.5-4B; select and pin exact revision after the compatibility spike |
| Precision | BF16 reference; evaluate quantization separately |
| Input | Text or canonical JSON; maximum 8,192 tokens per fully compiled question |
| Questions | 1–32 independent questions/request |
| Choices | 2–16 options/question; stable caller-owned IDs |
| Deployment | Localhost by default; offline operation after artifact installation |
| Persistence | Model artifacts and aggregate metrics; raw inputs/results are not logged by default |

Model selection requires a compatible license, working direct-logit access, verified tokenizer behavior, and a measured memory budget. An open-weight model does not necessarily include its original training data or complete training pipeline. Record those distinctions in the release manifest.

## 3. Decision contract

| Primitive | Input | Output semantics |
| --- | --- | --- |
| Boolean | A proposition evaluated against supplied state | Probability assigned to `true`; complementary `false` probability |
| Choice | Mutually exclusive option descriptions | Distribution over exactly those options and the highest-scoring option ID |
| Score | Ordered, described levels with numeric values | Distribution over levels, selected level, and expected value |

Choice probabilities are conditional on the offered alternatives. The caller must include an `other` or `insufficient_information` option when needed. Boolean cannot represent a separate unknown category; use Choice when that distinction matters. Missing evidence must be defined in the question's instructions.

Score uses `expected_value = sum(p_i * value_i)`. This assumes the numeric distances are meaningful; otherwise consume only the selected ordinal level and distribution. Multi-label classification is represented as separate Boolean questions.

Raw softmax scores are not automatically calibrated probabilities of correctness. Return `calibration_status` explicitly. Do not invent a separate confidence field or interpret the maximum score as a universal reliability estimate. Automatic acceptance is enabled only for a validated, versioned workload policy.

## 4. Architecture decisions

```text
HTTP / Python client
        |
Schema validation + canonicalization + token budget check
        |
Question compiler: shared state prefix + independent question suffixes
        |
Bounded scheduler / microbatcher
        |
Pinned backbone -> selected option logits
        |
Calibration profile -> decision + acceptance policy -> typed response
```

**ADR-01 — direct option logits first.** Describe candidate answers in the prompt, assign each a verified single-token label, and evaluate the next-position logits after the answer prefix. Gather only allowed label logits and normalize them. Do not sample a token, append an answer, invoke `generate()`, or produce JSON with the model. Ordinary code serializes the response. Transformers exposes model logits for this purpose [1].

For logits `z_i`, compute `p_i = softmax(z_i / T)` in FP32; `T = 1` without a fitted calibration profile. Boolean uses two labels. Score uses one label per level. Use the actual last unpadded input position when gathering logits.

Label mappings must be verified against the exact tokenizer and answer boundary at startup. Reject unsupported cardinality instead of silently switching to multi-token labels. Measure label/order bias; tokenize option descriptions normally. The resulting values represent preference among supplied labels, not proof that one candidate is true.

**ADR-02 — full-prompt batching is the correctness reference.** Compile one prompt per question and batch independent questions using length buckets. GPU memory determines microbatch size. A batch of 32 questions still contains 32 evaluated sequences; report both request and decision counts. Start with ordinary attention/cache behavior supported by the chosen backbone.

**ADR-03 — shared-prefix reuse is an optional optimization.** Put invariant instructions and state before question-specific content. Reuse a prefix only when tokens, positions, attention layout, model revision, precision, and prompt version match exactly. Each branch must own its mutable cache state. Hybrid/recurrent backbones may require more than a standard KV cache; implement against the actual model contract.

Ship reuse only after equivalence and performance gates pass. Budget cache bytes, bound entry lifetime, and invalidate on model/prompt changes. Keep reuse within a request initially. OpenJev demonstrates this general approach but also reports numerical differences across execution paths [2].

**ADR-04 — calibration is a versioned artifact.** Fit temperature scaling on a dedicated calibration partition. Start with one scalar per supported workload/primitive where data permits. Evaluate binary isotonic calibration only with sufficient data and cross-validation. Calibration profiles include model, precision, prompt, label mapping, task family, and option-cardinality range. Unknown or mismatched profiles return `uncalibrated`, with acceptance disabled.

**ADR-05 — train only after measuring the frozen baseline.** If quality misses the gate, fine-tune a LoRA adapter using labeled choice distributions or hard labels and cross-entropy. Keep new-task and new-template holdouts. A later learned scoring head may read backbone representations conditioned on candidate descriptions. A fixed category head alone cannot support arbitrary runtime labels. RL is not required for the first trained version.

**ADR-06 — no hidden fallback.** Unsupported inputs and runtime failures are explicit. The caller can route to a human or another model. Optional teacher models may help create training candidates, but their answers are not evaluation ground truth.

## 5. API specification

`POST /v1/evaluate`; a Python client mirrors the request/response types. `GET /health/live` reports process availability; `GET /health/ready` becomes healthy only after artifact validation and warmup.

Example request:

```json
{
  "state": "I was charged twice. Please help today.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Select the responsible team.",
      "options": [
        {"id": "billing", "description": "Charges and payments"},
        {"id": "technical", "description": "Software failures"},
        {"id": "other", "description": "Neither applies or evidence is insufficient"}
      ]
    }
  },
  "profile": "support-routing-v1"
}
```

Illustrative response; values below are not measured predictions:

```json
{
  "request_id": "req_example",
  "answers": {
    "route": {
      "type": "choice",
      "value": "billing",
      "probabilities": {"billing": 0.92, "technical": 0.03, "other": 0.05},
      "calibration_status": "validated",
      "disposition": "review"
    }
  },
  "artifact_id": "jah-reference-v1",
  "profile_id": "support-routing-v1",
  "usage": {"unique_state_tokens": 12, "processed_input_tokens": 90, "decisions": 1},
  "timing_ms": {"queue": 2, "inference": 80, "total": 91}
}
```

`profile` resolves a server-side calibration/policy bundle. Omission produces uncalibrated output and `review`. A validated profile still returns `review` when its acceptance threshold is unmet. Scope matching beyond schema checks is partly the caller's responsibility; unfamiliar content is not reliably detected automatically.

Boolean adds `p_true` and a threshold-0.5 label; workflow acceptance uses the policy's own thresholds. Score adds `level_id`, `expected_value`, and level probabilities. All output keys map to request IDs; no generated explanations.

Reject duplicate IDs, invalid levels, non-finite numbers, empty questions, and token overflow with HTTP 422. Return 429 for queue saturation, 503 for unavailable inference, and 504 for deadline expiry. Requests are atomic in v0.1: return all answers or an error. Never truncate silently. Schemas prohibit additional fields. Request cancellation must release queued work and caches when safe.

## 6. Implementation and artifact layout

```text
src/jah/
  schemas.py       # primitive and HTTP contracts
  compiler.py      # canonical state, prompt, label mappings
  backends/        # direct-logit reference; optional prefix reuse
  scheduler.py     # bounded queue, microbatching, deadlines
  calibration.py  # fitting and profile compatibility checks
  policy.py        # acceptance / review rules
  service.py       # HTTP endpoints and metrics
evals/             # dataset manifest, runners, paired comparisons
training/          # optional adapter training and export
configs/           # versioned model and workload settings
tests/             # contracts, scoring, caching, failure behavior
```

Release one immutable artifact manifest containing weight/tokenizer revisions and hashes, license, prompt and label versions, dependency lockfile, precision, supported limits, adapter/calibrator hashes, policy, and evaluation report ID. Keep the last passing bundle for rollback. Pin dependencies only after the initial compatibility run; do not use floating `latest` artifacts in releases.

Record request/decision counts, queue/inference/total latency, processed tokens, batch size, cache bytes/hits, VRAM peak, failure rates, and review rate. GPU timings must synchronize appropriately. Keep model loading and cold-start timings separate from warm inference. No input content in metrics labels. Treat state as untrusted data in the compiler, and test instruction injection behavior rather than assuming delimiters eliminate it.

## 7. Evaluation specification

### Dataset and splits

Create an initial manually reviewed set of at least 1,000 decisions spanning routing, binary relevance, and ordinal assessment. Expand it for release claims; 1,000 total examples may be insufficient to certify low error rates across many slices. Include representative traffic, hard negatives, missing information, conflicting evidence, paraphrases, and varying option counts. Keep challenge cases separate from prevalence-representative metrics.

Each row contains `example_id`, `source_group_id`, state, question/options, reference answer or label distribution, rubric, task/template IDs, provenance, and annotation status. Two annotators label independently; adjudicate disagreements. Report agreement and unresolved ambiguity. Use categorical unknown labels where the contract permits them; do not force unknowable examples into confident binary ground truth.

Split by source/customer/document and near-duplicate cluster before augmentation: 60% training, 15% development, 10% calibration, 15% locked test. For a frozen model, reserve training data for possible later adapters. Tune prompts/model selection on development only; fit calibration and thresholds on calibration only. Reserve an additional task/template holdout to test unfamiliar criteria. A reviewed test failure becomes development data only after replacing the affected locked test for the next release.

### Baselines and comparisons

Run the same cases through (a) a trivial majority/rules baseline, (b) a local generative model producing compact constrained answers, (c) the same backbone with direct-logit scoring, and (d) optimized scoring and any trained adapter. For classification workloads, optionally add GLiClass, which supports zero-shot classification with supplied labels [3].

Compare generation versus direct scoring with identical state/questions and equivalent answer content. Do not inflate generation cost with unnecessary prose or confidence JSON. Compare probability quality only where the baseline produces meaningful probability estimates. Report both per-question and all-questions completion latency. Any hosted Jev comparison requires actual comparable runs; published vendor scores are context, not local benchmark results.

### Metrics and provisional release gates

All numerical gates below are proposed product targets. Freeze them with the workload manifest before opening the locked test. Do not claim they have been achieved.

| Dimension | Measure | Initial gate |
| --- | --- | --- |
| Decision quality | Macro-F1 and balanced accuracy for Choice/Boolean; MAE and weighted kappa for Score | Classification macro-F1 >= 0.85 per approved workload; normalized Score MAE <= 0.15; paired quality drop vs. generative baseline <= 2 percentage points |
| Calibration | Brier score, NLL, 10-bin equal-count ECE, reliability plots | ECE <= 0.05 on approved workloads; Brier no worse than raw scorer and prevalence baseline |
| Selective decisions | Error among accepted cases and fraction accepted | One-sided 95% upper bound on accepted error <= 5%, at coverage >= 50% per approved workload |
| Contract | Keys, bounds, sums, types, errors | 100% pass for finite validated results; distributions sum to 1 within 1e-5 |
| Order/paraphrase robustness | Label-aligned prediction changes on meaning-preserving paired inputs | <= 5% flips on adjudicated unambiguous pairs; report probability shifts separately |
| Optimization equivalence | Compare optimized path to reference | >= 99.5% argmax agreement, max probability deviation <= 0.01, and no accept/review policy flips on regression set |
| Latency | Warm local HTTP completion, 2,048-token state, 16 Boolean questions, concurrency 1 | P95 <= 2 seconds and >= 2x faster median completion than equivalent generative baseline |
| Capacity | Same workload, sustained offered rate 1 request/second for 10 minutes | >= 16 decisions/second served, P95 <= 3 seconds, no OOM or unbounded queue growth |
| Memory | Measured process/device peak on reference host | <= 22 GB VRAM; retain operating headroom |

Report bootstrap 95% intervals grouped by source for quality differences. Use an exact binomial bound for accepted-case error when observations are independent; use cluster-aware analysis when multiple decisions share a source. Insufficient samples mean unvalidated, not pass. Publish accuracy/coverage curves rather than selecting a favorable threshold after test inspection. Reliability must be assessed per relevant domain/cardinality slice; pooled calibration can conceal failures.

Benchmark lengths 256/2,048/8,192 tokens, 1/8/16/32 questions, and 2/4/8/16 options where compiled limits permit. Include concurrency 1/4/8, full prompts versus prefix reuse, cold versus warm execution, and BF16 versus any candidate quantization. Use 20 warmup requests and at least 200 measured requests for each latency case; record hardware, versions, power settings, and failures. Report saturated throughput separately from latency at the target offered load.

### Required tests and evidence

Unit/contract tests cover single-token label validation, last-token indexing, Boolean complement, ordinal expectation, canonical JSON, malformed inputs, and calibrator mismatch. Metamorphic tests permute labels/options, paraphrase instructions, add irrelevant text, and batch questions differently. Integration tests cover cache branch isolation, deadlines, queue overload, cancellation, unavailable GPU, and complete offline startup from installed artifacts.

Each candidate exports dataset/split hashes, configuration, row-level predictions, timings, metrics, uncertainty intervals, reliability/risk-coverage plots, and failure slices. Review these together before promoting the artifact. Changing weights, tokenizer, prompts, precision, cache path, or calibration invalidates prior approval until the relevant gates are rerun.

## 8. Delivery milestones and decisions

| Milestone | Deliverable | Exit decision |
| --- | --- | --- |
| M0: compatibility + data contract | Verified GPU/model/tokenizer setup; 50 adjudicated examples; frozen workload definitions | Backbone fits and exposes usable logits; initial task feasibility established |
| M1: reproducible baseline | Direct scorer, compact generative baseline, API, initial 1,000-example suite | Correct contracts and reproducible quality/latency report |
| M2: calibrated local MVP | Calibration/policy profiles, error handling, immutable artifact bundle | Approved workloads meet quality, calibration, and selective-error gates |
| M3: optimize | Batching, optional prefix reuse, optional quantization | Equivalence plus latency/memory gates pass; otherwise retain reference path |
| M4: adapt if justified | LoRA or conditioned scoring-head experiment | Beats the frozen baseline on held-out tasks at acceptable inference cost |
| M5: release candidate | Offline install, rollback, documentation, shadow-run report | Operational gates pass and workload owner accepts measured tradeoffs |

Planning envelope: roughly 4–6 engineer-weeks for M0–M3 with one experienced ML engineer and ready labeled data. This is a provisional estimate; annotation, model compatibility, and GPU access can dominate elapsed time. M4 is separately scoped after observing the actual failure modes. Set experiment compute budgets and stop conditions at M0.

During shadow evaluation, the application records proposed decisions without applying them. Label a representative sample, compare with offline results, and enable acceptance only for profiles whose evidence passes. New domains start in review mode. If model quality requires generated reasoning, the caller may use a separate reasoning model; record that as a distinct workflow with its own latency and cost.

## 9. Open decisions and source notes

Before implementation begins, confirm actual GPU availability, intended first workload, annotation owner, expected traffic, and error costs. The defaults in this RFC permit prototyping; workload-specific acceptance thresholds require real labels and a named product owner.

The central engineering uncertainty is whether a frozen model's immediate decision scores retain enough task quality. Large-model reasoning quality does not automatically transfer to a single readout. Evaluate that before investing in custom kernels, broader model training, or a native non-autoregressive architecture.

Sources checked 2026-09-18:

1. [Transformers model outputs](https://huggingface.co/docs/transformers/main_classes/output) — logits and model output interfaces.
2. [OpenJev](https://github.com/TheoLeeCJ/openjev) — community direct-logit baseline, shared-state experiments, and documented limitations. It is evidence of feasibility, not a performance guarantee for this project.
3. [GLiClass](https://github.com/knowledgator/GLiClass) — optional classification baseline.
4. [TypeSafe founder's introduction](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — product inspiration and vendor-described approach; not a reproducible training specification.

Implementation requirements and numerical targets in this document are proposed design choices unless explicitly attributed to a source.
