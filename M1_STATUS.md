# M1 reproducible-baseline status

Status: **contract, scheduler, and paired evaluation tooling implemented; the M1 data exit
criterion is met by the public human benchmark, not by the model-adjudicated suite described
below.**

The 141-row model-adjudicated suite has been retired as an evaluation set (see
[M2_STATUS.md](M2_STATUS.md) section 2). Real evaluation uses `evals/data/public/suite.jsonl`,
33,126 upstream-human-labeled decisions. Gate verdicts are rendered from artifacts into
[EVIDENCE.md](EVIDENCE.md).

## Implemented

- `POST /v1/evaluate`, `GET /health/live`, and `GET /health/ready` with strict request/response
  schemas. Boolean, Choice, and Score responses are typed and always return `uncalibrated` plus
  `review` until M2 introduces a compatible validated profile.
- Atomic sequential direct-logit evaluation. M0 showed prediction changes under naive padding,
  so batching remains outside the M1 correctness reference until M3 equivalence tests pass.
- Bounded admission control, queue saturation (429), deadline expiry (504), unavailable inference
  (503), and schema/token-budget errors (422). A timed-out blocking inference retains its scheduler
  slot until the worker thread actually exits.
- A typed synchronous Python client.
- A same-backbone generative baseline that emits exactly one token constrained to the verified
  answer labels. It does not generate prose or JSON.
- A generic M1 dataset contract for runtime Boolean, Choice, and Score questions supporting both
  two-annotator human adjudication and `model-adjudicated` workflows for rapid product iteration.
- Deterministic connected-component splitting: any rows sharing a source group or near-duplicate
  cluster stay together. The ordinary split is 60% train, 15% development, 10% calibration, and
  15% locked test; task/template holdout rows are separate.
- Reproducible direct and generative reports with dataset/model/split/prediction hashes, row-level
  outputs, classification macro-F1 and balanced accuracy, Score normalized MAE and quadratic
  weighted kappa, and request latency. A paired comparator checks the provisional quality and
  latency gates.
- Benchmark request fixture `evals/fixtures/benchmark-2048-16b.json` tokenized to exactly 2,048 state
  tokens by the pinned tokenizer with 16 Boolean questions.

## Data status

`evals/data/m1-suite.jsonl` holds 141 decisions generated and labeled by `qwen/qwen3.8-27b` via
LM Studio. ADR-06 excludes teacher-model answers from evaluation ground truth, and the measured
consequence is visible in `artifacts/m1/paired-comparison.json`: macro-F1 1.000 and a Brier score
of 3.15e-16 on 22 development decisions. That is a model family agreeing with itself.

The file is retained only as a fast smoke fixture covering all three primitives.
[configs/evals/m1-suite.yaml](configs/evals/m1-suite.yaml) records `purpose: smoke-fixture-only`
and `evaluation_approved: false`.

The evaluation suite is `evals/data/public/suite.jsonl`: 33,126 decisions carrying upstream human
labels from banking77, WikiQA, and ASAP 2.0, imported with pinned source revisions and SHA-256
checksums ([configs/evals/public-sources.json](configs/evals/public-sources.json)). Upstream human
labels are not local independent adjudication, so `release_annotation_ready` stays `false`.

All `jah-eval` defaults now resolve to the public suite and the `jah-public-v1` split seed.

## Reproducible run sequence

```powershell
# 1. Validate data and freeze split assignments.
.venv\Scripts\jah-eval.exe validate `
  --dataset evals/data/m1-suite.jsonl `
  --output artifacts/m1/suite-validation.json

# 2. Run each path in a separate process so model memory is released between runs.
.venv\Scripts\jah-eval.exe run --backend direct --split development `
  --output artifacts/m1/direct-development.json `
  --predictions artifacts/m1/direct-development.jsonl

.venv\Scripts\jah-eval.exe run --backend generative --split development `
  --output artifacts/m1/generative-development.json `
  --predictions artifacts/m1/generative-development.jsonl

# 3. Pair only reports with identical dataset and split hashes.
.venv\Scripts\jah-eval.exe compare `
  --direct-report artifacts/m1/direct-development.json `
  --generative-report artifacts/m1/generative-development.json `
  --direct-predictions artifacts/m1/direct-development.jsonl `
  --generative-predictions artifacts/m1/generative-development.jsonl `
  --output artifacts/m1/paired-comparison.json
```

The specification's latency claim additionally requires 20 warmup and at least 200 measured HTTP
requests at the frozen benchmark shape. `jah-benchmark` enforces those minimums, 2,048 observed
state tokens, 16 Boolean decisions, concurrency one, and zero failed measured requests. It needs a
checked-in request fixture tokenized to exactly 2,048 state tokens by the pinned tokenizer.

## Remaining M1 exit work

1. ~~Collect at least 1,000 decisions under the checked-in schema.~~ Met by the public suite
   (33,126 decisions), with the adjudication caveat above.
2. ~~Freeze the dataset and split manifest hashes.~~ Recorded in
   `artifacts/public/import-report.json`.
3. Run direct, generative, and paired development reports within the M1 compute budget.
4. Create/freeze the exact 2,048-token, 16-Boolean benchmark request and execute the checked-in
   HTTP latency protocol on the reference host.
5. Review failures and freeze the M1 quality/latency report. Do not inspect the locked test until
   prompt/model selection is complete.
