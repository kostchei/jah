# M1 reproducible-baseline status

Status: **unblocked; initial model-adjudicated baseline generated and paired reports recorded**.

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

The M1 intake file is `evals/data/m1-suite.jsonl`. It contains 141 model-adjudicated decisions
spanning Choice (`support-routing-v1`), Boolean (`document-relevance-v1`), and Score (`rubric-assessment-v1`),
generated and adjudicated by `qwen/qwen3.8-27b` via LM Studio.

`jah-eval validate` passes with zero failures and establishes the deterministic 60/15/10/15 split.
The suite can be refined and augmented with real production traffic and human reviews over time.

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

1. Collect and adjudicate at least 1,000 decisions under the checked-in schema.
2. Freeze the resulting dataset and split manifest hashes.
3. Run direct, generative, and paired development reports within the M1 compute budget.
4. Create/freeze the exact 2,048-token, 16-Boolean benchmark request and execute the checked-in
   HTTP latency protocol on the reference host.
5. Review failures and freeze the M1 quality/latency report. Do not inspect the locked test until
   prompt/model selection is complete.
