# Fresh-data validation for optimized scoring

## What this test can establish

Keep two questions separate:

1. **Numerical/runtime equivalence:** on new request text, compare the sequential full-prompt
   reference with the optimized engine entry point. Require every decision to agree, probability
   deviation at or below 0.01, no policy flips, and peak reserved VRAM below 22 GB. This checks
   the scoring implementation; it does not measure whether an answer is correct.
2. **Decision quality:** compare the model with labels created independently of JAH/model output
   on a held-out source split. Report per-class and per-slice quality, not just pooled accuracy.
   Equivalence passing cannot substitute for this quality test.

## External benchmark now available

The UCI CLINC150 archive has a test partition of 4,500 in-scope utterances and 1,000
out-of-scope utterances. It is a crowdsourced intent benchmark with held-out labels, and its
travel domain has 15 intents. The preparation script selects all 450 travel-domain test items,
450 small-talk items as cross-domain negatives, and all 1,000 out-of-scope test items (nominally
1,900 decisions). It does not use CLINC150 train/dev examples as evaluation rows, and it excludes
and reports any normalized held-out utterance that exactly overlaps train/dev text.

This is an external benchmark against JAH's previous model-adjudicated responses, not a proof
that the base model never saw CLINC150 during pretraining. It also does not meet JAH's stricter
two-local-human-annotator release-evidence requirement: keep it benchmark-only. For the strongest
freshness claim, collect a later, representative sample after freezing the model and prompt, then
have two humans label it blind to model predictions and adjudicate disagreements.

The UCI page describes the archive as CC BY 4.0, while the downloaded archive contains a CC BY
3.0 license file. The preparation tool records the more conservative embedded archive license.
Do not commit the downloaded archive, derived utterances, or prediction rows; the `evals/cache/`
directory is ignored by Git. Source, transformed-data, and split hashes plus the preparation
counts are recorded in [the CLINC150 test manifest](../../evals/fixtures/external/clinc150-travel-test-manifest.json).
Preserve attribution and verify licensing before redistributing any derived data.

## Reproduce the CLINC150 test

From the repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Path evals/cache/clinc150 -Force | Out-Null
Invoke-WebRequest `
  -Uri 'https://archive.ics.uci.edu/static/public/570/clinc150.zip' `
  -OutFile 'evals/cache/clinc150/clinc150.zip'
Expand-Archive `
  -LiteralPath 'evals/cache/clinc150/clinc150.zip' `
  -DestinationPath 'evals/cache/clinc150' -Force
python scripts/prepare_clinc150_travel_test.py
python -m jah.evaluation --root . validate `
  --dataset evals/cache/clinc150/clinc150-travel-test.jsonl `
  --suite-config configs/evals/clinc150-travel-test.yaml `
  --split-seed clinc150-test-v1 `
  --output artifacts/clinc150/validation.json
```

The validation report must show at least 1,000 decisions, only the configured `choice`
primitive, the `upstream-human` annotation status, and `release_annotation_ready: false` (this
benchmark is not a two-local-reviewer release suite).

Run the implementation-equivalence check on 200 deterministic fresh decisions packed into
100 two-question requests. The paired utterances are placed in the same state so the optimized
engine path is exercised; the answer keys are not used by this check:

```powershell
.\scripts\run_m3_equivalence.ps1 `
  -Requests evals/cache/clinc150/equivalence-requests `
  -MicrobatchSize 1
```

Finally, score the complete 1,900-decision held-out set against its labels. Keep acceptance
profiles disabled unless they were calibrated before opening this test set:

```powershell
python -m jah.evaluation --root . run `
  --backend direct `
  --dataset evals/cache/clinc150/clinc150-travel-test.jsonl `
  --suite-config configs/evals/clinc150-travel-test.yaml `
  --split-seed clinc150-test-v1 `
  --split locked_test `
  --model-config configs/models/qwen3.5-4b.yaml `
  --output artifacts/clinc150/travel-test-report.json `
  --predictions artifacts/clinc150/travel-test-predictions.jsonl `
  --resource-limit 0.85
```

The quality report is descriptive, not an automatic release approval. Inspect travel-intent
macro-F1, per-intent recall, and the separate rates for small-talk rejection and true OOS
rejection. If any prompt, profile, threshold, or model setting is changed after examining these
results, this test set is development data for that change; obtain a new blind holdout before
making a fresh quality claim.

## Use a newly collected human-reviewed holdout

For the release-quality run, build a new M1 JSONL suite with stable source and duplicate-cluster
IDs, `annotation_status` set to `adjudicated-agreement` or `adjudicated-resolution`, and two
distinct independent human annotations per decision. Annotators must not see model predictions.
Freeze the dataset SHA-256, split seed, model revision/precision, prompt and label versions, and
profile files before running the locked split. Keep calibration and development examples
disjoint by source and near-duplicate cluster. Use `task_template_holdout: true` only for
pre-registered unseen task families; it routes those examples to the separate `task_holdout`
split.

The same evaluator can run a choice-only or other single-primitive suite by setting
`required_primitives` in its suite YAML. Keep the default public-suite requirement of all three
primitives when evaluating the mixed JAH suite. Do not tune on `locked_test` or `task_holdout`.

## Sources

- [UCI CLINC150 dataset record](https://archive.ics.uci.edu/dataset/570/clinc150) — archive, split sizes, and repository license metadata.
- [CLINC150 authors' repository](https://github.com/clinc/oos-eval) — original dataset and collection project.
- [CLINC150 original paper](https://aclanthology.org/D19-1131/) — crowdsourced intent/OOS benchmark.
- [Published CLINC150 domain/intent list](https://aclanthology.org/2024.customnlp4u-1.15.pdf) — travel taxonomy used by the adapter.
