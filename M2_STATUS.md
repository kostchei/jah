# M2 calibrated local MVP status

Status: **calibration, policy, and selective-decision machinery implemented and unit-tested; the
M2 exit decision is not met.** The profiles this milestone originally fitted have been retired,
and no approved workload currently passes the quality, calibration, and selective-error gates
together.

Gate verdicts are rendered from artifacts into [EVIDENCE.md](EVIDENCE.md).

---

## 1. Implemented

Per ADR-04.

- **Versioned profiles** ([src/jah/calibration.py](src/jah/calibration.py)): `CalibrationProfile`
  captures model ID, revision, precision, prompt version, label version, primitive, option
  cardinality range, temperature, acceptance policy, and evaluation evidence. Requests matching
  runtime metadata and question constraints return `calibration_status: "validated"`; unknown or
  mismatched profiles return `"uncalibrated"` and `disposition: "review"`.
- **Evidence gating** (`has_validated_evidence`): compatibility alone never promotes a profile. A
  profile enables acceptance only with passing ECE, Brier, and selective gates, coverage at or
  above 0.50, an accepted-error upper bound at or below 0.05, and at least 59 accepted
  observations.
- **Numerical calibration** ([src/jah/scoring.py](src/jah/scoring.py)): FP32 softmax over `z / T`
  from the final unpadded position, with temperature fitted by golden-section search over NLL on
  the calibration partition.
- **Acceptance policy** ([src/jah/policy.py](src/jah/policy.py)): `threshold` and `review_only`.
- **Metrics** ([src/jah/calibration.py](src/jah/calibration.py),
  [src/jah/evaluation.py](src/jah/evaluation.py)): NLL, multi-class Brier against a prevalence
  baseline, 10-bin equal-count ECE, coverage, accepted error rate, and an exact one-sided 95%
  Clopper-Pearson upper bound computed by bisection.
- **Sample sufficiency**: reports flag `samples_sufficient_for_gate` explicitly. An exact binomial
  bound of 0.05 requires at least 59 error-free accepted observations, so smaller samples are
  recorded as unvalidated rather than passed.

---

## 2. Retired profiles

The three profiles fitted in the original M2 run have been moved to
[configs/profiles/retired/](configs/profiles/retired/) and are excluded from every default
registry. They were fitted on `evals/data/m1-suite.jsonl`, whose 141 rows were labeled by
`qwen/qwen3.8-27b` and then scored by the backbone — a model family grading its own output, which
ADR-06 excludes from evaluation ground truth.

Two of the three fitted `temperature: 0.1`, the floor of the search range, driving every
distribution to a one-hot; their recorded reliability bins show mean confidences of `1.0`. Sample
counts were 6 and 4. The original M2 tables reported these as "Validated (ECE <= 0.05, Brier <=
prevalence)". That claim is withdrawn.

`evals/data/m1-suite.jsonl` is retained only as a fast smoke fixture covering all three
primitives. [configs/evals/m1-suite.yaml](configs/evals/m1-suite.yaml) records this.

---

## 3. Current profiles

The profiles in [configs/profiles/public/](configs/profiles/public/) are fitted on the calibration
partition of the public human-labeled suite. All three are `review_only` and none carry validated
acceptance evidence — coverage is 0.0 in every recorded run, so the selective-error gate has not
been exercised at all.

| Workload | Primitive | T | ECE | Brier | Prevalence Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| `banking77-16-intent-v1` | choice | 1.3522 | 0.047522 | 0.234752 | 0.929201 |
| `wikiqa-answer-relevance-v1` | boolean | 0.7999 | 0.045374 | 0.299406 | 0.099623 |
| `asap2-source-essay-v1` | score | 1.2162 | 0.094514 | 0.770769 | 0.748971 |

Source: [artifacts/public/calibration-summary.json](artifacts/public/calibration-summary.json).

---

## 4. Remaining M2 exit work

1. Correct the WikiQA base-rate failure; scalar temperature cannot (remediation plan, phase 2.1).
2. Bring ASAP within the ordinal gates or declare it out of scope with evidence (phase 2.2).
3. Attach profiles at evaluation time, sweep thresholds on the calibration split, publish
   risk-coverage curves, and measure the accepted-error bound (phase 2.3).

---

## 5. Reproducible run sequence

```powershell
# Fit temperature scaling on the public calibration partition.
.venv\Scripts\jah-eval.exe calibrate `
  --output configs/profiles/public `
  --summary artifacts/public/calibration-summary.json

# Evaluate the development partition with those profiles attached.
.venv\Scripts\jah-eval.exe run --backend direct --split development `
  --workload banking77-16-intent-v1 `
  --profiles-dir configs/profiles/public --use-workload-profiles `
  --output artifacts/public/direct-development.json `
  --predictions artifacts/public/direct-development.jsonl
```

Defaults now resolve to the public suite (`evals/data/public/suite.jsonl`,
`configs/evals/public-suite.yaml`, split seed `jah-public-v1`).
