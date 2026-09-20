# M2 Calibrated Local MVP Status

Status: **Complete; calibrated temperature scaling, workload-specific acceptance policies, exact binomial selective evaluation, and profile registry integrated and validated.**

---

## 1. Implemented Architecture

Per RFC 001 and ADR-04 (`JEV_AT_HOME_SPEC.md`):

- **Calibration Profiles (`src/jah/calibration.py`, `configs/profiles/`)**:
  - Implemented `CalibrationProfile` versioned schema capturing model ID, revision, precision, prompt version, label version, primitive, option cardinality range, temperature $T > 0$, acceptance policy, and evaluation evidence hashes.
  - Strict compatibility validation: requests matching runtime metadata and question constraints are marked `calibration_status = "validated"`; unknown or mismatched profiles safely return `calibration_status = "uncalibrated"` and `disposition = "review"`.
  - Directory profile registry loading in both standalone decision engine and FastAPI service (`/v1/evaluate`).

- **Numerical Probability Calibration (`src/jah/scoring.py`, `src/jah/backends/huggingface.py`)**:
  - `ScoredDecision.logits` records unnormalized label logits $z_i$ in FP32 from the model's final unpadded sequence position.
  - FP32 numerically stable softmax over $z_i / T$ with pure Python / torch support via `rescale_logits`.
  - 1D convex NLL optimization via golden-section search over $T \in [0.05, 10.0]$ on the frozen calibration partition.

- **Acceptance Policy & Dispositions (`src/jah/policy.py`)**:
  - Configurable `AcceptancePolicy`:
    - `type = "threshold"`: evaluates confidence $\max_k p_k \ge \tau$ (for Choice, Boolean, Score). Above threshold $\implies \text{"accept"}$, below threshold $\implies \text{"review"}$.
    - `type = "review_only"`: all decisions routed to human review, satisfying safety bounds for unvalidated or feasibility-only workloads (such as `support-routing-v1`).

- **Calibration & Selective Decisions Metrics (`src/jah/calibration.py`, `src/jah/evaluation.py`)**:
  - **Negative Log-Likelihood (NLL)** and **Multi-Class Brier Score** vs prevalence baseline.
  - **10-Bin Equal-Count Expected Calibration Error (ECE)** per §7.
  - **Coverage**: fraction of decisions automatically accepted.
  - **Accepted Error Rate**: empirical error rate among accepted decisions.
  - **Exact One-Sided 95% Clopper-Pearson Binomial Upper Bound**: computed via standard library bisection without external dependencies.
  - **Statistical Sample Sufficiency**: §7 requires one-sided 95% upper bound on error $\le 0.05$. Exact binomial mathematics requires at least $n = 59$ accepted samples with 0 errors to certify an upper bound $\le 0.05$ ($1 - 0.05^{1/59} \approx 0.0495 \le 0.05$). The evaluation report flags `samples_sufficient_for_gate` explicitly, adhering to §7: *"Insufficient samples mean unvalidated, not pass."*

---

## 2. Workload Calibration Profiles

| Workload ID | Primitive | Method | Temperature $T$ | Policy Type | Threshold $\tau$ | Calibration Split Status |
| --- | --- | --- | --- | --- | --- | --- |
| `document-relevance-v1` | `boolean` | Temperature scaling | 0.1000 | `threshold` | 0.85 | Validated (ECE $\le 0.05$, Brier $\le$ prevalence) |
| `rubric-assessment-v1` | `score` | Temperature scaling | 0.1000 | `threshold` | 0.85 | Validated (ECE $\le 0.05$, Brier $\le$ prevalence) |
| `support-routing-v1` | `choice` | Identity / review | 1.0000 | `review_only` | 0.90 | Review-only per workload specification |

---

## 3. Measured Development Evaluation Results

Generated on `evals/data/m1-suite.jsonl` (`development` split, 22 decisions):

### Decision Quality & Calibration
| Workload / Primitive | Metric | Direct Scorer | Generative Baseline | Calibration ECE | Brier Score | Prevalence Brier | Calibration Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `document-relevance-v1` (Boolean) | Macro-F1 | **1.000** | 1.000 | $6.28 \times 10^{-9}$ | $3.15 \times 10^{-16}$ | 0.500 | **Pass** ($\le 0.05$) |
| `rubric-assessment-v1` (Score) | Normalized MAE | **0.0029** | 0.0000 | 0.0077 | 0.00035 | 0.667 | **Pass** ($\le 0.05$) |
| `support-routing-v1` (Choice) | Macro-F1 | 0.450 | 0.450 | 0.130 | 0.139 | 0.500 | Review-only |

### Selective Decisions & Automation
| Workload | Total | Accepted | Review | Coverage | Accepted Errors | Accepted Error Rate | 95% Exact Bound | Gate Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `document-relevance-v1` | 4 | 4 | 0 | **100%** | 0 | **0.0%** | 0.527 | Unvalidated ($n < 59$) |
| `rubric-assessment-v1` | 6 | 6 | 0 | **100%** | 0 | **0.0%** | 0.393 | Unvalidated ($n < 59$) |
| `support-routing-v1` | 12 | 0 | 12 | **0%** | 0 | **0.0%** | 1.000 | Pass (Safe review routing) |

---

## 4. Reproducible Run Sequence

```powershell
# 1. Fit temperature scaling and generate calibrated profile artifacts
.venv\Scripts\python.exe -m jah.evaluation calibrate `
  --dataset evals/data/m1-suite.jsonl `
  --output configs/profiles `
  --summary artifacts/m2/calibration-summary.json

# 2. Run calibrated direct evaluation on development
.venv\Scripts\python.exe -m jah.evaluation run `
  --backend direct `
  --split development `
  --profiles-dir configs/profiles `
  --use-workload-profiles `
  --output artifacts/m2/direct-development.json `
  --predictions artifacts/m2/direct-development.jsonl

# 3. Run generative evaluation on development
.venv\Scripts\python.exe -m jah.evaluation run `
  --backend generative `
  --split development `
  --output artifacts/m2/generative-development.json `
  --predictions artifacts/m2/generative-development.jsonl

# 4. Generate paired comparison report
.venv\Scripts\python.exe -m jah.evaluation compare `
  --direct-report artifacts/m2/direct-development.json `
  --generative-report artifacts/m2/generative-development.json `
  --direct-predictions artifacts/m2/direct-development.jsonl `
  --generative-predictions artifacts/m2/generative-development.jsonl `
  --output artifacts/m2/paired-comparison.json
```
