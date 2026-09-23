# Operational Runbook: Recalibration Procedure (ADR-08 / Step 3C)

**Document ID:** `RUNBOOK-001-CALIBRATION`  
**Status:** Active Operational Standard  
**Governing ADRs:** ADR-04 (Versioned Calibration), ADR-08 (Calibration Strategy)

---

## 1. When to Recalibrate (Trigger Conditions)

Recalibration is mandatory whenever any of the following triggers occur:

1. **Drift Monitoring Alert** (`jah-eval drift`):
   - **Critical breach**: Realized 95% error upper bound exceeds the target certified bound ($> 0.05$).
   - **Coverage collapse**: Realized coverage falls below the minimum requirement ($< 50\%$).
   - **Approaching bound**: Realized 95% error upper bound $\ge 80\%$ of threshold ($\ge 0.040$) across consecutive evaluation runs.
2. **Backbone Model or Precision Change**:
   - Any upgrade, re-quantization, or revision bump of the foundation model.
   - Any modification to attention layout, precision, or chat template.
3. **Adapter Addition or Retraining**:
   - Any LoRA or decision head fine-tuning run (e.g. `artifacts/adapters/`).
4. **Data Distribution Shift**:
   - Upstream traffic domain shift or significant change in class prevalence.

---

## 2. Sample Size Requirements (Clopper-Pearson 95% Bound)

Post-hoc calibration uses exact one-sided finite-sample binomial Clopper-Pearson bounds ($1 - \alpha = 0.95$). To certify an error bound $\le 5\%$ at 95% confidence, the calibration split must contain sufficient independent accepted observations:

| Observed Accepted Errors ($k$) | Minimum Accepted Cases Needed ($N$) | Realized Error Rate ($k/N$) | Exact 95% Upper Bound |
| :---: | :---: | :---: | :---: |
| **0** | **59** | 0.00% | 4.96% |
| **1** | **93** | 1.08% | 4.97% |
| **2** | **124** | 1.61% | 4.99% |
| **3** | **153** | 1.96% | 4.98% |
| **4** | **181** | 2.21% | 4.99% |
| **5** | **208** | 2.40% | 4.99% |
| **10** | **336** | 2.98% | 4.99% |
| **20** | **574** | 3.48% | 4.99% |

> [!IMPORTANT]
> Because coverage must be $\ge 50\%$, the raw calibration partition size must be at least **$2\times$** the minimum accepted sample count (e.g., at least $120$ samples for $k=0$, and $250+$ samples when anticipating $k \le 2$).

---

## 3. Step-by-Step Recalibration Execution

### Step 3.1: Validate Dataset Integrity & Boundaries
Ensure upstream dataset provenance and deterministic partition splits are frozen:
```powershell
python -m jah.evaluation validate --dataset evals/data/public/suite.jsonl --suite-config configs/evals/public-suite.yaml
```

### Step 3.2: Fit Temperature and Vector Calibration
Run calibration on the held-out `calibration` split only. Never fit on development or locked test data:
```powershell
python -m jah.evaluation calibrate `
  --dataset evals/data/public/suite.jsonl `
  --suite-config configs/evals/public-suite.yaml `
  --model-config configs/models/qwen3.5-4b.yaml `
  --output configs/profiles/public `
  --summary artifacts/public/calibration-summary.json
```

### Step 3.3: Evaluate Realized Selective Performance on Development Split
Verify calibration effectiveness, coverage ($\ge 50\%$), and selective error bound ($\le 5\%$):
```powershell
python -m jah.evaluation run `
  --backend direct `
  --split development `
  --use-workload-profiles `
  --output artifacts/public/post-recalibration-dev.json `
  --predictions artifacts/public/post-recalibration-dev.jsonl
```

### Step 3.4: Run Drift and Compliance Verification
Verify that the newly fitted profile eliminates drift alerts:
```powershell
python -m jah.evaluation drift `
  --predictions artifacts/public/post-recalibration-dev.jsonl `
  --profile configs/profiles/public/banking77-16-intent-v1.yaml
```

---

## 4. Sign-Off & Governance

Before committing and deploying updated calibration profiles:

1. **Backbone Integrity**: Profile YAML must contain exact `model_id`, `revision`, and `precision` matching the serving configuration.
2. **Quality Verification**:
   - `gate_ece_passed: true` (ECE $\le 0.05$).
   - `gate_brier_passed: true` (Brier score beats prevalence baseline).
   - `gate_selective_passed: true` (Coverage $\ge 50\%$, 95% error bound $\le 5\%$).
3. **Approval**:
   - Requires review sign-off by the Workload Owner and ML Systems Lead.
   - Profile YAML must be committed to `configs/profiles/` with git commit SHA recorded.
