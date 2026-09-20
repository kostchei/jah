# Walkthrough: Public Human Benchmark Calibration & LoRA Head Adaptation

We have calibrated the frozen Qwen 3.5-4B baseline on the 3 public human-labeled workloads, built the M4 LoRA head adaptation engine with standalone CLI tooling (`jah-train`), trained an adapted decision head on the RTX 4090, and demonstrated significant quality improvements on held-out development data.

---

## 1. Temperature Calibration Summary ([artifacts/public/calibration-summary.json](artifacts/public/calibration-summary.json))

All **2,565 calibration split decisions** were evaluated and calibrated via 1D golden-section search over negative log-likelihood:

| Workload | Decisions | Primitive | Fitted Temp $T$ | 10-Bin ECE | Brier Score | Baseline Prevalence Brier | Gate Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`banking77-16-intent-v1`** | 240 | Choice (16-way) | **1.3522** | **0.0475** | **0.2348** | 0.9292 | **ECE & Brier Passed** |
| **`wikiqa-answer-relevance-v1`** | 1,883 | Boolean | **0.7999** | **0.0454** | 0.2994 | 0.0996 | **ECE Passed** |
| **`asap2-source-essay-v1`** | 442 | Score (Ordinal) | **1.2162** | 0.0945 | 0.7708 | 0.7490 | Review-only (Zero-shot gap) |

Profiles are versioned and stored in [configs/profiles/public/](configs/profiles/public/).

---

## 2. LoRA Head Adaptation & Training Trajectory

We built the standalone trainer [`jah-train`](src/jah/training/train.py) with dynamic module injection, activation memory optimization via gradient checkpointing, and candidate-restricted cross-entropy loss.

### Training Configuration & Compute:
- **Backbone**: Qwen 3.5-4B (BF16, RTX 4090)
- **Target Modules**: `q_proj`, `v_proj`, `lm_head` (17 LoRA modules)
- **Trainable Parameters**: **2,924,544** / 4,542,190,080 (**0.064%**)
- **Training Samples**: 1,000 balanced decisions across public workloads
- **Optimizer**: AdamW ($\text{lr} = 10^{-4}$, $\text{batch\_size} = 4$, $\text{accum\_steps} = 4$, $\text{max\_grad\_norm} = 1.0$)
- **Compute Time**: **311.3 seconds** (5.18 minutes) across 250 optimizer steps

### Loss Convergence:
```text
Step   1  | Example   4 | Loss: 0.9340
Step  25  | Example 100 | Loss: 0.6541
Step  50  | Example 200 | Loss: 0.0185
Step 100  | Example 400 | Loss: 0.0143
Step 150  | Example 600 | Loss: 0.0011
Step 200  | Example 800 | Loss: 0.0650
Step 250  | Example 999 | Loss: 0.0001 (Final: 0.00005)
```

Exported artifact bundle in [`artifacts/adapters/qwen3.5-4b-public-head-v1/`](artifacts/adapters/qwen3.5-4b-public-head-v1/):
- `adapter_weights.pt` (5.8 MB, SHA-256: `b8c2b772...`)
- `adapter_manifest.json` (SHA-verified against dataset hash `5a817155...`)
- `training-report.json` (loss history and training telemetry)

---

## 3. Side-by-Side Evaluation: Baseline vs. Tuned Head

Evaluated on the **`banking77-16-intent-v1` development partition** (297 held-out human decisions, 16-way intent classification) under identical prompt and hardware conditions:

| Metric | Frozen Baseline (Direct Logits) | Adapted LoRA Head (`qwen3.5-4b-public-head-v1`) | Absolute Change ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 82.83% | **91.58%** | **+8.75%** | **Significant Lift** |
| **Balanced Accuracy** | 82.19% | **91.42%** | **+9.23%** | **Significant Lift** |
| **Macro-F1** | 0.8126 | **0.9112** | **+0.0986** | **Exceeds 0.85 Target** |
| **Brier Score** | 0.2646 | **0.1525** | **-42.4% error** | **Improved Reliability** |
| **Negative Log-Likelihood** | 0.7373 | **0.4440** | **-39.8% NLL** | **Sharper Calibration** |
| **Median Request Latency** | 89.59 ms | **88.51 ms** | -1.08 ms | **Sub-100ms Preserved** |
| **$P_{95}$ Request Latency** | 118.13 ms | **114.48 ms** | -3.65 ms | **Sub-200ms Preserved** |

Evidence files:
- Baseline: [`artifacts/public/baseline-banking77-dev.json`](artifacts/public/baseline-banking77-dev.json)
- Adapted Head: [`artifacts/public/adapted-banking77-dev.json`](artifacts/public/adapted-banking77-dev.json)

---

## 4. Test Suite Status
- **117 / 117 tests pass** in `pytest` across all suites.
- **Ruff check**: 0 warnings.
