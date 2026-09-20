# M4 Adaptation Status

Status: **Complete; LoRA parameter-efficient adaptation engine (`jah-train`), dynamic module injection (`q_proj`, `v_proj`, `lm_head`), activation gradient checkpointing, and adapted evaluation verified. Decision head fine-tuning yields +8.75% accuracy and +0.0986 Macro-F1 lift on held-out development data.**

---

## 1. Implemented Architecture

Per RFC 001 and ADR-05 (`JEV_AT_HOME_SPEC.md`):

### LoRA Adaptation Engine
- **Module Injection & Removal (`src/jah/training/adapter.py`)**:
  - `inject_lora(model, config)` traverses arbitrary PyTorch module hierarchies, replacing targeted `nn.Linear` layers (`q_proj`, `v_proj`, and `lm_head`) with `LoRALinear` modules.
  - Frozen base model: 4.54B parameters remain non-trainable (`requires_grad = False`).
  - Low-rank adapters ($r=8$, $\alpha=16.0$): 2,924,544 trainable parameters (**0.064%** of backbone).
  - Preserves base layer device and dtype (BF16 on CUDA).
- **Candidate-Restricted Decision Objective (`src/jah/training/adapter.py`)**:
  - Restricts cross-entropy loss to tokenizer-verified candidate decision tokens:
    $$\mathcal{L}_{CE} = -\sum_{c \in \mathcal{C}} y_c \log p_c$$
- **Training Runner CLI (`jah-train`, `src/jah/training/train.py`)**:
  - Memory-safe training with PyTorch gradient checkpointing (`model.gradient_checkpointing_enable()`) and `enable_input_require_grads()`.
  - Max input sequence token filtering (default: 2,048 tokens) preventing quadratic attention allocation spikes.
  - Gradient accumulation, gradient clipping (`max_grad_norm = 1.0`), and deterministic dataset shuffling.
  - Export pipeline producing versioned `adapter_weights.pt` and `adapter_manifest.json`.
- **Inference & Evaluation Integration (`src/jah/backends/huggingface.py`, `src/jah/evaluation.py`)**:
  - `HuggingFaceDirectLogitBackend` dynamically attaches and initializes adapter weights when `adapter_dir` is supplied.
  - `jah-eval run --adapter-dir <path>` evaluates adapted models with identical CLI workflow.

---

## 2. Measured Benchmark Evidence (Head Adaptation vs. Frozen Baseline)

Evaluated on the **`banking77-16-intent-v1` development partition** (297 held-out human decisions, 16-way intent classification) on the reference host (NVIDIA GeForce RTX 4090, BF16):

| Metric | Frozen Baseline (Direct Logits) | Adapted LoRA Head (`qwen3.5-4b-public-head-v1`) | Absolute Change ($\Delta$) | Specification Gate | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 82.83% | **91.58%** | **+8.75%** | - | **Substantial Lift** |
| **Balanced Accuracy** | 82.19% | **91.42%** | **+9.23%** | - | **Substantial Lift** |
| **Macro-F1** | 0.8126 | **0.9112** | **+0.0986** | $\ge 0.85$ | **PASS** |
| **Brier Score** | 0.2646 | **0.1525** | **-42.4% error** | $\le \text{Prevalence}$ | **PASS** |
| **Negative Log-Likelihood** | 0.7373 | **0.4440** | **-39.8% NLL** | - | **Sharper Calibration** |
| **Median Request Latency** | 89.59 ms | **88.51 ms** | -1.08 ms | - | **Sub-100ms Preserved** |
| **$P_{95}$ Request Latency** | 118.13 ms | **114.48 ms** | -3.65 ms | $\le 2000\text{ ms}$ | **PASS** |

Artifact evidence:
- Trained Adapter Bundle: [artifacts/adapters/qwen3.5-4b-public-head-v1/](artifacts/adapters/qwen3.5-4b-public-head-v1/)
- Training Telemetry: [artifacts/adapters/qwen3.5-4b-public-head-v1/training-report.json](artifacts/adapters/qwen3.5-4b-public-head-v1/training-report.json)
- Baseline Evaluation: [artifacts/public/baseline-banking77-dev.json](artifacts/public/baseline-banking77-dev.json)
- Adapted Head Evaluation: [artifacts/public/adapted-banking77-dev.json](artifacts/public/adapted-banking77-dev.json)

---

## 3. Test Suite Verification

- `pytest` executes **117 tests with 0 failures** across all modules:
  - `tests/test_train.py` (2 passed)
  - `tests/test_adapter.py` (6 passed)
  - `tests/test_batch_equivalence.py` (4 passed)
  - Prior suites (105 passed)
- `ruff check .`: 0 warnings, formatting strictly compliant.
