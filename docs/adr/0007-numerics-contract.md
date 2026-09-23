# ADR-0007: Numerics Contract and FP32 Reduction Accumulation

- **Status**: Approved / Implemented
- **Date**: 2026-09-20
- **Authors**: JAH Core Engineering
- **References**: `claude_gap.md` (Gap 2A, 2C), ADR-0003, `src/jah/backends/huggingface.py`, `src/jah/service.py`, `tests/test_determinism.py`

---

## Context

On modern NVIDIA GPU architectures (Ampere, Ada Lovelace, Hopper), cuBLAS enables reduced-precision reduction accumulation by default for BF16 and FP16 matrix multiplications (`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True`). In this mode, intermediate summation operations during GEMM reductions are truncated to 16-bit precision to maximize tensor core utilization.

In large-scale language model inference, this introduces substantial non-deterministic numerical drift across different batch sizes, sequence lengths, and prefix caching layouts:
1. Intermediate partial sums suffer precision loss in BF16 (which has only 7 mantissa bits).
2. Reduction trees vary with batch shape, producing 1–2 ULP logit fluctuations.
3. These fluctuations disrupt reproducible scoring, cause non-deterministic argmax flips on competing candidates, and destabilize calibrated probability boundaries.

---

## Decision

We establish an immutable **Numerics Contract** governing all inference paths in JAH:

### 1. Mandatory FP32 Reduction Accumulation
On all CUDA-accelerated execution paths:
```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
```
This forces all GEMM reduction accumulation steps to be performed in 32-bit floating point (FP32) while model weights and activations remain in BF16/FP16.

### 2. Startup Assertion in the Serving Layer
The setting is not merely a test harness fixture. It is enforced in:
- `src/jah/backends/huggingface.py` upon backend initialization.
- `src/jah/service.py` at application startup.

If CUDA is active and either reduction flag is set to `True`, the service **refuses to start** and immediately raises `RuntimeError("FP32 accumulation required on decision path")`. Silent degradation or fallback is forbidden.

### 3. Version Pinning as a Numerics Artifact
The compute stack versions:
- CUDA Runtime (12.x)
- cuBLAS library
- PyTorch release (`torch >= 2.5.0`)
- NVIDIA driver family
are defined as **numerics artifacts**. A version update in any of these components cannot be treated as a routine dependency bump; it constitutes a gated change requiring execution and verification of the full CI determinism suite (`tests/test_determinism.py`).

### 4. Continuous Determinism CI Suite
The CI determinism suite tests batch configurations ($B \in \{1, 8, 16, 32\}$) with and without KV-cache prefix reuse, asserting:
- Zero argmax flips outside the tie band ($m \ge 2.80$).
- $\max \Delta p \le 0.01$ outside the tie band.
- Strict preservation of the empirical tie-band threshold.

---

## Consequences

- **Determinism**: Argmax disagreement across batch shapes and prefix reuse is eliminated (measured 0 flips across 69 evaluation decisions on RTX 4090).
- **Latency / Throughput Impact**: Throughput measurement demonstrates that disabling reduced-precision reductions incurs less than **1.5%** throughput overhead on modern tensor cores—well below the 10% re-evaluation threshold.
- **Maintenance Discipline**: Driver, CUDA, and PyTorch upgrades must pass numerical regression gates before merge.
