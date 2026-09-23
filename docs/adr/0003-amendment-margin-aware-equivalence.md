# ADR-0003 (Amendment): Diagnostic Margin Analysis for Batched and Prefix-Cached Inference

- **Status**: Superseded for release gating; retained for diagnostic analysis
- **Date**: 2026-09-20
- **Authors**: JAH Core Engineering
- **Amends**: ADR-03 (Shared-Prefix Reuse Optimization)
- **References**: `claude_gap.md` (Gap 2B), `M3_STATUS.md`, `EVIDENCE.md`

---

## Context

The initial ADR-03 equivalence contract specified that shared-prefix reuse and microbatching must achieve $\max \Delta p \le 0.01$ and exact argmax identity across all evaluated decisions compared to the full-prompt sequential reference.

During GPU validation of Milestone M3 on NVIDIA Ada Lovelace hardware (RTX 4090), the system exhibited $\max \Delta p = 0.0897$, failing the strict global $0.01$ gate. Detailed diagnostic investigation revealed:

1. **Non-Associativity of cuBLAS GEMM Reductions**:
   Batching multiple questions or reusing prefix KV cache states alters the sequence length, microbatch shapes, and tiling strategies chosen by cuBLAS. Floating-point addition is non-associative. Under BF16 accumulation, differing summation reduction trees induce small logit deviations of $1 \text{ to } 2 \text{ ULPs}$ ($0.0625 \text{ to } 0.125$).
2. **Derivative Amplification Near Decision Boundaries**:
   For a binary or multi-candidate decision near parity ($p_i \approx 0.5$), the derivative of softmax with respect to the logit difference is maximal:
   $$\frac{\partial p}{\partial (z_1 - z_2)} = p(1 - p) \approx 0.25$$
   A minimal hardware-induced logit shift of $\Delta z = 0.125$ thus produces a probability delta:
   $$\Delta p \approx 0.25 \times 0.125 \approx 0.03125$$
   In cases with multiple competing options near equality, $\Delta p$ reached $0.0897$. This is not a software bug or sampling divergence; it is an intrinsic consequence of IEEE 754 non-associativity in cuBLAS reduction trees.
3. **Flawed Alternative Proposals**:
   - *Loosening global $\Delta p \le 0.09$*: Rejected because it would permit genuine optimization regressions on well-separated decisions to go undetected.
   - *Serving exclusively at batch size 1*: Rejected because it forfeits GPU throughput gains (sacrificing an 8x–16x speedup) to satisfy an unphysical numerical ideal.

---

## Decision

The 2.80 margin is retained as a diagnostic slice for investigating near-tie behavior. It does
not amend release equivalence requirements and does not exempt any decision from them.

### 1. Diagnostic: Decision Agreement by Margin
For all decisions where the top-1 logit margin:
$$m = z_{(1)} - z_{(2)} \ge \tau_{\text{margin}}$$
the argmax prediction of the optimized path (prefix reuse and microbatching) must match the sequential reference **exactly** ($100\%$ argmax agreement, zero flips allowed).

### 2. Release Probability Equivalence
Across all decisions, the maximum absolute probability difference across all candidates must satisfy:
$$\max_{i} |\hat{p}_i - p_i^{\text{ref}}| \le 0.01$$

### 3. Why the Margin Is Not a Release Bound
A top-two margin of 2.80 does not imply a top-class posterior of 0.94 for general
multiclass decisions. The posterior depends on every competing logit; the binary formula
cannot be applied as a multiclass guarantee. A local softmax derivative also does not establish
a global bound under arbitrary logit perturbations. The measured 69-decision result is a
diagnostic observation, not a derivation of a universal bound.

### 4. Tie-Band Reporting
Decisions falling within the diagnostic band ($m < 2.80$) may be counted and reported to
guide targeted numerical work. They remain included in every release gate. No statistical
tie-band alert is currently part of the release process.

---

## Consequences

- **Correctness Assurance**: No decision is omitted from release equivalence.
- **Hardware Realism**: Numerical differences are measured on target hardware, not excused by an unvalidated margin heuristic.
- **Production Efficiency**: Optimization remains a performance objective and must meet the full equivalence contract before production reliance.
