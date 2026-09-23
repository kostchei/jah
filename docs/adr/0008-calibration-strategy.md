# ADR-0008: Calibration Strategy — Post-Hoc Calibration with Exact Binomial Bounds

- **Status**: Approved / Implemented
- **Date**: 2026-09-20
- **Authors**: JAH Core Engineering
- **References**: `claude_gap.md` (Gap 3), `docs/runbooks/recalibration.md`, `src/jah/calibration.py`, `src/jah/drift.py`, `src/jah/service.py`

---

## Context

The Jev reference architecture relies on Reinforcement Learning with Calibration Dialogue (RLCD) during foundation pre-training to align model logits with true posterior probabilities $P(Y|X)$.

In designing JAH, we evaluated whether to emulate native RLCD in-weight calibration or retain a post-hoc calibration layer fitting a scalar temperature $T$ and bias vector $b$ on a dedicated validation split, guarded by exact Clopper-Pearson binomial bounds.

---

## Decision

We deliberately select and solidify **post-hoc calibration with exact Clopper-Pearson binomial confidence bounds** as the enterprise calibration strategy for JAH. Native in-weight calibration is rejected.

### 1. Rationale for Post-Hoc Calibration
1. **Verifiable, Auditable Guarantees**:
   Post-hoc calibration provides a mathematically rigorous, exact one-sided 95% upper bound on accepted error:
   $$P(\text{error} \le \epsilon_{\text{upper}}) \ge 0.95 \quad \text{at } \text{coverage} \ge 50\%$$
   RLCD produces empirical heuristics embedded in weights that cannot provide formal statistical bounds.
2. **Independent Inspectability & Provenance**:
   Calibration profiles exist as decoupled, version-controlled JSON artifacts (`artifacts/profiles/`). They can be inspected, mathematically re-verified, or audited by compliance teams without inspecting black-box neural network weights.
3. **Low-Cost Recalibration**:
   When customer traffic distribution shifts, post-hoc recalibration requires only evaluating a held-out split of a few hundred examples and optimizing two scalar parameters. Recalibrating an RLCD system requires full model fine-tuning or reinforcement learning.

### 2. Implementation & Hardening Measures

- **Artifact Discipline (Step 3A)**:
  Every `CalibrationProfile` must record cryptographic provenance:
  - `backbone_hash`: SHA256 of base model weights.
  - `lora_hash`: SHA256 of adapter weights (or null for base zero-shot).
  - `prompt_version`: canonical schema string (`decision-prompt-v2`).
  - `sample_count`, `fit_timestamp`, `coverage`, and `error_bound`.
  The serving layer (`src/jah/service.py`) verifies these hashes against running model weights at startup. If a mismatch is detected, startup is aborted. No fallback to uncalibrated logits or default profiles is permitted.
- **Drift Monitoring (Step 3B)**:
  `src/jah/drift.py` and the `jah-eval drift` CLI monitor production decision logs against the Clopper-Pearson 95% upper bound. Alerts are raised on:
  - `ERROR_BOUND_WARNING`: Realized error exceeds 80% of the upper bound.
  - `ERROR_BOUND_BREACH`: Realized error exceeds the upper bound.
  - `COVERAGE_BELOW_MINIMUM`: Realized acceptance coverage falls below 50%.
- **Recalibration Runbook (Step 3C)**:
  A standardized operational procedure is documented in `docs/runbooks/recalibration.md`, including required sample size curves, parameter bounds ($0.1 \le T \le 10.0$), and human sign-off gates.

### 3. Explicit Re-Evaluation Trigger Condition (Step 3D)
In-weight calibration (such as RLCD) will **not** be investigated unless post-hoc calibration demonstrably fails. Specifically, the line of work remains closed unless:
> A single $(T, b)$ parameterization fails to achieve the target bound ($\le 5\%$ accepted error at $\ge 50\%$ coverage) across the multi-task mixture following Phase 1C multi-task decision tuning.

---

## Consequences

- Calibration remains modular, auditable, and easily updated across enterprise customer deployments.
- Serving startup guarantees that running models match their certified calibration bounds.
- Operational runbooks govern drift alerting and recalibration lifecycles.
