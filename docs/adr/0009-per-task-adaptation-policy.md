# ADR-0009: Per-Task Adaptation and LoRA Escalation Policy

- **Status**: Approved / Implemented
- **Date**: 2026-09-20
- **Authors**: JAH Core Engineering
- **References**: `claude_gap.md` (Gap 1), `docs/architecture/gap1_multitask_tuning.md`, `src/jah/exemplars.py`, `src/jah/compiler.py`

---

## Context

In initial implementations of JAH, tasks involving nuanced rubrics or ordinal grading (such as ASAP essay evaluation) could not achieve acceptable accuracy zero-shot on the 4B chat backbone. To ship these tasks, developers trained task-specific LoRA adapters.

However, treating per-task LoRA adapters as the default onboarding mechanism introduces substantial operational overhead:
1. Every new task requires a dedicated training run, hyperparameter tuning, and checkpoint management.
2. Every LoRA adapter requires an independent calibration split and versioned calibration profile artifact.
3. Serving multiple adapters requires dynamic adapter swapping or separate serving endpoints, eroding throughput and memory density.

---

## Decision

We establish a formal policy defining the conditions under which per-task adaptation (LoRA adapters) is permitted:

### 1. Default Onboarding: Zero-Weight Canonical Conditioning
All new tasks must onboard through **data and prompt configuration only**, requiring **zero parameter updates**:
1. **Canonical Schema (`decision-prompt-v2`)**: Questions, criteria boundaries, and explicit tie-break rules are rendered through the standard compiler.
2. **Adjudicated Exemplar Store (`ExemplarStore`)**: Up to $k=8$ human-adjudicated examples are retrieved dynamically via token overlap or semantic similarity and injected into the prompt `<EXEMPLARS>` block.
3. **Temperature Scaling Profile**: A task-level calibration profile is fitted on a held-out split without updating backbone weights.

### 2. Mandatory Failure Gate Prior to LoRA Training
A task-specific LoRA adapter head is **strictly prohibited** unless the task demonstrably fails the Zero-Weight Acceptance Gate:
> A task may train a LoRA adapter only if, after rubric refinement, canonical prompt structuring, and 8-shot exemplar retrieval, the zero-shot accuracy falls below **85%** of target human or benchmark parity (or Quadratic-Weighted Kappa $\Delta\text{QWK} > 0.15$ for ordinal rubrics).

### 3. Requirements for Permitted LoRA Adapters
If a task satisfies the escalation criteria and a LoRA adapter is authorized:
- **Registry & Provenance**: The adapter must be checked into the versioned model registry with a unique commit SHA, configuration YAML, and cryptographic hash (`lora_hash`).
- **Calibration Coupling**: A dedicated calibration profile must be fitted specifically for the `(backbone_hash, lora_hash)` pair. The serving layer refuses to serve the adapter if the profile hash does not match.
- **Deprecation Path**: Each LoRA adapter is flagged with a review target to be retired once the multi-task tuned foundation backbone (Phase 1C) achieves parity.

---

## Consequences

- **Operational Scalability**: Most customer and operational tasks onboard instantly without GPU training jobs.
- **Preventing Adapter Sprawl**: Engineering effort is focused on improving general rubric conditioning and multi-task tuning rather than proliferating hundreds of micro-adapters.
- **Strict Compliance**: Auditing and deployment overhead are bounded.
