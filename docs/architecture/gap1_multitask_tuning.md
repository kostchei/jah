# Gap 1: Multi-Task Decision Tuning Architecture Specification (Phase 1C)

- **Status**: Proposed / Architecture Specification
- **Owner**: JAH Architecture & ML Engineering
- **Target Horizon**: 1 Quarter (following Phase 1A/1B completion)
- **Reference**: `D:\Code\ash-rpg\claude_gap.md` (Gap 1)

---

## 1. Context and Problem Statement

JAH currently employs a general-purpose chat foundation backbone (`Qwen/Qwen3.5-4B`). While nominal intent classification (`banking77`) and factuality ranking (`wikiqa`) perform reliably zero-shot, complex rubric-heavy evaluation (such as ASAP essay scoring) exhibits severe performance degradation zero-shot (collapsing to ≈35% accuracy) unless a task-specific LoRA adapter is trained.

Phase 1A diagnostic probes separate rubric comprehension from ordinal mapping limitations. Phase 1B establishes a runtime rubric conditioning layer with a canonical prompt schema and exemplar retrieval. Phase 1C specifies the multi-task decision tuning architecture required to teach the foundation backbone to reliably interpret arbitrary runtime rubrics, criteria boundaries, and tie-break rules zero-shot without fine-tuning per-task weights.

---

## 2. Multi-Task Training Mixture & Data Composition

All training instances are converted into the canonical Phase 1B decision prompt schema (`decision-prompt-v2`), ensuring train-serve representation parity.

### 2.1 Corpus Composition by Task Family

| Task Family | Representative Datasets | Nature of Task | Mixture Weight Cap |
| :--- | :--- | :--- | :--- |
| **Intent Classification** | Banking77, MASSIVE, CLINC150 | Multi-class short-text routing (10–77 labels) | **20%** |
| **Factuality & Ranking** | WikiQA, MS MARCO, TREC-QA | Pairwise & multi-candidate relevancy ranking | **20%** |
| **Natural Language Inference** | MNLI, QNLI, RTE, SNLI | Textual entailment, contradiction, neutrality | **20%** |
| **Ordinal & Rubric Scoring** | ASAP (Automated Student Assessment Prize, prompts 1–8), CoNLL/EFL rubrics | Multi-level criteria boundaries, trait-based scoring | **25%** |
| **Enterprise Adjudicated Traffic** | Historical audit logs, compliance checks, support escalations | Structured decision records with human adjudication | **15%** |

### 2.2 Mixture Weight Balancing & Capping

To prevent large uniform datasets (e.g., MNLI or high-volume intent logs) from dominating the gradients:
- **Maximum Mass per Family**: No single task family may exceed **25%** of the total sampled batch volume.
- **Sampling Strategy**: Temperature-based multi-task sampling:
  $$P(\text{task}_k) \propto \left(\frac{N_k}{\sum_j N_j}\right)^\alpha$$
  where $\alpha = 0.5$ (square-root smoothing), clamped at the family maximum cap.

---

## 3. Generalization & Held-Out Task Partitioning

### 3.1 Held-Out Task Families

Generalization is assessed strictly across **held-out task families** that share no domains, task formulations, or schemas with the training mixture:
1. **Bio-Medical Clinical Triage**: Medical triage priority assessment based on clinical observation notes (unseen terminology and high-stakes ordinal severity).
2. **Multi-Aspect Contract Compliance**: Legal/regulatory clause adherence against explicit criteria descriptions.
3. **Complex Incident Routing & Escalation**: Novel multi-tier IT infrastructure incident classification with hierarchical rubrics.

> [!IMPORTANT]
> A held-out task split is required, not merely a held-out example split. Evaluation on rows from trained tasks measures memorization; evaluation on unseen families measures true zero-shot rubric interpretation.

---

## 4. Robustness & Invariance Augmentation

To prevent the model from overfitting to superficial syntactic artifacts or positional priors, three augmentations are applied during dataset synthesis:

1. **Option-Order & Label Shuffling**:
   - For every sample, the assignment of candidates to labels (`[A]`, `[B]`, etc.) is uniformly permuted at training time.
   - Preserves label-invariance and eliminates letter preference (neutralizing positional bias).
2. **Rubric Paraphrase Perturbation**:
   - Instructions and candidate descriptions are augmented with syntactically distinct, semantically invariant paraphrases (e.g., active vs. passive phrasing, synonymous qualifiers).
3. **State Format Variation**:
   - Evidence states alternate between raw natural language text, canonical compact JSON (`{"key": "value"}`), and indented key-value representations.

---

## 5. Escalation Policy: The LoRA Path

Multi-task decision tuning aims to make zero-shot the default onboarding path. However, specialized domains with proprietary vocabularies or highly strict regulatory constraints may fail zero-shot quality gates.

- **Standard Path**: Tasks onboard with **zero weight training** via configuration, rubric specification, and optional few-shot exemplars in `ExemplarStore`.
- **Escalation Path**: If a newly onboarded task fails the acceptance threshold ($< 85\%$ of human or benchmark parity) after rubric refinement and exemplar retrieval, training a task-specific LoRA adapter is permitted per ADR-09.

---

## 6. Acceptance Gates & Verification Criteria

Phase 1C must satisfy four hard gates before replacing the base backbone:

1. **ASAP Ordinal Scoring Gate**:
   - Zero-shot Quadratic-Weighted Kappa ($\text{QWK}$) across held-out ASAP essay sets must be within **$0.05$** of the task-specific LoRA head baseline ($\text{QWK}_{\text{zero-shot}} \ge \text{QWK}_{\text{LoRA}} - 0.05$).
2. **Regression Invariant Gate**:
   - Zero regression exceeding **$1.0$ point absolute** on established benchmark tasks (`banking77` accuracy $\ge 92.5\%$, `wikiqa` MRR within $1.0\%$).
3. **Held-Out Family Generalization Gate**:
   - Across the three unseen held-out task families, zero-shot accuracy must achieve $\ge \mathbf{85\%}$ of an oracle task-tuned LoRA head.
4. **Operational Zero-Weight Gate**:
   - Standard new task onboarding must require configuration, prompt rubric definition, and exemplar storage only—zero parameter gradient updates.

---

## 7. Risks and Calibration Integration (Gap 3 Alignment)

1. **Calibration Flattening**:
   - Multi-task training alters logit distributions and margin geometry.
   - *Mitigation*: Multi-task checkpoint commits trigger automatic post-hoc recalibration. New `(T, b)` temperature scaling parameters and Clopper-Pearson 95% upper bounds must be computed and verified before deployment per `docs/runbooks/recalibration.md`.
2. **Model Capacity Saturation**:
   - A 4B parameter model may have finite representation capacity for simultaneous general knowledge, complex instruction following, and nuanced ordinal boundary deduction.
   - *Mitigation*: If the 4B backbone cannot meet the ASAP gate ($\Delta\text{QWK} \le 0.05$) after multi-task tuning, the engineering team must formally decide via ADR between scaling to an 8B/14B backbone or maintaining the LoRA registry (ADR-09).
