# M4 adaptation status

Status: **adaptation engine implemented and exercised; development-split lift measured on one of
three workloads; the M4 exit decision is not yet supported.**

The specification's M4 exit decision is "beats the frozen baseline on held-out tasks at acceptable
inference cost". A single workload's development partition is held-out data, but it is not
"tasks", and the ordinal and boolean workloads were not re-measured after adaptation. Gate
verdicts for every artifact referenced here are rendered in [EVIDENCE.md](EVIDENCE.md).

---

## 1. Implemented

Per ADR-05.

### Adaptation engine

- **Module injection** ([src/jah/training/adapter.py](src/jah/training/adapter.py)):
  `inject_lora` walks arbitrary module hierarchies and replaces targeted `nn.Linear` layers
  (`q_proj`, `v_proj`, `lm_head`) with `LoRALinear`. The base model stays frozen
  (`requires_grad = False`), and base layer device and dtype are preserved.
- **Rank-8 adapters** (alpha 16.0): 2,924,544 trainable parameters, 0.064% of the backbone.
- **Candidate-restricted objective**: cross-entropy over tokenizer-verified single-token candidate
  labels only. Hard labels and soft label distributions are both supported.
- **Training runner** (`jah-train`, [src/jah/training/train.py](src/jah/training/train.py)):
  gradient checkpointing, `enable_input_require_grads`, input-length filtering (default 2,048
  tokens), gradient accumulation, gradient clipping, deterministic shuffling, and export of
  `adapter_weights.pt` plus `adapter_manifest.json`.
- **Split discipline**: training rows are filtered to the `train` assignment of the split manifest
  before any workload filter or sampling, so the development partition used below is genuinely
  held out.
- **Inference integration**: `HuggingFaceDirectLogitBackend` attaches adapter weights when
  `adapter_dir` is supplied, and `jah-eval run --adapter-dir <path>` evaluates the adapted model.
  A directory without `adapter_manifest.json` raises rather than silently scoring unadapted
  (ADR-06).

---

## 2. Measured: adapted head against frozen baseline

`banking77-16-intent-v1`, **development** partition, 297 decisions, RTX 4090, BF16.

| Measure | Frozen baseline | Adapted head | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 0.828283 | 0.915825 | +0.087542 |
| Balanced accuracy | 0.821928 | 0.914154 | +0.092226 |
| Macro-F1 | 0.812567 | 0.911191 | +0.098624 |
| Brier score | 0.264617 | 0.152478 | -0.112139 |
| NLL | 0.737278 | 0.444016 | -0.293262 |
| ECE | 0.054125 | 0.054195 | +0.000070 |
| Coverage | 0.0 | 0.0 | unchanged |
| Median request latency (ms) | 89.59 | 88.51 | -1.08 |
| P95 request latency (ms) | 118.13 | 114.48 | -3.65 |

Recorded gate state in both artifacts:

- `gate_brier_passed: true` — both runs beat the prevalence baseline of 0.929201.
- `gate_ece_passed: false` — both runs exceed the 0.05 reliability gate. Adaptation did not
  change this.
- `selective.gate_passed: false`, `samples_sufficient_for_gate: false` — coverage is 0.0 because
  no calibration profile was attached at evaluation time, so the accepted-error gate was never
  exercised.

Macro-F1 of 0.911191 exceeds the 0.85 product target **on development data for one workload**.
The specification requires that target per approved workload on the locked test, which has not
been opened.

Artifacts:
[baseline-banking77-dev.json](artifacts/public/baseline-banking77-dev.json),
[adapted-banking77-dev.json](artifacts/public/adapted-banking77-dev.json),
[training-report.json](artifacts/adapters/qwen3.5-4b-public-head-v1/training-report.json).

---

## 3. Known limits of this result

1. **Final training loss is 5e-05 over 999 samples.** The adapter has largely memorized its
   training partition. The development lift is still real, but the margin between memorization
   and generalization has not been probed with a task or template holdout.
2. **One workload.** The ordinal (`asap2-source-essay-v1`) and boolean
   (`wikiqa-answer-relevance-v1`) workloads were not re-evaluated after adaptation. Both fail
   calibration gates in their frozen state.
3. **No uncertainty interval.** The accuracy difference of +0.0875 carries no bootstrap interval
   grouped by source group, which the specification requires for quality differences.
4. **Contamination unquantified.** See [WALKTHROUGH.md](WALKTHROUGH.md) section 4.

Remediation for items 2 and 3 is scoped in [REMEDIATION_PLAN.md](REMEDIATION_PLAN.md), phases 2
and 3.

---

## 4. Verification

- Default suite: mocked backends, seconds to run.
- `pytest -m gpu`: adapter attachment and manifest failure behavior against the real backbone.
- `ruff check .`
- `jah-report check`: fails if [EVIDENCE.md](EVIDENCE.md) drifts from the artifacts.
