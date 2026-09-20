# jev-at-home

`jev-at-home` is a local direct-logit decision service. The architecture and evaluation contract
live in [JEV_AT_HOME_SPEC.md](JEV_AT_HOME_SPEC.md).

## Where the evidence lives

[EVIDENCE.md](EVIDENCE.md) is generated from the recorded artifacts by `jah-report generate` and
is the only place gate verdicts are stated. Status documents describe what was run; they do not
assert verdicts of their own. `jah-report check` fails when the two drift apart.

[REMEDIATION_PLAN.md](REMEDIATION_PLAN.md) lists the gaps between what has been measured and what
the specification requires, in the order they are being closed.

## Setup (Windows + NVIDIA GPU)

```powershell
.\scripts\setup_gpu.ps1
.venv\Scripts\python.exe -m pytest
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m jah.report check
```

The setup script creates an isolated Python 3.12 environment and installs the CUDA 12.8 PyTorch
wheel. The first model run downloads approximately 9.3 GB into the Hugging Face cache.

### Test tiers

The default suite scores mocked tensors and runs in seconds. Properties that only exist with the
real weights resident — tokenizer label boundaries, last-token indexing, prefix-cache agreement,
adapter attachment — are in the `gpu` tier and are excluded by default:

```powershell
.venv\Scripts\python.exe -m pytest -m gpu
```

## Compatibility and diagnostics

```powershell
.venv\Scripts\python.exe -m jah.compat
.venv\Scripts\python.exe -m jah.diagnostics.order_bias
```

`jah.compat` uses the pinned model revision, verifies answer labels at the real tokenizer
boundary, loads the BF16 checkpoint on CUDA, evaluates all 50 M0 cases without calling
`generate()`, and writes evidence under `artifacts/m0/`. See [M0_RESULTS.md](M0_RESULTS.md).

## Service

```powershell
.venv\Scripts\jah-serve.exe --model-config configs/models/qwen3.5-4b.yaml
```

Profiles load from `configs/profiles/public/` by default. None of them carry validated acceptance
evidence, so every decision returns `uncalibrated` and `review` until a profile passes the gates
in `has_validated_evidence()`.

## Evaluation

All `jah-eval` defaults resolve to the public human benchmark
(`evals/data/public/suite.jsonl`, `configs/evals/public-suite.yaml`, split seed `jah-public-v1`).

```powershell
# Validate the suite and freeze split assignments.
.venv\Scripts\jah-eval.exe validate --output artifacts/public/suite-validation.json

# Score one split with one backend.
.venv\Scripts\jah-eval.exe run --backend direct --split development `
  --workload banking77-16-intent-v1 `
  --output artifacts/public/direct-development.json `
  --predictions artifacts/public/direct-development.jsonl

# Optimization equivalence: optimized path against the single-item reference.
.venv\Scripts\jah-eval.exe equivalence --output artifacts/m3/equivalence.json

# Warm HTTP latency protocol: 20 warmup, 200 measured, concurrency 1.
.venv\Scripts\jah-benchmark.exe --request evals/fixtures/benchmark-2048-16b.json `
  --output artifacts/m3/http-latency.json
```

A `--smoke` benchmark run records `protocol_compliant: false` and can never pass the latency gate.

`evals/data/m1-suite.jsonl` is a smoke fixture only; its labels came from another model and
cannot measure decision quality (see [M1_STATUS.md](M1_STATUS.md)).

## Status documents

[M0_RESULTS.md](M0_RESULTS.md) · [M1_STATUS.md](M1_STATUS.md) · [M2_STATUS.md](M2_STATUS.md) ·
[M3_STATUS.md](M3_STATUS.md) · [M4_STATUS.md](M4_STATUS.md) · [WALKTHROUGH.md](WALKTHROUGH.md)
