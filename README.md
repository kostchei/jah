# jev-at-home

`jev-at-home` is a local direct-logit decision service. The architecture and evaluation contract live in [JEV_AT_HOME_SPEC.md](JEV_AT_HOME_SPEC.md).

## M0 quick start (Windows + NVIDIA GPU)

```powershell
.\scripts\setup_gpu.ps1
.venv\Scripts\python.exe -m pytest
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m jah.compat
.venv\Scripts\python.exe -m jah.diagnostics.order_bias
```

The setup script creates an isolated Python 3.12 environment and installs the CUDA 12.8 PyTorch wheel. The compatibility command uses the pinned model revision, verifies answer labels at the real tokenizer boundary, loads the BF16 checkpoint on CUDA, evaluates all 50 M0 cases without calling `generate()`, and writes evidence under `artifacts/m0/`. The order-bias command runs the frozen M0 position/wording diagnostic and writes `artifacts/m0/order_bias.json`.

The first model run downloads approximately 9.3 GB into the Hugging Face cache. Raw evaluation states are synthetic; no customer data is included.

See [M0_RESULTS.md](M0_RESULTS.md) for the milestone decision and limitations.

## M1 service and evaluation workflow

M1 adds the typed HTTP/Python contract, a bounded reference scheduler, and paired
direct-logit/constrained-generation evaluation tooling. Start the local service only after the
pinned model is installed:

```powershell
.venv\Scripts\jah-serve.exe --model-config configs/models/qwen3.5-4b.yaml
```

Validate the reviewed dataset and deterministic source/near-duplicate splits before any model
run:

```powershell
.venv\Scripts\jah-eval.exe validate
```

That command currently exits nonzero by design: the independently reviewed 1,000-decision M1
suite has not been supplied. See [M1_STATUS.md](M1_STATUS.md) for the exact implemented surface,
data intake contract, evaluation commands, and remaining exit criteria.
