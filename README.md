# jev-at-home

`jev-at-home` is a local direct-logit decision service. The architecture and evaluation contract live in [JEV_AT_HOME_SPEC.md](JEV_AT_HOME_SPEC.md).

## M0 quick start (Windows + NVIDIA GPU)

```powershell
.\scripts\setup_gpu.ps1
.venv\Scripts\python.exe -m pytest
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m jah.compat
```

The setup script creates an isolated Python 3.12 environment and installs the CUDA 12.8 PyTorch wheel. The compatibility command uses the pinned model revision, verifies answer labels at the real tokenizer boundary, loads the BF16 checkpoint on CUDA, evaluates all 50 M0 cases without calling `generate()`, and writes evidence under `artifacts/m0/`.

The first model run downloads approximately 9.3 GB into the Hugging Face cache. Raw evaluation states are synthetic; no customer data is included.

See [M0_RESULTS.md](M0_RESULTS.md) for the milestone decision and limitations.

