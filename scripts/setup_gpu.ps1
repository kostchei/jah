$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Push-Location $RepoRoot
try {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        uv venv --python 3.12 .venv
    }
    uv sync --extra inference --extra dev
    & $VenvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(f'torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}')"
}
finally {
    Pop-Location
}

