# Recalibration Runbook

See the comprehensive operational guide at [docs/runbooks/recalibration.md](docs/runbooks/recalibration.md).

## Quick Reference Commands

```powershell
# 1. Run drift check
python -m jah.evaluation drift --predictions <predictions.jsonl> --profile configs/profiles/public/<profile>.yaml

# 2. Refit calibration profile on calibration split
python -m jah.evaluation calibrate `
  --dataset evals/data/public/suite.jsonl `
  --suite-config configs/evals/public-suite.yaml `
  --model-config configs/models/qwen3.5-4b.yaml `
  --output configs/profiles/public `
  --summary artifacts/public/calibration-summary.json

# 3. Verify on development split
python -m jah.evaluation run `
  --backend direct `
  --split development `
  --use-workload-profiles `
  --output artifacts/public/post-recalibration-dev.json `
  --predictions artifacts/public/post-recalibration-dev.jsonl
```
