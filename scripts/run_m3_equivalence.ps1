param(
    [ValidateSet("bfloat16", "float16", "float32")]
    [string]$Precision = "bfloat16",
    [ValidateRange(1, 16)]
    [int]$MicrobatchSize = 1,
    [ValidateRange(0.1, 1.0)]
    [double]$ResourceLimit = 0.85,
    [string]$Requests = "evals/fixtures/equivalence",
    [switch]$EnablePrefixCache
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Requests -PathType Container)) {
    throw "Request directory does not exist: $Requests"
}
$gpuStats = & nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or -not $gpuStats) {
    throw "Could not read GPU utilization and free memory; refusing to start evaluation."
}
$gpuValues = $gpuStats.Trim().Split(",")
$gpuUtilization = [int]$gpuValues[0].Trim()
$freeMemoryMiB = [int]$gpuValues[1].Trim()
$minimumFreeMemoryMiB = switch ($Precision) {
    "float32" { 19000 }
    "float16" { 11000 }
    default { 11000 }
}
if ($gpuUtilization -gt 30 -or $freeMemoryMiB -lt $minimumFreeMemoryMiB) {
    throw "GPU is busy or lacks headroom (utilization=${gpuUtilization}%, free=${freeMemoryMiB}MiB; requires <=30% and >=${minimumFreeMemoryMiB}MiB). Retry when available."
}

$stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$uniqueSuffix = [Guid]::NewGuid().ToString("N").Substring(0, 8)
$runName = "${stamp}-${Precision}-mb${MicrobatchSize}-${uniqueSuffix}"
if (-not $EnablePrefixCache) {
    $runName += "-no-prefix"
}
$outputDirectory = Join-Path "artifacts/m3/runs" $runName
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$pythonPath = Join-Path ".venv\Scripts" "python.exe"
$arguments = @(
    "-m", "jah.evaluation",
    "--root", ".",
    "equivalence",
    "--requests", $Requests,
    "--model-config", "configs/models/qwen3.5-4b.yaml",
    "--precision", $Precision,
    "--microbatch-size", "$MicrobatchSize",
    "--resource-limit", "$ResourceLimit",
    "--throttle-ms", "5",
    "--output", (Join-Path $outputDirectory "equivalence.json"),
    "--rows", (Join-Path $outputDirectory "equivalence.jsonl")
)
if (-not $EnablePrefixCache) {
    $arguments += "--disable-prefix-cache"
}

& $pythonPath @arguments
exit $LASTEXITCODE
