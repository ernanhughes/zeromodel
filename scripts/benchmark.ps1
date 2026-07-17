param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",

    [string]$OutputDirectory = "build\visual-phase-one-local",

    [switch]$AllowModelDownload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$out = $OutputDirectory
New-Item -ItemType Directory -Force $out | Out-Null

$arguments = @(
    "examples/arcade_visual_address_benchmark.py",
    "--encoder", "dinov2",
    "--device", $Device,
    "--variants-per-family", "3",
    "--include-traces",
    "--output-dir", $out
)
if (-not $AllowModelDownload) {
    $arguments += "--local-files-only"
}

$command = "python " + (($arguments | ForEach-Object {
    if ($_ -match '\s') {
        '"' + ($_ -replace '"', '\"') + '"'
    }
    else {
        $_
    }
}) -join " ")

$command | Set-Content -Encoding utf8 "$out\command.txt"
$arguments | ConvertTo-Json | Set-Content -Encoding utf8 "$out\argv.json"
$env:ZEROMODEL_BENCHMARK_COMMAND = $command

$gitSha = (& git rev-parse HEAD).Trim()
$gitBranch = (& git branch --show-current).Trim()
$dirtyOutput = (& git status --porcelain)
$gitDirty = [bool]($dirtyOutput)

$pythonProbe = @'
import json
import os
import platform
import sys

payload = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
}
try:
    import numpy
    payload["numpy_version"] = numpy.__version__
except Exception as exc:
    payload["numpy_error"] = repr(exc)
try:
    import torch
    payload["torch_version"] = torch.__version__
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["cuda_version"] = torch.version.cuda
    payload["cudnn_version"] = (
        None if not torch.backends.cudnn.is_available()
        else torch.backends.cudnn.version()
    )
    payload["gpu_names"] = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]
except Exception as exc:
    payload["torch_error"] = repr(exc)
try:
    import transformers
    payload["transformers_version"] = transformers.__version__
except Exception as exc:
    payload["transformers_error"] = repr(exc)
try:
    import huggingface_hub
    payload["huggingface_hub_version"] = huggingface_hub.__version__
except Exception as exc:
    payload["huggingface_hub_error"] = repr(exc)
print(json.dumps(payload, indent=2, sort_keys=True))
'@
& python -c $pythonProbe |
    Set-Content -Encoding utf8 "$out\environment.json"

$started = Get-Date

& python @arguments 2>&1 |
    Tee-Object -FilePath "$out\dino-full-run.log"

$exitCode = $LASTEXITCODE
$finished = Get-Date

$expectedReport = Join-Path $out "arcade-visual-phase-one.json"
$reportExists = Test-Path $expectedReport
$reportSha256 = $null
if ($reportExists) {
    $reportSha256 = (Get-FileHash -Algorithm SHA256 $expectedReport).Hash.ToLowerInvariant()
}

$runManifest = [ordered]@{
    version = "zeromodel-visual-benchmark-run-manifest/v1"
    git_commit_sha = $gitSha
    git_branch = $gitBranch
    git_worktree_dirty = $gitDirty
    command = $command
    argv = $arguments
    device = $Device
    allow_model_download = [bool]$AllowModelDownload
    started_utc = $started.ToUniversalTime().ToString("o")
    finished_utc = $finished.ToUniversalTime().ToString("o")
    elapsed_seconds = ($finished - $started).TotalSeconds
    exit_code = $exitCode
    report_path = $expectedReport
    report_exists = $reportExists
    report_file_sha256 = $reportSha256
}

$runManifest |
    ConvertTo-Json -Depth 8 |
    Set-Content -Encoding utf8 "$out\run-manifest.json"

@"
started_utc=$($runManifest.started_utc)
finished_utc=$($runManifest.finished_utc)
elapsed_seconds=$($runManifest.elapsed_seconds)
exit_code=$exitCode
device=$Device
git_commit_sha=$gitSha
git_branch=$gitBranch
git_worktree_dirty=$gitDirty
report_exists=$reportExists
report_file_sha256=$reportSha256
"@ | Set-Content -Encoding utf8 "$out\runtime.txt"

if ($exitCode -ne 0) {
    throw "Full DINO benchmark failed with exit code $exitCode"
}
if (-not $reportExists) {
    throw "Full DINO benchmark did not produce $expectedReport"
}

Write-Host "Benchmark evidence written to $out"
Write-Host "Report SHA-256: $reportSha256"
