[CmdletBinding()]
param(
    [string]$OutputPath = "test-results/arcade-validation-record.md",
    [string]$LogPath = "test-results/arcade-exhaustive-pytest.txt",
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE} $Command $($Arguments -join ' ')"
    }
}

function Get-TrimmedOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE $Command $($Arguments -join ' ')"
    }

    return ($output | Out-String).Trim()
}

# Resolve the repository root even when this script lives in scripts/.
$repoRoot = Get-TrimmedOutput -Command "git" -Arguments @(
    "rev-parse",
    "--show-toplevel"
)

Set-Location $repoRoot

$testFile = "tests/test_arcade_shooter_exhaustive.py"

if (-not (Test-Path $testFile)) {
    throw "Required test file not found: $testFile"
}

# Confirm that the exhaustive test exists in HEAD, rather than only in the
# working tree or staging area.
& git ls-files --error-unmatch $testFile *> $null
if ($LASTEXITCODE -ne 0) {
    throw "$testFile is not committed. Commit it before generating a publication record."
}

& git diff --quiet HEAD -- $testFile
if ($LASTEXITCODE -ne 0) {
    throw "$testFile differs from the committed version. Commit or restore it first."
}

& git diff --cached --quiet HEAD -- $testFile
if ($LASTEXITCODE -ne 0) {
    throw "$testFile has staged changes not contained in HEAD. Commit them first."
}

$commitSha = Get-TrimmedOutput -Command "git" -Arguments @(
    "rev-parse",
    "HEAD"
)

# Verify that the selected Python interpreter works.
$pythonVersion = Get-TrimmedOutput -Command $PythonCommand -Arguments @(
    "-c",
    "import platform; print(platform.python_version())"
)

$pytestVersion = Get-TrimmedOutput -Command $PythonCommand -Arguments @(
    "-c",
    "import pytest; print(pytest.__version__)"
)

$numpyVersion = Get-TrimmedOutput -Command $PythonCommand -Arguments @(
    "-c",
    "import numpy; print(numpy.__version__)"
)

$zeroModelVersion = Get-TrimmedOutput -Command $PythonCommand -Arguments @(
    "-c",
    "import importlib.metadata as m; print(m.version('zeromodel'))"
)

# Compile the public arcade fixture and print its canonical artifact ID.
$artifactPython = @'
import importlib.util
from pathlib import Path
import sys

path = Path("examples/arcade_shooter_policy.py").resolve()
spec = importlib.util.spec_from_file_location("arcade_shooter_policy", path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {path}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

print(module.compile_policy_artifact().artifact_id)
'@

$artifactId = Get-TrimmedOutput -Command $PythonCommand -Arguments @(
    "-c",
    $artifactPython
)

$outputFile = Join-Path $repoRoot $OutputPath
$logFile = Join-Path $repoRoot $LogPath

New-Item -ItemType Directory -Force -Path (Split-Path $outputFile -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $logFile -Parent) | Out-Null

$testCommandDisplay = "$PythonCommand -m pytest $testFile -q -s"

Write-Host ""
Write-Host "Running exhaustive arcade validation..." -ForegroundColor Cyan
Write-Host $testCommandDisplay
Write-Host ""

$pytestOutput = & $PythonCommand -m pytest $testFile -q -s 2>&1
$pytestExitCode = $LASTEXITCODE

$pytestOutput | Tee-Object -FilePath $logFile | ForEach-Object {
    Write-Host $_
}

if ($pytestExitCode -ne 0) {
    throw "Exhaustive arcade validation failed. See: $logFile"
}

$pytestText = ($pytestOutput | Out-String)

$resultMatch = [regex]::Match(
    $pytestText,
    '(?<result>\d+\s+passed(?:,\s+\d+\s+\w+)*\s+in\s+[0-9.]+s)'
)

if (-not $resultMatch.Success) {
    throw "Tests passed, but the pytest summary could not be parsed from: $logFile"
}

$testResult = $resultMatch.Groups["result"].Value

$record = @"
repository commit:
$commitSha

Python:
$pythonVersion

pytest:
$pytestVersion

NumPy:
$numpyVersion

ZeroModel:
$zeroModelVersion

artifact:
$artifactId

test command:
$testCommandDisplay

result:
$testResult
"@

$fence = '```'
$markdown = "$($fence)text`r`n$record`r`n$fence"

Set-Content -Path $outputFile -Value $markdown -Encoding UTF8

Write-Host ""
Write-Host "Validation record generated:" -ForegroundColor Green
Write-Host $outputFile
Write-Host ""
Write-Host $markdown
