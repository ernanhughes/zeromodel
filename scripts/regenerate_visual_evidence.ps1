#requires -Version 7.0
<#
.SYNOPSIS
Regenerate the current ZeroModel visual evidence pipeline from a clean repository state.

.DESCRIPTION
This script performs a fresh, versioned rerun of:

  1. The current Phase 1 arcade visual-address benchmark, including DINOv2.
  2. The corrected System B v2 adjudication.
  3. The repository test suite, unless -SkipTests is supplied.
  4. Environment, command, runtime, and SHA-256 manifest generation.

IMPORTANT:
- This script does NOT overwrite the historical Phase 1 v1 recovery package.
- It writes into a new timestamped directory under build/ by default.
- The current repository generator uses the corrected current protocol and may
  not reproduce the historical v1 dataset or report digests.
- Run from a clean committed revision whenever the result will be cited.

.EXAMPLE
pwsh -File scripts/regenerate_visual_evidence.ps1

.EXAMPLE
pwsh -File scripts/regenerate_visual_evidence.ps1 `
  -Device cuda `
  -LocalFilesOnly `
  -OutputRoot build/visual-regeneration/manual-run

.EXAMPLE
pwsh -File scripts/regenerate_visual_evidence.ps1 -SkipTests
#>

[CmdletBinding()]
param(
    [Parameter()]
    [string]$RepoRoot = (Get-Location).Path,

    [Parameter()]
    [string]$Python = "python",

    [Parameter()]
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",

    [Parameter()]
    [switch]$LocalFilesOnly,

    [Parameter()]
    [ValidateRange(1, 100)]
    [int]$VariantsPerFamily = 3,

    [Parameter()]
    [string]$OutputRoot,

    [Parameter()]
    [switch]$SkipPhaseOne,

    [Parameter()]
    [switch]$SkipSystemB,

    [Parameter()]
    [switch]$SkipTests,

    [Parameter()]
    [switch]$AllowDirty,

    [Parameter()]
    [switch]$CreateZip
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [AllowEmptyString()]
        [string]$Text
    )

    $parent = Split-Path -Parent $Path
    if ($parent) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    [System.IO.File]::WriteAllText($Path, $Text, $script:Utf8NoBom)
}

function ConvertTo-CanonicalJson {
    param(
        [Parameter(Mandatory)]
        $Value
    )

    return ($Value | ConvertTo-Json -Depth 100)
}

function Format-CommandArgument {
    param(
        [Parameter(Mandatory)]
        [string]$Argument
    )

    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }

    return '"' + ($Argument -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
}

function Format-CommandLine {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $parts = @((Format-CommandArgument $FilePath))
    $parts += $Arguments | ForEach-Object { Format-CommandArgument $_ }
    return ($parts -join " ")
}

function Invoke-CapturedProcess {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [Parameter(Mandatory)]
        [string[]]$Arguments,

        [Parameter(Mandatory)]
        [string]$WorkingDirectory,

        [Parameter(Mandatory)]
        [string]$StdoutPath,

        [Parameter(Mandatory)]
        [string]$StderrPath,

        [Parameter(Mandatory)]
        [string]$RuntimePath
    )

    $commandLine = Format-CommandLine -FilePath $FilePath -Arguments $Arguments
    Write-Host ""
    Write-Host "Running:" -ForegroundColor Cyan
    Write-Host "  $commandLine"

    $started = [DateTimeOffset]::UtcNow
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    foreach ($argument in $Arguments) {
        [void]$psi.ArgumentList.Add($argument)
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $psi

    if (-not $process.Start()) {
        throw "Failed to start process: $FilePath"
    }

    # Read both streams asynchronously to avoid deadlocks on verbose processes.
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()

    $process.WaitForExit()

    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()

    $stopwatch.Stop()
    $completed = [DateTimeOffset]::UtcNow

    Write-Utf8NoBom -Path $StdoutPath -Text $stdout
    Write-Utf8NoBom -Path $StderrPath -Text $stderr

    $runtimeLines = @(
        "started_utc=$($started.ToString('o'))"
        "finished_utc=$($completed.ToString('o'))"
        "elapsed_seconds=$([Math]::Round($stopwatch.Elapsed.TotalSeconds, 6))"
        "exit_code=$($process.ExitCode)"
        "command=$commandLine"
    )
    Write-Utf8NoBom -Path $RuntimePath -Text (($runtimeLines -join "`n") + "`n")

    if ($stdout) {
        Write-Host $stdout.TrimEnd()
    }

    if ($stderr) {
        Write-Host $stderr.TrimEnd() -ForegroundColor DarkGray
    }

    if ($process.ExitCode -ne 0) {
        throw "Command failed with exit code $($process.ExitCode): $commandLine"
    }

    return [ordered]@{
        command          = $commandLine
        file             = $FilePath
        argv             = @($FilePath) + $Arguments
        started_utc      = $started.ToString("o")
        completed_utc    = $completed.ToString("o")
        elapsed_seconds  = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 6)
        exit_code        = $process.ExitCode
        stdout_path      = $StdoutPath
        stderr_path      = $StderrPath
        runtime_path     = $RuntimePath
    }
}

function Get-GitValue {
    param(
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $value = & git -C $script:ResolvedRepoRoot @Arguments 2>$null
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    return (($value | Out-String).Trim())
}

function Get-RelativePathPortable {
    param(
        [Parameter(Mandatory)]
        [string]$BasePath,

        [Parameter(Mandatory)]
        [string]$TargetPath
    )

    return [System.IO.Path]::GetRelativePath(
        [System.IO.Path]::GetFullPath($BasePath),
        [System.IO.Path]::GetFullPath($TargetPath)
    ).Replace("\", "/")
}

function Get-FileRecord {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [Parameter(Mandatory)]
        [string]$BasePath
    )

    $item = Get-Item -LiteralPath $FilePath
    $hash = Get-FileHash -LiteralPath $FilePath -Algorithm SHA256

    return [ordered]@{
        path   = Get-RelativePathPortable -BasePath $BasePath -TargetPath $item.FullName
        size   = [int64]$item.Length
        sha256 = $hash.Hash.ToLowerInvariant()
    }
}

function Assert-JsonFile {
    param(
        [Parameter(Mandatory)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Expected JSON file was not generated: $Path"
    }

    try {
        return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
    }
    catch {
        throw "Generated file is not valid JSON: $Path`n$($_.Exception.Message)"
    }
}

# ---------------------------------------------------------------------------
# Resolve and validate repository.
# ---------------------------------------------------------------------------

$script:ResolvedRepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path

$requiredPaths = @(
    "examples/arcade_visual_address_benchmark.py",
    "examples/arcade_visual_system_b_adjudication.py",
    "zeromodel",
    "tests"
)

foreach ($relativePath in $requiredPaths) {
    $candidate = Join-Path $ResolvedRepoRoot $relativePath
    if (-not (Test-Path -LiteralPath $candidate)) {
        throw "Repository path is missing: $candidate"
    }
}

$pythonCommand = Get-Command $Python -ErrorAction Stop
$resolvedPython = $pythonCommand.Source
if (-not $resolvedPython) {
    $resolvedPython = $pythonCommand.Path
}

$gitCommit = Get-GitValue -Arguments @("rev-parse", "HEAD")
if (-not $gitCommit) {
    throw "RepoRoot is not a Git repository: $ResolvedRepoRoot"
}

$gitBranch = Get-GitValue -Arguments @("branch", "--show-current")
if (-not $gitBranch) {
    $gitBranch = "detached"
}

$dirtyOutput = Get-GitValue -Arguments @("status", "--porcelain=v1", "--untracked-files=all")
$dirtyAtStart = [bool]$dirtyOutput

if ($dirtyAtStart -and -not $AllowDirty) {
    Write-Host ""
    Write-Host "Working tree changes:" -ForegroundColor Yellow
    Write-Host $dirtyOutput
    throw "Repository is dirty. Commit/stash changes or rerun with -AllowDirty."
}

# ---------------------------------------------------------------------------
# Output directory.
# ---------------------------------------------------------------------------

if (-not $OutputRoot) {
    $timestamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMdd-HHmmssZ")
    $OutputRoot = Join-Path $ResolvedRepoRoot "build/visual-regeneration/$timestamp"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $ResolvedRepoRoot $OutputRoot
}

$ResolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

if (Test-Path -LiteralPath $ResolvedOutputRoot) {
    $existing = Get-ChildItem -LiteralPath $ResolvedOutputRoot -Force -ErrorAction SilentlyContinue
    if ($existing) {
        throw "Output directory already exists and is not empty: $ResolvedOutputRoot"
    }
}

[System.IO.Directory]::CreateDirectory($ResolvedOutputRoot) | Out-Null

$phaseOneDir = Join-Path $ResolvedOutputRoot "phase-one-current-rerun"
$systemBDir = Join-Path $ResolvedOutputRoot "system-b-v2-current-rerun"
$validationDir = Join-Path $ResolvedOutputRoot "validation"

[System.IO.Directory]::CreateDirectory($phaseOneDir) | Out-Null
[System.IO.Directory]::CreateDirectory($systemBDir) | Out-Null
[System.IO.Directory]::CreateDirectory($validationDir) | Out-Null

# ---------------------------------------------------------------------------
# Resolve compute device.
# ---------------------------------------------------------------------------

$resolvedDevice = $Device
if ($Device -eq "auto") {
    $deviceProbe = & $resolvedPython -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to detect PyTorch device."
    }
    $resolvedDevice = (($deviceProbe | Out-String).Trim())
}

Write-Host ""
Write-Host "ZeroModel visual evidence regeneration" -ForegroundColor Green
Write-Host "Repository: $ResolvedRepoRoot"
Write-Host "Commit:    $gitCommit"
Write-Host "Branch:    $gitBranch"
Write-Host "Python:    $resolvedPython"
Write-Host "Device:    $resolvedDevice"
Write-Host "Output:    $ResolvedOutputRoot"

# ---------------------------------------------------------------------------
# Environment record.
# ---------------------------------------------------------------------------

$pythonEnvironmentCode = @'
import json
import platform
import sys

payload = {
    "python_version": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
}

for name in ("numpy", "torch", "torchvision", "transformers"):
    try:
        module = __import__(name)
        payload[f"{name}_version"] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        payload[f"{name}_version"] = f"NOT_AVAILABLE:{type(exc).__name__}"

try:
    import torch
    payload["cuda_available"] = bool(torch.cuda.is_available())
    payload["cuda_version"] = torch.version.cuda
    payload["cudnn_version"] = (
        torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None
    )
    payload["gpu"] = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    )
except Exception:
    pass

print(json.dumps(payload, indent=2, sort_keys=True))
'@

$pythonEnvironmentRaw = & $resolvedPython -c $pythonEnvironmentCode
if ($LASTEXITCODE -ne 0) {
    throw "Failed to capture Python environment."
}
$pythonEnvironment = ($pythonEnvironmentRaw -join "`n") | ConvertFrom-Json

$environment = [ordered]@{
    captured_utc        = [DateTimeOffset]::UtcNow.ToString("o")
    repository_root     = $ResolvedRepoRoot
    output_root         = $ResolvedOutputRoot
    git_commit          = $gitCommit
    git_branch          = $gitBranch
    dirty_at_start      = $dirtyAtStart
    dirty_paths_at_start = if ($dirtyOutput) { @($dirtyOutput -split "`r?`n") } else { @() }
    powershell_version  = $PSVersionTable.PSVersion.ToString()
    powershell_edition  = $PSVersionTable.PSEdition
    os                  = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    process_architecture = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
    requested_device    = $Device
    resolved_device     = $resolvedDevice
    local_files_only    = [bool]$LocalFilesOnly
    variants_per_family = $VariantsPerFamily
    python              = $pythonEnvironment
}

$environmentPath = Join-Path $ResolvedOutputRoot "environment.json"
Write-Utf8NoBom -Path $environmentPath -Text ((ConvertTo-CanonicalJson $environment) + "`n")

$commands = [System.Collections.Generic.List[object]]::new()
$results = [ordered]@{}

# ---------------------------------------------------------------------------
# Phase 1 current-protocol rerun.
# ---------------------------------------------------------------------------

if (-not $SkipPhaseOne) {
    $phaseOneArguments = @(
        "examples/arcade_visual_address_benchmark.py",
        "--encoder", "dinov2",
        "--device", $resolvedDevice,
        "--variants-per-family", [string]$VariantsPerFamily,
        "--include-traces",
        "--output-dir", $phaseOneDir
    )

    if ($LocalFilesOnly) {
        $phaseOneArguments += "--local-files-only"
    }

    $phaseOneCommandText = Format-CommandLine -FilePath $resolvedPython -Arguments $phaseOneArguments
    Write-Utf8NoBom -Path (Join-Path $phaseOneDir "command.txt") -Text ($phaseOneCommandText + "`n")

    $phaseOneArgv = [ordered]@{
        executable = $resolvedPython
        arguments  = $phaseOneArguments
        argv       = @($resolvedPython) + $phaseOneArguments
    }
    Write-Utf8NoBom `
        -Path (Join-Path $phaseOneDir "argv.json") `
        -Text ((ConvertTo-CanonicalJson $phaseOneArgv) + "`n")

    $phaseOneRun = Invoke-CapturedProcess `
        -FilePath $resolvedPython `
        -Arguments $phaseOneArguments `
        -WorkingDirectory $ResolvedRepoRoot `
        -StdoutPath (Join-Path $phaseOneDir "phase-one-summary.json") `
        -StderrPath (Join-Path $phaseOneDir "dino-full-run.log") `
        -RuntimePath (Join-Path $phaseOneDir "runtime.txt")

    $commands.Add($phaseOneRun)

    $phaseOnePayloadPath = Join-Path $phaseOneDir "arcade-visual-phase-one.json"
    $phaseOnePayload = Assert-JsonFile -Path $phaseOnePayloadPath
    $phaseOneSummary = Assert-JsonFile -Path (Join-Path $phaseOneDir "phase-one-summary.json")

    $results.phase_one = [ordered]@{
        output_directory = Get-RelativePathPortable -BasePath $ResolvedOutputRoot -TargetPath $phaseOneDir
        payload_path     = Get-RelativePathPortable -BasePath $ResolvedOutputRoot -TargetPath $phaseOnePayloadPath
        dataset_digest   = $phaseOneSummary.dataset_digest
        report_digest    = $phaseOneSummary.report_digest
        observation_count = $phaseOneSummary.observation_count
        validation_status = $phaseOneSummary.validation_status
        source_scope     = $phaseOnePayload.dataset_manifest.source_scope
    }
}

# ---------------------------------------------------------------------------
# System B v2 current-protocol rerun.
# ---------------------------------------------------------------------------

if (-not $SkipSystemB) {
    $systemBArguments = @(
        "examples/arcade_visual_system_b_adjudication.py",
        "--output-dir", $systemBDir,
        "--variants-per-family", [string]$VariantsPerFamily
    )

    $systemBCommandText = Format-CommandLine -FilePath $resolvedPython -Arguments $systemBArguments
    Write-Utf8NoBom -Path (Join-Path $systemBDir "outer-command.txt") -Text ($systemBCommandText + "`n")

    $systemBArgv = [ordered]@{
        executable = $resolvedPython
        arguments  = $systemBArguments
        argv       = @($resolvedPython) + $systemBArguments
    }
    Write-Utf8NoBom `
        -Path (Join-Path $systemBDir "outer-argv.json") `
        -Text ((ConvertTo-CanonicalJson $systemBArgv) + "`n")

    $systemBRun = Invoke-CapturedProcess `
        -FilePath $resolvedPython `
        -Arguments $systemBArguments `
        -WorkingDirectory $ResolvedRepoRoot `
        -StdoutPath (Join-Path $systemBDir "system-b-console.json") `
        -StderrPath (Join-Path $systemBDir "system-b-stderr.log") `
        -RuntimePath (Join-Path $systemBDir "outer-runtime.txt")

    $commands.Add($systemBRun)

    $systemBSummary = Assert-JsonFile -Path (Join-Path $systemBDir "final-summary.json")
    $systemBRunManifest = Assert-JsonFile -Path (Join-Path $systemBDir "run-manifest.json")
    $systemBDatasetManifest = Assert-JsonFile -Path (Join-Path $systemBDir "dataset-manifest.json")

    $results.system_b = [ordered]@{
        output_directory    = Get-RelativePathPortable -BasePath $ResolvedOutputRoot -TargetPath $systemBDir
        outcome             = $systemBSummary.outcome
        usefulness_status   = $systemBSummary.usefulness_status
        next_action         = $systemBSummary.next_action
        selected_quantile   = $systemBSummary.selected_quantile
        dataset_digest      = $systemBRunManifest.dataset_digest
        selection_digest    = $systemBRunManifest.selection_digest
        calibration_digest  = $systemBSummary.calibration_digest
        run_manifest_digest = $systemBSummary.run_manifest_digest
        source_scope        = $systemBDatasetManifest.source_scope
        final_metrics       = $systemBSummary.final_metrics
    }
}

# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------

if (-not $SkipTests) {
    $pytestArguments = @("-m", "pytest", "-q")

    $testRun = Invoke-CapturedProcess `
        -FilePath $resolvedPython `
        -Arguments $pytestArguments `
        -WorkingDirectory $ResolvedRepoRoot `
        -StdoutPath (Join-Path $validationDir "pytest-stdout.txt") `
        -StderrPath (Join-Path $validationDir "pytest-stderr.txt") `
        -RuntimePath (Join-Path $validationDir "pytest-runtime.txt")

    $commands.Add($testRun)
    $results.validation = [ordered]@{
        full_pytest_exit_code = $testRun.exit_code
        full_pytest_stdout = Get-RelativePathPortable `
            -BasePath $ResolvedOutputRoot `
            -TargetPath (Join-Path $validationDir "pytest-stdout.txt")
        full_pytest_stderr = Get-RelativePathPortable `
            -BasePath $ResolvedOutputRoot `
            -TargetPath (Join-Path $validationDir "pytest-stderr.txt")
    }
}

# ---------------------------------------------------------------------------
# Final bundle manifest and checksums.
# ---------------------------------------------------------------------------

$completedUtc = [DateTimeOffset]::UtcNow.ToString("o")
$dirtyAfterRunOutput = Get-GitValue -Arguments @("status", "--porcelain=v1", "--untracked-files=all")

$manifestPath = Join-Path $ResolvedOutputRoot "regeneration-manifest.json"
$checksumsPath = Join-Path $ResolvedOutputRoot "checksums.sha256"

$filesBeforeManifest = Get-ChildItem -LiteralPath $ResolvedOutputRoot -File -Recurse |
    Where-Object {
        $_.FullName -ne $manifestPath -and
        $_.FullName -ne $checksumsPath
    } |
    Sort-Object FullName |
    ForEach-Object {
        Get-FileRecord -FilePath $_.FullName -BasePath $ResolvedOutputRoot
    }

$regenerationManifest = [ordered]@{
    version              = "zeromodel-visual-regeneration-bundle/v1"
    purpose              = "fresh current-protocol visual evidence rerun"
    historical_v1_replacement = $false
    warning              = "This bundle must not overwrite or be represented as the historical Phase 1 v1 run."
    started_from_commit   = $gitCommit
    started_from_branch   = $gitBranch
    dirty_at_start        = $dirtyAtStart
    completed_utc         = $completedUtc
    dirty_after_run       = [bool]$dirtyAfterRunOutput
    dirty_paths_after_run = if ($dirtyAfterRunOutput) { @($dirtyAfterRunOutput -split "`r?`n") } else { @() }
    resolved_device       = $resolvedDevice
    variants_per_family   = $VariantsPerFamily
    commands              = $commands
    results               = $results
    files                 = @($filesBeforeManifest)
    digest_scope_note     = "File SHA-256 values are computed over exact bytes stored in this generated bundle."
}

Write-Utf8NoBom -Path $manifestPath -Text ((ConvertTo-CanonicalJson $regenerationManifest) + "`n")

$allFilesForChecksums = Get-ChildItem -LiteralPath $ResolvedOutputRoot -File -Recurse |
    Where-Object { $_.FullName -ne $checksumsPath } |
    Sort-Object FullName

$checksumLines = foreach ($file in $allFilesForChecksums) {
    $record = Get-FileRecord -FilePath $file.FullName -BasePath $ResolvedOutputRoot
    "$($record.sha256)  $($record.path)"
}
Write-Utf8NoBom -Path $checksumsPath -Text (($checksumLines -join "`n") + "`n")

# Verify checksums immediately.
foreach ($line in $checksumLines) {
    $parts = $line -split '\s{2}', 2
    if ($parts.Count -ne 2) {
        throw "Invalid checksum line: $line"
    }
    $expectedHash = $parts[0]
    $relativePath = $parts[1]
    $actualPath = Join-Path $ResolvedOutputRoot $relativePath
    $actualHash = (Get-FileHash -LiteralPath $actualPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "Checksum verification failed: $relativePath"
    }
}

if ($CreateZip) {
    $zipPath = "$ResolvedOutputRoot.zip"
    if (Test-Path -LiteralPath $zipPath) {
        throw "ZIP path already exists: $zipPath"
    }
    Compress-Archive -LiteralPath $ResolvedOutputRoot -DestinationPath $zipPath -CompressionLevel Optimal
    Write-Host "ZIP:       $zipPath"
}

Write-Host ""
Write-Host "Regeneration completed successfully." -ForegroundColor Green
Write-Host "Bundle:    $ResolvedOutputRoot"
Write-Host "Manifest:  $manifestPath"
Write-Host "Checksums: $checksumsPath"

if ($results.phase_one) {
    Write-Host ""
    Write-Host "Phase 1 current rerun:" -ForegroundColor Cyan
    Write-Host "  dataset digest: $($results.phase_one.dataset_digest)"
    Write-Host "  report digest:  $($results.phase_one.report_digest)"
    Write-Host "  source scope:   $($results.phase_one.source_scope)"
}

if ($results.system_b) {
    Write-Host ""
    Write-Host "System B current rerun:" -ForegroundColor Cyan
    Write-Host "  outcome:            $($results.system_b.outcome)"
    Write-Host "  dataset digest:     $($results.system_b.dataset_digest)"
    Write-Host "  selection digest:   $($results.system_b.selection_digest)"
    Write-Host "  calibration digest: $($results.system_b.calibration_digest)"
    Write-Host "  next action:        $($results.system_b.next_action)"
}
