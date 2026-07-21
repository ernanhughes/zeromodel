[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RepositoryPath,
    [Parameter(Mandatory = $true)][string]$ExpectedCommit,
    [string]$ExpectedBranch = "codex/video-finalization-integration-validation",
    [string]$Python = "python",
    [switch]$RemoveSuccessfulFixtures
)

$ErrorActionPreference = "Stop"
$ForbiddenStage8Path = [IO.Path]::GetFullPath("C:\Projects\zeromodel-stage8")
$ExpectedCommitPattern = "^[0-9a-fA-F]{40}$"

function ConvertTo-NativeArgument {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)
    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Test-PathWithin {
    param(
        [Parameter(Mandatory = $true)][string]$Child,
        [Parameter(Mandatory = $true)][string]$Parent
    )
    $childPath = [IO.Path]::GetFullPath($Child).TrimEnd('\')
    $parentPath = [IO.Path]::GetFullPath($Parent).TrimEnd('\')
    return $childPath.Equals($parentPath, [StringComparison]::OrdinalIgnoreCase) -or
        $childPath.StartsWith($parentPath + '\', [StringComparison]::OrdinalIgnoreCase)
}

function Invoke-GitText {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $value = & git -C $script:Repository @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed: $value"
    }
    return (($value | Out-String).Trim())
}

function Write-RunState {
    param(
        [Parameter(Mandatory = $true)][string]$Status,
        [AllowNull()][string]$CurrentGroup,
        [AllowNull()][Nullable[int]]$GroupPid
    )
    [ordered]@{
        version = "zeromodel-video-finalization-integration-run-state/v1"
        status = $Status
        runner_pid = $PID
        group_pid = $GroupPid
        current_group = $CurrentGroup
        repository = $script:Repository
        expected_branch = $ExpectedBranch
        expected_commit = $ExpectedCommit.ToLowerInvariant()
        updated_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $script:RunStatePath -Encoding UTF8
}

if ($ExpectedCommit -notmatch $ExpectedCommitPattern) {
    throw "ExpectedCommit must be a full 40-character Git SHA."
}
$Repository = [IO.Path]::GetFullPath($RepositoryPath)
if (-not (Test-Path -LiteralPath $Repository -PathType Container)) {
    throw "RepositoryPath is not an existing directory: $Repository"
}
if (Test-PathWithin -Child $Repository -Parent $ForbiddenStage8Path) {
    throw "The Stage 8 repository is forbidden for this validation."
}

$branch = Invoke-GitText -Arguments @("branch", "--show-current")
$head = Invoke-GitText -Arguments @("rev-parse", "HEAD")
$status = Invoke-GitText -Arguments @("status", "--short")
if ($branch -cne $ExpectedBranch) {
    throw "Wrong branch. Expected '$ExpectedBranch', found '$branch'."
}
if ($head -cne $ExpectedCommit.ToLowerInvariant()) {
    throw "Wrong commit. Expected '$ExpectedCommit', found '$head'."
}
if ($status.Length -ne 0) {
    throw "The validation worktree must be clean.`n$status"
}

$ValidationRoot = Join-Path ([IO.Path]::GetTempPath()) (
    "zeromodel-finalization-validation-" + [Guid]::NewGuid().ToString("N")
)
$ValidationRoot = [IO.Path]::GetFullPath($ValidationRoot)
if (Test-PathWithin -Child $ValidationRoot -Parent $ForbiddenStage8Path) {
    throw "Validation root resolved under the forbidden Stage 8 path."
}
New-Item -ItemType Directory -Path $ValidationRoot | Out-Null
Set-Content -LiteralPath (Join-Path $ValidationRoot ".zeromodel-synthetic-validation") `
    -Value "synthetic-only" -Encoding ASCII
$RunStatePath = Join-Path $ValidationRoot "run-state.json"
$SummaryJsonPath = Join-Path $ValidationRoot "summary.json"
$SummaryMarkdownPath = Join-Path $ValidationRoot "summary.md"
$LogsPath = Join-Path $ValidationRoot "logs"
$FixturesPath = Join-Path $ValidationRoot "fixtures"
New-Item -ItemType Directory -Path $LogsPath, $FixturesPath | Out-Null

$groups = @(
    [ordered]@{ Name = "01-schema-authority"; PredictedSeconds = 8; TimeoutSeconds = 60; Tests = @("tests/integration/test_video_finalization_schema_authority.py") },
    [ordered]@{ Name = "02-sqlite-transactions"; PredictedSeconds = 15; TimeoutSeconds = 90; Tests = @("tests/integration/test_video_finalization_sqlite_concurrency.py") },
    [ordered]@{ Name = "03-authorized-observations"; PredictedSeconds = 10; TimeoutSeconds = 60; Tests = @("tests/integration/test_video_finalization_authorized_observations.py") },
    [ordered]@{ Name = "04-historical-authority"; PredictedSeconds = 10; TimeoutSeconds = 60; Tests = @("tests/test_video_final_historical_authority.py", "tests/integration/test_video_finalization_historical_evaluator.py::test_historical_manifest_bindings_reject_declared_mismatch", "tests/integration/test_video_finalization_historical_evaluator.py::test_historical_authority_rejects_relative_paths") },
    [ordered]@{ Name = "05-evaluator"; PredictedSeconds = 8; TimeoutSeconds = 60; Tests = @("tests/test_video_final_access_kernel.py::test_evaluator_is_order_independent_and_digest_bearing", "tests/test_video_final_access_kernel.py::test_evaluator_is_isolated_from_hostile_global_decimal_contexts", "tests/test_video_final_access_kernel.py::test_decimal_aggregate_threshold_equality_remains_inclusive", "tests/test_video_final_access_kernel.py::test_pass_fail_and_indeterminate_are_deterministic", "tests/test_video_final_access_kernel.py::test_evaluator_rejects_unsupported_decision_rule_shapes", "tests/integration/test_video_finalization_historical_evaluator.py::test_duplicate_evidence_identity_is_rejected", "tests/integration/test_video_finalization_historical_evaluator.py::test_missing_or_undeclared_evidence_provider_is_rejected", "tests/integration/test_video_finalization_historical_evaluator.py::test_unknown_protocol_keys_fail_before_evaluation", "tests/integration/test_video_finalization_historical_evaluator.py::test_incomplete_required_evidence_is_indeterminate") },
    [ordered]@{ Name = "06-publication"; PredictedSeconds = 20; TimeoutSeconds = 90; Tests = @("tests/integration/test_video_finalization_executor_publication.py", "tests/test_video_final_publication.py") },
    [ordered]@{ Name = "07-failure-injection"; PredictedSeconds = 20; TimeoutSeconds = 90; Tests = @("tests/integration/test_video_finalization_failure_injection.py") },
    [ordered]@{ Name = "08-reconstruction"; PredictedSeconds = 35; TimeoutSeconds = 120; Tests = @("tests/integration/test_video_finalization_reconstruction.py") },
    [ordered]@{ Name = "09-cli-scripts"; PredictedSeconds = 45; TimeoutSeconds = 120; Tests = @("tests/integration/test_video_finalization_cli_scripts.py") },
    [ordered]@{ Name = "10-package-boundary"; PredictedSeconds = 90; TimeoutSeconds = 180; Tests = @("tests/integration/test_video_finalization_package_boundary.py") }
)

$startedUtc = [DateTime]::UtcNow
$results = [Collections.Generic.List[object]]::new()
$overallStatus = "passed"
Write-Host "Synthetic validation root: $ValidationRoot"
Write-Host "Predicted total: $((($groups | Measure-Object PredictedSeconds -Sum).Sum)) seconds"
Write-RunState -Status "running" -CurrentGroup $null -GroupPid $null

foreach ($group in $groups) {
    $groupFixture = Join-Path $FixturesPath $group.Name
    $stdoutPath = Join-Path $LogsPath ($group.Name + ".stdout.log")
    $stderrPath = Join-Path $LogsPath ($group.Name + ".stderr.log")
    $arguments = @(
        "-m", "pytest", "--run-integration", "-q", "--maxfail=1",
        "--basetemp", $groupFixture
    ) + $group.Tests
    $display = (ConvertTo-NativeArgument $Python) + " " + (
        ($arguments | ForEach-Object { ConvertTo-NativeArgument ([string]$_) }) -join " "
    )
    Write-Host ""
    Write-Host "[$($group.Name)] predicted $($group.PredictedSeconds)s; hard timeout $($group.TimeoutSeconds)s"
    Write-Host $display

    $start = [DateTime]::UtcNow
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Python
    $info.Arguments = ($arguments | ForEach-Object {
        ConvertTo-NativeArgument ([string]$_)
    }) -join " "
    $info.WorkingDirectory = $Repository
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.CreateNoWindow = $true
    $info.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    if (-not $process.Start()) {
        throw "Failed to start group $($group.Name)."
    }
    Write-RunState -Status "running" -CurrentGroup $group.Name -GroupPid $process.Id
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $completed = $process.WaitForExit([int]$group.TimeoutSeconds * 1000)
    $timedOut = -not $completed
    if ($timedOut) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        $process.WaitForExit()
    }
    $stdoutTask.Result | Set-Content -LiteralPath $stdoutPath -Encoding UTF8
    $stderrTask.Result | Set-Content -LiteralPath $stderrPath -Encoding UTF8
    $end = [DateTime]::UtcNow
    $exitCode = if ($timedOut) { 124 } else { $process.ExitCode }
    $groupStatus = if ($exitCode -eq 0) { "passed" } elseif ($timedOut) { "timed_out" } else { "failed" }
    $results.Add([ordered]@{
        name = $group.Name
        status = $groupStatus
        predicted_seconds = $group.PredictedSeconds
        timeout_seconds = $group.TimeoutSeconds
        started_utc = $start.ToString("o")
        ended_utc = $end.ToString("o")
        duration_seconds = [Math]::Round(($end - $start).TotalSeconds, 3)
        pid = $process.Id
        exit_code = $exitCode
        command = $display
        stdout = $stdoutPath
        stderr = $stderrPath
        fixture_root = $groupFixture
    })
    Write-Host "[$($group.Name)] $groupStatus (exit $exitCode, $([Math]::Round(($end - $start).TotalSeconds, 1))s)"
    if ($exitCode -eq 0 -and $RemoveSuccessfulFixtures -and (Test-Path -LiteralPath $groupFixture)) {
        Remove-Item -LiteralPath $groupFixture -Recurse -Force
    }
    if ($exitCode -ne 0) {
        $overallStatus = $groupStatus
        break
    }
}

$endedUtc = [DateTime]::UtcNow
$summary = [ordered]@{
    version = "zeromodel-video-finalization-integration-summary/v1"
    synthetic_only = $true
    status = $overallStatus
    repository = $Repository
    branch = $branch
    commit = $head
    runner_pid = $PID
    started_utc = $startedUtc.ToString("o")
    ended_utc = $endedUtc.ToString("o")
    duration_seconds = [Math]::Round(($endedUtc - $startedUtc).TotalSeconds, 3)
    validation_root = $ValidationRoot
    full_integration_selection_run = $false
    slow_tests_run = $false
    groups = $results
}
$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $SummaryJsonPath -Encoding UTF8

$markdown = @(
    "# Video Finalization Integration Validation",
    "",
    "- Status: **$overallStatus**",
    "- Synthetic only: **yes**",
    "- Branch: ``$branch``",
    "- Commit: ``$head``",
    "- Started UTC: ``$($startedUtc.ToString('o'))``",
    "- Ended UTC: ``$($endedUtc.ToString('o'))``",
    "- Duration: $([Math]::Round(($endedUtc - $startedUtc).TotalSeconds, 1)) seconds",
    "",
    "| Group | Status | Exit | Duration (s) |",
    "|---|---:|---:|---:|"
)
foreach ($result in $results) {
    $markdown += "| $($result.name) | $($result.status) | $($result.exit_code) | $($result.duration_seconds) |"
}
$markdown | Set-Content -LiteralPath $SummaryMarkdownPath -Encoding UTF8
Write-RunState -Status $overallStatus -CurrentGroup $null -GroupPid $null

Write-Host ""
Write-Host "JSON summary: $SummaryJsonPath"
Write-Host "Markdown summary: $SummaryMarkdownPath"
if ($overallStatus -ne "passed") {
    Write-Host "Failed fixtures were preserved under: $FixturesPath"
    exit 1
}
exit 0
