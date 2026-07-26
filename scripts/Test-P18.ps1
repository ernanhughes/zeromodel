<#
.SYNOPSIS
    Runs ZeroModel perception P18A-P18G tests from PowerShell.

.EXAMPLE
    .\scripts\Test-P18.ps1
    .\scripts\Test-P18.ps1 -Mode Focused -Stage P18D
    .\scripts\Test-P18.ps1 -Mode CleanWheel -RecreateVenv
    .\scripts\Test-P18.ps1 -Mode Scaffold
#>
[CmdletBinding()]
param(
    [ValidateSet("Smoke", "Focused", "Full", "CleanWheel", "Scaffold")]
    [string]$Mode = "Focused",

    [ValidateSet("All", "P18A", "P18B", "P18C", "P18D", "P18E", "P18F", "P18G")]
    [string]$Stage = "All",

    [string]$RepoRoot = "",
    [string]$Python = "",
    [string]$VenvPath = ".venv-p18",
    [string]$ResultsDir = "artifacts/p18-test-results",
    [string]$ExperimentRoot = "artifacts/p18-experiment",
    [string]$ExpectedStage = "P18G",
    [switch]$RecreateVenv,
    [switch]$ContinueOnFailure,
    [switch]$OpenResults
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Find-RepoRoot {
    param([string]$Requested, [string]$Start)
    if ($Requested) {
        $root = (Resolve-Path -LiteralPath $Requested).Path
    }
    else {
        $root = (Resolve-Path -LiteralPath $Start).Path
        while (-not (Test-Path -LiteralPath (Join-Path $root "packages/perception/tests"))) {
            $parent = Split-Path -Path $root -Parent
            if (-not $parent -or $parent -eq $root) {
                throw "Could not find the ZeroModel repository root."
            }
            $root = $parent
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $root "packages/perception/tests"))) {
        throw "Not a ZeroModel repository root: $root"
    }
    return $root
}

function Get-AbsolutePath {
    param([string]$Root, [string]$Path)
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $Root $Path))
}

function Get-BootstrapPython {
    param([string]$Requested)
    if ($Requested) {
        $command = Get-Command $Requested -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return [pscustomobject]@{ File = $command.Source; Prefix = @() }
        }
        if (Test-Path -LiteralPath $Requested) {
            return [pscustomobject]@{ File = (Resolve-Path $Requested).Path; Prefix = @() }
        }
        throw "Python executable not found: $Requested"
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $py) { return [pscustomobject]@{ File = $py.Source; Prefix = @("-3") } }
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        return [pscustomobject]@{ File = $pythonCommand.Source; Prefix = @() }
    }
    throw "Python 3 was not found. Install Python 3.12 or pass -Python <path>."
}

function Get-VenvPython {
    param([string]$Path)
    foreach ($candidate in @(
        (Join-Path $Path "Scripts/python.exe"),
        (Join-Path $Path "bin/python")
    )) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "Virtual-environment Python not found under: $Path"
}

function Invoke-CommandLogged {
    param(
        [string]$File,
        [string[]]$Args,
        [string]$Log,
        [switch]$AllowFailure
    )
    $display = (@($File) + $Args) -join " "
    Write-Host "`n> $display" -ForegroundColor Cyan
    Add-Content -LiteralPath $Log -Value "`n> $display" -Encoding UTF8
    & $File @Args 2>&1 | ForEach-Object {
        Write-Host $_
        Add-Content -LiteralPath $Log -Value $_.ToString() -Encoding UTF8
    }
    $code = $LASTEXITCODE
    if ($code -ne 0 -and -not $AllowFailure) {
        throw "Command failed with exit code ${code}: $display"
    }
    return $code
}

function Invoke-Bootstrap {
    param([pscustomobject]$Bootstrap, [string[]]$Args, [string]$Log)
    Invoke-CommandLogged -File $Bootstrap.File -Args (@($Bootstrap.Prefix) + $Args) -Log $Log | Out-Null
}

function New-TestVenv {
    param(
        [pscustomobject]$Bootstrap,
        [string]$Path,
        [string]$Log,
        [switch]$Recreate
    )
    if ($Recreate -and (Test-Path -LiteralPath $Path)) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path (Split-Path $Path -Parent) -Force | Out-Null
        Invoke-Bootstrap -Bootstrap $Bootstrap -Args @("-m", "venv", $Path) -Log $Log
    }
    return Get-VenvPython -Path $Path
}

function Install-Editable {
    param([string]$TestPython, [string]$Root, [string]$Log)
    Invoke-CommandLogged -File $TestPython -Args @("-m", "pip", "install", "--upgrade", "pip") -Log $Log | Out-Null
    Invoke-CommandLogged -File $TestPython -Args @("-m", "pip", "install", "pytest", "build", "twine", "numpy<2.0") -Log $Log | Out-Null
    foreach ($name in @("core", "observation", "perception")) {
        $package = Join-Path $Root "packages/$name"
        Invoke-CommandLogged -File $TestPython -Args @("-m", "pip", "install", "--editable", $package) -Log $Log | Out-Null
    }
    Invoke-CommandLogged -File $TestPython -Args @("-m", "pip", "check") -Log $Log | Out-Null
}

function Install-CleanWheels {
    param(
        [string]$BuildPython,
        [pscustomobject]$Bootstrap,
        [string]$Root,
        [string]$RunDir,
        [string]$Log
    )
    Invoke-CommandLogged -File $BuildPython -Args @("-m", "pip", "install", "--upgrade", "pip") -Log $Log | Out-Null
    Invoke-CommandLogged -File $BuildPython -Args @("-m", "pip", "install", "build", "twine", "numpy<2.0") -Log $Log | Out-Null

    $wheelFiles = @()
    foreach ($name in @("core", "observation", "perception")) {
        $package = Join-Path $Root "packages/$name"
        $output = Join-Path $RunDir "dist/$name"
        New-Item -ItemType Directory -Path $output -Force | Out-Null
        Invoke-CommandLogged -File $BuildPython -Args @("-m", "build", "--outdir", $output, $package) -Log $Log | Out-Null
        $files = @(Get-ChildItem -LiteralPath $output -File | Sort-Object Name)
        Invoke-CommandLogged -File $BuildPython -Args (@("-m", "twine", "check") + @($files.FullName)) -Log $Log | Out-Null
        $wheels = @($files | Where-Object { $_.Extension -eq ".whl" })
        if ($wheels.Count -ne 1) { throw "Expected one $name wheel; found $($wheels.Count)." }
        $wheelFiles += $wheels[0].FullName
    }

    $cleanVenv = Join-Path $RunDir "clean-wheel-venv"
    Invoke-Bootstrap -Bootstrap $Bootstrap -Args @("-m", "venv", $cleanVenv) -Log $Log
    $cleanPython = Get-VenvPython -Path $cleanVenv
    Invoke-CommandLogged -File $cleanPython -Args @("-m", "pip", "install", "--upgrade", "pip") -Log $Log | Out-Null
    Invoke-CommandLogged -File $cleanPython -Args @("-m", "pip", "install", "pytest") -Log $Log | Out-Null
    foreach ($wheel in $wheelFiles) {
        Invoke-CommandLogged -File $cleanPython -Args @("-m", "pip", "install", $wheel) -Log $Log | Out-Null
    }
    Invoke-CommandLogged -File $cleanPython -Args @("-m", "pip", "check") -Log $Log | Out-Null
    return $cleanPython
}

function Test-Imports {
    param(
        [string]$TestPython,
        [string]$RunDir,
        [string]$Expected,
        [bool]$RequireSitePackages,
        [string]$Log
    )
    $probe = Join-Path $RunDir "probe.py"
    $json = Join-Path $RunDir "import-info.json"
    $code = @'
import inspect, json, sys
from pathlib import Path
import zeromodel.observation as observation
import zeromodel.perception as perception

paths = {
    "observation": str(Path(inspect.getfile(observation)).resolve()),
    "perception": str(Path(inspect.getfile(perception)).resolve()),
}
result = {
    "python": sys.executable,
    "stage": perception.PERCEPTION_STAGE,
    "package_version": perception.PERCEPTION_PACKAGE_VERSION,
    "paths": paths,
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if result["stage"] != sys.argv[2]:
    raise SystemExit(f"Expected stage {sys.argv[2]}, got {result['stage']}")
if sys.argv[3].lower() == "true":
    for path in paths.values():
        if "site-packages" not in {part.lower() for part in Path(path).parts}:
            raise SystemExit(f"Import did not come from clean site-packages: {path}")
'@
    Set-Content -LiteralPath $probe -Value $code -Encoding UTF8
    $siteFlag = if ($RequireSitePackages) { "true" } else { "false" }
    Invoke-CommandLogged -File $TestPython -Args @($probe, $json, $Expected, $siteFlag) -Log $Log | Out-Null
    return $json
}

function Get-StageGroups {
    param([string]$Root)
    $tests = Join-Path $Root "packages/perception/tests"
    $definitions = @(
        @("P18A", "test_transition_evidence_p18a.py", "test_transition_evidence_p18a_public_surface.py"),
        @("P18B", "test_transition_conformance_p18b.py", "test_transition_conformance_p18b_public_surface.py"),
        @("P18C", "test_transition_discovery_p18c.py", "test_transition_discovery_p18c_public_surface.py"),
        @("P18D", "test_candidate_validation_p18d.py", "test_candidate_validation_p18d_public_surface.py"),
        @("P18E", "test_candidate_promotion_p18e.py", "test_candidate_promotion_p18e_public_surface.py"),
        @("P18F", "test_promotion_materialization_p18f.py", "test_promotion_materialization_p18f_public_surface.py"),
        @("P18G", "test_promotion_activation_p18g.py", "test_promotion_activation_p18g_public_surface.py")
    )
    $groups = @()
    foreach ($definition in $definitions) {
        $groups += [pscustomobject]@{
            Name = $definition[0]
            Tests = @(
                (Join-Path $tests $definition[1]),
                (Join-Path $tests $definition[2])
            )
        }
    }
    return $groups
}

function Invoke-TestGroup {
    param([string]$TestPython, [pscustomobject]$Group, [string]$RunDir)
    foreach ($target in $Group.Tests) {
        $file = $target.Split("::")[0]
        if (-not (Test-Path -LiteralPath $file)) { throw "Test target not found: $target" }
    }
    $name = $Group.Name.ToLowerInvariant()
    $log = Join-Path $RunDir "pytest-$name.log"
    $junit = Join-Path $RunDir "pytest-$name.xml"
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $exit = Invoke-CommandLogged -File $TestPython -Args (@("-m", "pytest", "-q") + $Group.Tests + @("--junitxml=$junit", "--durations=20")) -Log $log -AllowFailure
    $watch.Stop()
    return [pscustomobject]@{
        Group = $Group.Name
        Passed = ($exit -eq 0)
        ExitCode = $exit
        Seconds = [Math]::Round($watch.Elapsed.TotalSeconds, 3)
        Log = $log
        JUnit = $junit
    }
}

function New-ExperimentScaffold {
    param([string]$Root, [string]$Path)
    $target = Get-AbsolutePath -Root $Root -Path $Path
    foreach ($folder in @(
        "discovery/before", "discovery/after", "discovery/annotations",
        "validation/before", "validation/after", "validation/annotations",
        "negative-control/before", "negative-control/after", "negative-control/annotations"
    )) {
        New-Item -ItemType Directory -Path (Join-Path $target $folder) -Force | Out-Null
    }
    @'
interaction_id,cohort,split,before_image,after_image,annotation_file,expected_candidate,notes
discovery-001,discovery/train,positive,discovery/before/001.png,discovery/after/001.png,discovery/annotations/001.json,projectile,Deliberately omit projectile annotation
validation-001,validation/held-out,positive,validation/before/001.png,validation/after/001.png,validation/annotations/001.json,projectile,Do not reuse discovery artifacts
negative-001,validation/negative-control,negative,negative-control/before/001.png,negative-control/after/001.png,negative-control/annotations/001.json,,No projectile present
'@ | Set-Content -LiteralPath (Join-Path $target "manifest.template.csv") -Encoding UTF8
    @'
# P18 Experiment Scaffold

Keep discovery, held-out validation, and negative-control interactions disjoint.
Annotate known components while deliberately omitting the component being tested.

This scaffold prepares evidence only. The repository does not yet provide a generic
image-folder-to-P18 benchmark CLI, so Test-P18.ps1 validates implementation contracts
but does not ingest this folder automatically.
'@ | Set-Content -LiteralPath (Join-Path $target "README.md") -Encoding UTF8
    return $target
}

function Get-GitSha {
    param([string]$Root)
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($null -eq $git) { return $null }
    $sha = & $git.Source -C $Root rev-parse HEAD 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    return ($sha | Select-Object -First 1).ToString().Trim()
}

$started = Get-Date
$watch = [Diagnostics.Stopwatch]::StartNew()
$root = Find-RepoRoot -Requested $RepoRoot -Start $PSScriptRoot
Set-Location -LiteralPath $root
$resultsRoot = Get-AbsolutePath -Root $root -Path $ResultsDir
$runDir = Join-Path $resultsRoot ((Get-Date -Format "yyyyMMdd-HHmmss") + "-" + $Mode.ToLowerInvariant())
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$runnerLog = Join-Path $runDir "runner.log"
$results = @()
$status = "failed"
$errorMessage = $null
$importInfo = $null

try {
    Write-Host "ZeroModel P18 test runner" -ForegroundColor Green
    Write-Host "Repository: $root"
    Write-Host "Mode:       $Mode"
    Write-Host "Stage:      $Stage"
    Write-Host "Results:    $runDir"

    if ($Mode -eq "Scaffold") {
        $path = New-ExperimentScaffold -Root $root -Path $ExperimentRoot
        Write-Host "Created experiment scaffold: $path" -ForegroundColor Green
        $status = "passed"
    }
    else {
        $bootstrap = Get-BootstrapPython -Requested $Python
        $venv = Get-AbsolutePath -Root $root -Path $VenvPath
        $buildPython = New-TestVenv -Bootstrap $bootstrap -Path $venv -Log $runnerLog -Recreate:$RecreateVenv

        if ($Mode -eq "CleanWheel") {
            $testPython = Install-CleanWheels -BuildPython $buildPython -Bootstrap $bootstrap -Root $root -RunDir $runDir -Log $runnerLog
            $clean = $true
        }
        else {
            Install-Editable -TestPython $buildPython -Root $root -Log $runnerLog
            $testPython = $buildPython
            $clean = $false
        }

        $importInfo = Test-Imports -TestPython $testPython -RunDir $runDir -Expected $ExpectedStage -RequireSitePackages $clean -Log $runnerLog
        $allGroups = Get-StageGroups -Root $root
        $groups = @()
        switch ($Mode) {
            "Smoke" {
                $tests = Join-Path $root "packages/perception/tests"
                $groups = @([pscustomobject]@{
                    Name = "Smoke-P18G"
                    Tests = @(
                        ((Join-Path $tests "test_promotion_activation_p18g.py") + "::test_activates_all_operations_and_persists_exact_inverse_plan"),
                        (Join-Path $tests "test_promotion_activation_p18g_public_surface.py")
                    )
                })
            }
            "Focused" {
                if ($Stage -eq "All") { $groups = $allGroups }
                else { $groups = @($allGroups | Where-Object { $_.Name -eq $Stage }) }
            }
            "Full" {
                $groups = @([pscustomobject]@{ Name = "Full-Perception"; Tests = @((Join-Path $root "packages/perception/tests")) })
            }
            "CleanWheel" {
                $groups = @([pscustomobject]@{ Name = "CleanWheel-Perception"; Tests = @((Join-Path $root "packages/perception/tests")) })
            }
        }

        foreach ($group in $groups) {
            $result = Invoke-TestGroup -TestPython $testPython -Group $group -RunDir $runDir
            $results += $result
            if (-not $result.Passed -and -not $ContinueOnFailure) { break }
        }
        $status = if (@($results | Where-Object { -not $_.Passed }).Count -eq 0) { "passed" } else { "failed" }
    }
}
catch {
    $errorMessage = $_.Exception.Message
    Write-Host $errorMessage -ForegroundColor Red
    Add-Content -LiteralPath $runnerLog -Value "ERROR: $errorMessage" -Encoding UTF8
}
finally {
    $watch.Stop()
    $info = $null
    if ($importInfo -and (Test-Path -LiteralPath $importInfo)) {
        $info = Get-Content -LiteralPath $importInfo -Raw | ConvertFrom-Json
    }
    $summary = [ordered]@{
        status = $status
        mode = $Mode
        requested_stage = $Stage
        expected_stage = $ExpectedStage
        observed_stage = if ($null -ne $info) { $info.stage } else { $null }
        package_version = if ($null -ne $info) { $info.package_version } else { $null }
        repository = $root
        git_commit = Get-GitSha -Root $root
        powershell = $PSVersionTable.PSVersion.ToString()
        started_at = $started.ToString("o")
        duration_seconds = [Math]::Round($watch.Elapsed.TotalSeconds, 3)
        error = $errorMessage
        test_results = $results
    }
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runDir "summary.json") -Encoding UTF8

    $lines = @(
        "# P18 Test Summary", "",
        ("- Status: **{0}**" -f $status),
        ('- Mode: `{0}`' -f $Mode),
        ('- Expected package stage: `{0}`' -f $ExpectedStage),
        ('- Results: `{0}`' -f $runDir), "",
        "| Group | Result | Exit | Seconds |", "|---|---:|---:|---:|"
    )
    foreach ($result in $results) {
        $label = if ($result.Passed) { "PASS" } else { "FAIL" }
        $lines += ("| {0} | {1} | {2} | {3} |" -f $result.Group, $label, $result.ExitCode, $result.Seconds)
    }
    $lines | Set-Content -LiteralPath (Join-Path $runDir "summary.md") -Encoding UTF8
    Write-Host "`nSummary: $(Join-Path $runDir 'summary.md')" -ForegroundColor Cyan
    if ($OpenResults) { Invoke-Item -LiteralPath $runDir }
}

if ($status -ne "passed") { exit 1 }
exit 0
