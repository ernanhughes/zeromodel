[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version,

    [string]$Python = "python",

    [string]$PyPIToken = $env:PYPI_API_TOKEN,

    [switch]$SkipValidation,

    [switch]$SkipSmokeTest,

    [switch]$ResumePartial,

    [switch]$Yes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"


function Step {
    param([Parameter(Mandatory = $true)][string]$Message)

    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}


function Fail {
    param([Parameter(Mandatory = $true)][string]$Message)

    throw $Message
}


function Run {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string[]]$Arguments = @()
    )

    & $Command @Arguments

    if ($LASTEXITCODE -ne 0) {
        Fail "Command failed ($LASTEXITCODE): $Command $($Arguments -join ' ')"
    }
}


function Capture {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string[]]$Arguments = @()
    )

    $Output = & $Command @Arguments 2>&1

    if ($LASTEXITCODE -ne 0) {
        Fail @"
Command failed ($LASTEXITCODE): $Command $($Arguments -join ' ')
$(($Output | Out-String).Trim())
"@
    }

    return (($Output | Out-String).Trim())
}


function Need {
    param([Parameter(Mandatory = $true)][string]$Command)

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        Fail "Required command '$Command' was not found on PATH."
    }
}


function Read-Text {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Fail "Required file not found: $Path"
    }

    return [IO.File]::ReadAllText($Path)
}


function Get-PackageDefinitions {
    param([Parameter(Mandatory = $true)][string]$Root)

    $BoundaryPath = Join-Path $Root "package-boundaries.toml"
    $Lines = Get-Content -LiteralPath $BoundaryPath

    $Packages = [System.Collections.Generic.List[object]]::new()
    $Current = $null

    foreach ($RawLine in $Lines) {
        $Line = $RawLine.Trim()

        if ($Line -match '^\[packages\.([A-Za-z0-9_-]+)\]$') {
            if ($null -ne $Current) {
                $Packages.Add([pscustomobject]$Current)
            }

            $Current = [ordered]@{
                Key          = $Matches[1]
                Distribution = $null
                Namespace    = $null
                SourceRoot   = $null
                PackageRoot  = $null
                DependsOn    = @()
                Publishable  = $false
            }

            continue
        }

        if ($null -eq $Current) {
            continue
        }

        if ($Line -match '^distribution\s*=\s*"([^"]+)"$') {
            $Current.Distribution = $Matches[1]
            continue
        }

        if ($Line -match '^namespace\s*=\s*"([^"]+)"$') {
            $Current.Namespace = $Matches[1]
            continue
        }

        if ($Line -match '^source_root\s*=\s*"([^"]+)"$') {
            $Current.SourceRoot = $Matches[1]

            $Normalized = $Matches[1].Replace("\", "/")
            if ($Normalized -notmatch '^(.+)/src$') {
                Fail "Package '$($Current.Key)' has unsupported source_root '$Normalized'."
            }

            $Current.PackageRoot = $Matches[1].Substring(
                0,
                $Matches[1].Length - 4
            )

            continue
        }

        if ($Line -match '^depends_on\s*=\s*\[(.*)\]$') {
            $Dependencies = [System.Collections.Generic.List[string]]::new()

            foreach ($Match in [regex]::Matches($Matches[1], '"([^"]+)"')) {
                $Dependencies.Add($Match.Groups[1].Value)
            }

            $Current.DependsOn = @($Dependencies)
            continue
        }

        if ($Line -match '^publishable\s*=\s*(true|false)$') {
            $Current.Publishable = $Matches[1] -eq "true"
            continue
        }
    }

    if ($null -ne $Current) {
        $Packages.Add([pscustomobject]$Current)
    }

    $Publishable = @(
        $Packages |
            Where-Object { $_.Publishable } |
            Sort-Object Key
    )

    if ($Publishable.Count -eq 0) {
        Fail "package-boundaries.toml declares no publishable packages."
    }

    foreach ($Package in $Publishable) {
        if (
            -not $Package.Distribution -or
            -not $Package.Namespace -or
            -not $Package.SourceRoot -or
            -not $Package.PackageRoot
        ) {
            Fail "Incomplete package definition for '$($Package.Key)'."
        }

        $AbsoluteRoot = Join-Path $Root $Package.PackageRoot

        if (-not (Test-Path -LiteralPath $AbsoluteRoot -PathType Container)) {
            Fail "Package root not found for '$($Package.Key)': $AbsoluteRoot"
        }

        $PyProject = Join-Path $AbsoluteRoot "pyproject.toml"

        if (-not (Test-Path -LiteralPath $PyProject -PathType Leaf)) {
            Fail "Package pyproject.toml not found: $PyProject"
        }
    }

    return $Publishable
}


function Get-BoundaryReleaseVersion {
    param([Parameter(Mandatory = $true)][string]$Root)

    $Text = Read-Text (Join-Path $Root "package-boundaries.toml")
    $Match = [regex]::Match(
        $Text,
        '(?m)^release_version\s*=\s*"([^"]+)"\s*$'
    )

    if (-not $Match.Success) {
        Fail "Could not find release_version in package-boundaries.toml."
    }

    return $Match.Groups[1].Value
}


function Get-PackageMetadata {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)]$Package
    )

    $Path = Join-Path $Root "$($Package.PackageRoot)\pyproject.toml"
    $Text = Read-Text $Path

    $NameMatch = [regex]::Match(
        $Text,
        '(?m)^name\s*=\s*"([^"]+)"\s*$'
    )

    $VersionMatch = [regex]::Match(
        $Text,
        '(?m)^version\s*=\s*"([^"]+)"\s*$'
    )

    if (-not $NameMatch.Success) {
        Fail "Could not read project name from $Path."
    }

    if (-not $VersionMatch.Success) {
        Fail "Could not read project version from $Path."
    }

    return [pscustomobject]@{
        Key          = $Package.Key
        Distribution = $NameMatch.Groups[1].Value
        Namespace    = $Package.Namespace
        PackageRoot  = $Package.PackageRoot
        Version      = $VersionMatch.Groups[1].Value
        DependsOn    = @($Package.DependsOn)
        PyProject    = $Path
    }
}


function Assert-PackageVersions {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][object[]]$Packages,
        [Parameter(Mandatory = $true)][string]$ExpectedVersion
    )

    $BoundaryVersion = Get-BoundaryReleaseVersion $Root

    if ($BoundaryVersion -ne $ExpectedVersion) {
        Fail @"
package-boundaries.toml declares release_version=$BoundaryVersion,
but this publish requested $ExpectedVersion.
"@
    }

    $DistributionNames = @(
        $Packages | ForEach-Object { $_.Distribution }
    )

    foreach ($Package in $Packages) {
        if ($Package.Version -ne $ExpectedVersion) {
            Fail @"
Package '$($Package.Distribution)' declares version $($Package.Version),
expected $ExpectedVersion.
File: $($Package.PyProject)
"@
        }

        $Text = Read-Text $Package.PyProject

        foreach ($Distribution in $DistributionNames) {
            $Pattern = [regex]::Escape($Distribution) +
                '([<>=!~]=?|===)[^",\]\s]+'

            foreach ($Match in [regex]::Matches($Text, $Pattern)) {
                $Requirement = $Match.Value

                if (
                    $Requirement.StartsWith("$Distribution==") -and
                    $Requirement -ne "$Distribution==$ExpectedVersion"
                ) {
                    Fail @"
Internal dependency mismatch in $($Package.PyProject):
$Requirement
Expected: $Distribution==$ExpectedVersion
"@
                }
            }
        }
    }
}


function Get-TopologicalPackages {
    param([Parameter(Mandatory = $true)][object[]]$Packages)

    $ByKey = @{}

    foreach ($Package in $Packages) {
        $ByKey[$Package.Key] = $Package
    }

    $Remaining = [System.Collections.Generic.List[object]]::new()

    foreach ($Package in $Packages) {
        $Remaining.Add($Package)
    }

    $Ordered = [System.Collections.Generic.List[object]]::new()
    $Completed = [System.Collections.Generic.HashSet[string]]::new()

    while ($Remaining.Count -gt 0) {
        $Progress = $false

        foreach ($Package in @($Remaining)) {
            $DependenciesSatisfied = $true

            foreach ($Dependency in $Package.DependsOn) {
                if (-not $ByKey.ContainsKey($Dependency)) {
                    Fail @"
Package '$($Package.Key)' depends on unknown package '$Dependency'.
"@
                }

                if (-not $Completed.Contains($Dependency)) {
                    $DependenciesSatisfied = $false
                    break
                }
            }

            if ($DependenciesSatisfied) {
                $Ordered.Add($Package)
                [void]$Completed.Add($Package.Key)
                [void]$Remaining.Remove($Package)
                $Progress = $true
            }
        }

        if (-not $Progress) {
            $Names = ($Remaining | ForEach-Object { $_.Key }) -join ", "
            Fail "Package dependency cycle detected among: $Names"
        }
    }

    return @($Ordered)
}


function Test-PyPIVersionExists {
    param(
        [Parameter(Mandatory = $true)][string]$PythonCommand,
        [Parameter(Mandatory = $true)][string]$Distribution,
        [Parameter(Mandatory = $true)][string]$ReleaseVersion
    )

    $Code = @"
import sys
import urllib.error
import urllib.parse
import urllib.request

name = urllib.parse.quote("$Distribution", safe="")
version = urllib.parse.quote("$ReleaseVersion", safe="")
url = f"https://pypi.org/pypi/{name}/{version}/json"

try:
    with urllib.request.urlopen(url, timeout=30):
        sys.exit(0)
except urllib.error.HTTPError as exc:
    if exc.code == 404:
        sys.exit(1)
    raise
"@

    & $PythonCommand -c $Code

    if ($LASTEXITCODE -eq 0) {
        return $true
    }

    if ($LASTEXITCODE -eq 1) {
        return $false
    }

    Fail "Could not query PyPI for $Distribution==$ReleaseVersion."
}


function Wait-ForPyPIVersion {
    param(
        [Parameter(Mandatory = $true)][string]$PythonCommand,
        [Parameter(Mandatory = $true)][string]$Distribution,
        [Parameter(Mandatory = $true)][string]$ReleaseVersion
    )

    for ($Attempt = 1; $Attempt -le 30; $Attempt++) {
        if (
            Test-PyPIVersionExists `
                -PythonCommand $PythonCommand `
                -Distribution $Distribution `
                -ReleaseVersion $ReleaseVersion
        ) {
            return
        }

        Write-Host (
            "Waiting for PyPI: {0}=={1} ({2}/30)" -f
            $Distribution,
            $ReleaseVersion,
            $Attempt
        )

        Start-Sleep -Seconds 10
    }

    Fail "PyPI did not expose $Distribution==$ReleaseVersion in time."
}


function Get-PackageArtifacts {
    param(
        [Parameter(Mandatory = $true)][string]$DistRoot,
        [Parameter(Mandatory = $true)][string]$Distribution,
        [Parameter(Mandatory = $true)][string]$ReleaseVersion
    )

    $Normalized = $Distribution.Replace("-", "_")
    $EscapedOriginal = [regex]::Escape($Distribution)
    $EscapedNormalized = [regex]::Escape($Normalized)
    $EscapedVersion = [regex]::Escape($ReleaseVersion)

    return @(
        Get-ChildItem -LiteralPath $DistRoot -File |
            Where-Object {
                $_.Name -match (
                    "^(?:$EscapedOriginal|$EscapedNormalized)" +
                    "[-_]$EscapedVersion(?:-|\.)(?:.*\.(?:whl|tar\.gz))$"
                ) -or
                $_.Name -match (
                    "^(?:$EscapedOriginal|$EscapedNormalized)" +
                    "[-_]$EscapedVersion.*\.(?:whl|tar\.gz)$"
                )
            }
    )
}


function New-CleanEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$PythonCommand,
        [Parameter(Mandatory = $true)][string]$Path
    )

    Remove-Item -Recurse -Force $Path -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $Path | Out-Null

    Run $PythonCommand @("-m", "venv", (Join-Path $Path ".venv"))

    $VenvPython = Join-Path $Path ".venv\Scripts\python.exe"

    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        Fail "Clean-environment Python was not created: $VenvPython"
    }

    Run $VenvPython @("-m", "pip", "install", "--upgrade", "pip")

    return $VenvPython
}


function Invoke-LocalWheelSmokeTest {
    param(
        [Parameter(Mandatory = $true)][string]$PythonCommand,
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$DistRoot,
        [Parameter(Mandatory = $true)][object[]]$Packages,
        [Parameter(Mandatory = $true)][string]$ReleaseVersion
    )

    $SmokeRoot = Join-Path $env:TEMP (
        "zeromodel-local-wheel-smoke-$ReleaseVersion"
    )

    $VenvPython = New-CleanEnvironment `
        -PythonCommand $PythonCommand `
        -Path $SmokeRoot

    $Wheels = @(
        Get-ChildItem -LiteralPath $DistRoot -Filter "*.whl" -File |
            Sort-Object Name |
            ForEach-Object { $_.FullName }
    )

    if ($Wheels.Count -ne $Packages.Count) {
        Fail @"
Expected $($Packages.Count) wheels in $DistRoot,
found $($Wheels.Count).
"@
    }

    Run $VenvPython (@("-m", "pip", "install") + $Wheels)

    $ImportNames = @(
        $Packages |
            ForEach-Object { $_.Namespace } |
            Sort-Object -Unique
    )

    $ImportLiteral = (
        $ImportNames |
            ForEach-Object { "'$($_.Replace("'", "\'"))'" }
    ) -join ", "

    $Code = @"
import importlib
import importlib.metadata as metadata

packages = [$ImportLiteral]

for module_name in packages:
    importlib.import_module(module_name)

expected = "$ReleaseVersion"

distributions = [
$(
    (
        $Packages |
            ForEach-Object { "    '$($_.Distribution)'," }
    ) -join "`n"
)
]

for distribution in distributions:
    actual = metadata.version(distribution)
    assert actual == expected, (distribution, actual, expected)

from zeromodel import LayoutRecipe, ScoreTable, VPMPolicyLookup, build_vpm

table = ScoreTable(
    values=[[1.0, 0.0], [0.0, 1.0]],
    row_ids=["state:left", "state:right"],
    metric_ids=["LEFT", "RIGHT"],
)

recipe = LayoutRecipe.from_dict({
    "version": "vpm-layout/0",
    "name": "release-smoke",
    "row_order": {"kind": "source", "tie_break": "row_id"},
    "column_order": {"kind": "source"},
    "normalization": {
        "kind": "per_metric_minmax",
        "clip": True,
    },
})

artifact = build_vpm(table, recipe)
decision = VPMPolicyLookup(artifact).read("state:right")
assert decision.action == "RIGHT", decision

print("Local wheel-set smoke test passed.")
"@

    Run $VenvPython @("-c", $Code)
}


function Invoke-PyPISmokeTest {
    param(
        [Parameter(Mandatory = $true)][string]$PythonCommand,
        [Parameter(Mandatory = $true)][object[]]$Packages,
        [Parameter(Mandatory = $true)][string]$ReleaseVersion
    )

    $SmokeRoot = Join-Path $env:TEMP (
        "zeromodel-pypi-smoke-$ReleaseVersion"
    )

    $VenvPython = New-CleanEnvironment `
        -PythonCommand $PythonCommand `
        -Path $SmokeRoot

    $Requirements = @(
        $Packages |
            ForEach-Object {
                "$($_.Distribution)==$ReleaseVersion"
            }
    )

    Run $VenvPython (
        @("-m", "pip", "install", "--no-cache-dir") +
        $Requirements
    )

    $ImportNames = @(
        $Packages |
            ForEach-Object { $_.Namespace } |
            Sort-Object -Unique
    )

    $ImportLiteral = (
        $ImportNames |
            ForEach-Object { "'$($_.Replace("'", "\'"))'" }
    ) -join ", "

    $Code = @"
import importlib
import importlib.metadata as metadata

modules = [$ImportLiteral]

for module_name in modules:
    importlib.import_module(module_name)

expected = "$ReleaseVersion"

distributions = [
$(
    (
        $Packages |
            ForEach-Object { "    '$($_.Distribution)'," }
    ) -join "`n"
)
]

for distribution in distributions:
    actual = metadata.version(distribution)
    assert actual == expected, (distribution, actual, expected)

from zeromodel import LayoutRecipe, ScoreTable, VPMPolicyLookup, build_vpm

table = ScoreTable(
    values=[[1.0, 0.0], [0.0, 1.0]],
    row_ids=["state:left", "state:right"],
    metric_ids=["LEFT", "RIGHT"],
)

recipe = LayoutRecipe.from_dict({
    "version": "vpm-layout/0",
    "name": "pypi-release-smoke",
    "row_order": {"kind": "source", "tie_break": "row_id"},
    "column_order": {"kind": "source"},
    "normalization": {
        "kind": "per_metric_minmax",
        "clip": True,
    },
})

artifact = build_vpm(table, recipe)
assert VPMPolicyLookup(artifact).read("state:right").action == "RIGHT"

print("PyPI smoke test passed.")
"@

    Run $VenvPython @("-c", $Code)
}


$Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$DistRoot = Join-Path $Root "dist\release-$Version"


Push-Location $Root

try {
    Step "Checking repository and release tools"

    Need "git"
    Need $Python

    $GitRoot = [IO.Path]::GetFullPath(
        (Capture "git" @("rev-parse", "--show-toplevel"))
    )

    if ($GitRoot.TrimEnd("\") -ne $Root.TrimEnd("\")) {
        Fail "Run this script from the ZeroModel repository."
    }

    $Branch = Capture "git" @("branch", "--show-current")

    if ($Branch -ne "main") {
        Fail "Publishing is allowed only from main. Current branch: $Branch"
    }

    $Status = Capture "git" @("status", "--porcelain")

    if ($Status) {
        Fail "Working tree is not clean.`n$Status"
    }

    $Head = Capture "git" @("rev-parse", "HEAD")
    $RemoteHead = Capture "git" @("rev-parse", "origin/main")

    if ($Head -ne $RemoteHead) {
        Fail @"
Local main and origin/main differ.
Local:  $Head
Remote: $RemoteHead
"@
    }

    Step "Reading package topology"

    $Definitions = Get-PackageDefinitions $Root

    $Packages = @(
        $Definitions |
            ForEach-Object {
                Get-PackageMetadata -Root $Root -Package $_
            }
    )

    $OrderedPackages = Get-TopologicalPackages $Packages

    Write-Host "Release version: $Version"
    Write-Host "Commit: $Head"
    Write-Host "Packages:"

    foreach ($Package in $OrderedPackages) {
        Write-Host (
            "  {0,-24} {1}" -f
            $Package.Distribution,
            $Package.PackageRoot
        )
    }

    Assert-PackageVersions `
        -Root $Root `
        -Packages $OrderedPackages `
        -ExpectedVersion $Version

    Step "Installing release tooling"

    Run $Python @("-m", "pip", "install", "--upgrade", "pip")
    Run $Python @("-m", "pip", "install", "build", "twine")

    if (-not $SkipValidation) {
        Step "Running authoritative release-candidate validation"
        Run $Python @("scripts/validate_release_candidate.py")

        Step "Running repository quality gate"
        Run $Python @("scripts/check_quality.py")

        Step "Running bounded fast tests"
        Run $Python @("scripts/run_fast_tests.py")

        $StatusAfterValidation = Capture "git" @("status", "--porcelain")

        if ($StatusAfterValidation) {
            Fail @"
Release validation changed the working tree.

Commit generated release evidence before publishing:

$StatusAfterValidation
"@
        }
    }

    Step "Cleaning package build output"

    Remove-Item -Recurse -Force $DistRoot -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $DistRoot | Out-Null

    foreach ($Package in $OrderedPackages) {
        $PackageBuild = Join-Path $Root "$($Package.PackageRoot)\build"
        $PackageDist = Join-Path $Root "$($Package.PackageRoot)\dist"

        Remove-Item -Recurse -Force `
            $PackageBuild,
            $PackageDist `
            -ErrorAction SilentlyContinue

        Get-ChildItem `
            -LiteralPath (Join-Path $Root $Package.PackageRoot) `
            -Directory `
            -Filter "*.egg-info" `
            -ErrorAction SilentlyContinue |
                Remove-Item -Recurse -Force
    }

    Step "Building all publishable packages"

    foreach ($Package in $OrderedPackages) {
        Write-Host ""
        Write-Host "Building $($Package.Distribution)" -ForegroundColor Yellow

        Run $Python @(
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--outdir",
            $DistRoot,
            (Join-Path $Root $Package.PackageRoot)
        )
    }

    $Artifacts = @(
        Get-ChildItem -LiteralPath $DistRoot -File |
            Sort-Object Name
    )

    $ExpectedArtifactCount = $OrderedPackages.Count * 2

    if ($Artifacts.Count -ne $ExpectedArtifactCount) {
        Fail @"
Expected $ExpectedArtifactCount artifacts
(two per package), found $($Artifacts.Count).

$(
    ($Artifacts | ForEach-Object { $_.Name }) -join "`n"
)
"@
    }

    Step "Checking package metadata"

    Run $Python (
        @("-m", "twine", "check") +
        @($Artifacts | ForEach-Object { $_.FullName })
    )

    Step "Testing complete local wheel set"

    Invoke-LocalWheelSmokeTest `
        -PythonCommand $Python `
        -Root $Root `
        -DistRoot $DistRoot `
        -Packages $OrderedPackages `
        -ReleaseVersion $Version

    Step "Checking existing PyPI publication state"

    $Existing = [System.Collections.Generic.List[object]]::new()
    $Missing = [System.Collections.Generic.List[object]]::new()

    foreach ($Package in $OrderedPackages) {
        if (
            Test-PyPIVersionExists `
                -PythonCommand $Python `
                -Distribution $Package.Distribution `
                -ReleaseVersion $Version
        ) {
            $Existing.Add($Package)
            Write-Host "EXISTS  $($Package.Distribution)==$Version"
        }
        else {
            $Missing.Add($Package)
            Write-Host "MISSING $($Package.Distribution)==$Version"
        }
    }

    if (
        $Existing.Count -gt 0 -and
        $Missing.Count -gt 0 -and
        -not $ResumePartial
    ) {
        Fail @"
PyPI contains a partial ZeroModel $Version release.

Existing:
$(
    ($Existing | ForEach-Object { "  $($_.Distribution)" }) -join "`n"
)

Missing:
$(
    ($Missing | ForEach-Object { "  $($_.Distribution)" }) -join "`n"
)

Review the partial publication and rerun with -ResumePartial
only when the existing files are known to be correct.
"@
    }

    if ($Missing.Count -eq 0) {
        Write-Host ""
        Write-Host (
            "All ZeroModel $Version packages already exist on PyPI."
        ) -ForegroundColor Yellow
    }
    else {
        if (-not $PyPIToken) {
            $SecureToken = Read-Host `
                "Enter production PyPI API token" `
                -AsSecureString

            $Bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR(
                $SecureToken
            )

            try {
                $PyPIToken = (
                    [Runtime.InteropServices.Marshal]::PtrToStringAuto($Bstr)
                )
            }
            finally {
                [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Bstr)
            }
        }

        if (-not $PyPIToken.StartsWith("pypi-")) {
            Fail "Production PyPI token should start with 'pypi-'."
        }

        $UploadArtifacts = [System.Collections.Generic.List[string]]::new()

        foreach ($Package in $Missing) {
            $PackageArtifacts = Get-PackageArtifacts `
                -DistRoot $DistRoot `
                -Distribution $Package.Distribution `
                -ReleaseVersion $Version

            if ($PackageArtifacts.Count -ne 2) {
                Fail @"
Expected two artifacts for $($Package.Distribution),
found $($PackageArtifacts.Count).
"@
            }

            foreach ($Artifact in $PackageArtifacts) {
                $UploadArtifacts.Add($Artifact.FullName)
            }
        }

        Write-Host ""
        Write-Host (
            "Ready to publish ZeroModel $Version to production PyPI."
        ) -ForegroundColor Yellow
        Write-Host (
            "Packages to upload: $($Missing.Count)"
        ) -ForegroundColor Yellow
        Write-Host (
            "PyPI versions are immutable."
        ) -ForegroundColor Yellow

        if (-not $Yes) {
            $Confirmation = Read-Host (
                "Type PUBLISH $Version to continue"
            )

            if ($Confirmation -ne "PUBLISH $Version") {
                Fail "Upload cancelled."
            }
        }

        Step "Uploading missing packages to production PyPI"

        $PreviousUsername = $env:TWINE_USERNAME
        $PreviousPassword = $env:TWINE_PASSWORD

        try {
            $env:TWINE_USERNAME = "__token__"
            $env:TWINE_PASSWORD = $PyPIToken

            Run $Python (
                @("-m", "twine", "upload") +
                @($UploadArtifacts)
            )
        }
        finally {
            $env:TWINE_USERNAME = $PreviousUsername
            $env:TWINE_PASSWORD = $PreviousPassword
        }
    }

    Step "Waiting for all package versions to appear on PyPI"

    foreach ($Package in $OrderedPackages) {
        Wait-ForPyPIVersion `
            -PythonCommand $Python `
            -Distribution $Package.Distribution `
            -ReleaseVersion $Version
    }

    if (-not $SkipSmokeTest) {
        Step "Testing complete release from production PyPI"

        Invoke-PyPISmokeTest `
            -PythonCommand $Python `
            -Packages $OrderedPackages `
            -ReleaseVersion $Version
    }

    Step "PyPI publication complete"

    Write-Host ""
    Write-Host (
        "Published and verified ZeroModel $Version."
    ) -ForegroundColor Green
    Write-Host "Artifacts: $DistRoot"
}
finally {
    Pop-Location
}
