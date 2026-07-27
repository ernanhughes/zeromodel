[CmdletBinding()]
param(
    [ValidateSet("Prepare", "Publish")]
    [string]$Mode = "Prepare",

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version,

    [string]$Python = "python",

    [string]$Remote = "origin",

    [string]$BaseBranch = "main",

    [string]$Repository = "ernanhughes/zeromodel",

    [string]$ReleaseNotesPath = "",

    [switch]$SkipQuality,

    [switch]$SkipTests,

    [switch]$SkipReleaseValidator,

    [switch]$SkipCI,

    [switch]$SkipPyPI,

    [switch]$SkipGitHubRelease,

    [switch]$ResumePartialPyPI,

    [switch]$Yes,

    [switch]$DryRun
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


function Write-Text {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )

    $Parent = Split-Path -Parent $Path

    if ($Parent -and -not (Test-Path -LiteralPath $Parent)) {
        New-Item -ItemType Directory -Path $Parent | Out-Null
    }

    [IO.File]::WriteAllText(
        $Path,
        $Content,
        (New-Object Text.UTF8Encoding($false))
    )
}


function Replace-ExactlyOne {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string]$Replacement
    )

    $Text = Read-Text $Path
    $Matches = [regex]::Matches($Text, $Pattern)

    if ($Matches.Count -ne 1) {
        Fail @"
Expected exactly one match in:
$Path

Pattern:
$Pattern

Found:
$($Matches.Count)
"@
    }

    $Updated = [regex]::Replace(
        $Text,
        $Pattern,
        $Replacement,
        1
    )

    Write-Text -Path $Path -Content $Updated
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
                Fail "Unsupported source_root: $Normalized"
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
        Fail "No publishable packages found in package-boundaries.toml."
    }

    return $Publishable
}


function Get-PackageVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)]$Package
    )

    $Path = Join-Path $Root "$($Package.PackageRoot)\pyproject.toml"
    $Text = Read-Text $Path

    $Match = [regex]::Match(
        $Text,
        '(?m)^version\s*=\s*"([^"]+)"\s*$'
    )

    if (-not $Match.Success) {
        Fail "Could not read package version from $Path."
    }

    return $Match.Groups[1].Value
}


function Get-BoundaryVersion {
    param([Parameter(Mandatory = $true)][string]$Root)

    $Text = Read-Text (Join-Path $Root "package-boundaries.toml")
    $Match = [regex]::Match(
        $Text,
        '(?m)^release_version\s*=\s*"([^"]+)"\s*$'
    )

    if (-not $Match.Success) {
        Fail "Could not read release_version from package-boundaries.toml."
    }

    return $Match.Groups[1].Value
}


function Assert-Clean {
    $Status = Capture "git" @("status", "--porcelain")

    if ($Status) {
        Fail "Working tree is not clean.`n$Status"
    }
}


function Assert-SynchronizedBranch {
    param(
        [Parameter(Mandatory = $true)][string]$RemoteName,
        [Parameter(Mandatory = $true)][string]$BranchName
    )

    Run "git" @("fetch", "--prune", "--tags", $RemoteName)

    $CurrentBranch = Capture "git" @("branch", "--show-current")

    if ($CurrentBranch -ne $BranchName) {
        Fail @"
Expected branch '$BranchName'.
Current branch: '$CurrentBranch'
"@
    }

    $LocalHead = Capture "git" @("rev-parse", "HEAD")
    $RemoteHead = Capture "git" @(
        "rev-parse",
        "$RemoteName/$BranchName"
    )

    if ($LocalHead -ne $RemoteHead) {
        Fail @"
Local and remote $BranchName differ.
Local:  $LocalHead
Remote: $RemoteHead
"@
    }

    return $LocalHead
}


function Set-WorkspaceVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][object[]]$Packages,
        [Parameter(Mandatory = $true)][string]$ReleaseVersion
    )

    Step "Updating package-boundaries.toml"

    Replace-ExactlyOne `
        -Path (Join-Path $Root "package-boundaries.toml") `
        -Pattern '(?m)^release_version\s*=\s*"[^"]+"\s*$' `
        -Replacement "release_version = `"$ReleaseVersion`""

    $DistributionNames = @(
        $Packages | ForEach-Object { $_.Distribution }
    )

    foreach ($Package in $Packages) {
        $Path = Join-Path $Root "$($Package.PackageRoot)\pyproject.toml"

        Step "Updating $($Package.Distribution)"

        Replace-ExactlyOne `
            -Path $Path `
            -Pattern '(?m)^version\s*=\s*"[^"]+"\s*$' `
            -Replacement "version = `"$ReleaseVersion`""

        $Text = Read-Text $Path

        foreach ($Distribution in $DistributionNames) {
            $Escaped = [regex]::Escape($Distribution)

            $Pattern = (
                "(?m)(`"$Escaped)==[0-9]+\.[0-9]+\.[0-9]+(`")"
            )

            if ([regex]::IsMatch($Text, $Pattern)) {
                $Text = [regex]::Replace(
                    $Text,
                    $Pattern,
                    "`${1}$ReleaseVersion`${2}"
                )
            }
        }

        Write-Text -Path $Path -Content $Text
    }

    $ReadmePath = Join-Path $Root "README.md"

    if (Test-Path -LiteralPath $ReadmePath -PathType Leaf) {
        $Readme = Read-Text $ReadmePath
        $Readme = [regex]::Replace(
            $Readme,
            'zeromodel==[0-9]+\.[0-9]+\.[0-9]+',
            "zeromodel==$ReleaseVersion"
        )

        Write-Text -Path $ReadmePath -Content $Readme
    }
}


function Assert-WorkspaceVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][object[]]$Packages,
        [Parameter(Mandatory = $true)][string]$ExpectedVersion
    )

    $BoundaryVersion = Get-BoundaryVersion $Root

    if ($BoundaryVersion -ne $ExpectedVersion) {
        Fail @"
package-boundaries.toml has release_version=$BoundaryVersion,
expected $ExpectedVersion.
"@
    }

    $DistributionNames = @(
        $Packages | ForEach-Object { $_.Distribution }
    )

    foreach ($Package in $Packages) {
        $VersionFound = Get-PackageVersion `
            -Root $Root `
            -Package $Package

        if ($VersionFound -ne $ExpectedVersion) {
            Fail @"
Package $($Package.Distribution) has version $VersionFound,
expected $ExpectedVersion.
"@
        }

        $Path = Join-Path $Root "$($Package.PackageRoot)\pyproject.toml"
        $Text = Read-Text $Path

        foreach ($Distribution in $DistributionNames) {
            $Escaped = [regex]::Escape($Distribution)
            $Pattern = (
                "`"$Escaped==([0-9]+\.[0-9]+\.[0-9]+)`""
            )

            foreach ($Match in [regex]::Matches($Text, $Pattern)) {
                $DependencyVersion = $Match.Groups[1].Value

                if ($DependencyVersion -ne $ExpectedVersion) {
                    Fail @"
Internal dependency mismatch in ${Path}:
$($Match.Value)
Expected version: $ExpectedVersion
"@
                }
            }
        }
    }
}


function Update-Changelog {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$ReleaseVersion,
        [Parameter(Mandatory = $true)][string]$NotesRelativePath
    )

    $Path = Join-Path $Root "CHANGELOG.md"
    $Text = Read-Text $Path

    if (
        $Text -match (
            "(?m)^##\s+" +
            [regex]::Escape($ReleaseVersion) +
            "(?:\s|$)"
        )
    ) {
        return
    }

    if (-not $Text.StartsWith("# Changelog")) {
        Fail "CHANGELOG.md has an unexpected heading."
    }

    $Date = Get-Date -Format "yyyy-MM-dd"

    $Section = @"

## $ReleaseVersion - $Date

See the [ZeroModel $ReleaseVersion release notes]($NotesRelativePath).
"@

    $Updated = (
        "# Changelog" +
        $Section +
        $Text.Substring("# Changelog".Length)
    )

    Write-Text -Path $Path -Content $Updated
}


function Invoke-ReleaseGates {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$PythonCommand
    )

    Run $PythonCommand @("-m", "pip", "install", "--upgrade", "pip")
    Run $PythonCommand @("-m", "pip", "install", "build", "twine")

    if (-not $SkipReleaseValidator) {
        Step "Authoritative release-candidate validator"
        Run $PythonCommand @("scripts/validate_release_candidate.py")
    }

    if (-not $SkipQuality) {
        Step "Repository quality gate"
        Run $PythonCommand @("scripts/check_quality.py")
    }

    if (-not $SkipTests) {
        Step "Bounded fast suite"
        Run $PythonCommand @("scripts/run_fast_tests.py")
    }
}


function Wait-ForCommitCI {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryName,
        [Parameter(Mandatory = $true)][string]$Commit
    )

    Step "Waiting for GitHub Actions on $Commit"

    $Runs = $null

    for ($Attempt = 1; $Attempt -le 30; $Attempt++) {
        $Json = & gh run list `
            --repo $RepositoryName `
            --commit $Commit `
            --limit 100 `
            --json databaseId,name,status,conclusion,event 2>$null

        if ($LASTEXITCODE -eq 0 -and $Json) {
            $CandidateRuns = @($Json | ConvertFrom-Json)

            $Runs = @(
                $CandidateRuns |
                    Where-Object {
                        $_.event -in @(
                            "push",
                            "workflow_dispatch",
                            "pull_request"
                        )
                    }
            )

            if ($Runs.Count -gt 0) {
                break
            }
        }

        Start-Sleep -Seconds 5
    }

    if (-not $Runs -or $Runs.Count -eq 0) {
        Fail "No GitHub Actions runs found for commit $Commit."
    }

    foreach ($Run in $Runs) {
        if ($Run.status -ne "completed") {
            Write-Host "Watching workflow: $($Run.name)"

            Run "gh" @(
                "run",
                "watch",
                [string]$Run.databaseId,
                "--repo",
                $RepositoryName,
                "--exit-status"
            )
        }
    }

    $FinalJson = Capture "gh" @(
        "run",
        "list",
        "--repo",
        $RepositoryName,
        "--commit",
        $Commit,
        "--limit",
        "100",
        "--json",
        "databaseId,name,status,conclusion,event"
    )

    $FinalRuns = @(
        $FinalJson |
            ConvertFrom-Json |
            Where-Object {
                $_.event -in @(
                    "push",
                    "workflow_dispatch",
                    "pull_request"
                )
            }
    )

    $Failures = @(
        $FinalRuns |
            Where-Object {
                $_.status -ne "completed" -or
                $_.conclusion -notin @("success", "skipped", "neutral")
            }
    )

    if ($Failures.Count -gt 0) {
        $Summary = (
            $Failures |
                ForEach-Object {
                    "  $($_.name): status=$($_.status) conclusion=$($_.conclusion)"
                }
        ) -join "`n"

        Fail "GitHub Actions did not pass:`n$Summary"
    }

    Write-Host "GitHub Actions passed for $Commit." -ForegroundColor Green
}


function Ensure-Tag {
    param(
        [Parameter(Mandatory = $true)][string]$Tag,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)][string]$RemoteName
    )

    $LocalTag = & git rev-parse -q --verify "$Tag^{commit}" 2>$null

    if ($LASTEXITCODE -eq 0) {
        $Resolved = (($LocalTag | Out-String).Trim())

        if ($Resolved -ne $Commit) {
            Fail "Local tag $Tag points to $Resolved, expected $Commit."
        }
    }
    else {
        Run "git" @(
            "tag",
            "-a",
            $Tag,
            "-m",
            "ZeroModel $Version",
            $Commit
        )
    }

    $RemoteTags = & git ls-remote `
        --tags `
        $RemoteName `
        "refs/tags/$Tag" `
        "refs/tags/$Tag^{}" 2>$null

    if ($LASTEXITCODE -ne 0) {
        Fail "Could not inspect remote tag $Tag."
    }

    if ($RemoteTags) {
        $Peeled = @(
            $RemoteTags |
                Where-Object {
                    $_ -match (
                        "refs/tags/" +
                        [regex]::Escape($Tag) +
                        "\^\{\}$"
                    )
                }
        ) | Select-Object -First 1

        $Selected = if ($Peeled) {
            $Peeled
        }
        else {
            @($RemoteTags) | Select-Object -First 1
        }

        $RemoteCommit = (($Selected -split "`t")[0]).Trim()

        if ($RemoteCommit -ne $Commit) {
            Fail @"
Remote tag $Tag points to $RemoteCommit,
expected $Commit.
"@
        }
    }
    else {
        Run "git" @("push", $RemoteName, $Tag)
    }
}


function Ensure-GitHubRelease {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryName,
        [Parameter(Mandatory = $true)][string]$Tag,
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$NotesPath,
        [Parameter(Mandatory = $true)][string]$ArtifactRoot
    )

    & gh release view $Tag --repo $RepositoryName *> $null

    if ($LASTEXITCODE -eq 0) {
        Write-Host "GitHub release $Tag already exists."
        return
    }

    $Artifacts = @(
        Get-ChildItem -LiteralPath $ArtifactRoot -File |
            Sort-Object Name |
            ForEach-Object { $_.FullName }
    )

    if ($Artifacts.Count -eq 0) {
        Fail "No release artifacts found in $ArtifactRoot."
    }

    Run "gh" (
        @(
            "release",
            "create",
            $Tag,
            "--repo",
            $RepositoryName,
            "--title",
            $Title,
            "--notes-file",
            $NotesPath,
            "--verify-tag"
        ) +
        $Artifacts
    )
}


$Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

if (-not $ReleaseNotesPath) {
    $ReleaseNotesPath = "docs/releases/$Version.md"
}

$NotesFullPath = if ([IO.Path]::IsPathRooted($ReleaseNotesPath)) {
    [IO.Path]::GetFullPath($ReleaseNotesPath)
}
else {
    [IO.Path]::GetFullPath(
        (Join-Path $Root $ReleaseNotesPath)
    )
}

$NotesRelativePath = $NotesFullPath.Substring($Root.Length)
$NotesRelativePath = $NotesRelativePath.TrimStart(
    [char[]]@("\", "/")
)
$NotesRelativePath = $NotesRelativePath.Replace("\", "/")

$ReleaseBranch = "release/$Version"
$Tag = "v$Version"
$DistRoot = Join-Path $Root "dist\release-$Version"


Push-Location $Root

try {
    Need "git"
    Need $Python

    $RepositoryRoot = [IO.Path]::GetFullPath(
        (Capture "git" @("rev-parse", "--show-toplevel"))
    )

    if ($RepositoryRoot.TrimEnd("\") -ne $Root.TrimEnd("\")) {
        Fail "Run this script from the ZeroModel repository."
    }

    Assert-Clean

    if (-not (Test-Path -LiteralPath $NotesFullPath -PathType Leaf)) {
        Fail @"
Release notes not found:

$NotesFullPath

Create the release notes before running the release workflow.
"@
    }

    $Packages = Get-PackageDefinitions $Root

    if ($Mode -eq "Prepare") {
        $BaseCommit = Assert-SynchronizedBranch `
            -RemoteName $Remote `
            -BranchName $BaseBranch

        $CurrentVersion = Get-BoundaryVersion $Root

        if ([version]$Version -le [version]$CurrentVersion) {
            Fail @"
Release version $Version must be greater than
current version $CurrentVersion.
"@
        }

        Step "ZeroModel $Version release preparation"

        Write-Host "Base commit: $BaseCommit"
        Write-Host "Release branch: $ReleaseBranch"
        Write-Host "Release notes: $NotesRelativePath"
        Write-Host "Publishable packages: $($Packages.Count)"

        if ($DryRun) {
            Write-Host ""
            Write-Host "Dry run complete; nothing changed." `
                -ForegroundColor Yellow
            return
        }

        Need "gh"
        Run "gh" @("auth", "status")

        Run "git" @("switch", "-c", $ReleaseBranch)

        Set-WorkspaceVersion `
            -Root $Root `
            -Packages $Packages `
            -ReleaseVersion $Version

        Update-Changelog `
            -Root $Root `
            -ReleaseVersion $Version `
            -NotesRelativePath $NotesRelativePath

        Assert-WorkspaceVersion `
            -Root $Root `
            -Packages $Packages `
            -ExpectedVersion $Version

        Invoke-ReleaseGates `
            -Root $Root `
            -PythonCommand $Python

        $Changes = Capture "git" @("status", "--porcelain")

        if (-not $Changes) {
            Fail "Release preparation produced no changes."
        }

        Step "Committing release metadata and validation evidence"

        Run "git" @("add", "--all")
        Run "git" @(
            "commit",
            "-m",
            "chore(release): prepare ZeroModel $Version"
        )
        Run "git" @(
            "push",
            "-u",
            $Remote,
            $ReleaseBranch
        )

        $Body = @"
## Objective

Prepare ZeroModel $Version across the complete package workspace.

## Included

- updates package-boundaries.toml release_version;
- updates all publishable package versions;
- updates exact internal ZeroModel dependency pins;
- updates release documentation;
- records generated release-candidate evidence;
- validates quality, fast tests, package builds, integration tests, visual-transition tests, and clean-wheel imports.

## Publication

This PR does not publish packages, create a tag, or create a GitHub release.

After merge, run:

``````powershell
.\scripts\create-release.ps1 `
    -Mode Publish `
    -Version $Version
```````

"@

```
    Step "Opening release pull request"

    Run "gh" @(
        "pr",
        "create",
        "--repo",
        $Repository,
        "--base",
        $BaseBranch,
        "--head",
        $ReleaseBranch,
        "--title",
        "chore(release): prepare ZeroModel $Version",
        "--body",
        $Body
    )

    Write-Host ""
    Write-Host (
        "Release PR created for ZeroModel $Version."
    ) -ForegroundColor Green

    return
}

$ReleaseCommit = Assert-SynchronizedBranch `
    -RemoteName $Remote `
    -BranchName $BaseBranch

Assert-WorkspaceVersion `
    -Root $Root `
    -Packages $Packages `
    -ExpectedVersion $Version

Step "ZeroModel $Version publication preflight"

Write-Host "Release commit: $ReleaseCommit"
Write-Host "Release tag: $Tag"
Write-Host "Release notes: $NotesRelativePath"

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run complete; nothing changed." `
        -ForegroundColor Yellow
    return
}

Need "gh"
Run "gh" @("auth", "status")

Invoke-ReleaseGates `
    -Root $Root `
    -PythonCommand $Python

$StatusAfterGates = Capture "git" @("status", "--porcelain")

if ($StatusAfterGates) {
    Fail @"
```

Release gates changed the working tree.

Commit those generated results before publication:

$StatusAfterGates
"@
}

```
if (-not $SkipCI) {
    Wait-ForCommitCI `
        -RepositoryName $Repository `
        -Commit $ReleaseCommit
}

if (-not $Yes) {
    Write-Host ""
    Write-Host (
        "This will publish ten immutable package versions,"
    ) -ForegroundColor Yellow
    Write-Host (
        "create tag $Tag, and create a GitHub release."
    ) -ForegroundColor Yellow

    $Confirmation = Read-Host (
        "Type RELEASE $Version to continue"
    )

    if ($Confirmation -ne "RELEASE $Version") {
        Fail "Release cancelled."
    }
}

if (-not $SkipPyPI) {
    Step "Publishing complete package workspace to PyPI"

    $PublishArguments = @(
        "-NoProfile",
        "-File",
        (Join-Path $Root "scripts\publish-pypi.ps1"),
        "-Version",
        $Version,
        "-Python",
        $Python,
        "-SkipValidation",
        "-Yes"
    )

    if ($ResumePartialPyPI) {
        $PublishArguments += "-ResumePartial"
    }

    Run "powershell" $PublishArguments
}

Step "Creating and pushing release tag"

Ensure-Tag `
    -Tag $Tag `
    -Commit $ReleaseCommit `
    -RemoteName $Remote

if (-not $SkipGitHubRelease) {
    Step "Creating GitHub release"

    Ensure-GitHubRelease `
        -RepositoryName $Repository `
        -Tag $Tag `
        -Title "ZeroModel $Version" `
        -NotesPath $NotesFullPath `
        -ArtifactRoot $DistRoot
}

Step "Release complete"

Write-Host ""
Write-Host (
    "ZeroModel $Version has been released."
) -ForegroundColor Green
Write-Host "Commit: $ReleaseCommit"
Write-Host "Tag: $Tag"
```

}
finally {
Pop-Location
}
