[CmdletBinding()]
param(
    [ValidateSet("Prepare", "Publish")]
    [string]$Mode = "Prepare",

    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version = "1.0.12",

    [string]$Python = "python",
    [string]$Remote = "origin",
    [string]$BaseBranch = "main",
    [string]$Repository = "ernanhughes/zeromodel",
    [string]$ReleaseNotesPath = "",

    [switch]$SkipQuality,
    [switch]$SkipTests,
    [switch]$SkipDemos,
    [switch]$SkipFinalizationIntegration,
    [switch]$SkipPyPI,
    [switch]$SkipGitHubRelease,
    [switch]$Yes,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Fail([string]$Message) {
    throw $Message
}

function Run([string]$Command, [string[]]$Arguments = @()) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "Command failed ($LASTEXITCODE): $Command $($Arguments -join ' ')"
    }
}

function Capture([string]$Command, [string[]]$Arguments = @()) {
    $Output = & $Command @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        Fail "Command failed ($LASTEXITCODE): $Command $($Arguments -join ' ')`n$(($Output | Out-String).Trim())"
    }
    return (($Output | Out-String).Trim())
}

function Need([string]$Command) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        Fail "Required command '$Command' was not found on PATH."
    }
}

function Read-Text([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Fail "Required file not found: $Path"
    }
    return [IO.File]::ReadAllText($Path)
}

function Write-Text([string]$Path, [string]$Content) {
    [IO.File]::WriteAllText($Path, $Content, (New-Object Text.UTF8Encoding($false)))
}

function Replace-One([string]$Path, [string]$Pattern, [string]$Replacement) {
    $Text = Read-Text $Path
    $Matches = [regex]::Matches($Text, $Pattern)
    if ($Matches.Count -ne 1) {
        Fail "Expected one version marker in '$Path'; found $($Matches.Count)."
    }
    Write-Text $Path ([regex]::Replace($Text, $Pattern, $Replacement, 1))
}

function Project-Version([string]$Root) {
    $Match = [regex]::Match((Read-Text (Join-Path $Root "pyproject.toml")), '(?m)^version\s*=\s*"([^"]+)"\s*$')
    if (-not $Match.Success) { Fail "Could not read pyproject.toml version." }
    return $Match.Groups[1].Value
}

function Runtime-Version([string]$Root) {
    $Match = [regex]::Match((Read-Text (Join-Path $Root "zeromodel\__init__.py")), '(?m)^__version__\s*=\s*["'']([^"'']+)["'']\s*$')
    if (-not $Match.Success) { Fail "Could not read zeromodel.__version__." }
    return $Match.Groups[1].Value
}

function Assert-Clean {
    $Status = Capture "git" @("status", "--porcelain")
    if ($Status) { Fail "Working tree is not clean.`n$Status" }
}

function Assert-SyncedMain([string]$RemoteName, [string]$BranchName) {
    [void](Run "git" @("fetch", "--prune", "--tags", $RemoteName))
    $Branch = Capture "git" @("branch", "--show-current")
    if ($Branch -ne $BranchName) { Fail "Expected '$BranchName'; current branch is '$Branch'." }
    $Head = Capture "git" @("rev-parse", "HEAD")
    $RemoteHead = Capture "git" @("rev-parse", "$RemoteName/$BranchName")
    if ($Head -ne $RemoteHead) { Fail "Local and remote $BranchName differ. Local=$Head Remote=$RemoteHead" }
    return $Head
}

function Update-Changelog([string]$Root, [string]$ReleaseVersion, [string]$NotesPath) {
    $Path = Join-Path $Root "CHANGELOG.md"
    $Text = Read-Text $Path
    if ($Text -match "(?m)^##\s+$([regex]::Escape($ReleaseVersion))(?:\s|$)") { return }
    if (-not $Text.StartsWith("# Changelog")) { Fail "CHANGELOG.md has an unexpected heading." }
    $Date = Get-Date -Format "yyyy-MM-dd"
    $Section = "`r`n`r`n## $ReleaseVersion - $Date`r`n`r`nSee the [ZeroModel $ReleaseVersion release notes]($NotesPath)."
    Write-Text $Path ("# Changelog" + $Section + $Text.Substring("# Changelog".Length))
}

function Set-ReleaseFiles([string]$Root, [string]$ReleaseVersion, [string]$NotesPath) {
    Replace-One (Join-Path $Root "pyproject.toml") '(?m)^version\s*=\s*"[^"]+"\s*$' "version = `"$ReleaseVersion`""
    Replace-One (Join-Path $Root "zeromodel\__init__.py") '(?m)^__version__\s*=\s*["''][^"'']+["'']\s*$' "__version__ = `"$ReleaseVersion`""
    Replace-One (Join-Path $Root "README.md") '(?m)python -m pip install zeromodel==[0-9]+\.[0-9]+\.[0-9]+' "python -m pip install zeromodel==$ReleaseVersion"
    Update-Changelog $Root $ReleaseVersion $NotesPath
}

function Assert-ReleaseFiles([string]$Root, [string]$ReleaseVersion) {
    $Project = Project-Version $Root
    $Runtime = Runtime-Version $Root
    if ($Project -ne $Runtime -or $Project -ne $ReleaseVersion) {
        Fail "Expected both package versions to be $ReleaseVersion; found project=$Project runtime=$Runtime."
    }
    if ((Read-Text (Join-Path $Root "README.md")) -notmatch [regex]::Escape("zeromodel==$ReleaseVersion")) {
        Fail "README production install pin is not $ReleaseVersion."
    }
    if ((Read-Text (Join-Path $Root "CHANGELOG.md")) -notmatch "(?m)^##\s+$([regex]::Escape($ReleaseVersion))(?:\s|$)") {
        Fail "CHANGELOG.md has no $ReleaseVersion section."
    }
}

function Gates([string]$Root, [string]$PythonCommand) {
    Step "Installing release dependencies"
    Run $PythonCommand @("-m", "pip", "install", "--upgrade", "pip")
    Run $PythonCommand @("-m", "pip", "install", "-e", ".[dev,release]")

    if (-not $SkipQuality) {
        Step "Repository quality gate"
        Run $PythonCommand @("scripts/check_quality.py")
    }
    if (-not $SkipTests) {
        Step "Bounded fast suite"
        Run $PythonCommand @("scripts/run_fast_tests.py")
    }
    if (-not $SkipDemos) {
        Step "Release demos"
        Run $PythonCommand @("examples/arcade_shooter_policy.py")
        Run $PythonCommand @("examples/criticality_verification.py", "--output-dir", (Join-Path $Root "build\release\criticality-verification"))
    }

    Step "Building distributions"
    Remove-Item -Recurse -Force (Join-Path $Root "dist"), (Join-Path $Root "build") -ErrorAction SilentlyContinue
    Get-ChildItem -Path $Root -Directory -Filter "*.egg-info" | Remove-Item -Recurse -Force
    Run $PythonCommand @("-m", "build")

    Step "Checking distribution metadata"
    $Artifacts = Get-ChildItem -Path (Join-Path $Root "dist") -File
    if (-not $Artifacts) { Fail "Build produced no distribution files." }
    Run $PythonCommand (@("-m", "twine", "check") + @($Artifacts | ForEach-Object { $_.FullName }))
}

function Shell-Command {
    if (Get-Command "pwsh" -ErrorAction SilentlyContinue) { return "pwsh" }
    if (Get-Command "powershell" -ErrorAction SilentlyContinue) { return "powershell" }
    Fail "Neither pwsh nor powershell is available."
}

function Finalization-Gate([string]$Root, [string]$Commit) {
    Step "Bounded video-finalization integration validation"
    Run (Shell-Command) @(
        "-NoProfile",
        "-File", (Join-Path $Root "scripts\run-video-finalization-integration.ps1"),
        "-RepositoryPath", $Root,
        "-ExpectedCommit", $Commit,
        "-ExpectedBranch", $BaseBranch,
        "-Python", $Python,
        "-RemoveSuccessfulFixtures"
    )
}

function PyPI-Exists([string]$PythonCommand, [string]$ReleaseVersion) {
    $Code = @"
import sys, urllib.error, urllib.request
try:
    urllib.request.urlopen('https://pypi.org/pypi/zeromodel/$ReleaseVersion/json', timeout=20)
    sys.exit(0)
except urllib.error.HTTPError as exc:
    sys.exit(1 if exc.code == 404 else 2)
"@
    & $PythonCommand -c $Code
    if ($LASTEXITCODE -eq 0) { return $true }
    if ($LASTEXITCODE -eq 1) { return $false }
    Fail "Could not query PyPI for zeromodel==$ReleaseVersion."
}

function Wait-CI([string]$Repo, [string]$Commit) {
    Step "Waiting for GitHub Actions"
    $RunId = $null
    for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
        $Json = & gh run list --repo $Repo --commit $Commit --workflow python.yml --limit 1 --json databaseId 2>$null
        if ($LASTEXITCODE -eq 0 -and $Json) {
            $Runs = $Json | ConvertFrom-Json
            if ($Runs -and $Runs.Count -gt 0) { $RunId = [string]$Runs[0].databaseId; break }
        }
        Start-Sleep -Seconds 2
    }
    if (-not $RunId) { Fail "No package workflow appeared for commit $Commit." }
    Run "gh" @("run", "watch", $RunId, "--repo", $Repo, "--exit-status")
}

function Ensure-Tag([string]$Tag, [string]$Commit, [string]$RemoteName) {
    $Local = & git rev-parse -q --verify "$Tag^{commit}" 2>$null
    if ($LASTEXITCODE -eq 0) {
        if ((($Local | Out-String).Trim()) -ne $Commit) { Fail "Local tag $Tag points at the wrong commit." }
    }
    else {
        Run "git" @("tag", "-a", $Tag, "-m", "ZeroModel $Version", $Commit)
    }

    $RemoteTags = & git ls-remote --tags $RemoteName "refs/tags/$Tag" "refs/tags/$Tag^{}" 2>$null
    if ($LASTEXITCODE -ne 0) { Fail "Could not inspect remote tag $Tag." }
    if ($RemoteTags) {
        $Peeled = $RemoteTags | Where-Object { $_ -match "refs/tags/$([regex]::Escape($Tag))\^\{\}$" } | Select-Object -First 1
        $Selected = if ($Peeled) { $Peeled } else { $RemoteTags | Select-Object -First 1 }
        if (((($Selected -split "`t")[0]).Trim()) -ne $Commit) { Fail "Remote tag $Tag points at the wrong commit." }
    }
    else {
        Run "git" @("push", $RemoteName, $Tag)
    }
}

$Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $ReleaseNotesPath) { $ReleaseNotesPath = "docs/releases/$Version.md" }
$NotesFull = if ([IO.Path]::IsPathRooted($ReleaseNotesPath)) {
    [IO.Path]::GetFullPath($ReleaseNotesPath)
}
else {
    [IO.Path]::GetFullPath((Join-Path $Root $ReleaseNotesPath))
}
$NotesRelative = $NotesFull.Substring($Root.Length).TrimStart([char[]]@('\', '/')).Replace('\', '/')
$ReleaseBranch = "release/$Version"
$Tag = "v$Version"

Push-Location $Root
try {
    Need "git"
    Need $Python
    if (([IO.Path]::GetFullPath((Capture "git" @("rev-parse", "--show-toplevel")))) -ne $Root) {
        Fail "This script must run from the ZeroModel repository."
    }
    Assert-Clean
    if (-not (Test-Path -LiteralPath $NotesFull -PathType Leaf)) { Fail "Release notes not found: $NotesFull" }

    $BaseCommit = Assert-SyncedMain $Remote $BaseBranch
    Step "ZeroModel $Version $Mode preflight"
    Write-Host "Base commit: $BaseCommit"
    Write-Host "Release notes: $NotesRelative"
    if ($DryRun) { Write-Host "Dry run complete; nothing changed." -ForegroundColor Yellow; return }

    Need "gh"
    Run "gh" @("auth", "status")

    if ($Mode -eq "Prepare") {
        $Current = Project-Version $Root
        if ($Current -ne (Runtime-Version $Root)) { Fail "Current package versions do not match." }
        if ([version]$Version -le [version]$Current) { Fail "$Version must be greater than current version $Current." }

        Step "Creating $ReleaseBranch"
        Run "git" @("switch", "-c", $ReleaseBranch)
        Set-ReleaseFiles $Root $Version $NotesRelative
        Assert-ReleaseFiles $Root $Version
        Gates $Root $Python

        Step "Committing release preparation"
        Run "git" @("add", "--", "pyproject.toml", "zeromodel/__init__.py", "README.md", "CHANGELOG.md", $NotesRelative)
        Run "git" @("diff", "--cached", "--check")
        Run "git" @("commit", "-m", "release: ZeroModel $Version")
        Run "git" @("push", "-u", $Remote, $ReleaseBranch)

        $BodyPath = Join-Path $env:TEMP "zeromodel-release-$Version-pr.md"
        $Body = @(
            "## Summary", "",
            "Prepare ZeroModel $Version as the final monolithic 1.x release.", "",
            "## Validation", "",
            "- quality gate", "- bounded fast suite", "- release demos", "- wheel and sdist build", "- twine check", "",
            "After merge run: .\scripts\create-release.ps1 -Mode Publish -Version $Version"
        ) -join [Environment]::NewLine
        Write-Text $BodyPath $Body
        Run "gh" @("pr", "create", "--repo", $Repository, "--base", $BaseBranch, "--head", $ReleaseBranch, "--title", "release: ZeroModel $Version", "--body-file", $BodyPath)
        Write-Host "Release PR created." -ForegroundColor Green
        return
    }

    Assert-ReleaseFiles $Root $Version
    $Commit = Capture "git" @("rev-parse", "HEAD")
    if (-not $Yes) {
        $Confirm = Read-Host "Type RELEASE $Version to publish"
        if ($Confirm -ne "RELEASE $Version") { Fail "Release cancelled." }
    }

    Gates $Root $Python
    if (-not $SkipFinalizationIntegration) { Assert-Clean; Finalization-Gate $Root $Commit }

    Step "Validating merged commit on GitHub"
    Run "git" @("push", $Remote, $BaseBranch)
    Wait-CI $Repository $Commit

    if (-not $SkipPyPI) {
        if (PyPI-Exists $Python $Version) {
            Write-Host "zeromodel==$Version already exists on PyPI; upload skipped." -ForegroundColor Yellow
        }
        else {
            Step "Publishing to PyPI"
            Run (Shell-Command) @("-NoProfile", "-File", (Join-Path $Root "scripts\publish-pypi.ps1"), "-Python", $Python, "-SkipTests", "-SkipDemo", "-Yes")
        }
    }

    Step "Creating $Tag"
    Ensure-Tag $Tag $Commit $Remote

    if (-not $SkipGitHubRelease) {
        & gh release view $Tag --repo $Repository 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "GitHub release already exists; creation skipped." -ForegroundColor Yellow
        }
        else {
            $Assets = Get-ChildItem -Path (Join-Path $Root "dist") -File | ForEach-Object { $_.FullName }
            Run "gh" (@("release", "create", $Tag, "--repo", $Repository, "--title", "ZeroModel $Version", "--notes-file", $NotesFull, "--verify-tag") + @($Assets))
        }
    }

    Write-Host "ZeroModel $Version release complete." -ForegroundColor Green
    Write-Host "Next development version: 2.0.0.dev0" -ForegroundColor Green
}
finally {
    Pop-Location
}
