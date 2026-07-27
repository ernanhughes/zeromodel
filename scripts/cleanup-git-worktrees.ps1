[CmdletBinding()]
param(
    # Actually perform the deletions.
    # Without this switch, the script only previews the changes.
    [switch]$Execute,

    # Allow removal of dirty or locked worktrees.
    # This can permanently discard uncommitted files.
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Git {
    [CmdletBinding()]
    param(
        [Parameter(
            Mandatory = $true,
            Position = 0,
            ValueFromRemainingArguments = $true
        )]
        [string[]]$GitArguments
    )

    $output = @(& git @GitArguments 2>&1)

    if ($LASTEXITCODE -ne 0) {
        $command = "git " + ($GitArguments -join " ")
        $message = $output -join [Environment]::NewLine

        throw "$command failed:`n$message"
    }

    return $output
}

function Normalize-FileSystemPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)

    return $fullPath.TrimEnd(
        [char[]]@(
            [System.IO.Path]::DirectorySeparatorChar,
            [System.IO.Path]::AltDirectorySeparatorChar
        )
    )
}

function Get-GitWorktrees {
    $lines = @(Invoke-Git worktree list --porcelain)
    $worktrees = @()
    $entry = $null

    # Add an empty line to flush the final entry.
    foreach ($line in @($lines) + "") {
        if ([string]::IsNullOrWhiteSpace($line)) {
            if ($null -ne $entry) {
                $worktrees += [PSCustomObject]$entry
                $entry = $null
            }

            continue
        }

        if ($line.StartsWith("worktree ")) {
            $entry = [ordered]@{
                Path       = $line.Substring("worktree ".Length)
                Branch     = $null
                Detached   = $false
                Bare       = $false
                Locked     = $false
                LockReason = $null
                Prunable   = $false
            }

            continue
        }

        if ($null -eq $entry) {
            continue
        }

        if ($line.StartsWith("branch ")) {
            $branchReference = $line.Substring("branch ".Length)
            $entry.Branch = $branchReference -replace "^refs/heads/", ""
        }
        elseif ($line -eq "detached") {
            $entry.Detached = $true
        }
        elseif ($line -eq "bare") {
            $entry.Bare = $true
        }
        elseif ($line.StartsWith("locked")) {
            $entry.Locked = $true

            if ($line.Length -gt "locked".Length) {
                $entry.LockReason = $line.Substring("locked".Length).Trim()
            }
        }
        elseif ($line.StartsWith("prunable")) {
            $entry.Prunable = $true
        }
    }

    return $worktrees
}

# ---------------------------------------------------------------------
# Validate repository and determine what must be kept
# ---------------------------------------------------------------------

$repositoryRootOutput = @(Invoke-Git rev-parse --show-toplevel)

if ($repositoryRootOutput.Count -eq 0) {
    throw "Unable to determine the repository root."
}

$repositoryRoot = Normalize-FileSystemPath $repositoryRootOutput[0]

$currentBranchOutput = @(Invoke-Git branch --show-current)
$currentBranch = ($currentBranchOutput -join "").Trim()

if ([string]::IsNullOrWhiteSpace($currentBranch)) {
    throw @"
The current worktree is in detached HEAD state.

Check out the branch you want to keep before running this script:

    git switch <branch-name>
"@
}

$mainExistsOutput = @(
    & git show-ref --verify --quiet "refs/heads/main" 2>&1
)

if ($LASTEXITCODE -ne 0) {
    throw "The local 'main' branch does not exist. No changes were made."
}

$worktrees = @(Get-GitWorktrees)

if ($worktrees.Count -eq 0) {
    throw "Git returned no worktrees."
}

# The primary worktree should be the first porcelain entry.
$primaryWorktreePath = Normalize-FileSystemPath $worktrees[0].Path

if ($repositoryRoot -ine $primaryWorktreePath) {
    throw @"
Run this script from the repository's primary worktree.

Current worktree:
    $repositoryRoot

Primary worktree:
    $primaryWorktreePath

This restriction prevents the script from attempting to remove the primary
working directory while it is being executed from a linked worktree.
"@
}

$worktreesToRemove = @(
    $worktrees | Where-Object {
        (Normalize-FileSystemPath $_.Path) -ine $repositoryRoot
    }
)

$branches = @(
    Invoke-Git for-each-ref "--format=%(refname:short)" refs/heads |
        ForEach-Object { $_.Trim() } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)

$branchesToKeep = @(
    "main"
    $currentBranch
) | Select-Object -Unique

$branchesToDelete = @(
    $branches | Where-Object {
        $branch = $_
        -not ($branchesToKeep | Where-Object { $_ -ieq $branch })
    }
)

# ---------------------------------------------------------------------
# Inspect linked worktrees before deleting anything
# ---------------------------------------------------------------------

$unsafeWorktrees = @()

foreach ($worktree in $worktreesToRemove) {
    $exists = Test-Path -LiteralPath $worktree.Path

    if (-not $exists) {
        # A missing worktree path is normally a stale/prunable registration.
        continue
    }

    $statusOutput = @(
        & git -C $worktree.Path status --porcelain --untracked-files=all 2>&1
    )

    if ($LASTEXITCODE -ne 0) {
        throw @"
Could not inspect worktree:

    $($worktree.Path)

Git reported:

$($statusOutput -join [Environment]::NewLine)
"@
    }

    $isDirty = $statusOutput.Count -gt 0

    if ($isDirty -or $worktree.Locked) {
        $unsafeWorktrees += [PSCustomObject]@{
            Path       = $worktree.Path
            Branch     = $worktree.Branch
            Dirty      = $isDirty
            Locked     = $worktree.Locked
            LockReason = $worktree.LockReason
        }
    }
}

# ---------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------

Write-Host ""
Write-Host "Git cleanup preview" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Current worktree kept:"
Write-Host "  $repositoryRoot" -ForegroundColor Green

Write-Host ""
Write-Host "Branches kept:"
foreach ($branch in $branchesToKeep) {
    Write-Host "  $branch" -ForegroundColor Green
}

Write-Host ""
Write-Host "Worktrees to remove: $($worktreesToRemove.Count)"

foreach ($worktree in $worktreesToRemove) {
    $branchDisplay = if ($worktree.Branch) {
        $worktree.Branch
    }
    elseif ($worktree.Detached) {
        "<detached HEAD>"
    }
    else {
        "<unknown>"
    }

    Write-Host "  $($worktree.Path)" -ForegroundColor Yellow
    Write-Host "    Branch: $branchDisplay"
}

Write-Host ""
Write-Host "Local branches to delete: $($branchesToDelete.Count)"

foreach ($branch in $branchesToDelete) {
    Write-Host "  $branch" -ForegroundColor Yellow
}

if ($unsafeWorktrees.Count -gt 0) {
    Write-Host ""
    Write-Host "Worktrees requiring -Force:" -ForegroundColor Red

    foreach ($worktree in $unsafeWorktrees) {
        Write-Host "  $($worktree.Path)" -ForegroundColor Red
        Write-Host "    Branch: $($worktree.Branch)"
        Write-Host "    Dirty:  $($worktree.Dirty)"
        Write-Host "    Locked: $($worktree.Locked)"

        if ($worktree.LockReason) {
            Write-Host "    Lock reason: $($worktree.LockReason)"
        }
    }
}

if (-not $Execute) {
    Write-Host ""
    Write-Host "Preview only. Nothing was deleted." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Run the cleanup with:"
    Write-Host "  .\cleanup-git-worktrees.ps1 -Execute" -ForegroundColor Green
    Write-Host ""

    if ($unsafeWorktrees.Count -gt 0) {
        Write-Host "To deliberately discard dirty or locked worktrees:"
        Write-Host "  .\cleanup-git-worktrees.ps1 -Execute -Force" -ForegroundColor Red
        Write-Host ""
    }

    exit 0
}

if (($unsafeWorktrees.Count -gt 0) -and (-not $Force)) {
    throw @"
One or more worktrees contain uncommitted files or are locked.

No worktrees or branches were deleted.

Review them, commit anything valuable, and run again. To deliberately discard
their contents, use:

    .\cleanup-git-worktrees.ps1 -Execute -Force
"@
}

# ---------------------------------------------------------------------
# Remove linked worktrees
# ---------------------------------------------------------------------

Write-Host ""
Write-Host "Removing linked worktrees..." -ForegroundColor Cyan

foreach ($worktree in $worktreesToRemove) {
    if (-not (Test-Path -LiteralPath $worktree.Path)) {
        Write-Host "  Skipping missing path; stale record will be pruned:"
        Write-Host "    $($worktree.Path)"
        continue
    }

    if ($worktree.Locked) {
        if (-not $Force) {
            throw "Locked worktree encountered without -Force: $($worktree.Path)"
        }

        Write-Host "  Unlocking $($worktree.Path)"
        Invoke-Git worktree unlock $worktree.Path | Out-Null
    }

    Write-Host "  Removing $($worktree.Path)"

    if ($Force) {
        Invoke-Git worktree remove --force $worktree.Path | Out-Null
    }
    else {
        Invoke-Git worktree remove $worktree.Path | Out-Null
    }
}

Write-Host "  Pruning stale worktree registrations"
Invoke-Git worktree prune --expire now | Out-Null

# ---------------------------------------------------------------------
# Delete all local branches except main and the current branch
# ---------------------------------------------------------------------

Write-Host ""
Write-Host "Deleting local branches..." -ForegroundColor Cyan

foreach ($branch in $branchesToDelete) {
    Write-Host "  Deleting $branch"

    # -D is intentional: the user requested deletion even when Git does not
    # consider the branch merged.
    Invoke-Git -GitArguments @("branch", "-D", "--", $branch) | Out-Null
}

# ---------------------------------------------------------------------
# Verify final state
# ---------------------------------------------------------------------

Write-Host ""
Write-Host "Cleanup complete." -ForegroundColor Green

Write-Host ""
Write-Host "Remaining worktrees:"
Invoke-Git worktree list | ForEach-Object {
    Write-Host "  $_"
}

Write-Host ""
Write-Host "Remaining local branches:"
Invoke-Git branch --format="%(refname:short)" | ForEach-Object {
    Write-Host "  $_"
}

Write-Host ""
Write-Host "Remote branches were not changed." -ForegroundColor Cyan
