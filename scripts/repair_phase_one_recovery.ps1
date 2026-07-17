#requires -Version 7.0
<#
.SYNOPSIS
Restore the frozen Phase 1 recovered files to the exact bytes declared by
recovery-manifest.json.

.DESCRIPTION
The script ignores the current working-tree copies. For every recovered file it:

1. Searches the file's Git history.
2. Tests the raw Git blob, LF form, and CRLF form.
3. Finds the exact byte sequence whose size and SHA-256 match the manifest.
4. Writes those exact bytes to the working tree.
5. Ensures Git treats the recovery directory as binary (-text).
6. Repairs the dino-full-run.log .gitignore exception.
7. Stages the recovered files.
8. Verifies the exact staged Git blobs against the manifest.

It does not change recovery-manifest.json.
#>

[CmdletBinding()]
param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
Set-Location -LiteralPath $repo

$manifest = "docs/results/visual-address-phase-one-v1/recovery-manifest.json"
$recovered = "docs/results/visual-address-phase-one-v1/recovered-originals"
$attributes = ".gitattributes"
$ignore = ".gitignore"

if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "Manifest not found: $manifest"
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not available on PATH."
}

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python command is not available: $Python"
}

Write-Host "Repairing frozen Phase 1 recovery files..." -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Ensure byte-preserving Git attributes.
# ---------------------------------------------------------------------------

$attributeRule = "docs/results/visual-address-phase-one-v1/recovered-originals/* -text"

if (Test-Path -LiteralPath $attributes) {
    $attributeLines = @(Get-Content -LiteralPath $attributes -Encoding UTF8)
} else {
    $attributeLines = @()
}

$attributeLines = @(
    $attributeLines |
    Where-Object {
        $_ -notmatch '^\s*docs/results/visual-address-phase-one-v1/recovered-originals/'
    }
)

if ($attributeLines.Count -gt 0 -and $attributeLines[-1] -ne "") {
    $attributeLines += ""
}

$attributeLines += "# Historical recovered evidence: preserve exact bytes."
$attributeLines += $attributeRule

[System.IO.File]::WriteAllText(
    (Join-Path $repo $attributes),
    (($attributeLines -join "`n") + "`n"),
    [System.Text.UTF8Encoding]::new($false)
)

# ---------------------------------------------------------------------------
# 2. Repair the malformed .gitignore exception and place it last.
# ---------------------------------------------------------------------------

if (Test-Path -LiteralPath $ignore) {
    $ignoreLines = @(Get-Content -LiteralPath $ignore -Encoding UTF8)
} else {
    $ignoreLines = @()
}

$ignoreLines = @(
    $ignoreLines |
    Where-Object {
        $_ -notmatch '^\s*!docs/results/visual-address-phase-one-v1/recovered-originals/dino-full-run\.log(?:\s.*)?$'
    }
)

while ($ignoreLines.Count -gt 0 -and $ignoreLines[-1] -eq "") {
    if ($ignoreLines.Count -eq 1) {
        $ignoreLines = @()
    } else {
        $ignoreLines = $ignoreLines[0..($ignoreLines.Count - 2)]
    }
}

$ignoreLines += ""
$ignoreLines += "# Preserve the recovered historical benchmark log."
$ignoreLines += "!docs/results/visual-address-phase-one-v1/recovered-originals/dino-full-run.log"

[System.IO.File]::WriteAllText(
    (Join-Path $repo $ignore),
    (($ignoreLines -join "`n") + "`n"),
    [System.Text.UTF8Encoding]::new($false)
)

# ---------------------------------------------------------------------------
# 3. Search Git history and restore the exact manifest-declared bytes.
# ---------------------------------------------------------------------------

$restoreCode = @'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

manifest_path = Path(
    "docs/results/visual-address-phase-one-v1/recovery-manifest.json"
)
root = manifest_path.parent / "recovered-originals"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def run_bytes(*args: str) -> bytes:
    result = subprocess.run(
        list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(args)}\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    return result.stdout

def matches(payload: bytes, item: dict) -> bool:
    return (
        len(payload) == int(item["size"])
        and hashlib.sha256(payload).hexdigest() == item["sha256"]
    )

restored = []

for item in manifest["recovered_files"]:
    output_path = root / item["name"]
    repo_path = output_path.as_posix()

    history_raw = run_bytes(
        "git", "log", "--format=%H", "--all", "--", repo_path
    )
    commits = [
        line.strip()
        for line in history_raw.decode("ascii", errors="strict").splitlines()
        if line.strip()
    ]

    if not commits:
        raise RuntimeError(f"No Git history found for {repo_path}")

    found = None

    for commit in commits:
        result = subprocess.run(
            ["git", "show", f"{commit}:{repo_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            continue

        raw = result.stdout
        lf = raw.replace(b"\r\n", b"\n")
        crlf = lf.replace(b"\n", b"\r\n")

        candidates = (
            ("raw", raw),
            ("lf", lf),
            ("crlf", crlf),
        )

        for representation, payload in candidates:
            if matches(payload, item):
                found = (commit, representation, payload)
                break

        if found is not None:
            break

    if found is None:
        current = output_path.read_bytes() if output_path.exists() else b""
        raise RuntimeError(
            f"Could not reconstruct {item['name']} from Git history.\n"
            f"Manifest size={item['size']} sha256={item['sha256']}\n"
            f"Current size={len(current)} "
            f"sha256={hashlib.sha256(current).hexdigest() if current else 'missing'}"
        )

    commit, representation, payload = found
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)

    actual_hash = hashlib.sha256(payload).hexdigest()
    print(
        f"RESTORED {item['name']}: "
        f"source={commit[:12]} form={representation} "
        f"size={len(payload)} sha256={actual_hash}"
    )

    restored.append(
        {
            "name": item["name"],
            "source_commit": commit,
            "representation": representation,
            "size": len(payload),
            "sha256": actual_hash,
        }
    )

print(f"Restored {len(restored)} manifest-declared files.")
'@

$restoreCode | & $Python -
if ($LASTEXITCODE -ne 0) {
    throw "Historical-byte reconstruction failed. Nothing should be committed."
}

# ---------------------------------------------------------------------------
# 4. Stage attributes first, then exact evidence files.
# ---------------------------------------------------------------------------

& git add -- $attributes $ignore
if ($LASTEXITCODE -ne 0) {
    throw "Failed to stage .gitattributes or .gitignore."
}

& git add -f -- $recovered
if ($LASTEXITCODE -ne 0) {
    throw "Failed to stage recovered evidence files."
}

# ---------------------------------------------------------------------------
# 5. Verify exact bytes stored in the index.
# ---------------------------------------------------------------------------

$verifyCode = @'
import hashlib
import json
import subprocess
from pathlib import Path

manifest_path = Path(
    "docs/results/visual-address-phase-one-v1/recovery-manifest.json"
)
root = manifest_path.parent / "recovered-originals"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

for item in manifest["recovered_files"]:
    repo_path = (root / item["name"]).as_posix()

    result = subprocess.run(
        ["git", "show", f":{repo_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Unable to read staged blob for {repo_path}: "
            f"{result.stderr.decode(errors='replace')}"
        )

    payload = result.stdout
    size = len(payload)
    digest = hashlib.sha256(payload).hexdigest()

    print(
        f"STAGED {item['name']}: "
        f"size={size} sha256={digest}"
    )

    assert size == int(item["size"]), (
        item["name"],
        "staged size",
        item["size"],
        size,
    )
    assert digest == item["sha256"], (
        item["name"],
        "staged sha256",
        item["sha256"],
        digest,
    )

print("All staged recovery blobs match recovery-manifest.json exactly.")
'@

$verifyCode | & $Python -
if ($LASTEXITCODE -ne 0) {
    throw "Staged-blob verification failed. Do not commit."
}

Write-Host ""
Write-Host "Historical recovery repair succeeded." -ForegroundColor Green
Write-Host ""
Write-Host "Run the focused test next:"
Write-Host "  $Python -m pytest tests/test_visual_result_records.py::test_phase_one_recovery_manifest_matches_attached_files -q"
Write-Host ""
Write-Host "Then inspect the staged changes:"
Write-Host "  git diff --cached --stat"
