# Developer convenience wrapper. Public users should install the complete
# supported runtime with: python -m pip install zeromodel

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$manifest = Get-Content -LiteralPath (Join-Path $repoRoot "package-boundaries.toml")
$packages = @()
$current = $null
$currentKind = "runtime"

foreach ($line in $manifest) {
    if ($line -match '^\[packages\.([A-Za-z0-9_-]+)\]$') {
        if ($current -and $currentKind -ne "meta") {
            $packages += $current
        }
        $current = $Matches[1]
        $currentKind = "runtime"
        continue
    }
    if ($current -and $line -match '^kind = "([^"]+)"$') {
        $currentKind = $Matches[1]
    }
}
if ($current -and $currentKind -ne "meta") {
    $packages += $current
}

foreach ($package in $packages) {
    Write-Host "Installing editable runtime package $package..."
    python -m pip install -e (Join-Path $repoRoot "packages\$package")
}

Write-Host "Installing editable umbrella package meta..."
python -m pip install -e (Join-Path $repoRoot "packages\meta")
python -m pip install pytest ruff mypy
