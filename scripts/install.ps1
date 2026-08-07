# scripts/install-dev.ps1

$packages = @(
    "core",
    "analysis",
    "observation",
    "vision",
    "perception",
    "observer",
    "video",
    "sqlalchemy",
    "artifacts",
    "trust",
    "navigation",
    "critic",
    "search"
)

foreach ($package in $packages) {
    Write-Host "Installing $package..."
    python -m pip install -e ".\packages\$package"
}

python -m pip install pytest ruff mypy