param(
    [string]$Python = "python",
    [string]$PyPIToken = $env:PYPI_API_TOKEN,
    [switch]$SkipTests,
    [switch]$SkipDemo,
    [switch]$SkipSmokeTest,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

function Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Fail($Message) {
    Write-Host ""
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Get-ZeroModelVersionFromInit {
    $InitPath = Join-Path (Get-Location) "zeromodel\__init__.py"
    if (-not (Test-Path $InitPath)) {
        Fail "zeromodel\__init__.py not found. Run this script from the repository root."
    }

    $InitText = Get-Content $InitPath -Raw
    $Match = [regex]::Match($InitText, '(?m)^__version__\s*=\s*["'']([^"'']+)["'']')
    if (-not $Match.Success) {
        Fail "Could not find __version__ in zeromodel\__init__.py."
    }

    return $Match.Groups[1].Value
}

function Get-PyProjectVersion {
    $PyProjectPath = Join-Path (Get-Location) "pyproject.toml"
    if (-not (Test-Path $PyProjectPath)) {
        Fail "pyproject.toml not found. Run this script from the repository root."
    }

    $PyProjectText = Get-Content $PyProjectPath -Raw
    $Match = [regex]::Match($PyProjectText, '(?m)^version\s*=\s*"([^"]+)"')
    if (-not $Match.Success) {
        Fail "Could not find project version in pyproject.toml."
    }

    return $Match.Groups[1].Value
}

Step "Checking repo root"

if (-not (Test-Path "pyproject.toml")) {
    Fail "pyproject.toml not found. Run this script from the repository root."
}

$Version = Get-ZeroModelVersionFromInit
$PyProjectVersion = Get-PyProjectVersion

Step "Checking release version from zeromodel\__init__.py: $Version"

if ($PyProjectVersion -ne $Version) {
    Fail "Version mismatch: zeromodel\__init__.py has $Version but pyproject.toml has $PyProjectVersion."
}

Step "Installing package locally for runtime version check"

& $Python -m pip install --upgrade pip
& $Python -m pip install -e ".[dev,release]"

$RuntimeVersion = & $Python -c "import zeromodel; print(zeromodel.__version__)"
if ($RuntimeVersion -ne $Version) {
    Fail "Runtime zeromodel.__version__ is $RuntimeVersion, expected $Version."
}

Step "Cleaning build artifacts"

Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
Get-ChildItem -Directory -Filter "*.egg-info" | Remove-Item -Recurse -Force

if (-not $SkipTests) {
    Step "Running fast tests"
    & $Python scripts/run_fast_tests.py
}

if (-not $SkipDemo) {
    Step "Running arcade shooter demo"
    & $Python examples/arcade_shooter_policy.py
}

Step "Building source and wheel distributions"

& $Python -m build

Step "Checking package metadata"

& $Python -m twine check dist/*

Step "Preparing PyPI upload"

if (-not $PyPIToken) {
    $SecureToken = Read-Host "Enter PyPI API token" -AsSecureString
    $Bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureToken)
    try {
        $PyPIToken = [Runtime.InteropServices.Marshal]::PtrToStringAuto($Bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Bstr)
    }
}

if (-not $PyPIToken.StartsWith("pypi-")) {
    Fail "PyPI token should usually start with 'pypi-'. Refusing to upload."
}

Write-Host ""
Write-Host "Ready to publish zeromodel==$Version to production PyPI." -ForegroundColor Yellow
Write-Host "This is irreversible: PyPI will not let you reuse the same version number." -ForegroundColor Yellow

if (-not $Yes) {
    $Confirm = Read-Host "Type PUBLISH to continue"
    if ($Confirm -ne "PUBLISH") {
        Fail "Upload cancelled."
    }
}

Step "Uploading zeromodel==$Version to production PyPI"

$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = $PyPIToken

& $Python -m twine upload dist/*

if (-not $SkipSmokeTest) {
    Step "Verifying clean PyPI install for zeromodel==$Version"

    $SmokeDir = Join-Path $env:TEMP "zeromodel-pypi-smoke-$Version"
    Remove-Item -Recurse -Force $SmokeDir -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $SmokeDir | Out-Null

    Push-Location $SmokeDir

    try {
        & $Python -m venv .venv
        $VenvPython = Join-Path $SmokeDir ".venv\Scripts\python.exe"

        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install "zeromodel==$Version"

        & $VenvPython -c @"
from zeromodel import LayoutRecipe, ScoreTable, VPMPolicyLookup, build_vpm
import zeromodel

assert zeromodel.__version__ == "$Version", zeromodel.__version__

score_table = ScoreTable(
    values=[[1.0, 0.0], [0.0, 1.0]],
    row_ids=["state:left", "state:right"],
    metric_ids=["LEFT", "RIGHT"],
)

recipe = LayoutRecipe.from_dict({
    "version": "vpm-layout/0",
    "name": "policy-source-order",
    "row_order": {"kind": "source", "tie_break": "row_id"},
    "column_order": {"kind": "source"},
    "normalization": {"kind": "per_metric_minmax", "clip": True},
})

artifact = build_vpm(score_table, recipe)
assert VPMPolicyLookup(artifact).read("state:right").action == "RIGHT"

print("zeromodel $Version PyPI smoke test passed")
"@
    }
    finally {
        Pop-Location
    }
}

Step "Release complete"

Write-Host ""
Write-Host "Published zeromodel==$Version to PyPI." -ForegroundColor Green
