from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

_IMPLEMENTATION_PATH = Path(__file__).with_name("_validate_release_candidate_impl.py")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_BOUNDARIES_PATH = _REPO_ROOT / "package-boundaries.toml"
_ORIGINAL_MODULE_NAME = __name__
_IMPLEMENTATION_MODULE_NAME = "_zeromodel_validate_release_candidate_impl"
_RELEASE_VERSION = str(
    tomllib.loads(_PACKAGE_BOUNDARIES_PATH.read_text(encoding="utf-8"))["release_version"]
)

_CURRENT_MODULE = sys.modules[_ORIGINAL_MODULE_NAME]
sys.modules[_IMPLEMENTATION_MODULE_NAME] = _CURRENT_MODULE
globals()["__name__"] = _IMPLEMENTATION_MODULE_NAME
try:
    exec(
        compile(
            _IMPLEMENTATION_PATH.read_text(encoding="utf-8"),
            str(_IMPLEMENTATION_PATH),
            "exec",
        ),
        globals(),
        globals(),
    )
finally:
    globals()["__name__"] = _ORIGINAL_MODULE_NAME
    sys.modules.pop(_IMPLEMENTATION_MODULE_NAME, None)

# The implementation remains the large stable release harness. The coordinated
# release line comes from package-boundaries.toml, the package-system authority,
# so release preparation does not require rewriting the complete implementation.
globals()["VERSION"] = _RELEASE_VERSION
globals()["PACKAGE_RELEASE_ARTIFACTS_PATH"] = globals()["ARCHITECTURE_REPORT_DIR"] / (
    f"package-release-artifacts-{_RELEASE_VERSION}.json"
)
globals()["PACKAGE_PUBLIC_API_PATH"] = globals()["ARCHITECTURE_REPORT_DIR"] / (
    f"package-public-api-{_RELEASE_VERSION}.csv"
)
globals()["PACKAGE_RELEASE_TEST_LAYERS_PATH"] = globals()[
    "ARCHITECTURE_REPORT_DIR"
] / f"package-release-test-layers-{_RELEASE_VERSION}.json"
globals()["RELEASE_CANDIDATE_REPORT_DIR"] = (
    globals()["REPO_ROOT"]
    / "docs"
    / "results"
    / f"release-candidate-{_RELEASE_VERSION}"
)

for package in globals()["PACKAGES"].values():
    package["requires"] = {
        (
            f"{requirement.split('==', 1)[0]}=={_RELEASE_VERSION}"
            if requirement.startswith("zeromodel") and "==" in requirement
            else requirement
        )
        for requirement in package["requires"]
    }

if _ORIGINAL_MODULE_NAME == "__main__":
    raise SystemExit(globals()["main"]())
