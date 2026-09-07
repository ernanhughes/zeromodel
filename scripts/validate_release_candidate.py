from __future__ import annotations

import sys
from pathlib import Path

_IMPLEMENTATION_PATH = Path(__file__).with_name("_validate_release_candidate_impl.py")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_VERSION_PATH = _REPO_ROOT / "VERSION"
_ORIGINAL_MODULE_NAME = __name__
_IMPLEMENTATION_MODULE_NAME = "_zeromodel_validate_release_candidate_impl"
_RELEASE_VERSION = _VERSION_PATH.read_text(encoding="utf-8").strip()

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

# The implementation remains the large stable release harness. VERSION is the
# repository-wide human-edited authority; package metadata and generated report
# names are synchronized mirrors validated by scripts/release_version.py.
globals()["VERSION"] = _RELEASE_VERSION
globals()["PACKAGE_RELEASE_ARTIFACTS_PATH"] = globals()["ARCHITECTURE_REPORT_DIR"] / (
    f"package-release-artifacts-{_RELEASE_VERSION}.json"
)
globals()["PACKAGE_PUBLIC_API_PATH"] = globals()["ARCHITECTURE_REPORT_DIR"] / (
    f"package-public-api-{_RELEASE_VERSION}.csv"
)
globals()["PACKAGE_RELEASE_TEST_LAYERS_PATH"] = (
    globals()["ARCHITECTURE_REPORT_DIR"]
    / f"package-release-test-layers-{_RELEASE_VERSION}.json"
)
globals()["RELEASE_CANDIDATE_REPORT_DIR"] = (
    globals()["REPO_ROOT"]
    / "docs"
    / "results"
    / f"release-candidate-{_RELEASE_VERSION}"
)

_MANIFEST = tomllib.loads(
    (_REPO_ROOT / "package-boundaries.toml").read_text(encoding="utf-8")
)
_RUNTIME_BOUNDARIES = {
    key: config
    for key, config in _MANIFEST["packages"].items()
    if config.get("kind", "runtime") == "runtime"
}


def _wheel_stem(distribution: str) -> str:
    return distribution.replace("-", "_").replace(".", "_")


def _package_path(source_root: str) -> Path:
    path = Path(source_root)
    if path.name == "src":
        return path.parent
    return path


globals()["PACKAGES"] = {
    key: {
        "path": _package_path(config["source_root"]),
        "distribution": config["distribution"],
        "wheel_stem": _wheel_stem(config["distribution"]),
        "namespace": config["namespace"],
        "requires": set(
            tomllib.loads(
                (_REPO_ROOT / _package_path(config["source_root"]) / "pyproject.toml")
                .read_text(encoding="utf-8")
            )["project"].get("dependencies", [])
        ),
        "depends_on": tuple(config.get("depends_on", ())),
    }
    for key, config in _RUNTIME_BOUNDARIES.items()
}

globals()["UMBRELLA_PACKAGE"] = {
    "key": "meta",
    "path": _package_path(_MANIFEST["packages"]["meta"]["source_root"]),
    "distribution": _MANIFEST["packages"]["meta"]["distribution"],
    "requires": set(
        tomllib.loads(
            (
                _REPO_ROOT
                / _package_path(_MANIFEST["packages"]["meta"]["source_root"])
                / "pyproject.toml"
            ).read_text(encoding="utf-8")
        )["project"].get("dependencies", [])
    ),
}

globals()["load_package_boundaries"] = lambda: dict(_RUNTIME_BOUNDARIES)

if _ORIGINAL_MODULE_NAME == "__main__":
    raise SystemExit(globals()["main"]())
