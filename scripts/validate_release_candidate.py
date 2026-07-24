from __future__ import annotations

from pathlib import Path

_IMPLEMENTATION_PATH = Path(__file__).with_name("_validate_release_candidate_impl.py")
_ORIGINAL_MODULE_NAME = __name__

globals()["__name__"] = "_zeromodel_validate_release_candidate_impl"
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

_VERSION = globals()["VERSION"]
globals()["PACKAGES"]["perception"] = {
    "path": Path("packages/perception"),
    "distribution": "zeromodel-perception",
    "wheel_stem": "zeromodel_perception",
    "namespace": "zeromodel.perception",
    "requires": {
        "numpy>=1.23",
        "pillow>=9.0",
        f"zeromodel=={_VERSION}",
        f"zeromodel-observation=={_VERSION}",
    },
    "depends_on": ("core", "observation"),
}

if _ORIGINAL_MODULE_NAME == "__main__":
    raise SystemExit(globals()["main"]())
