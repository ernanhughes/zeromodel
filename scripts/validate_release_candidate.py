from __future__ import annotations

import sys
from pathlib import Path

_IMPLEMENTATION_PATH = Path(__file__).with_name("_validate_release_candidate_impl.py")
_ORIGINAL_MODULE_NAME = __name__
_IMPLEMENTATION_MODULE_NAME = "_zeromodel_validate_release_candidate_impl"

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
