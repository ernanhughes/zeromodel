"""Compatibility entry point for the repository package inventory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_IMPLEMENTATION_NAME = "_zeromodel_analyze_package_inventory_impl"
_IMPLEMENTATION_PATH = Path(__file__).with_name("_analyze_package_inventory_impl.py")
_IMPLEMENTATION_SPEC = importlib.util.spec_from_file_location(
    _IMPLEMENTATION_NAME,
    _IMPLEMENTATION_PATH,
)
if _IMPLEMENTATION_SPEC is None or _IMPLEMENTATION_SPEC.loader is None:
    raise ImportError(f"cannot load inventory implementation from {_IMPLEMENTATION_PATH}")

_implementation = importlib.util.module_from_spec(_IMPLEMENTATION_SPEC)
sys.modules[_IMPLEMENTATION_NAME] = _implementation
_IMPLEMENTATION_SPEC.loader.exec_module(_implementation)

for _name in dir(_implementation):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_implementation, _name)

_implementation.CLASSIFICATIONS.add("perception")

if __name__ == "__main__":
    _implementation.main()
