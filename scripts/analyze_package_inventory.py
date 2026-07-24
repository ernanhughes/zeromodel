"""Compatibility entry point for the repository package inventory."""

from __future__ import annotations

try:
    from . import _analyze_package_inventory_impl as _implementation
    from ._analyze_package_inventory_impl import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - direct script execution
    import _analyze_package_inventory_impl as _implementation
    from _analyze_package_inventory_impl import *  # type: ignore[no-redef]  # noqa: F401,F403

CLASSIFICATIONS.add("perception")

if __name__ == "__main__":
    _implementation.main()
