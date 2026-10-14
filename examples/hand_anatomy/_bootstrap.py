"""Make workspace packages importable when they are not pip-installed."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_workspace_on_path() -> None:
    try:
        import zeromodel.core  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    for src in sorted((REPO_ROOT / "packages").glob("*/src")):
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


ensure_workspace_on_path()
