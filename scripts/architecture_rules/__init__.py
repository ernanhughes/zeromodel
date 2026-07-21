from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any


TRACKED_EXTERNAL_MODULES = frozenset(
    {
        "csv",
        "json",
        "os",
        "pathlib",
        "shutil",
        "sqlite3",
        "sqlalchemy",
        "subprocess",
        "tempfile",
    }
)


def print_violations(violations: Sequence[Any]) -> None:
    for violation in sorted(
        violations, key=lambda item: (item.rule, item.importer, item.imported)
    ):
        print(f"Architecture violation: {violation.rule}", file=sys.stderr)
        print(f"  importer: {violation.importer}", file=sys.stderr)
        print(f"  imported: {violation.imported}", file=sys.stderr)
        print(f"  detail: {violation.detail}", file=sys.stderr)
