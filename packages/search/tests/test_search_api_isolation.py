from __future__ import annotations

import ast
from pathlib import Path


def test_public_api_imports_without_heavy_modules():
    import zeromodel.search as search

    assert search.SEARCH_PACKAGE_VERSION == "1.2.0"


def test_production_modules_do_not_import_research_or_relate():
    root = Path(__file__).resolve().parents[1] / "src"
    forbidden = {"research", "relate", "torch", "transformers", "PIL"}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            assert forbidden.isdisjoint(names), path
