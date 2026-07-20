from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
STAGE7A_IMPORTER = "zeromodel.domains.video_action_set.provider_measurement"


def _load_checker() -> ModuleType:
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location(
        "stage7a_check_architecture", SCRIPTS_ROOT / "check_architecture.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


@pytest.mark.parametrize(
    ("source", "expected_import"),
    [
        ("from pathlib import Path\n", "pathlib"),
        ("import sqlite3\n", "sqlite3"),
        ("from json import dumps\n", "json"),
    ],
)
def test_stage7a_tracked_external_imports_are_collected_and_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
    expected_import: str,
) -> None:
    module_path = tmp_path / "provider_measurement.py"
    module_path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(CHECKER, "relative_path", lambda path: path.name)

    edges = CHECKER.collect_import_edges(
        STAGE7A_IMPORTER,
        module_path,
        {STAGE7A_IMPORTER},
    )
    assert [(edge.importer, edge.imported) for edge in edges] == [
        (STAGE7A_IMPORTER, expected_import)
    ]

    violations = CHECKER.forbidden_edge_violations(edges)

    assert any(
        violation.importer == STAGE7A_IMPORTER
        and violation.imported == expected_import
        and violation.rule.startswith("provider_measurement must not import")
        for violation in violations
    )


@pytest.mark.parametrize(
    "imported",
    [
        "zeromodel.runtime",
        "zeromodel.db.orm.video_action_set",
        "zeromodel.db.stores.video_action_set",
        "zeromodel.domains.video_action_set.episode_planning",
        "zeromodel.domains.video_action_set.episode_materialization",
        "zeromodel.video_action_equivalence",
        "zeromodel.video_action_set.mutation_audit",
    ],
)
def test_stage7a_forbidden_layers_are_rejected(imported: str) -> None:
    edge = CHECKER.ImportEdge(
        importer=STAGE7A_IMPORTER,
        imported=imported,
        line=7,
    )

    violations = CHECKER.forbidden_edge_violations([edge])

    assert any(
        violation.importer == STAGE7A_IMPORTER
        and violation.imported == imported
        and violation.rule.startswith("provider_measurement must not import")
        for violation in violations
    )
