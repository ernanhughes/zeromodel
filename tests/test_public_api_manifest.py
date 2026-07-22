from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "docs" / "architecture" / "package-public-api-1.0.13.csv"

SPEC = importlib.util.spec_from_file_location(
    "validate_release_candidate", REPO_ROOT / "scripts" / "validate_release_candidate.py"
)
assert SPEC is not None
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _read_manifest_rows() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _real_all(namespace: str) -> list[str]:
    module = importlib.import_module(namespace)
    all_symbols = getattr(module, "__all__", None)
    assert all_symbols is not None, f"{namespace} has no __all__"
    return list(all_symbols)


def test_manifest_has_the_required_columns() -> None:
    rows = _read_manifest_rows()
    assert list(rows[0].keys()) == [
        "distribution",
        "namespace",
        "exported_symbol",
        "owning_module",
        "object_kind",
        "source_module",
        "is_reexport",
    ]


def test_manifest_contains_all_nine_distributions() -> None:
    rows = _read_manifest_rows()
    distributions = {row["distribution"] for row in rows}
    assert distributions == {
        "zeromodel",
        "zeromodel-analysis",
        "zeromodel-observation",
        "zeromodel-vision",
        "zeromodel-video",
        "zeromodel-sqlalchemy",
        "zeromodel-artifacts",
        "zeromodel-trust",
        "zeromodel-navigation",
    }


def test_manifest_contains_more_than_six_rows() -> None:
    rows = _read_manifest_rows()
    assert len(rows) > 6, "manifest looks like a one-row-per-distribution placeholder"


def test_manifest_is_not_a_placeholder() -> None:
    rows = _read_manifest_rows()
    assert not any(row["exported_symbol"] == "__all__" for row in rows)


@pytest.mark.parametrize(
    "namespace",
    [
        "zeromodel.core",
        "zeromodel.analysis",
        "zeromodel.observation",
        "zeromodel.vision",
        "zeromodel.video",
        "zeromodel.persistence.sqlalchemy",
        "zeromodel.artifacts",
        "zeromodel.trust",
        "zeromodel.navigation",
    ],
)
def test_every_actual_all_symbol_appears_exactly_once(namespace: str) -> None:
    rows = [row for row in _read_manifest_rows() if row["namespace"] == namespace]
    manifest_symbols = [row["exported_symbol"] for row in rows]
    real_symbols = _real_all(namespace)

    assert sorted(manifest_symbols) == sorted(set(manifest_symbols)), (
        f"{namespace}: manifest lists a symbol more than once"
    )
    assert set(manifest_symbols) == set(real_symbols), (
        f"{namespace}: manifest symbols do not match the real __all__ "
        f"(missing={set(real_symbols) - set(manifest_symbols)}, "
        f"extra={set(manifest_symbols) - set(real_symbols)})"
    )


def test_no_undeclared_export_appears() -> None:
    # Every row's exported_symbol must be resolvable on its own namespace
    # module - i.e. the manifest never invents a symbol that isn't real.
    for row in _read_manifest_rows():
        module = importlib.import_module(row["namespace"])
        assert hasattr(module, row["exported_symbol"]), (
            f"{row['namespace']}.{row['exported_symbol']} is not a real attribute"
        )


def test_optional_packages_are_not_imported_through_another_packages_public_api() -> None:
    # Every row's source_module must belong to that row's own distribution or
    # one of its declared dependencies (packages/*/pyproject.toml), never an
    # undeclared sibling package.
    allowed_prefixes: dict[str, tuple[str, ...]] = {}
    for key, expected in validator.PACKAGES.items():
        own = (expected["namespace"],)
        deps = tuple(validator.PACKAGES[dep]["namespace"] for dep in expected["depends_on"])
        allowed_prefixes[expected["distribution"]] = own + deps

    for row in _read_manifest_rows():
        prefixes = allowed_prefixes[row["distribution"]]
        source_module = row["source_module"]
        assert any(
            source_module == prefix or source_module.startswith(f"{prefix}.")
            for prefix in prefixes
        ), (
            f"{row['distribution']}.{row['exported_symbol']} is implemented in "
            f"{source_module!r}, which is outside its declared dependency closure {prefixes}"
        )


@pytest.mark.slow
def test_manifest_generation_is_byte_identical_across_runs(tmp_path: Path) -> None:
    # Full end-to-end regeneration (build six wheels, clean venv, probe) is
    # too expensive for the fast suite; this proves determinism directly.
    validator.clean_artifacts()
    validator.build_packages()
    validator.install_and_probe()
    validator.write_public_exports()
    first = MANIFEST_PATH.read_bytes()
    validator.write_public_exports()
    second = MANIFEST_PATH.read_bytes()
    assert first == second
