from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path("scripts/analyze_package_inventory.py")
SPEC = importlib.util.spec_from_file_location("analyze_package_inventory", SCRIPT_PATH)
assert SPEC is not None
inventory = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)

PRODUCTION_PACKAGE_KEYS = {
    "core",
    "analysis",
    "observation",
    "vision",
    "video",
    "sqlalchemy",
    "artifacts",
    "trust",
    "navigation",
}


def test_inventory_rows_cover_unique_modules_and_allowed_classifications() -> None:
    data = inventory.make_inventory("2026-01-01T00:00:00Z")
    rows = data["rows"]

    assert rows
    assert len({row["path"] for row in rows}) == len(rows)
    assert len({row["module"] for row in rows}) == len(rows)
    assert {row["classification"] for row in rows} <= inventory.CLASSIFICATIONS
    assert all(
        row["target_distribution"] and row["target_namespace"]
        for row in rows
        if row["classification"] in PRODUCTION_PACKAGE_KEYS
    )
    assert all(
        row["blocking_questions"]
        for row in rows
        if row["classification"] == "undecided"
    )


def test_all_nine_production_source_roots_are_discovered() -> None:
    """No production package may be silently omitted from discovery: every
    key configured in package-boundaries.toml must contribute at least one
    classified row."""
    data = inventory.make_inventory("2026-01-01T00:00:00Z")
    rows = data["rows"]
    discovered = {row["classification"] for row in rows} & PRODUCTION_PACKAGE_KEYS
    assert discovered == PRODUCTION_PACKAGE_KEYS



def test_all_nine_package_test_roots_are_discovered() -> None:
    boundaries = inventory.load_package_boundaries()
    data = inventory.make_inventory("2026-01-01T00:00:00Z")
    paths = {row["path"] for row in data["rows"]}
    for key, config in boundaries.items():
        test_root = (Path(config["source_root"]).parent / "tests").as_posix()
        assert any(path.startswith(f"{test_root}/") for path in paths), key


def test_package_keys_and_namespaces_agree_with_package_boundaries_toml() -> None:
    boundaries = inventory.load_package_boundaries()
    data = inventory.make_inventory("2026-01-01T00:00:00Z")
    rows = data["rows"]
    for row in rows:
        cls = row["classification"]
        if cls not in PRODUCTION_PACKAGE_KEYS:
            continue
        config = boundaries[cls]
        assert row["target_distribution"] == config["distribution"]
        assert row["target_namespace"].startswith(config["namespace"])


def test_missing_package_source_root_fails_loudly(tmp_path: Path) -> None:
    boundaries = inventory.load_package_boundaries()
    broken = {k: dict(v) for k, v in boundaries.items()}
    broken["trust"]["source_root"] = str(tmp_path / "does-not-exist")
    try:
        inventory.make_inventory("2026-01-01T00:00:00Z", boundaries=broken)
    except SystemExit as exc:
        assert "trust" in str(exc)
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("expected SystemExit for a missing package source root")


def test_zero_module_package_source_root_fails_loudly(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty-src"
    empty_dir.mkdir()
    boundaries = inventory.load_package_boundaries()
    broken = {k: dict(v) for k, v in boundaries.items()}
    broken["navigation"]["source_root"] = str(empty_dir)
    try:
        inventory.make_inventory("2026-01-01T00:00:00Z", boundaries=broken)
    except SystemExit as exc:
        assert "navigation" in str(exc)
        assert "zero Python modules" in str(exc)
    else:
        raise AssertionError("expected SystemExit for an empty package source root")


def test_import_graph_is_deterministic_with_fixed_timestamp() -> None:
    first = inventory.make_inventory("2026-01-01T00:00:00Z")["graph"]
    second = inventory.make_inventory("2026-01-01T00:00:00Z")["graph"]

    assert first == second
    assert first["schema_version"] == 1
    assert first["generator_version"] == inventory.GENERATOR_VERSION
    assert first["inventory_kind"] == "current_architecture"
    assert "examples.arcade_shooter_policy" in first["modules"]
    assert all(
        {"importer", "imported", "line", "kind", "resolved"} <= set(edge)
        for edge in first["edges"]
    )


def test_write_outputs_emit_parseable_csv_and_json_without_mutating_docs(
    tmp_path: Path,
) -> None:
    canonical_paths = [
        Path("docs/architecture/package-module-map-1.0.13.csv"),
        Path("docs/architecture/package-import-graph-1.0.13.json"),
        Path("docs/architecture/package-inventory-1.0.13.md"),
        Path("docs/architecture/package-dependency-findings-1.0.13.md"),
    ]
    before = {path: path.read_bytes() for path in canonical_paths}

    data = inventory.make_inventory("2026-01-01T00:00:00Z")
    output_dir = tmp_path / "architecture"
    inventory.write_outputs(data, output_dir)

    csv_path = output_dir / "package-module-map-1.0.13.csv"
    json_path = output_dir / "package-import-graph-1.0.13.json"
    inv_path = output_dir / "package-inventory-1.0.13.md"

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = json.loads(json_path.read_text(encoding="utf-8"))

    assert len(rows) == len(data["rows"])
    assert parsed["generated_at_utc"] == "2026-01-01T00:00:00Z"
    assert parsed["generator_version"] == inventory.GENERATOR_VERSION
    assert parsed["source_tree_dirty"] == data["source_tree_dirty"]

    report_text = inv_path.read_text(encoding="utf-8")
    assert "current architecture inventory" in report_text.lower()
    assert data["baseline"] in report_text
    assert inventory.GENERATOR_VERSION in report_text
    assert "source tree dirty" in report_text.lower()
    assert "1.0.12" not in report_text
    assert "ships the monolithic" not in report_text.lower()
    assert {path: path.read_bytes() for path in canonical_paths} == before
