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
        if row["classification"]
        in {"core", "analysis", "observation", "vision", "video", "sqlalchemy"}
    )
    assert all(
        row["blocking_questions"] for row in rows if row["classification"] == "undecided"
    )


def test_import_graph_is_deterministic_with_fixed_timestamp() -> None:
    first = inventory.make_inventory("2026-01-01T00:00:00Z")["graph"]
    second = inventory.make_inventory("2026-01-01T00:00:00Z")["graph"]

    assert first == second
    assert first["schema_version"] == 1
    assert "examples.arcade_shooter_policy" in first["modules"]
    assert all(
        {"importer", "imported", "line", "kind", "resolved"} <= set(edge)
        for edge in first["edges"]
    )


def test_write_outputs_emit_parseable_csv_and_json() -> None:
    data = inventory.make_inventory("2026-01-01T00:00:00Z")
    inventory.write_outputs(data)

    csv_path = Path("docs/architecture/package-module-map-1.0.13.csv")
    json_path = Path("docs/architecture/package-import-graph-1.0.13.json")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = json.loads(json_path.read_text(encoding="utf-8"))

    assert len(rows) == len(data["rows"])
    assert parsed["generated_at_utc"] == "2026-01-01T00:00:00Z"
