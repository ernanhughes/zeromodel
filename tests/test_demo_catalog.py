from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_demos", ROOT / "scripts" / "build_demos.py"
)
assert SPEC is not None and SPEC.loader is not None
build_demos = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_demos
SPEC.loader.exec_module(build_demos)


def test_demo_catalogue_and_inventory_are_valid() -> None:
    demos = build_demos.load_catalog()
    build_demos.validate(demos)


def test_second_wave_catalogue_contains_seven_fast_demos() -> None:
    demos = build_demos.load_catalog()
    assert [demo.id for demo in demos] == [
        "vpm-artifact",
        "visual-sign-reader",
        "lua-edge-policy",
        "signs-rendering",
        "criticality-verification",
        "policy-lookup-benchmark",
        "bertin-pattern-detection",
    ]
    assert all(demo.execution_profile == "fast" for demo in demos)
    assert all(not demo.network for demo in demos)


def test_example_inventory_tracks_published_and_planned_work() -> None:
    entries = build_demos.load_inventory()
    assert len(entries) >= 38
    published = {
        demo_id
        for entry in entries
        if entry.status == "published"
        for demo_id in entry.demo_ids
    }
    assert published == {
        "vpm-artifact",
        "visual-sign-reader",
        "lua-edge-policy",
        "signs-rendering",
        "criticality-verification",
        "policy-lookup-benchmark",
        "bertin-pattern-detection",
    }
    assert any(entry.status == "planned" for entry in entries)
    assert any(entry.status == "supporting" for entry in entries)


def test_pages_workflow_deploys_the_generated_site() -> None:
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(
        encoding="utf-8"
    )
    assert "python scripts/build_demos.py all --profile fast" in workflow
    assert "path: build/site" in workflow
    assert "actions/upload-pages-artifact@v3" in workflow
    assert "actions/deploy-pages@v4" in workflow


def test_site_build_injects_demo_navigation() -> None:
    source = (ROOT / "scripts" / "build_demos.py").read_text(encoding="utf-8")
    assert "def _inject_demo_links()" in source
    assert 'href="demos/"' in source
