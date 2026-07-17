from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_adjudication_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "arcade_visual_system_b_adjudication.py"
    )
    spec = importlib.util.spec_from_file_location(
        "arcade_visual_system_b_adjudication_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase_one_recovery_manifest_matches_attached_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "docs" / "results" / "visual-address-phase-one-v1" / "recovery-manifest.json"
    recovered_root = manifest_path.parent / "recovered-originals"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    for item in payload["recovered_files"]:
        path = recovered_root / item["name"]
        assert path.exists(), item["name"]
        assert path.stat().st_size == item["size"], item["name"]
        import hashlib

        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["name"]


def test_system_b_adjudication_writes_run_manifest_and_row_confusion_atlas(tmp_path: Path) -> None:
    module = _load_adjudication_module()
    summary = module.run(
        output_dir=tmp_path,
        variants_per_family=1,
        argv=["python", "examples/arcade_visual_system_b_adjudication.py", "--variants-per-family", "1"],
        command="python examples/arcade_visual_system_b_adjudication.py --variants-per-family 1",
    )

    final_report = json.loads((tmp_path / "final-report.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((tmp_path / "run-manifest.json").read_text(encoding="utf-8"))
    row_confusion = json.loads((tmp_path / "row-confusion-atlas.json").read_text(encoding="utf-8"))

    assert summary["run_manifest_digest"] == run_manifest["run_manifest_digest"]
    assert final_report["run_manifest_digest"] == run_manifest["run_manifest_digest"]
    assert run_manifest["final_report_digest"]
    assert run_manifest["trace_digest"]
    assert run_manifest["git_commit"]
    assert run_manifest["branch"]
    assert run_manifest["argv"]
    assert row_confusion["atlas_type"] == "observed_benign_row_confusion"
    assert not (tmp_path / "state-equivalence-atlas.json").exists()
