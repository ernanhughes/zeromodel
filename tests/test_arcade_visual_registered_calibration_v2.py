from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "arcade_visual_registered_calibration_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "arcade_visual_registered_calibration_v2_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registered_calibration_v2_generates_required_artifacts(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    def fake_git_output(*args: str) -> str:
        mapping = {
            ("rev-parse", "HEAD"): module.EXPECTED_STARTING_MAIN_SHA,
            ("merge-base", "HEAD", "origin/main"): module.EXPECTED_STARTING_MAIN_SHA,
            ("status", "--short"): "",
            ("branch", "--show-current"): "research/visual-local-correlation-evidence",
        }
        return mapping[args]

    monkeypatch.setattr(module, "_git_output", fake_git_output)
    monkeypatch.setattr(module, "QUANTILES", (0.0, 1.0))
    output_dir = tmp_path / "registered-v2"
    summary = module.run_registered_calibration_v2(
        output_dir=output_dir,
        variants_per_family=1,
        max_dx=3,
        max_dy=3,
        minimum_overlap_fraction=0.6,
        argv=["python", "examples/arcade_visual_registered_calibration_v2.py"],
        command="python examples/arcade_visual_registered_calibration_v2.py",
    )
    assert summary["outcome"] in {"A", "B", "C"}
    assert (output_dir / "candidate-grid.json").exists()
    assert (output_dir / "selected-calibration.json").exists()
    assert (output_dir / "risk-coverage-curve.json").exists()
