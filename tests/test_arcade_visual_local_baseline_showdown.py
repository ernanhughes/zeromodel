from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "arcade_visual_local_baseline_showdown.py"
    )
    spec = importlib.util.spec_from_file_location(
        "arcade_visual_local_baseline_showdown_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_showdown_generates_required_artifacts(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    dataset = module.build_arcade_benchmark_dataset(variants_per_family=1)

    def fake_git_output(*args: str) -> str:
        mapping = {
            ("status", "--short"): "",
            ("rev-parse", "HEAD"): "stage1-test-sha",
            ("branch", "--show-current"): "research/visual-local-baseline-showdown",
            ("merge-base", "HEAD", "origin/main"): "832bca74fa05a6222ed02c65419bc2f551dfc7c0",
        }
        return mapping[args]

    monkeypatch.setattr(module, "_git_output", fake_git_output)
    monkeypatch.setattr(module, "QUANTILES", (0.0, 1.0))
    monkeypatch.setitem(module.FROZEN_SYSTEM_B_IDENTITIES, "dataset_digest", dataset.manifest.digest)
    monkeypatch.setattr(
        module,
        "_frozen_system_b_reference",
        lambda: {
            "dataset_digest": dataset.manifest.digest,
            "selection_digest": "frozen-selection",
            "calibration_digest": "frozen-calibration",
            "run_manifest_digest": "frozen-run",
            "trace_digest": "frozen-trace",
            "outcome": "C",
            "headline_metrics": {
                "raw_top1_exact_row_accuracy": 0.75,
                "raw_top1_action_accuracy": 0.96875,
                "accepted_benign_count": 0,
                "false_accept_count": 0,
                "false_reject_count": 448,
            },
        },
    )
    monkeypatch.setattr(module, "_paired_comparison", lambda frozen_b, r1: {"observation_count": len(r1), "row": {}, "action": {}})
    monkeypatch.setattr(module, "_load_jsonl", lambda path: ())
    output_dir = tmp_path / "showdown"
    summary = module.run_showdown(
        output_dir=output_dir,
        variants_per_family=1,
        max_dx=3,
        max_dy=3,
        minimum_overlap_fraction=0.6,
        argv=["python", "examples/arcade_visual_local_baseline_showdown.py"],
        command="python examples/arcade_visual_local_baseline_showdown.py",
    )
    assert summary["outcome"] in {"A", "B", "C"}
    assert (output_dir / "bundle-manifest.json").exists()
    assert (output_dir / "system-b-frozen-reference.json").exists()
    assert (output_dir / "registration-displacement-atlas.json").exists()
