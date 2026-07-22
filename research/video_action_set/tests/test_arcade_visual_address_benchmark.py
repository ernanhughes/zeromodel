from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def _load_demo():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "arcade_visual_address_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "arcade_visual_address_benchmark_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_arcade_dataset_has_family_holdout_controls_and_rejection_sets() -> None:
    demo = _load_demo()
    dataset = demo.build_arcade_benchmark_dataset(
        variants_per_family=1,
        ood_examples_per_family=2,
    )

    dataset.manifest.validate()
    records = dataset.manifest.records
    assert {record.split for record in records} == {
        "prototype",
        "benign_calibration",
        "rejection_calibration",
        "final_evaluation",
    }
    assert any(
        family.critical_evidence_removed for family in dataset.manifest.families
    )
    target_controls = tuple(
        record
        for record in records
        if record.family_id == "final-information-target"
    )
    assert len(target_controls) == 98
    assert all(
        record.evaluation_role == demo.IMPOSSIBILITY_CONTROL
        for record in target_controls
    )
    assert sum(record.split == "final_evaluation" and record.row_id is None for record in records) == 6
    assert len(dataset.observations) == len(records)

    impossible = dataset.observations["final-ood-impossible-state:00"].pixels
    left_band = impossible[11:14, : demo.CELL_PIXELS]
    right_band = impossible[11:14, -demo.CELL_PIXELS :]
    assert np.count_nonzero(left_band == demo.TANK_VALUE) > 0
    assert np.count_nonzero(right_band == demo.TANK_VALUE) > 0


def test_default_benchmark_runs_deterministic_and_template_systems() -> None:
    demo = _load_demo()
    dataset, report, artifacts = demo.run_benchmark(
        variants_per_family=1,
        encoder_name="none",
        include_traces=False,
    )

    assert tuple(system.system_id for system in report.systems) == ("A", "B")
    assert not report.deployment_permitted
    assert report.validation_status == "research"
    assert report.dataset_manifest_digest == dataset.manifest.digest
    assert "address_B" in artifacts

    for system in report.systems:
        system.metrics.validate()
        assert system.metrics.evaluation_count > 0
        assert system.metrics.false_accept_opportunities == 248
        assert system.metrics.false_reject_opportunities == 448
        assert 0.0 <= system.metrics.false_acceptance_rate <= 1.0
        assert 0.0 <= system.metrics.false_rejection_rate <= 1.0
        assert system.notes["impossibility_control_count"] == 98
        assert (
            system.notes["observation_count_including_controls"]
            == system.metrics.evaluation_count + 98
        )
        assert tuple(system.notes["evaluated_splits"]) == ("final_evaluation",)

    assert json.loads(json.dumps(report.to_dict())) == report.to_dict()
