from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from zeromodel.artifact import VPMValidationError
from zeromodel.visual_system_b import (
    build_system_b_candidates,
    select_system_b_operating_point,
    validate_selection_records,
)


def _load_demo():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "arcade_visual_address_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "arcade_visual_address_benchmark_system_b_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selection_records_reject_final_evaluation_records() -> None:
    demo = _load_demo()
    dataset = demo.build_arcade_benchmark_dataset(variants_per_family=1, ood_examples_per_family=1)
    final_records = tuple(
        record for record in dataset.manifest.records if record.split == "final_evaluation"
    )
    with pytest.raises(VPMValidationError, match="final_evaluation records cannot enter"):
        validate_selection_records(final_records)


def test_system_b_candidate_enumeration_and_selection_are_deterministic() -> None:
    demo = _load_demo()
    dataset = demo.build_arcade_benchmark_dataset(variants_per_family=1, ood_examples_per_family=1)
    candidates_a = build_system_b_candidates(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        source_scope=demo.SOURCE_SCOPE,
        quantiles=(0.0, 0.5, 1.0),
    )
    candidates_b = build_system_b_candidates(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        source_scope=demo.SOURCE_SCOPE,
        quantiles=(0.0, 0.5, 1.0),
    )
    assert [item.to_dict() for item in candidates_a] == [item.to_dict() for item in candidates_b]

    selection_a = select_system_b_operating_point(
        dataset_manifest=dataset.manifest,
        candidates=candidates_a,
    )
    selection_b = select_system_b_operating_point(
        dataset_manifest=dataset.manifest,
        candidates=candidates_b,
    )
    assert selection_a.to_dict() == selection_b.to_dict()
    assert selection_a.digest == selection_b.digest

