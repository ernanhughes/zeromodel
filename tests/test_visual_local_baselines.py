from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from zeromodel.visual_experiment import EXPECTED_ACCEPT
from zeromodel.visual_local_baselines import (
    build_registered_pixel_candidates,
    build_registered_pixel_provider,
    select_registered_pixel_candidate,
)
from zeromodel.visual_registration import RegistrationConfig


def _load_demo():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "arcade_visual_address_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "arcade_visual_address_benchmark_local_baseline_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registered_pixel_selection_excludes_final_evaluation_ids() -> None:
    demo = _load_demo()
    dataset = demo.build_arcade_benchmark_dataset(variants_per_family=1, ood_examples_per_family=1)
    seen_ids: set[str] = set()
    _ = build_registered_pixel_candidates(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        registration_config=RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6),
        quantiles=(0.0, 1.0),
        source_scope=demo.SOURCE_SCOPE,
        capture_ids=seen_ids,
    )
    final_ids = {
        record.observation_id
        for record in dataset.manifest.records
        if record.split == "final_evaluation"
    }
    assert final_ids.isdisjoint(seen_ids)


def test_registered_pixel_provider_emits_registration_trace_and_raw_top1_when_rejected() -> None:
    demo = _load_demo()
    dataset = demo.build_arcade_benchmark_dataset(variants_per_family=1, ood_examples_per_family=1)
    candidates = build_registered_pixel_candidates(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        registration_config=RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6),
        quantiles=(1.0,),
        source_scope=demo.SOURCE_SCOPE,
    )
    selection = select_registered_pixel_candidate(
        dataset_manifest=dataset.manifest,
        registration_config=RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6),
        candidates=candidates,
        source_scope=demo.SOURCE_SCOPE,
    )
    calibration = (
        candidates[0].calibration
        if selection.selected_calibration_digest is None
        else next(
            candidate.calibration
            for candidate in candidates
            if candidate.calibration.digest == selection.selected_calibration_digest
        )
    )
    provider = build_registered_pixel_provider(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        registration_config=RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6),
        calibration=calibration,
    )
    record = next(
        record
        for record in dataset.manifest.records
        if record.split == "final_evaluation" and record.evaluation_role == EXPECTED_ACCEPT
    )
    decision = provider.read(dataset.observations[record.observation_id])
    assert "registration" in decision.trace
    assert "raw_top1_row_id" in decision.trace
    assert decision.nearest_row_id is not None
    if not decision.accepted:
        assert decision.matched_row_id is None
        assert decision.trace["raw_top1_row_id"] is not None
