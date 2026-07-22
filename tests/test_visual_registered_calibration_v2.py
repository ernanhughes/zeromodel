from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from research.visual.visual_local_baselines import (
    RegisteredPixelCandidate,
    RegisteredPixelAddressProvider,
    build_registered_pixel_candidates_v2,
    select_registered_pixel_candidate_v2,
)
from research.visual.visual_registration import RegistrationConfig, RegistrationResult


def _load_module(name: str, relative: str):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_registered_pixel_v2_generates_full_independent_grid_and_excludes_final_ids() -> None:
    demo = _load_module(
        "arcade_visual_local_evidence_benchmark_v3_grid_test",
        "examples/arcade_visual_local_evidence_benchmark.py",
    )
    dataset = demo.build_arcade_local_evidence_dataset(variants_per_family=1)
    seen_ids: set[str] = set()
    candidates = build_registered_pixel_candidates_v2(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        registration_config=RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6),
        distance_quantiles=(0.0, 0.5, 1.0),
        ambiguity_margin_quantiles=(0.0, 0.5, 1.0),
        source_scope=demo.SOURCE_SCOPE,
        capture_ids=seen_ids,
    )
    assert len(candidates) == 9
    assert len({(candidate.distance_quantile, candidate.ambiguity_margin_quantile) for candidate in candidates}) == 9
    final_ids = {
        record.observation_id
        for record in dataset.manifest.records
        if record.split == "final_evaluation"
    }
    assert final_ids.isdisjoint(seen_ids)


@pytest.mark.slow
def test_registered_pixel_v2_selection_uses_declared_strictness_directions() -> None:
    demo = _load_module(
        "arcade_visual_local_evidence_benchmark_v3_select_test",
        "examples/arcade_visual_local_evidence_benchmark.py",
    )
    dataset = demo.build_arcade_local_evidence_dataset(variants_per_family=1)
    candidates = build_registered_pixel_candidates_v2(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        registration_config=RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6),
        distance_quantiles=(0.0, 0.5),
        ambiguity_margin_quantiles=(0.0,),
        source_scope=demo.SOURCE_SCOPE,
    )
    selection = select_registered_pixel_candidate_v2(
        dataset_manifest=dataset.manifest,
        registration_config=RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6),
        candidates=candidates,
        source_scope=demo.SOURCE_SCOPE,
    )
    assert selection.selected_distance_quantile == 0.5
    assert selection.selected_ambiguity_margin_quantile == 0.0


def test_registered_pixel_raw_tie_break_ignores_registered_metadata() -> None:
    provider = RegisteredPixelAddressProvider.__new__(RegisteredPixelAddressProvider)
    first = RegisteredPixelCandidate(
        row_id="row-a",
        action_id="LEFT",
        prototype_observation_id="prototype-b",
        observation_digest="obs-a",
        registration=RegistrationResult(
            dx=0,
            dy=0,
            distance_before=0.25,
            distance_after=0.01,
            distance_improvement=0.24,
            overlap_fraction=1.0,
            valid_pixel_count=16,
            score_before=-0.25,
            score_after=-0.01,
            registration_succeeded=True,
            rejection_reason=None,
        ),
    )
    second = RegisteredPixelCandidate(
        row_id="row-b",
        action_id="LEFT",
        prototype_observation_id="prototype-a",
        observation_digest="obs-b",
        registration=RegistrationResult(
            dx=3,
            dy=3,
            distance_before=0.25,
            distance_after=0.0,
            distance_improvement=0.25,
            overlap_fraction=0.6,
            valid_pixel_count=10,
            score_before=-0.25,
            score_after=-0.0,
            registration_succeeded=True,
            rejection_reason=None,
        ),
    )
    ordered = sorted((second, first), key=lambda candidate: provider._sort_key(candidate, use_registered_distance=False))
    assert [candidate.row_id for candidate in ordered] == ["row-a", "row-b"]
    registered = sorted((second, first), key=lambda candidate: provider._sort_key(candidate, use_registered_distance=True))
    assert registered[0].row_id == "row-b"
