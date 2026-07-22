from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

import pytest

from zeromodel.core.artifact import VPMValidationError
from zeromodel.observation.visual_address import ImageObservation
from zeromodel.vision.visual_encoder import EncoderManifest
from research.visual.visual_system_b import (
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


def test_system_b_v2_dataset_digest_differs_from_historical_v1() -> None:
    demo = _load_demo()
    dataset = demo.build_arcade_benchmark_dataset(variants_per_family=1, ood_examples_per_family=1)
    assert dataset.manifest.source_scope == "arcade-visual-system-b-adjudication/v2"
    assert dataset.manifest.digest != "91b1b422482eeeef20eb182162eb2a745f9b50524cc7f94ec95a0aba5f2fa37e"


class _SpyEncoder:
    def __init__(self) -> None:
        self.seen_batches: list[tuple[ImageObservation, ...]] = []
        self._manifest = EncoderManifest(
            provider_kind="test",
            model_id="spy",
            revision="test",
            architecture="identity",
            weights_digest="weights:test",
            preprocessing_digest="preprocess:test",
            output_dimension=4,
            normalization="none",
            framework="pytest",
            framework_version="1",
            license_id="test-only",
            source_record="unit-test",
        )

    def manifest(self) -> EncoderManifest:
        return self._manifest

    def encode_batch(
        self,
        observations: Sequence[ImageObservation],
    ) -> np.ndarray:
        batch = tuple(observations)
        self.seen_batches.append(batch)
        rows = []
        for index, _observation in enumerate(batch, start=1):
            rows.append(np.full((self._manifest.output_dimension,), float(index), dtype=np.float32))
        return np.vstack(rows)


def test_system_b_candidate_builder_never_encodes_final_evaluation_observations() -> None:
    demo = _load_demo()
    dataset = demo.build_arcade_benchmark_dataset(variants_per_family=1, ood_examples_per_family=1)
    final_observations = {
        id(dataset.observations[record.observation_id])
        for record in dataset.manifest.records
        if record.split == "final_evaluation"
    }
    spy = _SpyEncoder()

    build_system_b_candidates(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        source_scope=demo.SOURCE_SCOPE,
        quantiles=(0.0, 1.0),
        encoder=spy,
    )

    seen_ids = {id(observation) for batch in spy.seen_batches for observation in batch}
    assert seen_ids
    assert final_observations.isdisjoint(seen_ids)
