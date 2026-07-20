from __future__ import annotations

from copy import deepcopy
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import zeromodel.video_action_set_benchmark as benchmark
from test_video_episode_plan_rmdto import plan_dto, sample_identity
from zeromodel import build_runtime
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.domains.video_action_set.contracts import (
    BENCHMARK_VERSION,
    FRAME_SHAPE,
    GENERATOR_VERSION,
    OBSERVATION_OPERATION_CHAIN_VERSION,
)
from zeromodel.domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
    ObservationOperationChainDTO,
    ObservationOperationDTO,
    ProviderObservationDescriptorDTO,
)
from zeromodel.domains.video_action_set.store import (
    MATRIX_BLOB_CONFLICT_MESSAGE,
    OBSERVATION_CONFLICT_MESSAGE,
    OBSERVATION_SEQUENCE_CONFLICT_MESSAGE,
    UNKNOWN_BENCHMARK_IDENTITY_MESSAGE,
    UNKNOWN_EPISODE_PLAN_MESSAGE,
)
from zeromodel.matrix_blob import MatrixBlob
from zeromodel.stores.video_action_set_memory import InMemoryVideoActionSetStore
from zeromodel.visual_address import ImageObservation


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pixel_digest(pixels: np.ndarray) -> str:
    return (
        "sha256:"
        + hashlib.sha256(np.ascontiguousarray(pixels).tobytes(order="C")).hexdigest()
    )


def _operation(
    *,
    index: int = 0,
    operation: str = "emit_observation",
    operation_version: str = OBSERVATION_OPERATION_CHAIN_VERSION,
    input_digests: tuple[str | None, ...],
    output_digest: str | None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parameters = {"event_type": "frame"} if parameters is None else parameters
    payload = {
        "index": index,
        "operation": operation,
        "operation_version": operation_version,
        "input_digests": list(input_digests),
        "parameters": parameters,
        "parameter_digest": canonical_sha256(parameters),
        "output_digest": output_digest,
    }
    return payload | {"operation_digest": canonical_sha256(payload)}


def _chain(
    final_digest: str | None,
    *,
    operation: str = "emit_observation",
) -> dict[str, Any]:
    parameters = {"event_type": "gap_unknown" if final_digest is None else "frame"}
    op = _operation(
        operation=operation,
        input_digests=(final_digest,),
        output_digest=final_digest,
        parameters=parameters,
    )
    payload = {
        "version": OBSERVATION_OPERATION_CHAIN_VERSION,
        "operations": [op],
        "final_emitted_digest": final_digest,
    }
    return payload | {"operation_chain_digest": canonical_sha256(payload)}


def _pixels(offset: int = 0) -> np.ndarray:
    values = (
        np.arange(FRAME_SHAPE[0] * FRAME_SHAPE[1], dtype=np.uint16) + offset
    ) % 251
    return values.astype(np.uint8).reshape(FRAME_SHAPE)


def _provider_descriptor(pixels: np.ndarray, frame_id: str) -> dict[str, Any]:
    return ImageObservation(pixels, source_id=frame_id).to_descriptor()


def sample_record(
    *,
    split: str = "development",
    sequence_number: int = 0,
    frame_index: int | None = None,
    pixels: np.ndarray | None = None,
    event_type: str = "frame",
    plan=None,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = sample_identity()
    plan = plan or plan_dto(identity=identity, split=split, frame_count=2)
    frame_index = sequence_number if frame_index is None else frame_index
    frame_id = f"{split}:{plan.episode_id}:frame-{frame_index:02d}"
    pixel_digest = None if pixels is None else _pixel_digest(pixels)
    metadata = {
        "episode_seed": plan.episode_seed,
        "seed_digest": plan.benchmark_seed_digest,
        "derived_seed_identity": plan.derived_seed_identity,
        "episode_plan_digest": plan.plan_digest,
        "frame_seed_identity": plan.frame_plans[0].to_value()["frame_seed_identity"],
        "observation_operation_chain": _chain(pixel_digest),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    if pixels is not None:
        descriptor = _provider_descriptor(pixels, frame_id)
        descriptor_dto = ProviderObservationDescriptorDTO.from_dict(descriptor)
        metadata.update(
            {
                "provider_observation_boundary_version": (
                    benchmark.PROVIDER_OBSERVATION_BOUNDARY_VERSION
                ),
                "provider_observation_descriptor": descriptor,
                "provider_observation_digest": descriptor_dto.descriptor_digest,
            }
        )
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "split": split,
        "episode_id": plan.episode_id,
        "clip_id": f"{split}:{plan.episode_id}:clip",
        "frame_id": frame_id,
        "sequence_number": sequence_number,
        "event_type": event_type,
        "family": "bounded_translation",
        "expected_disposition": "valid",
        "episode_family": plan.episode_family,
        "episode_disposition": plan.episode_disposition,
        "frame_disposition": "valid_frame_payload",
        "denominator_class": plan.denominator_class,
        "expected_row": plan.source_row_id,
        "expected_action": "left",
        "actual_executed_action": "left",
        "action_known": True,
        "gap_declaration": None,
        "observation_pixel_digest": pixel_digest,
        "metadata": metadata,
        "pixels": pixels,
    }


def sample_gap_record(
    *,
    plan=None,
    sequence_number: int = 0,
    frame_index: int | None = None,
) -> dict[str, Any]:
    record = sample_record(
        plan=plan,
        sequence_number=sequence_number,
        frame_index=frame_index,
        pixels=None,
        event_type="gap_unknown",
    )
    record["expected_row"] = None
    record["expected_action"] = None
    record["actual_executed_action"] = None
    record["action_known"] = False
    record["gap_declaration"] = "declared_gap"
    record["metadata"]["observation_operation_chain"] = _chain(None)
    return record


def assert_records_equivalent(left: dict[str, Any], right: dict[str, Any]) -> None:
    left = dict(left)
    right = dict(right)
    left_pixels = left.pop("pixels", None)
    right_pixels = right.pop("pixels", None)
    assert left == right
    if left_pixels is None or right_pixels is None:
        assert left_pixels is right_pixels
    else:
        np.testing.assert_array_equal(left_pixels, right_pixels)


def test_operation_dto_round_trip_hashing_and_tamper_rejection() -> None:
    digest = "sha256:" + "1" * 64
    payload = _operation(input_digests=(digest,), output_digest=digest)
    dto = ObservationOperationDTO.from_dict(payload)

    assert dto.to_dict() == payload
    returned = dto.to_dict()
    returned["parameters"]["event_type"] = "changed"  # type: ignore[index]
    assert dto.to_dict() == payload

    bad_index = payload | {"index": -1}
    with pytest.raises(VPMValidationError, match="index cannot be negative"):
        ObservationOperationDTO.from_dict(bad_index)

    bad_parameter = deepcopy(payload)
    bad_parameter["parameters"]["event_type"] = "changed"
    with pytest.raises(VPMValidationError, match="parameter digest mismatch"):
        ObservationOperationDTO.from_dict(bad_parameter)

    bad_output = payload | {"output_digest": "not-sha256"}
    with pytest.raises(VPMValidationError, match="digest is not sha256"):
        ObservationOperationDTO.from_dict(bad_output)


def test_operation_chain_round_trip_typed_gap_and_tamper_rejection() -> None:
    digest = "sha256:" + "2" * 64
    payload = _chain(digest)
    dto = ObservationOperationChainDTO.from_dict(payload)

    assert dto.to_dict() == payload
    assert (
        ObservationOperationChainDTO.from_dict(_chain(None)).final_emitted_digest
        is None
    )

    bad_indexes = deepcopy(payload)
    bad_indexes["operations"][0]["index"] = 1
    bad_indexes["operations"][0]["operation_digest"] = canonical_sha256(
        {
            key: value
            for key, value in bad_indexes["operations"][0].items()
            if key != "operation_digest"
        }
    )
    bad_indexes["operation_chain_digest"] = canonical_sha256(
        {
            key: value
            for key, value in bad_indexes.items()
            if key != "operation_chain_digest"
        }
    )
    with pytest.raises(VPMValidationError, match="indexes are not contiguous"):
        ObservationOperationChainDTO.from_dict(bad_indexes)

    bad_final = deepcopy(payload)
    bad_final["final_emitted_digest"] = "sha256:" + "0" * 64
    bad_final["operation_chain_digest"] = canonical_sha256(
        {
            key: value
            for key, value in bad_final.items()
            if key != "operation_chain_digest"
        }
    )
    with pytest.raises(VPMValidationError, match="final digest mismatch"):
        ObservationOperationChainDTO.from_dict(bad_final)


def test_provider_descriptor_round_trip_and_validation() -> None:
    descriptor = _provider_descriptor(_pixels(), "frame:source")
    dto = ProviderObservationDescriptorDTO.from_dict(descriptor)

    assert dto.to_dict() == descriptor
    returned = dto.to_dict()
    returned["metadata"]["changed"] = True  # type: ignore[index]
    assert dto.to_dict() == descriptor

    with pytest.raises(VPMValidationError, match="shape mismatch"):
        ProviderObservationDescriptorDTO.from_dict(descriptor | {"shape": [0, 28]})
    with pytest.raises(VPMValidationError, match="raw digest"):
        ProviderObservationDescriptorDTO.from_dict(descriptor | {"raw_digest": "bad"})


def test_materialized_observation_record_round_trip_and_identity_separation() -> None:
    record = sample_record(pixels=_pixels())
    item = MaterializedObservationDTO.from_record(record)

    assert item.matrix_blob is not None
    assert item.observation.observation_pixel_digest != item.matrix_blob.blob_id
    assert item.observation.matrix_blob_id == item.matrix_blob.blob_id
    assert item.matrix_blob.metadata == {
        "kind": "video_action_set_frame_pixels",
        "pixel_digest": item.observation.observation_pixel_digest,
    }
    assert "matrix_blob_id" not in item.to_record(include_pixels=False)
    assert not any(
        key in item.observation.metadata.to_value()
        for key in (
            "observation_operation_chain",
            "provider_observation_descriptor",
            "provider_observation_digest",
        )
    )
    assert_records_equivalent(record, item.to_record(include_pixels=True))

    second = MaterializedObservationDTO.from_record(
        sample_record(sequence_number=1, pixels=_pixels())
    )
    assert second.matrix_blob is not None
    assert second.matrix_blob.blob_id == item.matrix_blob.blob_id


def test_typed_gap_and_observation_validation() -> None:
    gap = MaterializedObservationDTO.from_record(sample_gap_record())

    assert gap.matrix_blob is None
    assert gap.observation.matrix_blob_id is None
    assert gap.observation.observation_pixel_digest is None
    assert gap.to_record(include_pixels=True)["pixels"] is None

    bad_clip = sample_record(pixels=_pixels()) | {"clip_id": "bad"}
    with pytest.raises(VPMValidationError, match="clip id mismatch"):
        MaterializedObservationDTO.from_record(bad_clip)

    bad_action = sample_record(pixels=_pixels()) | {"action_known": False}
    with pytest.raises(VPMValidationError, match="action_known mismatch"):
        MaterializedObservationDTO.from_record(bad_action)

    final_plan = plan_dto(identity=sample_identity(), split="final", frame_count=2)
    final_record = sample_record(split="final", plan=final_plan, pixels=_pixels())
    with pytest.raises(VPMValidationError, match="final split observation"):
        MaterializedObservationDTO.from_record(final_record)


def test_in_memory_store_observation_ownership_atomicity_and_queries() -> None:
    identity = sample_identity()
    plan = plan_dto(identity=identity, split="development", frame_count=3)
    store = InMemoryVideoActionSetStore()
    first = MaterializedObservationDTO.from_record(
        sample_record(plan=plan, sequence_number=0, pixels=_pixels())
    )

    with pytest.raises(VPMValidationError, match=UNKNOWN_BENCHMARK_IDENTITY_MESSAGE):
        store.save_observation(first.observation, matrix_blob=first.matrix_blob)
    store.save_identity(identity)
    with pytest.raises(VPMValidationError, match=UNKNOWN_EPISODE_PLAN_MESSAGE):
        store.save_observation(first.observation, matrix_blob=first.matrix_blob)
    store.save_episode_plan(plan)

    assert store.save_observation(first.observation, matrix_blob=first.matrix_blob) == (
        first.observation
    )
    assert store.save_observation(first.observation, matrix_blob=first.matrix_blob) == (
        first.observation
    )
    assert store.get_observation(first.observation.frame_id) == first.observation
    assert store.get_materialized_observation(first.observation.frame_id) == first

    second = MaterializedObservationDTO.from_record(
        sample_record(plan=plan, sequence_number=1, pixels=_pixels())
    )
    gap = MaterializedObservationDTO.from_record(
        sample_gap_record(plan=plan, sequence_number=2)
    )
    store.save_observations((second, gap))

    assert store.list_observations(split="development") == (
        first.observation,
        second.observation,
        gap.observation,
    )
    assert store.list_observations(family="bounded_translation", has_pixels=True) == (
        first.observation,
        second.observation,
    )
    assert store.list_observations(event_type="gap_unknown", has_pixels=False) == (
        gap.observation,
    )
    assert store.list_observations_by_operation(operation="emit_observation") == (
        first.observation,
        second.observation,
        gap.observation,
    )
    assert store.list_observations_by_output_digest(
        str(first.observation.observation_pixel_digest)
    ) == (first.observation, second.observation)
    assert store.list_observation_consumers_of_digest(
        str(first.observation.observation_pixel_digest)
    ) == (first.observation, second.observation)

    conflict = MaterializedObservationDTO.from_record(
        sample_record(plan=plan, sequence_number=0, pixels=_pixels(3))
    )
    with pytest.raises(VPMValidationError, match=OBSERVATION_CONFLICT_MESSAGE):
        store.save_observation(conflict.observation, matrix_blob=conflict.matrix_blob)

    sequence_conflict = MaterializedObservationDTO.from_record(
        sample_record(
            plan=plan,
            sequence_number=0,
            frame_index=3,
            pixels=_pixels(4),
            metadata_extra={"original_frame_index": 3, "materialized_order": [3, 0]},
        )
    )
    with pytest.raises(VPMValidationError, match=OBSERVATION_SEQUENCE_CONFLICT_MESSAGE):
        store.save_observation(
            sequence_conflict.observation,
            matrix_blob=sequence_conflict.matrix_blob,
        )

    before = store.list_observations()
    third = MaterializedObservationDTO.from_record(
        sample_record(plan=plan, sequence_number=2, pixels=_pixels(5))
    )
    with pytest.raises(VPMValidationError, match=OBSERVATION_CONFLICT_MESSAGE):
        store.save_observations((third, conflict))
    assert store.list_observations() == before


def test_matrix_blob_save_idempotence_and_conflict() -> None:
    store = InMemoryVideoActionSetStore()
    blob = MatrixBlob.from_array(
        _pixels(),
        dtype="uint8",
        metadata={"kind": "test", "pixel_digest": _pixel_digest(_pixels())},
    )
    assert store.save_matrix_blob(blob) == blob
    assert store.save_matrix_blob(blob) == blob

    conflict = MatrixBlob.from_array(
        _pixels(7),
        dtype="uint8",
        metadata={"kind": "test", "pixel_digest": _pixel_digest(_pixels(7))},
    )
    object.__setattr__(conflict, "blob_id", blob.blob_id)
    with pytest.raises(VPMValidationError, match=MATRIX_BLOB_CONFLICT_MESSAGE):
        store.save_matrix_blob(conflict)


def test_runtime_facade_observation_methods_share_one_store() -> None:
    identity = sample_identity()
    plan = plan_dto(identity=identity, split="development", frame_count=2)
    store = InMemoryVideoActionSetStore()
    runtime = build_runtime(video_action_set_store=store)
    record = sample_record(plan=plan, pixels=_pixels())

    assert runtime.video_action_set.engine.observation_service.store is store
    runtime.video_action_set.save_identity(identity)
    runtime.video_action_set.save_episode_plan(plan)
    observation = runtime.video_action_set.save_observation_record(record)

    assert runtime.video_action_set.get_observation(observation.frame_id) == observation
    assert_records_equivalent(
        record,
        runtime.video_action_set.get_observation_record(observation.frame_id),
    )
    assert runtime.video_action_set.list_observations_by_operation(
        operation="emit_observation"
    ) == (observation,)


def test_legacy_frame_descriptor_round_trips_through_runtime() -> None:
    identity = sample_identity()
    plan = plan_dto(identity=identity, split="development", frame_count=2)
    runtime = build_runtime()
    runtime.video_action_set.save_identity(identity)
    runtime.video_action_set.save_episode_plan(plan)
    pixels = _pixels()
    digest = _pixel_digest(pixels)
    record = benchmark._frame_descriptor(
        split="development",
        episode_id=plan.episode_id,
        frame_index=0,
        row_id=plan.source_row_id,
        expected_action="left",
        actual_action="left",
        family="bounded_translation",
        pixels=pixels,
        expected_disposition="valid",
        episode_family=plan.episode_family,
        episode_disposition=plan.episode_disposition,
        frame_disposition="valid_frame_payload",
        denominator_class=plan.denominator_class,
        metadata={
            "episode_seed": plan.episode_seed,
            "seed_digest": identity.seed_digest,
            "derived_seed_identity": plan.derived_seed_identity,
            "episode_plan_digest": plan.plan_digest,
            "frame_seed_identity": plan.frame_plans[0].to_value()[
                "frame_seed_identity"
            ],
            "observation_operation_chain": _chain(digest),
        },
    ) | {"pixels": pixels}

    observation = runtime.video_action_set.save_observation_record(record)

    assert_records_equivalent(
        record,
        runtime.video_action_set.get_observation_record(observation.frame_id),
    )


def test_core_import_remains_sqlalchemy_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import zeromodel; print('sqlalchemy' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
