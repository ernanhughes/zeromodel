from __future__ import annotations

import numpy as np

from zeromodel.perception import (
    DiscreteActionSchemaDTO,
    RecordedInteractionDTO,
    SourceImageEncoderSpecDTO,
    TemporalWindowSpecDTO,
    build_dataset_manifest,
    build_temporal_source_vpms,
    diagnose_temporal_state_completeness,
    encode_discrete_action,
    encode_source_array,
)


def _fixture(*, unresolved: bool = False):
    encoder = SourceImageEncoderSpecDTO(color_space="L")
    action_schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])

    prior_left = encode_source_array(np.full((2, 2), 10, dtype=np.uint8), encoder)
    prior_right = encode_source_array(
        np.full((2, 2), 10 if unresolved else 240, dtype=np.uint8), encoder
    )
    shared_current = encode_source_array(np.full((2, 2), 100, dtype=np.uint8), encoder)

    interactions = (
        RecordedInteractionDTO.from_vpms(
            sequence_id="left-sequence",
            step_index=0,
            source=prior_left,
            target=encode_discrete_action("LEFT", action_schema),
        ),
        RecordedInteractionDTO.from_vpms(
            sequence_id="left-sequence",
            step_index=1,
            source=shared_current,
            target=encode_discrete_action("LEFT", action_schema),
        ),
        RecordedInteractionDTO.from_vpms(
            sequence_id="right-sequence",
            step_index=0,
            source=prior_right,
            target=encode_discrete_action("RIGHT", action_schema),
        ),
        RecordedInteractionDTO.from_vpms(
            sequence_id="right-sequence",
            step_index=1,
            source=shared_current,
            target=encode_discrete_action("RIGHT", action_schema),
        ),
    )
    manifest = build_dataset_manifest(
        interactions,
        source_encoder_spec_ids=[encoder.encoder_spec_id],
        reject_errors=False,
    )
    sources = {
        source.source_vpm_id: source
        for source in (prior_left, prior_right, shared_current)
    }
    return manifest, sources, shared_current


def test_temporal_montage_is_deterministic_and_addressable() -> None:
    manifest, sources, _ = _fixture()
    spec = TemporalWindowSpecDTO(frame_count=2)

    first = build_temporal_source_vpms(manifest, sources, spec)
    second = build_temporal_source_vpms(manifest, sources, spec)

    assert first == second
    assert len(first) == 2
    assert all(item.montage_source_vpm.width == 4 for item in first)
    assert all(item.montage_source_vpm.height == 2 for item in first)
    assert all(len(item.frame_source_vpm_ids) == 2 for item in first)
    assert all(item.frame_source_vpm_ids[-1] == item.current_source_vpm_id for item in first)


def test_prior_context_confirms_incomplete_single_frame_state() -> None:
    manifest, sources, current = _fixture()
    spec = TemporalWindowSpecDTO(frame_count=2)
    temporal = build_temporal_source_vpms(manifest, sources, spec)

    report = diagnose_temporal_state_completeness(manifest, temporal, spec)
    conflict = next(
        item for item in report.groups if item.current_pixel_digest == current.pixel_digest
    )

    assert conflict.action_labels == ("LEFT", "RIGHT")
    assert conflict.temporal_context_count == 2
    assert conflict.temporally_conflicting_context_count == 0
    assert conflict.status == "incomplete_state_confirmed"
    assert report.single_frame_conflict_group_count == 1
    assert report.resolved_by_temporal_context_count == 1
    assert report.unresolved_temporal_group_count == 0


def test_identical_temporal_context_remains_unresolved() -> None:
    manifest, sources, current = _fixture(unresolved=True)
    spec = TemporalWindowSpecDTO(frame_count=2)
    temporal = build_temporal_source_vpms(manifest, sources, spec)

    report = diagnose_temporal_state_completeness(manifest, temporal, spec)
    conflict = next(
        item for item in report.groups if item.current_pixel_digest == current.pixel_digest
    )

    assert conflict.temporal_context_count == 1
    assert conflict.temporally_conflicting_context_count == 1
    assert conflict.status == "insufficient_temporal_support"
    assert report.insufficient_support_group_count == 1


def test_temporal_identity_changes_with_window_contract() -> None:
    first = TemporalWindowSpecDTO(frame_count=2)
    second = TemporalWindowSpecDTO(frame_count=3)

    assert first.temporal_window_spec_id != second.temporal_window_spec_id
