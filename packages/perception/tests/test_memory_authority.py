"""Unit tests for future-memory authority (memory-about-memory)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from zeromodel.perception import build_grid_field_schema, encode_source_array
from zeromodel.perception.expected_transition import (
    ExpectedTransitionFieldDTO,
    ExpectedTransitionVPMDTO,
    render_expected_transition_png,
)
from zeromodel.perception.memory_authority import (
    MEMORY_AUTHORITIES,
    FutureMemoryValidityDTO,
    MemoryAuthorityContextDTO,
    MemoryAuthorityPolicyDTO,
    PerceptionMemoryAuthorityError,
    assess_memory_authority,
    record_verification_event,
)
from zeromodel.perception.representation import SourceImageEncoderSpecDTO
from zeromodel.perception.transition_verification import (
    FutureTransitionVerificationDTO,
)


def _verification(status: str, index: int = 0) -> FutureTransitionVerificationDTO:
    return FutureTransitionVerificationDTO(
        verification_id=f"verification-{index}",
        expected_transition_id="expected",
        observed_transition_evidence_id="observed",
        status=status,
        mean_absolute_error=0.1,
        direction_error_rate=0.2,
        changed_field_error_rate=0.3,
    )


def _recorded(statuses: list[str]) -> FutureMemoryValidityDTO:
    validity = None
    for index, status in enumerate(statuses):
        validity = record_verification_event(
            validity,
            _verification(status, index),
            transition_model_id="model",
            action_label="left",
            field_schema_id="schema",
            training_dataset_id="dataset",
        )
    assert validity is not None
    return validity


def test_authority_levels_are_bounded() -> None:
    assert MEMORY_AUTHORITIES == frozenset({"SUPPORT_ONLY", "MAY_VETO", "STALE", "OOD"})
    with pytest.raises(PerceptionMemoryAuthorityError):
        MemoryAuthorityPolicyDTO(minimum_events=0)
    with pytest.raises(PerceptionMemoryAuthorityError):
        MemoryAuthorityPolicyDTO(stale_negative_rate=1.5)


def test_record_accumulates_bounded_evidence() -> None:
    validity = _recorded(["confirmed", "confirmed", "future_projection_mismatch"])
    assert validity.event_count == 3
    assert validity.confirmed_count == 2
    assert validity.mismatch_count == 1
    assert validity.recent_outcomes == ("confirmed", "confirmed", "mismatch")
    assert validity.recent_negative_rate == 1 / 3
    assert validity.staleness_score == 1 / 3
    assert validity.mean_absolute_error == pytest.approx(0.1)
    assert validity.mean_direction_error == pytest.approx(0.2)
    assert validity.mean_changed_error == pytest.approx(0.3)
    first_id = validity.validity_id
    again = record_verification_event(
        None,
        _verification("confirmed", 99),
        transition_model_id="model",
        action_label="left",
        field_schema_id="schema",
        training_dataset_id="dataset",
    )
    assert again.event_count == 1
    assert again.validity_id != first_id


def test_recent_window_is_bounded() -> None:
    validity = _recorded(["confirmed"] * 20)
    assert validity.event_count == 20
    assert len(validity.recent_outcomes) == 16
    assert validity.recent_negative_rate == 0.0
    assert validity.staleness_score == 0.0


def test_record_rejects_cross_model_events() -> None:
    validity = _recorded(["confirmed"])
    with pytest.raises(
        PerceptionMemoryAuthorityError, match="different transition model"
    ):
        record_verification_event(
            validity,
            _verification("confirmed"),
            transition_model_id="other-model",
            action_label="left",
            field_schema_id="schema",
            training_dataset_id="dataset",
        )
    with pytest.raises(PerceptionMemoryAuthorityError, match="different action"):
        record_verification_event(
            validity,
            _verification("confirmed"),
            transition_model_id="model",
            action_label="right",
            field_schema_id="schema",
            training_dataset_id="dataset",
        )


def test_violation_counts_as_negative_evidence() -> None:
    validity = _recorded(
        ["declared_expectation_violation", "declared_expectation_violation"]
    )
    assert validity.violation_count == 2
    assert validity.recent_negative_rate == 1.0


def test_assess_cold_memory_is_permissive_when_confident() -> None:
    projection = _supported_projection(confidence=0.8)
    assessment = assess_memory_authority(None, projection)
    assert assessment.authority == "MAY_VETO"
    assert assessment.validity_id is None
    timid = _supported_projection(confidence=0.1)
    assert assess_memory_authority(None, timid).authority == "SUPPORT_ONLY"


def test_assess_stale_memory_loses_veto() -> None:
    projection = _supported_projection(confidence=0.9)
    stale = _recorded(["future_projection_mismatch"] * 4)
    assessment = assess_memory_authority(stale, projection)
    assert assessment.authority == "STALE"
    assert assessment.staleness_score == 1.0
    fresh = _recorded(["confirmed"] * 4)
    assert assess_memory_authority(fresh, projection).authority == "MAY_VETO"
    few = _recorded(["future_projection_mismatch"] * 2)
    assert assess_memory_authority(few, projection).authority == "MAY_VETO"


def test_assess_maps_projection_status() -> None:
    ood = _supported_projection(confidence=0.9, status="out_of_distribution")
    assert assess_memory_authority(None, ood).authority == "OOD"
    weak = _supported_projection(confidence=0.9, status="insufficient_examples")
    assert assess_memory_authority(None, weak).authority == "SUPPORT_ONLY"


def test_context_sorts_and_resolves() -> None:
    fresh = _recorded(["confirmed"])
    stale = _recorded(["future_projection_mismatch"])
    context = MemoryAuthorityContextDTO.create({"right": stale, "left": fresh})
    assert [action for action, _ in context.validity_by_action] == [
        "left",
        "right",
    ]
    assert context.validity_for("left") == fresh
    assert context.validity_for("jump") is None
    with pytest.raises(PerceptionMemoryAuthorityError):
        MemoryAuthorityContextDTO(
            validity_by_action=(("left", fresh), ("left", fresh)),
            policy=MemoryAuthorityPolicyDTO(),
        )


def _supported_projection(
    confidence: float, status: str = "supported"
) -> ExpectedTransitionVPMDTO:
    spec = SourceImageEncoderSpecDTO(color_space="L")
    source = encode_source_array(np.zeros((4, 4), dtype=np.uint8), spec)
    schema = build_grid_field_schema(
        source, tile_width=2, tile_height=2, channel_mode="joint"
    )
    fields = tuple(
        ExpectedTransitionFieldDTO(
            field_id=field.field_id,
            expected_after_mean=0.1,
            expected_mean_signed_change=0.05,
            expected_mean_absolute_change=0.05,
            expected_changed_fraction=0.5,
            signed_change_dispersion=0.01,
            absolute_change_dispersion=0.01,
            support_count=6,
        )
        for field in schema.fields
    )
    png, digest = render_expected_transition_png(
        fields, schema, schema.width, schema.height
    )
    payload = {
        "action_label": "left",
        "confidence": confidence,
        "fields": [item.canonical_payload() for item in fields],
        "field_schema_id": schema.field_schema_id,
        "model_id": "model",
        "png_digest": digest,
        "render_semantics": "rounded_uint8_expected_mean_absolute_change_max_over_channels",
        "source_encoder_spec_id": schema.source_encoder_spec_id,
        "source_vpm_id": source.source_vpm_id,
        "status": status,
        "support_count": 6,
        "training_dataset_id": "dataset",
        "version": "perception-expected-transition-vpm/1",
    }
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    hasher = hashlib.sha256()
    hasher.update(len(raw).to_bytes(8, "big"))
    hasher.update(raw)
    return ExpectedTransitionVPMDTO(
        expected_transition_id=f"sha256:{hasher.hexdigest()}",
        source_vpm_id=source.source_vpm_id,
        action_label="left",
        field_schema_id=schema.field_schema_id,
        source_encoder_spec_id=schema.source_encoder_spec_id,
        fields=fields,
        model_id="model",
        training_dataset_id="dataset",
        support_count=6,
        confidence=confidence,
        status=status,
        png_digest=digest,
        png_bytes=png,
    )
