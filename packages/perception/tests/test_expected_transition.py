"""Unit tests for structured expected-transition memory contracts."""

from __future__ import annotations

import numpy as np
import pytest

from zeromodel.perception import (
    EXPECTED_TRANSITION_STATUSES,
    build_grid_field_schema,
    encode_source_array,
)
from zeromodel.perception.expected_transition import (
    EXPECTED_TRANSITION_VPM_VERSION,
    ExpectedTransitionFieldDTO,
    ExpectedTransitionVPMDTO,
    PerceptionExpectedTransitionError,
    render_expected_transition_png,
)
from zeromodel.perception.representation import SourceImageEncoderSpecDTO

_SPEC = SourceImageEncoderSpecDTO(color_space="L")


def _fixture():
    before = encode_source_array(np.zeros((8, 12), dtype=np.uint8), _SPEC)
    schema = build_grid_field_schema(
        before, tile_width=4, tile_height=4, channel_mode="joint"
    )
    return before, schema


def _fields(schema, *, support=6):
    return tuple(
        ExpectedTransitionFieldDTO(
            field_id=field.field_id,
            expected_after_mean=0.1,
            expected_mean_signed_change=0.05,
            expected_mean_absolute_change=0.05,
            expected_changed_fraction=0.5,
            signed_change_dispersion=0.01,
            absolute_change_dispersion=0.01,
            support_count=support,
        )
        for field in schema.fields
    )


def _assemble(schema, fields, **overrides):
    png, digest = render_expected_transition_png(
        fields, schema, schema.width, schema.height
    )
    values = {
        "expected_transition_id": "sha256:" + "0" * 64,
        "source_vpm_id": "sha256:" + "1" * 64,
        "action_label": "left",
        "field_schema_id": schema.field_schema_id,
        "source_encoder_spec_id": schema.source_encoder_spec_id,
        "fields": fields,
        "model_id": "sha256:" + "2" * 64,
        "training_dataset_id": "sha256:" + "3" * 64,
        "support_count": 6,
        "confidence": 0.8,
        "status": "supported",
        "png_digest": digest,
        "png_bytes": png,
    }
    values.update(overrides)
    return values


def _valid_vpm(schema, fields, **overrides):
    import hashlib
    import json

    values = _assemble(schema, fields, **overrides)
    payload = {
        "action_label": values["action_label"],
        "confidence": values["confidence"],
        "fields": [item.canonical_payload() for item in values["fields"]],
        "field_schema_id": values["field_schema_id"],
        "model_id": values["model_id"],
        "png_digest": values["png_digest"],
        "render_semantics": "rounded_uint8_expected_mean_absolute_change_max_over_channels",
        "source_encoder_spec_id": values["source_encoder_spec_id"],
        "source_vpm_id": values["source_vpm_id"],
        "status": values["status"],
        "support_count": values["support_count"],
        "training_dataset_id": values["training_dataset_id"],
        "version": EXPECTED_TRANSITION_VPM_VERSION,
    }
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    hasher = hashlib.sha256()
    hasher.update(len(raw).to_bytes(8, "big"))
    hasher.update(raw)
    values["expected_transition_id"] = f"sha256:{hasher.hexdigest()}"
    return ExpectedTransitionVPMDTO(**values)


def test_expected_field_rejects_out_of_range_moments() -> None:
    _, schema = _fixture()
    field_id = schema.fields[0].field_id
    with pytest.raises(PerceptionExpectedTransitionError):
        ExpectedTransitionFieldDTO(
            field_id=field_id,
            expected_after_mean=1.5,
            expected_mean_signed_change=0.0,
            expected_mean_absolute_change=0.0,
            expected_changed_fraction=0.0,
            signed_change_dispersion=0.0,
            absolute_change_dispersion=0.0,
            support_count=1,
        )
    with pytest.raises(PerceptionExpectedTransitionError):
        ExpectedTransitionFieldDTO(
            field_id=field_id,
            expected_after_mean=0.5,
            expected_mean_signed_change=0.0,
            expected_mean_absolute_change=0.0,
            expected_changed_fraction=0.0,
            signed_change_dispersion=-0.1,
            absolute_change_dispersion=0.0,
            support_count=1,
        )


def test_expected_vpm_status_vocabulary() -> None:
    assert EXPECTED_TRANSITION_STATUSES == frozenset(
        {
            "supported",
            "insufficient_examples",
            "out_of_distribution",
            "ambiguous_future",
            "unsupported_action",
        }
    )
    _, schema = _fixture()
    fields = _fields(schema)
    with pytest.raises(PerceptionExpectedTransitionError):
        _valid_vpm(schema, fields, status="confident_guess")


def test_expected_vpm_roundtrip_and_png() -> None:
    _, schema = _fixture()
    fields = _fields(schema)
    vpm = _valid_vpm(schema, fields)
    array = vpm.to_array()
    assert array.shape == (schema.height, schema.width)
    assert vpm.field_evidence(fields[0].field_id) == fields[0]
    with pytest.raises(KeyError):
        vpm.field_evidence("no-such-field")
    restored = ExpectedTransitionVPMDTO.from_dict(vpm.to_dict())
    assert restored == vpm
