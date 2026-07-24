from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from zeromodel.perception import (
    DiscreteActionSchemaDTO,
    PerceptionTemporalInferenceError,
    SourceImageEncoderSpecDTO,
    TemporalSourceVPMDTO,
    TemporalWindowSpecDTO,
    build_grid_field_schema,
    encode_source_array,
    fit_temporal_translator,
)
from zeromodel.perception.temporal_inference import (
    TEMPORAL_FIT_ORDER_SEMANTICS,
    TEMPORAL_TRANSLATOR_VERSION,
)


def _source(value0: int, value1: int, action: str, index: int) -> TemporalSourceVPMDTO:
    frame_spec = SourceImageEncoderSpecDTO(color_space="L")
    first = encode_source_array(np.asarray([[value0]], dtype=np.uint8), frame_spec)
    current = encode_source_array(np.asarray([[value1]], dtype=np.uint8), frame_spec)
    montage_spec = SourceImageEncoderSpecDTO(
        color_space="L",
        max_width=2,
        max_height=1,
        max_pixels=2,
        version="test-p16c-montage/1",
    )
    montage = encode_source_array(
        np.asarray([[value0, value1]], dtype=np.uint8),
        montage_spec,
    )
    window = TemporalWindowSpecDTO(frame_count=2)
    return TemporalSourceVPMDTO(
        temporal_source_id=f"sha256:p16c-{index}",
        temporal_window_spec_id=window.temporal_window_spec_id,
        sequence_id=f"sequence-{index}",
        target_interaction_id=f"interaction-{index}",
        target_step_index=1,
        action_label=action,
        frame_source_vpm_ids=(first.source_vpm_id, current.source_vpm_id),
        frame_pixel_digests=(first.pixel_digest, current.pixel_digest),
        current_source_vpm_id=current.source_vpm_id,
        current_pixel_digest=current.pixel_digest,
        montage_source_vpm=montage,
    )


def test_every_caller_permutation_produces_exactly_one_translator() -> None:
    actions = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    window = TemporalWindowSpecDTO(frame_count=2)
    examples = (
        _source(0, 128, "LEFT", 0),
        _source(255, 128, "RIGHT", 1),
        _source(10, 128, "LEFT", 2),
        _source(245, 128, "RIGHT", 3),
    )
    fields = build_grid_field_schema(
        examples[0].montage_source_vpm,
        tile_width=1,
        tile_height=1,
    )

    fitted = tuple(
        fit_temporal_translator(tuple(order), window, fields, actions)
        for order in permutations(examples)
    )

    assert len({item.temporal_translator_id for item in fitted}) == 1
    assert len({item.coefficients for item in fitted}) == 1
    assert len({item.intercepts for item in fitted}) == 1
    assert all(
        item.training_temporal_source_ids
        == tuple(sorted(source.temporal_source_id for source in examples))
        for item in fitted
    )
    assert all(item.fit_order_semantics == TEMPORAL_FIT_ORDER_SEMANTICS for item in fitted)
    assert all(item.version == TEMPORAL_TRANSLATOR_VERSION for item in fitted)


def test_duplicate_temporal_source_identity_is_rejected() -> None:
    actions = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    window = TemporalWindowSpecDTO(frame_count=2)
    source = _source(0, 128, "LEFT", 0)
    fields = build_grid_field_schema(
        source.montage_source_vpm,
        tile_width=1,
        tile_height=1,
    )

    with pytest.raises(PerceptionTemporalInferenceError, match="identities must be unique"):
        fit_temporal_translator((source, source), window, fields, actions)
