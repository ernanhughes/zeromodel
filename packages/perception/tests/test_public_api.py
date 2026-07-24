from __future__ import annotations

import numpy as np

from zeromodel.perception import (
    FIELD_RELEVANCE_SEMANTICS,
    PERCEPTION_PACKAGE_VERSION,
    PERCEPTION_STAGE,
    WEIGHTED_DISTANCE_SEMANTICS,
    BaselineInferenceConfigDTO,
    DiscreteActionSchemaDTO,
    InMemoryPerceptionDatasetStore,
    SourceImageEncoderSpecDTO,
    build_grid_field_schema,
    encode_source_array,
)


def test_phase_four_c_public_contract() -> None:
    assert PERCEPTION_PACKAGE_VERSION == "1.0.13"
    assert PERCEPTION_STAGE == "P4C"
    assert FIELD_RELEVANCE_SEMANTICS == "eta_squared_of_field_mean_by_action"
    assert WEIGHTED_DISTANCE_SEMANTICS == (
        "field_relevance_weighted_normalized_mean_absolute_distance"
    )
    assert SourceImageEncoderSpecDTO().color_space == "RGB"
    assert DiscreteActionSchemaDTO.from_labels(["RIGHT", "LEFT"]).labels == (
        "LEFT",
        "RIGHT",
    )
    assert InMemoryPerceptionDatasetStore().list_ids() == ()
    assert BaselineInferenceConfigDTO().neighbor_count == 3

    source = encode_source_array(
        np.zeros((2, 2), dtype=np.uint8),
        SourceImageEncoderSpecDTO(color_space="L"),
    )
    schema = build_grid_field_schema(source, tile_width=1, tile_height=1)
    assert len(schema.fields) == 4
