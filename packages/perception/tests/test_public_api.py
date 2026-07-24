from __future__ import annotations

import numpy as np

from zeromodel.perception import (
    PERCEPTION_PACKAGE_VERSION,
    PERCEPTION_STAGE,
    BaselineInferenceConfigDTO,
    DiscreteActionSchemaDTO,
    InMemoryPerceptionDatasetStore,
    SourceImageEncoderSpecDTO,
    build_grid_field_schema,
    encode_source_array,
)


def test_phase_four_a_public_contract() -> None:
    assert PERCEPTION_PACKAGE_VERSION == "1.0.13"
    assert PERCEPTION_STAGE == "P4A"
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
