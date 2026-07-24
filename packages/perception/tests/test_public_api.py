from __future__ import annotations

import numpy as np

from zeromodel.perception import (
    COEFFICIENT_SEMANTICS,
    FIELD_RELEVANCE_SEMANTICS,
    PERCEPTION_PACKAGE_VERSION,
    PERCEPTION_STAGE,
    RECONSTRUCTION_ERROR_SEMANTICS,
    REGISTRATION_SEMANTICS,
    REJECTION_SEMANTICS,
    TARGET_SCORE_SEMANTICS,
    WEIGHTED_DISTANCE_SEMANTICS,
    BaselineInferenceConfigDTO,
    DiscreteActionSchemaDTO,
    InMemoryPerceptionDatasetStore,
    SourceImageEncoderSpecDTO,
    TranslatorConfigDTO,
    build_grid_field_schema,
    encode_source_array,
)


def test_phase_six_public_contract() -> None:
    assert PERCEPTION_PACKAGE_VERSION == "1.0.13"
    assert PERCEPTION_STAGE == "P6"
    assert FIELD_RELEVANCE_SEMANTICS == "eta_squared_of_field_mean_by_action"
    assert WEIGHTED_DISTANCE_SEMANTICS == (
        "field_relevance_weighted_normalized_mean_absolute_distance"
    )
    assert COEFFICIENT_SEMANTICS == (
        "ridge_linear_mapping_with_unregularized_intercept"
    )
    assert TARGET_SCORE_SEMANTICS == "clipped_ridge_predicted_one_hot_field_value"
    assert RECONSTRUCTION_ERROR_SEMANTICS == (
        "mean_absolute_error_against_one_hot_target"
    )
    assert REJECTION_SEMANTICS == (
        "reject_when_top_two_margin_below_calibrated_threshold"
    )
    assert REGISTRATION_SEMANTICS == (
        "share_of_absolute_translator_coefficient_mass_in_declared_source_fields"
    )
    assert TranslatorConfigDTO().ridge_alpha == 1e-6
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
