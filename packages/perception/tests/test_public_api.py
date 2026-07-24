from __future__ import annotations

import numpy as np
from zeromodel.perception import (
    COEFFICIENT_SEMANTICS,
    DIFFERENCE_SURFACE_SEMANTICS,
    FIELD_RELEVANCE_SEMANTICS,
    PERCEPTION_PACKAGE_VERSION,
    PERCEPTION_STAGE,
    RECONSTRUCTION_ERROR_SEMANTICS,
    REGISTRATION_SEMANTICS,
    REJECTION_SEMANTICS,
    TARGET_SCORE_SEMANTICS,
    TEMPORAL_COMPARISON_SEMANTICS,
    TEMPORAL_DIAGNOSIS_SEMANTICS,
    TEMPORAL_FEATURE_SEMANTICS,
    TEMPORAL_LAYOUT_SEMANTICS,
    TEMPORAL_REJECTION_SEMANTICS,
    UNEXPLAINED_SURFACE_SEMANTICS,
    WEIGHTED_DISTANCE_SEMANTICS,
    BaselineInferenceConfigDTO,
    DiscreteActionSchemaDTO,
    InMemoryPerceptionDatasetStore,
    SourceImageEncoderSpecDTO,
    TemporalWindowSpecDTO,
    TranslatorConfigDTO,
    build_grid_field_schema,
    encode_source_array,
)


def test_phase_nine_public_contract() -> None:
    assert PERCEPTION_PACKAGE_VERSION == "1.0.13"
    assert PERCEPTION_STAGE == "P9"
    assert FIELD_RELEVANCE_SEMANTICS == "eta_squared_of_field_mean_by_action"
    assert WEIGHTED_DISTANCE_SEMANTICS == (
        "field_relevance_weighted_normalized_mean_absolute_distance"
    )
    assert COEFFICIENT_SEMANTICS == "ridge_linear_mapping_with_unregularized_intercept"
    assert TARGET_SCORE_SEMANTICS == "clipped_ridge_predicted_one_hot_field_value"
    assert RECONSTRUCTION_ERROR_SEMANTICS == "mean_absolute_error_against_one_hot_target"
    assert REJECTION_SEMANTICS == "reject_when_top_two_margin_below_calibrated_threshold"
    assert REGISTRATION_SEMANTICS == (
        "share_of_absolute_translator_coefficient_mass_in_declared_source_fields"
    )
    assert DIFFERENCE_SURFACE_SEMANTICS == "signed_observed_minus_expected_registration"
    assert UNEXPLAINED_SURFACE_SEMANTICS == (
        "observed_registration_outside_declared_expected_annotations"
    )
    assert TEMPORAL_LAYOUT_SEMANTICS == "oldest_to_current_horizontal_frame_montage"
    assert TEMPORAL_DIAGNOSIS_SEMANTICS == (
        "exact_current_pixel_identity_conflict_resolved_by_exact_prior_context_identity"
    )
    assert TEMPORAL_FEATURE_SEMANTICS == (
        "normalized_mean_intensity_per_declared_temporal_montage_field"
    )
    assert TEMPORAL_COMPARISON_SEMANTICS == (
        "aligned_held_out_single_frame_vs_fixed_window_ridge_translation"
    )
    assert TEMPORAL_REJECTION_SEMANTICS == (
        "reject_when_top_two_margin_below_declared_comparison_threshold"
    )
    assert TemporalWindowSpecDTO(frame_count=2).frame_count == 2
    assert TranslatorConfigDTO().ridge_alpha == 1e-6
    assert SourceImageEncoderSpecDTO().color_space == "RGB"
    assert DiscreteActionSchemaDTO.from_labels(["RIGHT", "LEFT"]).labels == (
        "LEFT", "RIGHT"
    )
    assert InMemoryPerceptionDatasetStore().list_ids() == ()
    assert BaselineInferenceConfigDTO().neighbor_count == 3

    source = encode_source_array(
        np.zeros((2, 2), dtype=np.uint8), SourceImageEncoderSpecDTO(color_space="L")
    )
    schema = build_grid_field_schema(source, tile_width=1, tile_height=1)
    assert len(schema.fields) == 4
