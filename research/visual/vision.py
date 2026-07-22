"""Public research surface for governed visual-address benchmarks.

The root package remains conservative. Experimental frozen-vision components are
collected here so importing :mod:`zeromodel` does not load model runtimes or
suggest that the learned path is already validated.
"""
from __future__ import annotations

from research.visual.visual_corruptions import (
    add_integer_noise,
    canonical_uint8_frame,
    checkerboard_frame,
    mask_box,
    overlay_background_patch,
    remap_levels,
    scale_intensity,
    translate_frame,
)
from research.visual.visual_encoder import (
    DINO_V2_SMALL_LICENSE,
    DINO_V2_SMALL_MODEL_ID,
    DINO_V2_SMALL_REVISION,
    ENCODER_MANIFEST_VERSION,
    EncoderManifest,
    FrozenVisualEncoder,
    HuggingFaceDinoV2Encoder,
    square_letterbox_uint8,
)
from research.visual.visual_experiment import (
    EXPECTED_ACCEPT,
    EXPECTED_REJECT,
    IMPOSSIBILITY_CONTROL,
    VisualEvaluationTrace,
    build_research_report,
    encode_observations,
    evaluate_visual_provider,
    records_for_split,
    vectors_for_records,
)
from research.visual.visual_precomputed import (
    PrecomputedVectorAddressProvider,
    VectorAddressMatcher,
)
from research.visual.visual_retrieval import (
    LINEAR_PROBE_VERSION,
    VECTOR_ADDRESS_READER_VERSION,
    VECTOR_CALIBRATION_VERSION,
    FrozenVectorAddressProvider,
    LinearProbeBuild,
    LinearProbeIndex,
    NormalizedPixelEncoder,
    VectorAddressBuild,
    VectorAddressIndex,
    VectorCalibration,
    build_linear_probe,
    build_vector_address,
    l2_normalize_rows,
)

__all__ = [
    "DINO_V2_SMALL_LICENSE",
    "DINO_V2_SMALL_MODEL_ID",
    "DINO_V2_SMALL_REVISION",
    "ENCODER_MANIFEST_VERSION",
    "EXPECTED_ACCEPT",
    "EXPECTED_REJECT",
    "EncoderManifest",
    "FrozenVisualEncoder",
    "FrozenVectorAddressProvider",
    "HuggingFaceDinoV2Encoder",
    "IMPOSSIBILITY_CONTROL",
    "LINEAR_PROBE_VERSION",
    "LinearProbeBuild",
    "LinearProbeIndex",
    "NormalizedPixelEncoder",
    "PrecomputedVectorAddressProvider",
    "VECTOR_ADDRESS_READER_VERSION",
    "VECTOR_CALIBRATION_VERSION",
    "VectorAddressBuild",
    "VectorAddressIndex",
    "VectorAddressMatcher",
    "VectorCalibration",
    "VisualEvaluationTrace",
    "add_integer_noise",
    "build_linear_probe",
    "build_research_report",
    "build_vector_address",
    "canonical_uint8_frame",
    "checkerboard_frame",
    "encode_observations",
    "evaluate_visual_provider",
    "l2_normalize_rows",
    "mask_box",
    "overlay_background_patch",
    "records_for_split",
    "remap_levels",
    "scale_intensity",
    "square_letterbox_uint8",
    "translate_frame",
    "vectors_for_records",
]
