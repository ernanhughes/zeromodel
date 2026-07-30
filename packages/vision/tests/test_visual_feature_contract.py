from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zeromodel.video.arcade_policy import (
    CELL_PIXELS,
    FRAME_HEIGHT,
    ShooterConfig,
    enumerate_visual_frames,
)
from zeromodel.vision import (
    VisualFeatureSpec,
    extract_visual_features,
    visual_feature_digest,
    visual_input_digest,
    visual_raw_input_digest,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "visual_feature_contract_v1.json"


def _fixture_frame(name: str) -> np.ndarray:
    if name == "grayscale_min":
        return np.zeros((4, 4), dtype=np.uint8)
    if name == "grayscale_max":
        return np.full((4, 4), 255, dtype=np.uint8)
    if name == "rgb_gray":
        return np.repeat(
            np.arange(16, dtype=np.uint8).reshape(4, 4)[:, :, None], 3, axis=2
        )
    if name == "rgba_gray":
        rgb = np.repeat(
            np.arange(16, dtype=np.uint8).reshape(4, 4)[:, :, None], 3, axis=2
        )
        return np.concatenate([rgb, np.full((4, 4, 1), 255, dtype=np.uint8)], axis=2)
    if name == "pooling_boundary":
        return np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [248, 249, 250, 251], [252, 253, 254, 255]],
            dtype=np.uint8,
        )
    if name == "rounding_boundary":
        return np.array(
            [[0, 0, 1, 1], [1, 1, 2, 2], [127, 128, 129, 130], [131, 132, 133, 134]],
            dtype=np.uint8,
        )
    if name == "quantization_boundary":
        return np.array(
            [
                [8, 9, 24, 25],
                [42, 43, 59, 60],
                [127, 128, 144, 145],
                [246, 247, 254, 255],
            ],
            dtype=np.uint8,
        )
    if name == "non_contiguous_view":
        return np.arange(64, dtype=np.uint8).reshape(8, 8)[::2, ::2]
    if name == "signed_int_inside_range":
        return np.arange(16, dtype=np.int16).reshape(4, 4)
    if name == "larger_uint_inside_range":
        return np.arange(16, dtype=np.uint16).reshape(4, 4)
    raise AssertionError(name)


def _reference_features(frame: np.ndarray, spec: VisualFeatureSpec) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        gray = [
            [int(array[row, col]) for col in range(spec.input_width)]
            for row in range(spec.input_height)
        ]
    else:
        gray = []
        for row in range(spec.input_height):
            gray_row = []
            for col in range(spec.input_width):
                red = int(array[row, col, 0])
                green = int(array[row, col, 1])
                blue = int(array[row, col, 2])
                gray_row.append((77 * red + 150 * green + 29 * blue + 128) // 256)
            gray.append(gray_row)

    block_height = spec.input_height // spec.target_height
    block_width = spec.input_width // spec.target_width
    area = block_height * block_width
    out = []
    for target_row in range(spec.target_height):
        for target_col in range(spec.target_width):
            total = 0
            for offset_row in range(block_height):
                for offset_col in range(block_width):
                    total += gray[target_row * block_height + offset_row][
                        target_col * block_width + offset_col
                    ]
            pooled = (total + area // 2) // area
            quantized = (pooled * (spec.quantization_levels - 1) + 127) // 255
            out.append(int(quantized))
    result = np.array(out, dtype=np.uint8)
    result.flags.writeable = False
    return result


def test_visual_feature_contract_golden_vectors() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    spec = VisualFeatureSpec.from_dict(payload["feature_spec"])

    assert spec.digest == payload["feature_spec_digest"]
    for item in payload["fixtures"]:
        frame = _fixture_frame(item["name"])
        features = extract_visual_features(frame, spec)
        assert features.tolist() == item["features"]
        assert _reference_features(frame, spec).tolist() == item["features"]
        assert visual_raw_input_digest(frame, spec) == item["raw_input_digest"]
        assert visual_input_digest(frame, spec) == item["canonical_input_digest"]
        assert visual_feature_digest(features, spec) == item["feature_digest"]
        assert extract_visual_features(frame.copy(), spec).tolist() == item["features"]


def test_reference_feature_contract_parity_for_seeded_random_frames() -> None:
    spec = VisualFeatureSpec(4, 4, 2, 2, quantization_levels=16)
    rng = np.random.default_rng(20260730)

    for _ in range(64):
        frame = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
        assert np.array_equal(
            extract_visual_features(frame, spec),
            _reference_features(frame, spec),
        )


def test_reference_feature_contract_parity_for_all_arcade_canonical_frames() -> None:
    config = ShooterConfig()
    frames = enumerate_visual_frames(config)
    spec = VisualFeatureSpec(
        FRAME_HEIGHT,
        config.width * CELL_PIXELS,
        8,
        config.width * 2,
        quantization_levels=16,
    )

    assert len(frames) == 112
    for frame in frames.values():
        assert np.array_equal(
            extract_visual_features(frame, spec),
            _reference_features(frame, spec),
        )
