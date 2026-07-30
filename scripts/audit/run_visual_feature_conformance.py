"""Generate visual feature-contract conformance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zeromodel.vision import (
    VisualFeatureSpec,
    extract_visual_features,
    visual_feature_digest,
    visual_input_digest,
    visual_raw_input_digest,
)


FIXTURE = Path("packages/vision/tests/fixtures/visual_feature_contract_v1.json")


def _frame(name: str) -> np.ndarray:
    if name == "grayscale_min":
        return np.zeros((4, 4), dtype=np.uint8)
    if name == "grayscale_max":
        return np.full((4, 4), 255, dtype=np.uint8)
    if name in {"rgb_gray", "rgba_gray"}:
        rgb = np.repeat(
            np.arange(16, dtype=np.uint8).reshape(4, 4)[:, :, None], 3, axis=2
        )
        if name == "rgb_gray":
            return rgb
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
    raise ValueError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    spec = VisualFeatureSpec.from_dict(payload["feature_spec"])
    rows = []
    passed = 0
    for item in payload["fixtures"]:
        frame = _frame(item["name"])
        features = extract_visual_features(frame, spec)
        checks = {
            "features": features.tolist() == item["features"],
            "raw_input_digest": visual_raw_input_digest(frame, spec)
            == item["raw_input_digest"],
            "canonical_input_digest": visual_input_digest(frame, spec)
            == item["canonical_input_digest"],
            "feature_digest": visual_feature_digest(features, spec)
            == item["feature_digest"],
        }
        passed += int(all(checks.values()))
        rows.append({"name": item["name"], "checks": checks})
    result = {
        "fixture_path": str(FIXTURE),
        "feature_spec_digest": spec.digest,
        "fixture_count": len(rows),
        "passed_count": passed,
        "rows": rows,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
