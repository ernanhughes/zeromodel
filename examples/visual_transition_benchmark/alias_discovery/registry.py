from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zeromodel.video.arcade_policy import ShooterConfig

from visual_transition_benchmark.alias_discovery._json import digest, write_json
from visual_transition_benchmark.alias_discovery.transforms import TransformSpec

REGISTRY_VERSION = "visual-alias-transform-registry/v1"
REGISTRY_FILE = Path(__file__).with_name("registry-v1.json")


def default_registry() -> tuple[TransformSpec, ...]:
    width = ShooterConfig().width * 4
    shape = (16, width)

    def spec(tid: str, family: str, params: dict[str, Any], *, rgb: bool = False, expected: bool = False) -> TransformSpec:
        return TransformSpec(
            transform_id=tid,
            family=family,
            version="v1",
            parameters=params,
            input_shape=shape,
            output_shape=(16, width, 3) if rgb else shape,
            raw_format_effect=("channel encoding" if rgb else "pixel values/layout"),
            severity_definition="family-local predeclared parameter plus pixel deltas",
            seed_use="fixed seed where stochastic, otherwise unused",
            canonical_identity_expected=expected,
        )

    specs = [
        spec("identity", "representation", {}, expected=True),
        spec("grayscale_to_rgb", "representation", {}, rgb=True, expected=True),
        spec("contiguous_copy", "representation", {}, expected=True),
        spec("noncontiguous_roundtrip", "representation", {}, expected=True),
        spec("uint_roundtrip", "representation", {}, expected=True),
        spec("png_roundtrip", "representation", {}, expected=True),
        spec("translate", "geometric", {"dx": 1, "dy": 0, "fill": 0}),
        spec("translate", "geometric", {"dx": -1, "dy": 0, "fill": 0}),
        spec("translate", "geometric", {"dx": 0, "dy": 1, "fill": 0}),
        spec("translate", "geometric", {"dx": 0, "dy": -1, "fill": 0}),
        spec("downsample_restore", "geometric", {"scale": 0.5, "method": "nearest"}),
        spec("downsample_restore", "geometric", {"scale": 0.5, "method": "bilinear"}),
        spec("crop_pad", "geometric", {"pixels": 1, "fill": 0}),
        spec("shear", "geometric", {"direction": "horizontal"}),
        spec("shear", "geometric", {"direction": "vertical"}),
        spec("brightness", "photometric", {"offset": 16}),
        spec("brightness", "photometric", {"offset": -16}),
        spec("contrast", "photometric", {"factor": 0.75}),
        spec("contrast", "photometric", {"factor": 1.25}),
        spec("gamma", "photometric", {"gamma": 0.8}),
        spec("gamma", "photometric", {"gamma": 1.25}),
        spec("quantize", "photometric", {"levels": 8}),
        spec("box_blur", "blur_compression", {"radius": 1}),
        spec("gaussian_blur", "blur_compression", {"radius": 0.75}),
        spec("median_filter", "blur_compression", {"size": 3}),
        spec("jpeg_roundtrip", "blur_compression", {"quality": 70}),
        spec("occlusion", "occlusion", {"x": 0, "y": 0, "w": 4, "h": 4, "fill": 0}),
        spec("occlusion", "occlusion", {"x": 12, "y": 6, "w": 6, "h": 4, "fill": 0}),
        spec("occlusion", "occlusion", {"x": 8, "y": 11, "w": 8, "h": 3, "fill": 0}),
        spec("occlusion", "occlusion", {"x": width - 8, "y": 7, "w": 8, "h": 2, "fill": 0}),
        spec("salt_pepper", "noise", {"count": 12}),
        spec("uniform_noise", "noise", {"amount": 20}),
        spec("gaussian_noise", "noise", {"sigma": 12.0}),
        spec("dropout", "noise", {"fraction": 0.08, "fill": 0}),
        spec("stripe_noise", "noise", {"every": 5, "fill": 255}),
        spec("pixel_mutation", "local_corruption", {"x": 0, "y": 0, "value": 1}),
        spec("square_mutation", "local_corruption", {"x": 10, "y": 10, "size": 2, "value": 255}),
        spec("row_stripe", "local_corruption", {"y": 8, "value": 255}),
        spec("column_stripe", "local_corruption", {"x": 4, "value": 255}),
        spec("checkerboard", "local_corruption", {"value": 255}),
        spec("band_mask", "local_corruption", {"band": "tank", "value": 0}),
        spec("band_mask", "local_corruption", {"band": "alien", "value": 0}),
        spec("band_mask", "local_corruption", {"band": "cooldown", "value": 0}),
        spec("invert", "negative_control", {}),
        spec("translate", "negative_control", {"dx": 8, "dy": 0, "fill": 0}),
        spec("uniform_noise", "negative_control", {"amount": 120}),
    ]
    return tuple(specs)


def registry_payload(specs: tuple[TransformSpec, ...] | None = None) -> dict[str, Any]:
    specs = specs or default_registry()
    return {
        "registry_version": REGISTRY_VERSION,
        "registry_kind": "target_agnostic_static_observation_transforms",
        "target_row_leakage_policy": "transform inputs are source observation, transform spec, and optional fixed seed only",
        "transition_leakage_policy": "no after frames or transition outcomes are inputs to discovery or membership",
        "transforms": [item.to_dict() for item in specs],
    }


def registry_id(specs: tuple[TransformSpec, ...] | None = None) -> str:
    return digest(registry_payload(specs))


def write_default_registry(path: Path = REGISTRY_FILE) -> str:
    payload = registry_payload()
    write_json(path, payload)
    return digest(payload)


def load_registry(path: Path) -> tuple[TransformSpec, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        TransformSpec(
            transform_id=str(item["transform_id"]),
            family=str(item["family"]),
            version=str(item["version"]),
            parameters=dict(item["parameters"]),
            input_shape=tuple(int(x) for x in item["input_shape"]),
            output_shape=tuple(int(x) for x in item["output_shape"]),
            raw_format_effect=str(item["raw_format_effect"]),
            severity_definition=str(item["severity_definition"]),
            seed_use=str(item["seed_use"]),
            canonical_identity_expected=bool(item["canonical_identity_expected"]),
        )
        for item in data["transforms"]
    )
