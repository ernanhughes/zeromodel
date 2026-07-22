from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_json_value, canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import FRAME_SHAPE, TRANSFORMATION_FAMILY_VERSION
from zeromodel.video.domains.video_action_set.pixel_digest import array_digest


def _transformation_contract() -> dict[str, Any]:
    return {
        "version": TRANSFORMATION_FAMILY_VERSION,
        "families": {
            "exact": {
                "classification": "valid",
                "dx": [0, 0],
                "dy": [0, 0],
                "scale_percent": [100, 100],
                "offset": [0, 0],
                "occlusion": False,
            },
            "bounded_translation": {
                "classification": "valid",
                "dx": [-1, 1],
                "dy": [-1, 1],
                "scale_percent": [100, 100],
                "offset": [0, 0],
                "occlusion": False,
            },
            "bounded_photometric": {
                "classification": "valid",
                "dx": [0, 0],
                "dy": [0, 0],
                "scale_percent": [90, 105],
                "offset": [0, 5],
                "occlusion": False,
            },
            "bounded_translation_photometric": {
                "classification": "valid",
                "dx": [-1, 1],
                "dy": [-1, 1],
                "scale_percent": [90, 105],
                "offset": [0, 5],
                "occlusion": False,
            },
            "bounded_translation_occlusion": {
                "classification": "valid",
                "dx": [-1, 1],
                "dy": [-1, 1],
                "scale_percent": [100, 100],
                "offset": [0, 0],
                "occlusion": True,
            },
            "compound_bounded": {
                "classification": "valid",
                "dx": [-1, 1],
                "dy": [-1, 1],
                "scale_percent": [90, 105],
                "offset": [0, 5],
                "occlusion": True,
            },
        },
        "occlusion_bounds": {
            "top": [0, 2],
            "left": [0, 2],
            "height": 2,
            "width": 3,
            "value": 64,
        },
    }


def _translation_values_for_seed(seed: int) -> tuple[int, int]:
    rng = random.Random(int(seed))
    return int(rng.choice((-1, 0, 1))), int(rng.choice((-1, 0, 1)))


def _transformation_parameters(family: str, seed: int) -> dict[str, Any]:
    contract = _transformation_contract()["families"]
    if family not in contract:
        raise VPMValidationError("unsupported transformation family")
    rng = random.Random(int(seed))
    spec = contract[family]
    dx = dy = 0
    if spec["dx"] != [0, 0] or spec["dy"] != [0, 0]:
        dx, dy = _translation_values_for_seed(seed)
    scale_percent = int(spec["scale_percent"][0])
    offset = int(spec["offset"][0])
    if spec["scale_percent"][0] != spec["scale_percent"][1]:
        scale_percent = 90 + rng.randint(0, 15)
    if spec["offset"][0] != spec["offset"][1]:
        offset = rng.randint(0, 5)
    params: dict[str, Any] = {
        "version": TRANSFORMATION_FAMILY_VERSION,
        "family": family,
        "seed": int(seed),
        "dx": dx,
        "dy": dy,
        "scale_percent": scale_percent,
        "offset": offset,
        "occlusion": None,
    }
    if bool(spec["occlusion"]):
        occlusion = _transformation_contract()["occlusion_bounds"]
        params["occlusion"] = {
            "top": rng.randint(int(occlusion["top"][0]), int(occlusion["top"][1])),
            "left": rng.randint(
                int(occlusion["left"][0]),
                int(occlusion["left"][1]),
            ),
            "height": int(occlusion["height"]),
            "width": int(occlusion["width"]),
            "value": int(occlusion["value"]),
        }
    params["parameter_digest"] = canonical_sha256(
        {key: value for key, value in params.items() if key != "parameter_digest"}
    )
    return params


def _validate_transformation_parameters(
    params: Mapping[str, Any],
    *,
    image_shape: tuple[int, int] = FRAME_SHAPE,
) -> None:
    family = str(params["family"])
    contract = _transformation_contract()
    if family not in contract["families"]:
        raise VPMValidationError("unsupported transformation family")
    spec = contract["families"][family]
    dx = int(params["dx"])
    dy = int(params["dy"])
    scale_percent = int(params["scale_percent"])
    offset = int(params["offset"])
    if not (int(spec["dx"][0]) <= dx <= int(spec["dx"][1])):
        raise VPMValidationError("transformation dx out of bounds")
    if not (int(spec["dy"][0]) <= dy <= int(spec["dy"][1])):
        raise VPMValidationError("transformation dy out of bounds")
    if not (
        int(spec["scale_percent"][0]) <= scale_percent <= int(spec["scale_percent"][1])
    ):
        raise VPMValidationError("transformation scale out of bounds")
    if not (int(spec["offset"][0]) <= offset <= int(spec["offset"][1])):
        raise VPMValidationError("transformation offset out of bounds")
    if bool(spec["occlusion"]):
        occlusion = params.get("occlusion")
        if not isinstance(occlusion, Mapping):
            raise VPMValidationError("occlusion parameters required")
        top = int(occlusion["top"])
        left = int(occlusion["left"])
        height = int(occlusion["height"])
        width = int(occlusion["width"])
        value = int(occlusion["value"])
        bounds = contract["occlusion_bounds"]
        if not (int(bounds["top"][0]) <= top <= int(bounds["top"][1])):
            raise VPMValidationError("occlusion top out of bounds")
        if not (int(bounds["left"][0]) <= left <= int(bounds["left"][1])):
            raise VPMValidationError("occlusion left out of bounds")
        if (
            height != int(bounds["height"])
            or width != int(bounds["width"])
            or value != int(bounds["value"])
        ):
            raise VPMValidationError("occlusion dimensions or value out of bounds")
        if top + height > image_shape[0] or left + width > image_shape[1]:
            raise VPMValidationError("occlusion exceeds image bounds")
    elif params.get("occlusion") is not None:
        raise VPMValidationError(
            "occlusion parameters supplied for non-occlusion family"
        )
    expected = canonical_sha256(
        {
            key: canonical_json_value(value)
            for key, value in params.items()
            if key != "parameter_digest"
        }
    )
    if str(params.get("parameter_digest")) != expected:
        raise VPMValidationError("foreign transformation parameter digest")


def _apply_transformation(
    frame: np.ndarray,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.ascontiguousarray(frame, dtype=np.uint8)
    _validate_transformation_parameters(params, image_shape=tuple(source.shape))
    result = np.array(source, copy=True)
    dx = int(params["dx"])
    dy = int(params["dy"])
    if dx or dy:
        translated = np.full_like(result, 0)
        h, w = result.shape
        x0, x1 = max(0, dx), min(w, w + dx)
        y0, y1 = max(0, dy), min(h, h + dy)
        if x1 > x0 and y1 > y0:
            translated[y0:y1, x0:x1] = result[
                y0 - dy : y1 - dy,
                x0 - dx : x1 - dx,
            ]
        result = translated
    if int(params["scale_percent"]) != 100 or int(params["offset"]) != 0:
        result = np.clip(
            np.round(result.astype(np.float32) * (int(params["scale_percent"]) / 100.0))
            + int(params["offset"]),
            0,
            255,
        ).astype(np.uint8)
    if params.get("occlusion") is not None:
        occlusion = params["occlusion"]
        top = int(occlusion["top"])
        left = int(occlusion["left"])
        height = int(occlusion["height"])
        width = int(occlusion["width"])
        result[top : top + height, left : left + width] = int(occlusion["value"])
    result = np.ascontiguousarray(result, dtype=np.uint8)
    source_digest = array_digest(source)
    output_digest = array_digest(result)
    return result, {
        "source_observation_digest": source_digest,
        "transformed_observation_digest": output_digest,
        "transformation_parameter_digest": params["parameter_digest"],
        "changed_pixel_count": int(np.count_nonzero(source != result)),
    }


__all__ = [
    "_apply_transformation",
    "_transformation_contract",
    "_transformation_parameters",
    "_translation_values_for_seed",
    "_validate_transformation_parameters",
]
