from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class TransformSpec:
    transform_id: str
    family: str
    version: str
    parameters: Mapping[str, Any]
    input_shape: tuple[int, int]
    output_shape: tuple[int, ...]
    raw_format_effect: str
    severity_definition: str
    seed_use: str
    canonical_identity_expected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "family": self.family,
            "version": self.version,
            "parameters": dict(sorted(self.parameters.items())),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "raw_format_effect": self.raw_format_effect,
            "severity_definition": self.severity_definition,
            "seed_use": self.seed_use,
            "canonical_identity_expected": self.canonical_identity_expected,
        }


def changed_stats(source: np.ndarray, transformed: np.ndarray) -> tuple[float, float, int]:
    left = np.asarray(source, dtype=np.int16)
    right = np.asarray(transformed, dtype=np.int16)
    if right.ndim == 3:
        right = right[:, :, 0]
    delta = np.abs(left - right)
    return (
        float(np.count_nonzero(delta) / delta.size),
        float(delta.mean()),
        int(delta.max(initial=0)),
    )


def _pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="L")


def transform_frame(
    source_observation: np.ndarray, spec: TransformSpec, *, seed: int | None = None
) -> np.ndarray:
    frame = np.asarray(source_observation, dtype=np.uint8)
    rng = np.random.default_rng(seed if seed is not None else 0)
    p = spec.parameters
    tid = spec.transform_id
    if tid == "identity":
        return np.array(frame, copy=True)
    if tid == "grayscale_to_rgb":
        return np.stack((frame, frame, frame), axis=-1).astype(np.uint8, copy=False)
    if tid == "contiguous_copy":
        return np.ascontiguousarray(frame)
    if tid == "noncontiguous_roundtrip":
        return np.array(frame[:, ::-1][:, ::-1], copy=False)
    if tid == "uint_roundtrip":
        return np.rint(frame.astype(np.float32) / 255.0 * 255.0).astype(np.uint8)
    if tid == "png_roundtrip":
        out = io.BytesIO()
        _pil(frame).save(out, format="PNG")
        out.seek(0)
        return np.array(Image.open(out).convert("L"), dtype=np.uint8)
    if tid == "jpeg_roundtrip":
        out = io.BytesIO()
        _pil(frame).save(out, format="JPEG", quality=int(p["quality"]))
        out.seek(0)
        return np.array(Image.open(out).convert("L"), dtype=np.uint8)
    if tid == "translate":
        dx, dy, fill = int(p["dx"]), int(p["dy"]), int(p.get("fill", 0))
        out = np.full_like(frame, fill)
        src_y0, src_y1 = max(0, -dy), frame.shape[0] - max(0, dy)
        src_x0, src_x1 = max(0, -dx), frame.shape[1] - max(0, dx)
        dst_y0, dst_y1 = max(0, dy), frame.shape[0] - max(0, -dy)
        dst_x0, dst_x1 = max(0, dx), frame.shape[1] - max(0, -dx)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
        return out
    if tid == "downsample_restore":
        scale = float(p["scale"])
        resample = Image.Resampling.BILINEAR if p["method"] == "bilinear" else Image.Resampling.NEAREST
        small = _pil(frame).resize(
            (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
            resample=resample,
        )
        return np.array(small.resize((frame.shape[1], frame.shape[0]), resample=resample), dtype=np.uint8)
    if tid == "crop_pad":
        pixels, fill = int(p["pixels"]), int(p.get("fill", 0))
        out = np.full_like(frame, fill)
        out[pixels:, pixels:] = frame[:-pixels, :-pixels]
        return out
    if tid == "shear":
        direction = str(p["direction"])
        out = np.array(frame, copy=True)
        if direction == "horizontal":
            out[1::2] = np.roll(out[1::2], 1, axis=1)
        else:
            out[:, 1::2] = np.roll(out[:, 1::2], 1, axis=0)
        return out
    if tid == "brightness":
        return np.clip(frame.astype(np.int16) + int(p["offset"]), 0, 255).astype(np.uint8)
    if tid == "contrast":
        factor = float(p["factor"])
        return np.clip((frame.astype(np.float32) - 128.0) * factor + 128.0, 0, 255).astype(np.uint8)
    if tid == "gamma":
        gamma = float(p["gamma"])
        return np.clip(((frame.astype(np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    if tid == "quantize":
        levels = int(p["levels"])
        step = 255 / (levels - 1)
        return np.rint(np.rint(frame.astype(np.float32) / step) * step).astype(np.uint8)
    if tid == "box_blur":
        return np.array(_pil(frame).filter(ImageFilter.BoxBlur(int(p["radius"]))), dtype=np.uint8)
    if tid == "gaussian_blur":
        return np.array(_pil(frame).filter(ImageFilter.GaussianBlur(float(p["radius"]))), dtype=np.uint8)
    if tid == "median_filter":
        return np.array(_pil(frame).filter(ImageFilter.MedianFilter(int(p["size"]))), dtype=np.uint8)
    if tid == "occlusion":
        out = np.array(frame, copy=True)
        x, y, w, h, fill = (int(p[k]) for k in ("x", "y", "w", "h", "fill"))
        out[y : y + h, x : x + w] = fill
        return out
    if tid == "salt_pepper":
        out = np.array(frame, copy=True)
        count = int(p["count"])
        ys = rng.integers(0, out.shape[0], size=count)
        xs = rng.integers(0, out.shape[1], size=count)
        vals = rng.choice(np.array([0, 255], dtype=np.uint8), size=count)
        out[ys, xs] = vals
        return out
    if tid == "uniform_noise":
        amount = int(p["amount"])
        noise = rng.integers(-amount, amount + 1, size=frame.shape)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if tid == "gaussian_noise":
        sigma = float(p["sigma"])
        noise = rng.normal(0.0, sigma, size=frame.shape)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if tid == "dropout":
        out = np.array(frame, copy=True)
        mask = rng.random(frame.shape) < float(p["fraction"])
        out[mask] = int(p.get("fill", 0))
        return out
    if tid == "stripe_noise":
        out = np.array(frame, copy=True)
        every = int(p["every"])
        out[:, ::every] = int(p.get("fill", 255))
        return out
    if tid == "pixel_mutation":
        out = np.array(frame, copy=True)
        out[int(p["y"]), int(p["x"])] = int(p["value"])
        return out
    if tid == "square_mutation":
        out = np.array(frame, copy=True)
        x, y, size, value = (int(p[k]) for k in ("x", "y", "size", "value"))
        out[y : y + size, x : x + size] = value
        return out
    if tid == "row_stripe":
        out = np.array(frame, copy=True)
        out[int(p["y"]), :] = int(p["value"])
        return out
    if tid == "column_stripe":
        out = np.array(frame, copy=True)
        out[:, int(p["x"])] = int(p["value"])
        return out
    if tid == "checkerboard":
        out = np.array(frame, copy=True)
        out[::2, ::2] = int(p["value"])
        out[1::2, 1::2] = int(p["value"])
        return out
    if tid == "band_mask":
        out = np.array(frame, copy=True)
        band = str(p["band"])
        if band == "tank":
            out[11:14, :] = int(p["value"])
        elif band == "alien":
            out[2:5, :] = int(p["value"])
        elif band == "cooldown":
            out[7:9, -4:] = int(p["value"])
        else:
            out[0:2, :] = int(p["value"])
        return out
    if tid == "invert":
        return (255 - frame).astype(np.uint8)
    raise ValueError(f"unknown transform_id: {tid}")


def transform_callable_accepts_no_target() -> Callable[..., np.ndarray]:
    return transform_frame
