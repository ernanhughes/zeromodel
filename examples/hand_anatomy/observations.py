"""Identified observations: an external image/mask, preserved and normalized apart."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from ._common import (
    MAX_IMAGE_SIDE,
    MIN_MASK_PIXELS,
    NOMINAL_AREA,
    WORK_CANVAS,
    UnsupportedInput,
    digest,
    mask_digest,
)

OBSERVATION_SCHEMA = "hand-observation/1"
EXTRACTION_VERSION = "hand-silhouette-evidence/1"
MAX_FILE_BYTES = 25 * 1024 * 1024
ALLOWED_FORMATS = {"PNG", "JPEG", "BMP"}


@dataclass(frozen=True)
class Observation:
    observation_id: str
    mask: np.ndarray = field(repr=False)  # original-resolution bool mask (preserved)
    work_mask: np.ndarray = field(repr=False)  # normalized WORK_CANVAS bool mask
    image_digest: Optional[str]
    image_shape: Tuple[int, ...]
    mask_id: str
    work_mask_id: str
    bbox: Tuple[int, int, int, int]  # y0, x0, y1, x1 (exclusive)
    normalization: Dict[str, float]
    method: Dict[str, Any]
    border_contact_fraction: float
    landmarks: Optional[np.ndarray] = field(default=None, repr=False)  # (21, 2) in work frame
    unknowns: Tuple[str, ...] = ()
    image: Optional[np.ndarray] = field(default=None, repr=False)  # original RGB, if supplied

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def summary(self) -> Dict[str, Any]:
        return {
            "schema_version": OBSERVATION_SCHEMA,
            "observation_id": self.observation_id,
            "image_digest": self.image_digest,
            "image_shape": list(self.image_shape),
            "mask_id": self.mask_id,
            "work_mask_id": self.work_mask_id,
            "bbox_yxyx": list(self.bbox),
            "normalization": self.normalization,
            "method": self.method,
            "border_contact_fraction": round(self.border_contact_fraction, 6),
            "has_landmarks": self.landmarks is not None,
            "unknowns": list(self.unknowns),
        }


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _open_checked(path: Path, role: str) -> Image.Image:
    if not path.is_file():
        raise UnsupportedInput("%s file not found: %s" % (role, path))
    if path.stat().st_size > MAX_FILE_BYTES:
        raise UnsupportedInput("%s file exceeds %d bytes" % (role, MAX_FILE_BYTES))
    try:
        img = Image.open(path)
        fmt = img.format
        img.load()
    except Exception as exc:  # PIL raises many types
        raise UnsupportedInput("%s is not a readable image: %s" % (role, exc)) from exc
    if fmt not in ALLOWED_FORMATS:
        raise UnsupportedInput("%s format %s unsupported (allowed: %s)" % (role, fmt, sorted(ALLOWED_FORMATS)))
    if max(img.size) > MAX_IMAGE_SIDE:
        raise UnsupportedInput("%s larger than %d px on a side" % (role, MAX_IMAGE_SIDE))
    return img


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from .comparison import connected_components

    labels, areas = connected_components(mask)
    if not areas:
        return mask
    return labels == (1 + int(np.argmax(areas)))


def segment_uniform_background(rgb: np.ndarray, threshold: float = 40.0) -> np.ndarray:
    """Bounded segmentation.  APPLIES ONLY when the border is a near-uniform colour.

    Foreground = colour distance from the median border colour > ``threshold``,
    largest component only.  Raises when the border is not uniform.
    """
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(float)
    ref = np.median(border, axis=0)
    if float(np.percentile(np.linalg.norm(border - ref, axis=1), 95)) > threshold / 2:
        raise UnsupportedInput("background is not uniform; supply an explicit mask")
    dist = np.linalg.norm(rgb.astype(float) - ref, axis=2)
    return _largest_component(dist > threshold)


def normalize_to_work_frame(mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Centroid at canvas centre, area scaled to NOMINAL_AREA (declared, uniform)."""
    ys, xs = np.nonzero(mask)
    area = float(len(ys))
    cy0, cx0 = float(ys.mean()), float(xs.mean())
    cy, cx = cy0, cx0
    scale = math.sqrt(NOMINAL_AREA / area)
    img = Image.fromarray((mask * 255).astype(np.uint8), "L")
    pre = 1
    while scale * pre < 0.5 and pre < 64:
        pre *= 2
    if pre > 1:
        img = img.reduce(pre)
        cy, cx = (cy + 0.5) / pre - 0.5, (cx + 0.5) / pre - 0.5
    s = scale * pre  # source px -> work px
    c = WORK_CANVAS / 2.0
    # output (x, y) -> input: in = centroid + (out - c) / s
    data = (1 / s, 0.0, cx - c / s, 0.0, 1 / s, cy - c / s)
    work = img.transform((WORK_CANVAS, WORK_CANVAS), Image.AFFINE, data, resample=Image.BILINEAR)
    out = np.asarray(work) > 127
    return out, {"scale": scale, "centroid_y": cy0, "centroid_x": cx0}


def observation_from_arrays(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    image_digest: Optional[str] = None,
    method: Optional[Dict[str, Any]] = None,
    landmarks: Optional[np.ndarray] = None,
) -> Observation:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise UnsupportedInput("mask must be 2-D")
    if max(mask.shape) > MAX_IMAGE_SIDE:
        raise UnsupportedInput("mask larger than %d px on a side" % MAX_IMAGE_SIDE)
    m = mask.astype(bool)
    area = int(m.sum())
    if area < MIN_MASK_PIXELS:
        raise UnsupportedInput("mask too small or empty (%d px < %d)" % (area, MIN_MASK_PIXELS))
    if area > 0.9 * m.size:
        raise UnsupportedInput("mask covers >90%% of the image (inverted or unsegmented?)")
    if image is not None and image.shape[:2] != m.shape:
        raise UnsupportedInput("image and mask dimensions differ")
    ys, xs = np.nonzero(m)
    bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
    work, norm = normalize_to_work_frame(m)
    clipped = int(((work[0] | work[-1]).sum() + (work[:, 0] | work[:, -1]).sum()))
    if clipped:
        raise UnsupportedInput("normalized silhouette exceeds the working canvas (extreme aspect ratio)")
    ring = np.zeros_like(m)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    from .comparison import boundary_of

    boundary = int(boundary_of(m).sum())
    border = float((m & ring).sum()) / max(1, boundary)
    unknowns = ["handedness", "palm_vs_back_facing", "camera_pose", "occlusion_extent", "depth"]
    method = dict(method or {"mask_source": "array"})
    method["extraction_version"] = EXTRACTION_VERSION
    mid = mask_digest(m.astype(np.uint8))
    lm = None
    if landmarks is not None:
        lm = np.asarray(landmarks, dtype=float)
        if lm.shape != (21, 2) or not np.isfinite(lm).all():
            raise UnsupportedInput("landmarks must be a finite (21, 2) array")
        s = norm["scale"]
        lm = (lm - np.array([norm["centroid_x"], norm["centroid_y"]])) * s + WORK_CANVAS / 2.0
        method["landmark_provider"] = method.get("landmark_provider", "external-unspecified")
    obs_id = digest(
        {
            "schema": OBSERVATION_SCHEMA,
            "image_digest": image_digest,
            "mask_id": mid,
            "method": method,
            "landmarks": None if lm is None else [[round(float(v), 4) for v in p] for p in lm],
        }
    )
    return Observation(
        observation_id=obs_id,
        mask=m,
        work_mask=work,
        image_digest=image_digest,
        image_shape=tuple(image.shape if image is not None else m.shape),
        mask_id=mid,
        work_mask_id=mask_digest(work.astype(np.uint8)),
        bbox=bbox,
        normalization=norm,
        method=method,
        border_contact_fraction=border,
        landmarks=lm,
        unknowns=tuple(unknowns),
        image=image,
    )


def load_observation(
    image_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    landmarks_path: str | Path | None = None,
    segment_background: bool = False,
) -> Observation:
    if image_path is None and mask_path is None:
        raise UnsupportedInput("an image and/or a mask is required")
    image_arr: Optional[np.ndarray] = None
    image_digest: Optional[str] = None
    mask: Optional[np.ndarray] = None
    method: Dict[str, Any] = {}
    if image_path is not None:
        p = Path(image_path)
        img = _open_checked(p, "image")
        image_digest = _file_digest(p)
        has_alpha = img.mode in ("RGBA", "LA", "PA")
        rgba = img.convert("RGBA")
        image_arr = np.asarray(rgba.convert("RGB"))
        alpha = np.asarray(rgba)[:, :, 3]
        if mask_path is None:
            if has_alpha and alpha.min() < 255 and alpha.max() > 0:
                mask = alpha > 127
                method["mask_source"] = "alpha-channel"
            elif segment_background:
                mask = segment_uniform_background(image_arr)
                method["mask_source"] = "uniform-background-threshold"
                method["assumption"] = "near-uniform border colour; largest component"
            else:
                raise UnsupportedInput(
                    "no mask supplied; pass --mask, an RGBA image, or --segment-background"
                )
    if mask_path is not None:
        mp = Path(mask_path)
        mimg = _open_checked(mp, "mask")
        mask = np.asarray(mimg.convert("L")) > 127
        method["mask_source"] = "supplied-mask"
        method["mask_file_digest"] = _file_digest(mp)
        if image_arr is not None and mask.shape != image_arr.shape[:2]:
            raise UnsupportedInput("mask dimensions %s differ from image %s" % (mask.shape, image_arr.shape[:2]))
    assert mask is not None
    landmarks = None
    if landmarks_path is not None:
        try:
            data = json.loads(Path(landmarks_path).read_text(encoding="utf-8"))
            landmarks = np.asarray(data["landmarks"] if isinstance(data, dict) else data, dtype=float)
        except (OSError, ValueError, KeyError) as exc:
            raise UnsupportedInput("invalid landmarks file: %s" % exc) from exc
    return observation_from_arrays(
        mask, image=image_arr, image_digest=image_digest, method=method, landmarks=landmarks
    )
