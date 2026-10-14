"""Deterministic CPU renderer: posed capsule skeleton -> labelled 2D silhouette.

Limitations (declared): orthographic projection only; occlusion is simplified
to painter's-order by mean depth; no mesh, no soft tissue, no perspective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ._common import (
    RENDER_RECIPE_VERSION,
    TEMPLATE_CANVAS,
    HandAnatomyError,
    digest,
    mask_digest,
)
from .geometry import (
    COMPONENT_IDS,
    FINGERS,
    Camera,
    HandPose,
    Skeleton,
    build_skeleton,
)


@dataclass(frozen=True)
class RenderRecipe:
    canvas: int = TEMPLATE_CANVAS
    pixels_per_unit: float = 40.0
    palm_padding: float = 0.10
    version: str = RENDER_RECIPE_VERSION

    def validate(self) -> None:
        if not (32 <= self.canvas <= 512):
            raise HandAnatomyError("render canvas must be within [32, 512]")
        if not (4.0 <= self.pixels_per_unit <= 400.0) or not math.isfinite(self.pixels_per_unit):
            raise HandAnatomyError("pixels_per_unit out of range")

    def to_dict(self) -> Dict[str, object]:
        return {
            "canvas": self.canvas,
            "pixels_per_unit": self.pixels_per_unit,
            "palm_padding": self.palm_padding,
            "version": self.version,
        }

    @property
    def recipe_id(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class RenderedTemplate:
    template_id: str
    geometry_id: str
    render_id: str
    name: str
    pose: HandPose
    camera: Camera
    recipe: RenderRecipe
    labels: np.ndarray  # uint8 component-id map
    landmarks: np.ndarray  # (21, 2) pixel coordinates
    visibility: np.ndarray  # (21,) bool

    @property
    def mask(self) -> np.ndarray:
        return self.labels > 0

    @property
    def mask_id(self) -> str:
        return mask_digest(self.labels)

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def _capsule(draw: ImageDraw.ImageDraw, a: np.ndarray, b: np.ndarray, ra: float, rb: float, fill: int) -> None:
    for p, r in ((a, ra), (b, rb)):
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill)
    d = b - a
    n = float(np.hypot(*d))
    if n < 1e-9:
        return
    perp = np.array([-d[1], d[0]]) / n
    quad = [a + perp * ra, b + perp * rb, b - perp * rb, a - perp * ra]
    draw.polygon([tuple(q) for q in quad], fill=fill)


def _project(skel: Skeleton, recipe: RenderRecipe) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    s = recipe.pixels_per_unit

    def p2(p: np.ndarray) -> np.ndarray:
        return np.array([p[0] * s, -p[1] * s])

    lm = np.stack([p2(p) for p in skel.landmarks])
    extents = [lm]
    for segs in skel.capsules.values():
        for a, b, ra, rb in segs:
            for p, r in ((a, ra), (b, rb)):
                c = p2(p)
                extents.append(np.array([c - r * s, c + r * s]))
    for p in skel.palm_polygon:
        c = p2(p)
        extents.append(np.array([c - recipe.palm_padding * s, c + recipe.palm_padding * s]))
    fa, fb, fr = skel.forearm
    for p in (fa, fb):
        c = p2(p)
        extents.append(np.array([c - fr * s, c + fr * s]))
    allpts = np.concatenate(extents)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    shift = np.array([recipe.canvas / 2.0, recipe.canvas / 2.0]) - (lo + hi) / 2.0
    if (hi - lo).max() > recipe.canvas - 4:
        raise HandAnatomyError("posed hand does not fit the render canvas")
    return {"shift": shift, "scale": np.array([s])}, lm + shift


def render_template(
    pose: HandPose,
    camera: Camera = Camera(),
    recipe: RenderRecipe = RenderRecipe(),
    name: str = "",
) -> RenderedTemplate:
    recipe.validate()
    skel = build_skeleton(pose, camera)
    info, landmarks = _project(skel, recipe)
    shift, s = info["shift"], float(info["scale"][0])

    def p2(p: np.ndarray) -> np.ndarray:
        return np.array([p[0] * s, -p[1] * s]) + shift

    img = Image.new("L", (recipe.canvas, recipe.canvas), 0)
    draw = ImageDraw.Draw(img)
    fa, fb, fr = skel.forearm
    _capsule(draw, p2(fa), p2(fb), fr * s, fr * s * 0.95, COMPONENT_IDS["forearm"])
    pad = recipe.palm_padding * s
    poly = [tuple(p2(p)) for p in skel.palm_polygon]
    draw.polygon(poly, fill=COMPONENT_IDS["palm"])
    for i in range(len(poly)):
        a, b = np.array(poly[i]), np.array(poly[(i + 1) % len(poly)])
        _capsule(draw, a, b, pad, pad, COMPONENT_IDS["palm"])
    order = sorted(
        FINGERS,
        key=lambda f: (float(np.mean([(a[2] + b[2]) / 2 for a, b, _, _ in skel.capsules[f]])), f),
    )
    for finger in order:
        for a, b, ra, rb in skel.capsules[finger]:
            _capsule(draw, p2(a), p2(b), ra * s, rb * s, COMPONENT_IDS[finger])
    labels = np.asarray(img, dtype=np.uint8).copy()
    geometry_id = pose.geometry_id
    render_id = digest(
        {"geometry_id": geometry_id, "camera": camera.to_dict(), "recipe": recipe.to_dict()}
    )
    return RenderedTemplate(
        template_id=render_id,
        geometry_id=geometry_id,
        render_id=render_id,
        name=name or "%s-%s" % (pose.handedness, geometry_id[7:15]),
        pose=pose,
        camera=camera,
        recipe=recipe,
        labels=labels,
        landmarks=landmarks,
        visibility=skel.visibility.copy(),
    )
