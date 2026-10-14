"""Deterministic, bounded alignment of library templates to an observation.

Both masks are first normalized (centroid -> canvas centre, area -> NOMINAL_AREA).
The search then covers ONLY: in-plane rotation, residual uniform scale and
translation, over a declared grid.  No shear, no anisotropic scale, no warping:
an alignment cannot deform an impossible hand into a plausible one.

Two stages: coarse (half resolution, every template) then fine (full
resolution, shortlist).  Selection by IoU is *candidate retrieval only*; it is
not the anomaly decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from ._common import NOMINAL_AREA, WORK_CANVAS, HandAnatomyError, digest
from .observations import Observation
from .pose_library import TemplateLibrary
from .template_renderer import RenderedTemplate

ALIGNMENT_VERSION = "hand-alignment/1"


@dataclass(frozen=True)
class AlignmentRecipe:
    rotation_range_deg: float = 45.0
    coarse_rotation_step_deg: float = 7.5
    scale_ratio: float = 1.065  # residual scale grid = ratio ** k
    scale_steps: int = 2  # k in [-2, 2]
    coarse_translation_px: int = 3  # in half-resolution pixels
    fine_rotation_offsets_deg: Tuple[float, ...] = (-3.0, -1.5, 0.0, 1.5, 3.0)
    fine_scale_factors: Tuple[float, ...] = (0.97, 1.0, 1.03)
    fine_translation_px: int = 4
    shortlist_size: int = 12
    version: str = ALIGNMENT_VERSION

    def validate(self) -> None:
        if self.rotation_range_deg <= 0 or self.rotation_range_deg > 90:
            raise HandAnatomyError("rotation range must be in (0, 90]")
        if self.shortlist_size < 1:
            raise HandAnatomyError("shortlist_size must be >= 1")

    def to_dict(self) -> Dict[str, object]:
        return {
            "rotation_range_deg": self.rotation_range_deg,
            "coarse_rotation_step_deg": self.coarse_rotation_step_deg,
            "scale_ratio": self.scale_ratio,
            "scale_steps": self.scale_steps,
            "coarse_translation_px": self.coarse_translation_px,
            "fine_rotation_offsets_deg": list(self.fine_rotation_offsets_deg),
            "fine_scale_factors": list(self.fine_scale_factors),
            "fine_translation_px": self.fine_translation_px,
            "shortlist_size": self.shortlist_size,
            "tie_break": "iou desc, |rot|, |log scale|, |t|1, params, template_id",
            "version": self.version,
        }

    @property
    def recipe_id(self) -> str:
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "AlignmentRecipe":
        kw = {k: v for k, v in data.items() if k != "tie_break"}
        for k in ("fine_rotation_offsets_deg", "fine_scale_factors"):
            kw[k] = tuple(kw[k])  # type: ignore[arg-type]
        return cls(**kw)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Alignment:
    template_id: str
    rotation_deg: float
    residual_scale: float
    tx: float  # work-frame full-resolution pixels
    ty: float
    iou_pre: float  # normalization only (no search)
    iou_post: float

    def params(self) -> Dict[str, float]:
        return {
            "rotation_deg": self.rotation_deg,
            "residual_scale": self.residual_scale,
            "tx": self.tx,
            "ty": self.ty,
        }


def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


class _TemplateGeom:
    """Per-template constants for warping."""

    def __init__(self, template: RenderedTemplate) -> None:
        mask = template.mask
        self.template = template
        self.img = Image.fromarray((mask * 255).astype(np.uint8), "L")
        self.cy, self.cx = _centroid(mask)
        self.norm_scale = math.sqrt(NOMINAL_AREA / float(mask.sum()))


def _warp(
    g: _TemplateGeom, rot_deg: float, residual: float, tx: float, ty: float, size: int, shrink: int
) -> np.ndarray:
    """Forward map  p_w = c + s*R(rot)*(p_t - c_t) + t  rendered on a size x size canvas."""
    s = g.norm_scale * residual / shrink
    c = size / 2.0
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    # inverse: p_t = c_t + R^-1 (p_w - c - t) / s
    data = (
        ca / s, sa / s, g.cx - (ca * (c + tx) + sa * (c + ty)) / s,
        -sa / s, ca / s, g.cy - (-sa * (c + tx) + ca * (c + ty)) / s,
    )
    out = g.img.transform((size, size), Image.AFFINE, data, resample=Image.NEAREST)
    return np.asarray(out) > 127


def warp_template(template: RenderedTemplate, params: Dict[str, float]) -> np.ndarray:
    """Re-create an aligned template mask (full working resolution) from stored params."""
    g = _TemplateGeom(template)
    return _warp(
        g, params["rotation_deg"], params["residual_scale"], params["tx"], params["ty"],
        WORK_CANVAS, 1,
    )


def _windows(obs: np.ndarray, radius: int) -> np.ndarray:
    """(n_shifts, H*W) float32 matrix of shifted observation windows."""
    h, w = obs.shape
    padded = np.pad(obs.astype(np.float32), radius)
    view = np.lib.stride_tricks.sliding_window_view(padded, (h, w))
    n = 2 * radius + 1
    return np.ascontiguousarray(view[:n, :n]).reshape(n * n, h * w)


def _shift_grid(radius: int) -> List[Tuple[int, int]]:
    return [(sx, sy) for sy in range(-radius, radius + 1) for sx in range(-radius, radius + 1)]


def _best(
    stack: np.ndarray, areas: np.ndarray, obs_area: float, wmat: np.ndarray,
    combos: Sequence[Tuple[float, float]], shifts: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[float, float, float, int, int], float]:
    inter = wmat @ stack.reshape(len(stack), -1).T  # (n_shifts, n_combos)
    union = areas[None, :] + obs_area - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)
    best_key = None
    best = None
    for ci, (rot, res) in enumerate(combos):
        for si, (sx, sy) in enumerate(shifts):
            val = round(float(iou[si, ci]), 9)
            key = (-val, abs(rot), abs(math.log(res)), abs(sx) + abs(sy), rot, res, sx, sy)
            if best_key is None or key < best_key:
                best_key = key
                best = (rot, res, val, sx, sy)
    assert best is not None
    return (best[0], best[1], best[2], best[3], best[4]), best[2]


def align(
    observation: Observation,
    library: TemplateLibrary,
    recipe: AlignmentRecipe = AlignmentRecipe(),
) -> List[Alignment]:
    """Return the shortlist (best first) of aligned candidates."""
    recipe.validate()
    obs_full = observation.work_mask
    obs_area = float(obs_full.sum())
    half = np.asarray(
        Image.fromarray((obs_full * 255).astype(np.uint8), "L").reduce(2)
    ) > 127
    half_area = float(half.sum())
    rr = recipe.coarse_translation_px
    w_half = _windows(half, rr)
    shifts_half = _shift_grid(rr)
    rots = list(np.arange(-recipe.rotation_range_deg, recipe.rotation_range_deg + 1e-9, recipe.coarse_rotation_step_deg))
    scales = [recipe.scale_ratio ** k for k in range(-recipe.scale_steps, recipe.scale_steps + 1)]
    combos = [(float(r), float(s)) for r in rots for s in scales]
    coarse: List[Tuple[float, str, float, float, int, int]] = []
    geoms: Dict[str, _TemplateGeom] = {}
    for tpl in library.templates:
        g = _TemplateGeom(tpl)
        geoms[tpl.template_id] = g
        stack = np.stack([_warp(g, r, s, 0, 0, WORK_CANVAS // 2, 2) for r, s in combos]).astype(np.float32)
        areas = stack.reshape(len(combos), -1).sum(axis=1)
        (r, s, iou, sx, sy), _ = _best(stack, areas, half_area, w_half, combos, shifts_half)
        coarse.append((-iou, tpl.template_id, r, s, sx, sy))
    coarse.sort()
    shortlist = coarse[: recipe.shortlist_size]
    fr = recipe.fine_translation_px
    w_full = _windows(obs_full, fr)
    shifts_full = _shift_grid(fr)
    results: List[Alignment] = []
    for _, tid, r1, s1, sx1, sy1 in shortlist:
        g = geoms[tid]
        fine_combos = [
            (float(r1 + dr), float(s1 * f))
            for dr in recipe.fine_rotation_offsets_deg
            for f in recipe.fine_scale_factors
        ]
        stack = np.stack(
            [_warp(g, r, s, 2 * sx1, 2 * sy1, WORK_CANVAS, 1) for r, s in fine_combos]
        ).astype(np.float32)
        areas = stack.reshape(len(fine_combos), -1).sum(axis=1)
        (r, s, iou, sx, sy), _ = _best(stack, areas, obs_area, w_full, fine_combos, shifts_full)
        pre = _warp(g, 0.0, 1.0, 0, 0, WORK_CANVAS, 1)
        inter = float((pre & obs_full).sum())
        union = float(pre.sum()) + obs_area - inter
        results.append(
            Alignment(
                template_id=tid,
                rotation_deg=r,
                residual_scale=s,
                tx=float(2 * sx1 + sx),
                ty=float(2 * sy1 + sy),
                iou_pre=inter / union if union else 0.0,
                iou_post=iou,
            )
        )
    results.sort(key=lambda a: (-a.iou_post, a.template_id))
    return results
