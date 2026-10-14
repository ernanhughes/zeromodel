"""Residual analysis: what remains unexplained after the best valid fit, and
whether it looks like an anatomical digit attached to the hand.

Finger-likeness of a residual component (all factors in [0, 1], constants in
:class:`ResidualRecipe`, frozen in PREREGISTRATION.md before the test split):

    clip(length / L_ref) * clip(elongation / E_ref) * clip(width / W_min) * attached

``attached`` = the component touches the aligned template body (dilated by
``attach_dilation``).  An unattached sliver (e.g. segmentation noise) scores 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from examples.hand_anatomy._common import digest
from examples.hand_anatomy.alignment import _TemplateGeom, _warp
from examples.hand_anatomy._common import WORK_CANVAS
from examples.hand_anatomy.comparison import connected_components, dilate
from examples.hand_anatomy.geometry import COMPONENT_IDS
from examples.hand_anatomy.template_renderer import RenderedTemplate

RESIDUAL_VERSION = "hand-residual/1"
_LABEL_NAMES = {v: k for k, v in COMPONENT_IDS.items()}


@dataclass(frozen=True)
class ResidualRecipe:
    tolerance_px: int = 2
    min_component_px: int = 10
    length_ref: float = 28.0
    elongation_ref: float = 2.5
    width_min: float = 6.0
    attach_dilation: int = 3
    digit_min_px: int = 15
    digit_covered_fraction: float = 0.6
    version: str = RESIDUAL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tolerance_px": self.tolerance_px, "min_component_px": self.min_component_px,
            "length_ref": self.length_ref, "elongation_ref": self.elongation_ref,
            "width_min": self.width_min, "attach_dilation": self.attach_dilation,
            "digit_min_px": self.digit_min_px, "digit_covered_fraction": self.digit_covered_fraction,
            "version": self.version,
        }

    @property
    def recipe_id(self) -> str:
        return digest(self.to_dict())


def warp_label_masks(template: RenderedTemplate, params: Dict[str, float]) -> Dict[int, np.ndarray]:
    """Aligned per-component masks (palm, forearm, five digits) at work resolution."""
    g = _TemplateGeom(template)
    out: Dict[int, np.ndarray] = {}
    for cid in range(1, 8):
        comp = template.labels == cid
        if not comp.any():
            continue
        g.img = Image.fromarray((comp * 255).astype(np.uint8), "L")
        out[cid] = _warp(g, params["rotation_deg"], params["residual_scale"], params["tx"], params["ty"],
                         WORK_CANVAS, 1)
    return out


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def analyze_residual(
    obs_mask: np.ndarray, template_mask: np.ndarray, label_masks: Dict[int, np.ndarray],
    recipe: ResidualRecipe = ResidualRecipe(),
) -> Dict[str, Any]:
    tol = recipe.tolerance_px
    residual = obs_mask & ~dilate(template_mask, tol)
    labels, areas = connected_components(residual, min_area=recipe.min_component_px)
    near_template = dilate(template_mask, recipe.attach_dilation)
    comps: List[Dict[str, Any]] = []
    for k, area in enumerate(areas, start=1):
        comp = labels == k
        ys, xs = np.nonzero(comp)
        pts = np.stack([xs, ys], axis=1).astype(float)
        centered = pts - pts.mean(axis=0)
        if len(pts) > 2:
            evals, evecs = np.linalg.eigh(np.cov(centered.T))
            axis = evecs[:, int(np.argmax(evals))]
        else:
            axis = np.array([1.0, 0.0])
        proj = centered @ axis
        length = float(proj.max() - proj.min() + 1.0)
        width = float(area) / length
        elong = length / max(width, 1.0)
        attached = bool((dilate(comp, recipe.attach_dilation) & near_template).any())
        touch = dilate(comp, recipe.attach_dilation)
        attach_label: Optional[str] = None
        best = 0
        for cid, lm in label_masks.items():
            ov = int((touch & lm).sum())
            if ov > best:
                best, attach_label = ov, _LABEL_NAMES[cid]
        likeness = (
            _clip01(length / recipe.length_ref) * _clip01(elong / recipe.elongation_ref)
            * _clip01(width / recipe.width_min) * (1.0 if attached else 0.0)
        )
        comps.append({
            "area": int(area), "length": round(length, 2), "width": round(width, 2),
            "elongation": round(elong, 2), "attached": attached, "attached_to": attach_label,
            "finger_likeness": round(likeness, 6),
            "centroid_xy": [round(float(pts[:, 0].mean()), 1), round(float(pts[:, 1].mean()), 1)],
            "bbox_yxyx": [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1],
        })
    comps.sort(key=lambda c: (-c["finger_likeness"], -c["area"]))
    present = explained = 0
    obs_near = dilate(obs_mask, tol)
    for cid in range(3, 8):
        lm = label_masks.get(cid)
        if lm is None or int(lm.sum()) < recipe.digit_min_px:
            continue
        present += 1
        if float((lm & obs_near).sum()) / float(lm.sum()) >= recipe.digit_covered_fraction:
            explained += 1
    return {
        "components": comps,
        "max_finger_likeness": comps[0]["finger_likeness"] if comps else 0.0,
        "digit_coverage": (explained / present) if present else 0.0,
        "digits_present": present,
        "digits_explained": explained,
        "unexplained_pixels": int(residual.sum()),
        "residual_mask": residual,
    }
