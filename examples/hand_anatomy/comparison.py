"""Independent comparison metrics between an observation and aligned templates.

Every metric is expressed as a *compatibility* in [0, 1] where higher means the
observation is more compatible with the template (polarity is uniform and
declared in :data:`METRICS`).  Distances are transformed with a documented
formula.  Metrics are kept separate on purpose; nothing here averages away a
contradiction, and nothing here makes a verdict (see ``assessment.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from ._common import WORK_CANVAS, HandAnatomyError, digest
from .alignment import Alignment, _TemplateGeom, warp_template
from .observations import Observation
from .template_renderer import RenderedTemplate

COMPARISON_VERSION = "hand-comparison/1"

METRICS: Dict[str, Dict[str, str]] = {
    "silhouette_agreement": {
        "definition": "IoU(observed, aligned template); empty-union -> 0",
    },
    "boundary_agreement": {
        "definition": "exp(-symmetric_chamfer_px / chamfer_tau_px) over 1-px boundaries",
    },
    "geometry_agreement": {
        "definition": "1 - min(1, (|d solidity|/sol_tol + |d eccentricity|/ecc_tol)/2); alignment-free shape descriptors",
    },
    "observed_explainability": {
        "definition": "1 - |obs AND NOT dilate(tpl, tol)| / |obs|",
    },
    "protrusion_explainability": {
        "definition": "1 - min(1, largest unexplained connected component area / (ref_fraction*|obs|))",
    },
    "missing_explainability": {
        "definition": "1 - |tpl AND NOT dilate(obs, tol)| / |tpl|  (low may be ordinary occlusion)",
    },
    "structure_agreement": {
        "definition": "1 - min(1, |n_protrusions(obs) - n_protrusions(tpl)| / 2); heuristic, see applicability",
    },
    "landmark_compatibility": {
        "definition": "exp(-RMS landmark error / palm length / 0.15) after template alignment (optional column)",
    },
}
for _m in METRICS.values():
    _m["direction"] = "higher_is_more_compatible"
    _m["range"] = "[0, 1]"

CORE_METRICS: Tuple[str, ...] = (
    "silhouette_agreement",
    "boundary_agreement",
    "geometry_agreement",
    "observed_explainability",
    "protrusion_explainability",
    "missing_explainability",
    "structure_agreement",
)


@dataclass(frozen=True)
class ComparisonRecipe:
    tolerance_px: int = 2
    chamfer_tau_px: float = 3.0
    protrusion_ref_fraction: float = 0.04
    min_component_px: int = 10
    protrusion_radius_factor: float = 1.5
    polar_bins: int = 120
    min_run_bins: int = 2
    solidity_tol: float = 0.15
    eccentricity_tol: float = 0.25
    max_boundary_points: int = 1500
    version: str = COMPARISON_VERSION

    def validate(self) -> None:
        if self.tolerance_px < 0 or self.chamfer_tau_px <= 0:
            raise HandAnatomyError("invalid comparison recipe")
        if not 0 < self.protrusion_ref_fraction < 1:
            raise HandAnatomyError("protrusion_ref_fraction must be in (0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tolerance_px": self.tolerance_px,
            "chamfer_tau_px": self.chamfer_tau_px,
            "protrusion_ref_fraction": self.protrusion_ref_fraction,
            "min_component_px": self.min_component_px,
            "protrusion_radius_factor": self.protrusion_radius_factor,
            "polar_bins": self.polar_bins,
            "min_run_bins": self.min_run_bins,
            "solidity_tol": self.solidity_tol,
            "eccentricity_tol": self.eccentricity_tol,
            "max_boundary_points": self.max_boundary_points,
            "connectivity": 8,
            "metrics": {k: dict(v) for k, v in METRICS.items()},
            "version": self.version,
        }

    @property
    def recipe_id(self) -> str:
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparisonRecipe":
        skip = ("metrics", "connectivity")
        return cls(**{k: v for k, v in data.items() if k not in skip})


# --- morphology / components (numpy + PIL only) ---------------------------
def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    img = Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), "L")
    from PIL import ImageFilter

    return np.asarray(img.filter(ImageFilter.MaxFilter(2 * radius + 1))) > 127


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    from PIL import ImageFilter

    img = Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), "L")
    return np.asarray(img.filter(ImageFilter.MinFilter(2 * radius + 1))) > 127


def boundary_of(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    return m & ~erode(m, 1)


def connected_components(mask: np.ndarray, min_area: int = 1) -> Tuple[np.ndarray, List[int]]:
    """8-connected labelling. Returns (labels starting at 1, areas of kept components)."""
    m = np.asarray(mask, dtype=bool)
    labels = np.zeros(m.shape, dtype=np.int32)
    h, w = m.shape
    pending = set(map(tuple, np.argwhere(m).tolist()))
    areas: List[int] = []
    nxt = 0
    for start in sorted(pending):
        if labels[start] != 0:
            continue
        stack = [start]
        labels[start] = -1
        members = []
        while stack:
            y, x = stack.pop()
            members.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = -1
                        stack.append((ny, nx))
        if len(members) >= min_area:
            nxt += 1
            areas.append(len(members))
            for p in members:
                labels[p] = nxt
        else:
            for p in members:
                labels[p] = 0
    return labels, areas


def _points(mask: np.ndarray, limit: int) -> np.ndarray:
    pts = np.argwhere(mask).astype(np.float32)
    if len(pts) > limit:
        pts = pts[np.linspace(0, len(pts) - 1, limit).astype(int)]
    return pts


def symmetric_chamfer(a: np.ndarray, b: np.ndarray, limit: int = 1500) -> float:
    """Mean of the two directed mean nearest-boundary distances (pixels)."""
    pa, pb = _points(boundary_of(a), limit), _points(boundary_of(b), limit)
    if len(pa) == 0 or len(pb) == 0:
        return float("inf")
    d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2))
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def _convex_hull_area(points: np.ndarray) -> float:
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    x = np.array([p[0] for p in hull])
    y = np.array([p[1] for p in hull])
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def shape_descriptors(mask: np.ndarray) -> Tuple[float, float]:
    """(solidity, eccentricity = sqrt(lambda_min / lambda_max))."""
    pts = np.argwhere(mask).astype(float)
    area = float(len(pts))
    if area < 3:
        return 0.0, 0.0
    hull = _convex_hull_area(_points(boundary_of(mask), 4000).astype(float))
    solidity = min(1.0, area / hull) if hull > 0 else 0.0
    cov = np.cov(pts.T)
    ev = np.sort(np.linalg.eigvalsh(cov))
    ecc = math.sqrt(max(ev[0], 0.0) / ev[1]) if ev[1] > 0 else 0.0
    return solidity, ecc


def palm_center(mask: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """Centre/radius of the largest inscribed circle (grid-sampled, deterministic)."""
    b = _points(boundary_of(mask), 2000)
    cand = np.argwhere(mask)[::3].astype(np.float32)
    if len(b) == 0 or len(cand) == 0:
        return (0.0, 0.0), 0.0
    best_d, best_c = -1.0, cand[0]
    for chunk in np.array_split(cand, max(1, len(cand) // 400)):
        d = np.sqrt(((chunk[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        i = int(np.argmax(d))
        if d[i] > best_d:
            best_d, best_c = float(d[i]), chunk[i]
    return (float(best_c[0]), float(best_c[1])), best_d


def protrusion_runs(mask: np.ndarray, recipe: ComparisonRecipe) -> Dict[str, Any]:
    """Count finger-like angular runs beyond the palm circle (heuristic evidence)."""
    (cy, cx), r0 = palm_center(mask)
    pts = np.argwhere(mask).astype(float)
    if r0 <= 0 or len(pts) == 0:
        return {"count": 0, "palm_radius": r0, "runs": []}
    dy, dx = pts[:, 0] - cy, pts[:, 1] - cx
    r = np.hypot(dx, dy)
    ang = (np.arctan2(dy, dx) + math.pi) / (2 * math.pi)
    bins = np.minimum((ang * recipe.polar_bins).astype(int), recipe.polar_bins - 1)
    extent = np.zeros(recipe.polar_bins)
    np.maximum.at(extent, bins, r)
    over = extent > recipe.protrusion_radius_factor * r0
    runs: List[Tuple[int, int]] = []
    n = recipe.polar_bins
    if over.all():
        return {"count": 1, "palm_radius": r0, "runs": [[0, n]]}
    start = int(np.argmin(over))  # a non-protruding bin to rotate to
    seq = np.roll(over, -start)
    i = 0
    while i < n:
        if seq[i]:
            j = i
            while j < n and seq[j]:
                j += 1
            if j - i >= recipe.min_run_bins:
                runs.append(((i + start) % n, j - i))
            i = j
        else:
            i += 1
    return {"count": len(runs), "palm_radius": r0, "runs": [list(r) for r in runs]}


@dataclass(frozen=True)
class ObservationFeatures:
    """Candidate-independent evidence computed once per observation."""

    area: int
    solidity: float
    eccentricity: float
    component_count: int
    connectivity: float  # largest component share of area
    protrusions: Dict[str, Any]


def observation_features(observation: Observation, recipe: ComparisonRecipe) -> ObservationFeatures:
    m = observation.work_mask
    _, areas = connected_components(m, min_area=recipe.min_component_px)
    total = float(m.sum())
    sol, ecc = shape_descriptors(m)
    return ObservationFeatures(
        area=int(total),
        solidity=sol,
        eccentricity=ecc,
        component_count=len(areas),
        connectivity=(max(areas) / total) if areas else 0.0,
        protrusions=protrusion_runs(m, recipe),
    )


@dataclass(frozen=True)
class CandidateEvidence:
    template_id: str
    template_name: str
    alignment: Alignment
    metrics: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _transform_points(
    template: RenderedTemplate, pts: np.ndarray, al: Alignment
) -> np.ndarray:
    g = _TemplateGeom(template)
    s = g.norm_scale * al.residual_scale
    a = math.radians(al.rotation_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = WORK_CANVAS / 2.0
    rel = np.asarray(pts, dtype=float) - np.array([g.cx, g.cy])  # points are (x, y)
    return (rel @ R.T) * s + np.array([c + al.tx, c + al.ty])


def compare_candidate(
    observation: Observation,
    features: ObservationFeatures,
    template: RenderedTemplate,
    alignment: Alignment,
    recipe: ComparisonRecipe = ComparisonRecipe(),
) -> CandidateEvidence:
    recipe.validate()
    obs_mask = observation.work_mask
    T = warp_template(template, alignment.params())
    inter = int((obs_mask & T).sum())
    union = int((obs_mask | T).sum())
    iou = inter / union if union else 0.0
    chamfer = symmetric_chamfer(obs_mask, T, recipe.max_boundary_points)
    boundary = math.exp(-chamfer / recipe.chamfer_tau_px) if math.isfinite(chamfer) else 0.0
    t_sol, t_ecc = shape_descriptors(T)
    geometry = 1.0 - min(
        1.0,
        (
            abs(features.solidity - t_sol) / recipe.solidity_tol
            + abs(features.eccentricity - t_ecc) / recipe.eccentricity_tol
        )
        / 2.0,
    )
    tol = recipe.tolerance_px
    unexplained = obs_mask & ~dilate(T, tol)
    missing = T & ~dilate(obs_mask, tol)
    o_area, t_area = float(obs_mask.sum()), float(T.sum())
    obs_expl = 1.0 - float(unexplained.sum()) / o_area
    labels, u_areas = connected_components(unexplained, min_area=recipe.min_component_px)
    largest = max(u_areas) if u_areas else 0
    prot_expl = 1.0 - min(1.0, largest / (recipe.protrusion_ref_fraction * o_area))
    miss_expl = 1.0 - float(missing.sum()) / t_area if t_area else 0.0
    t_prot = protrusion_runs(T, recipe)
    n_o, n_t = features.protrusions["count"], t_prot["count"]
    structure = 1.0 - min(1.0, abs(n_o - n_t) / 2.0)
    loc = None
    if largest:
        k = 1 + int(np.argmax(u_areas))
        ys, xs = np.nonzero(labels == k)
        loc = {"centroid_yx": [round(float(ys.mean()), 1), round(float(xs.mean()), 1)],
               "bbox_yxyx": [int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1]}
    metrics = {
        "silhouette_agreement": iou,
        "boundary_agreement": boundary,
        "geometry_agreement": geometry,
        "observed_explainability": obs_expl,
        "protrusion_explainability": prot_expl,
        "missing_explainability": miss_expl,
        "structure_agreement": structure,
    }
    diagnostics: Dict[str, Any] = {
        "chamfer_px": chamfer,
        "unexplained_pixels": int(unexplained.sum()),
        "unexplained_fraction": float(unexplained.sum()) / o_area,
        "unexplained_components": sorted(u_areas, reverse=True)[:6],
        "largest_unexplained": loc,
        "missing_pixels": int(missing.sum()),
        "missing_fraction": float(missing.sum()) / t_area if t_area else 0.0,
        "protrusions_observed": n_o,
        "protrusions_template": n_t,
        "matched_pixels": inter,
    }
    if observation.landmarks is not None:
        metrics["landmark_compatibility"], diagnostics["landmark_rmse_norm"] = landmark_compatibility(
            observation, template, alignment
        )
    for k, v in metrics.items():
        if not math.isfinite(v) or not -1e-12 <= v <= 1 + 1e-12:
            raise HandAnatomyError("metric %s is not a finite value in [0, 1]: %r" % (k, v))
        metrics[k] = float(min(1.0, max(0.0, v)))
    return CandidateEvidence(template.template_id, template.name, alignment, metrics, diagnostics)


def landmark_compatibility(
    observation: Observation, template: RenderedTemplate, alignment: Alignment
) -> Tuple[float, float]:
    assert observation.landmarks is not None
    tpl = _transform_points(template, template.landmarks, alignment)
    obs = observation.landmarks
    palm = float(np.linalg.norm(obs[9] - obs[0]))
    if palm <= 0:
        raise HandAnatomyError("degenerate observed landmarks")
    rmse = float(np.sqrt(((tpl - obs) ** 2).sum(axis=1).mean())) / palm
    return math.exp(-rmse / 0.15), rmse
