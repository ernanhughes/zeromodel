"""Constraint-respecting pose refinement of a library candidate.

Coordinate descent over joint angles, shape parameters, camera yaw and a rigid
transform (rotation, uniform scale, translation).  Every parameter is clipped to
the anatomical ranges of ``geometry.CONSTRAINTS`` and the pose is re-validated,
so refinement can only produce hands the model deems valid: it cannot grow a
sixth digit, and each digit can bend only within its joint limits.

Deterministic: fixed parameter order, fixed step schedule, strict-improvement
acceptance.  No random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from examples.hand_anatomy._common import WORK_CANVAS, HandAnatomyError
from examples.hand_anatomy.alignment import Alignment, _TemplateGeom
from examples.hand_anatomy.geometry import CONSTRAINTS, FINGERS, Camera, HandPose, HandShape, pose_violations
from examples.hand_anatomy.template_renderer import RenderedTemplate, render_template

REFINE_VERSION = "hand-refine/1"
ANGLE_STEPS = (8.0, 4.0, 2.0)
SCALE_STEPS = (0.06, 0.03, 0.015)
PX_STEPS = (3.0, 1.5, 0.75)
ROT_STEPS = (3.0, 1.5, 0.75)
MAX_SWEEPS_PER_LEVEL = 2
MIN_GAIN = 1e-4


@dataclass(frozen=True)
class Fit:
    template: RenderedTemplate
    transform: Dict[str, float]  # rotation_deg, scale, qx, qy  (anchor mapping)
    iou: float
    evaluations: int


def _anchor(template: RenderedTemplate) -> Tuple[float, float]:
    lm = template.landmarks
    return float((lm[0, 0] + lm[9, 0]) / 2.0), float((lm[0, 1] + lm[9, 1]) / 2.0)


def warp_anchored(img: Image.Image, anchor: Tuple[float, float], tf: Dict[str, float]) -> np.ndarray:
    """p_w = q + s R(rot) (p_t - anchor), rendered on the work canvas."""
    a = math.radians(tf["rotation_deg"])
    ca, sa = math.cos(a), math.sin(a)
    s = tf["scale"]
    ax, ay = anchor
    qx, qy = tf["qx"], tf["qy"]
    data = (ca / s, sa / s, ax - (ca * qx + sa * qy) / s,
            -sa / s, ca / s, ay - (-sa * qx + ca * qy) / s)
    out = img.transform((WORK_CANVAS, WORK_CANVAS), Image.AFFINE, data, resample=Image.NEAREST)
    return np.asarray(out) > 127


def initial_transform(template: RenderedTemplate, al: Alignment) -> Dict[str, float]:
    g = _TemplateGeom(template)
    s = g.norm_scale * al.residual_scale
    a = math.radians(al.rotation_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    ax, ay = _anchor(template)
    c = WORK_CANVAS / 2.0
    q = np.array([c + al.tx, c + al.ty]) + s * (R @ np.array([ax - g.cx, ay - g.cy]))
    return {"rotation_deg": al.rotation_deg, "scale": s, "qx": float(q[0]), "qy": float(q[1])}


def _param_names() -> List[str]:
    names: List[str] = []
    for f in FINGERS[1:]:
        names += ["%s.mcp" % f, "%s.pip" % f, "%s.dip" % f, "%s.abd" % f]
    names += ["thumb.cmc", "thumb.mcp", "thumb.ip", "thumb.abd", "thumb.opp"]
    names += ["shape.palm_width", "shape.thickness"] + ["shape.len.%s" % f for f in FINGERS]
    names += ["cam.yaw", "tf.rot", "tf.scale", "tf.qx", "tf.qy"]
    return names


PARAMS = _param_names()


def _vector(pose: HandPose, yaw: float, tf: Dict[str, float]) -> Dict[str, float]:
    v: Dict[str, float] = {}
    for f in FINGERS[1:]:
        for j, k in enumerate(("mcp", "pip", "dip")):
            v["%s.%s" % (f, k)] = float(pose.flex[f][j])
        v["%s.abd" % f] = float(pose.abduction[f])
    for j, k in enumerate(("cmc", "mcp", "ip")):
        v["thumb.%s" % k] = float(pose.flex["thumb"][j])
    v["thumb.abd"] = float(pose.abduction["thumb"])
    v["thumb.opp"] = float(pose.thumb_opposition)
    v["shape.palm_width"] = pose.shape.palm_width
    v["shape.thickness"] = pose.shape.thickness
    for i, f in enumerate(FINGERS):
        v["shape.len.%s" % f] = pose.shape.length_scale[i]
    v.update({"cam.yaw": yaw, "tf.rot": tf["rotation_deg"], "tf.scale": tf["scale"],
              "tf.qx": tf["qx"], "tf.qy": tf["qy"]})
    return v


def _bounds(name: str) -> Tuple[float, float]:
    c = CONSTRAINTS
    if name.startswith("thumb."):
        k = name.split(".")[1]
        return c["thumb"]["abduction" if k == "abd" else "opposition" if k == "opp" else k]
    if name.endswith((".mcp", ".pip", ".dip")):
        return c["flex"][name.split(".")[1]]
    if name.endswith(".abd"):
        return c["abduction"][name.split(".")[0]]
    if name == "shape.palm_width":
        return c["shape"]["palm_width"]
    if name == "shape.thickness":
        return c["shape"]["thickness"]
    if name.startswith("shape.len."):
        return c["shape"]["length_scale"]
    if name == "cam.yaw":
        return (0.0, 75.0)
    return (-1e9, 1e9)


def _step(name: str, level: int) -> float:
    if name.startswith("shape."):
        return SCALE_STEPS[level]
    if name == "tf.scale":
        return SCALE_STEPS[level] * 0.5
    if name == "tf.rot":
        return ROT_STEPS[level]
    if name in ("tf.qx", "tf.qy"):
        return PX_STEPS[level]
    return ANGLE_STEPS[level]


def _build(v: Dict[str, float], handedness: str, topology_id: str) -> Tuple[HandPose, Camera, Dict[str, float]]:
    flex = {}
    for f in FINGERS[1:]:
        mcp, pip, dip = v["%s.mcp" % f], v["%s.pip" % f], v["%s.dip" % f]
        dip = min(dip, pip + CONSTRAINTS["dip_over_pip_margin"])
        flex[f] = (mcp, pip, dip)
    flex["thumb"] = (v["thumb.cmc"], v["thumb.mcp"], v["thumb.ip"])
    abd = {f: v["%s.abd" % f] for f in FINGERS[1:]}
    abd["thumb"] = v["thumb.abd"]
    shape = HandShape(v["shape.palm_width"], tuple(v["shape.len.%s" % f] for f in FINGERS), v["shape.thickness"])
    pose = HandPose(handedness, flex, abd, v["thumb.opp"], shape, topology_id)
    tf = {"rotation_deg": v["tf.rot"], "scale": v["tf.scale"], "qx": v["tf.qx"], "qy": v["tf.qy"]}
    return pose, Camera(yaw=v["cam.yaw"]), tf


def _score(v: Dict[str, float], template: RenderedTemplate, obs: np.ndarray, obs_area: float,
           cache: Dict[Tuple[float, ...], float]) -> Tuple[float, Optional[RenderedTemplate], Dict[str, float]]:
    key = tuple(round(v[n], 6) for n in PARAMS)
    pose, cam, tf = _build(v, template.pose.handedness, template.pose.topology_id)
    if pose_violations(pose):
        return -1.0, None, tf
    try:
        tpl = render_template(pose, cam, template.recipe)
    except HandAnatomyError:
        return -1.0, None, tf
    img = Image.fromarray((tpl.mask * 255).astype(np.uint8), "L")
    m = warp_anchored(img, _anchor(tpl), tf)
    inter = float((m & obs).sum())
    union = float(m.sum()) + obs_area - inter
    iou = inter / union if union else 0.0
    cache[key] = iou
    return iou, tpl, tf


def refine(observation_mask: np.ndarray, template: RenderedTemplate, al: Alignment) -> Fit:
    obs = np.asarray(observation_mask, dtype=bool)
    obs_area = float(obs.sum())
    tf0 = initial_transform(template, al)
    v = _vector(template.pose, template.camera.yaw, tf0)
    cache: Dict[Tuple[float, ...], float] = {}
    best, best_tpl, best_tf = _score(v, template, obs, obs_area, cache)
    if best_tpl is None:
        raise HandAnatomyError("initial candidate is not renderable")
    evals = 1
    for level in range(len(ANGLE_STEPS)):
        for _ in range(MAX_SWEEPS_PER_LEVEL):
            improved = False
            for name in PARAMS:
                lo, hi = _bounds(name)
                for sign in (1.0, -1.0):
                    cand = dict(v)
                    cand[name] = min(hi, max(lo, v[name] + sign * _step(name, level)))
                    if cand[name] == v[name]:
                        continue
                    sc, tpl, tf = _score(cand, template, obs, obs_area, cache)
                    evals += 1
                    if tpl is not None and sc > best + MIN_GAIN:
                        v, best, best_tpl, best_tf, improved = cand, sc, tpl, tf, True
                        break
            if not improved:
                break
    return Fit(best_tpl, best_tf, best, evals)
