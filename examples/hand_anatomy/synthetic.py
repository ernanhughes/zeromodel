"""Synthetic benchmark cases with ground-truth labels independent of the inspector.

Group ``independent``: a DIFFERENT rendering path from ``template_renderer``
(heading-integrated tapered strokes + elliptic palm + Gaussian-smoothed
outline, own parameterisation, own corruption routines).  It shares no
geometry code or parameter ranges with the library, but it is still a
synthetic silhouette generator, not real imagery.

Group ``same_renderer``: valid hands rendered by the library renderer with
off-grid parameters, and defects painted onto its labelled masks.  This is the
WEAK test (same renderer family) and is always reported separately.

Labels come from construction.  ``ambiguous`` marks additive defects whose
extra area is tiny, computed from the construction (never from the inspector).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ._common import digest, mask_digest

HAND_CANVAS = 420
ORIGIN = (210.0, 340.0)  # wrist centre in hand-frame canvas (x, y-down)
OUT_CANVAS = 520


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    group: str
    kind: str
    label: str  # "valid" | "malformed"
    ambiguous: bool
    mask: np.ndarray = field(repr=False)
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def mask_id(self) -> str:
        return mask_digest(self.mask.astype(np.uint8))


# ---------------------------------------------------------------- blob renderer
def _finger(
    draw: ImageDraw.ImageDraw, base: Tuple[float, float], heading: float, length: float,
    width: float, bend: float = 0.0, hook: float = 0.0, n: int = 36,
) -> Tuple[float, float]:
    x, y = base
    pts = []
    for i in range(n):
        s = i / (n - 1)
        pts.append((x, y, width / 2.0 * (1.0 - 0.28 * s)))
        theta = math.radians(heading + bend * s + hook * s ** 3)
        x += length / n * math.sin(theta)
        y -= length / n * math.cos(theta)
    for px, py, r in pts:
        draw.ellipse([ORIGIN[0] + px - r, ORIGIN[1] + py - r - 0, ORIGIN[0] + px + r, ORIGIN[1] + py + r], fill=255)
    return x, y


def _to_img(x: float, y: float) -> Tuple[float, float]:
    return ORIGIN[0] + x, ORIGIN[1] + y


def default_hand() -> Dict[str, Any]:
    return {
        "palm_w": 74.0, "palm_h": 82.0, "wrist_w": 46.0, "forearm": 60.0,
        "fingers": {
            "index": dict(bx=0.62, heading=4.0, length=70.0, width=15.0, bend=0.0, hook=0.0),
            "middle": dict(bx=0.20, heading=0.0, length=78.0, width=16.0, bend=0.0, hook=0.0),
            "ring": dict(bx=-0.22, heading=-3.0, length=72.0, width=15.0, bend=0.0, hook=0.0),
            "little": dict(bx=-0.62, heading=-9.0, length=56.0, width=13.0, bend=0.0, hook=0.0),
        },
        "thumb": dict(heading=48.0, length=62.0, width=19.0, bend=-8.0, hook=-10.0),
        "extras": [],
        "gaps": [],
        "erase": [],
    }


def render_blob_hand(spec: Dict[str, Any]) -> np.ndarray:
    """Hand-frame mask (HAND_CANVAS^2 bool) from a blob-hand spec."""
    img = Image.new("L", (HAND_CANVAS, HAND_CANVAS), 0)
    d = ImageDraw.Draw(img)
    pw, ph, ww, fa = spec["palm_w"], spec["palm_h"], spec["wrist_w"], spec["forearm"]
    # forearm: tapered rectangle with rounded end
    d.polygon([_to_img(-ww / 2, 8), _to_img(ww / 2, 8), _to_img(ww / 2 * 1.04, fa), _to_img(-ww / 2 * 1.04, fa)], fill=255)
    d.ellipse([ORIGIN[0] - ww / 2 * 1.04, ORIGIN[1] + fa - ww / 4, ORIGIN[0] + ww / 2 * 1.04, ORIGIN[1] + fa + ww / 4], fill=255)
    # palm: elliptic outline wider at the knuckles
    poly = []
    for k in range(64):
        t = 2 * math.pi * k / 64
        poly.append(_to_img((pw / 2) * math.cos(t) * (1 + 0.12 * math.sin(t)), -ph * (0.5 + 0.5 * math.sin(t)) + 6))
    d.polygon(poly, fill=255)
    top = -ph + 6
    for f in spec["fingers"].values():
        base = (f["bx"] * pw / 2, top + 10 + 8 * abs(f["bx"]) ** 2)
        _finger(d, base, f["heading"], f["length"], f["width"], f["bend"], f["hook"])
    if spec.get("thumb"):
        t = spec["thumb"]
        side = t.get("side", 1.0)
        _finger(d, (side * pw / 2 * 0.85, -ph * 0.28), side * t["heading"], t["length"], t["width"], side * t["bend"], side * t["hook"])
    for e in spec["extras"]:
        _finger(d, (e["bx"] * pw / 2, e.get("by", top + 12)), e["heading"], e["length"], e["width"], e.get("bend", 0.0), e.get("hook", 0.0))
        for seg in e.get("more", []):
            pass
    for lobe in spec.get("lobes", []):
        cx, cy, rx, ry = lobe
        a, b = _to_img(cx - rx, cy - ry), _to_img(cx + rx, cy + ry)
        d.ellipse([a[0], a[1], b[0], b[1]], fill=255)
    for zig in spec.get("zigzags", []):
        x, y = zig["base"]
        for seg_heading, seg_len in zig["segments"]:
            x, y = _finger(d, (x, y), seg_heading, seg_len, zig["width"], 0.0, 0.0, n=18)
    img = img.filter(ImageFilter.GaussianBlur(1.4))
    arr = np.asarray(img) > 127
    out = Image.fromarray((arr * 255).astype(np.uint8), "L")
    d2 = ImageDraw.Draw(out)
    for gap in spec["gaps"]:  # disconnect wrist: erase a band across the wrist
        y0, y1 = gap
        d2.rectangle([0, ORIGIN[1] + y0, HAND_CANVAS, ORIGIN[1] + y1], fill=0)
    for ex in spec["erase"]:  # occluders
        cx, cy, rx, ry = ex
        a, b = _to_img(cx - rx, cy - ry), _to_img(cx + rx, cy + ry)
        d2.ellipse([a[0], a[1], b[0], b[1]], fill=0)
    return np.asarray(out) > 127


def place(hand: np.ndarray, rng: np.random.Generator, rot_deg: float, scale: float,
          margin_shift: bool = True) -> np.ndarray:
    """Rotate/scale about the hand centre and translate into a larger output canvas."""
    img = Image.fromarray((hand * 255).astype(np.uint8), "L")
    ys, xs = np.nonzero(hand)
    c = (float(xs.mean()), float(ys.mean()))
    img = img.rotate(rot_deg, resample=Image.BILINEAR, center=c)
    size = int(round(HAND_CANVAS * scale))
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img) > 127
    out = np.zeros((OUT_CANVAS, OUT_CANVAS), dtype=bool)
    ys, xs = np.nonzero(arr)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    max_y, max_x = OUT_CANVAS - h - 12, OUT_CANVAS - w - 12
    oy = int(rng.integers(12, max(13, max_y))) if margin_shift else (OUT_CANVAS - h) // 2
    ox = int(rng.integers(12, max(13, max_x))) if margin_shift else (OUT_CANVAS - w) // 2
    out[oy:oy + h, ox:ox + w] = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return out


# ------------------------------------------------------------------ valid kinds
def _jitter(rng: np.random.Generator, spec: Dict[str, Any], amount: float = 0.06) -> Dict[str, Any]:
    for f in spec["fingers"].values():
        f["length"] *= 1 + rng.uniform(-amount, amount)
        f["width"] *= 1 + rng.uniform(-amount, amount)
        f["heading"] += rng.uniform(-2, 2)
    return spec


def valid_spec(kind: str, rng: np.random.Generator) -> Dict[str, Any]:
    s = default_hand()
    fing = s["fingers"]
    if kind == "open_palm":
        _jitter(rng, s)
    elif kind == "spread":
        for name, dh in zip(("index", "middle", "ring", "little"), (14, 3, -10, -24)):
            fing[name]["heading"] += dh * rng.uniform(0.8, 1.2)
        s["thumb"]["heading"] += 18
        _jitter(rng, s)
    elif kind == "curled":
        amount = rng.uniform(0.45, 0.65)
        for f in fing.values():
            f["length"] *= amount
            f["bend"] = rng.uniform(15, 30)
            f["hook"] = rng.uniform(30, 55)
        s["thumb"]["length"] *= 0.8
        s["thumb"]["hook"] = -35
    elif kind == "sideways":
        s["palm_w"] = 44.0
        for name, (bx, dh) in zip(("index", "middle", "ring", "little"), ((0.6, 1), (0.2, 0), (-0.2, -1), (-0.6, -2))):
            fing[name]["bx"], fing[name]["heading"] = bx, dh
        s["thumb"]["heading"] = 18
        s["thumb"]["length"] *= 0.8
    elif kind in ("scaled", "translated", "rotated"):
        _jitter(rng, s, 0.05)
    elif kind == "proportions":
        s["palm_w"] *= rng.uniform(0.85, 1.2)
        for f in fing.values():
            f["length"] *= rng.uniform(0.8, 1.2)
            f["width"] *= rng.uniform(0.8, 1.3)
        s["thumb"]["length"] *= rng.uniform(0.85, 1.15)
    elif kind == "partial_occlusion":
        _jitter(rng, s)
        names = list(fing)
        pick = names[int(rng.integers(0, 3))]
        cx = fing[pick]["bx"] * s["palm_w"] / 2 + rng.uniform(-6, 6)
        top = -s["palm_h"] - fing[pick]["length"]
        s["erase"].append((cx + 10, top + 14, 30.0, 24.0))
    elif kind == "close_overlap":
        for name, dh in zip(("index", "middle", "ring", "little"), (-1, 0, 1, 2)):
            fing[name]["heading"] = dh
            fing[name]["width"] *= 1.15
        for name, bx in zip(("index", "middle", "ring", "little"), (0.45, 0.15, -0.15, -0.45)):
            fing[name]["bx"] = bx
        s["thumb"]["heading"] = 20
    else:
        raise ValueError("unknown valid kind %s" % kind)
    return s


VALID_KINDS = ("open_palm", "spread", "curled", "sideways", "scaled", "translated", "rotated",
               "proportions", "partial_occlusion", "close_overlap")
UNUSUAL = {"proportions", "close_overlap", "partial_occlusion", "sideways"}
KIND_TRANSFORM = {
    "scaled": dict(rot=(-10, 10), scale=(0.65, 1.5)),
    "translated": dict(rot=(-8, 8), scale=(0.9, 1.1)),
    "rotated": dict(rot=(-38, 38), scale=(0.85, 1.15)),
}


# ---------------------------------------------------------------- malformed kinds
def malformed_spec(kind: str, rng: np.random.Generator) -> Tuple[Dict[str, Any], Callable[[Dict[str, Any]], Dict[str, Any]], bool]:
    """(defective spec, function yielding the same hand WITHOUT the defect, additive?)"""
    base = valid_spec(str(rng.choice(["open_palm", "spread", "open_palm"])), rng)
    import copy

    clean = copy.deepcopy(base)
    s = copy.deepcopy(base)
    additive = True
    if kind == "six_fingers":
        gap_kind = str(rng.choice(["outer", "between", "adjacent"]))
        if gap_kind == "outer":
            s["extras"].append(dict(bx=-0.95, heading=-32 + rng.uniform(-6, 6), length=60, width=14, bend=-4))
        elif gap_kind == "between":
            s["extras"].append(dict(bx=rng.choice([0.4, -0.02]), heading=rng.uniform(-6, 6), length=66, width=14))
        else:
            s["extras"].append(dict(bx=0.20 + rng.uniform(-0.05, 0.05), heading=rng.uniform(5, 9), length=70, width=14))
    elif kind == "missing_finger":
        additive = False
        name = str(rng.choice(["index", "middle", "ring", "little"]))
        del s["fingers"][name]
    elif kind == "duplicated_thumb":
        side = 1.0 if rng.random() < 0.5 else -1.0
        s["extras"].append(dict(bx=side * 1.0, by=-s["palm_h"] * 0.28, heading=side * 50, length=58, width=19, bend=-6 * side))
    elif kind == "abnormal_attachment":
        which = str(rng.choice(["back_of_palm", "from_finger"]))
        if which == "back_of_palm":
            s["extras"].append(dict(bx=0.0, by=-s["palm_h"] * 0.45, heading=rng.choice([-25, 25]) * 1.0, length=58, width=14))
        else:
            host = s["fingers"]["middle"]
            by = -s["palm_h"] - host["length"] * 0.55
            s["extras"].append(dict(bx=host["bx"], by=by, heading=rng.choice([-55, 55]), length=48, width=12))
    elif kind == "disconnected_wrist":
        additive = False
        s["gaps"].append((22 + float(rng.uniform(0, 8)), 34 + float(rng.uniform(0, 8))))
    elif kind == "implausible_segments":
        name = str(rng.choice(["index", "middle", "ring"]))
        f = s["fingers"].pop(name)
        base = (f["bx"] * s["palm_w"] / 2, -s["palm_h"] + 10)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        s["zigzags"] = [dict(base=base, width=f["width"],
                             segments=[(0, 50), (sign * 75, 52), (-sign * 70, 55), (sign * 10, 40)])]
    elif kind == "large_protrusion":
        side = 1.0 if rng.random() < 0.5 else -1.0
        s["extras"].append(dict(bx=side * 1.0, by=-s["palm_h"] * 0.55, heading=side * rng.uniform(70, 100), length=85, width=15))
    elif kind == "malformed_palm":
        s["lobes"] = [(float(rng.choice([-1, 1])) * s["palm_w"] * 0.62, -s["palm_h"] * float(rng.uniform(0.35, 0.65)),
                       s["palm_w"] * 0.38, s["palm_h"] * 0.3)]
    else:
        raise ValueError("unknown malformed kind %s" % kind)
    return s, (lambda _s, c=clean: c), additive


MALFORMED_KINDS = ("six_fingers", "missing_finger", "duplicated_thumb", "abnormal_attachment",
                   "disconnected_wrist", "implausible_segments", "large_protrusion", "malformed_palm")


# --------------------------------------------------------------- same-renderer set
def same_renderer_case(kind: str, rng: np.random.Generator) -> Tuple[np.ndarray, bool, str]:
    """Library renderer with OFF-GRID parameters; defects painted on its label map."""
    from .geometry import Camera, HandShape
    from .pose_library import make_pose
    from .template_renderer import render_template, RenderRecipe

    gesture = str(rng.choice(["open", "spread", "relaxed", "partial_curl", "pinch"]))
    hand = str(rng.choice(["left", "right"]))
    shape = HandShape(
        palm_width=float(rng.uniform(0.8, 1.1)),
        length_scale=tuple(float(v) for v in rng.uniform(0.9, 1.1, 5)),
        thickness=float(rng.uniform(0.9, 1.2)),
    )
    pose = make_pose(gesture, hand, shape)
    cam = Camera(yaw=float(rng.uniform(0, 70)), pitch=float(rng.uniform(-10, 10)))
    t = render_template(pose, cam, RenderRecipe(canvas=200))
    labels = t.labels.copy()
    additive = True
    if kind == "six_fingers":
        img = Image.fromarray(labels, "L")
        d = ImageDraw.Draw(img)
        ys, xs = np.nonzero(labels == 1)
        cx, cy = float(xs.mean()), float(ys.min())
        side = 1.0 if rng.random() < 0.5 else -1.0
        ang = math.radians(side * rng.uniform(25, 45))
        for r_, L in ((4.5, 44.0),):
            ex, ey = cx + side * 16, cy + 16
            tx, ty = ex + L * math.sin(ang), ey - L * math.cos(ang)
            d.line([ex, ey, tx, ty], fill=9, width=int(2 * r_))
            d.ellipse([tx - r_, ty - r_, tx + r_, ty + r_], fill=9)
        labels = np.asarray(img)
    elif kind == "missing_finger":
        additive = False
        f = int(rng.integers(1, 5))
        labels[labels == 3 + f] = 0
    elif kind == "duplicated_thumb":
        img = Image.fromarray(labels, "L")
        d = ImageDraw.Draw(img)
        ys, xs = np.nonzero(labels == 3)
        if len(xs):
            dx = -int(2 * (xs.mean() - np.nonzero(labels == 1)[1].mean()))
            thumb = (labels == 3)
            shifted = np.roll(thumb, dx, axis=1)
            labels = np.where(shifted, 9, labels).astype(np.uint8)
    return labels > 0, additive, "%s/%s" % (hand, gesture)


# --------------------------------------------------------------------- builders
def _seed(*parts: Any) -> int:
    return int(digest(list(parts))[7:23], 16) % (2 ** 32)


def _additive_ambiguity(spec_def: Dict[str, Any], spec_clean: Dict[str, Any]) -> float:
    a, b = render_blob_hand(spec_def), render_blob_hand(spec_clean)
    return float((a & ~b).sum()) / float(b.sum())


def make_independent(split: str, kind: str, index: int, malformed: bool) -> Case:
    rng = np.random.default_rng(_seed("independent", split, kind, index))
    ambiguous = False
    if malformed:
        spec, clean_fn, additive = malformed_spec(kind, rng)
        hand = render_blob_hand(spec)
        if additive and kind in ("six_fingers", "duplicated_thumb", "abnormal_attachment",
                                 "large_protrusion", "malformed_palm"):
            ambiguous = _additive_ambiguity(spec, clean_fn(spec)) < 0.03
        rot, scale = rng.uniform(-25, 25), rng.uniform(0.85, 1.2)
    else:
        spec = valid_spec(kind, rng)
        hand = render_blob_hand(spec)
        tr = KIND_TRANSFORM.get(kind, dict(rot=(-20, 20), scale=(0.85, 1.2)))
        rot, scale = rng.uniform(*tr["rot"]), rng.uniform(*tr["scale"])
    mask = place(hand, rng, float(rot), float(scale))
    params = {"group": "independent", "kind": kind, "malformed": malformed, "seed_parts": [split, kind, index],
              "rot": round(float(rot), 4), "scale": round(float(scale), 4)}
    return Case(
        case_id=digest(params), split=split, group="independent", kind=kind,
        label="malformed" if malformed else "valid", ambiguous=ambiguous, mask=mask, params=params,
    )


def make_same_renderer(split: str, kind: str, index: int) -> Case:
    rng = np.random.default_rng(_seed("same_renderer", split, kind, index))
    mask, additive, name = same_renderer_case(kind, rng)
    arr = place(np.pad(mask, 110), rng, float(rng.uniform(-25, 25)), float(rng.uniform(0.9, 1.5)) * 1.1)
    params = {"group": "same_renderer", "kind": kind, "seed_parts": [split, kind, index], "pose": name}
    malformed = kind != "valid"
    return Case(
        case_id=digest(params), split=split, group="same_renderer", kind=kind,
        label="malformed" if malformed else "valid", ambiguous=False, mask=arr, params=params,
    )


# kind -> (calibration count, test count)
INDEPENDENT_COUNTS = (3, 5)
SAME_RENDERER_COUNTS = {"valid": (6, 12), "six_fingers": (3, 5), "missing_finger": (3, 5), "duplicated_thumb": (3, 5)}


def build_cases() -> Dict[str, List[Case]]:
    splits: Dict[str, List[Case]] = {"calibration": [], "test": []}
    for si, split in enumerate(("calibration", "test")):
        n = INDEPENDENT_COUNTS[si]
        for kind in VALID_KINDS:
            splits[split] += [make_independent(split, kind, i, False) for i in range(n)]
        for kind in MALFORMED_KINDS:
            splits[split] += [make_independent(split, kind, i, True) for i in range(n)]
        for kind, counts in SAME_RENDERER_COUNTS.items():
            splits[split] += [make_same_renderer(split, kind, i) for i in range(counts[si])]
    return splits


def write_demo_fixtures(directory: str) -> Dict[str, str]:
    """Write one plausible and one six-finger silhouette (independent generator)."""
    from pathlib import Path

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    good = place(render_blob_hand(valid_spec("open_palm", rng)), rng, 12.0, 1.0)
    rng = np.random.default_rng(11)
    spec = default_hand()
    spec["extras"].append(dict(bx=-0.97, heading=-34.0, length=62, width=14, bend=-5))
    bad = place(render_blob_hand(spec), rng, -9.0, 1.0)
    paths = {}
    for name, m in (("plausible_open_hand", good), ("six_finger_hand", bad)):
        p = out / ("%s-mask.png" % name)
        Image.fromarray((m * 255).astype(np.uint8), "L").save(p)
        paths[name] = str(p)
    return paths
