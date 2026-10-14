"""Counterfactual pairs: the same hand (pose, proportions, placement), once valid
and once with exactly ONE defect.  Viewpoint, pose and nearly all silhouette
variation are therefore controlled.

Rendering uses the independent blob generator of ZM-HAND-01 (``synthetic.py``),
not the template renderer.  Seeds use new split names ("cal02" / "test02"), so
no case coincides with a ZM-HAND-01 case.

``indistinguishable`` is computed from the two rendered masks (symmetric
difference < 2% of the valid area), never from any inspector output.  Such pairs
are reported separately; detection there is not credited.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from examples.hand_anatomy._common import digest, mask_digest
from examples.hand_anatomy.synthetic import (
    HAND_CANVAS,
    OUT_CANVAS,
    make_independent,
    render_blob_hand,
    valid_spec,
)

INDISTINGUISHABLE_DELTA = 0.02
FINGER_NAMES = ("index", "middle", "ring", "little")

# (category, subtype) -> (calibration pairs, test pairs)
PAIR_PLAN: Dict[Tuple[str, str], Tuple[int, int]] = {
    ("extra_finger", "outer"): (8, 15),
    ("extra_finger", "between"): (10, 30),
    ("extra_finger", "adjacent"): (10, 30),
    ("extra_finger", "hidden"): (4, 10),
    ("duplicated_thumb", "side"): (4, 12),
    ("missing_finger", "any"): (4, 12),
    ("disconnected_wrist", "gap"): (4, 12),
    ("abnormal_attachment", "mixed"): (4, 12),
}
PRIMARY_SUBTYPES = ("outer", "between", "adjacent")
VALID_POOL_KINDS = ("open_palm", "spread", "curled", "sideways", "scaled", "translated", "rotated",
                    "proportions", "partial_occlusion", "close_overlap")
VALID_POOL_PER_KIND = (3, 5)


@dataclass(frozen=True)
class Pair:
    pair_id: str
    split: str
    category: str
    subtype: str
    valid: np.ndarray = field(repr=False)
    defect: np.ndarray = field(repr=False)
    silhouette_delta: float = 0.0
    indistinguishable: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


def _seed(*parts: Any) -> int:
    return int(digest(list(parts))[7:23], 16) % (2 ** 32)


def place_fixed(hand: np.ndarray, rot: float, scale: float, ox: int, oy: int) -> np.ndarray:
    """Identical rigid placement for both members of a pair (rotation about the hand frame centre)."""
    img = Image.fromarray((hand * 255).astype(np.uint8), "L")
    c = (HAND_CANVAS / 2.0, HAND_CANVAS / 2.0)
    img = img.rotate(rot, resample=Image.BILINEAR, center=c)
    size = int(round(HAND_CANVAS * scale))
    arr = np.asarray(img.resize((size, size), Image.BILINEAR)) > 127
    out = np.zeros((OUT_CANVAS, OUT_CANVAS), dtype=bool)
    h = min(size, OUT_CANVAS - oy)
    w = min(size, OUT_CANVAS - ox)
    out[oy:oy + h, ox:ox + w] = arr[:h, :w]
    return out


def _apply_defect(spec: Dict[str, Any], category: str, subtype: str, rng: np.random.Generator) -> None:
    fing = spec["fingers"]
    if category == "extra_finger":
        names = list(fing)
        if subtype == "outer":
            spec["extras"].append(dict(bx=-0.97, heading=-32.0 + rng.uniform(-6, 6), length=60.0, width=14.0, bend=-4.0))
        elif subtype == "between":
            i = int(rng.integers(0, len(names) - 1))
            a, b = fing[names[i]], fing[names[i + 1]]
            spec["extras"].append(dict(
                bx=(a["bx"] + b["bx"]) / 2, heading=(a["heading"] + b["heading"]) / 2 + rng.uniform(-3, 3),
                length=0.85 * (a["length"] + b["length"]) / 2, width=13.0))
        elif subtype == "adjacent":
            host = fing[names[int(rng.integers(0, len(names)))]]
            spec["extras"].append(dict(
                bx=host["bx"] + 0.10, heading=host["heading"] + rng.uniform(5, 9),
                length=host["length"] * 0.95, width=host["width"] * 0.95))
        elif subtype == "hidden":
            host = fing[names[int(rng.integers(0, len(names)))]]
            spec["extras"].append(dict(
                bx=host["bx"], heading=host["heading"] + rng.uniform(-1, 1),
                length=host["length"] * 0.9, width=host["width"] * 0.7))
    elif category == "duplicated_thumb":
        side = 1.0 if rng.random() < 0.5 else -1.0
        spec["extras"].append(dict(bx=side * 1.0, by=-spec["palm_h"] * 0.28, heading=side * 50.0,
                                   length=58.0, width=19.0, bend=-6.0 * side))
    elif category == "missing_finger":
        del fing[str(rng.choice(list(fing)))]
    elif category == "disconnected_wrist":
        spec["gaps"].append((22 + float(rng.uniform(0, 8)), 34 + float(rng.uniform(0, 8))))
    elif category == "abnormal_attachment":
        if rng.random() < 0.5:
            spec["extras"].append(dict(bx=0.0, by=-spec["palm_h"] * 0.45, heading=float(rng.choice([-25, 25])),
                                       length=58.0, width=14.0))
        else:
            host = fing["middle"]
            spec["extras"].append(dict(bx=host["bx"], by=-spec["palm_h"] - host["length"] * 0.55,
                                       heading=float(rng.choice([-55, 55])), length=48.0, width=12.0))
    else:
        raise ValueError("unknown category %s" % category)


def make_pair(split: str, category: str, subtype: str, index: int) -> Pair:
    rng = np.random.default_rng(_seed("pair02", split, category, subtype, index))
    base_kind = str(rng.choice(["open_palm", "spread", "proportions", "close_overlap", "rotated"]))
    spec_valid = valid_spec(base_kind, rng)
    spec_defect = copy.deepcopy(spec_valid)
    _apply_defect(spec_defect, category, subtype, rng)
    rot = float(rng.uniform(-30, 30))
    scale = float(rng.uniform(0.85, 1.15))
    ox, oy = int(rng.integers(0, 14)), int(rng.integers(0, 14))
    valid = place_fixed(render_blob_hand(spec_valid), rot, scale, ox, oy)
    defect = place_fixed(render_blob_hand(spec_defect), rot, scale, ox, oy)
    delta = float((valid ^ defect).sum()) / float(valid.sum())
    params = {"split": split, "category": category, "subtype": subtype, "index": index, "base_kind": base_kind,
              "rot": round(rot, 4), "scale": round(scale, 4), "offset": [ox, oy]}
    return Pair(
        pair_id=digest(params), split=split, category=category, subtype=subtype, valid=valid, defect=defect,
        silhouette_delta=delta, indistinguishable=delta < INDISTINGUISHABLE_DELTA, params=params,
    )


@dataclass(frozen=True)
class Case02:
    case_id: str
    split: str
    role: str  # "valid_twin" | "defect" | "valid_pool"
    label: str
    pair_id: str
    category: str
    subtype: str
    indistinguishable: bool
    mask: np.ndarray = field(repr=False)

    @property
    def mask_id(self) -> str:
        return mask_digest(self.mask.astype(np.uint8))


def build_dataset() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for si, split in enumerate(("cal02", "test02")):
        pairs: List[Pair] = []
        for (category, subtype), counts in PAIR_PLAN.items():
            pairs += [make_pair(split, category, subtype, i) for i in range(counts[si])]
        cases: List[Case02] = []
        for p in pairs:
            cases.append(Case02(digest(["valid", p.pair_id]), split, "valid_twin", "valid", p.pair_id,
                                p.category, p.subtype, p.indistinguishable, p.valid))
            cases.append(Case02(digest(["defect", p.pair_id]), split, "defect", "malformed", p.pair_id,
                                p.category, p.subtype, p.indistinguishable, p.defect))
        for kind in VALID_POOL_KINDS:
            for i in range(VALID_POOL_PER_KIND[si]):
                c = make_independent(split, kind, i, False)
                cases.append(Case02(digest(["pool", c.case_id]), split, "valid_pool", "valid", "", "valid_pool",
                                    kind, False, c.mask))
        out[split] = {"pairs": pairs, "cases": cases}
    return out
