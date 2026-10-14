"""Bounded, deterministic, versioned library of anatomically valid templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ._common import GEOMETRY_VERSION, HandAnatomyError, digest, mask_digest
from .geometry import (
    CONSTRAINTS,
    FINGERS,
    STANDARD_TOPOLOGY,
    Camera,
    HandPose,
    HandShape,
    validate_pose,
)
from .template_renderer import RenderedTemplate, RenderRecipe, render_template

LIBRARY_VERSION = "hand-template-library/1"
MAX_LIBRARY_SIZE = 512

F = Tuple[float, float, float]


def _gesture(
    index: F, middle: F, ring: F, little: F, thumb: F,
    abd: Tuple[float, float, float, float], thumb_abd: float, opposition: float,
) -> Dict[str, object]:
    return {
        "flex": {"thumb": thumb, "index": index, "middle": middle, "ring": ring, "little": little},
        "abduction": {
            "thumb": thumb_abd, "index": abd[0], "middle": abd[1], "ring": abd[2], "little": abd[3],
        },
        "thumb_opposition": opposition,
    }


_Z: F = (0.0, 0.0, 0.0)
_FIST: F = (88.0, 105.0, 70.0)
GESTURES: Dict[str, Dict[str, object]] = {
    "open": _gesture(_Z, _Z, _Z, _Z, (0.0, 5.0, 5.0), (4, 0, -4, -9), 38.0, 0.0),
    "spread": _gesture(_Z, _Z, _Z, _Z, (0.0, 0.0, 0.0), (20, 5, -14, -32), 65.0, 0.0),
    "relaxed": _gesture(
        (14, 22, 10), (18, 26, 12), (22, 30, 14), (26, 34, 16), (10, 15, 12),
        (3, 0, -3, -8), 30.0, 10.0,
    ),
    "partial_curl": _gesture(
        (45, 55, 25), (45, 55, 25), (45, 55, 25), (45, 55, 25), (15, 25, 20),
        (2, 0, -2, -5), 30.0, 20.0,
    ),
    "fist": _gesture(_FIST, _FIST, _FIST, _FIST, (20, 40, 40), (0, 0, 0, 0), 20.0, 40.0),
    "pointing": _gesture(_Z, _FIST, _FIST, _FIST, (15, 30, 30), (2, 0, 0, 0), 25.0, 35.0),
    "pinch": _gesture(
        (35, 40, 20), (20, 25, 10), (25, 30, 15), (30, 35, 15), (25, 30, 25),
        (5, 0, -3, -6), 30.0, 45.0,
    ),
    "grip": _gesture(
        (60, 75, 45), (60, 75, 45), (60, 75, 45), (60, 75, 45), (20, 30, 35),
        (6, 0, -6, -12), 45.0, 30.0,
    ),
}
SHAPES: Dict[str, HandShape] = {
    "standard": HandShape(),
    "wide": HandShape(
        palm_width=1.02, length_scale=(1.05, 0.97, 1.03, 1.0, 0.96), thickness=1.1
    ),
}
YAWS: Tuple[float, ...] = (0.0, 40.0, 75.0)
HANDS: Tuple[str, ...] = ("right", "left")


THUMB_VARIANTS: Dict[str, Tuple[float, float]] = {
    "base": (0.0, 0.0),  # (abduction delta, opposition delta), degrees
    "out": (20.0, 0.0),
    "in": (0.0, 20.0),
}


def make_pose(
    gesture: str, handedness: str, shape: HandShape, thumb_variant: str = "base"
) -> HandPose:
    g = GESTURES[gesture]
    d_abd, d_opp = THUMB_VARIANTS[thumb_variant]
    lo_a, hi_a = CONSTRAINTS["thumb"]["abduction"]
    lo_o, hi_o = CONSTRAINTS["thumb"]["opposition"]
    abduction = dict(g["abduction"])  # type: ignore[arg-type]
    abduction["thumb"] = min(hi_a, max(lo_a, float(abduction["thumb"]) + d_abd))
    opposition = min(hi_o, max(lo_o, float(g["thumb_opposition"]) + d_opp))  # type: ignore[arg-type]
    pose = HandPose(
        handedness=handedness,
        flex={k: tuple(float(x) for x in v) for k, v in g["flex"].items()},  # type: ignore[union-attr]
        abduction=abduction,
        thumb_opposition=opposition,
        shape=shape,
    )
    validate_pose(pose)
    return pose


def pose_from_dict(data: Mapping[str, object]) -> HandPose:
    shape = data["shape"]
    pose = HandPose(
        handedness=str(data["handedness"]),
        flex={k: tuple(float(x) for x in data["flex"][k]) for k in FINGERS},  # type: ignore[index]
        abduction={k: float(data["abduction"][k]) for k in FINGERS},  # type: ignore[index]
        thumb_opposition=float(data["thumb_opposition"]),  # type: ignore[arg-type]
        shape=HandShape(
            palm_width=float(shape["palm_width"]),  # type: ignore[index]
            length_scale=tuple(float(x) for x in shape["length_scale"]),  # type: ignore[index]
            thickness=float(shape["thickness"]),  # type: ignore[index]
        ),
        topology_id=str(data["topology_id"]),
    )
    validate_pose(pose)
    return pose


class TemplateLibrary:
    def __init__(self, templates: Sequence[RenderedTemplate], recipe: RenderRecipe) -> None:
        if not templates:
            raise HandAnatomyError("template library is empty")
        if len(templates) > MAX_LIBRARY_SIZE:
            raise HandAnatomyError("template library exceeds MAX_LIBRARY_SIZE")
        ids = [t.template_id for t in templates]
        if len(set(ids)) != len(ids):
            raise HandAnatomyError("duplicate template ids in library")
        self.templates: Tuple[RenderedTemplate, ...] = tuple(sorted(templates, key=lambda t: t.template_id))
        self.recipe = recipe
        self._by_id = {t.template_id: t for t in self.templates}

    def __len__(self) -> int:
        return len(self.templates)

    def get(self, template_id: str) -> RenderedTemplate:
        try:
            return self._by_id[template_id]
        except KeyError as exc:
            raise HandAnatomyError("unknown template id: %s" % template_id) from exc

    @property
    def library_id(self) -> str:
        return digest(
            {
                "version": LIBRARY_VERSION,
                "geometry_version": GEOMETRY_VERSION,
                "topology_id": STANDARD_TOPOLOGY.topology_id,
                "recipe_id": self.recipe.recipe_id,
                "templates": [[t.template_id, t.mask_id] for t in self.templates],
            }
        )

    def metadata(self) -> Dict[str, object]:
        return {
            "schema_version": LIBRARY_VERSION,
            "library_id": self.library_id,
            "geometry_version": GEOMETRY_VERSION,
            "topology_id": STANDARD_TOPOLOGY.topology_id,
            "render_recipe": self.recipe.to_dict(),
            "render_recipe_id": self.recipe.recipe_id,
            "size": len(self),
            "templates": [
                {
                    "template_id": t.template_id,
                    "geometry_id": t.geometry_id,
                    "render_id": t.render_id,
                    "name": t.name,
                    "handedness": t.pose.handedness,
                    "camera": t.camera.to_dict(),
                    "pose": t.pose.to_dict(),
                    "mask_id": t.mask_id,
                    "area": t.area,
                    "visibility": [bool(v) for v in t.visibility],
                    "landmarks": [[round(float(x), 4) for x in p] for p in t.landmarks],
                }
                for t in self.templates
            ],
        }


def build_library(
    gestures: Sequence[str] | None = None,
    yaws: Sequence[float] = YAWS,
    shapes: Sequence[str] | None = None,
    hands: Sequence[str] = HANDS,
    recipe: RenderRecipe = RenderRecipe(),
    thumb_variants: Sequence[str] = ("base",),
) -> TemplateLibrary:
    names = list(gestures) if gestures is not None else list(GESTURES)
    shape_names = list(shapes) if shapes is not None else list(SHAPES)
    templates: List[RenderedTemplate] = []
    for hand in hands:
        for gesture in names:
            for shape_name in shape_names:
              for variant in thumb_variants:
                pose = make_pose(gesture, hand, SHAPES[shape_name], variant)
                suffix = "" if variant == "base" else "/thumb-" + variant
                for yaw in yaws:
                    templates.append(
                        render_template(
                            pose,
                            Camera(yaw=float(yaw)),
                            recipe,
                            name="%s/%s/%s/yaw%d%s" % (hand, gesture, shape_name, int(yaw), suffix),
                        )
                    )
    return TemplateLibrary(templates, recipe)


def save_library(library: TemplateLibrary, directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "library.json").write_text(
        json.dumps(library.metadata(), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        out / "masks.npz", **{t.template_id.split(":")[1]: t.labels for t in library.templates}
    )
    return out


def load_library(directory: str | Path) -> TemplateLibrary:
    """Load and fully re-verify a saved library (re-renders every template)."""
    root = Path(directory)
    meta = json.loads((root / "library.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != LIBRARY_VERSION:
        raise HandAnatomyError("unsupported library schema")
    recipe = RenderRecipe(**{k: v for k, v in meta["render_recipe"].items()})
    stored = np.load(root / "masks.npz")
    templates = []
    for entry in meta["templates"]:
        pose = pose_from_dict(entry["pose"])
        cam = Camera(**entry["camera"])
        tpl = render_template(pose, cam, recipe, name=entry["name"])
        if tpl.template_id != entry["template_id"]:
            raise HandAnatomyError("template identity mismatch for %s" % entry["name"])
        saved = stored[entry["template_id"].split(":")[1]]
        if mask_digest(saved) != entry["mask_id"] or mask_digest(tpl.labels) != entry["mask_id"]:
            raise HandAnatomyError("template mask mismatch for %s" % entry["name"])
        templates.append(tpl)
    lib = TemplateLibrary(templates, recipe)
    if lib.library_id != meta["library_id"]:
        raise HandAnatomyError("library identity mismatch")
    return lib


def validate_library(library: TemplateLibrary) -> List[str]:
    """Internal-consistency problems (empty list = consistent)."""
    problems: List[str] = []
    for t in library.templates:
        problems += ["%s: %s" % (t.name, p) for p in _template_problems(t)]
    return problems


def _template_problems(t: RenderedTemplate) -> List[str]:
    from .geometry import pose_violations

    out = list(pose_violations(t.pose))
    present = set(np.unique(t.labels).tolist())
    if not {1, 2}.issubset(present):
        out.append("palm/forearm component missing")
    mask = t.mask
    if mask.sum() == 0:
        out.append("empty mask")
    return out
