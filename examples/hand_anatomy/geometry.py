"""Hand topology, pose, anatomical constraints and a 3D capsule skeleton.

Layers kept deliberately separate:

1. Strict topology (what connects to what)         -> :class:`HandTopology`
2. Broad anatomical plausibility (joint ranges ...) -> :data:`CONSTRAINTS`
3. Heuristic visual comparison                      -> ``comparison.py``

Convention (right-hand frame, palm toward camera): x right, y toward the
fingertips, z toward the camera.  A right hand has its thumb at +x; a left
hand is the exact mirror image (x -> -x).  Finger flexion rotates segments
toward the camera (+z), which foreshortens them under zero yaw.

This is an orthographic *capsule skeleton*, not a mesh and not MANO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Mapping, Tuple

import numpy as np

from ._common import GEOMETRY_VERSION, HandAnatomyError, digest

FINGERS: Tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")
LANDMARK_NAMES: Tuple[str, ...] = ("wrist",) + tuple(
    "%s_%s" % (f, j)
    for f in FINGERS
    for j in (("cmc", "mcp", "ip", "tip") if f == "thumb" else ("mcp", "pip", "dip", "tip"))
)
COMPONENT_IDS = {
    "background": 0,
    "palm": 1,
    "forearm": 2,
    **{f: 3 + i for i, f in enumerate(FINGERS)},
}


@dataclass(frozen=True)
class HandTopology:
    """Connectivity only: five digits, each a 3-segment chain, on one palm."""

    chains: Mapping[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "thumb": ("cmc", "mcp", "ip", "tip"),
            **{f: ("mcp", "pip", "dip", "tip") for f in FINGERS[1:]},
        }
    )
    attachment: Mapping[str, str] = field(
        default_factory=lambda: {f: "palm" for f in FINGERS}
    )
    root: str = "wrist"

    def validate(self) -> None:
        if tuple(self.chains) != FINGERS:
            raise HandAnatomyError("topology must declare exactly: %s" % (FINGERS,))
        for name, chain in self.chains.items():
            if len(chain) != 4 or len(set(chain)) != 4:
                raise HandAnatomyError("digit %s must be a 3-segment chain" % name)
        if dict(self.attachment) != {f: "palm" for f in FINGERS}:
            raise HandAnatomyError("every digit must attach to the palm")
        if self.root != "wrist":
            raise HandAnatomyError("topology root must be the wrist")

    def to_dict(self) -> Dict[str, object]:
        return {
            "chains": {k: list(v) for k, v in self.chains.items()},
            "attachment": dict(self.attachment),
            "root": self.root,
        }

    @property
    def topology_id(self) -> str:
        return digest({"geometry_version": GEOMETRY_VERSION, "topology": self.to_dict()})


STANDARD_TOPOLOGY = HandTopology()
STANDARD_TOPOLOGY.validate()


# --- anatomical constraints (broad plausibility, degrees) -----------------
CONSTRAINTS = {
    "flex": {"mcp": (-15.0, 95.0), "pip": (0.0, 115.0), "dip": (0.0, 90.0)},
    "abduction": {
        "index": (-10.0, 28.0),
        "middle": (-18.0, 18.0),
        "ring": (-28.0, 10.0),
        "little": (-40.0, 10.0),
    },
    "thumb": {
        "abduction": (0.0, 80.0),
        "opposition": (0.0, 60.0),
        "cmc": (-10.0, 50.0),
        "mcp": (-10.0, 80.0),
        "ip": (-15.0, 90.0),
    },
    "shape": {
        "palm_width": (0.75, 1.15),
        "length_scale": (0.85, 1.15),
        "thickness": (0.8, 1.25),
    },
    "camera": {"yaw": (-90.0, 90.0), "pitch": (-60.0, 60.0)},
    "dip_over_pip_margin": 5.0,
}

# Mean proportions relative to palm length P = 1 (wrist -> middle MCP).
BASE_SEGMENTS = {
    "thumb": (0.45, 0.32, 0.26),
    "index": (0.40, 0.24, 0.18),
    "middle": (0.44, 0.27, 0.19),
    "ring": (0.41, 0.25, 0.18),
    "little": (0.32, 0.19, 0.16),
}
BASE_MCP_X = {"index": 0.36, "middle": 0.12, "ring": -0.12, "little": -0.34}  # * palm width
BASE_MCP_Y = {"index": 0.97, "middle": 1.0, "ring": 0.94, "little": 0.84}
THUMB_CMC = (0.38, 0.16)  # * (palm width, 1)
BASE_RADIUS = {"thumb": 0.15, "index": 0.115, "middle": 0.118, "ring": 0.11, "little": 0.096}
WRIST_HALF_WIDTH = 0.34  # * palm width
FOREARM_LENGTH = 0.55


@dataclass(frozen=True)
class HandShape:
    palm_width: float = 0.90
    length_scale: Tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)
    thickness: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "palm_width": round(self.palm_width, 6),
            "length_scale": [round(v, 6) for v in self.length_scale],
            "thickness": round(self.thickness, 6),
        }


@dataclass(frozen=True)
class HandPose:
    """Joint configuration in the right-hand frame (degrees)."""

    handedness: str  # "left" | "right"
    flex: Mapping[str, Tuple[float, float, float]]  # per digit, proximal -> distal
    abduction: Mapping[str, float]
    thumb_opposition: float = 0.0
    shape: HandShape = field(default_factory=HandShape)
    topology_id: str = STANDARD_TOPOLOGY.topology_id

    def to_dict(self) -> Dict[str, object]:
        return {
            "topology_id": self.topology_id,
            "handedness": self.handedness,
            "flex": {k: [round(float(x), 6) for x in self.flex[k]] for k in FINGERS},
            "abduction": {k: round(float(self.abduction[k]), 6) for k in FINGERS},
            "thumb_opposition": round(float(self.thumb_opposition), 6),
            "shape": self.shape.to_dict(),
        }

    @property
    def geometry_id(self) -> str:
        """Identity of validated geometry parameters + representation version."""
        return digest({"geometry_version": GEOMETRY_VERSION, "pose": self.to_dict()})


@dataclass(frozen=True)
class Camera:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "yaw": round(self.yaw, 6),
            "pitch": round(self.pitch, 6),
            "roll": round(self.roll, 6),
        }


def _in(value: float, bounds: Tuple[float, float], label: str, problems: List[str]) -> None:
    if not math.isfinite(value):
        problems.append("%s is not finite" % label)
    elif not bounds[0] - 1e-9 <= value <= bounds[1] + 1e-9:
        problems.append("%s=%.2f outside [%.1f, %.1f]" % (label, value, bounds[0], bounds[1]))


def pose_violations(pose: HandPose) -> List[str]:
    """Return human-readable anatomical-constraint violations (empty = valid)."""
    problems: List[str] = []
    if pose.handedness not in ("left", "right"):
        problems.append("handedness must be 'left' or 'right'")
    if pose.topology_id != STANDARD_TOPOLOGY.topology_id:
        problems.append("unknown topology id")
    for finger in FINGERS:
        if finger not in pose.flex or len(pose.flex[finger]) != 3:
            problems.append("flex for %s must have three joint angles" % finger)
            return problems
        if finger not in pose.abduction:
            problems.append("abduction missing for %s" % finger)
            return problems
    for finger in FINGERS[1:]:
        for joint, angle in zip(("mcp", "pip", "dip"), pose.flex[finger]):
            _in(angle, CONSTRAINTS["flex"][joint], "%s.%s" % (finger, joint), problems)
        _in(
            pose.abduction[finger],
            CONSTRAINTS["abduction"][finger],
            "%s.abduction" % finger,
            problems,
        )
        pip, dip = pose.flex[finger][1], pose.flex[finger][2]
        if dip > pip + CONSTRAINTS["dip_over_pip_margin"] + 1e-9:
            problems.append("%s.dip exceeds pip by more than the coupling margin" % finger)
    th = CONSTRAINTS["thumb"]
    for joint, angle in zip(("cmc", "mcp", "ip"), pose.flex["thumb"]):
        _in(angle, th[joint], "thumb.%s" % joint, problems)
    _in(pose.abduction["thumb"], th["abduction"], "thumb.abduction", problems)
    _in(pose.thumb_opposition, th["opposition"], "thumb.opposition", problems)
    sh = CONSTRAINTS["shape"]
    _in(pose.shape.palm_width, sh["palm_width"], "palm_width", problems)
    _in(pose.shape.thickness, sh["thickness"], "thickness", problems)
    for finger, scale in zip(FINGERS, pose.shape.length_scale):
        _in(scale, sh["length_scale"], "%s.length_scale" % finger, problems)
    return problems


def validate_pose(pose: HandPose) -> None:
    problems = pose_violations(pose)
    if problems:
        raise HandAnatomyError("invalid hand pose: " + "; ".join(problems))


def validate_camera(camera: Camera) -> None:
    c = CONSTRAINTS["camera"]
    problems: List[str] = []
    _in(camera.yaw, c["yaw"], "camera.yaw", problems)
    _in(camera.pitch, c["pitch"], "camera.pitch", problems)
    if not math.isfinite(camera.roll):
        problems.append("camera.roll is not finite")
    if problems:
        raise HandAnatomyError("invalid camera: " + "; ".join(problems))


def _rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


@dataclass(frozen=True)
class Skeleton:
    """Posed 3D skeleton in camera coordinates (units of palm length)."""

    landmarks: np.ndarray  # (21, 3)
    palm_polygon: np.ndarray  # (5, 3)
    forearm: Tuple[np.ndarray, np.ndarray, float]
    capsules: Dict[str, List[Tuple[np.ndarray, np.ndarray, float, float]]]
    visibility: np.ndarray  # (21,) bool


def build_skeleton(pose: HandPose, camera: Camera = Camera()) -> Skeleton:
    validate_pose(pose)
    validate_camera(camera)
    sh = pose.shape
    W = sh.palm_width
    pts: List[np.ndarray] = [np.zeros(3)]
    capsules: Dict[str, List[Tuple[np.ndarray, np.ndarray, float, float]]] = {}
    for fi, finger in enumerate(FINGERS):
        lengths = [seg * sh.length_scale[fi] for seg in BASE_SEGMENTS[finger]]
        r0 = BASE_RADIUS[finger] * sh.thickness
        if finger == "thumb":
            base = np.array([THUMB_CMC[0] * W, THUMB_CMC[1], 0.0])
            in_plane = math.radians(pose.abduction["thumb"] - pose.thumb_opposition)
        else:
            base = np.array([BASE_MCP_X[finger] * W, BASE_MCP_Y[finger], 0.0])
            in_plane = math.radians(pose.abduction[finger])
        d = np.array([math.sin(in_plane), math.cos(in_plane), 0.0])
        cum = 0.0
        cur = base.copy()
        pts.append(cur.copy())
        segs = []
        for si in range(3):
            cum += math.radians(pose.flex[finger][si])
            direction = math.cos(cum) * d + math.sin(cum) * np.array([0.0, 0.0, 1.0])
            nxt = cur + lengths[si] * direction
            segs.append((cur.copy(), nxt.copy(), r0 * (1.0 - 0.18 * si), r0 * (1.0 - 0.18 * (si + 1))))
            pts.append(nxt.copy())
            cur = nxt
        capsules[finger] = segs
    local = np.stack(pts)
    half = WRIST_HALF_WIDTH * W
    palm = np.array(
        [
            [-half, 0.0, 0.0],
            [half, 0.0, 0.0],
            [BASE_MCP_X["little"] * W - 0.02, BASE_MCP_Y["little"], 0.0],
            [BASE_MCP_X["index"] * W + 0.03, BASE_MCP_Y["index"], 0.0],
            [THUMB_CMC[0] * W + 0.03, THUMB_CMC[1], 0.0],
        ]
    )
    forearm = (np.zeros(3), np.array([0.0, -FOREARM_LENGTH, 0.0]), half * 1.05)
    mirror = np.array([-1.0, 1.0, 1.0]) if pose.handedness == "left" else np.ones(3)
    R = _rot(camera.yaw, camera.pitch, camera.roll)

    def tf(p: np.ndarray) -> np.ndarray:
        return R @ (p * mirror)

    landmarks = np.stack([tf(p) for p in local])
    palm_t = np.stack([tf(p) for p in palm])
    caps_t = {
        f: [(tf(a), tf(b), ra, rb) for a, b, ra, rb in segs] for f, segs in capsules.items()
    }
    fore_t = (tf(forearm[0]), tf(forearm[1]), forearm[2])
    palm_z = float(palm_t[:, 2].mean())
    visibility = landmarks[:, 2] >= palm_z - 0.02
    return Skeleton(landmarks, palm_t, fore_t, caps_t, visibility)


def landmark_segment_ratios(landmarks: np.ndarray) -> Dict[str, float]:
    """Segment lengths relative to wrist -> middle-MCP, for plausibility checks."""
    pts = np.asarray(landmarks, dtype=float)
    if pts.shape != (21, 2) and pts.shape != (21, 3):
        raise HandAnatomyError("landmarks must have shape (21, 2|3)")
    ref = float(np.linalg.norm(pts[9] - pts[0]))
    if not math.isfinite(ref) or ref <= 0:
        raise HandAnatomyError("degenerate landmark set (zero palm length)")
    out = {}
    for fi, finger in enumerate(FINGERS):
        base = 1 + fi * 4
        for si in range(3):
            out["%s_seg%d" % (finger, si)] = (
                float(np.linalg.norm(pts[base + si + 1] - pts[base + si])) / ref
            )
    return out
