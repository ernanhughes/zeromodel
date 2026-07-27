"""Warehouse decoding and conformance: component-level and value-level.

Architectural note (a genuine cross-domain finding, not an oversight): in the
arcade domain, "tank", "alien", and "cooldown" occupy fixed, mutually
exclusive row-bands, so stage 1's declared P6 annotations
(``PerceptionRegionAnnotationDTO``) and P18B conformance
(``evaluate_transition_conformance``) could operate directly on
presence/absence of change within each disjoint band. In this domain, the
robot and any crate can occupy the *same* set of possible cells -- their
regions overlap completely. P18A (``build_transition_evidence_vpm``, reused
here unchanged) still gives exact per-field before/after evidence, but P18B's
declarative "this annotation must change/stay stable" model assumes disjoint,
pre-known semantic regions, which does not hold here. Presence-level
attribution in this domain therefore requires a value-level step first
(classify what occupies a cell from its pixel intensity), before a
presence/absence judgement is even meaningful. That classification step, and
the conformance judgement built on top of it, are implemented directly here
in Python rather than through P18B's DTOs -- P18B was the one piece of
stage-1 machinery that did **not** transfer as-is.

P4A field partitioning and P18A transition evidence *are* reused unchanged
(via ``compilation.field_schema_compiler``, which is exactly the same code
path arcade's coarse and fine schemas both went through). Performance note:
``TransitionEvidenceVPMDTO.field_evidence()`` is an O(n) linear scan by
design (P18A does not promise indexed lookup); at this domain's 1x1px
resolution (~1,000 fields) that scan dominates runtime if called per-pixel.
``build_transition_evidence`` below walks ``.fields`` once and caches the
result into plain (H, W) numpy arrays -- everything downstream is O(1) array
indexing. This is a benchmark-harness optimization, not a change to P18A
itself.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from visual_transition_benchmark.compilation.evidence_requirements import VisualEvidenceRequirement
from visual_transition_benchmark.compilation.field_schema_compiler import CompiledFieldSchema, compile_field_schema
from visual_transition_benchmark.domains.warehouse import model as wm
from visual_transition_benchmark.domains.warehouse import rendering as wr
from visual_transition_benchmark.domains.protocol import AnalysisMetadata, ComponentAnalysisResult, ValueAnalysisResult

CANVAS_SHAPE = (wr.CANVAS_HEIGHT, wr.CANVAS_WIDTH)

REQUIREMENTS: Tuple[VisualEvidenceRequirement, ...] = (
    VisualEvidenceRequirement(
        component="grid",
        property_name="occupancy",
        evidence_kind="spatial_position",
        region=(0, wm.GRID_SIZE * wr.CELL_PIXELS, 0, wr.CANVAS_WIDTH),
        required_resolution=(1, 1),
        aggregation="mean",
    ),
    VisualEvidenceRequirement(
        component="battery",
        property_name="level",
        evidence_kind="numeric_intensity",
        region=(wm.GRID_SIZE * wr.CELL_PIXELS, wr.CANVAS_HEIGHT, 0, wr.CANVAS_WIDTH),
        required_resolution=(1, 1),
        aggregation="mean",
    ),
)

COMPILED: CompiledFieldSchema = compile_field_schema(CANVAS_SHAPE, REQUIREMENTS)

_CANONICAL_LEVELS: Dict[str, float] = {
    "empty": 0 / 255,
    "goal_ring": wr.GOAL_RING_VALUE / 255,
    "wall": wr.WALL_VALUE / 255,
    "crate": wr.CRATE_BODY_VALUE / 255,
    "robot": wr.ROBOT_VALUE / 255,
    "door": wr.DOOR_VALUE / 255,
}
_LEVEL_TOLERANCE = 4 / 255  # canonical levels are >= 10/255 apart; this must be < half that gap
_CHANGE_THRESHOLD = 8


def classify_level(value: float) -> str:
    best_name, best_distance = None, None
    for name, level in _CANONICAL_LEVELS.items():
        distance = abs(value - level)
        if best_distance is None or distance < best_distance:
            best_name, best_distance = name, distance
    if best_distance <= _LEVEL_TOLERANCE:
        return best_name
    return "out_of_domain"


@dataclass(frozen=True)
class DecodedGrids:
    """Per-pixel P18A evidence, cached into plain numpy arrays once per
    transition. All decode functions below take this, never the raw DTO."""

    before: np.ndarray  # (H, W) float64, normalized [0, 1]
    after: np.ndarray
    changed: np.ndarray  # (H, W) bool


def build_transition_evidence(frame_before: np.ndarray, frame_after: np.ndarray) -> DecodedGrids:
    evidence = COMPILED.build_transition_evidence(frame_before, frame_after, change_threshold=_CHANGE_THRESHOLD)
    height, width = CANVAS_SHAPE
    before = np.zeros((height, width), dtype=np.float64)
    after = np.zeros((height, width), dtype=np.float64)
    changed = np.zeros((height, width), dtype=bool)
    for field, item in zip(COMPILED.field_schema.fields, evidence.fields):
        before[field.y0, field.x0] = item.before_mean
        after[field.y0, field.x0] = item.after_mean
        changed[field.y0, field.x0] = item.changed_value_count > 0
    return DecodedGrids(before=before, after=after, changed=changed)


def _grid_of(grids: DecodedGrids, which: str) -> np.ndarray:
    return grids.before if which == "before_mean" else grids.after


def _mode_value(values: np.ndarray) -> float:
    counts = Counter(np.round(values * 255).astype(int).tolist())
    return counts.most_common(1)[0][0] / 255.0


def _cell_pixel_values(grids: DecodedGrids, row: int, col: int, which: str) -> np.ndarray:
    y0, x0 = wr.cell_origin(row, col)
    grid = _grid_of(grids, which)
    return grid[y0 : y0 + wr.CELL_PIXELS, x0 : x0 + wr.CELL_PIXELS].reshape(-1)


def classify_cell(grids: DecodedGrids, row: int, col: int, which: str) -> str:
    return classify_level(_mode_value(_cell_pixel_values(grids, row, col, which)))


def cell_changed(grids: DecodedGrids, row: int, col: int) -> bool:
    y0, x0 = wr.cell_origin(row, col)
    return bool(grids.changed[y0 : y0 + wr.CELL_PIXELS, x0 : x0 + wr.CELL_PIXELS].any())


def decode_crate_identity(grids: DecodedGrids, row: int, col: int, which: str) -> Optional[int]:
    """Count of the 3 fixed dot sub-positions reading >= the dot threshold.
    0 dots means the cell is not classified as a crate at all; 1/2/3 dots is
    the crate's decoded identity index (0/1/2)."""

    y0, x0 = wr.cell_origin(row, col)
    grid = _grid_of(grids, which)
    dot_threshold = (wr.CRATE_DOT_VALUE - 30) / 255
    lit = 0
    for dy, dx in wr._DOT_OFFSETS:
        patch = grid[y0 + dy : y0 + dy + 2, x0 + dx : x0 + dx + 2]
        if patch.min() >= dot_threshold:
            lit += 1
        else:
            break  # dots are always a prefix (identity 0 -> first dot only, etc.)
    return (lit - 1) if lit > 0 else None


_DOOR_LEVELS: Dict[str, float] = {
    "closed": wr.DOOR_VALUE / 255,
    "open": (wr.DOOR_VALUE / 255) * (wr.CELL_PIXELS // 2) / wr.CELL_PIXELS,
}
_DOOR_TOLERANCE = 20 / 255


def classify_door_state(grids: DecodedGrids, which: str) -> str:
    """The door bar is only 2px wide inside its 6px cell -- mode-based cell
    classification (used for robot/crate/wall) would read the cell's
    majority-empty background and misclassify every door state as "empty".
    This mirrors stage 2's cooldown-dilution bug exactly: a glyph that does
    not fill most of its cell needs its own dedicated sub-region decode, not
    a whole-cell aggregate."""

    door_y0, door_x0 = wr.cell_origin(*wm.DOOR_POSITION)
    grid = _grid_of(grids, which)
    region = grid[door_y0 : door_y0 + wr.CELL_PIXELS, door_x0 + 2 : door_x0 + 4]
    mean_value = float(region.mean())
    for name, level in _DOOR_LEVELS.items():
        if abs(mean_value - level) <= _DOOR_TOLERANCE:
            return name
    return "out_of_domain"


def battery_level(grids: DecodedGrids, which: str) -> int:
    segment_width = wr.CANVAS_WIDTH // wm.MAX_BATTERY
    y0 = wm.GRID_SIZE * wr.CELL_PIXELS
    grid = _grid_of(grids, which)
    level = 0
    for index in range(wm.MAX_BATTERY):
        x0 = index * segment_width
        if grid[y0, x0] >= (wr.BATTERY_ON_VALUE - 30) / 255:
            level += 1
    return level


ACTION_EXPECTED_ROBOT_DELTA: Dict[str, Optional[Tuple[int, int]]] = {
    "MOVE_UP": (-1, 0),
    "MOVE_DOWN": (1, 0),
    "MOVE_LEFT": (0, -1),
    "MOVE_RIGHT": (0, 1),
    "PUSH_UP": (-1, 0),
    "PUSH_DOWN": (1, 0),
    "PUSH_LEFT": (0, -1),
    "PUSH_RIGHT": (0, 1),
    "OPEN_DOOR": (0, 0),
    "WAIT": (0, 0),
}


def _sign(value: int) -> int:
    return -1 if value < 0 else (1 if value > 0 else 0)


_COMPONENT_OF_OCCUPANT: Dict[str, str] = {
    "robot": "robot",
    "crate": "crate",
    "wall": "wall",
}

_EXPECTED_CHANGE_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "MOVE_UP": ("robot",),
    "MOVE_DOWN": ("robot",),
    "MOVE_LEFT": ("robot",),
    "MOVE_RIGHT": ("robot",),
    "PUSH_UP": ("robot", "crate"),
    "PUSH_DOWN": ("robot", "crate"),
    "PUSH_LEFT": ("robot", "crate"),
    "PUSH_RIGHT": ("robot", "crate"),
    "OPEN_DOOR": ("door",),
    "WAIT": (),
}


class WarehouseComponentAnalyzer:
    """Component-level (presence/absence) analysis. See module docstring for
    why this is direct Python, not P18B, for robot/crate/wall/background."""

    def analyze(self, frame_before, frame_after, action, metadata: AnalysisMetadata) -> ComponentAnalysisResult:
        grids = build_transition_evidence(frame_before, frame_after)

        changed_components = set()
        predicted_cells = []
        evidence_scores = {name: 0.0 for name in ("robot", "crate", "door", "battery", "wall", "background")}

        for row in range(wm.GRID_SIZE):
            for col in range(wm.GRID_SIZE):
                if (row, col) == wm.DOOR_POSITION:
                    continue  # decoded separately below; its glyph does not fill the cell
                if not cell_changed(grids, row, col):
                    continue
                before_type = classify_cell(grids, row, col, "before_mean")
                after_type = classify_cell(grids, row, col, "after_mean")
                # A cell a real object vacates or enters is attributed to that
                # object, never to "background" -- otherwise every legitimate
                # robot/crate move would also register a spurious background
                # change at the cell it left.
                real_objects = {t for t in (before_type, after_type) if t in ("robot", "crate", "wall")}
                if real_objects:
                    for occupant in real_objects:
                        name = _COMPONENT_OF_OCCUPANT[occupant]
                        changed_components.add(name)
                        evidence_scores[name] = 1.0
                else:
                    changed_components.add("background")
                    evidence_scores["background"] = 1.0
                predicted_cells.append((row, col))

        door_before = classify_door_state(grids, "before_mean")
        door_after = classify_door_state(grids, "after_mean")
        if door_before != door_after:
            changed_components.add("door")
            evidence_scores["door"] = 1.0
            predicted_cells.append(wm.DOOR_POSITION)

        battery_changed = battery_level(grids, "before_mean") != battery_level(grids, "after_mean")
        if battery_changed:
            changed_components.add("battery")
            evidence_scores["battery"] = 1.0

        expected = _EXPECTED_CHANGE_COMPONENTS.get(action, ())
        stable_required = tuple(name for name in ("robot", "crate", "door", "wall", "background") if name not in expected)

        missing = tuple(sorted(name for name in expected if name not in changed_components))
        unexpected = tuple(sorted(name for name in stable_required if name in changed_components))

        predicted_region_mask = np.zeros((wr.CANVAS_HEIGHT, wr.CANVAS_WIDTH), dtype=bool)
        predicted_fields = []
        for row, col in predicted_cells:
            y0, x0 = wr.cell_origin(row, col)
            predicted_region_mask[y0 : y0 + wr.CELL_PIXELS, x0 : x0 + wr.CELL_PIXELS] = True
            predicted_fields.append(f"cell:{row}:{col}")
        if battery_changed:
            predicted_region_mask[wm.GRID_SIZE * wr.CELL_PIXELS :, :] = True
            predicted_fields.append("battery")

        return ComponentAnalysisResult(
            predicted_region_mask=predicted_region_mask,
            predicted_fields=tuple(sorted(set(predicted_fields))),
            predicted_components=tuple(sorted(changed_components)),
            expected_components=expected,
            unexpected_components=unexpected,
            missing_components=missing,
            evidence_scores=evidence_scores,
            diagnostics={"action": action},
        )


class WarehouseValueAnalyzer:
    """Value-level: decoded robot delta, battery level, crate identities, and
    one adjacency relation -- all from pixels + action, no hidden state."""

    def analyze(self, frame_before, frame_after, action, metadata: AnalysisMetadata) -> ValueAnalysisResult:
        grids = build_transition_evidence(frame_before, frame_after)

        robot_before = _decode_robot_cell(grids, "before_mean")
        robot_after = _decode_robot_cell(grids, "after_mean")
        delta = None
        if robot_before is not None and robot_after is not None:
            delta = (robot_after[0] - robot_before[0], robot_after[1] - robot_before[1])

        expected_delta = ACTION_EXPECTED_ROBOT_DELTA.get(action)
        if delta is None or expected_delta is None:
            direction_ok = None
            magnitude_ok = None
        else:
            direction_ok = tuple(_sign(v) for v in delta) == tuple(_sign(v) for v in expected_delta)
            magnitude_ok = delta == expected_delta

        battery_before = battery_level(grids, "before_mean")
        battery_after = battery_level(grids, "after_mean")
        moved = delta is not None and delta != (0, 0)
        expected_battery_after = (
            max(0, battery_before - 1) if (action != "WAIT" and action != "OPEN_DOOR" and moved) else battery_before
        )
        battery_ok = battery_after == expected_battery_after

        door_after_state = classify_door_state(grids, "after_mean")
        expected_door_state = "open" if action == "OPEN_DOOR" else "closed"
        door_ok = door_after_state == expected_door_state

        relation_violations = []
        new_crate_cells = _crate_cells_appearing(grids)
        for cell in new_crate_cells:
            if not _adjacent_to_any(cell, {c for c in (robot_before, robot_after) if c is not None}):
                relation_violations.append("crate_change_without_robot_adjacency")
                break

        identity_decoded_id = None
        if len(new_crate_cells) == 1:
            identity_decoded_id = decode_crate_identity(grids, *new_crate_cells[0], "after_mean")

        decoded = {
            "direction_decoded_sign": None if delta is None else tuple(_sign(v) for v in delta),
            "magnitude_decoded_delta": delta,
            "value_decoded_level": battery_after,
            "door_decoded_level": door_after_state,
            "relation_decoded_satisfied": "crate_change_without_robot_adjacency" not in relation_violations,
            "identity_decoded_id": identity_decoded_id,
        }

        flags = []
        if direction_ok is False:
            flags.append("robot_direction_violation")
        if magnitude_ok is False:
            flags.append("robot_magnitude_violation")
        if not battery_ok:
            flags.append("battery_value_violation")
        if not door_ok:
            flags.append("door_value_violation")
        flags.extend(f"relation:{name}" for name in relation_violations)

        return ValueAnalysisResult(
            decoded=decoded,
            value_flags=tuple(flags),
            diagnostics={
                "robot_before": robot_before,
                "robot_after": robot_after,
                "battery_before": battery_before,
                "battery_after": battery_after,
            },
        )


def _decode_robot_cell(grids: DecodedGrids, which: str) -> Optional[Tuple[int, int]]:
    best = None
    best_score = 0.0
    for row in range(1, wm.GRID_SIZE - 1):
        for col in range(1, wm.GRID_SIZE - 1):
            score = _mode_value(_cell_pixel_values(grids, row, col, which))
            if classify_level(score) == "robot" and score > best_score:
                best_score = score
                best = (row, col)
    return best


def _crate_cells_appearing(grids: DecodedGrids) -> Tuple[Tuple[int, int], ...]:
    cells = []
    for row in range(1, wm.GRID_SIZE - 1):
        for col in range(1, wm.GRID_SIZE - 1):
            before_type = classify_cell(grids, row, col, "before_mean")
            after_type = classify_cell(grids, row, col, "after_mean")
            if after_type == "crate" and before_type != "crate":
                cells.append((row, col))
    return tuple(cells)


def _adjacent_to_any(cell: Tuple[int, int], others) -> bool:
    for other in others:
        if abs(cell[0] - other[0]) + abs(cell[1] - other[1]) <= 1:
            return True
    return False


def decode_identity_at(frame_before, frame_after, row: int, col: int, which: str) -> Optional[int]:
    grids = build_transition_evidence(frame_before, frame_after)
    return decode_crate_identity(grids, row, col, which)
