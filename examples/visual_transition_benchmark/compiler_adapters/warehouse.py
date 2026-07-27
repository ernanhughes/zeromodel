"""Warehouse domain adapter: declares candidate regions and evidence
requirements for the warehouse benchmark cases, and builds development/
evaluation samples from ``domains/warehouse/faults.py``'s transition
generator.

Unlike the arcade domain, ``WarehouseTransitionRecord`` exposes
``rendered_state`` -- what was *actually* rendered, even for a faulty
transition -- so numeric/position/categorical cases here can honestly draw
from every category (ordinary and faulty), not just ordinary ones. This
asymmetry with the arcade adapter is a genuine data-availability difference
between the two environments, not an inconsistency in method.

``push_relation`` (crate-follows-robot adjacency) is deliberately excluded:
per ``MANUAL_REPRESENTATION_INVENTORY.md`` item 13, it is a relation over two
already-compiled decoders, out of scope for this per-property compiler, and
``decode_candidate``'s ``relation_over_decoded`` path is an intentional stub.
Running it through the compiler would report a manufactured
``insufficient_observability`` for something never genuinely attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from visual_transition_benchmark.domains.warehouse import faults as wf
from visual_transition_benchmark.domains.warehouse import model as wm
from visual_transition_benchmark.domains.warehouse import rendering as wr

from visual_transition_benchmark.compiler.candidates import RegionGeometry
from visual_transition_benchmark.compiler.contracts import VisualEvidenceRequirement
from visual_transition_benchmark.compiler.evaluate import DevelopmentSample

CANVAS_SHAPE = (wr.CANVAS_HEIGHT, wr.CANVAS_WIDTH)

INTERIOR_REGION = RegionGeometry(
    region_id="interior_grid", canvas_shape=CANVAS_SHAPE,
    y0=0, y1=wm.GRID_SIZE * wr.CELL_PIXELS, x0=0, x1=wr.CANVAS_WIDTH,
    cell_height=wr.CELL_PIXELS, cell_width=wr.CELL_PIXELS,
)
BATTERY_SEGMENT_WIDTH = wr.CANVAS_WIDTH // wm.MAX_BATTERY
BATTERY_REGION = RegionGeometry(
    region_id="battery_strip", canvas_shape=CANVAS_SHAPE,
    y0=wm.GRID_SIZE * wr.CELL_PIXELS, y1=wr.CANVAS_HEIGHT, x0=0, x1=wr.CANVAS_WIDTH,
    cell_height=wr.BATTERY_STRIP_HEIGHT, cell_width=BATTERY_SEGMENT_WIDTH,
)
_door_y0, _door_x0 = wr.cell_origin(*wm.DOOR_POSITION)
DOOR_REGION = RegionGeometry(
    region_id="door_cell", canvas_shape=CANVAS_SHAPE,
    y0=_door_y0, y1=_door_y0 + wr.CELL_PIXELS, x0=_door_x0, x1=_door_x0 + wr.CELL_PIXELS,
    cell_height=wr.CELL_PIXELS, cell_width=wr.CELL_PIXELS,
)
_crate_row, _crate_col = 2, 1  # a fixed interior cell, never the door or goal cell
_crate_y0, _crate_x0 = wr.cell_origin(_crate_row, _crate_col)
CRATE_REGION = RegionGeometry(
    region_id="crate_probe_cell", canvas_shape=CANVAS_SHAPE,
    y0=_crate_y0, y1=_crate_y0 + wr.CELL_PIXELS, x0=_crate_x0, x1=_crate_x0 + wr.CELL_PIXELS,
    cell_height=wr.CELL_PIXELS, cell_width=wr.CELL_PIXELS, sub_patch_size=2,
)

DOOR_CLOSED_LEVEL = wr.DOOR_VALUE / 255
DOOR_OPEN_LEVEL = (wr.DOOR_VALUE / 255) * (wr.CELL_PIXELS // 2) / wr.CELL_PIXELS
BATTERY_LEVELS: Tuple[float, ...] = tuple(
    (k / wm.MAX_BATTERY) * (wr.BATTERY_ON_VALUE / 255) for k in range(wm.MAX_BATTERY + 1)
)


@dataclass(frozen=True)
class CompilerCase:
    name: str
    requirement: VisualEvidenceRequirement
    region: RegionGeometry
    canonical_levels: Tuple[float, ...]
    sub_patch_offsets: Tuple[Tuple[int, int], ...]
    min_decoding_accuracy: float
    build_samples: Callable[[int, int], Tuple[DevelopmentSample, ...]]


def _robot_samples(count: int, seed: int) -> Tuple[DevelopmentSample, ...]:
    samples = []
    for i, category in enumerate(wf.ALL_CATEGORIES):
        for j in range(count):
            rec = wf.build_transition(episode_id=f"robot-{category}", step_number=j, seed=seed + i * 1000 + j, category=category)
            before_robot = tuple(rec.state_before["robot"])
            after_robot = tuple(rec.rendered_state["robot"])
            samples.append(DevelopmentSample(
                sample_id=rec.transition_id,
                frame_before=rec.frame_before,
                frame_after=rec.frame_after,
                true_before=before_robot,
                true_after=after_robot,
                is_unrelated=(before_robot == after_robot),
                fault_present=rec.is_faulty,
            ))
    return tuple(samples)


def _battery_level_of(count: int) -> float:
    return BATTERY_LEVELS[max(0, min(wm.MAX_BATTERY, count))]


def _battery_samples(count: int, seed: int) -> Tuple[DevelopmentSample, ...]:
    samples = []
    for i, category in enumerate(wf.ALL_CATEGORIES):
        for j in range(count):
            rec = wf.build_transition(episode_id=f"battery-{category}", step_number=j, seed=seed + i * 1000 + j, category=category)
            true_before = _battery_level_of(rec.state_before["battery"])
            true_after = _battery_level_of(rec.rendered_state["battery"])
            samples.append(DevelopmentSample(
                sample_id=rec.transition_id,
                frame_before=rec.frame_before,
                frame_after=rec.frame_after,
                true_before=true_before,
                true_after=true_after,
                is_unrelated=(true_before == true_after),
                fault_present=rec.is_faulty,
            ))
    return tuple(samples)


def _door_samples(count: int, seed: int) -> Tuple[DevelopmentSample, ...]:
    samples = []
    for i, category in enumerate(wf.ALL_CATEGORIES):
        for j in range(count):
            rec = wf.build_transition(episode_id=f"door-{category}", step_number=j, seed=seed + i * 1000 + j, category=category)
            true_before = DOOR_OPEN_LEVEL if rec.state_before["door_open"] else DOOR_CLOSED_LEVEL
            true_after = DOOR_OPEN_LEVEL if rec.rendered_state["door_open"] else DOOR_CLOSED_LEVEL
            samples.append(DevelopmentSample(
                sample_id=rec.transition_id,
                frame_before=rec.frame_before,
                frame_after=rec.frame_after,
                true_before=true_before,
                true_after=true_after,
                is_unrelated=(true_before == true_after),
                fault_present=rec.is_faulty,
            ))
    return tuple(samples)


def _crate_identity_samples(count: int, seed: int) -> Tuple[DevelopmentSample, ...]:
    """Direct state construction (not the episode generator): places a crate
    at a fixed interior probe cell at each of the 3 possible identity indices,
    repeated with varying filler crates/robot position for noise."""

    samples = []
    filler_cells = [(1, 1), (1, 2), (1, 3), (3, 1), (3, 2)]
    sample_id = 0
    for identity in range(wm.MAX_CRATES):
        for variant in range(count):
            fillers = [c for c in filler_cells if c != (_crate_row, _crate_col)][:identity]
            crates = tuple(fillers) + ((_crate_row, _crate_col),)
            robot_cell = (1, 1) if (1, 1) not in crates else (3, 3)
            state = wm.WarehouseState(robot=robot_cell, crates=crates, door_open=(variant % 2 == 0), battery=variant % (wm.MAX_BATTERY + 1))
            frame = wr.render_state_frame(state)
            samples.append(DevelopmentSample(
                sample_id=f"crate-identity-{identity}-{variant}",
                frame_before=frame,
                frame_after=frame,
                true_before=identity,
                true_after=identity,
                is_unrelated=False,
            ))
            sample_id += 1
    return tuple(samples)


def build_cases() -> Tuple[CompilerCase, ...]:
    return (
        CompilerCase(
            name="robot_position",
            requirement=VisualEvidenceRequirement(
                domain_name="warehouse", component_type="robot", property_name="position",
                evidence_kind="spatial_position", candidate_region_id="interior_grid", comparison="equal",
            ),
            region=INTERIOR_REGION, canonical_levels=(), sub_patch_offsets=(), min_decoding_accuracy=0.95,
            build_samples=_robot_samples,
        ),
        CompilerCase(
            name="robot_direction",
            requirement=VisualEvidenceRequirement(
                domain_name="warehouse", component_type="robot", property_name="direction",
                evidence_kind="signed_delta", candidate_region_id="interior_grid", comparison="signed_delta",
            ),
            region=INTERIOR_REGION, canonical_levels=(), sub_patch_offsets=(), min_decoding_accuracy=0.95,
            build_samples=_robot_samples,
        ),
        CompilerCase(
            name="robot_movement_magnitude",
            requirement=VisualEvidenceRequirement(
                domain_name="warehouse", component_type="robot", property_name="movement_magnitude",
                evidence_kind="exact_magnitude", candidate_region_id="interior_grid", comparison="exact_delta",
            ),
            region=INTERIOR_REGION, canonical_levels=(), sub_patch_offsets=(), min_decoding_accuracy=0.95,
            build_samples=_robot_samples,
        ),
        CompilerCase(
            name="battery_value",
            requirement=VisualEvidenceRequirement(
                domain_name="warehouse", component_type="battery", property_name="level",
                evidence_kind="numeric_value", candidate_region_id="battery_strip",
                expected_value_domain=BATTERY_LEVELS, required_precision=20 / 255, comparison="equal",
            ),
            region=BATTERY_REGION, canonical_levels=BATTERY_LEVELS, sub_patch_offsets=(), min_decoding_accuracy=0.95,
            build_samples=_battery_samples,
        ),
        CompilerCase(
            name="door_state",
            requirement=VisualEvidenceRequirement(
                domain_name="warehouse", component_type="door", property_name="state",
                evidence_kind="categorical_state", candidate_region_id="door_cell", comparison="equal",
            ),
            region=DOOR_REGION, canonical_levels=(DOOR_CLOSED_LEVEL, DOOR_OPEN_LEVEL), sub_patch_offsets=(), min_decoding_accuracy=0.95,
            build_samples=_door_samples,
        ),
        CompilerCase(
            name="crate_identity",
            requirement=VisualEvidenceRequirement(
                domain_name="warehouse", component_type="crate", property_name="identity",
                evidence_kind="visible_identity", candidate_region_id="crate_probe_cell",
                comparison="identity_equal", permits_identity_marker=True,
            ),
            region=CRATE_REGION, canonical_levels=(), sub_patch_offsets=wr._DOT_OFFSETS, min_decoding_accuracy=0.95,
            build_samples=_crate_identity_samples,
        ),
    )
