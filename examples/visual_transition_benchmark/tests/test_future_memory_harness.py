"""Unit tests for future-memory benchmark harness helpers (labels/perturbations)."""

from __future__ import annotations

import numpy as np

from visual_transition_benchmark.domains.warehouse import model as wh_model
from visual_transition_benchmark.future_memory_benchmark import (
    arcade_label,
    generate_arcade_episodes,
    generate_warehouse_episodes,
    perturb_background_shift,
    perturb_pixel_noise,
    warehouse_label,
)
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.perception import (
    build_grid_field_schema,
    build_transition_evidence_vpm,
    encode_source_array,
)
from zeromodel.perception.representation import SourceImageEncoderSpecDTO
from zeromodel.video.arcade_policy.model import (
    ShooterConfig,
    compile_policy_artifact,
    state_row_id,
)

_SPEC = SourceImageEncoderSpecDTO(color_space="L")


def test_arcade_labels_match_compiled_policy_artifact() -> None:
    """Harness labels reproduce the repo's own optimal policy table."""
    config = ShooterConfig()
    reader = VPMPolicyLookup(
        compile_policy_artifact(config),
        action_metric_ids=("LEFT", "RIGHT", "STAY", "FIRE"),
    )
    targets = (None,) + tuple(range(config.width))
    for tank_x in range(config.width):
        for target_x in targets:
            for cooldown in (0, 1):
                row_id = state_row_id(tank_x, target_x, cooldown)
                assert reader.choose(row_id) == arcade_label(
                    tank_x, target_x, cooldown
                )


def test_warehouse_labels_navigate_to_goal() -> None:
    state = wh_model.WarehouseState(
        robot=(1, 1), crates=(), door_open=True, battery=3
    )
    label = warehouse_label(state, (3, 3))
    assert label in ("MOVE_DOWN", "MOVE_RIGHT")
    at_goal = wh_model.WarehouseState(
        robot=(3, 3), crates=(), door_open=True, battery=3
    )
    assert warehouse_label(at_goal, (3, 3)) == "WAIT"
    boxed = wh_model.WarehouseState(
        robot=(1, 1),
        crates=((1, 2), (2, 1)),
        door_open=False,
        battery=3,
    )
    assert warehouse_label(boxed, (3, 3)) == "WAIT"


def test_background_shift_preserves_transition_deltas() -> None:
    records = generate_arcade_episodes(
        prefix="probe", episode_count=2, seed_offset=0, config=ShooterConfig()
    )
    shifted = perturb_background_shift(records, shift=30)
    schema = build_grid_field_schema(
        encode_source_array(records[0].frame_before, _SPEC),
        tile_width=4,
        tile_height=1,
        channel_mode="joint",
    )
    for clean, perturbed in zip(records, shifted):
        clean_evidence = build_transition_evidence_vpm(
            encode_source_array(clean.frame_before, _SPEC),
            encode_source_array(clean.frame_after, _SPEC),
            schema,
            change_threshold=8,
        )
        shifted_evidence = build_transition_evidence_vpm(
            encode_source_array(perturbed.frame_before, _SPEC),
            encode_source_array(perturbed.frame_after, _SPEC),
            schema,
            change_threshold=8,
        )
        for clean_field, shifted_field in zip(
            clean_evidence.fields, shifted_evidence.fields
        ):
            assert clean_field.mean_signed_change == shifted_field.mean_signed_change
            assert clean_field.mean_absolute_change == shifted_field.mean_absolute_change
            assert clean_field.changed_value_count == shifted_field.changed_value_count


def test_pixel_noise_stays_below_change_threshold() -> None:
    records = generate_warehouse_episodes(
        prefix="probe", episode_count=2, seed_offset=0, goal=(3, 3)
    )
    noisy = perturb_pixel_noise(records, seed=1, amplitude=3)
    schema = build_grid_field_schema(
        encode_source_array(records[0].frame_before, _SPEC),
        tile_width=5,
        tile_height=2,
        channel_mode="joint",
    )
    for clean, perturbed in zip(records, noisy):
        evidence = build_transition_evidence_vpm(
            encode_source_array(clean.frame_before, _SPEC),
            encode_source_array(perturbed.frame_before, _SPEC),
            schema,
            change_threshold=8,
        )
        assert all(
            item.changed_value_count == 0 for item in evidence.fields
        )


def test_changed_wave_produces_different_targets() -> None:
    default = generate_arcade_episodes(
        prefix="a", episode_count=6, seed_offset=0, config=ShooterConfig()
    )
    changed = generate_arcade_episodes(
        prefix="b",
        episode_count=6,
        seed_offset=0,
        config=ShooterConfig(wave=(6, 0, 5, 1)),
    )
    default_sequence = [
        (record.episode_id, record.step_number, record.state_before["target_x"])
        for record in default
    ]
    changed_sequence = [
        (record.episode_id, record.step_number, record.state_before["target_x"])
        for record in changed
    ]
    assert default_sequence != changed_sequence


def test_warehouse_generation_is_deterministic() -> None:
    first = generate_warehouse_episodes(
        prefix="a", episode_count=3, seed_offset=0, goal=(3, 3)
    )
    second = generate_warehouse_episodes(
        prefix="a", episode_count=3, seed_offset=0, goal=(3, 3)
    )
    assert [(r.action, r.label) for r in first] == [
        (r.action, r.label) for r in second
    ]
    assert all(
        np.array_equal(a.frame_before, b.frame_before)
        for a, b in zip(first, second)
    )
