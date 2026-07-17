from __future__ import annotations

import importlib.util
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any, Callable, Optional

import pytest

from zeromodel import (
    LayoutRecipe,
    ScoreTable,
    VPMPolicyLookup,
    build_vpm,
    from_bundle,
    to_bundle,
)


def _load_demo():
    path = Path(__file__).resolve().parents[1] / "examples" / "arcade_shooter_policy.py"
    spec = importlib.util.spec_from_file_location("arcade_shooter_policy_exhaustive", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expected_action(actions: tuple[str, ...], values: tuple[float, ...]) -> str:
    """Match VPMPolicyLookup(value_source='raw', tie_break='metric_order')."""
    winner_index = max(
        range(len(actions)),
        key=lambda index: (values[index], -index),
    )
    return actions[winner_index]


def _compile_with_value_function(
    demo,
    value_function: Callable[[int, Optional[int], int], tuple[float, ...]],
):
    """Compile with the same canonical metadata, recipe, and provenance as the demo."""
    config = demo.ShooterConfig()
    targets: tuple[Optional[int], ...] = (None,) + tuple(range(config.width))
    row_ids: list[str] = []
    values: list[tuple[float, ...]] = []

    for tank_x in range(config.width):
        for target_x in targets:
            for cooldown in (0, 1):
                row_ids.append(demo.state_row_id(tank_x, target_x, cooldown))
                values.append(value_function(tank_x, target_x, cooldown))

    table = ScoreTable(
        values=values,
        row_ids=row_ids,
        metric_ids=demo.ACTIONS,
        metadata={
            "kind": "arcade_shooter_policy",
            "world": "tiny_arcade_shooter",
            "addressing": "tank_x,target_x,cooldown",
            "slogan": "signs_not_directions",
        },
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "arcade-shooter-policy-source-order",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(
        table,
        recipe,
        provenance={
            "kind": "compiled_policy",
            "consumer": "VPMPolicyLookup",
            "compile_time_intelligence": "hand_scored_closed_world_policy",
        },
    )


def _run_episode_with_artifact(demo, artifact, config) -> dict[str, Any]:
    reader = VPMPolicyLookup(
        artifact,
        action_metric_ids=demo.ACTIONS,
        value_source="raw",
        tie_break="metric_order",
    )
    game = demo.TinyArcadeShooter(config)
    trace: list[dict[str, Any]] = []

    while not game.done:
        before = game.snapshot()
        decision = reader.read(game.row_id())
        game.step(decision.action)
        after = game.snapshot()
        trace.append(
            {
                "before": before,
                "decision": decision.to_dict(),
                "after": after,
            }
        )

    return {
        "artifact_id": artifact.artifact_id,
        "cleared": game.cleared,
        "score": game.score,
        "steps": game.steps,
        "trace": trace,
    }


def _run_semantic_trace_with_source_policy(demo, config) -> dict[str, Any]:
    game = demo.TinyArcadeShooter(config)
    trace: list[dict[str, Any]] = []

    while not game.done:
        before = game.snapshot()
        values = demo._action_values(game.tank_x, game.target_x, game.cooldown)
        action = _expected_action(demo.ACTIONS, values)
        game.step(action)
        after = game.snapshot()
        trace.append(
            {
                "before": before,
                "candidates": dict(zip(demo.ACTIONS, values)),
                "action": action,
                "after": after,
            }
        )

    return {
        "cleared": game.cleared,
        "score": game.score,
        "steps": game.steps,
        "trace": trace,
    }


def _run_semantic_trace_with_artifact(demo, artifact, config) -> dict[str, Any]:
    reader = VPMPolicyLookup(
        artifact,
        action_metric_ids=demo.ACTIONS,
        value_source="raw",
        tie_break="metric_order",
    )
    game = demo.TinyArcadeShooter(config)
    trace: list[dict[str, Any]] = []

    while not game.done:
        before = game.snapshot()
        decision = reader.read(game.row_id())
        game.step(decision.action)
        after = game.snapshot()
        trace.append(
            {
                "before": before,
                "candidates": dict(decision.candidates),
                "action": decision.action,
                "after": after,
            }
        )

    return {
        "cleared": game.cleared,
        "score": game.score,
        "steps": game.steps,
        "trace": trace,
    }


def test_exhaustive_policy_fidelity() -> None:
    demo = _load_demo()
    artifact = demo.compile_policy_artifact()
    reader = VPMPolicyLookup(
        artifact,
        action_metric_ids=demo.ACTIONS,
        value_source="raw",
        tie_break="metric_order",
    )

    checked_states = 0
    checked_values = 0
    matched_actions = 0
    targets = (None, *range(7))

    for tank_x in range(7):
        for target_x in targets:
            for cooldown in (0, 1):
                expected_values = demo._action_values(tank_x, target_x, cooldown)
                expected_action = _expected_action(demo.ACTIONS, expected_values)
                row_id = demo.state_row_id(tank_x, target_x, cooldown)
                decision = reader.read(row_id)
                actual_values = tuple(decision.candidates[action] for action in demo.ACTIONS)

                assert actual_values == pytest.approx(expected_values), row_id
                assert decision.action == expected_action, row_id
                assert decision.row_id == row_id
                assert decision.artifact_id == artifact.artifact_id

                checked_states += 1
                checked_values += len(demo.ACTIONS)
                matched_actions += 1

    assert checked_states == 112
    assert checked_values == 448
    assert matched_actions == 112

    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "declared_states": 112,
                "state_rows_checked": checked_states,
                "action_values_checked": checked_values,
                "value_matches": checked_values,
                "selected_actions_checked": matched_actions,
                "action_matches": matched_actions,
                "value_fidelity_percent": 100.0,
                "decision_fidelity_percent": 100.0,
            },
            indent=2,
            sort_keys=True,
        )
    )


def test_exhaustive_wave_coverage_and_source_equivalence() -> None:
    demo = _load_demo()
    artifact = demo.compile_policy_artifact()

    total_waves = 0
    cleared_waves = 0
    shortest_episode: int | None = None
    longest_episode = 0
    compared_steps = 0

    for wave in product(range(7), repeat=4):
        config = demo.ShooterConfig(width=7, wave=wave, max_steps=32)
        direct = _run_semantic_trace_with_source_policy(demo, config)
        compiled = _run_semantic_trace_with_artifact(demo, artifact, config)

        assert compiled == direct, wave
        assert compiled["cleared"] is True, wave
        assert compiled["score"] == len(wave), wave

        total_waves += 1
        cleared_waves += int(compiled["cleared"])
        compared_steps += len(compiled["trace"])
        shortest_episode = (
            compiled["steps"]
            if shortest_episode is None
            else min(shortest_episode, compiled["steps"])
        )
        longest_episode = max(longest_episode, compiled["steps"])

    assert total_waves == 7**4
    assert cleared_waves == total_waves
    assert shortest_episode == 7
    assert longest_episode <= 32

    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "artifact_count": 1,
                "possible_waves": 7**4,
                "waves_evaluated": total_waves,
                "waves_cleared": cleared_waves,
                "failed_waves": total_waves - cleared_waves,
                "source_artifact_steps_compared": compared_steps,
                "shortest_episode_steps": shortest_episode,
                "longest_episode_steps": longest_episode,
                "maximum_allowed_steps": 32,
                "scenario_coverage_percent": 100.0,
                "source_artifact_trace_equivalence_percent": 100.0,
            },
            indent=2,
            sort_keys=True,
        )
    )


def test_identity_mutation_localization_and_behaviour() -> None:
    demo = _load_demo()
    original = demo.compile_policy_artifact()

    # Test that the helper reproduces the exact canonical original fixture.
    rebuilt_original = _compile_with_value_function(demo, demo._action_values)
    assert rebuilt_original.artifact_id == original.artifact_id

    behavioural_state = demo.state_row_id(3, 3, 0)

    def behavioural_values(tank_x: int, target_x: Optional[int], cooldown: int) -> tuple[float, ...]:
        values = list(demo._action_values(tank_x, target_x, cooldown))
        if demo.state_row_id(tank_x, target_x, cooldown) == behavioural_state:
            values[demo.ACTIONS.index("STAY")] = 1.0
        return tuple(values)

    behavioural = _compile_with_value_function(demo, behavioural_values)
    assert behavioural.artifact_id != original.artifact_id

    before = VPMPolicyLookup(original, action_metric_ids=demo.ACTIONS).read(behavioural_state)
    after = VPMPolicyLookup(behavioural, action_metric_ids=demo.ACTIONS).read(behavioural_state)
    assert before.action == "FIRE"
    assert after.action == "STAY"

    changed_cells: list[tuple[int, int, float, float]] = []
    for row_index in range(original.source.values.shape[0]):
        for metric_index in range(original.source.values.shape[1]):
            before_value = float(original.source.values[row_index, metric_index])
            after_value = float(behavioural.source.values[row_index, metric_index])
            if before_value != after_value:
                changed_cells.append((row_index, metric_index, before_value, after_value))

    assert changed_cells == [(56, demo.ACTIONS.index("STAY"), 0.0, 1.0)]

    non_behavioural_state = demo.state_row_id(3, 0, 0)

    def non_behavioural_values(
        tank_x: int,
        target_x: Optional[int],
        cooldown: int,
    ) -> tuple[float, ...]:
        values = list(demo._action_values(tank_x, target_x, cooldown))
        if demo.state_row_id(tank_x, target_x, cooldown) == non_behavioural_state:
            values[demo.ACTIONS.index("STAY")] = 0.2
        return tuple(values)

    non_behavioural = _compile_with_value_function(demo, non_behavioural_values)
    assert non_behavioural.artifact_id != original.artifact_id

    non_behavioural_before = VPMPolicyLookup(original, action_metric_ids=demo.ACTIONS).read(
        non_behavioural_state
    )
    non_behavioural_after = VPMPolicyLookup(
        non_behavioural,
        action_metric_ids=demo.ACTIONS,
    ).read(non_behavioural_state)
    assert non_behavioural_before.action == non_behavioural_after.action == "LEFT"

    print(
        json.dumps(
            {
                "original_artifact_id": original.artifact_id,
                "behavioural_mutation_artifact_id": behavioural.artifact_id,
                "non_behavioural_mutation_artifact_id": non_behavioural.artifact_id,
                "behavioural_changed_cells": len(changed_cells),
                "behavioural_changed_state": behavioural_state,
                "behavioural_changed_metric": "STAY",
                "behavioural_before_action": before.action,
                "behavioural_after_action": after.action,
                "non_behavioural_changed_state": non_behavioural_state,
                "non_behavioural_before_action": non_behavioural_before.action,
                "non_behavioural_after_action": non_behavioural_after.action,
            },
            indent=2,
            sort_keys=True,
        )
    )


def test_bundle_roundtrip_replays_identical_episode(tmp_path: Path) -> None:
    demo = _load_demo()
    artifact = demo.compile_policy_artifact()
    bundle_path = to_bundle(artifact, tmp_path / "shooter-policy.vpm")
    loaded = from_bundle(bundle_path)

    original_run = _run_episode_with_artifact(demo, artifact, demo.ShooterConfig())
    loaded_run = _run_episode_with_artifact(demo, loaded, demo.ShooterConfig())

    assert loaded.artifact_id == artifact.artifact_id
    assert original_run == loaded_run
    assert original_run["cleared"] is True
    assert original_run["score"] == 4
    assert original_run["steps"] == 22

    print(
        json.dumps(
            {
                "original_artifact_id": artifact.artifact_id,
                "loaded_artifact_id": loaded.artifact_id,
                "artifact_identity_preserved": True,
                "wave_cleared_before_save": original_run["cleared"],
                "wave_cleared_after_load": loaded_run["cleared"],
                "decision_traces_compared": len(original_run["trace"]),
                "decision_traces_matched": len(original_run["trace"]),
                "environment_states_matched": len(original_run["trace"]),
                "complete_reexecution_match": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


def test_every_runtime_decision_has_complete_resolvable_trace() -> None:
    demo = _load_demo()
    artifact = demo.compile_policy_artifact()
    result = _run_episode_with_artifact(demo, artifact, demo.ShooterConfig())

    assert result["trace"]
    complete = 0

    for step_number, step in enumerate(result["trace"]):
        decision = step["decision"]

        assert decision["artifact_id"] == artifact.artifact_id
        assert decision["row_id"]
        assert decision["action"] in demo.ACTIONS
        assert decision["metric_id"] == decision["action"]
        assert isinstance(decision["value"], float)
        assert set(decision["candidates"]) == set(demo.ACTIONS)
        assert decision["value"] == decision["candidates"][decision["action"]], step_number
        assert decision["source_row_index"] >= 0
        assert decision["source_metric_index"] >= 0
        assert decision["view_row"] >= 0
        assert decision["view_column"] >= 0

        cell = artifact.cell(decision["view_row"], decision["view_column"])
        assert cell.source_row_index == decision["source_row_index"]
        assert cell.source_metric_index == decision["source_metric_index"]
        assert cell.row_id == decision["row_id"]
        assert cell.metric_id == decision["action"]
        assert cell.raw_value == decision["value"]

        # Confirm the complete step record survives ordinary JSON serialization.
        assert json.loads(json.dumps(step, allow_nan=False)) == step
        complete += 1

    assert complete == len(result["trace"])
    assert complete == 22

    print(
        json.dumps(
            {
                "artifact_id": artifact.artifact_id,
                "decisions_executed": len(result["trace"]),
                "decisions_with_complete_trace": complete,
                "resolved_cell_matches": complete,
                "json_roundtrip_matches": complete,
                "trace_completeness_percent": 100.0,
            },
            indent=2,
            sort_keys=True,
        )
    )
