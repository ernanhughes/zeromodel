from __future__ import annotations

import dataclasses

import pytest

from zeromodel.navigation import compile_hierarchy, replay_traversal, traverse
from zeromodel.navigation.dto import TraversalRequestDTO, build_traversal_receipt
from zeromodel.navigation.rules import DeclaredPriorityRule, FixedKeySelectorRule

from conftest import make_source_artifacts


def test_traversal_reaches_expected_leaf(compiler_spec, artifact_store):
    artifacts = make_source_artifacts(3)
    small_spec = dataclasses.replace(compiler_spec, max_children_per_tile=3)
    manifest = compile_hierarchy(
        spec=small_spec, source_artifacts=artifacts, store=artifact_store
    )

    rule = FixedKeySelectorRule(routing_key="target_order_key")
    request = TraversalRequestDTO(
        request_id="req-1", attributes=(("target_order_key", "00000001"),)
    )
    result = traverse(
        manifest=manifest, store=artifact_store, rule=rule, request=request
    )

    assert result.success
    assert result.final_leaf_id is not None
    assert result.final_artifact_kind == artifacts[1].artifact_kind
    assert result.final_artifact_id == artifacts[1].artifact_id
    assert len(result.steps) == 1
    assert result.steps[0].selected_child == result.final_leaf_id


def test_traversal_through_multiple_levels_reaches_expected_leaf(
    compiler_spec, artifact_store
):
    tiny_branching_spec = dataclasses.replace(
        compiler_spec, max_children_per_tile=2, max_depth=8
    )
    artifacts = make_source_artifacts(5)
    manifest = compile_hierarchy(
        spec=tiny_branching_spec, source_artifacts=artifacts, store=artifact_store
    )

    # Route by declared priority index at every level: 0 then 0 walks to
    # the first leaf under a branching factor of 2.
    rule = DeclaredPriorityRule(priority_attribute="priority")
    request = TraversalRequestDTO(request_id="req-2", attributes=(("priority", "0"),))
    result = traverse(
        manifest=manifest, store=artifact_store, rule=rule, request=request
    )

    assert result.success
    assert result.final_artifact_id == artifacts[0].artifact_id
    assert len(result.steps) >= 2


def test_deterministic_tie_resolution(compiler_spec, artifact_store):
    artifacts = make_source_artifacts(3)
    manifest = compile_hierarchy(
        spec=compiler_spec, source_artifacts=artifacts, store=artifact_store
    )

    class AlwaysTieRule(FixedKeySelectorRule):
        def select_child(self, request, tile, children):  # type: ignore[override]
            from zeromodel.navigation.rules import ChildSelection

            winner = min(children, key=lambda child: child.order_key)
            return ChildSelection(
                selected=winner,
                eligible=children,
                tie_candidates=tuple(c.order_key for c in children),
                tie_resolution="lowest_order_key",
            )

    rule = AlwaysTieRule(routing_key="unused")
    request = TraversalRequestDTO(request_id="req-3")
    first = traverse(
        manifest=manifest, store=artifact_store, rule=rule, request=request
    )
    second = traverse(
        manifest=manifest, store=artifact_store, rule=rule, request=request
    )
    assert first == second
    assert first.steps[0].tie_resolution == "lowest_order_key"
    assert len(first.steps[0].tie_candidates) == len(artifacts)


def test_failure_is_represented_as_data_not_an_exception(compiler_spec, artifact_store):
    artifacts = make_source_artifacts(3)
    manifest = compile_hierarchy(
        spec=compiler_spec, source_artifacts=artifacts, store=artifact_store
    )

    rule = FixedKeySelectorRule(routing_key="target_order_key")
    request = TraversalRequestDTO(
        request_id="req-4", attributes=(("target_order_key", "does-not-exist"),)
    )
    result = traverse(
        manifest=manifest, store=artifact_store, rule=rule, request=request
    )

    assert result.success is False
    assert result.failure is not None
    assert result.failure.failure_code == "no_matching_child"
    assert result.steps[-1].failure_condition == "no_matching_child"


def test_max_depth_enforced_during_traversal(compiler_spec, artifact_store):
    tiny_branching_spec = dataclasses.replace(
        compiler_spec, max_children_per_tile=2, max_depth=8
    )
    artifacts = make_source_artifacts(5)
    manifest = compile_hierarchy(
        spec=tiny_branching_spec, source_artifacts=artifacts, store=artifact_store
    )

    rule = DeclaredPriorityRule(priority_attribute="priority")
    request = TraversalRequestDTO(request_id="req-5", attributes=(("priority", "0"),))
    result = traverse(
        manifest=manifest, store=artifact_store, rule=rule, request=request, max_depth=0
    )

    assert result.success is False
    assert result.failure.failure_code == "max_depth_exceeded"


def test_traversal_receipt_replays_to_the_same_path(compiler_spec, artifact_store):
    artifacts = make_source_artifacts(4)
    manifest = compile_hierarchy(
        spec=compiler_spec, source_artifacts=artifacts, store=artifact_store
    )

    rule = FixedKeySelectorRule(routing_key="target_order_key")
    request = TraversalRequestDTO(
        request_id="req-6", attributes=(("target_order_key", "00000002"),)
    )
    result = traverse(
        manifest=manifest, store=artifact_store, rule=rule, request=request
    )
    receipt = build_traversal_receipt(
        hierarchy_id=manifest.hierarchy_id, request=request, result=result
    )

    replayed = replay_traversal(
        receipt=receipt, manifest=manifest, store=artifact_store, rule=rule
    )

    assert replayed.success == result.success
    assert replayed.final_leaf_id == result.final_leaf_id
    assert replayed.steps == result.steps


def test_replay_rejects_mismatched_hierarchy(compiler_spec, artifact_store):
    artifacts = make_source_artifacts(2)
    manifest_a = compile_hierarchy(
        spec=compiler_spec, source_artifacts=artifacts, store=artifact_store
    )
    other_store = artifact_store
    other_artifacts = make_source_artifacts(3)
    manifest_b = compile_hierarchy(
        spec=compiler_spec, source_artifacts=other_artifacts, store=other_store
    )

    rule = FixedKeySelectorRule(routing_key="target_order_key")
    request = TraversalRequestDTO(
        request_id="req-7", attributes=(("target_order_key", "00000000"),)
    )
    result = traverse(
        manifest=manifest_a, store=artifact_store, rule=rule, request=request
    )
    receipt = build_traversal_receipt(
        hierarchy_id=manifest_a.hierarchy_id, request=request, result=result
    )

    from zeromodel.core.artifact import VPMValidationError

    with pytest.raises(VPMValidationError):
        replay_traversal(
            receipt=receipt, manifest=manifest_b, store=other_store, rule=rule
        )
