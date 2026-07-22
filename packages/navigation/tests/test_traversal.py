from __future__ import annotations

import dataclasses

import pytest

from zeromodel.navigation import compile_hierarchy, replay_traversal, traverse
from zeromodel.navigation.dto import (
    TraversalRequestDTO,
    TraversalResultDTO,
    TraversalRuleDescriptorDTO,
    TraversalStepDTO,
    build_traversal_receipt,
)
from zeromodel.navigation.rules import DeclaredPriorityRule, FixedKeySelectorRule


def test_traversal_reaches_expected_leaf(
    compiler_spec, artifact_store, make_source_artifacts
):
    artifacts = make_source_artifacts(3, store=artifact_store)
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
    compiler_spec, artifact_store, make_source_artifacts
):
    tiny_branching_spec = dataclasses.replace(
        compiler_spec, max_children_per_tile=2, max_depth=8
    )
    artifacts = make_source_artifacts(5, store=artifact_store)
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


def test_deterministic_tie_resolution(
    compiler_spec, artifact_store, make_source_artifacts
):
    artifacts = make_source_artifacts(3, store=artifact_store)
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


def test_failure_is_represented_as_data_not_an_exception(
    compiler_spec, artifact_store, make_source_artifacts
):
    artifacts = make_source_artifacts(3, store=artifact_store)
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


def test_max_depth_enforced_during_traversal(
    compiler_spec, artifact_store, make_source_artifacts
):
    tiny_branching_spec = dataclasses.replace(
        compiler_spec, max_children_per_tile=2, max_depth=8
    )
    artifacts = make_source_artifacts(5, store=artifact_store)
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


def test_traversal_receipt_replays_to_the_same_path(
    compiler_spec, artifact_store, make_source_artifacts
):
    artifacts = make_source_artifacts(4, store=artifact_store)
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


def test_replay_rejects_mismatched_hierarchy(
    compiler_spec, artifact_store, make_source_artifacts
):
    artifacts = make_source_artifacts(2, store=artifact_store)
    manifest_a = compile_hierarchy(
        spec=compiler_spec, source_artifacts=artifacts, store=artifact_store
    )
    other_store = artifact_store
    other_artifacts = make_source_artifacts(3, store=other_store)
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


def test_receipt_identity_distinguishes_same_leaf_different_bound_artifact() -> None:
    """Two traversals that reach the same leaf id but resolve to a
    different bound artifact (or a different rule/eligible-children/tie
    trace) must never collide on `receipt_id`."""
    descriptor = TraversalRuleDescriptorDTO(
        rule_kind="fixed_selector_by_request_key", parameters=(("routing_key", "k"),)
    )
    leaf_id = "sha256:" + "2" * 64
    step = TraversalStepDTO(
        tile_id="sha256:" + "1" * 64,
        rule_descriptor=descriptor,
        request_id="req",
        eligible_children=((leaf_id, "0"),),
        selected_child=leaf_id,
        failure_condition=None,
    )
    request = TraversalRequestDTO(request_id="req")
    hierarchy_id = "sha256:" + "5" * 64

    result_a = TraversalResultDTO(
        success=True,
        final_leaf_id=leaf_id,
        final_artifact_kind="kind-a",
        final_artifact_id="sha256:" + "3" * 64,
        steps=(step,),
    )
    result_b = TraversalResultDTO(
        success=True,
        final_leaf_id=leaf_id,
        final_artifact_kind="kind-b",
        final_artifact_id="sha256:" + "4" * 64,
        steps=(step,),
    )

    receipt_a = build_traversal_receipt(
        hierarchy_id=hierarchy_id, request=request, result=result_a
    )
    receipt_b = build_traversal_receipt(
        hierarchy_id=hierarchy_id, request=request, result=result_b
    )

    assert receipt_a.result.final_leaf_id == receipt_b.result.final_leaf_id
    assert receipt_a.receipt_id != receipt_b.receipt_id


def test_duplicate_attribute_key_is_rejected_not_silently_collapsed() -> None:
    request = TraversalRequestDTO(
        request_id="req", attributes=(("k", "first"), ("k", "second"))
    )
    from zeromodel.core.artifact import VPMValidationError

    with pytest.raises(VPMValidationError, match="duplicate key"):
        request.attributes_map
