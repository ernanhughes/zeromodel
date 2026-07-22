"""Deterministic traversal and replay.

`traverse` walks from the hierarchy root, consulting `rule.select_child`
at each tile. Every step - including ties and failures - is recorded as
data in `TraversalStepDTO`, so a `TraversalReceiptDTO` built from the
result can be replayed to the same path via `replay_traversal`.
"""

from __future__ import annotations

from typing import List, Optional

from zeromodel.artifacts import ArtifactResolver
from zeromodel.core.artifact import VPMValidationError

from zeromodel.navigation.dto import (
    HierarchyManifestDTO,
    TraversalFailureDTO,
    TraversalReceiptDTO,
    TraversalRequestDTO,
    TraversalResultDTO,
    TraversalStepDTO,
)
from zeromodel.navigation.rules import ChildSelection, TraversalRule
from zeromodel.navigation.storage import load_leaf_binding, load_tile


def _build_step(
    *,
    tile_id: str,
    descriptor,
    request: TraversalRequestDTO,
    eligible: tuple,
    selected_child: Optional[str],
    selection: ChildSelection,
    failure_condition: Optional[str],
) -> TraversalStepDTO:
    return TraversalStepDTO(
        tile_id=tile_id,
        rule_descriptor=descriptor,
        request_id=request.request_id,
        eligible_children=eligible,
        selected_child=selected_child,
        tie_candidates=selection.tie_candidates,
        tie_resolution=selection.tie_resolution,
        failure_condition=failure_condition,
    )


def _failure_result(
    failure: TraversalFailureDTO, steps: List[TraversalStepDTO]
) -> TraversalResultDTO:
    return TraversalResultDTO(
        success=False,
        final_leaf_id=None,
        final_artifact_kind=None,
        final_artifact_id=None,
        steps=tuple(steps),
        failure=failure,
    )


def traverse(
    *,
    manifest: HierarchyManifestDTO,
    store: ArtifactResolver,
    rule: TraversalRule,
    request: TraversalRequestDTO,
    max_depth: Optional[int] = None,
) -> TraversalResultDTO:
    depth_limit = manifest.max_depth if max_depth is None else max_depth
    current_tile_id = manifest.root_ref.artifact_id
    descriptor = rule.descriptor()
    steps: List[TraversalStepDTO] = []
    depth = 0

    while True:
        tile = load_tile(store, current_tile_id)
        selection = rule.select_child(request, tile, tile.children)
        eligible = tuple((child.target_id, child.order_key) for child in tile.children)

        if selection.selected is None:
            failure = TraversalFailureDTO(
                failure_code="no_matching_child",
                message=f"no child matched request {request.request_id!r} at tile {current_tile_id}",
                at_tile_id=current_tile_id,
            )
            steps.append(
                _build_step(
                    tile_id=current_tile_id,
                    descriptor=descriptor,
                    request=request,
                    eligible=eligible,
                    selected_child=None,
                    selection=selection,
                    failure_condition=failure.failure_code,
                )
            )
            return _failure_result(failure, steps)

        selected = selection.selected

        if selected.pointer_kind == "leaf":
            steps.append(
                _build_step(
                    tile_id=current_tile_id,
                    descriptor=descriptor,
                    request=request,
                    eligible=eligible,
                    selected_child=selected.target_id,
                    selection=selection,
                    failure_condition=None,
                )
            )
            binding = load_leaf_binding(store, selected.target_id)
            return TraversalResultDTO(
                success=True,
                final_leaf_id=selected.target_id,
                final_artifact_kind=binding.artifact_ref.artifact_kind,
                final_artifact_id=binding.artifact_ref.artifact_id,
                steps=tuple(steps),
                failure=None,
            )

        # selected.pointer_kind == "tile": descending is subject to the depth limit.
        if depth + 1 > depth_limit:
            failure = TraversalFailureDTO(
                failure_code="max_depth_exceeded",
                message=f"traversal would exceed max_depth={depth_limit} descending from {current_tile_id}",
                at_tile_id=current_tile_id,
            )
            steps.append(
                _build_step(
                    tile_id=current_tile_id,
                    descriptor=descriptor,
                    request=request,
                    eligible=eligible,
                    selected_child=selected.target_id,
                    selection=selection,
                    failure_condition=failure.failure_code,
                )
            )
            return _failure_result(failure, steps)

        steps.append(
            _build_step(
                tile_id=current_tile_id,
                descriptor=descriptor,
                request=request,
                eligible=eligible,
                selected_child=selected.target_id,
                selection=selection,
                failure_condition=None,
            )
        )
        current_tile_id = selected.target_id
        depth += 1


def replay_traversal(
    *,
    receipt: TraversalReceiptDTO,
    manifest: HierarchyManifestDTO,
    store: ArtifactResolver,
    rule: TraversalRule,
) -> TraversalResultDTO:
    """Re-execute the traversal a receipt recorded, against the same
    hierarchy and rule, and return the freshly computed result for
    comparison against the receipt."""
    if receipt.hierarchy_id != manifest.hierarchy_id:
        raise VPMValidationError(
            "TraversalReceiptDTO.hierarchy_id does not match the supplied HierarchyManifestDTO"
        )
    return traverse(manifest=manifest, store=store, rule=rule, request=receipt.request)
