"""Deterministic compilation and structural closure validation.

`compile_hierarchy` never reorders its input: identical `source_artifacts`
order plus an identical `spec` always yields an identical `HierarchyManifestDTO.
hierarchy_id`; a deliberately different order or a changed spec parameter
yields a different one. `validate_hierarchy` is an exhaustive, finite
structural walk of an already-compiled hierarchy - it performs no
similarity comparison and is not search.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from zeromodel.artifacts import ArtifactRef, ArtifactResolver, ArtifactStore

from zeromodel.navigation.dto import (
    HierarchyCompilerSpecDTO,
    HierarchyManifestDTO,
    NavigationTileDTO,
    TileCoverageDTO,
    TilePointerDTO,
    compute_hierarchy_id,
    compute_leaf_id,
    compute_source_artifact_digest,
    compute_tile_id,
)
from zeromodel.navigation.dto import LeafBindingDTO
from zeromodel.navigation.errors import HierarchyClosureError, HierarchyCompilationError
from zeromodel.navigation.storage import (
    leaf_binding_exists,
    load_leaf_binding,
    load_tile,
    store_leaf_binding,
    store_tile,
    tile_exists,
)


def _chunk(items: Sequence, size: int) -> List[Sequence]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _validate_corpus_contract(
    spec: HierarchyCompilerSpecDTO, source_artifacts: Sequence[ArtifactRef]
) -> None:
    if not source_artifacts:
        raise HierarchyCompilationError(
            "compile_hierarchy requires at least one source artifact"
        )
    for ref in source_artifacts:
        if ref.artifact_kind != spec.corpus_artifact_kind:
            raise HierarchyCompilationError(
                f"artifact {ref.artifact_id} has kind {ref.artifact_kind!r}, "
                f"but this corpus requires {spec.corpus_artifact_kind!r}"
            )


def _build_leaf_level(
    spec: HierarchyCompilerSpecDTO,
    source_artifacts: Sequence[ArtifactRef],
    store: ArtifactStore,
) -> List[Tuple[TilePointerDTO, int]]:
    """Level 0: one leaf binding per source artifact, in declared order.

    Each entry is (pointer, leaf_count_under_this_pointer) - leaf_count is 1
    for every leaf and is summed up through parent levels for coverage.
    """
    level: List[Tuple[TilePointerDTO, int]] = []
    for index, ref in enumerate(source_artifacts):
        leaf_id = compute_leaf_id(artifact_ref=ref, leaf_semantics=spec.leaf_semantics)
        binding = LeafBindingDTO(
            leaf_id=leaf_id, artifact_ref=ref, leaf_semantics=spec.leaf_semantics
        )
        store_leaf_binding(store, binding)
        pointer = TilePointerDTO(
            pointer_kind="leaf", target_id=leaf_id, order_key=f"{index:08d}"
        )
        level.append((pointer, 1))
    return level


def _build_parent_level(
    current: List[Tuple[TilePointerDTO, int]],
    spec: HierarchyCompilerSpecDTO,
    depth_from_leaves: int,
    store: ArtifactStore,
) -> List[Tuple[TilePointerDTO, int]]:
    next_level: List[Tuple[TilePointerDTO, int]] = []
    for chunk_index, chunk_entries in enumerate(
        _chunk(current, spec.max_children_per_tile)
    ):
        children = tuple(entry[0] for entry in chunk_entries)
        leaf_count = sum(entry[1] for entry in chunk_entries)
        coverage = TileCoverageDTO(
            corpus_id=spec.corpus_id,
            partition_key=f"L{depth_from_leaves}-{chunk_index}",
            child_count=len(children),
            leaf_count=leaf_count,
        )
        tile_id = compute_tile_id(
            depth=depth_from_leaves,
            coverage=coverage,
            children=children,
            tie_rule=spec.tie_rule,
        )
        tile = NavigationTileDTO(
            tile_id=tile_id,
            depth=depth_from_leaves,
            coverage=coverage,
            children=children,
            tie_rule=spec.tie_rule,
        )
        store_tile(store, tile)
        pointer = TilePointerDTO(
            pointer_kind="tile", target_id=tile_id, order_key=f"{chunk_index:08d}"
        )
        next_level.append((pointer, leaf_count))
    return next_level


def _build_manifest(
    spec: HierarchyCompilerSpecDTO, root_ref: ArtifactRef, source_artifact_digest: str
) -> HierarchyManifestDTO:
    hierarchy_id = compute_hierarchy_id(
        root_ref=root_ref, source_artifact_digest=source_artifact_digest, spec=spec
    )
    return HierarchyManifestDTO(
        hierarchy_id=hierarchy_id,
        root_ref=root_ref,
        source_artifact_digest=source_artifact_digest,
        compiler_id=spec.compiler_id,
        compiler_version=spec.compiler_version,
        corpus_id=spec.corpus_id,
        corpus_artifact_kind=spec.corpus_artifact_kind,
        leaf_semantics=spec.leaf_semantics,
        max_children_per_tile=spec.max_children_per_tile,
        max_depth=spec.max_depth,
        tie_rule=spec.tie_rule,
        failure_rule=spec.failure_rule,
        navigation_rule_contract=spec.navigation_rule_contract,
        child_ordering_rule=spec.child_ordering_rule,
        partition_parameters=spec.partition_parameters,
    )


def compile_hierarchy(
    *,
    spec: HierarchyCompilerSpecDTO,
    source_artifacts: Sequence[ArtifactRef],
    store: ArtifactStore,
) -> HierarchyManifestDTO:
    _validate_corpus_contract(spec, source_artifacts)

    current = _build_leaf_level(spec, source_artifacts, store)
    depth_from_leaves = 0
    while True:
        next_level = _build_parent_level(current, spec, depth_from_leaves, store)
        if len(next_level) == 1:
            root_ref = ArtifactRef(
                artifact_kind="navigation-tile", artifact_id=next_level[0][0].target_id
            )
            break
        current = next_level
        depth_from_leaves += 1

    source_artifact_digest = compute_source_artifact_digest(tuple(source_artifacts))
    manifest = _build_manifest(spec, root_ref, source_artifact_digest)

    # Defense in depth: the compiler never hands back a hierarchy that
    # would not itself pass closure validation (this also enforces
    # max_depth against the tree the grouping loop actually produced).
    try:
        validate_hierarchy(manifest, store)
    except HierarchyClosureError as exc:
        raise HierarchyCompilationError(
            f"compiled hierarchy failed closure validation: {exc}"
        ) from exc
    return manifest


def validate_hierarchy(
    manifest: HierarchyManifestDTO, store: ArtifactResolver
) -> Tuple[str, ...]:
    """Exhaustively validate structural closure. Returns every reachable
    tile_id (root first) on success; raises `HierarchyClosureError` on the
    first violation found."""
    if not tile_exists(store, manifest.root_ref.artifact_id):
        raise HierarchyClosureError(
            f"hierarchy root {manifest.root_ref.artifact_id} does not resolve"
        )

    visited_tiles: List[str] = []
    path_stack: set = set()

    def _walk(tile_id: str, depth_from_root: int) -> None:
        if tile_id in path_stack:
            raise HierarchyClosureError(f"cycle detected at tile {tile_id}")
        if depth_from_root > manifest.max_depth:
            raise HierarchyClosureError(
                f"hierarchy exceeds declared max_depth={manifest.max_depth} at tile {tile_id}"
            )
        tile = load_tile(store, tile_id)
        visited_tiles.append(tile_id)
        path_stack.add(tile_id)

        if tile.coverage.child_count != len(tile.children):
            raise HierarchyClosureError(
                f"tile {tile_id} declares child_count that does not match its children"
            )

        seen_targets = set()
        for child in tile.children:
            if child.target_id == tile_id:
                raise HierarchyClosureError(
                    f"tile {tile_id} references itself as a child"
                )
            if child.target_id in seen_targets:
                raise HierarchyClosureError(
                    f"tile {tile_id} has a duplicate child reference {child.target_id}"
                )
            seen_targets.add(child.target_id)

            if child.pointer_kind == "tile":
                if not tile_exists(store, child.target_id):
                    raise HierarchyClosureError(
                        f"tile {tile_id} child tile {child.target_id} does not resolve"
                    )
                _walk(child.target_id, depth_from_root + 1)
            elif child.pointer_kind == "leaf":
                if not leaf_binding_exists(store, child.target_id):
                    raise HierarchyClosureError(
                        f"tile {tile_id} child leaf {child.target_id} does not resolve"
                    )
                binding = load_leaf_binding(store, child.target_id)
                if binding.leaf_semantics != manifest.leaf_semantics:
                    raise HierarchyClosureError(
                        f"leaf {child.target_id} declares semantics {binding.leaf_semantics!r}, "
                        f"expected {manifest.leaf_semantics!r}"
                    )
                if binding.artifact_ref.artifact_kind != manifest.corpus_artifact_kind:
                    raise HierarchyClosureError(
                        f"leaf {child.target_id} artifact kind {binding.artifact_ref.artifact_kind!r} "
                        f"violates corpus contract {manifest.corpus_artifact_kind!r}"
                    )
            else:  # pragma: no cover - TilePointerDTO already restricts this
                raise HierarchyClosureError(
                    f"tile {tile_id} has a child with an unknown pointer_kind"
                )

        path_stack.discard(tile_id)

    _walk(manifest.root_ref.artifact_id, 0)
    return tuple(visited_tiles)
