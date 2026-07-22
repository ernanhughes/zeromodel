from __future__ import annotations

import dataclasses
import types

import pytest

from zeromodel.artifacts import ArtifactRef, InMemoryArtifactStore, sha256_digest
from zeromodel.core.artifact import VPMValidationError
from zeromodel.navigation import compile_hierarchy, validate_hierarchy
from zeromodel.navigation.dto import (
    NavigationTileDTO,
    TileCoverageDTO,
    TilePointerDTO,
    compute_leaf_id,
    compute_tile_id,
)
from zeromodel.navigation.errors import HierarchyClosureError, HierarchyCompilationError
from zeromodel.navigation.storage import store_tile


def test_identical_input_and_spec_yield_identical_hierarchy_identity(
    compiler_spec, make_source_artifacts
):
    store_a = InMemoryArtifactStore()
    store_b = InMemoryArtifactStore()
    artifacts_a = make_source_artifacts(5, store=store_a)
    artifacts_b = make_source_artifacts(5, store=store_b)
    manifest_a = compile_hierarchy(
        spec=compiler_spec, source_artifacts=artifacts_a, store=store_a
    )
    manifest_b = compile_hierarchy(
        spec=compiler_spec, source_artifacts=artifacts_b, store=store_b
    )
    assert manifest_a.hierarchy_id == manifest_b.hierarchy_id
    assert manifest_a.root_ref == manifest_b.root_ref


def test_changed_child_order_changes_hierarchy_identity(
    compiler_spec, make_source_artifacts
):
    store_original = InMemoryArtifactStore()
    store_reordered = InMemoryArtifactStore()
    artifacts = make_source_artifacts(5, store=store_original)
    artifacts_for_reorder = make_source_artifacts(5, store=store_reordered)
    reordered = (
        artifacts_for_reorder[1],
        artifacts_for_reorder[0],
        *artifacts_for_reorder[2:],
    )
    manifest_original = compile_hierarchy(
        spec=compiler_spec, source_artifacts=artifacts, store=store_original
    )
    manifest_reordered = compile_hierarchy(
        spec=compiler_spec, source_artifacts=reordered, store=store_reordered
    )
    assert manifest_original.hierarchy_id != manifest_reordered.hierarchy_id


def test_changed_spec_rule_changes_hierarchy_identity(
    compiler_spec, make_source_artifacts
):
    store_original = InMemoryArtifactStore()
    store_changed = InMemoryArtifactStore()
    artifacts = make_source_artifacts(5, store=store_original)
    artifacts_for_changed = make_source_artifacts(5, store=store_changed)
    manifest_original = compile_hierarchy(
        spec=compiler_spec, source_artifacts=artifacts, store=store_original
    )
    changed_spec = dataclasses.replace(compiler_spec, tie_rule="highest_order_key")
    manifest_changed = compile_hierarchy(
        spec=changed_spec, source_artifacts=artifacts_for_changed, store=store_changed
    )
    assert manifest_original.hierarchy_id != manifest_changed.hierarchy_id


def test_incompatible_corpus_artifact_fails_compilation(
    compiler_spec, artifact_store, make_source_artifacts
):
    artifacts = make_source_artifacts(3, store=artifact_store) + (
        ArtifactRef(
            artifact_kind="not-the-right-kind",
            artifact_id=sha256_digest(b"wrong-kind-payload"),
        ),
    )
    with pytest.raises(HierarchyCompilationError):
        compile_hierarchy(
            spec=compiler_spec, source_artifacts=artifacts, store=artifact_store
        )


def test_max_depth_enforced_at_compile_time(
    compiler_spec, artifact_store, make_source_artifacts
):
    tiny_depth_spec = dataclasses.replace(
        compiler_spec, max_children_per_tile=2, max_depth=1
    )
    # 9 artifacts with a branching factor of 2 requires more than 1 level.
    artifacts = make_source_artifacts(9, store=artifact_store)
    with pytest.raises(HierarchyCompilationError):
        compile_hierarchy(
            spec=tiny_depth_spec, source_artifacts=artifacts, store=artifact_store
        )


def test_root_and_every_reference_resolves_after_compilation(
    compiler_spec, artifact_store, make_source_artifacts
):
    artifacts = make_source_artifacts(7, store=artifact_store)
    manifest = compile_hierarchy(
        spec=compiler_spec, source_artifacts=artifacts, store=artifact_store
    )
    visited = validate_hierarchy(manifest, artifact_store)
    assert manifest.root_ref.artifact_id in visited
    assert len(visited) == len(set(visited))  # every tile visited exactly once


def test_missing_leaf_reference_fails_closure(
    compiler_spec, artifact_store, make_source_artifacts
):
    ref = make_source_artifacts(1)[0]
    leaf_id = compute_leaf_id(
        artifact_ref=ref, leaf_semantics=compiler_spec.leaf_semantics
    )
    # Deliberately never store the leaf binding this pointer targets.
    pointer = TilePointerDTO(
        pointer_kind="leaf", target_id=leaf_id, order_key="00000000"
    )
    coverage = TileCoverageDTO(
        corpus_id=compiler_spec.corpus_id,
        partition_key="root",
        child_count=1,
        leaf_count=1,
    )
    tile_id = compute_tile_id(
        depth=0, coverage=coverage, children=(pointer,), tie_rule=compiler_spec.tie_rule
    )
    tile = NavigationTileDTO(
        tile_id=tile_id,
        depth=0,
        coverage=coverage,
        children=(pointer,),
        tie_rule=compiler_spec.tie_rule,
    )
    store_tile(artifact_store, tile)

    root_ref = ArtifactRef(artifact_kind="navigation-tile", artifact_id=tile_id)
    from zeromodel.navigation.dto import (
        HierarchyManifestDTO,
        compute_hierarchy_id,
        compute_source_artifact_digest,
    )

    source_digest = compute_source_artifact_digest((ref,))
    hierarchy_id = compute_hierarchy_id(
        root_ref=root_ref, source_artifact_digest=source_digest, spec=compiler_spec
    )
    manifest = HierarchyManifestDTO(
        hierarchy_id=hierarchy_id,
        root_ref=root_ref,
        source_artifact_digest=source_digest,
        compiler_id=compiler_spec.compiler_id,
        compiler_version=compiler_spec.compiler_version,
        corpus_id=compiler_spec.corpus_id,
        corpus_artifact_kind=compiler_spec.corpus_artifact_kind,
        leaf_semantics=compiler_spec.leaf_semantics,
        max_children_per_tile=compiler_spec.max_children_per_tile,
        max_depth=compiler_spec.max_depth,
        tie_rule=compiler_spec.tie_rule,
        failure_rule=compiler_spec.failure_rule,
        navigation_rule_contract=compiler_spec.navigation_rule_contract,
        child_ordering_rule=compiler_spec.child_ordering_rule,
    )
    with pytest.raises(HierarchyClosureError):
        validate_hierarchy(manifest, artifact_store)


def test_duplicate_child_reference_is_rejected_at_construction(compiler_spec):
    """Content-addressed identity makes a genuinely duplicate child
    structurally detectable at tile-construction time already."""
    pointer = TilePointerDTO(
        pointer_kind="leaf", target_id="sha256:" + "a" * 64, order_key="00000000"
    )
    coverage = TileCoverageDTO(
        corpus_id=compiler_spec.corpus_id,
        partition_key="root",
        child_count=2,
        leaf_count=2,
    )
    with pytest.raises(VPMValidationError):
        NavigationTileDTO(
            tile_id="sha256:" + "b" * 64,
            depth=0,
            coverage=coverage,
            children=(pointer, pointer),
            tie_rule=compiler_spec.tie_rule,
        )


def test_self_referencing_tile_fails_closure(
    compiler_spec, artifact_store, monkeypatch, make_source_artifacts
):
    """Content-addressed identity makes an honest self-reference
    infeasible to construct (a tile's id is a hash of its own children).
    This test exercises the defensive guard in `validate_hierarchy`
    directly, simulating a corrupted/tampered store."""
    root_id = "sha256:" + "c" * 64
    fake_self_pointer = types.SimpleNamespace(pointer_kind="tile", target_id=root_id)
    fake_tile = types.SimpleNamespace(
        tile_id=root_id,
        depth=0,
        children=(fake_self_pointer,),
        coverage=types.SimpleNamespace(
            child_count=1, corpus_id=compiler_spec.corpus_id
        ),
    )

    monkeypatch.setattr(
        "zeromodel.navigation.compiler.tile_exists", lambda store, tile_id: True
    )
    monkeypatch.setattr(
        "zeromodel.navigation.compiler.load_tile", lambda store, tile_id: fake_tile
    )

    from zeromodel.navigation.dto import (
        HierarchyManifestDTO,
        compute_hierarchy_id,
        compute_source_artifact_digest,
    )

    ref = make_source_artifacts(1)[0]
    root_ref = ArtifactRef(artifact_kind="navigation-tile", artifact_id=root_id)
    source_digest = compute_source_artifact_digest((ref,))
    manifest = HierarchyManifestDTO(
        hierarchy_id=compute_hierarchy_id(
            root_ref=root_ref, source_artifact_digest=source_digest, spec=compiler_spec
        ),
        root_ref=root_ref,
        source_artifact_digest=source_digest,
        compiler_id=compiler_spec.compiler_id,
        compiler_version=compiler_spec.compiler_version,
        corpus_id=compiler_spec.corpus_id,
        corpus_artifact_kind=compiler_spec.corpus_artifact_kind,
        leaf_semantics=compiler_spec.leaf_semantics,
        max_children_per_tile=compiler_spec.max_children_per_tile,
        max_depth=compiler_spec.max_depth,
        tie_rule=compiler_spec.tie_rule,
        failure_rule=compiler_spec.failure_rule,
        navigation_rule_contract=compiler_spec.navigation_rule_contract,
        child_ordering_rule=compiler_spec.child_ordering_rule,
    )
    with pytest.raises(HierarchyClosureError, match="references itself"):
        validate_hierarchy(manifest, artifact_store)


def test_cyclic_hierarchy_fails_closure(
    compiler_spec, artifact_store, monkeypatch, make_source_artifacts
):
    """As with self-reference, a genuine cycle cannot arise from honest
    content-addressed hashing (B's id would have to depend on A's id and
    vice versa). This exercises the path-tracking cycle guard directly
    against a simulated corrupted store."""
    tile_a_id = "sha256:" + "d" * 64
    tile_b_id = "sha256:" + "e" * 64
    pointer_to_b = types.SimpleNamespace(pointer_kind="tile", target_id=tile_b_id)
    pointer_to_a = types.SimpleNamespace(pointer_kind="tile", target_id=tile_a_id)
    tile_a = types.SimpleNamespace(
        tile_id=tile_a_id,
        depth=1,
        children=(pointer_to_b,),
        coverage=types.SimpleNamespace(
            child_count=1, corpus_id=compiler_spec.corpus_id
        ),
    )
    tile_b = types.SimpleNamespace(
        tile_id=tile_b_id,
        depth=0,
        children=(pointer_to_a,),
        coverage=types.SimpleNamespace(
            child_count=1, corpus_id=compiler_spec.corpus_id
        ),
    )
    fake_tiles = {tile_a_id: tile_a, tile_b_id: tile_b}

    monkeypatch.setattr(
        "zeromodel.navigation.compiler.tile_exists",
        lambda store, tile_id: tile_id in fake_tiles,
    )
    monkeypatch.setattr(
        "zeromodel.navigation.compiler.load_tile",
        lambda store, tile_id: fake_tiles[tile_id],
    )

    from zeromodel.navigation.dto import (
        HierarchyManifestDTO,
        compute_hierarchy_id,
        compute_source_artifact_digest,
    )

    ref = make_source_artifacts(1)[0]
    root_ref = ArtifactRef(artifact_kind="navigation-tile", artifact_id=tile_a_id)
    source_digest = compute_source_artifact_digest((ref,))
    manifest = HierarchyManifestDTO(
        hierarchy_id=compute_hierarchy_id(
            root_ref=root_ref, source_artifact_digest=source_digest, spec=compiler_spec
        ),
        root_ref=root_ref,
        source_artifact_digest=source_digest,
        compiler_id=compiler_spec.compiler_id,
        compiler_version=compiler_spec.compiler_version,
        corpus_id=compiler_spec.corpus_id,
        corpus_artifact_kind=compiler_spec.corpus_artifact_kind,
        leaf_semantics=compiler_spec.leaf_semantics,
        max_children_per_tile=compiler_spec.max_children_per_tile,
        max_depth=compiler_spec.max_depth,
        tie_rule=compiler_spec.tie_rule,
        failure_rule=compiler_spec.failure_rule,
        navigation_rule_contract=compiler_spec.navigation_rule_contract,
        child_ordering_rule=compiler_spec.child_ordering_rule,
    )
    with pytest.raises(HierarchyClosureError, match="cycle detected"):
        validate_hierarchy(manifest, artifact_store)


def test_navigation_owns_no_independent_persistence(
    compiler_spec, artifact_store, make_source_artifacts
):
    """All tiles/leaf bindings must be retrievable through the injected
    Artifacts store alone - Navigation keeps no side-channel storage."""
    artifacts = make_source_artifacts(4, store=artifact_store)
    manifest = compile_hierarchy(
        spec=compiler_spec, source_artifacts=artifacts, store=artifact_store
    )
    other_store = InMemoryArtifactStore()
    with pytest.raises(Exception):
        validate_hierarchy(manifest, other_store)
    # But it resolves fully through the store it was actually compiled into.
    validate_hierarchy(manifest, artifact_store)
