"""Regression coverage for hierarchy closure over the *represented corpus*,
not merely the routing structure.

An external review of the first merged commit found that `validate_hierarchy`
checked that a leaf-binding record resolved and declared the right kind, but
never checked that the leaf binding's *referenced source artifact* actually
resolved through the store - so a hierarchy could pass closure while
pointing at source artifacts that were never stored. These tests exercise
the three closure checks added in response: source-artifact resolution,
leaf-count reconciliation, and source-artifact-set digest reconciliation.
"""

from __future__ import annotations

import pytest

from zeromodel.artifacts import ArtifactRef, sha256_digest
from zeromodel.navigation import compile_hierarchy, validate_hierarchy
from zeromodel.navigation.dto import (
    HierarchyManifestDTO,
    LeafBindingDTO,
    NavigationTileDTO,
    TileCoverageDTO,
    TilePointerDTO,
    compute_hierarchy_id,
    compute_leaf_id,
    compute_source_artifact_digest,
    compute_tile_id,
)
from zeromodel.navigation.errors import HierarchyClosureError, HierarchyCompilationError
from zeromodel.navigation.storage import store_leaf_binding, store_tile


def _build_single_leaf_manifest(
    compiler_spec,
    artifact_store,
    make_source_artifacts,
    *,
    leaf_count_override=None,
    source_digest_override=None,
):
    """Build a manifest for a one-tile, one-leaf hierarchy, with the leaf
    binding and its source artifact genuinely stored, but with optional
    deliberate lies in coverage.leaf_count / source_artifact_digest."""
    ref = make_source_artifacts(1, store=artifact_store)[0]
    leaf_id = compute_leaf_id(
        artifact_ref=ref, leaf_semantics=compiler_spec.leaf_semantics
    )
    binding = LeafBindingDTO(
        leaf_id=leaf_id, artifact_ref=ref, leaf_semantics=compiler_spec.leaf_semantics
    )
    store_leaf_binding(artifact_store, binding)

    pointer = TilePointerDTO(
        pointer_kind="leaf", target_id=leaf_id, order_key="00000000"
    )
    coverage = TileCoverageDTO(
        corpus_id=compiler_spec.corpus_id,
        partition_key="root",
        child_count=1,
        leaf_count=leaf_count_override if leaf_count_override is not None else 1,
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
    source_digest = source_digest_override or compute_source_artifact_digest((ref,))
    return HierarchyManifestDTO(
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
    ), ref


def test_leaf_referencing_an_unstored_source_artifact_fails_closure(
    compiler_spec, artifact_store
):
    """The leaf-binding record itself resolves and declares the right
    kind, but the source artifact it points at was never stored - this
    must fail even though the routing structure is perfectly closed."""
    unstored_ref = ArtifactRef(
        artifact_kind=compiler_spec.corpus_artifact_kind,
        artifact_id=sha256_digest(b"never-actually-stored"),
    )
    leaf_id = compute_leaf_id(
        artifact_ref=unstored_ref, leaf_semantics=compiler_spec.leaf_semantics
    )
    binding = LeafBindingDTO(
        leaf_id=leaf_id,
        artifact_ref=unstored_ref,
        leaf_semantics=compiler_spec.leaf_semantics,
    )
    store_leaf_binding(artifact_store, binding)  # binding resolves...

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
    source_digest = compute_source_artifact_digest((unstored_ref,))
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

    # ... but its referenced source artifact was never put() into the store.
    with pytest.raises(HierarchyClosureError, match="does not resolve"):
        validate_hierarchy(manifest, artifact_store)


def test_declared_leaf_count_mismatch_fails_closure(
    compiler_spec, artifact_store, make_source_artifacts
):
    manifest, _ = _build_single_leaf_manifest(
        compiler_spec, artifact_store, make_source_artifacts, leaf_count_override=2
    )
    with pytest.raises(HierarchyClosureError, match="leaf_count"):
        validate_hierarchy(manifest, artifact_store)


def test_source_artifact_digest_mismatch_fails_closure(
    compiler_spec, artifact_store, make_source_artifacts
):
    wrong_digest = compute_source_artifact_digest(
        (
            ArtifactRef(
                artifact_kind=compiler_spec.corpus_artifact_kind,
                artifact_id=sha256_digest(b"unrelated"),
            ),
        )
    )
    manifest, _ = _build_single_leaf_manifest(
        compiler_spec,
        artifact_store,
        make_source_artifacts,
        source_digest_override=wrong_digest,
    )
    with pytest.raises(HierarchyClosureError, match="source_artifact_digest"):
        validate_hierarchy(manifest, artifact_store)


def test_valid_single_leaf_hierarchy_still_passes_closure(
    compiler_spec, artifact_store, make_source_artifacts
):
    manifest, ref = _build_single_leaf_manifest(
        compiler_spec, artifact_store, make_source_artifacts
    )
    visited = validate_hierarchy(manifest, artifact_store)
    assert manifest.root_ref.artifact_id in visited


def test_duplicate_source_artifact_is_rejected_at_compilation(
    compiler_spec, artifact_store, make_source_artifacts
):
    ref = make_source_artifacts(1, store=artifact_store)[0]
    with pytest.raises(HierarchyCompilationError, match="duplicate source artifact"):
        compile_hierarchy(
            spec=compiler_spec, source_artifacts=(ref, ref), store=artifact_store
        )
