"""Regression coverage for the load-time identity hardening in storage.py.

An external review of the first merged commit found that `load_tile` and
`load_leaf_binding` reconstructed DTOs from the store's `manifest` (which
could, in principle, be swapped for a different tile/leaf binding's
content under the same `ArtifactRef` key) rather than from canonical
bytes with a verified digest. These tests prove the fix: loading is
manifest-blind and fails closed on any digest mismatch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from zeromodel.artifacts import InMemoryArtifactStore
from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.content_identity import canonical_json_bytes
from zeromodel.navigation.dto import (
    LeafBindingDTO,
    NavigationTileDTO,
    TileCoverageDTO,
    TilePointerDTO,
    compute_leaf_id,
    compute_tile_id,
    leaf_binding_identity_payload,
    tile_identity_payload,
)
from zeromodel.navigation.storage import (
    LEAF_BINDING_ARTIFACT_KIND,
    TILE_ARTIFACT_KIND,
    load_leaf_binding,
    load_tile,
)

CORPUS_ARTIFACT_KIND = "policy-snapshot"


def test_load_tile_ignores_a_manifest_that_lies_about_the_content() -> None:
    """Even if a store's manifest were substituted for a different tile's
    full content (as the Artifacts package's `put()` manifest-conflict
    check now prevents), `load_tile` must not be fooled - it never reads
    the manifest at all."""
    store = InMemoryArtifactStore()
    coverage = TileCoverageDTO(
        corpus_id="c", partition_key="root", child_count=1, leaf_count=1
    )
    pointer = TilePointerDTO(
        pointer_kind="leaf", target_id="sha256:" + "a" * 64, order_key="0"
    )
    tile_id = compute_tile_id(
        depth=0, coverage=coverage, children=(pointer,), tie_rule="lowest_order_key"
    )
    tile = NavigationTileDTO(
        tile_id=tile_id,
        depth=0,
        coverage=coverage,
        children=(pointer,),
        tie_rule="lowest_order_key",
    )
    canonical_bytes = canonical_json_bytes(tile_identity_payload(tile))
    ref = store.put(
        TILE_ARTIFACT_KIND,
        canonical_bytes,
        manifest={"tile_id": "sha256:" + "f" * 64, "lying": True},
    )

    loaded = load_tile(store, ref.artifact_id)
    assert loaded.tile_id == tile_id
    assert loaded.tile_id != "sha256:" + "f" * 64


def test_load_tile_rejects_a_digest_mismatch_from_a_misbehaving_resolver() -> None:
    """A resolver that returns bytes not matching the requested id (e.g. a
    corrupted or malicious backend) must be rejected, not trusted."""
    coverage = TileCoverageDTO(
        corpus_id="c", partition_key="root", child_count=1, leaf_count=1
    )
    pointer = TilePointerDTO(
        pointer_kind="leaf", target_id="sha256:" + "a" * 64, order_key="0"
    )
    tile_id = compute_tile_id(
        depth=0, coverage=coverage, children=(pointer,), tie_rule="lowest_order_key"
    )
    other_tile_id = compute_tile_id(
        depth=1, coverage=coverage, children=(pointer,), tie_rule="lowest_order_key"
    )
    other_tile = NavigationTileDTO(
        tile_id=other_tile_id,
        depth=1,
        coverage=coverage,
        children=(pointer,),
        tie_rule="lowest_order_key",
    )
    wrong_bytes = canonical_json_bytes(tile_identity_payload(other_tile))

    misbehaving_resolver = SimpleNamespace(
        has=lambda ref: True,
        resolve_canonical_bytes=lambda ref: wrong_bytes,
        resolve_manifest=lambda ref: {},
    )

    with pytest.raises(VPMValidationError, match="do not hash to the requested id"):
        load_tile(misbehaving_resolver, tile_id)


def test_load_tile_rejects_a_non_digest_tile_id() -> None:
    store = InMemoryArtifactStore()
    with pytest.raises(VPMValidationError, match="sha256"):
        load_tile(store, "not-a-digest")


def test_load_leaf_binding_rejects_a_non_digest_leaf_id() -> None:
    store = InMemoryArtifactStore()
    with pytest.raises(VPMValidationError, match="sha256"):
        load_leaf_binding(store, "not-a-digest")


def test_load_leaf_binding_ignores_a_manifest_that_lies_about_the_content(
    make_source_artifacts,
) -> None:
    store = InMemoryArtifactStore()
    ref_artifact = make_source_artifacts(1)[0]
    leaf_id = compute_leaf_id(
        artifact_ref=ref_artifact, leaf_semantics="terminal-policy-artifact"
    )
    binding = LeafBindingDTO(
        leaf_id=leaf_id,
        artifact_ref=ref_artifact,
        leaf_semantics="terminal-policy-artifact",
    )
    canonical_bytes = canonical_json_bytes(leaf_binding_identity_payload(binding))
    ref = store.put(
        LEAF_BINDING_ARTIFACT_KIND,
        canonical_bytes,
        manifest={
            "leaf_id": "sha256:" + "9" * 64,
            "artifact_ref": {
                "artifact_kind": "lying",
                "artifact_id": "sha256:" + "9" * 64,
            },
        },
    )

    loaded = load_leaf_binding(store, ref.artifact_id)
    assert loaded.leaf_id == leaf_id
    assert loaded.artifact_ref.artifact_kind == CORPUS_ARTIFACT_KIND
