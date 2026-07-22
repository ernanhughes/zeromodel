"""Persist and resolve tiles/leaf bindings through the Artifacts protocol.

Navigation defines no store of its own. A tile or leaf binding's identity
digest (`tile_id` / `leaf_id`) is computed over its identity payload; the
same payload bytes are handed to the `ArtifactStore` as `canonical_bytes`
so the store's own content digest equals that identity exactly. The full
DTO content (including the id) is kept as the store's `manifest` for
resolution back into a DTO.
"""

from __future__ import annotations

from zeromodel.artifacts import (
    ArtifactRef,
    ArtifactResolver,
    ArtifactStore,
    canonical_json_bytes,
)
from zeromodel.core.artifact import VPMValidationError

from zeromodel.navigation.dto import (
    LeafBindingDTO,
    NavigationTileDTO,
    TileCoverageDTO,
    TilePointerDTO,
    leaf_binding_identity_payload,
    tile_identity_payload,
)

TILE_ARTIFACT_KIND = "navigation-tile"
LEAF_BINDING_ARTIFACT_KIND = "navigation-leaf-binding"


def store_tile(store: ArtifactStore, tile: NavigationTileDTO) -> ArtifactRef:
    canonical_bytes = canonical_json_bytes(tile_identity_payload(tile))
    manifest = {
        "tile_id": tile.tile_id,
        "depth": tile.depth,
        "coverage": {
            "corpus_id": tile.coverage.corpus_id,
            "partition_key": tile.coverage.partition_key,
            "child_count": tile.coverage.child_count,
            "leaf_count": tile.coverage.leaf_count,
            "spec_version": tile.coverage.spec_version,
        },
        "children": [
            {
                "pointer_kind": child.pointer_kind,
                "target_id": child.target_id,
                "order_key": child.order_key,
                "spec_version": child.spec_version,
            }
            for child in tile.children
        ],
        "tie_rule": tile.tie_rule,
        "spec_version": tile.spec_version,
    }
    ref = store.put(TILE_ARTIFACT_KIND, canonical_bytes, manifest=manifest)
    if ref.artifact_id != tile.tile_id:
        raise VPMValidationError(
            "stored tile digest does not match its declared tile_id"
        )
    return ref


def store_leaf_binding(store: ArtifactStore, binding: LeafBindingDTO) -> ArtifactRef:
    canonical_bytes = canonical_json_bytes(leaf_binding_identity_payload(binding))
    manifest = {
        "leaf_id": binding.leaf_id,
        "artifact_ref": {
            "artifact_kind": binding.artifact_ref.artifact_kind,
            "artifact_id": binding.artifact_ref.artifact_id,
            "spec_version": binding.artifact_ref.spec_version,
        },
        "leaf_semantics": binding.leaf_semantics,
        "spec_version": binding.spec_version,
    }
    ref = store.put(LEAF_BINDING_ARTIFACT_KIND, canonical_bytes, manifest=manifest)
    if ref.artifact_id != binding.leaf_id:
        raise VPMValidationError(
            "stored leaf binding digest does not match its declared leaf_id"
        )
    return ref


def load_tile(store: ArtifactResolver, tile_id: str) -> NavigationTileDTO:
    ref = ArtifactRef(artifact_kind=TILE_ARTIFACT_KIND, artifact_id=tile_id)
    manifest = store.resolve_manifest(ref)
    coverage = TileCoverageDTO(**manifest["coverage"])
    children = tuple(TilePointerDTO(**child) for child in manifest["children"])
    return NavigationTileDTO(
        tile_id=manifest["tile_id"],
        depth=manifest["depth"],
        coverage=coverage,
        children=children,
        tie_rule=manifest["tie_rule"],
        spec_version=manifest["spec_version"],
    )


def load_leaf_binding(store: ArtifactResolver, leaf_id: str) -> LeafBindingDTO:
    ref = ArtifactRef(artifact_kind=LEAF_BINDING_ARTIFACT_KIND, artifact_id=leaf_id)
    manifest = store.resolve_manifest(ref)
    artifact_ref = ArtifactRef(**manifest["artifact_ref"])
    return LeafBindingDTO(
        leaf_id=manifest["leaf_id"],
        artifact_ref=artifact_ref,
        leaf_semantics=manifest["leaf_semantics"],
        spec_version=manifest["spec_version"],
    )


def tile_exists(store: ArtifactResolver, tile_id: str) -> bool:
    return store.has(ArtifactRef(artifact_kind=TILE_ARTIFACT_KIND, artifact_id=tile_id))


def leaf_binding_exists(store: ArtifactResolver, leaf_id: str) -> bool:
    return store.has(
        ArtifactRef(artifact_kind=LEAF_BINDING_ARTIFACT_KIND, artifact_id=leaf_id)
    )
