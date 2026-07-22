"""Persist and resolve tiles/leaf bindings through the Artifacts protocol.

Navigation defines no store of its own. A tile or leaf binding's identity
digest (`tile_id` / `leaf_id`) is computed over its identity payload; the
same payload bytes are handed to the `ArtifactStore` as `canonical_bytes`
so the store's own content digest equals that identity exactly.

Loading is decode-and-verify, never trust-the-manifest: `load_tile` and
`load_leaf_binding` resolve canonical bytes, recompute the digest and
require it to equal the requested id, decode the canonical JSON payload
themselves, and reconstruct the DTO with the *requested* id (never an id
read back out of a manifest). `NavigationTileDTO`/`LeafBindingDTO`'s own
`__post_init__` then independently re-verifies that id against the
payload it was just given. The manifest each `store_*` call also writes
is kept only as non-authoritative, informational metadata (useful for a
real backend's lookup/indexing) - it is never read back to reconstruct a
DTO, so a manifest substitution at the storage layer cannot substitute a
different tile/leaf binding for the one actually requested.
"""

from __future__ import annotations

import json

from zeromodel.artifacts import (
    ArtifactRef,
    ArtifactResolver,
    ArtifactStore,
    canonical_json_bytes,
    is_sha256_digest,
    sha256_digest,
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


def _resolve_and_verify_canonical_payload(
    store: ArtifactResolver, ref: ArtifactRef
) -> dict:
    """Resolve canonical bytes for `ref`, verify their digest equals
    `ref.artifact_id`, and decode the canonical JSON payload.

    This is the only trusted path into a tile/leaf binding's content -
    never the manifest.
    """
    canonical_bytes = store.resolve_canonical_bytes(ref)
    actual_digest = sha256_digest(canonical_bytes)
    if actual_digest != ref.artifact_id:
        raise VPMValidationError(
            f"resolved canonical bytes for {ref.artifact_kind} {ref.artifact_id} "
            f"do not hash to the requested id (got {actual_digest})"
        )
    return json.loads(canonical_bytes)


def load_tile(store: ArtifactResolver, tile_id: str) -> NavigationTileDTO:
    if not is_sha256_digest(tile_id):
        raise VPMValidationError(f"tile_id must be a sha256: digest, got {tile_id!r}")
    ref = ArtifactRef(artifact_kind=TILE_ARTIFACT_KIND, artifact_id=tile_id)
    payload = _resolve_and_verify_canonical_payload(store, ref)
    coverage = TileCoverageDTO(**payload["coverage"])
    children = tuple(TilePointerDTO(**child) for child in payload["children"])
    # tile_id is the *requested* id, never a value read back out of stored
    # metadata; NavigationTileDTO.__post_init__ independently re-verifies
    # it against this exact payload.
    return NavigationTileDTO(
        tile_id=tile_id,
        depth=payload["depth"],
        coverage=coverage,
        children=children,
        tie_rule=payload["tie_rule"],
        spec_version=payload["spec_version"],
    )


def load_leaf_binding(store: ArtifactResolver, leaf_id: str) -> LeafBindingDTO:
    if not is_sha256_digest(leaf_id):
        raise VPMValidationError(f"leaf_id must be a sha256: digest, got {leaf_id!r}")
    ref = ArtifactRef(artifact_kind=LEAF_BINDING_ARTIFACT_KIND, artifact_id=leaf_id)
    payload = _resolve_and_verify_canonical_payload(store, ref)
    artifact_ref = ArtifactRef(
        artifact_kind=payload["artifact_kind"], artifact_id=payload["artifact_id"]
    )
    return LeafBindingDTO(
        leaf_id=leaf_id,
        artifact_ref=artifact_ref,
        leaf_semantics=payload["leaf_semantics"],
        spec_version=payload["spec_version"],
    )


def tile_exists(store: ArtifactResolver, tile_id: str) -> bool:
    return store.has(ArtifactRef(artifact_kind=TILE_ARTIFACT_KIND, artifact_id=tile_id))


def leaf_binding_exists(store: ArtifactResolver, leaf_id: str) -> bool:
    return store.has(
        ArtifactRef(artifact_kind=LEAF_BINDING_ARTIFACT_KIND, artifact_id=leaf_id)
    )
