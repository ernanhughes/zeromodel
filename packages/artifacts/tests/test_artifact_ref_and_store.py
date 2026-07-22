from __future__ import annotations

import pytest

from zeromodel.artifacts import (
    ArtifactManifestConflictError,
    ArtifactNotFoundError,
    ArtifactRef,
    InMemoryArtifactStore,
    canonical_json_bytes,
    sha256_digest,
)
from zeromodel.core.artifact import VPMValidationError


def test_artifact_ref_requires_a_sha256_digest() -> None:
    with pytest.raises(VPMValidationError):
        ArtifactRef(artifact_kind="zeromodel.core.vpm", artifact_id="not-a-digest")


def test_artifact_ref_requires_a_non_empty_kind() -> None:
    digest = sha256_digest(b"payload")
    with pytest.raises(VPMValidationError):
        ArtifactRef(artifact_kind="", artifact_id=digest)


def test_artifact_ref_rejects_unsupported_spec_version() -> None:
    digest = sha256_digest(b"payload")
    with pytest.raises(VPMValidationError):
        ArtifactRef(
            artifact_kind="zeromodel.core.vpm", artifact_id=digest, spec_version="v0"
        )


def test_artifact_ref_is_immutable() -> None:
    digest = sha256_digest(b"payload")
    ref = ArtifactRef(artifact_kind="zeromodel.core.vpm", artifact_id=digest)
    with pytest.raises(Exception):
        ref.artifact_id = sha256_digest(b"other")  # type: ignore[misc]


def test_store_put_then_resolve_round_trips_canonical_bytes() -> None:
    store = InMemoryArtifactStore()
    payload = canonical_json_bytes({"a": 1, "b": [1, 2, 3]})
    ref = store.put("zeromodel.navigation.tile", payload, manifest={"note": "test"})

    assert store.has(ref)
    assert store.resolve_canonical_bytes(ref) == payload
    assert store.resolve_manifest(ref) == {"note": "test"}


def test_store_ref_id_is_the_content_digest_not_an_arbitrary_label() -> None:
    store = InMemoryArtifactStore()
    payload = canonical_json_bytes({"x": 1})
    ref = store.put("zeromodel.trust.policy", payload)

    assert ref.artifact_id == sha256_digest(payload)


def test_store_resolving_an_unknown_ref_fails_closed() -> None:
    store = InMemoryArtifactStore()
    unknown = ArtifactRef(
        artifact_kind="zeromodel.trust.policy", artifact_id=sha256_digest(b"nope")
    )

    with pytest.raises(ArtifactNotFoundError):
        store.resolve_canonical_bytes(unknown)
    with pytest.raises(ArtifactNotFoundError):
        store.resolve_manifest(unknown)
    assert not store.has(unknown)


def test_store_resolving_the_right_digest_under_the_wrong_kind_fails_closed() -> None:
    store = InMemoryArtifactStore()
    payload = canonical_json_bytes({"x": 1})
    ref = store.put("zeromodel.trust.policy", payload)
    wrong_kind_ref = ArtifactRef(
        artifact_kind="zeromodel.navigation.tile", artifact_id=ref.artifact_id
    )

    with pytest.raises(ArtifactNotFoundError):
        store.resolve_canonical_bytes(wrong_kind_ref)


def test_store_put_is_idempotent_for_identical_payload_and_manifest() -> None:
    store = InMemoryArtifactStore()
    payload = canonical_json_bytes({"x": 1})
    manifest = {"note": "same"}

    first = store.put("zeromodel.trust.policy", payload, manifest=dict(manifest))
    second = store.put("zeromodel.trust.policy", payload, manifest=dict(manifest))

    assert first == second
    assert store.resolve_manifest(first) == manifest


def test_store_put_rejects_a_different_manifest_under_the_same_identity() -> None:
    store = InMemoryArtifactStore()
    payload = canonical_json_bytes({"x": 1})
    store.put("zeromodel.trust.policy", payload, manifest={"note": "original"})

    with pytest.raises(ArtifactManifestConflictError):
        store.put("zeromodel.trust.policy", payload, manifest={"note": "different"})

    # The original manifest must survive the rejected attempt untouched.
    ref = ArtifactRef(
        artifact_kind="zeromodel.trust.policy", artifact_id=sha256_digest(payload)
    )
    assert store.resolve_manifest(ref) == {"note": "original"}


def test_store_manifest_is_immutable_from_the_outside() -> None:
    store = InMemoryArtifactStore()
    manifest = {"note": "mutable-source-dict"}
    ref = store.put(
        "zeromodel.trust.policy", canonical_json_bytes({"x": 1}), manifest=manifest
    )
    manifest["note"] = "mutated-after-put"

    assert store.resolve_manifest(ref)["note"] == "mutable-source-dict"
    with pytest.raises(TypeError):
        store.resolve_manifest(ref)["note"] = "mutated-via-accessor"  # type: ignore[index]
