from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from zeromodel.core.artifact import VPMValidationError

from zeromodel.artifacts.canonicalization import sha256_digest
from zeromodel.artifacts.ref import ArtifactRef


class ArtifactNotFoundError(VPMValidationError):
    """Raised when a resolver/store has no record for the given ArtifactRef."""


class ArtifactIntegrityError(VPMValidationError):
    """Raised when stored content no longer matches its own declared digest."""


class ArtifactManifestConflictError(VPMValidationError):
    """Raised when a `put()` under an existing identity carries a different
    manifest than the one already stored.

    An `ArtifactRef`'s identity is immutable content-addressed data; nothing
    about that guarantees its *manifest* is immutable unless the store
    enforces it. Without this check, re-`put()`-ing the same canonical bytes
    with a different manifest would silently replace authoritative metadata
    (lineage, source bindings, contract identities) attached to the same
    reference - the store fails closed instead.
    """


@runtime_checkable
class ArtifactResolver(Protocol):
    """Read-side contract every artifact-aware package resolves through.

    Implementations must never silently substitute content: resolving a ref
    either returns bytes whose digest equals `ref.artifact_id`, or raises.
    """

    def has(self, ref: ArtifactRef) -> bool: ...

    def resolve_canonical_bytes(self, ref: ArtifactRef) -> bytes: ...

    def resolve_manifest(self, ref: ArtifactRef) -> Mapping[str, Any]: ...


@runtime_checkable
class ArtifactStore(ArtifactResolver, Protocol):
    """Write+read contract. `put` computes the ref; callers never invent one."""

    def put(
        self,
        artifact_kind: str,
        canonical_bytes: bytes,
        manifest: Mapping[str, Any] | None = None,
    ) -> ArtifactRef: ...


@dataclass(frozen=True)
class _StoredArtifact:
    ref: ArtifactRef
    canonical_bytes: bytes
    manifest: Mapping[str, Any]


class InMemoryArtifactStore:
    """Bounded, process-local content-addressed artifact store.

    A reference implementation for tests and single-process composition -
    not a distributed, durable, or persistent store. Navigation and Trust
    both persist their own DTOs through an `ArtifactStore` implementation
    like this one (or a real one backed by `zeromodel-sqlalchemy`) rather
    than defining their own repository.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], _StoredArtifact] = {}

    def put(
        self,
        artifact_kind: str,
        canonical_bytes: bytes,
        manifest: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        if not isinstance(canonical_bytes, (bytes, bytearray, memoryview)):
            raise VPMValidationError("canonical_bytes must be bytes-like")
        payload = bytes(canonical_bytes)
        digest = sha256_digest(payload)
        ref = ArtifactRef(artifact_kind=artifact_kind, artifact_id=digest)
        key = (ref.artifact_kind, ref.artifact_id)
        new_manifest = dict(manifest or {})
        existing = self._records.get(key)
        if existing is not None:
            if existing.canonical_bytes != payload:
                # sha256 collision within one artifact_kind - should be
                # unreachable in practice; fail closed rather than silently
                # keep the first write.
                raise ArtifactIntegrityError(
                    f"digest collision for {artifact_kind} {digest}: stored content differs"
                )
            if dict(existing.manifest) != new_manifest:
                raise ArtifactManifestConflictError(
                    f"manifest conflict for {artifact_kind} {digest}: a different "
                    "manifest is already stored under this artifact identity"
                )
            # Identical payload + identical manifest: idempotent no-op.
            return existing.ref
        self._records[key] = _StoredArtifact(
            ref=ref,
            canonical_bytes=payload,
            manifest=MappingProxyType(new_manifest),
        )
        return ref

    def has(self, ref: ArtifactRef) -> bool:
        return (ref.artifact_kind, ref.artifact_id) in self._records

    def resolve_canonical_bytes(self, ref: ArtifactRef) -> bytes:
        record = self._records.get((ref.artifact_kind, ref.artifact_id))
        if record is None:
            raise ArtifactNotFoundError(
                f"unknown artifact: kind={ref.artifact_kind} id={ref.artifact_id}"
            )
        return record.canonical_bytes

    def resolve_manifest(self, ref: ArtifactRef) -> Mapping[str, Any]:
        record = self._records.get((ref.artifact_kind, ref.artifact_id))
        if record is None:
            raise ArtifactNotFoundError(
                f"unknown artifact: kind={ref.artifact_kind} id={ref.artifact_id}"
            )
        return record.manifest
