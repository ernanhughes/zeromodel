from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from ...artifact import VPMValidationError
from .canonical_json import canonical_json_bytes, canonical_sha256
from .final_access_dto import validate_final_identifier


HISTORICAL_EVIDENCE_MANIFEST_VERSION = "zeromodel-video-historical-evidence-manifest/v1"
VERIFIED_HISTORICAL_AUTHORITY_VERSION = (
    "zeromodel-video-verified-historical-authority/v1"
)
HISTORICAL_AUTHORITY_KEYS = frozenset(
    {
        "version",
        "historical_authority_id",
        "historical_database_path",
        "historical_database_sha256",
        "evidence_manifest_path",
        "evidence_manifest_digest",
        "stage8_commit",
    }
)
HISTORICAL_EVIDENCE_MANIFEST_KEYS = (
    "version",
    "historical_authority_id",
    "historical_database_path",
    "historical_database_sha256",
    "stage8_commit",
    "evidence_manifest_digest",
)
VERIFIED_HISTORICAL_AUTHORITY_KEYS = (
    "version",
    "historical_authority_version",
    "historical_authority_id",
    "historical_database_path",
    "historical_database_sha256",
    "evidence_manifest_path",
    "evidence_manifest_digest",
    "stage8_commit",
    "historical_authority_digest",
)
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class HistoricalEvidenceManifestDTO:
    version: str
    historical_authority_id: str
    historical_database_path: str
    historical_database_sha256: str
    stage8_commit: str
    evidence_manifest_digest: str

    def __post_init__(self) -> None:
        if self.version != HISTORICAL_EVIDENCE_MANIFEST_VERSION:
            raise VPMValidationError("historical evidence manifest version mismatch")
        validate_final_identifier(
            self.historical_authority_id,
            "historical authority id mismatch",
        )
        _required_string(
            self.historical_database_path,
            "historical evidence database path mismatch",
        )
        _required_digest(
            self.historical_database_sha256,
            "historical evidence database digest mismatch",
        )
        _required_string(self.stage8_commit, "historical Stage 8 commit mismatch")
        _required_digest(
            self.evidence_manifest_digest,
            "historical evidence manifest digest mismatch",
        )
        if (
            canonical_sha256(
                {
                    key: value
                    for key, value in self.to_dict().items()
                    if key != "evidence_manifest_digest"
                }
            )
            != self.evidence_manifest_digest
        ):
            raise VPMValidationError("historical evidence manifest digest mismatch")

    @classmethod
    def create(
        cls,
        payload: Mapping[str, object],
    ) -> HistoricalEvidenceManifestDTO:
        _exact_keys(
            payload,
            HISTORICAL_EVIDENCE_MANIFEST_KEYS[:-1],
            "historical evidence manifest keys mismatch",
        )
        return cls.from_dict(
            dict(payload) | {"evidence_manifest_digest": canonical_sha256(payload)}
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> HistoricalEvidenceManifestDTO:
        _exact_keys(
            payload,
            HISTORICAL_EVIDENCE_MANIFEST_KEYS,
            "historical evidence manifest keys mismatch",
        )
        return cls(
            version=_required_string(
                payload["version"],
                "historical evidence manifest version mismatch",
            ),
            historical_authority_id=_required_string(
                payload["historical_authority_id"],
                "historical authority id mismatch",
            ),
            historical_database_path=_required_string(
                payload["historical_database_path"],
                "historical evidence database path mismatch",
            ),
            historical_database_sha256=_required_string(
                payload["historical_database_sha256"],
                "historical evidence database digest mismatch",
            ),
            stage8_commit=_required_string(
                payload["stage8_commit"],
                "historical Stage 8 commit mismatch",
            ),
            evidence_manifest_digest=_required_string(
                payload["evidence_manifest_digest"],
                "historical evidence manifest digest mismatch",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "historical_authority_id": self.historical_authority_id,
            "historical_database_path": self.historical_database_path,
            "historical_database_sha256": self.historical_database_sha256,
            "stage8_commit": self.stage8_commit,
            "evidence_manifest_digest": self.evidence_manifest_digest,
        }


@dataclass(frozen=True, slots=True)
class VerifiedHistoricalAuthorityDTO:
    version: str
    historical_authority_version: str
    historical_authority_id: str
    historical_database_path: str
    historical_database_sha256: str
    evidence_manifest_path: str
    evidence_manifest_digest: str
    stage8_commit: str
    historical_authority_digest: str

    def __post_init__(self) -> None:
        if self.version != VERIFIED_HISTORICAL_AUTHORITY_VERSION:
            raise VPMValidationError("verified historical authority version mismatch")
        for value, message in (
            (
                self.historical_authority_version,
                "historical authority version mismatch",
            ),
            (self.historical_database_path, "historical database path mismatch"),
            (self.evidence_manifest_path, "historical evidence manifest path mismatch"),
            (self.stage8_commit, "historical Stage 8 commit mismatch"),
        ):
            _required_string(value, message)
        validate_final_identifier(
            self.historical_authority_id,
            "historical authority id mismatch",
        )
        for value, message in (
            (
                self.historical_database_sha256,
                "historical database digest mismatch",
            ),
            (
                self.evidence_manifest_digest,
                "historical evidence manifest digest mismatch",
            ),
            (
                self.historical_authority_digest,
                "verified historical authority digest mismatch",
            ),
        ):
            _required_digest(value, message)
        if (
            canonical_sha256(
                {
                    key: value
                    for key, value in self.to_dict().items()
                    if key != "historical_authority_digest"
                }
            )
            != self.historical_authority_digest
        ):
            raise VPMValidationError("verified historical authority digest mismatch")

    @classmethod
    def create(
        cls,
        payload: Mapping[str, object],
    ) -> VerifiedHistoricalAuthorityDTO:
        _exact_keys(
            payload,
            VERIFIED_HISTORICAL_AUTHORITY_KEYS[:-1],
            "verified historical authority keys mismatch",
        )
        return cls.from_dict(
            dict(payload) | {"historical_authority_digest": canonical_sha256(payload)}
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> VerifiedHistoricalAuthorityDTO:
        _exact_keys(
            payload,
            VERIFIED_HISTORICAL_AUTHORITY_KEYS,
            "verified historical authority keys mismatch",
        )
        return cls(
            version=_required_string(
                payload["version"],
                "verified historical authority version mismatch",
            ),
            historical_authority_version=_required_string(
                payload["historical_authority_version"],
                "historical authority version mismatch",
            ),
            historical_authority_id=_required_string(
                payload["historical_authority_id"],
                "historical authority id mismatch",
            ),
            historical_database_path=_required_string(
                payload["historical_database_path"],
                "historical database path mismatch",
            ),
            historical_database_sha256=_required_string(
                payload["historical_database_sha256"],
                "historical database digest mismatch",
            ),
            evidence_manifest_path=_required_string(
                payload["evidence_manifest_path"],
                "historical evidence manifest path mismatch",
            ),
            evidence_manifest_digest=_required_string(
                payload["evidence_manifest_digest"],
                "historical evidence manifest digest mismatch",
            ),
            stage8_commit=_required_string(
                payload["stage8_commit"],
                "historical Stage 8 commit mismatch",
            ),
            historical_authority_digest=_required_string(
                payload["historical_authority_digest"],
                "verified historical authority digest mismatch",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "historical_authority_version": self.historical_authority_version,
            "historical_authority_id": self.historical_authority_id,
            "historical_database_path": self.historical_database_path,
            "historical_database_sha256": self.historical_database_sha256,
            "evidence_manifest_path": self.evidence_manifest_path,
            "evidence_manifest_digest": self.evidence_manifest_digest,
            "stage8_commit": self.stage8_commit,
            "historical_authority_digest": self.historical_authority_digest,
        }


def verify_historical_authority(
    declaration: Mapping[str, object],
) -> VerifiedHistoricalAuthorityDTO:
    """Recompute and bind the declared Stage 8 database and evidence manifest."""

    _exact_keys(
        declaration,
        HISTORICAL_AUTHORITY_KEYS,
        "final historical authority mismatch",
    )
    authority_version = _required_string(
        declaration["version"],
        "historical authority version mismatch",
    )
    authority_id = _required_string(
        declaration["historical_authority_id"],
        "historical authority id mismatch",
    )
    validate_final_identifier(authority_id, "historical authority id mismatch")
    stage8_commit = _required_string(
        declaration["stage8_commit"],
        "historical Stage 8 commit mismatch",
    )
    declared_database_digest = _required_digest(
        declaration["historical_database_sha256"],
        "historical database digest mismatch",
    )
    declared_manifest_digest = _required_digest(
        declaration["evidence_manifest_digest"],
        "historical evidence manifest digest mismatch",
    )

    database_path = _resolve_regular_file(
        declaration["historical_database_path"],
        "historical database path mismatch",
    )
    actual_database_digest = _sha256_file(database_path)
    if actual_database_digest != declared_database_digest:
        raise VPMValidationError("historical database digest mismatch")

    manifest_path = _resolve_regular_file(
        declaration["evidence_manifest_path"],
        "historical evidence manifest path mismatch",
    )
    manifest = _load_manifest(manifest_path)
    manifest_database_path = _resolve_regular_file(
        manifest.historical_database_path,
        "historical evidence database path mismatch",
    )
    if (
        manifest.historical_authority_id != authority_id
        or manifest_database_path != database_path
        or manifest.historical_database_sha256 != actual_database_digest
        or manifest.historical_database_sha256 != declared_database_digest
        or manifest.stage8_commit != stage8_commit
        or manifest.evidence_manifest_digest != declared_manifest_digest
    ):
        raise VPMValidationError("historical evidence manifest binding mismatch")

    return VerifiedHistoricalAuthorityDTO.create(
        {
            "version": VERIFIED_HISTORICAL_AUTHORITY_VERSION,
            "historical_authority_version": authority_version,
            "historical_authority_id": authority_id,
            "historical_database_path": str(database_path),
            "historical_database_sha256": actual_database_digest,
            "evidence_manifest_path": str(manifest_path),
            "evidence_manifest_digest": manifest.evidence_manifest_digest,
            "stage8_commit": stage8_commit,
        }
    )


def _load_manifest(path: Path) -> HistoricalEvidenceManifestDTO:
    try:
        data = path.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VPMValidationError("historical evidence manifest mismatch") from exc
    if not isinstance(payload, Mapping):
        raise VPMValidationError("historical evidence manifest mismatch")
    manifest = HistoricalEvidenceManifestDTO.from_dict(payload)
    if data != canonical_json_bytes(manifest.to_dict()):
        raise VPMValidationError("historical evidence manifest is not canonical")
    return manifest


def _resolve_regular_file(value: object, message: str) -> Path:
    raw = _required_string(value, message)
    declared = Path(raw)
    if not declared.is_absolute():
        raise VPMValidationError(message)
    absolute = Path(os.path.abspath(declared))
    _assert_no_link_components(absolute, message)
    try:
        resolved = absolute.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise VPMValidationError(message) from exc
    _assert_no_link_components(resolved, message)
    if not stat.S_ISREG(metadata.st_mode):
        raise VPMValidationError(message)
    return resolved


def _assert_no_link_components(path: Path, message: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            raise VPMValidationError(message)


def _sha256_file(path: Path) -> str:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise VPMValidationError("historical database read mismatch") from exc
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise VPMValidationError("historical database changed during verification")
    return "sha256:" + digest.hexdigest()


def _required_string(value: object, message: str) -> str:
    if not isinstance(value, str) or not value:
        raise VPMValidationError(message)
    return value


def _required_digest(value: object, message: str) -> str:
    text = _required_string(value, message)
    if DIGEST_RE.fullmatch(text) is None:
        raise VPMValidationError(message)
    return text


def _exact_keys(
    payload: Mapping[str, object],
    keys: object,
    message: str,
) -> None:
    if set(payload) != set(keys):  # type: ignore[arg-type]
        raise VPMValidationError(message)


__all__ = [
    "HISTORICAL_AUTHORITY_KEYS",
    "HISTORICAL_EVIDENCE_MANIFEST_VERSION",
    "HistoricalEvidenceManifestDTO",
    "VERIFIED_HISTORICAL_AUTHORITY_VERSION",
    "VerifiedHistoricalAuthorityDTO",
    "verify_historical_authority",
]
