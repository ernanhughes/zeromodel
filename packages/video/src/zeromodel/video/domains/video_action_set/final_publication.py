from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import ntpath
import os
from pathlib import Path
import re

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import (
    canonical_json_bytes,
    canonical_sha256,
)
from zeromodel.video.domains.video_action_set.final_access_dto import (
    FinalEvaluationResultDTO,
    FinalEvidenceBundleDTO,
    FinalExecutionReceiptDTO,
    FinalJsonDTO,
)


FINAL_ARTIFACT_MANIFEST_VERSION = "zeromodel-video-final-artifact-manifest/v1"
FINAL_ARTIFACT_MANIFEST_NAME = "final-artifact-manifest.json"
FINAL_EVIDENCE_NAME = "final-evidence.json"
FINAL_EVALUATION_NAME = "final-evaluation.json"
FINAL_RECEIPT_NAME = "final-execution-receipt.json"
ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
WINDOWS_RESERVED_ARTIFACT_STEMS = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
MANIFEST_KEYS = (
    "version",
    "access_id",
    "authorization_digest",
    "protocol_digest",
    "evidence_digest",
    "evaluation_digest",
    "execution_commit",
    "provider_order",
    "provider_versions",
    "expected_counts",
    "actual_counts",
    "files",
    "artifact_manifest_digest",
)
FILE_KEYS = frozenset({"filename", "sha256", "byte_length"})


@dataclass(frozen=True, slots=True)
class FinalArtifactManifestDTO:
    version: str
    access_id: str
    authorization_digest: str
    protocol_digest: str
    evidence_digest: str
    evaluation_digest: str
    execution_commit: str
    provider_order: tuple[str, ...]
    provider_versions: FinalJsonDTO
    expected_counts: FinalJsonDTO
    actual_counts: FinalJsonDTO
    files: FinalJsonDTO
    artifact_manifest_digest: str

    def __post_init__(self) -> None:
        if self.version != FINAL_ARTIFACT_MANIFEST_VERSION:
            raise VPMValidationError("final artifact manifest version mismatch")
        for value in (
            self.authorization_digest,
            self.protocol_digest,
            self.evidence_digest,
            self.evaluation_digest,
            self.artifact_manifest_digest,
        ):
            _require_digest(value, "final artifact manifest digest mismatch")
        files = _manifest_files(self.files.to_value())
        if files != sorted(files, key=lambda item: str(item["filename"])):
            raise VPMValidationError("final artifact manifest ordering mismatch")
        if len({str(item["filename"]).casefold() for item in files}) != len(files):
            raise VPMValidationError("duplicate canonical final artifact filename")
        if (
            canonical_sha256(
                {
                    key: value
                    for key, value in self.to_dict().items()
                    if key != "artifact_manifest_digest"
                }
            )
            != self.artifact_manifest_digest
        ):
            raise VPMValidationError("final artifact manifest digest mismatch")

    @classmethod
    def create(cls, payload: Mapping[str, object]) -> FinalArtifactManifestDTO:
        _require_exact_keys(
            payload,
            MANIFEST_KEYS[:-1],
            "final artifact manifest keys mismatch",
        )
        complete = dict(payload)
        complete["artifact_manifest_digest"] = canonical_sha256(payload)
        return cls.from_dict(complete)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FinalArtifactManifestDTO:
        _require_exact_keys(
            payload,
            MANIFEST_KEYS,
            "final artifact manifest keys mismatch",
        )
        provider_order = payload["provider_order"]
        if not isinstance(provider_order, list) or not provider_order:
            raise VPMValidationError("final artifact provider order mismatch")
        if any(not isinstance(item, str) or not item for item in provider_order):
            raise VPMValidationError("final artifact provider order mismatch")
        return cls(
            version=_string(
                payload["version"], "final artifact manifest version mismatch"
            ),
            access_id=_string(payload["access_id"], "final access id mismatch"),
            authorization_digest=_string(
                payload["authorization_digest"],
                "final artifact authorization mismatch",
            ),
            protocol_digest=_string(
                payload["protocol_digest"],
                "final artifact protocol mismatch",
            ),
            evidence_digest=_string(
                payload["evidence_digest"],
                "final artifact evidence mismatch",
            ),
            evaluation_digest=_string(
                payload["evaluation_digest"],
                "final artifact evaluation mismatch",
            ),
            execution_commit=_string(
                payload["execution_commit"],
                "final artifact execution commit mismatch",
            ),
            provider_order=tuple(provider_order),
            provider_versions=FinalJsonDTO.from_value(payload["provider_versions"]),
            expected_counts=FinalJsonDTO.from_value(payload["expected_counts"]),
            actual_counts=FinalJsonDTO.from_value(payload["actual_counts"]),
            files=FinalJsonDTO.from_value(payload["files"]),
            artifact_manifest_digest=_string(
                payload["artifact_manifest_digest"],
                "final artifact manifest digest mismatch",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "access_id": self.access_id,
            "authorization_digest": self.authorization_digest,
            "protocol_digest": self.protocol_digest,
            "evidence_digest": self.evidence_digest,
            "evaluation_digest": self.evaluation_digest,
            "execution_commit": self.execution_commit,
            "provider_order": list(self.provider_order),
            "provider_versions": self.provider_versions.to_value(),
            "expected_counts": self.expected_counts.to_value(),
            "actual_counts": self.actual_counts.to_value(),
            "files": self.files.to_value(),
            "artifact_manifest_digest": self.artifact_manifest_digest,
        }


def stage_final_artifacts(
    *,
    output_dir: Path,
    authorization_id: str,
    expected_executor_artifacts: Sequence[str],
    executor_artifacts: Mapping[str, bytes],
    evidence: FinalEvidenceBundleDTO,
    evaluation: FinalEvaluationResultDTO,
    before_validation: Callable[[], None] | None = None,
) -> tuple[Path, FinalArtifactManifestDTO]:
    output = _absolute_path(output_dir)
    _assert_no_symlink_components(output)
    if output.exists():
        raise VPMValidationError("canonical final artifacts already present")
    expected = _artifact_names(expected_executor_artifacts)
    if set(executor_artifacts) != set(expected):
        raise VPMValidationError("final artifact file set mismatch")
    if any(not isinstance(value, bytes) for value in executor_artifacts.values()):
        raise VPMValidationError("final artifact payload mismatch")
    staging = _staging_path(output, authorization_id)
    _assert_no_symlink_components(staging)
    if staging.exists():
        raise VPMValidationError("final staging directory already exists")
    staging.mkdir(parents=False)
    try:
        payloads = {
            FINAL_EVIDENCE_NAME: canonical_json_bytes(evidence.to_dict()),
            FINAL_EVALUATION_NAME: canonical_json_bytes(evaluation.to_dict()),
        } | dict(executor_artifacts)
        for filename, data in payloads.items():
            (staging / filename).write_bytes(data)
        files = [_file_record(staging / filename) for filename in sorted(payloads)]
        manifest = FinalArtifactManifestDTO.create(
            {
                "version": FINAL_ARTIFACT_MANIFEST_VERSION,
                "access_id": evidence.access_id,
                "authorization_digest": evidence.authorization_digest,
                "protocol_digest": evidence.protocol_digest,
                "evidence_digest": evidence.evidence_digest,
                "evaluation_digest": evaluation.evaluation_digest,
                "execution_commit": evidence.execution_commit,
                "provider_order": list(evidence.provider_order),
                "provider_versions": evidence.provider_versions.to_value(),
                "expected_counts": evidence.expected_counts.to_value(),
                "actual_counts": evidence.actual_counts.to_value(),
                "files": files,
            }
        )
        (staging / FINAL_ARTIFACT_MANIFEST_NAME).write_bytes(
            canonical_json_bytes(manifest.to_dict())
        )
        if before_validation is not None:
            before_validation()
        validate_staged_artifacts(staging, manifest)
        return staging, manifest
    except Exception:
        _remove_empty_or_files(staging)
        raise


def validate_staged_artifacts(
    staging_dir: Path,
    manifest: FinalArtifactManifestDTO,
    *,
    allow_receipt: bool = False,
) -> None:
    staging = _absolute_path(staging_dir)
    _assert_no_symlink_components(staging)
    if not staging.is_dir() or staging.is_symlink():
        raise VPMValidationError("final staging directory mismatch")
    expected = {
        str(item["filename"]) for item in _manifest_files(manifest.files.to_value())
    } | {FINAL_ARTIFACT_MANIFEST_NAME}
    actual_paths = tuple(staging.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in actual_paths):
        raise VPMValidationError("symlinked final staging path")
    actual_names = {path.name for path in actual_paths}
    if allow_receipt:
        actual_names.discard(FINAL_RECEIPT_NAME)
    if actual_names != expected:
        raise VPMValidationError("final artifact file set mismatch")
    for item in _manifest_files(manifest.files.to_value()):
        path = staging / str(item["filename"])
        if _file_record(path) != item:
            raise VPMValidationError("final artifact manifest validation mismatch")
    manifest_path = staging / FINAL_ARTIFACT_MANIFEST_NAME
    loaded = FinalArtifactManifestDTO.from_dict(_json_mapping(manifest_path))
    if loaded != manifest:
        raise VPMValidationError("final artifact manifest validation mismatch")


def promote_staged_artifacts(staging_dir: Path, output_dir: Path) -> None:
    staging = _absolute_path(staging_dir)
    output = _absolute_path(output_dir)
    _assert_no_symlink_components(staging)
    _assert_no_symlink_components(output)
    if staging.anchor.casefold() != output.anchor.casefold():
        raise VPMValidationError("final staging and output volumes differ")
    if output.exists():
        raise VPMValidationError("canonical final artifacts already present")
    os.replace(staging, output)


def validate_canonical_artifacts(
    output_dir: Path,
    manifest: FinalArtifactManifestDTO,
) -> None:
    validate_staged_artifacts(output_dir, manifest, allow_receipt=True)


def load_canonical_artifact_manifest(
    output_dir: Path,
) -> FinalArtifactManifestDTO | None:
    output = _absolute_path(output_dir)
    _assert_no_symlink_components(output)
    path = output / FINAL_ARTIFACT_MANIFEST_NAME
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise VPMValidationError("final artifact manifest path mismatch")
    manifest = FinalArtifactManifestDTO.from_dict(_json_mapping(path))
    validate_canonical_artifacts(output, manifest)
    return manifest


def publish_receipt_last(
    *,
    output_dir: Path,
    authorization_id: str,
    receipt: FinalExecutionReceiptDTO,
    before_write: Callable[[], None] | None = None,
    after_write: Callable[[], None] | None = None,
    before_publish: Callable[[], None] | None = None,
    before_rename: Callable[[], None] | None = None,
    during_rename: Callable[[], None] | None = None,
    after_publish: Callable[[], None] | None = None,
) -> Path:
    output = _absolute_path(output_dir)
    _assert_no_symlink_components(output)
    destination = output / FINAL_RECEIPT_NAME
    if destination.exists() or destination.is_symlink():
        raise VPMValidationError("final receipt already exists")
    token = canonical_sha256(authorization_id)[7:23]
    temporary = output.parent / f".{output.name}.receipt-{token}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise VPMValidationError("final receipt staging file already exists")
    try:
        if before_write is not None:
            before_write()
        temporary.write_bytes(canonical_json_bytes(receipt.to_dict()))
        if after_write is not None:
            after_write()
        loaded = FinalExecutionReceiptDTO.from_dict(_json_mapping(temporary))
        if loaded != receipt:
            raise VPMValidationError("final receipt staging validation mismatch")
        if before_publish is not None:
            before_publish()
        if before_rename is not None:
            before_rename()
        if during_rename is not None:
            during_rename()
        os.replace(temporary, destination)
        if after_publish is not None:
            after_publish()
    except Exception:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return destination


def load_published_receipt(output_dir: Path) -> FinalExecutionReceiptDTO | None:
    path = _absolute_path(output_dir) / FINAL_RECEIPT_NAME
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise VPMValidationError("final receipt path mismatch")
    return FinalExecutionReceiptDTO.from_dict(_json_mapping(path))


def _staging_path(output_dir: Path, authorization_id: str) -> Path:
    token = canonical_sha256(authorization_id)[7:23]
    return output_dir.parent / f".{output_dir.name}.final-staging-{token}"


def _artifact_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(values)
    if len({name.casefold() for name in names}) != len(names):
        raise VPMValidationError("duplicate canonical final artifact filename")
    reserved = {
        FINAL_ARTIFACT_MANIFEST_NAME,
        FINAL_EVIDENCE_NAME,
        FINAL_EVALUATION_NAME,
        FINAL_RECEIPT_NAME,
    }
    for name in names:
        validate_final_artifact_filename(name)
        if name.casefold() in {item.casefold() for item in reserved}:
            raise VPMValidationError("final artifact filename mismatch")
    return names


def _manifest_files(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise VPMValidationError("final artifact manifest files mismatch")
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != FILE_KEYS:
            raise VPMValidationError("final artifact manifest files mismatch")
        filename = item["filename"]
        digest = item["sha256"]
        byte_length = item["byte_length"]
        validate_final_artifact_filename(filename)
        _require_digest(digest, "final artifact file digest mismatch")
        if (
            not isinstance(byte_length, int)
            or isinstance(byte_length, bool)
            or byte_length < 0
        ):
            raise VPMValidationError("final artifact byte length mismatch")
        result.append(dict(item))
    return result


def validate_final_artifact_filename(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise VPMValidationError("final artifact filename mismatch")
    if (
        value in {".", ".."}
        or value.endswith((".", " "))
        or "/" in value
        or "\\" in value
        or ":" in value
        or ntpath.isabs(value)
        or bool(ntpath.splitdrive(value)[0])
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or ARTIFACT_NAME_RE.fullmatch(value) is None
    ):
        raise VPMValidationError("final artifact filename mismatch")
    normalized = value.rstrip(". ")
    stem = normalized.split(".", 1)[0].casefold()
    if stem in WINDOWS_RESERVED_ARTIFACT_STEMS:
        raise VPMValidationError("final artifact filename mismatch")
    return value


def _file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "filename": path.name,
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "byte_length": len(data),
    }


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise VPMValidationError("symlinked final path")


def _remove_empty_or_files(path: Path) -> None:
    if not path.exists() or path.is_symlink():
        return
    for child in path.iterdir():
        if child.is_file() and not child.is_symlink():
            child.unlink()
    try:
        path.rmdir()
    except OSError:
        pass


def _json_mapping(path: Path) -> Mapping[str, object]:
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VPMValidationError("final JSON artifact mismatch") from exc
    if not isinstance(value, Mapping):
        raise VPMValidationError("final JSON artifact mismatch")
    return value


def _string(value: object, message: str) -> str:
    if not isinstance(value, str) or not value:
        raise VPMValidationError(message)
    return value


def _require_digest(value: object, message: str) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
    ):
        raise VPMValidationError(message)


def _require_exact_keys(
    payload: Mapping[str, object],
    keys: Sequence[str],
    message: str,
) -> None:
    if set(payload) != set(keys):
        raise VPMValidationError(message)


__all__ = [
    "FINAL_ARTIFACT_MANIFEST_NAME",
    "FINAL_ARTIFACT_MANIFEST_VERSION",
    "FINAL_RECEIPT_NAME",
    "WINDOWS_RESERVED_ARTIFACT_STEMS",
    "FinalArtifactManifestDTO",
    "load_canonical_artifact_manifest",
    "load_published_receipt",
    "promote_staged_artifacts",
    "publish_receipt_last",
    "stage_final_artifacts",
    "validate_canonical_artifacts",
    "validate_final_artifact_filename",
    "validate_staged_artifacts",
]
