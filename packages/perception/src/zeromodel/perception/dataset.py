"""Immutable perception dataset ledger for Stage P2.

The dataset layer records authoritative image/action pairings. It does not infer
alignment from filenames or directory order, and it does not perform learning.

P16B strengthens split ownership: every interaction in one sequence/episode is
assigned to the same partition, and identical source pixels may not cross partitions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Iterable, Mapping, Protocol, Sequence

from .representation import SourceVPMDTO, TargetVPMDTO

DATASET_MANIFEST_VERSION: Final = "perception-dataset-manifest/2"
INTERACTION_VERSION: Final = "perception-recorded-interaction/1"
SPLIT_ASSIGNMENT_VERSION: Final = "perception-split-assignment/2"

_ALLOWED_SPLITS: Final = ("train", "validation", "test")


class PerceptionDatasetError(ValueError):
    """Raised when a dataset cannot be represented without ambiguity."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class RecordedInteractionDTO:
    """One authoritative source/action pairing with optional next state."""

    interaction_id: str
    sequence_id: str
    step_index: int
    source_vpm_id: str
    target_vpm_id: str
    action_schema_id: str
    action_label: str
    source_pixel_digest: str
    next_source_vpm_id: str | None = None
    observed_at_ns: int | None = None
    version: str = INTERACTION_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("interaction_id", self.interaction_id),
            ("sequence_id", self.sequence_id),
            ("source_vpm_id", self.source_vpm_id),
            ("target_vpm_id", self.target_vpm_id),
            ("action_schema_id", self.action_schema_id),
            ("action_label", self.action_label),
            ("source_pixel_digest", self.source_pixel_digest),
            ("version", self.version),
        ):
            if not value:
                raise PerceptionDatasetError(f"{name} must be non-empty")
        if self.step_index < 0:
            raise PerceptionDatasetError("step_index must be non-negative")
        if self.observed_at_ns is not None and self.observed_at_ns < 0:
            raise PerceptionDatasetError("observed_at_ns must be non-negative")

    @classmethod
    def from_vpms(
        cls,
        *,
        sequence_id: str,
        step_index: int,
        source: SourceVPMDTO,
        target: TargetVPMDTO,
        next_source: SourceVPMDTO | None = None,
        observed_at_ns: int | None = None,
    ) -> "RecordedInteractionDTO":
        payload: Mapping[str, object] = {
            "action_label": target.action_label,
            "action_schema_id": target.action_schema_id,
            "next_source_vpm_id": None if next_source is None else next_source.source_vpm_id,
            "observed_at_ns": observed_at_ns,
            "sequence_id": sequence_id,
            "source_pixel_digest": source.pixel_digest,
            "source_vpm_id": source.source_vpm_id,
            "step_index": step_index,
            "target_vpm_id": target.target_vpm_id,
            "version": INTERACTION_VERSION,
        }
        return cls(interaction_id=_digest(_canonical_json(payload)), **payload)  # type: ignore[arg-type]

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "action_label": self.action_label,
            "action_schema_id": self.action_schema_id,
            "interaction_id": self.interaction_id,
            "next_source_vpm_id": self.next_source_vpm_id,
            "observed_at_ns": self.observed_at_ns,
            "sequence_id": self.sequence_id,
            "source_pixel_digest": self.source_pixel_digest,
            "source_vpm_id": self.source_vpm_id,
            "step_index": self.step_index,
            "target_vpm_id": self.target_vpm_id,
            "version": self.version,
        }


@dataclass(frozen=True)
class SplitAssignmentDTO:
    interaction_id: str
    split: str
    version: str = SPLIT_ASSIGNMENT_VERSION

    def __post_init__(self) -> None:
        if not self.interaction_id:
            raise PerceptionDatasetError("interaction_id must be non-empty")
        if self.split not in _ALLOWED_SPLITS:
            raise PerceptionDatasetError(
                f"split must be one of {_ALLOWED_SPLITS}, got {self.split!r}"
            )


@dataclass(frozen=True)
class DatasetFindingDTO:
    code: str
    severity: str
    interaction_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class PerceptionDatasetManifestDTO:
    dataset_id: str
    action_schema_id: str
    source_encoder_spec_ids: tuple[str, ...]
    interactions: tuple[RecordedInteractionDTO, ...]
    split_assignments: tuple[SplitAssignmentDTO, ...]
    findings: tuple[DatasetFindingDTO, ...]
    version: str = DATASET_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.action_schema_id:
            raise PerceptionDatasetError("dataset_id and action_schema_id must be non-empty")
        ids = tuple(item.interaction_id for item in self.interactions)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise PerceptionDatasetError(
                "interactions must be unique and sorted by interaction_id"
            )
        assignment_ids = tuple(item.interaction_id for item in self.split_assignments)
        if assignment_ids != ids:
            raise PerceptionDatasetError(
                "split assignments must be ordered one-to-one with interactions"
            )
        assignment_by_id = {
            item.interaction_id: item.split for item in self.split_assignments
        }
        sequence_splits: dict[str, set[str]] = {}
        for interaction in self.interactions:
            sequence_splits.setdefault(interaction.sequence_id, set()).add(
                assignment_by_id[interaction.interaction_id]
            )
        if any(len(splits) != 1 for splits in sequence_splits.values()):
            raise PerceptionDatasetError(
                "all interactions in one sequence must belong to one split"
            )
        if self.version != DATASET_MANIFEST_VERSION:
            raise PerceptionDatasetError("unsupported dataset manifest version")

    def split_for(self, interaction_id: str) -> str:
        for assignment in self.split_assignments:
            if assignment.interaction_id == interaction_id:
                return assignment.split
        raise KeyError(interaction_id)


class PerceptionDatasetStore(Protocol):
    def put(self, manifest: PerceptionDatasetManifestDTO) -> None: ...

    def get(self, dataset_id: str) -> PerceptionDatasetManifestDTO: ...

    def list_ids(self) -> tuple[str, ...]: ...


class InMemoryPerceptionDatasetStore:
    """Small DTO-only store used until persistence boundaries are promoted."""

    def __init__(self) -> None:
        self._manifests: dict[str, PerceptionDatasetManifestDTO] = {}

    def put(self, manifest: PerceptionDatasetManifestDTO) -> None:
        existing = self._manifests.get(manifest.dataset_id)
        if existing is not None and existing != manifest:
            raise PerceptionDatasetError(
                "dataset identity already exists with different content"
            )
        self._manifests[manifest.dataset_id] = manifest

    def get(self, dataset_id: str) -> PerceptionDatasetManifestDTO:
        try:
            return self._manifests[dataset_id]
        except KeyError as exc:
            raise KeyError(dataset_id) from exc

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._manifests))


def _split_name(split_owner_id: str, *, seed: str) -> str:
    value = int(
        hashlib.sha256(f"{seed}\0{split_owner_id}".encode("utf-8")).hexdigest()[:16],
        16,
    )
    bucket = value % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "validation"
    return "test"


def _findings(
    interactions: Sequence[RecordedInteractionDTO],
    assignments: Sequence[SplitAssignmentDTO],
) -> tuple[DatasetFindingDTO, ...]:
    findings: list[DatasetFindingDTO] = []
    by_pixel: dict[str, list[RecordedInteractionDTO]] = {}
    by_sequence: dict[str, list[RecordedInteractionDTO]] = {}
    assignment_by_id = {item.interaction_id: item.split for item in assignments}
    for item in interactions:
        by_pixel.setdefault(item.source_pixel_digest, []).append(item)
        by_sequence.setdefault(item.sequence_id, []).append(item)

    for pixel_digest, group in sorted(by_pixel.items()):
        labels = {item.action_label for item in group}
        if len(labels) > 1:
            findings.append(
                DatasetFindingDTO(
                    code="conflicting_actions_for_identical_source",
                    severity="error",
                    interaction_ids=tuple(sorted(item.interaction_id for item in group)),
                    detail=f"source pixel digest {pixel_digest} maps to actions {sorted(labels)}",
                )
            )
        splits = {assignment_by_id[item.interaction_id] for item in group}
        if len(splits) > 1:
            findings.append(
                DatasetFindingDTO(
                    code="identical_source_across_splits",
                    severity="error",
                    interaction_ids=tuple(sorted(item.interaction_id for item in group)),
                    detail=(
                        f"source pixel digest {pixel_digest} appears in partitions "
                        f"{sorted(splits)}"
                    ),
                )
            )

    for sequence_id, group in sorted(by_sequence.items()):
        ordered = sorted(group, key=lambda item: item.step_index)
        indices = [item.step_index for item in ordered]
        if len(indices) != len(set(indices)):
            findings.append(
                DatasetFindingDTO(
                    code="duplicate_sequence_step",
                    severity="error",
                    interaction_ids=tuple(sorted(item.interaction_id for item in group)),
                    detail=f"sequence {sequence_id!r} contains duplicate step indices",
                )
            )
        timestamps = [item.observed_at_ns for item in ordered if item.observed_at_ns is not None]
        if timestamps and timestamps != sorted(timestamps):
            findings.append(
                DatasetFindingDTO(
                    code="non_monotonic_sequence_time",
                    severity="error",
                    interaction_ids=tuple(item.interaction_id for item in ordered),
                    detail=f"sequence {sequence_id!r} timestamps are not monotonic",
                )
            )
        splits = {assignment_by_id[item.interaction_id] for item in group}
        if len(splits) > 1:
            findings.append(
                DatasetFindingDTO(
                    code="sequence_crosses_splits",
                    severity="error",
                    interaction_ids=tuple(sorted(item.interaction_id for item in group)),
                    detail=f"sequence {sequence_id!r} spans partitions {sorted(splits)}",
                )
            )
    return tuple(findings)


def build_dataset_manifest(
    interactions: Iterable[RecordedInteractionDTO],
    *,
    source_encoder_spec_ids: Iterable[str],
    split_seed: str = "zeromodel-perception/p16b",
    reject_errors: bool = True,
) -> PerceptionDatasetManifestDTO:
    """Validate pairings and build a sequence-owned content-addressed manifest."""

    ordered = tuple(sorted(interactions, key=lambda item: item.interaction_id))
    if not ordered:
        raise PerceptionDatasetError("dataset requires at least one interaction")
    ids = [item.interaction_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise PerceptionDatasetError("duplicate interaction identity")
    schema_ids = {item.action_schema_id for item in ordered}
    if len(schema_ids) != 1:
        raise PerceptionDatasetError("all interactions must use one action schema")
    encoder_ids = tuple(sorted(set(source_encoder_spec_ids)))
    if not encoder_ids or any(not value for value in encoder_ids):
        raise PerceptionDatasetError("source_encoder_spec_ids must be non-empty")
    if not split_seed:
        raise PerceptionDatasetError("split_seed must be non-empty")

    assignments = tuple(
        SplitAssignmentDTO(
            item.interaction_id,
            _split_name(item.sequence_id, seed=split_seed),
        )
        for item in ordered
    )
    findings = _findings(ordered, assignments)
    if reject_errors and any(item.severity == "error" for item in findings):
        codes = sorted({item.code for item in findings if item.severity == "error"})
        raise PerceptionDatasetError(f"dataset validation failed: {codes}")

    payload: Mapping[str, object] = {
        "action_schema_id": next(iter(schema_ids)),
        "findings": [
            {
                "code": item.code,
                "detail": item.detail,
                "interaction_ids": list(item.interaction_ids),
                "severity": item.severity,
            }
            for item in findings
        ],
        "interactions": [item.canonical_payload() for item in ordered],
        "source_encoder_spec_ids": list(encoder_ids),
        "split_assignments": [
            {"interaction_id": item.interaction_id, "split": item.split, "version": item.version}
            for item in assignments
        ],
        "split_owner": "sequence_id",
        "split_seed": split_seed,
        "version": DATASET_MANIFEST_VERSION,
    }
    return PerceptionDatasetManifestDTO(
        dataset_id=_digest(_canonical_json(payload)),
        action_schema_id=next(iter(schema_ids)),
        source_encoder_spec_ids=encoder_ids,
        interactions=ordered,
        split_assignments=assignments,
        findings=findings,
    )
