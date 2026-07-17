"""Identity-bearing manifests and result contracts for visual-address research.

This module intentionally contains no encoder, training, or automatic mode
selection. It defines the evidence boundary required before any deterministic
or learned visual address artifact can be called validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import VPMValidationError


VISUAL_DATASET_MANIFEST_VERSION = "zeromodel-visual-dataset-manifest/v1"
VISUAL_BENCHMARK_VERSION = "zeromodel-visual-benchmark/v1"
VISUAL_BENCHMARK_REPORT_VERSION = "zeromodel-visual-benchmark-report/v1"

VISUAL_DATASET_SPLITS = ("prototype", "calibration", "test", "ood")
VISUAL_BENCHMARK_SYSTEMS = MappingProxyType(
    {
        "A": "current_deterministic_reader",
        "B": "normalized_template_matching",
        "C": "frozen_embedding_medoids",
        "D": "raw_embedding_knn",
        "G": "rejection_equipped_linear_probe",
        "H": "governance_parity_wrapper",
    }
)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        raise VPMValidationError(
            "visual benchmark JSON values must use plain JSON scalar types"
        )
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError(
            "visual benchmark values must be JSON-serializable"
        ) from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_rate(name: str, value: float) -> None:
    if not np.isfinite(float(value)) or not (0.0 <= float(value) <= 1.0):
        raise VPMValidationError("%s must be finite and in [0, 1]" % name)


@dataclass(frozen=True)
class CorruptionFamilySpec:
    """Declared observation-variation family used for split holdout."""

    family_id: str
    kind: str
    critical_evidence_removed: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        family_id: str,
        kind: str,
        critical_evidence_removed: bool = False,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> None:
        object.__setattr__(self, "family_id", str(family_id))
        object.__setattr__(self, "kind", str(kind))
        object.__setattr__(
            self,
            "critical_evidence_removed",
            bool(critical_evidence_removed),
        )
        object.__setattr__(self, "parameters", _freeze_json(parameters or {}))
        self.validate()

    def validate(self) -> None:
        if not self.family_id:
            raise VPMValidationError("corruption family_id cannot be empty")
        if not self.kind:
            raise VPMValidationError("corruption family kind cannot be empty")
        _canonical_json_bytes(self.parameters)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_id": self.family_id,
            "kind": self.kind,
            "critical_evidence_removed": self.critical_evidence_removed,
            "parameters": _thaw_json(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorruptionFamilySpec":
        return cls(
            family_id=str(data["family_id"]),
            kind=str(data["kind"]),
            critical_evidence_removed=bool(
                data.get("critical_evidence_removed", False)
            ),
            parameters=data.get("parameters") or {},
        )


@dataclass(frozen=True)
class VisualExampleRecord:
    """One content-addressed benchmark observation and its evaluation binding."""

    observation_id: str
    observation_digest: str
    split: str
    family_id: str
    row_id: Optional[str]
    action_id: Optional[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        observation_id: str,
        observation_digest: str,
        split: str,
        family_id: str,
        row_id: Optional[str] = None,
        action_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        object.__setattr__(self, "observation_id", str(observation_id))
        object.__setattr__(self, "observation_digest", str(observation_digest))
        object.__setattr__(self, "split", str(split))
        object.__setattr__(self, "family_id", str(family_id))
        object.__setattr__(
            self,
            "row_id",
            None if row_id is None else str(row_id),
        )
        object.__setattr__(
            self,
            "action_id",
            None if action_id is None else str(action_id),
        )
        object.__setattr__(self, "metadata", _freeze_json(metadata or {}))
        self.validate()

    def validate(self) -> None:
        for name, value in (
            ("observation_id", self.observation_id),
            ("observation_digest", self.observation_digest),
            ("family_id", self.family_id),
        ):
            if not value:
                raise VPMValidationError("%s cannot be empty" % name)
        if self.split not in VISUAL_DATASET_SPLITS:
            raise VPMValidationError(
                "visual example split must be one of: %s"
                % ", ".join(VISUAL_DATASET_SPLITS)
            )
        if self.split == "ood":
            if self.row_id is not None or self.action_id is not None:
                raise VPMValidationError(
                    "OOD examples cannot declare a valid row_id or action_id"
                )
        elif self.row_id is None or self.action_id is None:
            raise VPMValidationError(
                "non-OOD examples require row_id and action_id"
            )
        _canonical_json_bytes(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "observation_digest": self.observation_digest,
            "split": self.split,
            "family_id": self.family_id,
            "row_id": self.row_id,
            "action_id": self.action_id,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisualExampleRecord":
        return cls(
            observation_id=str(data["observation_id"]),
            observation_digest=str(data["observation_digest"]),
            split=str(data["split"]),
            family_id=str(data["family_id"]),
            row_id=data.get("row_id"),
            action_id=data.get("action_id"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True)
class VisualDatasetManifest:
    """Dataset identity with strict family-held-out split validation."""

    source_scope: str
    policy_artifact_id: str
    families: Tuple[CorruptionFamilySpec, ...]
    records: Tuple[VisualExampleRecord, ...]
    enforce_family_holdout: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VISUAL_DATASET_MANIFEST_VERSION

    def __init__(
        self,
        *,
        source_scope: str,
        policy_artifact_id: str,
        families: Sequence[CorruptionFamilySpec],
        records: Sequence[VisualExampleRecord],
        enforce_family_holdout: bool = True,
        metadata: Optional[Mapping[str, Any]] = None,
        version: str = VISUAL_DATASET_MANIFEST_VERSION,
    ) -> None:
        object.__setattr__(self, "source_scope", str(source_scope))
        object.__setattr__(self, "policy_artifact_id", str(policy_artifact_id))
        object.__setattr__(self, "families", tuple(families))
        object.__setattr__(self, "records", tuple(records))
        object.__setattr__(
            self,
            "enforce_family_holdout",
            bool(enforce_family_holdout),
        )
        object.__setattr__(self, "metadata", _freeze_json(metadata or {}))
        object.__setattr__(self, "version", str(version))
        self.validate()

    def validate(self) -> None:
        if self.version != VISUAL_DATASET_MANIFEST_VERSION:
            raise VPMValidationError(
                "unsupported visual dataset manifest version: %r" % self.version
            )
        if not self.source_scope:
            raise VPMValidationError("visual dataset source_scope cannot be empty")
        if not self.policy_artifact_id:
            raise VPMValidationError(
                "visual dataset policy_artifact_id cannot be empty"
            )
        if not self.families:
            raise VPMValidationError(
                "visual dataset must declare at least one corruption family"
            )
        if not self.records:
            raise VPMValidationError(
                "visual dataset must contain at least one observation record"
            )
        family_ids = [family.family_id for family in self.families]
        if len(set(family_ids)) != len(family_ids):
            raise VPMValidationError("visual dataset family ids must be unique")
        observation_ids = [record.observation_id for record in self.records]
        if len(set(observation_ids)) != len(observation_ids):
            raise VPMValidationError(
                "visual dataset observation ids must be unique"
            )
        family_id_set = set(family_ids)
        unknown_families = sorted(
            {record.family_id for record in self.records} - family_id_set
        )
        if unknown_families:
            raise VPMValidationError(
                "visual records reference unknown families: %s"
                % ", ".join(unknown_families)
            )
        required_splits = {"prototype", "calibration", "test"}
        present_splits = {record.split for record in self.records}
        missing_splits = sorted(required_splits - present_splits)
        if missing_splits:
            raise VPMValidationError(
                "visual dataset is missing required splits: %s"
                % ", ".join(missing_splits)
            )
        if self.enforce_family_holdout:
            splits_by_family: Dict[str, set[str]] = {}
            for record in self.records:
                splits_by_family.setdefault(record.family_id, set()).add(record.split)
            leaks = {
                family_id: sorted(splits)
                for family_id, splits in splits_by_family.items()
                if len(splits) > 1
            }
            if leaks:
                raise VPMValidationError(
                    "corruption families must be held out by split: %s"
                    % json.dumps(leaks, sort_keys=True)
                )
        row_sets = {
            split: {
                record.row_id
                for record in self.records
                if record.split == split and record.row_id is not None
            }
            for split in ("prototype", "calibration", "test")
        }
        if not (
            row_sets["prototype"]
            == row_sets["calibration"]
            == row_sets["test"]
        ):
            raise VPMValidationError(
                "prototype, calibration, and test splits must cover identical rows"
            )
        _canonical_json_bytes(self.metadata)

    @property
    def digest(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source_scope": self.source_scope,
            "policy_artifact_id": self.policy_artifact_id,
            "enforce_family_holdout": self.enforce_family_holdout,
            "families": [family.to_dict() for family in self.families],
            "records": [record.to_dict() for record in self.records],
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisualDatasetManifest":
        return cls(
            source_scope=str(data["source_scope"]),
            policy_artifact_id=str(data["policy_artifact_id"]),
            families=[
                CorruptionFamilySpec.from_dict(item)
                for item in data["families"]
            ],
            records=[
                VisualExampleRecord.from_dict(item) for item in data["records"]
            ],
            enforce_family_holdout=bool(
                data.get("enforce_family_holdout", True)
            ),
            metadata=data.get("metadata") or {},
            version=str(data.get("version", VISUAL_DATASET_MANIFEST_VERSION)),
        )
