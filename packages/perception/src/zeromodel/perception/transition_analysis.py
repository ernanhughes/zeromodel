"""Composition identity for bounded visual transition analysis.

This module intentionally does not infer causality.  It binds exact
before/action/after transition evidence to the declared expectation set and
conformance report under which the observation was evaluated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from types import MappingProxyType
from typing import Any, Final, Mapping

from .transition_conformance import (
    TransitionConformanceFindingDTO,
    TransitionConformanceReportDTO,
    TransitionExpectationDTO,
)
from .transition_evidence import TransitionEvidenceVPMDTO

TRANSITION_ACTION_DECLARATION_VERSION: Final = (
    "perception-transition-action-declaration/1"
)
TRANSITION_EXPECTATION_SET_VERSION: Final = "perception-transition-expectation-set/1"
VISUAL_TRANSITION_READER_TRACE_VERSION: Final = (
    "perception-visual-transition-reader-trace/1"
)
VISUAL_TRANSITION_ANALYSIS_VERSION: Final = "perception-visual-transition-analysis/1"

VISUAL_TRANSITION_ANALYSIS_STATUSES: Final = {
    "conformant",
    "attention_required",
    "nonconformant",
}
VISUAL_READER_ACCEPTANCE_PROFILES: Final = {
    "canonical_only",
    "exact_codeword",
    "calibrated_nearest",
    "evidence_only",
}


class PerceptionTransitionAnalysisError(ValueError):
    """Raised when transition-analysis composition is malformed."""


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise PerceptionTransitionAnalysisError(
        "transition action payload must contain only JSON scalar/container values"
    )


def _json(payload: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            _thaw(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PerceptionTransitionAnalysisError(
            "transition-analysis payload must be canonical JSON"
        ) from exc


def _digest(payload: Mapping[str, object]) -> str:
    value = _json(payload)
    hasher = hashlib.sha256()
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)
    return f"sha256:{hasher.hexdigest()}"


def _payload(value: object, identity_field: str) -> dict[str, object]:
    payload = {field.name: _thaw(getattr(value, field.name)) for field in fields(value)}
    payload.pop(identity_field)
    return payload


def _ordered(name: str, values: tuple[str, ...]) -> None:
    if values != tuple(sorted(set(values))):
        raise PerceptionTransitionAnalysisError(f"{name} must be unique and sorted")


@dataclass(frozen=True)
class TransitionActionDeclarationDTO:
    """Declared command under which a transition is evaluated."""

    action_id: str
    action_type: str
    payload: Mapping[str, Any]
    schema_version: str = "1"
    provider_id: str | None = None
    version: str = TRANSITION_ACTION_DECLARATION_VERSION

    def __post_init__(self) -> None:
        if not self.action_id or not self.action_type or not self.schema_version:
            raise PerceptionTransitionAnalysisError(
                "action identity, type, and schema_version must be non-empty"
            )
        object.__setattr__(self, "payload", _freeze(self.payload))
        if self.provider_id is not None:
            object.__setattr__(self, "provider_id", str(self.provider_id))
        if self.version != TRANSITION_ACTION_DECLARATION_VERSION:
            raise PerceptionTransitionAnalysisError(
                "unsupported transition action declaration version"
            )
        if self.action_id != _digest(_payload(self, "action_id")):
            raise PerceptionTransitionAnalysisError(
                "action identity disagrees with canonical payload"
            )

    @property
    def action_digest(self) -> str:
        return self.action_id

    @classmethod
    def create(
        cls,
        *,
        action_type: str,
        payload: Mapping[str, Any] | None = None,
        schema_version: str = "1",
        provider_id: str | None = None,
    ) -> "TransitionActionDeclarationDTO":
        values: dict[str, object] = {
            "action_type": str(action_type),
            "payload": _freeze(payload or {}),
            "schema_version": str(schema_version),
            "provider_id": provider_id,
            "version": TRANSITION_ACTION_DECLARATION_VERSION,
        }
        return cls(action_id=_digest(values), **values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "payload": _thaw(self.payload),
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TransitionActionDeclarationDTO":
        return cls(
            action_id=str(data["action_id"]),
            action_type=str(data["action_type"]),
            payload=data.get("payload") or {},
            schema_version=str(data.get("schema_version", "1")),
            provider_id=(
                None if data.get("provider_id") is None else str(data["provider_id"])
            ),
            version=str(data.get("version", TRANSITION_ACTION_DECLARATION_VERSION)),
        )


@dataclass(frozen=True)
class TransitionExpectationSetDTO:
    """Canonical identity for the exact expectation set used in evaluation."""

    expectation_set_id: str
    field_schema_id: str
    expectations: tuple[TransitionExpectationDTO, ...]
    version: str = TRANSITION_EXPECTATION_SET_VERSION

    def __post_init__(self) -> None:
        if not self.expectation_set_id or not self.field_schema_id:
            raise PerceptionTransitionAnalysisError(
                "expectation set identities must be non-empty"
            )
        if not self.expectations:
            raise PerceptionTransitionAnalysisError(
                "expectation set requires at least one expectation"
            )
        if self.version != TRANSITION_EXPECTATION_SET_VERSION:
            raise PerceptionTransitionAnalysisError(
                "unsupported transition expectation set version"
            )
        expectation_ids = tuple(item.expectation_id for item in self.expectations)
        _ordered("expectation_ids", expectation_ids)
        targets: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for item in self.expectations:
            if item.field_schema_id != self.field_schema_id:
                raise PerceptionTransitionAnalysisError(
                    "expectation field schema does not match expectation set"
                )
            target = (item.annotation_ids, item.relation_ids)
            if target in targets:
                raise PerceptionTransitionAnalysisError(
                    "conflicting expectations target the same declarations"
                )
            targets.add(target)
        if self.expectation_set_id != _digest(self.canonical_payload()):
            raise PerceptionTransitionAnalysisError(
                "expectation set identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        expectations: tuple[TransitionExpectationDTO, ...],
    ) -> "TransitionExpectationSetDTO":
        if not expectations:
            raise PerceptionTransitionAnalysisError(
                "expectation set requires at least one expectation"
            )
        field_schema_id = expectations[0].field_schema_id
        ordered = tuple(sorted(expectations, key=lambda item: item.expectation_id))
        values = {
            "field_schema_id": field_schema_id,
            "expectations": [item.expectation_id for item in ordered],
            "version": TRANSITION_EXPECTATION_SET_VERSION,
        }
        return cls(
            expectation_set_id=_digest(values),
            field_schema_id=field_schema_id,
            expectations=ordered,
        )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "field_schema_id": self.field_schema_id,
            "expectations": [item.expectation_id for item in self.expectations],
            "version": self.version,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "expectation_set_id": self.expectation_set_id,
            "field_schema_id": self.field_schema_id,
            "expectations": [_thaw(asdict(item)) for item in self.expectations],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TransitionExpectationSetDTO":
        return cls(
            expectation_set_id=str(data["expectation_set_id"]),
            field_schema_id=str(data["field_schema_id"]),
            expectations=tuple(
                TransitionExpectationDTO(
                    expectation_id=str(item["expectation_id"]),
                    field_schema_id=str(item["field_schema_id"]),
                    annotation_ids=tuple(item["annotation_ids"]),
                    relation_ids=tuple(item["relation_ids"]),
                    expected_change=str(item["expected_change"]),
                    minimum_mean_absolute_change=float(
                        item.get("minimum_mean_absolute_change", 0.0)
                    ),
                    maximum_mean_absolute_change=float(
                        item.get("maximum_mean_absolute_change", 1.0)
                    ),
                    minimum_changed_fraction=float(
                        item.get("minimum_changed_fraction", 0.0)
                    ),
                    maximum_changed_fraction=float(
                        item.get("maximum_changed_fraction", 1.0)
                    ),
                    minimum_signed_change_magnitude=float(
                        item.get("minimum_signed_change_magnitude", 0.0)
                    ),
                    version=str(
                        item.get("version", "perception-transition-expectation/1")
                    ),
                )
                for item in data["expectations"]  # type: ignore[index]
            ),
            version=str(data.get("version", TRANSITION_EXPECTATION_SET_VERSION)),
        )


@dataclass(frozen=True)
class VisualTransitionReaderTraceDTO:
    """Visual Sign Reader evidence preserved at transition-analysis level."""

    raw_input_digest: str
    canonical_input_digest: str
    feature_digest: str
    visual_index_artifact_id: str
    policy_artifact_id: str
    acceptance_profile: str
    policy_executed: bool
    nearest_row_id: str | None = None
    matched_row_id: str | None = None
    canonical_input_match: bool = False
    exact_feature_match: bool = False
    version: str = VISUAL_TRANSITION_READER_TRACE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "raw_input_digest",
            "canonical_input_digest",
            "feature_digest",
            "visual_index_artifact_id",
            "policy_artifact_id",
            "acceptance_profile",
        ):
            if not getattr(self, name):
                raise PerceptionTransitionAnalysisError(
                    f"visual reader trace {name} must be non-empty"
                )
        if self.acceptance_profile not in VISUAL_READER_ACCEPTANCE_PROFILES:
            raise PerceptionTransitionAnalysisError(
                "unsupported visual reader acceptance profile"
            )
        if self.acceptance_profile == "evidence_only" and self.policy_executed:
            raise PerceptionTransitionAnalysisError(
                "evidence_only reader traces cannot execute policy"
            )
        if self.version != VISUAL_TRANSITION_READER_TRACE_VERSION:
            raise PerceptionTransitionAnalysisError(
                "unsupported visual transition reader trace version"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_visual_decision(cls, decision: object) -> "VisualTransitionReaderTraceDTO":
        return cls(
            raw_input_digest=str(getattr(decision, "raw_input_digest")),
            canonical_input_digest=str(getattr(decision, "canonical_input_digest")),
            feature_digest=str(getattr(decision, "feature_digest")),
            visual_index_artifact_id=str(getattr(decision, "visual_index_artifact_id")),
            policy_artifact_id=str(getattr(decision, "policy_artifact_id")),
            acceptance_profile=str(getattr(decision, "acceptance_profile")),
            policy_executed=bool(getattr(decision, "policy_executed")),
            nearest_row_id=getattr(decision, "nearest_row_id"),
            matched_row_id=getattr(decision, "matched_row_id"),
            canonical_input_match=bool(getattr(decision, "canonical_input_match")),
            exact_feature_match=bool(getattr(decision, "exact_feature_match")),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "VisualTransitionReaderTraceDTO":
        return cls(
            raw_input_digest=str(data["raw_input_digest"]),
            canonical_input_digest=str(data["canonical_input_digest"]),
            feature_digest=str(data["feature_digest"]),
            visual_index_artifact_id=str(data["visual_index_artifact_id"]),
            policy_artifact_id=str(data["policy_artifact_id"]),
            acceptance_profile=str(data["acceptance_profile"]),
            policy_executed=bool(data["policy_executed"]),
            nearest_row_id=data.get("nearest_row_id"),  # type: ignore[arg-type]
            matched_row_id=data.get("matched_row_id"),  # type: ignore[arg-type]
            canonical_input_match=bool(data.get("canonical_input_match", False)),
            exact_feature_match=bool(data.get("exact_feature_match", False)),
            version=str(data.get("version", VISUAL_TRANSITION_READER_TRACE_VERSION)),
        )


@dataclass(frozen=True)
class VisualTransitionAnalysisDTO:
    """One immutable identity for before/action/after contract evaluation."""

    analysis_id: str
    transition_evidence_id: str
    before_source_vpm_id: str
    after_source_vpm_id: str
    field_schema_id: str
    action_id: str
    expectation_set_id: str
    conformance_report_id: str
    status: str
    action: TransitionActionDeclarationDTO
    expectation_set: TransitionExpectationSetDTO
    conformance_report: TransitionConformanceReportDTO
    before_reader_trace: VisualTransitionReaderTraceDTO | None = None
    after_reader_trace: VisualTransitionReaderTraceDTO | None = None
    version: str = VISUAL_TRANSITION_ANALYSIS_VERSION

    def __post_init__(self) -> None:
        if self.status not in VISUAL_TRANSITION_ANALYSIS_STATUSES:
            raise PerceptionTransitionAnalysisError(
                "unsupported visual transition analysis status"
            )
        if self.version != VISUAL_TRANSITION_ANALYSIS_VERSION:
            raise PerceptionTransitionAnalysisError(
                "unsupported visual transition analysis version"
            )
        if self.action_id != self.action.action_id:
            raise PerceptionTransitionAnalysisError("analysis action identity mismatch")
        if self.expectation_set_id != self.expectation_set.expectation_set_id:
            raise PerceptionTransitionAnalysisError(
                "analysis expectation set identity mismatch"
            )
        if self.conformance_report_id != self.conformance_report.report_id:
            raise PerceptionTransitionAnalysisError(
                "analysis conformance report identity mismatch"
            )
        if self.status != self.conformance_report.status:
            raise PerceptionTransitionAnalysisError(
                "analysis status must match conformance report status"
            )
        if (
            self.transition_evidence_id
            != self.conformance_report.transition_evidence_id
        ):
            raise PerceptionTransitionAnalysisError(
                "analysis transition evidence identity mismatch"
            )
        if self.field_schema_id != self.conformance_report.field_schema_id:
            raise PerceptionTransitionAnalysisError(
                "analysis field schema identity mismatch"
            )
        if tuple(self.conformance_report.expectation_ids) != tuple(
            item.expectation_id for item in self.expectation_set.expectations
        ):
            raise PerceptionTransitionAnalysisError(
                "conformance report was not evaluated against this expectation set"
            )
        if self.analysis_id != _digest(self.canonical_payload()):
            raise PerceptionTransitionAnalysisError(
                "analysis identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        *,
        transition: TransitionEvidenceVPMDTO,
        action: TransitionActionDeclarationDTO,
        expectation_set: TransitionExpectationSetDTO,
        conformance_report: TransitionConformanceReportDTO,
        before_reader_trace: VisualTransitionReaderTraceDTO | None = None,
        after_reader_trace: VisualTransitionReaderTraceDTO | None = None,
    ) -> "VisualTransitionAnalysisDTO":
        values: dict[str, object] = {
            "transition_evidence_id": transition.transition_evidence_id,
            "before_source_vpm_id": transition.before_source_vpm_id,
            "after_source_vpm_id": transition.after_source_vpm_id,
            "field_schema_id": transition.field_schema_id,
            "action_id": action.action_id,
            "expectation_set_id": expectation_set.expectation_set_id,
            "conformance_report_id": conformance_report.report_id,
            "status": conformance_report.status,
            "before_reader_trace": None
            if before_reader_trace is None
            else before_reader_trace.to_dict(),
            "after_reader_trace": None
            if after_reader_trace is None
            else after_reader_trace.to_dict(),
            "version": VISUAL_TRANSITION_ANALYSIS_VERSION,
        }
        return cls(
            analysis_id=_digest(values),
            action=action,
            expectation_set=expectation_set,
            conformance_report=conformance_report,
            before_reader_trace=before_reader_trace,
            after_reader_trace=after_reader_trace,
            **{
                k: v
                for k, v in values.items()
                if k not in {"before_reader_trace", "after_reader_trace", "version"}
            },  # type: ignore[arg-type]
        )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "action_id": self.action_id,
            "after_reader_trace": None
            if self.after_reader_trace is None
            else self.after_reader_trace.to_dict(),
            "after_source_vpm_id": self.after_source_vpm_id,
            "before_reader_trace": None
            if self.before_reader_trace is None
            else self.before_reader_trace.to_dict(),
            "before_source_vpm_id": self.before_source_vpm_id,
            "conformance_report_id": self.conformance_report_id,
            "expectation_set_id": self.expectation_set_id,
            "field_schema_id": self.field_schema_id,
            "status": self.status,
            "transition_evidence_id": self.transition_evidence_id,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.canonical_payload(),
            "analysis_id": self.analysis_id,
            "action": self.action.to_dict(),
            "expectation_set": self.expectation_set.to_dict(),
            "conformance_report": _thaw(asdict(self.conformance_report)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "VisualTransitionAnalysisDTO":
        report_data = data["conformance_report"]  # type: ignore[index]
        findings = tuple(
            TransitionConformanceFindingDTO(
                finding_id=str(item["finding_id"]),
                status=str(item["status"]),
                expectation_id=(
                    None
                    if item.get("expectation_id") is None
                    else str(item["expectation_id"])
                ),
                annotation_ids=tuple(item["annotation_ids"]),
                relation_ids=tuple(item["relation_ids"]),
                field_ids=tuple(item["field_ids"]),
                observed_mean_absolute_change=float(
                    item["observed_mean_absolute_change"]
                ),
                observed_mean_signed_change=float(item["observed_mean_signed_change"]),
                observed_changed_fraction=float(item["observed_changed_fraction"]),
                detail=str(item["detail"]),
                version=str(
                    item.get(
                        "version",
                        "perception-transition-conformance-finding/1",
                    )
                ),
            )
            for item in report_data["findings"]  # type: ignore[index]
        )
        report = TransitionConformanceReportDTO(
            report_id=str(report_data["report_id"]),  # type: ignore[index]
            status=str(report_data["status"]),  # type: ignore[index]
            transition_evidence_id=str(report_data["transition_evidence_id"]),  # type: ignore[index]
            field_schema_id=str(report_data["field_schema_id"]),  # type: ignore[index]
            expectation_ids=tuple(report_data["expectation_ids"]),  # type: ignore[index]
            annotation_ids=tuple(report_data["annotation_ids"]),  # type: ignore[index]
            relation_ids=tuple(report_data["relation_ids"]),  # type: ignore[index]
            minimum_unexplained_mean_absolute_change=float(
                report_data["minimum_unexplained_mean_absolute_change"]  # type: ignore[index]
            ),
            minimum_unexplained_changed_fraction=float(
                report_data["minimum_unexplained_changed_fraction"]  # type: ignore[index]
            ),
            findings=findings,
            semantics=str(
                report_data.get(  # type: ignore[attr-defined]
                    "semantics",
                    "weighted_field_transition_measurements_compared_with_declared_annotation_or_relation_thresholds",
                )
            ),
            version=str(
                report_data.get(  # type: ignore[attr-defined]
                    "version",
                    "perception-transition-conformance-report/1",
                )
            ),
        )
        return cls(
            analysis_id=str(data["analysis_id"]),
            transition_evidence_id=str(data["transition_evidence_id"]),
            before_source_vpm_id=str(data["before_source_vpm_id"]),
            after_source_vpm_id=str(data["after_source_vpm_id"]),
            field_schema_id=str(data["field_schema_id"]),
            action_id=str(data["action_id"]),
            expectation_set_id=str(data["expectation_set_id"]),
            conformance_report_id=str(data["conformance_report_id"]),
            status=str(data["status"]),
            action=TransitionActionDeclarationDTO.from_dict(data["action"]),  # type: ignore[arg-type]
            expectation_set=TransitionExpectationSetDTO.from_dict(
                data["expectation_set"]  # type: ignore[arg-type]
            ),
            conformance_report=report,
            before_reader_trace=(
                None
                if data.get("before_reader_trace") is None
                else VisualTransitionReaderTraceDTO.from_dict(
                    data["before_reader_trace"]  # type: ignore[arg-type]
                )
            ),
            after_reader_trace=(
                None
                if data.get("after_reader_trace") is None
                else VisualTransitionReaderTraceDTO.from_dict(
                    data["after_reader_trace"]  # type: ignore[arg-type]
                )
            ),
            version=str(data.get("version", VISUAL_TRANSITION_ANALYSIS_VERSION)),
        )
