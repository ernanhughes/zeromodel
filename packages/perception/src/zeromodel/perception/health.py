"""Operational drift and health diagnosis for Stage P16.

P16 freezes a reference profile from the promoted model's untouched P11 test report,
then compares an immutable P14/P15 production sequence window against that profile.
Coverage, margin, selected-action distribution, and labeled accuracy remain separate
signals. Missing production labels produce insufficient evidence rather than an
invented accuracy finding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .production import (
    PerceptionProductionLedgerError,
    PerceptionProductionLedgerStore,
    ProductionInferenceRecordDTO,
    build_production_metrics_report,
)
from .promoted_inference import PromotedTestEvaluationReportDTO

OPERATIONAL_REFERENCE_PROFILE_VERSION: Final = "perception-operational-reference-profile/1"
OPERATIONAL_DRIFT_POLICY_VERSION: Final = "perception-operational-drift-policy/1"
OPERATIONAL_HEALTH_FINDING_VERSION: Final = "perception-operational-health-finding/1"
OPERATIONAL_HEALTH_REPORT_VERSION: Final = "perception-operational-health-report/1"
OPERATIONAL_REFERENCE_SEMANTICS: Final = (
    "frozen_reference_profile_from_untouched_promoted_test_evaluation"
)
OPERATIONAL_HEALTH_SEMANTICS: Final = (
    "production_window_health_against_frozen_promoted_model_reference"
)
OPERATIONAL_HEALTH_STATUSES: Final = {"healthy", "drifted", "insufficient_evidence"}
OPERATIONAL_HEALTH_METRICS: Final = {
    "coverage",
    "mean_margin",
    "action_distribution",
    "raw_accuracy",
    "accepted_accuracy",
}


class PerceptionOperationalHealthError(ValueError):
    """Raised when P16 drift and health contracts are violated."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, object]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


@dataclass(frozen=True)
class ActionFrequencyDTO:
    action_label: str
    count: int
    frequency: float

    def __post_init__(self) -> None:
        if not self.action_label or self.count < 0:
            raise PerceptionOperationalHealthError("invalid action-frequency entry")
        if not 0.0 <= self.frequency <= 1.0:
            raise PerceptionOperationalHealthError("action frequency must be in [0, 1]")


@dataclass(frozen=True)
class OperationalReferenceProfileDTO:
    reference_profile_id: str
    promoted_model_id: str
    model_kind: str
    model_id: str
    test_report_id: str
    example_count: int
    coverage: float
    mean_margin: float
    raw_accuracy: float
    accepted_accuracy: float | None
    action_distribution: tuple[ActionFrequencyDTO, ...]
    semantics: str = OPERATIONAL_REFERENCE_SEMANTICS
    version: str = OPERATIONAL_REFERENCE_PROFILE_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.reference_profile_id,
                self.promoted_model_id,
                self.model_kind,
                self.model_id,
                self.test_report_id,
            )
        ):
            raise PerceptionOperationalHealthError("reference identities must be non-empty")
        if self.model_kind not in {"single_frame", "temporal"}:
            raise PerceptionOperationalHealthError("unsupported reference model kind")
        if self.example_count <= 0:
            raise PerceptionOperationalHealthError("reference profile requires examples")
        for value in (self.coverage, self.mean_margin, self.raw_accuracy):
            if not 0.0 <= value <= 1.0:
                raise PerceptionOperationalHealthError("reference metric outside [0, 1]")
        if self.accepted_accuracy is not None and not 0.0 <= self.accepted_accuracy <= 1.0:
            raise PerceptionOperationalHealthError("reference accepted accuracy outside [0, 1]")
        if not self.action_distribution:
            raise PerceptionOperationalHealthError("reference action distribution cannot be empty")
        if self.action_distribution != tuple(
            sorted(self.action_distribution, key=lambda item: item.action_label)
        ):
            raise PerceptionOperationalHealthError("reference actions must be sorted")
        if sum(item.count for item in self.action_distribution) != self.example_count:
            raise PerceptionOperationalHealthError("reference action counts must exhaust examples")
        if abs(sum(item.frequency for item in self.action_distribution) - 1.0) > 1e-9:
            raise PerceptionOperationalHealthError("reference action frequencies must sum to one")
        if self.semantics != OPERATIONAL_REFERENCE_SEMANTICS:
            raise PerceptionOperationalHealthError("unsupported reference semantics")


@dataclass(frozen=True)
class OperationalDriftPolicyDTO:
    maximum_coverage_drop: float = 0.10
    maximum_mean_margin_drop: float = 0.10
    maximum_action_distribution_distance: float = 0.20
    maximum_raw_accuracy_drop: float = 0.10
    maximum_accepted_accuracy_drop: float = 0.10
    minimum_labeled_count: int = 20
    version: str = OPERATIONAL_DRIFT_POLICY_VERSION

    def __post_init__(self) -> None:
        for value in (
            self.maximum_coverage_drop,
            self.maximum_mean_margin_drop,
            self.maximum_action_distribution_distance,
            self.maximum_raw_accuracy_drop,
            self.maximum_accepted_accuracy_drop,
        ):
            if not 0.0 <= value <= 1.0:
                raise PerceptionOperationalHealthError("drift threshold outside [0, 1]")
        if self.minimum_labeled_count <= 0:
            raise PerceptionOperationalHealthError("minimum_labeled_count must be positive")


@dataclass(frozen=True)
class OperationalHealthFindingDTO:
    finding_id: str
    metric: str
    status: str
    reference_value: float | None
    observed_value: float | None
    delta: float | None
    threshold: float
    evidence_count: int
    rationale: str
    version: str = OPERATIONAL_HEALTH_FINDING_VERSION

    def __post_init__(self) -> None:
        if self.metric not in OPERATIONAL_HEALTH_METRICS:
            raise PerceptionOperationalHealthError("unsupported health metric")
        if self.status not in OPERATIONAL_HEALTH_STATUSES:
            raise PerceptionOperationalHealthError("unsupported health status")
        if not self.finding_id or not self.rationale:
            raise PerceptionOperationalHealthError("health finding identity and rationale required")
        if self.evidence_count < 0 or not 0.0 <= self.threshold <= 1.0:
            raise PerceptionOperationalHealthError("invalid finding evidence or threshold")
        for value in (self.reference_value, self.observed_value):
            if value is not None and not 0.0 <= value <= 1.0:
                raise PerceptionOperationalHealthError("finding value outside [0, 1]")
        if self.delta is not None and not -1.0 <= self.delta <= 1.0:
            raise PerceptionOperationalHealthError("finding delta outside [-1, 1]")


@dataclass(frozen=True)
class OperationalHealthReportDTO:
    report_id: str
    reference_profile_id: str
    promoted_model_id: str
    start_sequence_number: int
    end_sequence_number: int
    production_metrics_report_id: str
    production_action_distribution: tuple[ActionFrequencyDTO, ...]
    action_distribution_distance: float
    overall_status: str
    findings: tuple[OperationalHealthFindingDTO, ...]
    inference_record_ids: tuple[str, ...]
    outcome_ids: tuple[str, ...]
    semantics: str = OPERATIONAL_HEALTH_SEMANTICS
    version: str = OPERATIONAL_HEALTH_REPORT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.report_id,
                self.reference_profile_id,
                self.promoted_model_id,
                self.production_metrics_report_id,
            )
        ):
            raise PerceptionOperationalHealthError("health report identities must be non-empty")
        if self.start_sequence_number <= 0 or self.end_sequence_number < self.start_sequence_number:
            raise PerceptionOperationalHealthError("invalid health report window")
        if self.overall_status not in OPERATIONAL_HEALTH_STATUSES:
            raise PerceptionOperationalHealthError("unsupported overall health status")
        if not 0.0 <= self.action_distribution_distance <= 1.0:
            raise PerceptionOperationalHealthError("action distribution distance outside [0, 1]")
        if tuple(item.metric for item in self.findings) != (
            "coverage",
            "mean_margin",
            "action_distribution",
            "raw_accuracy",
            "accepted_accuracy",
        ):
            raise PerceptionOperationalHealthError("health findings must use canonical order")
        if self.production_action_distribution != tuple(
            sorted(self.production_action_distribution, key=lambda item: item.action_label)
        ):
            raise PerceptionOperationalHealthError("production actions must be sorted")
        if self.semantics != OPERATIONAL_HEALTH_SEMANTICS:
            raise PerceptionOperationalHealthError("unsupported health report semantics")


def _distribution(labels: tuple[str, ...]) -> tuple[ActionFrequencyDTO, ...]:
    if not labels:
        raise PerceptionOperationalHealthError("action distribution requires labels")
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = len(labels)
    return tuple(
        ActionFrequencyDTO(action_label=label, count=count, frequency=count / total)
        for label, count in sorted(counts.items())
    )


def _frequency_map(items: tuple[ActionFrequencyDTO, ...]) -> dict[str, float]:
    return {item.action_label: item.frequency for item in items}


def build_operational_reference_profile(
    test_report: PromotedTestEvaluationReportDTO,
) -> OperationalReferenceProfileDTO:
    """Freeze the P11 untouched-test operating profile for later production comparison."""

    actions = _distribution(tuple(item.selected_action for item in test_report.examples))
    payload: Mapping[str, object] = {
        "accepted_accuracy": test_report.accepted_accuracy,
        "action_distribution": [
            {"action_label": item.action_label, "count": item.count, "frequency": item.frequency}
            for item in actions
        ],
        "coverage": test_report.coverage,
        "example_count": test_report.example_count,
        "mean_margin": test_report.mean_margin,
        "model_id": test_report.model_id,
        "model_kind": test_report.model_kind,
        "promoted_model_id": test_report.promoted_model_id,
        "raw_accuracy": test_report.raw_accuracy,
        "semantics": OPERATIONAL_REFERENCE_SEMANTICS,
        "test_report_id": test_report.report_id,
        "version": OPERATIONAL_REFERENCE_PROFILE_VERSION,
    }
    return OperationalReferenceProfileDTO(
        reference_profile_id=_digest(payload),
        promoted_model_id=test_report.promoted_model_id,
        model_kind=test_report.model_kind,
        model_id=test_report.model_id,
        test_report_id=test_report.report_id,
        example_count=test_report.example_count,
        coverage=test_report.coverage,
        mean_margin=test_report.mean_margin,
        raw_accuracy=test_report.raw_accuracy,
        accepted_accuracy=test_report.accepted_accuracy,
        action_distribution=actions,
    )


def _finding(
    *,
    metric: str,
    status: str,
    reference_value: float | None,
    observed_value: float | None,
    delta: float | None,
    threshold: float,
    evidence_count: int,
    rationale: str,
) -> OperationalHealthFindingDTO:
    payload: Mapping[str, object] = {
        "delta": delta,
        "evidence_count": evidence_count,
        "metric": metric,
        "observed_value": observed_value,
        "rationale": rationale,
        "reference_value": reference_value,
        "status": status,
        "threshold": threshold,
        "version": OPERATIONAL_HEALTH_FINDING_VERSION,
    }
    return OperationalHealthFindingDTO(finding_id=_digest(payload), **payload)  # type: ignore[arg-type]


def _drop_finding(
    metric: str,
    reference: float,
    observed: float,
    threshold: float,
    evidence_count: int,
) -> OperationalHealthFindingDTO:
    drop = reference - observed
    status = "drifted" if drop > threshold else "healthy"
    return _finding(
        metric=metric,
        status=status,
        reference_value=reference,
        observed_value=observed,
        delta=observed - reference,
        threshold=threshold,
        evidence_count=evidence_count,
        rationale=(
            f"observed {metric} is {drop:.6f} below reference; allowed drop is {threshold:.6f}"
        ),
    )


def diagnose_operational_health(
    reference: OperationalReferenceProfileDTO,
    store: PerceptionProductionLedgerStore,
    *,
    start_sequence_number: int,
    end_sequence_number: int | None = None,
    policy: OperationalDriftPolicyDTO | None = None,
) -> OperationalHealthReportDTO:
    """Diagnose one immutable production window against its frozen promoted-model reference."""

    resolved_policy = policy or OperationalDriftPolicyDTO()
    try:
        metrics = build_production_metrics_report(
            store,
            start_sequence_number=start_sequence_number,
            end_sequence_number=end_sequence_number,
            promoted_model_id=reference.promoted_model_id,
        )
    except PerceptionProductionLedgerError as exc:
        raise PerceptionOperationalHealthError(str(exc)) from exc

    all_records = store.list_inferences()
    selected: tuple[ProductionInferenceRecordDTO, ...] = tuple(
        item
        for item in all_records
        if metrics.start_sequence_number <= item.sequence_number <= metrics.end_sequence_number
        and item.promoted_model_id == reference.promoted_model_id
    )
    actions = _distribution(tuple(item.selected_action for item in selected))
    reference_map = _frequency_map(reference.action_distribution)
    production_map = _frequency_map(actions)
    labels = set(reference_map) | set(production_map)
    distance = 0.5 * sum(abs(reference_map.get(label, 0.0) - production_map.get(label, 0.0)) for label in labels)

    findings: list[OperationalHealthFindingDTO] = [
        _drop_finding(
            "coverage",
            reference.coverage,
            metrics.coverage,
            resolved_policy.maximum_coverage_drop,
            metrics.inference_count,
        ),
        _drop_finding(
            "mean_margin",
            reference.mean_margin,
            metrics.mean_margin,
            resolved_policy.maximum_mean_margin_drop,
            metrics.inference_count,
        ),
        _finding(
            metric="action_distribution",
            status=(
                "drifted"
                if distance > resolved_policy.maximum_action_distribution_distance
                else "healthy"
            ),
            reference_value=0.0,
            observed_value=distance,
            delta=distance,
            threshold=resolved_policy.maximum_action_distribution_distance,
            evidence_count=metrics.inference_count,
            rationale=(
                f"total-variation distance is {distance:.6f}; allowed distance is "
                f"{resolved_policy.maximum_action_distribution_distance:.6f}"
            ),
        ),
    ]

    if metrics.labeled_count < resolved_policy.minimum_labeled_count:
        findings.extend(
            (
                _finding(
                    metric="raw_accuracy",
                    status="insufficient_evidence",
                    reference_value=reference.raw_accuracy,
                    observed_value=metrics.raw_accuracy,
                    delta=None,
                    threshold=resolved_policy.maximum_raw_accuracy_drop,
                    evidence_count=metrics.labeled_count,
                    rationale=(
                        f"requires {resolved_policy.minimum_labeled_count} labeled outcomes; "
                        f"observed {metrics.labeled_count}"
                    ),
                ),
                _finding(
                    metric="accepted_accuracy",
                    status="insufficient_evidence",
                    reference_value=reference.accepted_accuracy,
                    observed_value=metrics.accepted_accuracy,
                    delta=None,
                    threshold=resolved_policy.maximum_accepted_accuracy_drop,
                    evidence_count=metrics.labeled_count,
                    rationale=(
                        f"requires {resolved_policy.minimum_labeled_count} labeled outcomes; "
                        f"observed {metrics.labeled_count}"
                    ),
                ),
            )
        )
    else:
        assert metrics.raw_accuracy is not None
        findings.append(
            _drop_finding(
                "raw_accuracy",
                reference.raw_accuracy,
                metrics.raw_accuracy,
                resolved_policy.maximum_raw_accuracy_drop,
                metrics.labeled_count,
            )
        )
        if reference.accepted_accuracy is None or metrics.accepted_accuracy is None:
            findings.append(
                _finding(
                    metric="accepted_accuracy",
                    status="insufficient_evidence",
                    reference_value=reference.accepted_accuracy,
                    observed_value=metrics.accepted_accuracy,
                    delta=None,
                    threshold=resolved_policy.maximum_accepted_accuracy_drop,
                    evidence_count=metrics.labeled_count,
                    rationale="accepted accuracy is undefined for the reference or production window",
                )
            )
        else:
            findings.append(
                _drop_finding(
                    "accepted_accuracy",
                    reference.accepted_accuracy,
                    metrics.accepted_accuracy,
                    resolved_policy.maximum_accepted_accuracy_drop,
                    metrics.labeled_count,
                )
            )

    if any(item.status == "drifted" for item in findings):
        overall = "drifted"
    elif any(item.status == "insufficient_evidence" for item in findings):
        overall = "insufficient_evidence"
    else:
        overall = "healthy"

    payload: Mapping[str, object] = {
        "action_distribution_distance": distance,
        "end_sequence_number": metrics.end_sequence_number,
        "findings": [item.finding_id for item in findings],
        "inference_record_ids": list(metrics.inference_record_ids),
        "outcome_ids": list(metrics.outcome_ids),
        "overall_status": overall,
        "production_action_distribution": [
            {"action_label": item.action_label, "count": item.count, "frequency": item.frequency}
            for item in actions
        ],
        "production_metrics_report_id": metrics.report_id,
        "promoted_model_id": reference.promoted_model_id,
        "reference_profile_id": reference.reference_profile_id,
        "semantics": OPERATIONAL_HEALTH_SEMANTICS,
        "start_sequence_number": metrics.start_sequence_number,
        "version": OPERATIONAL_HEALTH_REPORT_VERSION,
    }
    return OperationalHealthReportDTO(
        report_id=_digest(payload),
        reference_profile_id=reference.reference_profile_id,
        promoted_model_id=reference.promoted_model_id,
        start_sequence_number=metrics.start_sequence_number,
        end_sequence_number=metrics.end_sequence_number,
        production_metrics_report_id=metrics.report_id,
        production_action_distribution=actions,
        action_distribution_distance=distance,
        overall_status=overall,
        findings=tuple(findings),
        inference_record_ids=metrics.inference_record_ids,
        outcome_ids=metrics.outcome_ids,
    )
