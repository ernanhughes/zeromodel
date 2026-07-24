"""Statistically gated operational health diagnosis for Stage P16E.

This governed implementation preserves the immutable P16 report contracts while requiring
adequate inference, reference, label, and accepted-label evidence before any metric may be
classified as healthy or drifted. Accuracy comparison is withheld unless the production
label cohort is sufficiently complete to match the fully-labelled test reference estimand.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .health import (
    OPERATIONAL_HEALTH_FINDING_VERSION,
    OPERATIONAL_HEALTH_REPORT_VERSION,
    OPERATIONAL_HEALTH_SEMANTICS,
    ActionFrequencyDTO,
    OperationalHealthFindingDTO,
    OperationalHealthReportDTO,
    OperationalReferenceProfileDTO,
    PerceptionOperationalHealthError,
)
from .production import (
    PerceptionProductionLedgerError,
    PerceptionProductionLedgerStore,
    ProductionInferenceRecordDTO,
    build_production_metrics_report,
)

OPERATIONAL_DRIFT_POLICY_VERSION: Final = "perception-operational-drift-policy/2"
OPERATIONAL_EVIDENCE_SEMANTICS: Final = (
    "metric_specific_reference_inference_label_and_accepted_label_evidence_gates"
)


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
class OperationalDriftPolicyDTO:
    maximum_coverage_drop: float = 0.10
    maximum_mean_margin_drop: float = 0.10
    maximum_action_distribution_distance: float = 0.20
    maximum_raw_accuracy_drop: float = 0.10
    maximum_accepted_accuracy_drop: float = 0.10
    minimum_reference_count: int = 20
    minimum_inference_count: int = 20
    minimum_labeled_count: int = 20
    minimum_accepted_labeled_count: int = 20
    minimum_label_coverage: float = 1.0
    require_single_pointer_revision: bool = True
    evidence_semantics: str = OPERATIONAL_EVIDENCE_SEMANTICS
    version: str = OPERATIONAL_DRIFT_POLICY_VERSION

    def __post_init__(self) -> None:
        for value in (
            self.maximum_coverage_drop,
            self.maximum_mean_margin_drop,
            self.maximum_action_distribution_distance,
            self.maximum_raw_accuracy_drop,
            self.maximum_accepted_accuracy_drop,
            self.minimum_label_coverage,
        ):
            if not 0.0 <= value <= 1.0:
                raise PerceptionOperationalHealthError("drift threshold or coverage gate outside [0, 1]")
        for value in (
            self.minimum_reference_count,
            self.minimum_inference_count,
            self.minimum_labeled_count,
            self.minimum_accepted_labeled_count,
        ):
            if value <= 0:
                raise PerceptionOperationalHealthError("health evidence counts must be positive")
        if self.evidence_semantics != OPERATIONAL_EVIDENCE_SEMANTICS:
            raise PerceptionOperationalHealthError("unsupported health evidence semantics")
        if self.version != OPERATIONAL_DRIFT_POLICY_VERSION:
            raise PerceptionOperationalHealthError("unsupported drift policy version")


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


def _insufficient(
    metric: str,
    reference: float | None,
    observed: float | None,
    threshold: float,
    evidence_count: int,
    rationale: str,
) -> OperationalHealthFindingDTO:
    return _finding(
        metric=metric,
        status="insufficient_evidence",
        reference_value=reference,
        observed_value=observed,
        delta=None,
        threshold=threshold,
        evidence_count=evidence_count,
        rationale=rationale,
    )


def _drop_finding(
    metric: str,
    reference: float,
    observed: float,
    threshold: float,
    evidence_count: int,
) -> OperationalHealthFindingDTO:
    drop = reference - observed
    return _finding(
        metric=metric,
        status="drifted" if drop > threshold else "healthy",
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
    """Diagnose one production window only after metric-specific evidence gates pass."""

    resolved = policy or OperationalDriftPolicyDTO()
    try:
        metrics = build_production_metrics_report(
            store,
            start_sequence_number=start_sequence_number,
            end_sequence_number=end_sequence_number,
            promoted_model_id=reference.promoted_model_id,
        )
    except PerceptionProductionLedgerError as exc:
        raise PerceptionOperationalHealthError(str(exc)) from exc

    selected: tuple[ProductionInferenceRecordDTO, ...] = tuple(
        item
        for item in store.list_inferences()
        if metrics.start_sequence_number <= item.sequence_number <= metrics.end_sequence_number
        and item.promoted_model_id == reference.promoted_model_id
    )
    actions = _distribution(tuple(item.selected_action for item in selected))
    reference_map = {item.action_label: item.frequency for item in reference.action_distribution}
    production_map = {item.action_label: item.frequency for item in actions}
    labels = set(reference_map) | set(production_map)
    distance = 0.5 * sum(
        abs(reference_map.get(label, 0.0) - production_map.get(label, 0.0))
        for label in labels
    )

    reference_ready = reference.example_count >= resolved.minimum_reference_count
    inference_ready = metrics.inference_count >= resolved.minimum_inference_count
    pointer_ready = (
        not resolved.require_single_pointer_revision
        or len(metrics.model_pointer_revisions) == 1
    )
    unlabeled_ready = reference_ready and inference_ready and pointer_ready
    unlabeled_reason = (
        f"requires reference_count>={resolved.minimum_reference_count}, "
        f"inference_count>={resolved.minimum_inference_count}, and "
        f"single_pointer_revision={resolved.require_single_pointer_revision}; observed "
        f"reference_count={reference.example_count}, inference_count={metrics.inference_count}, "
        f"pointer_revisions={list(metrics.model_pointer_revisions)}"
    )

    if unlabeled_ready:
        findings: list[OperationalHealthFindingDTO] = [
            _drop_finding(
                "coverage",
                reference.coverage,
                metrics.coverage,
                resolved.maximum_coverage_drop,
                metrics.inference_count,
            ),
            _drop_finding(
                "mean_margin",
                reference.mean_margin,
                metrics.mean_margin,
                resolved.maximum_mean_margin_drop,
                metrics.inference_count,
            ),
            _finding(
                metric="action_distribution",
                status=(
                    "drifted"
                    if distance > resolved.maximum_action_distribution_distance
                    else "healthy"
                ),
                reference_value=0.0,
                observed_value=distance,
                delta=distance,
                threshold=resolved.maximum_action_distribution_distance,
                evidence_count=metrics.inference_count,
                rationale=(
                    f"total-variation distance is {distance:.6f}; allowed distance is "
                    f"{resolved.maximum_action_distribution_distance:.6f}"
                ),
            ),
        ]
    else:
        findings = [
            _insufficient(
                "coverage",
                reference.coverage,
                metrics.coverage,
                resolved.maximum_coverage_drop,
                metrics.inference_count,
                unlabeled_reason,
            ),
            _insufficient(
                "mean_margin",
                reference.mean_margin,
                metrics.mean_margin,
                resolved.maximum_mean_margin_drop,
                metrics.inference_count,
                unlabeled_reason,
            ),
            _insufficient(
                "action_distribution",
                0.0,
                distance,
                resolved.maximum_action_distribution_distance,
                metrics.inference_count,
                unlabeled_reason,
            ),
        ]

    label_coverage = metrics.labeled_count / metrics.inference_count
    accuracy_ready = (
        unlabeled_ready
        and metrics.labeled_count >= resolved.minimum_labeled_count
        and label_coverage >= resolved.minimum_label_coverage
        and metrics.raw_accuracy is not None
    )
    if accuracy_ready:
        assert metrics.raw_accuracy is not None
        findings.append(
            _drop_finding(
                "raw_accuracy",
                reference.raw_accuracy,
                metrics.raw_accuracy,
                resolved.maximum_raw_accuracy_drop,
                metrics.labeled_count,
            )
        )
    else:
        findings.append(
            _insufficient(
                "raw_accuracy",
                reference.raw_accuracy,
                metrics.raw_accuracy,
                resolved.maximum_raw_accuracy_drop,
                metrics.labeled_count,
                (
                    f"requires labeled_count>={resolved.minimum_labeled_count} and "
                    f"label_coverage>={resolved.minimum_label_coverage:.6f}; observed "
                    f"labeled_count={metrics.labeled_count}, label_coverage={label_coverage:.6f}"
                ),
            )
        )

    outcomes_by_record = {
        item.inference_record_id: item for item in store.list_outcomes()
    }
    accepted_labeled_count = sum(
        1
        for item in selected
        if item.status == "accepted" and item.record_id in outcomes_by_record
    )
    accepted_ready = (
        accuracy_ready
        and accepted_labeled_count >= resolved.minimum_accepted_labeled_count
        and reference.accepted_accuracy is not None
        and metrics.accepted_accuracy is not None
    )
    if accepted_ready:
        assert reference.accepted_accuracy is not None
        assert metrics.accepted_accuracy is not None
        findings.append(
            _drop_finding(
                "accepted_accuracy",
                reference.accepted_accuracy,
                metrics.accepted_accuracy,
                resolved.maximum_accepted_accuracy_drop,
                accepted_labeled_count,
            )
        )
    else:
        findings.append(
            _insufficient(
                "accepted_accuracy",
                reference.accepted_accuracy,
                metrics.accepted_accuracy,
                resolved.maximum_accepted_accuracy_drop,
                accepted_labeled_count,
                (
                    f"requires accepted_labeled_count>="
                    f"{resolved.minimum_accepted_labeled_count} and defined reference/observed "
                    f"accepted accuracy; observed accepted_labeled_count={accepted_labeled_count}"
                ),
            )
        )

    if any(item.status == "insufficient_evidence" for item in findings):
        overall = "insufficient_evidence"
    elif any(item.status == "drifted" for item in findings):
        overall = "drifted"
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
            {
                "action_label": item.action_label,
                "count": item.count,
                "frequency": item.frequency,
            }
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
