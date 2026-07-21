from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
import re
from typing import cast

from ...artifact import VPMValidationError
from .final_access_dto import (
    FINAL_EVALUATION_RESULT_VERSION,
    FinalEvaluationProtocolDTO,
    FinalEvaluationResultDTO,
    FinalEvidenceBundleDTO,
)


FORBIDDEN_DECISION_PHASES = frozenset(
    {"development", "calibration", "selection", "tuning", "candidate_selection"}
)
DECISION_RULE_KEYS = frozenset(
    {"kind", "aggregate", "metric_id", "operator", "threshold"}
)
REQUIRED_EVIDENCE_KEYS = frozenset({"provider_id", "expected_row_count"})
DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")


def evaluate_final_protocol(
    protocol: FinalEvaluationProtocolDTO,
    evidence: FinalEvidenceBundleDTO,
) -> FinalEvaluationResultDTO:
    """Evaluate canonical final evidence with exact decimal threshold semantics.

    Rows are already sorted by the evidence DTO's stable identity. Numeric inputs
    are integers or decimal strings; binary floats are rejected. ``gte`` and
    ``lte`` include threshold equality.
    """

    if not protocol.approved:
        raise VPMValidationError("final evaluation protocol is not approved")
    if protocol.protocol_digest != evidence.protocol_digest:
        raise VPMValidationError("final evaluation protocol mismatch")
    if protocol.benchmark_seed_digest != evidence.benchmark_seed_digest:
        raise VPMValidationError("final evaluation benchmark identity mismatch")
    if protocol.sealed_plan_digest != evidence.sealed_plan_digest:
        raise VPMValidationError("final evaluation sealed plan mismatch")

    decision_rule = _exact_mapping(
        protocol.decision_rule.to_value(),
        DECISION_RULE_KEYS,
        "final decision rule mismatch",
    )
    required = _exact_mapping(
        protocol.required_evidence.to_value(),
        REQUIRED_EVIDENCE_KEYS,
        "final required evidence mismatch",
    )
    _reject_tuning_or_selection(decision_rule)
    _reject_tuning_or_selection(required)
    if decision_rule["kind"] != "fixed_metric_threshold":
        raise VPMValidationError("final decision rule kind mismatch")
    aggregate_kind = _required_str(
        decision_rule["aggregate"],
        "final decision aggregate mismatch",
    )
    if aggregate_kind not in {"mean", "minimum", "maximum"}:
        raise VPMValidationError("final decision aggregate mismatch")
    operator = _required_str(
        decision_rule["operator"],
        "final decision operator mismatch",
    )
    if operator not in {"gte", "lte"}:
        raise VPMValidationError("final decision operator mismatch")
    metric_id = _required_str(
        decision_rule["metric_id"],
        "final metric mismatch",
    )
    threshold = _decimal(decision_rule["threshold"], "final threshold mismatch")
    provider_id = _required_str(
        required["provider_id"],
        "final required evidence mismatch",
    )
    expected_count = _nonnegative_int(
        required["expected_row_count"],
        "final required evidence mismatch",
    )

    rows_value = evidence.rows.to_value()
    if not isinstance(rows_value, list):
        raise VPMValidationError("final evidence rows mismatch")
    rows = tuple(
        cast(Mapping[str, object], row)
        for row in rows_value
        if isinstance(row, Mapping) and row.get("provider_id") == provider_id
    )
    actual_counts = evidence.actual_counts.to_value()
    if len(rows) != expected_count:
        return _result(
            protocol=protocol,
            evidence=evidence,
            decision="indeterminate",
            descriptive_measurements={
                "provider_id": provider_id,
                "metric_id": metric_id,
                "expected_row_count": expected_count,
                "actual_row_count": len(rows),
            },
            family_measurements=[],
            rejections=[],
            indeterminate_reasons=["incomplete_final_evidence"],
            actual_counts=cast(Mapping[str, object], actual_counts),
        )

    values_by_family: dict[str, list[Decimal]] = defaultdict(list)
    values: list[Decimal] = []
    for row in rows:
        family_id = _required_str(row.get("family_id"), "final family id mismatch")
        value = _metric_value(row, metric_id)
        values.append(value)
        values_by_family[family_id].append(value)
    if not values:
        return _result(
            protocol=protocol,
            evidence=evidence,
            decision="indeterminate",
            descriptive_measurements={
                "provider_id": provider_id,
                "metric_id": metric_id,
                "expected_row_count": expected_count,
                "actual_row_count": 0,
            },
            family_measurements=[],
            rejections=[],
            indeterminate_reasons=["missing_final_metric"],
            actual_counts=cast(Mapping[str, object], actual_counts),
        )

    aggregate = _aggregate(values, aggregate_kind)
    passed = aggregate >= threshold if operator == "gte" else aggregate <= threshold
    family_measurements = [
        {
            "family_id": family_id,
            "row_count": len(family_values),
            "aggregate": _decimal_text(_aggregate(family_values, aggregate_kind)),
        }
        for family_id, family_values in sorted(values_by_family.items())
    ]
    return _result(
        protocol=protocol,
        evidence=evidence,
        decision="passed" if passed else "failed",
        descriptive_measurements={
            "provider_id": provider_id,
            "metric_id": metric_id,
            "row_count": len(rows),
            "aggregate_kind": aggregate_kind,
            "aggregate": _decimal_text(aggregate),
            "operator": operator,
            "threshold": _decimal_text(threshold),
            "threshold_equality": "inclusive",
        },
        family_measurements=family_measurements,
        rejections=[] if passed else ["fixed_threshold_not_met"],
        indeterminate_reasons=[],
        actual_counts=cast(Mapping[str, object], actual_counts),
    )


def _result(
    *,
    protocol: FinalEvaluationProtocolDTO,
    evidence: FinalEvidenceBundleDTO,
    decision: str,
    descriptive_measurements: Mapping[str, object],
    family_measurements: Sequence[Mapping[str, object]],
    rejections: Sequence[str],
    indeterminate_reasons: Sequence[str],
    actual_counts: Mapping[str, object],
) -> FinalEvaluationResultDTO:
    return FinalEvaluationResultDTO.create(
        {
            "version": FINAL_EVALUATION_RESULT_VERSION,
            "protocol_digest": protocol.protocol_digest,
            "evidence_digest": evidence.evidence_digest,
            "decision": decision,
            "descriptive_measurements": dict(descriptive_measurements),
            "family_measurements": [dict(item) for item in family_measurements],
            "rejections": list(rejections),
            "indeterminate_reasons": list(indeterminate_reasons),
            "actual_counts": dict(actual_counts),
        }
    )


def _aggregate(values: Sequence[Decimal], kind: str) -> Decimal:
    if kind == "mean":
        return sum(values, Decimal(0)) / Decimal(len(values))
    if kind == "minimum":
        return min(values)
    if kind == "maximum":
        return max(values)
    raise VPMValidationError("final decision aggregate mismatch")


def _metric_value(row: Mapping[str, object], metric_id: str) -> Decimal:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping) or metric_id not in metrics:
        raise VPMValidationError("final metric mismatch")
    return _decimal(metrics[metric_id], "final metric mismatch")


def _decimal(value: object, message: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise VPMValidationError(message)
    text = str(value)
    if DECIMAL_RE.fullmatch(text) is None:
        raise VPMValidationError(message)
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise VPMValidationError(message) from exc


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _exact_mapping(
    value: object,
    keys: frozenset[str],
    message: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise VPMValidationError(message)
    return value


def _required_str(value: object, message: str) -> str:
    if not isinstance(value, str) or not value:
        raise VPMValidationError(message)
    return value


def _nonnegative_int(value: object, message: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise VPMValidationError(message)
    return value


def _reject_tuning_or_selection(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in {
                "tuning",
                "selection",
                "candidate_tuning",
                "candidate_selection",
                "operating_point_selection",
            }:
                raise VPMValidationError("final evaluation refuses tuning or selection")
            _reject_tuning_or_selection(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _reject_tuning_or_selection(item)


__all__ = [
    "FINAL_EVALUATION_RESULT_VERSION",
    "evaluate_final_protocol",
]
