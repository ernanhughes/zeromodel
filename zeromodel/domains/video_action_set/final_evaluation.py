from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...artifact import VPMValidationError
from .canonical_json import canonical_sha256
from .final_access_dto import FinalEvaluationProtocolDTO


FINAL_EVALUATION_RESULT_VERSION = "zeromodel-video-final-evaluation-result/v1"
FORBIDDEN_DECISION_PHASES = frozenset(
    {"development", "calibration", "selection", "tuning", "candidate_selection"}
)


def evaluate_final_protocol(
    protocol: FinalEvaluationProtocolDTO,
    evidence_rows: Sequence[Mapping[str, object]],
    *,
    benchmark_seed_digest: str,
    sealed_plan_digest: str,
) -> dict[str, Any]:
    """Evaluate final evidence against an already-approved deterministic protocol."""

    if not protocol.approved:
        raise VPMValidationError("final evaluation protocol is not approved")
    if protocol.benchmark_seed_digest != benchmark_seed_digest:
        raise VPMValidationError("final evaluation benchmark identity mismatch")
    if protocol.sealed_plan_digest != sealed_plan_digest:
        raise VPMValidationError("final evaluation sealed plan mismatch")
    decision_rule = _mapping(
        protocol.decision_rule.to_value(),
        "final decision rule mismatch",
    )
    required = _mapping(
        protocol.required_evidence.to_value(),
        "final required evidence mismatch",
    )
    _reject_tuning_or_selection(decision_rule)
    _reject_tuning_or_selection(required)
    _validate_evidence_rows(evidence_rows)
    provider_id = _optional_str(required.get("provider_id"))
    metric_id = _required_str(decision_rule.get("metric_id"), "final metric mismatch")
    expected_count = _optional_nonnegative_int(required.get("expected_row_count"))
    rows = tuple(
        row
        for row in evidence_rows
        if provider_id is None or row.get("provider_id") == provider_id
    )
    if expected_count is not None and len(rows) != expected_count:
        return _result(
            protocol,
            "indeterminate",
            {
                "reason": "incomplete_final_evidence",
                "expected_row_count": expected_count,
                "actual_row_count": len(rows),
            },
        )
    values = tuple(_metric_value(row, metric_id) for row in rows)
    if not values:
        return _result(
            protocol,
            "indeterminate",
            {"reason": "missing_final_metric", "metric_id": metric_id},
        )
    aggregate = _aggregate(values, decision_rule)
    threshold = _number(decision_rule.get("threshold"), "final threshold mismatch")
    operator = _required_str(
        decision_rule.get("operator"),
        "final decision operator mismatch",
    )
    if operator == "gte":
        passed = aggregate >= threshold
    elif operator == "lte":
        passed = aggregate <= threshold
    else:
        raise VPMValidationError("final decision operator mismatch")
    return _result(
        protocol,
        "passed" if passed else "failed",
        {
            "provider_id": provider_id,
            "metric_id": metric_id,
            "row_count": len(rows),
            "aggregate": aggregate,
            "operator": operator,
            "threshold": threshold,
        },
    )


def _result(
    protocol: FinalEvaluationProtocolDTO,
    decision: str,
    measurements: Mapping[str, object],
) -> dict[str, Any]:
    payload = {
        "version": FINAL_EVALUATION_RESULT_VERSION,
        "protocol_digest": protocol.protocol_digest,
        "decision": decision,
        "measurements": dict(measurements),
    }
    return payload | {"evaluation_digest": canonical_sha256(payload)}


def _aggregate(values: Sequence[float], decision_rule: Mapping[str, object]) -> float:
    aggregate = _required_str(
        decision_rule.get("aggregate"),
        "final decision aggregate mismatch",
    )
    if aggregate == "mean":
        return sum(values) / len(values)
    if aggregate == "minimum":
        return min(values)
    if aggregate == "maximum":
        return max(values)
    raise VPMValidationError("final decision aggregate mismatch")


def _metric_value(row: Mapping[str, object], metric_id: str) -> float:
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping) and metric_id in metrics:
        return _number(metrics[metric_id], "final metric mismatch")
    if metric_id in row:
        return _number(row[metric_id], "final metric mismatch")
    raise VPMValidationError("final metric mismatch")


def _validate_evidence_rows(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        split = row.get("split")
        if split != "final":
            raise VPMValidationError("final evaluation evidence split mismatch")
        phase = row.get("phase")
        if isinstance(phase, str) and phase in FORBIDDEN_DECISION_PHASES:
            raise VPMValidationError("final evaluation refuses tuning or selection")


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
                raise VPMValidationError(
                    "final evaluation refuses tuning or selection"
                )
            _reject_tuning_or_selection(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _reject_tuning_or_selection(item)


def _mapping(value: object, message: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise VPMValidationError(message)
    return value


def _required_str(value: object, message: str) -> str:
    if not isinstance(value, str):
        raise VPMValidationError(message)
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise VPMValidationError("final required evidence mismatch")
    return value


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise VPMValidationError("final required evidence mismatch")
    return value


def _number(value: object, message: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VPMValidationError(message)
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise VPMValidationError(message)
    return result


__all__ = [
    "FINAL_EVALUATION_RESULT_VERSION",
    "evaluate_final_protocol",
]
