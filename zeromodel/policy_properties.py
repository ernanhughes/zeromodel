"""Exhaustive declarative property checks over finite policy artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Optional, Sequence, Tuple

from .artifact import (
    LayoutRecipe,
    ScoreTable,
    VPMArtifact,
    VPMValidationError,
    build_vpm,
)
from .policy_lookup import VPMPolicyLookup

CHECKER_VERSION = "zeromodel.policy-property/v1"
VERIFICATION_METRICS = ("passed", "coverage", "violation_count")


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError(
            "Policy property content must be JSON-serializable"
        ) from exc


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value).decode("utf-8"))


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(
        r"-?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?",
        value,
    ):
        return float(value)
    return value


def decode_key_value_row_id(row_id: str) -> dict[str, Any]:
    """Decode ``key=value|...`` row ids into a typed state mapping.

    ``key-value-row-id/v1`` applies these deterministic scalar rules:

    - ``none`` and ``null`` become :data:`None`;
    - ``true`` and ``false`` become booleans;
    - integer-looking values become :class:`int`;
    - decimal-looking values become :class:`float`;
    - every other value remains a string.

    Property assertions must use the corresponding JSON literal. For example,
    compare ``state.target`` with ``null``/``None`` in the property spec rather
    than the string ``"none"``.
    """

    state: dict[str, Any] = {}
    for part in str(row_id).split("|"):
        key, separator, raw_value = part.partition("=")
        if not separator or not key:
            raise VPMValidationError(
                "key-value-row-id/v1 requires key=value components: %s"
                % row_id
            )
        if key in state:
            raise VPMValidationError(
                "Duplicate state key %r in row_id %s" % (key, row_id)
            )
        state[key] = _parse_scalar(raw_value)
    return state


@dataclass(frozen=True)
class PolicyPropertySpec:
    """One named declarative assertion over a finite policy row."""

    property_id: str
    version: str
    assertion: Mapping[str, Any]
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyPropertySpec":
        property_id = str(data.get("id") or "")
        version = str(data.get("version") or "")
        assertion = data.get("assert")
        description = str(data.get("description") or "")
        if not property_id:
            raise VPMValidationError("Policy property requires a non-empty id")
        if not version:
            raise VPMValidationError(
                "Policy property %s requires a non-empty version" % property_id
            )
        if not isinstance(assertion, Mapping):
            raise VPMValidationError(
                "Policy property %s requires an assertion mapping"
                % property_id
            )
        return cls(
            property_id=property_id,
            version=version,
            assertion=_json_copy(assertion),
            description=description,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.property_id,
            "version": self.version,
            "description": self.description,
            "assert": _json_copy(self.assertion),
        }


@dataclass(frozen=True)
class PolicyPropertyViolation:
    """A counterexample row for one failed policy property."""

    row_id: str
    action: str
    value: float
    source_row_index: int
    source_metric_index: int
    view_row: int
    view_column: int
    candidates: Mapping[str, float]
    evidence: Mapping[str, float]
    state: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "action": self.action,
            "value": float(self.value),
            "source_row_index": int(self.source_row_index),
            "source_metric_index": int(self.source_metric_index),
            "view_row": int(self.view_row),
            "view_column": int(self.view_column),
            "candidates": {
                str(key): float(value)
                for key, value in self.candidates.items()
            },
            "evidence": {
                str(key): float(value)
                for key, value in self.evidence.items()
            },
            "state": _json_copy(self.state),
        }


@dataclass(frozen=True)
class PolicyPropertyResult:
    """Exhaustive result for one named property."""

    property_id: str
    version: str
    description: str
    passed: bool
    rows_checked: int
    violations: Tuple[PolicyPropertyViolation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_id": self.property_id,
            "version": self.version,
            "description": self.description,
            "passed": bool(self.passed),
            "rows_checked": int(self.rows_checked),
            "violation_count": len(self.violations),
            "violations": [
                violation.to_dict() for violation in self.violations
            ],
        }


@dataclass(frozen=True)
class PolicyVerificationReport:
    """Deterministic report for named properties checked against one policy."""

    policy_artifact_id: str
    checker_version: str
    property_spec_digest: str
    rows_available: int
    results: Tuple[PolicyPropertyResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_artifact_id": self.policy_artifact_id,
            "checker_version": self.checker_version,
            "property_spec_digest": self.property_spec_digest,
            "rows_available": int(self.rows_available),
            "passed": bool(self.passed),
            "results": [result.to_dict() for result in self.results],
        }

    def to_vpm(self) -> VPMArtifact:
        """Materialize this report as an identity-bearing verification artifact."""

        values = []
        row_ids = []
        for result in self.results:
            row_ids.append("%s@%s" % (result.property_id, result.version))
            coverage = (
                float(result.rows_checked) / float(self.rows_available)
                if self.rows_available
                else 0.0
            )
            values.append(
                (
                    1.0 if result.passed else 0.0,
                    coverage,
                    float(len(result.violations)),
                )
            )

        table = ScoreTable(
            values=values,
            row_ids=row_ids,
            metric_ids=VERIFICATION_METRICS,
            metadata={
                "kind": "finite_policy_verification",
                "policy_artifact_id": self.policy_artifact_id,
                "checker_version": self.checker_version,
                "property_spec_digest": self.property_spec_digest,
                "report": self.to_dict(),
            },
        )
        recipe = LayoutRecipe.from_dict(
            {
                "version": "vpm-layout/0",
                "name": "finite-policy-verification-source-order",
                "row_order": {
                    "kind": "source",
                    "tie_break": "row_id",
                },
                "column_order": {"kind": "source"},
                "normalization": {
                    "kind": "per_metric_minmax",
                    "clip": True,
                },
            }
        )
        return build_vpm(
            table,
            recipe,
            provenance={
                "kind": "finite_policy_verification",
                "checker": self.checker_version,
                "parents": [
                    {
                        "artifact_id": self.policy_artifact_id,
                        "relation": "verifies",
                    }
                ],
            },
        )


class PolicyPropertyChecker:
    """Exhaustively check declarative properties over a finite VPM policy."""

    def __init__(
        self,
        artifact: VPMArtifact,
        *,
        action_metric_ids: Sequence[str],
        evidence_metric_ids: Optional[Sequence[str]] = None,
        state_encoding: str = "key-value-row-id/v1",
        value_source: str = "raw",
        tie_break: str = "metric_order",
    ) -> None:
        if state_encoding != "key-value-row-id/v1":
            raise VPMValidationError(
                "Unsupported state_encoding: %s" % state_encoding
            )
        self.artifact = artifact
        self.state_encoding = state_encoding
        self.reader = VPMPolicyLookup(
            artifact,
            action_metric_ids=action_metric_ids,
            evidence_metric_ids=evidence_metric_ids,
            value_source=value_source,
            tie_break=tie_break,
        )

    def check(
        self,
        properties: Sequence[PolicyPropertySpec],
    ) -> PolicyVerificationReport:
        specs = tuple(properties)
        if not specs:
            raise VPMValidationError(
                "PolicyPropertyChecker requires at least one property"
            )
        keys = [(spec.property_id, spec.version) for spec in specs]
        if len(set(keys)) != len(keys):
            raise VPMValidationError(
                "Policy property id/version pairs must be unique"
            )

        results = []
        for spec in specs:
            violations = []
            rows_checked = 0
            for row_id in self.artifact.source.row_ids:
                decision = self.reader.read(row_id)
                state = decode_key_value_row_id(row_id)
                context = {
                    "artifact_id": decision.artifact_id,
                    "row_id": decision.row_id,
                    "winner": decision.action,
                    "value": decision.value,
                    "state": state,
                    "candidate": dict(decision.candidates),
                    "evidence": dict(decision.evidence),
                }
                try:
                    passed = bool(_evaluate(spec.assertion, context))
                except (VPMValidationError, TypeError, ValueError) as exc:
                    raise VPMValidationError(
                        "Property %s@%s failed to evaluate for row %s: %s"
                        % (
                            spec.property_id,
                            spec.version,
                            row_id,
                            exc,
                        )
                    ) from exc

                rows_checked += 1
                if not passed:
                    violations.append(
                        PolicyPropertyViolation(
                            row_id=decision.row_id,
                            action=decision.action,
                            value=decision.value,
                            source_row_index=decision.source_row_index,
                            source_metric_index=decision.source_metric_index,
                            view_row=decision.view_row,
                            view_column=decision.view_column,
                            candidates=dict(decision.candidates),
                            evidence=dict(decision.evidence),
                            state=state,
                        )
                    )

            results.append(
                PolicyPropertyResult(
                    property_id=spec.property_id,
                    version=spec.version,
                    description=spec.description,
                    passed=not violations,
                    rows_checked=rows_checked,
                    violations=tuple(violations),
                )
            )

        spec_payload = [spec.to_dict() for spec in specs]
        spec_digest = hashlib.sha256(_canonical_json(spec_payload)).hexdigest()
        return PolicyVerificationReport(
            policy_artifact_id=self.artifact.artifact_id,
            checker_version=CHECKER_VERSION,
            property_spec_digest=spec_digest,
            rows_available=len(self.artifact.source.row_ids),
            results=tuple(results),
        )


def _resolve_var(path: str, context: Mapping[str, Any]) -> Any:
    current: Any = context
    for component in str(path).split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise VPMValidationError("Unknown property variable: %s" % path)
        current = current[component]
    return current


def _binary_args(name: str, value: Any) -> tuple[Any, Any]:
    if not isinstance(value, list) or len(value) != 2:
        raise VPMValidationError(
            "Property operator %s requires a two-item list" % name
        )
    return value[0], value[1]


def _comparison_error(
    operator: str,
    left_value: Any,
    right_value: Any,
    exc: Exception,
) -> VPMValidationError:
    return VPMValidationError(
        "Property operator %s cannot compare %r (%s) with %r (%s): %s"
        % (
            operator,
            left_value,
            type(left_value).__name__,
            right_value,
            type(right_value).__name__,
            exc,
        )
    )


def _evaluate(expression: Any, context: Mapping[str, Any]) -> Any:
    if not isinstance(expression, Mapping):
        if isinstance(expression, list):
            return [_evaluate(item, context) for item in expression]
        return expression

    if len(expression) != 1:
        raise VPMValidationError(
            "Property expression mappings require exactly one operator"
        )

    operator, value = next(iter(expression.items()))
    if operator == "var":
        return _resolve_var(str(value), context)
    if operator == "all":
        if not isinstance(value, list):
            raise VPMValidationError("all requires a list")
        return all(bool(_evaluate(item, context)) for item in value)
    if operator == "any":
        if not isinstance(value, list):
            raise VPMValidationError("any requires a list")
        return any(bool(_evaluate(item, context)) for item in value)
    if operator == "not":
        return not bool(_evaluate(value, context))
    if operator == "implies":
        left, right = _binary_args(operator, value)
        return (not bool(_evaluate(left, context))) or bool(
            _evaluate(right, context)
        )

    left, right = _binary_args(operator, value)
    left_value = _evaluate(left, context)
    right_value = _evaluate(right, context)

    try:
        if operator == "eq":
            return left_value == right_value
        if operator == "ne":
            return left_value != right_value
        if operator == "lt":
            return left_value < right_value
        if operator == "lte":
            return left_value <= right_value
        if operator == "gt":
            return left_value > right_value
        if operator == "gte":
            return left_value >= right_value
        if operator == "in":
            return left_value in right_value
    except (TypeError, ValueError) as exc:
        raise _comparison_error(
            str(operator),
            left_value,
            right_value,
            exc,
        ) from exc

    raise VPMValidationError("Unsupported property operator: %s" % operator)
