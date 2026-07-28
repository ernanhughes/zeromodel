"""Structured comparison recipes for Observer transition checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id, canonical_json

OBSERVER_FEATURE_COMPARISON_VERSION: Final = "observer-feature-comparison/1"
OBSERVER_FEATURE_COMPARISON_RESULT_VERSION: Final = (
    "observer-feature-comparison-result/1"
)
OBSERVER_HIDDEN_STATE_HYPOTHESIS_VERSION: Final = "observer-hidden-state-hypothesis/1"
OBSERVER_HIDDEN_STATE_HYPOTHESIS_SET_VERSION: Final = (
    "observer-hidden-state-hypothesis-set/1"
)
OBSERVER_POLICY_CONSEQUENCE_EVIDENCE_VERSION: Final = (
    "observer-policy-consequence-evidence/1"
)
OBSERVER_COMPARISON_RECIPE_VERSION: Final = "observer-comparison-recipe/2"
OBSERVER_LEGACY_COMPARISON_RECIPE_VERSION: Final = "observer-comparison-recipe/1"
OBSERVER_COMPARISON_RESULT_VERSION: Final = "observer-comparison-result/3"

COMPARISON_MODES: Final = frozenset({"exact", "categorical", "numeric_tolerance"})
EXPECTED_TYPES: Final = frozenset({"bool", "int", "float", "number", "str", "none"})
FEATURE_RESULT_STATUSES: Final = frozenset(
    {
        "match",
        "mismatch",
        "missing_predicted",
        "missing_observed",
        "type_mismatch",
        "invalid_value",
    }
)
HYPOTHESIS_STATUSES: Final = frozenset({"possible", "eliminated", "confirmed"})
INCONCLUSIVE_REASONS: Final = frozenset(
    {
        "missing_required_feature",
        "schema_mismatch",
        "missing_policy_consequence_evidence",
        "missing_hidden_state_hypothesis_set",
        "invalid_feature_value",
    }
)


class ObserverComparisonError(ValueError):
    """Raised when an Observer comparison DTO is invalid."""


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverComparisonError(f"{field_name} must be unique and sorted")


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverComparisonError(f"{field_name} must be non-empty")


def _validate_feature_key(feature_key: str) -> None:
    if (
        not feature_key
        or "." not in feature_key
        or feature_key.split(".", 1)[0] not in {"visible", "history", "hidden"}
        or not feature_key.split(".", 1)[1]
    ):
        raise ObserverComparisonError(f"malformed feature key: {feature_key!r}")


def _type_name(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, (tuple, list)):
        return "array"
    return type(value).__name__


def _is_finite_json_value(value: object) -> bool:
    try:
        canonical_json({"value": value})
    except (TypeError, ValueError):
        return False
    return True


def _matches_expected_type(value: object, expected_type: str | None) -> bool:
    if expected_type is None:
        return True
    if expected_type == "bool":
        return isinstance(value, bool)
    if expected_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "float":
        return isinstance(value, float) and math.isfinite(value)
    if expected_type == "number":
        return (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if expected_type == "str":
        return isinstance(value, str)
    if expected_type == "none":
        return value is None
    return False


@dataclass(frozen=True)
class ObserverFeatureComparisonDTO:
    """Declared comparison semantics for one qualified feature."""

    comparison_id: str
    feature_key: str
    mode: str
    absolute_tolerance: float | None = None
    relative_tolerance: float | None = None
    expected_type: str | None = None
    version: str = OBSERVER_FEATURE_COMPARISON_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_FEATURE_COMPARISON_VERSION:
            raise ObserverComparisonError("unsupported feature comparison version")
        _validate_feature_key(self.feature_key)
        if self.mode not in COMPARISON_MODES:
            raise ObserverComparisonError(f"unsupported comparison mode: {self.mode!r}")
        if self.expected_type is not None and self.expected_type not in EXPECTED_TYPES:
            raise ObserverComparisonError(
                f"unsupported expected_type: {self.expected_type!r}"
            )
        if self.mode == "numeric_tolerance":
            if self.absolute_tolerance is None or self.relative_tolerance is None:
                raise ObserverComparisonError(
                    "numeric_tolerance requires absolute and relative tolerances"
                )
            if self.absolute_tolerance < 0.0 or self.relative_tolerance < 0.0:
                raise ObserverComparisonError("numeric tolerances must be non-negative")
        elif self.absolute_tolerance is not None or self.relative_tolerance is not None:
            raise ObserverComparisonError(
                "only numeric_tolerance comparisons may declare tolerances"
            )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.comparison_id != expected_id:
            raise ObserverComparisonError(
                "comparison_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "absolute_tolerance": self.absolute_tolerance,
            "expected_type": self.expected_type,
            "feature_key": self.feature_key,
            "mode": self.mode,
            "relative_tolerance": self.relative_tolerance,
            "version": self.version,
        }
        if include_id:
            payload["comparison_id"] = self.comparison_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        feature_key: str,
        mode: str = "exact",
        absolute_tolerance: float | None = None,
        relative_tolerance: float | None = None,
        expected_type: str | None = None,
    ) -> "ObserverFeatureComparisonDTO":
        payload = {
            "absolute_tolerance": absolute_tolerance,
            "expected_type": expected_type,
            "feature_key": feature_key,
            "mode": mode,
            "relative_tolerance": relative_tolerance,
            "version": OBSERVER_FEATURE_COMPARISON_VERSION,
        }
        return cls(
            comparison_id=canonical_id(payload),
            feature_key=feature_key,
            mode=mode,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            expected_type=expected_type,
        )


@dataclass(frozen=True)
class ObserverFeatureComparisonResultDTO:
    """Canonical result for one evaluated feature comparison."""

    result_id: str
    feature_key: str
    comparison_id: str
    status: str
    predicted_type: str | None
    observed_type: str | None
    absolute_delta: float | None = None
    tolerance_limit: float | None = None
    version: str = OBSERVER_FEATURE_COMPARISON_RESULT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_FEATURE_COMPARISON_RESULT_VERSION:
            raise ObserverComparisonError("unsupported feature result version")
        _validate_feature_key(self.feature_key)
        _require_non_empty(self.comparison_id, "comparison_id")
        if self.status not in FEATURE_RESULT_STATUSES:
            raise ObserverComparisonError(
                f"unsupported feature result status: {self.status!r}"
            )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.result_id != expected_id:
            raise ObserverComparisonError("result_id disagrees with canonical payload")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "absolute_delta": self.absolute_delta,
            "comparison_id": self.comparison_id,
            "feature_key": self.feature_key,
            "observed_type": self.observed_type,
            "predicted_type": self.predicted_type,
            "status": self.status,
            "tolerance_limit": self.tolerance_limit,
            "version": self.version,
        }
        if include_id:
            payload["result_id"] = self.result_id
        return payload


def _feature_result(
    *,
    feature_key: str,
    comparison_id: str,
    status: str,
    predicted_type: str | None,
    observed_type: str | None,
    absolute_delta: float | None = None,
    tolerance_limit: float | None = None,
) -> ObserverFeatureComparisonResultDTO:
    payload = {
        "absolute_delta": absolute_delta,
        "comparison_id": comparison_id,
        "feature_key": feature_key,
        "observed_type": observed_type,
        "predicted_type": predicted_type,
        "status": status,
        "tolerance_limit": tolerance_limit,
        "version": OBSERVER_FEATURE_COMPARISON_RESULT_VERSION,
    }
    return ObserverFeatureComparisonResultDTO(
        result_id=canonical_id(payload),
        feature_key=feature_key,
        comparison_id=comparison_id,
        status=status,
        predicted_type=predicted_type,
        observed_type=observed_type,
        absolute_delta=absolute_delta,
        tolerance_limit=tolerance_limit,
    )


@dataclass(frozen=True)
class ObserverHiddenStateHypothesisDTO:
    """Evidence-bearing hidden-state hypothesis."""

    hypothesis_id: str
    state_key: str
    state_value: object
    evidence_ids: tuple[str, ...]
    status: str
    version: str = OBSERVER_HIDDEN_STATE_HYPOTHESIS_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HIDDEN_STATE_HYPOTHESIS_VERSION:
            raise ObserverComparisonError("unsupported hidden-state hypothesis version")
        _validate_feature_key(self.state_key)
        if not self.state_key.startswith("hidden."):
            raise ObserverComparisonError("state_key must be in the hidden namespace")
        if self.status not in HYPOTHESIS_STATUSES:
            raise ObserverComparisonError(
                f"unsupported hypothesis status: {self.status!r}"
            )
        _ensure_sorted_unique(self.evidence_ids, "evidence_ids")
        if not _is_finite_json_value(self.state_value):
            raise ObserverComparisonError(
                "state_value must be canonical JSON compatible"
            )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.hypothesis_id != expected_id:
            raise ObserverComparisonError(
                "hypothesis_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "evidence_ids": list(self.evidence_ids),
            "state_key": self.state_key,
            "state_value": self.state_value,
            "status": self.status,
            "version": self.version,
        }
        if include_id:
            payload["hypothesis_id"] = self.hypothesis_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        state_key: str,
        state_value: object,
        evidence_ids: tuple[str, ...] = (),
        status: str = "possible",
    ) -> "ObserverHiddenStateHypothesisDTO":
        payload = {
            "evidence_ids": list(evidence_ids),
            "state_key": state_key,
            "state_value": state_value,
            "status": status,
            "version": OBSERVER_HIDDEN_STATE_HYPOTHESIS_VERSION,
        }
        return cls(
            hypothesis_id=canonical_id(payload),
            state_key=state_key,
            state_value=state_value,
            evidence_ids=evidence_ids,
            status=status,
        )


@dataclass(frozen=True)
class ObserverHiddenStateHypothesisSetDTO:
    """Canonical set that owns hidden-state exhaustion evidence."""

    hypothesis_set_id: str
    observation_schema_id: str
    hypotheses: tuple[ObserverHiddenStateHypothesisDTO, ...]
    derivation_evidence_ids: tuple[str, ...] = ()
    version: str = OBSERVER_HIDDEN_STATE_HYPOTHESIS_SET_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HIDDEN_STATE_HYPOTHESIS_SET_VERSION:
            raise ObserverComparisonError("unsupported hypothesis set version")
        _require_non_empty(self.observation_schema_id, "observation_schema_id")
        ids = tuple(item.hypothesis_id for item in self.hypotheses)
        if ids != tuple(sorted(set(ids))):
            raise ObserverComparisonError(
                "hypotheses must have unique IDs in sorted order"
            )
        _ensure_sorted_unique(self.derivation_evidence_ids, "derivation_evidence_ids")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.hypothesis_set_id != expected_id:
            raise ObserverComparisonError(
                "hypothesis_set_id disagrees with canonical payload"
            )

    @property
    def possible_count(self) -> int:
        return sum(1 for item in self.hypotheses if item.status == "possible")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "derivation_evidence_ids": list(self.derivation_evidence_ids),
            "hypotheses": [item.canonical_payload() for item in self.hypotheses],
            "observation_schema_id": self.observation_schema_id,
            "version": self.version,
        }
        if include_id:
            payload["hypothesis_set_id"] = self.hypothesis_set_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        observation_schema_id: str,
        hypotheses: tuple[ObserverHiddenStateHypothesisDTO, ...],
        derivation_evidence_ids: tuple[str, ...] = (),
    ) -> "ObserverHiddenStateHypothesisSetDTO":
        hypotheses = tuple(sorted(hypotheses, key=lambda item: item.hypothesis_id))
        payload = {
            "derivation_evidence_ids": list(derivation_evidence_ids),
            "hypotheses": [item.canonical_payload() for item in hypotheses],
            "observation_schema_id": observation_schema_id,
            "version": OBSERVER_HIDDEN_STATE_HYPOTHESIS_SET_VERSION,
        }
        return cls(
            hypothesis_set_id=canonical_id(payload),
            observation_schema_id=observation_schema_id,
            hypotheses=hypotheses,
            derivation_evidence_ids=derivation_evidence_ids,
        )


@dataclass(frozen=True)
class ObserverPolicyConsequenceEvidenceDTO:
    """Externally computed policy consequence evidence."""

    policy_consequence_evidence_id: str
    policy_artifact_id: str
    predicted_state_artifact_id: str
    observed_state_artifact_id: str
    predicted_selected_action: str
    observed_selected_action: str
    predicted_decision_trace_id: str
    observed_decision_trace_id: str
    equivalent: bool
    reader_contract_id: str
    version: str = OBSERVER_POLICY_CONSEQUENCE_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_POLICY_CONSEQUENCE_EVIDENCE_VERSION:
            raise ObserverComparisonError(
                "unsupported policy consequence evidence version"
            )
        for field_name in (
            "policy_artifact_id",
            "predicted_state_artifact_id",
            "observed_state_artifact_id",
            "predicted_selected_action",
            "observed_selected_action",
            "predicted_decision_trace_id",
            "observed_decision_trace_id",
            "reader_contract_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.equivalent != (
            self.predicted_selected_action == self.observed_selected_action
        ):
            raise ObserverComparisonError(
                "equivalent must match selected action equality"
            )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.policy_consequence_evidence_id != expected_id:
            raise ObserverComparisonError(
                "policy_consequence_evidence_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "equivalent": self.equivalent,
            "observed_decision_trace_id": self.observed_decision_trace_id,
            "observed_selected_action": self.observed_selected_action,
            "observed_state_artifact_id": self.observed_state_artifact_id,
            "policy_artifact_id": self.policy_artifact_id,
            "predicted_decision_trace_id": self.predicted_decision_trace_id,
            "predicted_selected_action": self.predicted_selected_action,
            "predicted_state_artifact_id": self.predicted_state_artifact_id,
            "reader_contract_id": self.reader_contract_id,
            "version": self.version,
        }
        if include_id:
            payload["policy_consequence_evidence_id"] = (
                self.policy_consequence_evidence_id
            )
        return payload

    @classmethod
    def create(
        cls,
        *,
        policy_artifact_id: str,
        predicted_state_artifact_id: str,
        observed_state_artifact_id: str,
        predicted_selected_action: str,
        observed_selected_action: str,
        predicted_decision_trace_id: str,
        observed_decision_trace_id: str,
        reader_contract_id: str,
    ) -> "ObserverPolicyConsequenceEvidenceDTO":
        payload = {
            "equivalent": predicted_selected_action == observed_selected_action,
            "observed_decision_trace_id": observed_decision_trace_id,
            "observed_selected_action": observed_selected_action,
            "observed_state_artifact_id": observed_state_artifact_id,
            "policy_artifact_id": policy_artifact_id,
            "predicted_decision_trace_id": predicted_decision_trace_id,
            "predicted_selected_action": predicted_selected_action,
            "predicted_state_artifact_id": predicted_state_artifact_id,
            "reader_contract_id": reader_contract_id,
            "version": OBSERVER_POLICY_CONSEQUENCE_EVIDENCE_VERSION,
        }
        return cls(
            policy_consequence_evidence_id=canonical_id(payload),
            policy_artifact_id=policy_artifact_id,
            predicted_state_artifact_id=predicted_state_artifact_id,
            observed_state_artifact_id=observed_state_artifact_id,
            predicted_selected_action=predicted_selected_action,
            observed_selected_action=observed_selected_action,
            predicted_decision_trace_id=predicted_decision_trace_id,
            observed_decision_trace_id=observed_decision_trace_id,
            equivalent=predicted_selected_action == observed_selected_action,
            reader_contract_id=reader_contract_id,
        )


@dataclass(frozen=True)
class ObserverComparisonRecipeDTO:
    """Declared hypothesis for comparing predicted and observed transitions."""

    recipe_id: str
    feature_comparisons: tuple[ObserverFeatureComparisonDTO, ...]
    observable_feature_keys: tuple[str, ...]
    action_effect_keys: tuple[str, ...] = ()
    hidden_state_keys: tuple[str, ...] = ()
    require_policy_consequence_evidence: bool = False
    wake_on_observable_mismatch: bool = True
    wake_on_action_effect_mismatch: bool = True
    wake_on_policy_consequence_mismatch: bool = False
    wake_on_hidden_state_exhausted: bool = True
    version: str = OBSERVER_COMPARISON_RECIPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_COMPARISON_RECIPE_VERSION:
            raise ObserverComparisonError(
                "legacy comparison recipes are not accepted by the O3.0 contract; "
                "create observer-comparison-recipe/2 with feature_comparisons"
            )
        if not self.observable_feature_keys:
            raise ObserverComparisonError("observable_feature_keys must be non-empty")
        for field_name in (
            "observable_feature_keys",
            "action_effect_keys",
            "hidden_state_keys",
        ):
            values = getattr(self, field_name)
            _ensure_sorted_unique(values, field_name)
            for key in values:
                _validate_feature_key(key)
        comparison_keys = tuple(item.feature_key for item in self.feature_comparisons)
        if comparison_keys != tuple(sorted(set(comparison_keys))):
            raise ObserverComparisonError(
                "feature_comparisons must be unique and sorted by feature_key"
            )
        required_keys = tuple(
            sorted(
                set(self.observable_feature_keys)
                | set(self.action_effect_keys)
                | set(self.hidden_state_keys)
            )
        )
        missing_specs = set(required_keys) - set(comparison_keys)
        extra_specs = set(comparison_keys) - set(required_keys)
        if missing_specs or extra_specs:
            raise ObserverComparisonError(
                "feature comparison specs must exactly match required feature keys "
                f"(missing={sorted(missing_specs)}, extra={sorted(extra_specs)})"
            )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.recipe_id != expected_id:
            raise ObserverComparisonError("recipe_id disagrees with canonical payload")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "action_effect_keys": list(self.action_effect_keys),
            "feature_comparisons": [
                item.canonical_payload() for item in self.feature_comparisons
            ],
            "hidden_state_keys": list(self.hidden_state_keys),
            "observable_feature_keys": list(self.observable_feature_keys),
            "require_policy_consequence_evidence": (
                self.require_policy_consequence_evidence
            ),
            "version": self.version,
            "wake_on_action_effect_mismatch": self.wake_on_action_effect_mismatch,
            "wake_on_hidden_state_exhausted": self.wake_on_hidden_state_exhausted,
            "wake_on_observable_mismatch": self.wake_on_observable_mismatch,
            "wake_on_policy_consequence_mismatch": (
                self.wake_on_policy_consequence_mismatch
            ),
        }
        if include_id:
            payload["recipe_id"] = self.recipe_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        feature_comparisons: tuple[ObserverFeatureComparisonDTO, ...],
        observable_feature_keys: tuple[str, ...],
        action_effect_keys: tuple[str, ...] = (),
        hidden_state_keys: tuple[str, ...] = (),
        require_policy_consequence_evidence: bool = False,
        wake_on_observable_mismatch: bool = True,
        wake_on_action_effect_mismatch: bool = True,
        wake_on_policy_consequence_mismatch: bool = False,
        wake_on_hidden_state_exhausted: bool = True,
    ) -> "ObserverComparisonRecipeDTO":
        feature_comparisons = tuple(
            sorted(feature_comparisons, key=lambda item: item.feature_key)
        )
        payload = {
            "action_effect_keys": list(action_effect_keys),
            "feature_comparisons": [
                item.canonical_payload() for item in feature_comparisons
            ],
            "hidden_state_keys": list(hidden_state_keys),
            "observable_feature_keys": list(observable_feature_keys),
            "require_policy_consequence_evidence": require_policy_consequence_evidence,
            "version": OBSERVER_COMPARISON_RECIPE_VERSION,
            "wake_on_action_effect_mismatch": wake_on_action_effect_mismatch,
            "wake_on_hidden_state_exhausted": wake_on_hidden_state_exhausted,
            "wake_on_observable_mismatch": wake_on_observable_mismatch,
            "wake_on_policy_consequence_mismatch": (
                wake_on_policy_consequence_mismatch
            ),
        }
        return cls(
            recipe_id=canonical_id(payload),
            feature_comparisons=feature_comparisons,
            observable_feature_keys=observable_feature_keys,
            action_effect_keys=action_effect_keys,
            hidden_state_keys=hidden_state_keys,
            require_policy_consequence_evidence=require_policy_consequence_evidence,
            wake_on_observable_mismatch=wake_on_observable_mismatch,
            wake_on_action_effect_mismatch=wake_on_action_effect_mismatch,
            wake_on_policy_consequence_mismatch=(wake_on_policy_consequence_mismatch),
            wake_on_hidden_state_exhausted=wake_on_hidden_state_exhausted,
        )


@dataclass(frozen=True)
class ObserverComparisonResultDTO:
    """Structured transition comparison result."""

    comparison_result_id: str
    recipe_id: str
    predicted_observation_artifact_id: str
    observed_observation_artifact_id: str
    feature_results: tuple[ObserverFeatureComparisonResultDTO, ...]
    policy_consequence_evidence_id: str | None
    hidden_state_hypothesis_set_id: str | None
    observable_feature_match: bool
    action_effect_match: bool
    policy_consequence_match: bool
    hidden_state_exhausted: bool
    wake_required: bool
    contradiction: bool
    inconclusive_reasons: tuple[str, ...] = ()
    version: str = OBSERVER_COMPARISON_RESULT_VERSION

    @property
    def next_action_equivalent(self) -> bool:
        return self.policy_consequence_match

    @property
    def mismatched_feature_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                item.feature_key
                for item in self.feature_results
                if item.status in {"mismatch", "type_mismatch", "invalid_value"}
            )
        )

    @property
    def missing_predicted_feature_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                item.feature_key
                for item in self.feature_results
                if item.status == "missing_predicted"
            )
        )

    @property
    def missing_observed_feature_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                item.feature_key
                for item in self.feature_results
                if item.status == "missing_observed"
            )
        )

    def __post_init__(self) -> None:
        if self.version != OBSERVER_COMPARISON_RESULT_VERSION:
            raise ObserverComparisonError("unsupported comparison result version")
        for field_name in (
            "recipe_id",
            "predicted_observation_artifact_id",
            "observed_observation_artifact_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        keys = tuple(item.feature_key for item in self.feature_results)
        if keys != tuple(sorted(set(keys))):
            raise ObserverComparisonError("feature_results must be unique and sorted")
        _ensure_sorted_unique(self.inconclusive_reasons, "inconclusive_reasons")
        unknown = set(self.inconclusive_reasons) - INCONCLUSIVE_REASONS
        if unknown:
            raise ObserverComparisonError(
                f"unsupported inconclusive_reasons: {sorted(unknown)}"
            )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.comparison_result_id != expected_id:
            raise ObserverComparisonError(
                "comparison_result_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "action_effect_match": self.action_effect_match,
            "contradiction": self.contradiction,
            "feature_results": [
                item.canonical_payload() for item in self.feature_results
            ],
            "hidden_state_exhausted": self.hidden_state_exhausted,
            "hidden_state_hypothesis_set_id": self.hidden_state_hypothesis_set_id,
            "inconclusive_reasons": list(self.inconclusive_reasons),
            "observable_feature_match": self.observable_feature_match,
            "observed_observation_artifact_id": self.observed_observation_artifact_id,
            "policy_consequence_evidence_id": self.policy_consequence_evidence_id,
            "policy_consequence_match": self.policy_consequence_match,
            "predicted_observation_artifact_id": (
                self.predicted_observation_artifact_id
            ),
            "recipe_id": self.recipe_id,
            "version": self.version,
            "wake_required": self.wake_required,
        }
        if include_id:
            payload["comparison_result_id"] = self.comparison_result_id
        return payload


def _compare_value(
    comparison: ObserverFeatureComparisonDTO,
    predicted: object,
    observed: object,
) -> ObserverFeatureComparisonResultDTO:
    predicted_type = _type_name(predicted)
    observed_type = _type_name(observed)
    if not _is_finite_json_value(predicted) or not _is_finite_json_value(observed):
        return _feature_result(
            feature_key=comparison.feature_key,
            comparison_id=comparison.comparison_id,
            status="invalid_value",
            predicted_type=predicted_type,
            observed_type=observed_type,
        )
    if comparison.mode == "numeric_tolerance" and (
        isinstance(predicted, bool) or isinstance(observed, bool)
    ):
        return _feature_result(
            feature_key=comparison.feature_key,
            comparison_id=comparison.comparison_id,
            status="invalid_value",
            predicted_type=predicted_type,
            observed_type=observed_type,
        )
    if not _matches_expected_type(
        predicted, comparison.expected_type
    ) or not _matches_expected_type(observed, comparison.expected_type):
        return _feature_result(
            feature_key=comparison.feature_key,
            comparison_id=comparison.comparison_id,
            status="type_mismatch",
            predicted_type=predicted_type,
            observed_type=observed_type,
        )
    if comparison.mode in {"exact", "categorical"}:
        if type(predicted) is not type(observed):
            status = "type_mismatch"
        else:
            status = "match" if predicted == observed else "mismatch"
        return _feature_result(
            feature_key=comparison.feature_key,
            comparison_id=comparison.comparison_id,
            status=status,
            predicted_type=predicted_type,
            observed_type=observed_type,
        )
    if (
        not isinstance(predicted, Real)
        or not isinstance(observed, Real)
        or isinstance(predicted, bool)
        or isinstance(observed, bool)
        or not math.isfinite(predicted)
        or not math.isfinite(observed)
    ):
        return _feature_result(
            feature_key=comparison.feature_key,
            comparison_id=comparison.comparison_id,
            status="invalid_value",
            predicted_type=predicted_type,
            observed_type=observed_type,
        )
    absolute_delta = abs(float(predicted) - float(observed))
    assert comparison.absolute_tolerance is not None
    assert comparison.relative_tolerance is not None
    tolerance_limit = max(
        comparison.absolute_tolerance,
        comparison.relative_tolerance
        * max(abs(float(predicted)), abs(float(observed))),
    )
    return _feature_result(
        feature_key=comparison.feature_key,
        comparison_id=comparison.comparison_id,
        status="match" if absolute_delta <= tolerance_limit else "mismatch",
        predicted_type=predicted_type,
        observed_type=observed_type,
        absolute_delta=absolute_delta,
        tolerance_limit=tolerance_limit,
    )


def compare_observer_transition(
    *,
    recipe: ObserverComparisonRecipeDTO,
    predicted_observation_artifact_id: str,
    observed_observation_artifact_id: str,
    predicted_observation_schema_id: str,
    observed_observation_schema_id: str,
    predicted_features: Mapping[str, object],
    observed_features: Mapping[str, object],
    hidden_state_hypothesis_set: ObserverHiddenStateHypothesisSetDTO | None = None,
    policy_consequence_evidence: ObserverPolicyConsequenceEvidenceDTO | None = None,
) -> ObserverComparisonResultDTO:
    """Compare predicted and observed state using declared evidence contracts."""

    if recipe.version != OBSERVER_COMPARISON_RECIPE_VERSION:
        raise ObserverComparisonError(
            "compare_observer_transition requires observer-comparison-recipe/2"
        )
    for field_name, value in (
        ("predicted_observation_artifact_id", predicted_observation_artifact_id),
        ("observed_observation_artifact_id", observed_observation_artifact_id),
        ("predicted_observation_schema_id", predicted_observation_schema_id),
        ("observed_observation_schema_id", observed_observation_schema_id),
    ):
        _require_non_empty(value, field_name)

    inconclusive_reasons: set[str] = set()
    if predicted_observation_schema_id != observed_observation_schema_id:
        inconclusive_reasons.add("schema_mismatch")

    comparisons = {item.feature_key: item for item in recipe.feature_comparisons}
    required_keys = tuple(
        sorted(
            set(recipe.observable_feature_keys)
            | set(recipe.action_effect_keys)
            | set(recipe.hidden_state_keys)
        )
    )
    feature_results: list[ObserverFeatureComparisonResultDTO] = []
    if "schema_mismatch" not in inconclusive_reasons:
        for key in required_keys:
            comparison = comparisons[key]
            if key not in predicted_features:
                feature_results.append(
                    _feature_result(
                        feature_key=key,
                        comparison_id=comparison.comparison_id,
                        status="missing_predicted",
                        predicted_type=None,
                        observed_type=(
                            _type_name(observed_features[key])
                            if key in observed_features
                            else None
                        ),
                    )
                )
                inconclusive_reasons.add("missing_required_feature")
            elif key not in observed_features:
                feature_results.append(
                    _feature_result(
                        feature_key=key,
                        comparison_id=comparison.comparison_id,
                        status="missing_observed",
                        predicted_type=_type_name(predicted_features[key]),
                        observed_type=None,
                    )
                )
                inconclusive_reasons.add("missing_required_feature")
            else:
                result = _compare_value(
                    comparison, predicted_features[key], observed_features[key]
                )
                feature_results.append(result)
                if result.status == "invalid_value":
                    inconclusive_reasons.add("invalid_feature_value")

    feature_results_tuple = tuple(
        sorted(feature_results, key=lambda item: item.feature_key)
    )
    observable_feature_match = all(
        item.status == "match"
        for item in feature_results_tuple
        if item.feature_key in recipe.observable_feature_keys
    ) and not any(
        item.status.startswith("missing")
        for item in feature_results_tuple
        if item.feature_key in recipe.observable_feature_keys
    )
    action_effect_match = all(
        item.status == "match"
        for item in feature_results_tuple
        if item.feature_key in recipe.action_effect_keys
    ) and not any(
        item.status.startswith("missing")
        for item in feature_results_tuple
        if item.feature_key in recipe.action_effect_keys
    )

    if recipe.hidden_state_keys and hidden_state_hypothesis_set is None:
        hidden_state_exhausted = False
        inconclusive_reasons.add("missing_hidden_state_hypothesis_set")
        hidden_set_id = None
    else:
        hidden_state_exhausted = (
            hidden_state_hypothesis_set is not None
            and hidden_state_hypothesis_set.possible_count == 0
        )
        hidden_set_id = (
            None
            if hidden_state_hypothesis_set is None
            else hidden_state_hypothesis_set.hypothesis_set_id
        )

    if (
        recipe.require_policy_consequence_evidence
        and policy_consequence_evidence is None
    ):
        policy_consequence_match = False
        policy_evidence_id = None
        inconclusive_reasons.add("missing_policy_consequence_evidence")
    else:
        policy_consequence_match = (
            True
            if policy_consequence_evidence is None
            else policy_consequence_evidence.equivalent
        )
        policy_evidence_id = (
            None
            if policy_consequence_evidence is None
            else policy_consequence_evidence.policy_consequence_evidence_id
        )

    evaluated_bad_features = any(
        item.status in {"mismatch", "type_mismatch"} for item in feature_results_tuple
    )
    contradiction = (
        "schema_mismatch" not in inconclusive_reasons
        and "missing_required_feature" not in inconclusive_reasons
        and "invalid_feature_value" not in inconclusive_reasons
        and (
            evaluated_bad_features
            or hidden_state_exhausted
            or (
                policy_consequence_evidence is not None
                and not policy_consequence_evidence.equivalent
            )
        )
    )
    wake_required = (
        bool(inconclusive_reasons)
        or (
            recipe.wake_on_observable_mismatch
            and any(
                item.status in {"mismatch", "type_mismatch"}
                for item in feature_results_tuple
                if item.feature_key in recipe.observable_feature_keys
            )
        )
        or (
            recipe.wake_on_action_effect_mismatch
            and any(
                item.status in {"mismatch", "type_mismatch"}
                for item in feature_results_tuple
                if item.feature_key in recipe.action_effect_keys
            )
        )
        or (
            recipe.wake_on_policy_consequence_mismatch
            and policy_consequence_evidence is not None
            and not policy_consequence_evidence.equivalent
        )
        or (recipe.wake_on_hidden_state_exhausted and hidden_state_exhausted)
    )
    payload = {
        "action_effect_match": action_effect_match,
        "contradiction": contradiction,
        "feature_results": [item.canonical_payload() for item in feature_results_tuple],
        "hidden_state_exhausted": hidden_state_exhausted,
        "hidden_state_hypothesis_set_id": hidden_set_id,
        "inconclusive_reasons": sorted(inconclusive_reasons),
        "observable_feature_match": observable_feature_match,
        "observed_observation_artifact_id": observed_observation_artifact_id,
        "policy_consequence_evidence_id": policy_evidence_id,
        "policy_consequence_match": policy_consequence_match,
        "predicted_observation_artifact_id": predicted_observation_artifact_id,
        "recipe_id": recipe.recipe_id,
        "version": OBSERVER_COMPARISON_RESULT_VERSION,
        "wake_required": wake_required,
    }
    return ObserverComparisonResultDTO(
        comparison_result_id=canonical_id(payload),
        recipe_id=recipe.recipe_id,
        predicted_observation_artifact_id=predicted_observation_artifact_id,
        observed_observation_artifact_id=observed_observation_artifact_id,
        feature_results=feature_results_tuple,
        policy_consequence_evidence_id=policy_evidence_id,
        hidden_state_hypothesis_set_id=hidden_set_id,
        observable_feature_match=observable_feature_match,
        action_effect_match=action_effect_match,
        policy_consequence_match=policy_consequence_match,
        hidden_state_exhausted=hidden_state_exhausted,
        wake_required=wake_required,
        contradiction=contradiction,
        inconclusive_reasons=tuple(sorted(inconclusive_reasons)),
    )
