"""Canonical Observer promotion-evidence DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence, cast

from zeromodel.observer._canonical import canonical_id

OBSERVER_PROMOTION_EVIDENCE_RECIPE_VERSION: Final = (
    "observer-promotion-evidence-recipe/1"
)
OBSERVER_RULE_CHANGE_TEST_VERSION: Final = "observer-rule-change-test/1"
OBSERVER_NOVELTY_EVIDENCE_VERSION: Final = "observer-novelty-evidence/1"
OBSERVER_RULE_REGIME_VERSION: Final = "observer-rule-regime/1"
OBSERVER_TRANSITION_OCCURRENCE_VERSION: Final = "observer-transition-occurrence/1"
OBSERVER_TRANSITION_RECURRENCE_VERSION: Final = "observer-transition-recurrence/1"
OBSERVER_TRANSITION_STABILITY_VERSION: Final = "observer-transition-stability/1"
OBSERVER_EVIDENCE_INDEPENDENCE_VERSION: Final = "observer-evidence-independence/1"
OBSERVER_RULE_CHANGE_SURVIVAL_VERSION: Final = "observer-rule-change-survival/1"
OBSERVER_PROMOTION_CANDIDATE_VERSION: Final = "observer-promotion-candidate/1"
OBSERVER_PROMOTION_ANALYSIS_VERSION: Final = "observer-promotion-analysis/1"

VERIFICATION_STATUSES: Final = frozenset({"confirmed", "contradicted", "inconclusive"})
NOVELTY_STATUSES: Final = frozenset({"novel", "recurrent", "unknown"})
REGIME_KINDS: Final = frozenset(
    {"aligned", "environment_changed", "predictor_changed", "both_changed"}
)
STABILITY_STATUSES: Final = frozenset(
    {"stable", "unstable", "insufficient_evidence", "mixed"}
)
INDEPENDENCE_STATUSES: Final = frozenset({"sufficient", "insufficient"})
RULE_CHANGE_STATUSES: Final = frozenset({"survived", "failed", "not_tested"})
PROMOTION_DISPOSITIONS: Final = frozenset(
    {
        "eligible",
        "insufficient_evidence",
        "unstable",
        "contradicted",
        "not_independent",
        "not_rule_change_tested",
        "unsupported",
    }
)
PROMOTION_ANALYSIS_STATUSES: Final = frozenset({"built", "failed", "inconclusive"})
PROMOTION_ANALYSIS_FAILURE_CODES: Final = frozenset(
    {
        "failed_graph_build",
        "missing_graph",
        "graph_ledger_mismatch",
        "graph_grouping_mismatch",
        "schema_mismatch",
        "rule_change_test_missing",
        "edge_support_missing",
        "edge_occurrence_mismatch",
        "edge_count_mismatch",
        "transition_key_mismatch",
        "duplicate_occurrence",
    }
)
STABILITY_REASON_CODES: Final = frozenset(
    {
        "minimum_traversals_met",
        "minimum_traversals_not_met",
        "minimum_confirmations_met",
        "minimum_confirmations_not_met",
        "contradiction_limit_met",
        "contradiction_limit_exceeded",
        "inconclusive_limit_met",
        "inconclusive_limit_exceeded",
        "confirmation_ratio_met",
        "confirmation_ratio_not_met",
        "no_evaluated_evidence",
    }
)
INDEPENDENCE_REASON_CODES: Final = frozenset(
    {
        "enough_episodes",
        "not_enough_episodes",
        "enough_source_observations",
        "not_enough_source_observations",
        "enough_rule_regimes",
        "not_enough_rule_regimes",
    }
)
RULE_CHANGE_REASON_CODES: Final = frozenset(
    {
        "pre_change_observed",
        "post_change_observed",
        "post_change_confirmed",
        "post_change_contradicted",
        "rule_change_not_observed",
        "post_change_confirmation_missing",
    }
)
PROMOTION_REASON_CODES: Final = frozenset(
    {
        "minimum_traversals_met",
        "minimum_traversals_not_met",
        "minimum_confirmations_met",
        "minimum_confirmations_not_met",
        "episode_independence_met",
        "episode_independence_not_met",
        "source_diversity_met",
        "source_diversity_not_met",
        "rule_regime_requirement_met",
        "rule_regime_requirement_not_met",
        "contradiction_limit_met",
        "contradiction_limit_exceeded",
        "inconclusive_limit_met",
        "inconclusive_limit_exceeded",
        "confirmation_ratio_met",
        "confirmation_ratio_not_met",
        "rule_change_survival_met",
        "rule_change_survival_not_met",
        "stability_met",
        "stability_not_met",
    }
)


class ObserverPromotionAnalysisError(ValueError):
    """Raised when promotion-evidence contracts are malformed."""


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverPromotionAnalysisError(f"{field_name} must be non-empty")


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverPromotionAnalysisError(f"{field_name} must be unique and sorted")


def _ensure_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        raise ObserverPromotionAnalysisError(f"{field_name} must be non-negative")


@dataclass(frozen=True)
class ObserverRuleChangeTestDTO:
    rule_change_test_id: str
    baseline_environment_rule_set_id: str
    changed_environment_rule_set_id: str
    change_start_ledger_sequence: int
    predictor_rule_set_id: str | None
    version: str = OBSERVER_RULE_CHANGE_TEST_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_RULE_CHANGE_TEST_VERSION:
            raise ObserverPromotionAnalysisError("unsupported rule-change test version")
        _require_non_empty(
            self.baseline_environment_rule_set_id,
            "baseline_environment_rule_set_id",
        )
        _require_non_empty(
            self.changed_environment_rule_set_id,
            "changed_environment_rule_set_id",
        )
        if (
            self.baseline_environment_rule_set_id
            == self.changed_environment_rule_set_id
        ):
            raise ObserverPromotionAnalysisError(
                "rule-change test requires distinct environment rules"
            )
        _ensure_non_negative(
            self.change_start_ledger_sequence, "change_start_ledger_sequence"
        )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.rule_change_test_id != expected_id:
            raise ObserverPromotionAnalysisError("rule_change_test_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = {
            "baseline_environment_rule_set_id": self.baseline_environment_rule_set_id,
            "changed_environment_rule_set_id": self.changed_environment_rule_set_id,
            "change_start_ledger_sequence": self.change_start_ledger_sequence,
            "predictor_rule_set_id": self.predictor_rule_set_id,
            "version": self.version,
        }
        if include_id:
            payload["rule_change_test_id"] = self.rule_change_test_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        baseline_environment_rule_set_id: str,
        changed_environment_rule_set_id: str,
        change_start_ledger_sequence: int,
        predictor_rule_set_id: str | None = None,
    ) -> "ObserverRuleChangeTestDTO":
        payload = {
            "baseline_environment_rule_set_id": baseline_environment_rule_set_id,
            "changed_environment_rule_set_id": changed_environment_rule_set_id,
            "change_start_ledger_sequence": change_start_ledger_sequence,
            "predictor_rule_set_id": predictor_rule_set_id,
            "version": OBSERVER_RULE_CHANGE_TEST_VERSION,
        }
        return cls(
            rule_change_test_id=canonical_id(payload),
            baseline_environment_rule_set_id=baseline_environment_rule_set_id,
            changed_environment_rule_set_id=changed_environment_rule_set_id,
            change_start_ledger_sequence=change_start_ledger_sequence,
            predictor_rule_set_id=predictor_rule_set_id,
        )


@dataclass(frozen=True)
class ObserverPromotionEvidenceRecipeDTO:
    promotion_recipe_id: str
    observation_graph_id: str
    grouping_recipe_id: str
    minimum_traversal_count: int
    minimum_confirmed_count: int
    minimum_independent_episode_count: int
    minimum_distinct_source_state_count: int
    minimum_distinct_rule_regime_count: int
    maximum_contradicted_count: int
    maximum_inconclusive_count: int
    minimum_confirmation_ratio_numerator: int
    minimum_confirmation_ratio_denominator: int
    require_post_rule_change_confirmation: bool
    rule_change_test_id: str | None
    version: str = OBSERVER_PROMOTION_EVIDENCE_RECIPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_PROMOTION_EVIDENCE_RECIPE_VERSION:
            raise ObserverPromotionAnalysisError("unsupported promotion recipe version")
        _require_non_empty(self.observation_graph_id, "observation_graph_id")
        _require_non_empty(self.grouping_recipe_id, "grouping_recipe_id")
        for field_name in (
            "minimum_traversal_count",
            "minimum_confirmed_count",
            "minimum_independent_episode_count",
            "minimum_distinct_source_state_count",
            "minimum_distinct_rule_regime_count",
            "maximum_contradicted_count",
            "maximum_inconclusive_count",
            "minimum_confirmation_ratio_numerator",
            "minimum_confirmation_ratio_denominator",
        ):
            _ensure_non_negative(getattr(self, field_name), field_name)
        if self.minimum_confirmation_ratio_denominator <= 0:
            raise ObserverPromotionAnalysisError(
                "minimum_confirmation_ratio_denominator must be positive"
            )
        if (
            self.minimum_confirmation_ratio_numerator
            > self.minimum_confirmation_ratio_denominator
        ):
            raise ObserverPromotionAnalysisError("minimum confirmation ratio exceeds 1")
        if (
            self.require_post_rule_change_confirmation
            and self.rule_change_test_id is None
        ):
            raise ObserverPromotionAnalysisError(
                "rule-change confirmation requires a rule_change_test_id"
            )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.promotion_recipe_id != expected_id:
            raise ObserverPromotionAnalysisError(
                "promotion_recipe_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = {
            "grouping_recipe_id": self.grouping_recipe_id,
            "maximum_contradicted_count": self.maximum_contradicted_count,
            "maximum_inconclusive_count": self.maximum_inconclusive_count,
            "minimum_confirmation_ratio_denominator": (
                self.minimum_confirmation_ratio_denominator
            ),
            "minimum_confirmation_ratio_numerator": (
                self.minimum_confirmation_ratio_numerator
            ),
            "minimum_confirmed_count": self.minimum_confirmed_count,
            "minimum_distinct_rule_regime_count": (
                self.minimum_distinct_rule_regime_count
            ),
            "minimum_distinct_source_state_count": (
                self.minimum_distinct_source_state_count
            ),
            "minimum_independent_episode_count": (
                self.minimum_independent_episode_count
            ),
            "minimum_traversal_count": self.minimum_traversal_count,
            "observation_graph_id": self.observation_graph_id,
            "require_post_rule_change_confirmation": (
                self.require_post_rule_change_confirmation
            ),
            "rule_change_test_id": self.rule_change_test_id,
            "version": self.version,
        }
        if include_id:
            payload["promotion_recipe_id"] = self.promotion_recipe_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        observation_graph_id: str,
        grouping_recipe_id: str,
        minimum_traversal_count: int = 1,
        minimum_confirmed_count: int = 1,
        minimum_independent_episode_count: int = 1,
        minimum_distinct_source_state_count: int = 1,
        minimum_distinct_rule_regime_count: int = 1,
        maximum_contradicted_count: int = 0,
        maximum_inconclusive_count: int = 0,
        minimum_confirmation_ratio_numerator: int = 1,
        minimum_confirmation_ratio_denominator: int = 1,
        require_post_rule_change_confirmation: bool = False,
        rule_change_test_id: str | None = None,
    ) -> "ObserverPromotionEvidenceRecipeDTO":
        payload = {
            "grouping_recipe_id": grouping_recipe_id,
            "maximum_contradicted_count": maximum_contradicted_count,
            "maximum_inconclusive_count": maximum_inconclusive_count,
            "minimum_confirmation_ratio_denominator": (
                minimum_confirmation_ratio_denominator
            ),
            "minimum_confirmation_ratio_numerator": (
                minimum_confirmation_ratio_numerator
            ),
            "minimum_confirmed_count": minimum_confirmed_count,
            "minimum_distinct_rule_regime_count": minimum_distinct_rule_regime_count,
            "minimum_distinct_source_state_count": minimum_distinct_source_state_count,
            "minimum_independent_episode_count": minimum_independent_episode_count,
            "minimum_traversal_count": minimum_traversal_count,
            "observation_graph_id": observation_graph_id,
            "require_post_rule_change_confirmation": (
                require_post_rule_change_confirmation
            ),
            "rule_change_test_id": rule_change_test_id,
            "version": OBSERVER_PROMOTION_EVIDENCE_RECIPE_VERSION,
        }
        return cls(
            promotion_recipe_id=canonical_id(payload),
            observation_graph_id=observation_graph_id,
            grouping_recipe_id=grouping_recipe_id,
            minimum_traversal_count=minimum_traversal_count,
            minimum_confirmed_count=minimum_confirmed_count,
            minimum_independent_episode_count=minimum_independent_episode_count,
            minimum_distinct_source_state_count=minimum_distinct_source_state_count,
            minimum_distinct_rule_regime_count=minimum_distinct_rule_regime_count,
            maximum_contradicted_count=maximum_contradicted_count,
            maximum_inconclusive_count=maximum_inconclusive_count,
            minimum_confirmation_ratio_numerator=minimum_confirmation_ratio_numerator,
            minimum_confirmation_ratio_denominator=minimum_confirmation_ratio_denominator,
            require_post_rule_change_confirmation=require_post_rule_change_confirmation,
            rule_change_test_id=rule_change_test_id,
        )


def _payload_with_version(version: str, **values: object) -> dict[str, object]:
    payload = dict(values)
    payload["version"] = version
    return payload


def _canonical_tuple(items: tuple[object, ...]) -> list[object]:
    return [
        item.canonical_payload() if hasattr(item, "canonical_payload") else item
        for item in items
    ]


@dataclass(frozen=True)
class ObserverNoveltyEvidenceDTO:
    novelty_evidence_id: str
    ledger_snapshot_id: str
    observation_graph_id: str
    state_class_id: str
    first_observation_artifact_id: str
    first_ledger_entry_id: str
    first_ledger_sequence: int
    previously_observed: bool
    prior_observation_count: int
    novelty_status: str
    version: str = OBSERVER_NOVELTY_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_NOVELTY_EVIDENCE_VERSION:
            raise ObserverPromotionAnalysisError("unsupported novelty version")
        for field_name in (
            "ledger_snapshot_id",
            "observation_graph_id",
            "state_class_id",
            "first_observation_artifact_id",
            "first_ledger_entry_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        _ensure_non_negative(self.first_ledger_sequence, "first_ledger_sequence")
        _ensure_non_negative(self.prior_observation_count, "prior_observation_count")
        if self.novelty_status not in NOVELTY_STATUSES:
            raise ObserverPromotionAnalysisError("unsupported novelty status")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.novelty_evidence_id != expected_id:
            raise ObserverPromotionAnalysisError("novelty id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            first_ledger_entry_id=self.first_ledger_entry_id,
            first_ledger_sequence=self.first_ledger_sequence,
            first_observation_artifact_id=self.first_observation_artifact_id,
            ledger_snapshot_id=self.ledger_snapshot_id,
            novelty_status=self.novelty_status,
            observation_graph_id=self.observation_graph_id,
            previously_observed=self.previously_observed,
            prior_observation_count=self.prior_observation_count,
            state_class_id=self.state_class_id,
        )
        if include_id:
            payload["novelty_evidence_id"] = self.novelty_evidence_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverNoveltyEvidenceDTO":
        payload = _payload_with_version(OBSERVER_NOVELTY_EVIDENCE_VERSION, **values)
        return cls(novelty_evidence_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverRuleRegimeDTO:
    rule_regime_id: str
    predictor_rule_set_id: str
    environment_rule_set_id: str
    regime_kind: str
    version: str = OBSERVER_RULE_REGIME_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_RULE_REGIME_VERSION:
            raise ObserverPromotionAnalysisError("unsupported rule regime version")
        _require_non_empty(self.predictor_rule_set_id, "predictor_rule_set_id")
        _require_non_empty(self.environment_rule_set_id, "environment_rule_set_id")
        if self.regime_kind not in REGIME_KINDS:
            raise ObserverPromotionAnalysisError("unsupported regime kind")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.rule_regime_id != expected_id:
            raise ObserverPromotionAnalysisError("rule_regime_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            environment_rule_set_id=self.environment_rule_set_id,
            predictor_rule_set_id=self.predictor_rule_set_id,
            regime_kind=self.regime_kind,
        )
        if include_id:
            payload["rule_regime_id"] = self.rule_regime_id
        return payload

    @classmethod
    def create(
        cls, *, predictor_rule_set_id: str, environment_rule_set_id: str
    ) -> "ObserverRuleRegimeDTO":
        regime_kind = (
            "aligned"
            if predictor_rule_set_id == environment_rule_set_id
            else "environment_changed"
        )
        payload = _payload_with_version(
            OBSERVER_RULE_REGIME_VERSION,
            environment_rule_set_id=environment_rule_set_id,
            predictor_rule_set_id=predictor_rule_set_id,
            regime_kind=regime_kind,
        )
        return cls(
            rule_regime_id=canonical_id(payload),
            predictor_rule_set_id=predictor_rule_set_id,
            environment_rule_set_id=environment_rule_set_id,
            regime_kind=regime_kind,
        )


@dataclass(frozen=True)
class ObserverTransitionOccurrenceDTO:
    occurrence_id: str
    transition_key_id: str
    graph_edge_id: str
    ledger_entry_id: str
    ledger_sequence: int
    episode_id: str
    source_state_class_id: str
    source_observation_artifact_id: str
    target_state_class_id: str
    target_observation_artifact_id: str
    action: str
    verification_status: str
    comparison_result_id: str
    predictor_rule_set_id: str
    environment_rule_set_id: str
    rule_regime_id: str
    version: str = OBSERVER_TRANSITION_OCCURRENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_TRANSITION_OCCURRENCE_VERSION:
            raise ObserverPromotionAnalysisError("unsupported occurrence version")
        for field_name in (
            "transition_key_id",
            "graph_edge_id",
            "ledger_entry_id",
            "episode_id",
            "source_state_class_id",
            "source_observation_artifact_id",
            "target_state_class_id",
            "target_observation_artifact_id",
            "action",
            "comparison_result_id",
            "predictor_rule_set_id",
            "environment_rule_set_id",
            "rule_regime_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.verification_status not in VERIFICATION_STATUSES:
            raise ObserverPromotionAnalysisError("unsupported verification status")
        _ensure_non_negative(self.ledger_sequence, "ledger_sequence")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.occurrence_id != expected_id:
            raise ObserverPromotionAnalysisError("occurrence_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            action=self.action,
            comparison_result_id=self.comparison_result_id,
            environment_rule_set_id=self.environment_rule_set_id,
            episode_id=self.episode_id,
            graph_edge_id=self.graph_edge_id,
            ledger_entry_id=self.ledger_entry_id,
            ledger_sequence=self.ledger_sequence,
            predictor_rule_set_id=self.predictor_rule_set_id,
            rule_regime_id=self.rule_regime_id,
            source_observation_artifact_id=self.source_observation_artifact_id,
            source_state_class_id=self.source_state_class_id,
            target_observation_artifact_id=self.target_observation_artifact_id,
            target_state_class_id=self.target_state_class_id,
            transition_key_id=self.transition_key_id,
            verification_status=self.verification_status,
        )
        if include_id:
            payload["occurrence_id"] = self.occurrence_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverTransitionOccurrenceDTO":
        payload = _payload_with_version(
            OBSERVER_TRANSITION_OCCURRENCE_VERSION, **values
        )
        return cls(occurrence_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverTransitionRecurrenceDTO:
    recurrence_id: str
    transition_key_id: str
    observation_graph_id: str
    occurrence_ids: tuple[str, ...]
    supporting_ledger_entry_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    source_observation_artifact_ids: tuple[str, ...]
    target_observation_artifact_ids: tuple[str, ...]
    predictor_rule_set_ids: tuple[str, ...]
    environment_rule_set_ids: tuple[str, ...]
    rule_regime_ids: tuple[str, ...]
    traversal_count: int
    independent_episode_count: int
    distinct_source_observation_count: int
    distinct_target_observation_count: int
    distinct_rule_regime_count: int
    first_ledger_sequence: int
    last_ledger_sequence: int
    version: str = OBSERVER_TRANSITION_RECURRENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_TRANSITION_RECURRENCE_VERSION:
            raise ObserverPromotionAnalysisError("unsupported recurrence version")
        for field_name in (
            "occurrence_ids",
            "supporting_ledger_entry_ids",
            "episode_ids",
            "source_observation_artifact_ids",
            "target_observation_artifact_ids",
            "predictor_rule_set_ids",
            "environment_rule_set_ids",
            "rule_regime_ids",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        if self.traversal_count != len(self.occurrence_ids):
            raise ObserverPromotionAnalysisError("traversal_count mismatch")
        if self.independent_episode_count != len(self.episode_ids):
            raise ObserverPromotionAnalysisError("episode count mismatch")
        if self.distinct_source_observation_count != len(
            self.source_observation_artifact_ids
        ):
            raise ObserverPromotionAnalysisError("source observation count mismatch")
        if self.distinct_target_observation_count != len(
            self.target_observation_artifact_ids
        ):
            raise ObserverPromotionAnalysisError("target observation count mismatch")
        if self.distinct_rule_regime_count != len(self.rule_regime_ids):
            raise ObserverPromotionAnalysisError("rule regime count mismatch")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.recurrence_id != expected_id:
            raise ObserverPromotionAnalysisError("recurrence_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            distinct_rule_regime_count=self.distinct_rule_regime_count,
            distinct_source_observation_count=self.distinct_source_observation_count,
            distinct_target_observation_count=self.distinct_target_observation_count,
            environment_rule_set_ids=list(self.environment_rule_set_ids),
            episode_ids=list(self.episode_ids),
            first_ledger_sequence=self.first_ledger_sequence,
            independent_episode_count=self.independent_episode_count,
            last_ledger_sequence=self.last_ledger_sequence,
            observation_graph_id=self.observation_graph_id,
            occurrence_ids=list(self.occurrence_ids),
            predictor_rule_set_ids=list(self.predictor_rule_set_ids),
            rule_regime_ids=list(self.rule_regime_ids),
            source_observation_artifact_ids=list(self.source_observation_artifact_ids),
            supporting_ledger_entry_ids=list(self.supporting_ledger_entry_ids),
            target_observation_artifact_ids=list(self.target_observation_artifact_ids),
            transition_key_id=self.transition_key_id,
            traversal_count=self.traversal_count,
        )
        if include_id:
            payload["recurrence_id"] = self.recurrence_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverTransitionRecurrenceDTO":
        payload = _payload_with_version(
            OBSERVER_TRANSITION_RECURRENCE_VERSION, **values
        )
        return cls(recurrence_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverTransitionStabilityDTO:
    stability_id: str
    transition_key_id: str
    recurrence_id: str
    confirmed_occurrence_ids: tuple[str, ...]
    contradicted_occurrence_ids: tuple[str, ...]
    inconclusive_occurrence_ids: tuple[str, ...]
    confirmed_count: int
    contradicted_count: int
    inconclusive_count: int
    evaluated_count: int
    confirmation_ratio_numerator: int
    confirmation_ratio_denominator: int
    status: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_TRANSITION_STABILITY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_TRANSITION_STABILITY_VERSION:
            raise ObserverPromotionAnalysisError("unsupported stability version")
        for field_name in (
            "confirmed_occurrence_ids",
            "contradicted_occurrence_ids",
            "inconclusive_occurrence_ids",
            "reason_codes",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        if self.confirmed_count != len(self.confirmed_occurrence_ids):
            raise ObserverPromotionAnalysisError("confirmed_count mismatch")
        if self.contradicted_count != len(self.contradicted_occurrence_ids):
            raise ObserverPromotionAnalysisError("contradicted_count mismatch")
        if self.inconclusive_count != len(self.inconclusive_occurrence_ids):
            raise ObserverPromotionAnalysisError("inconclusive_count mismatch")
        if self.evaluated_count != self.confirmed_count + self.contradicted_count:
            raise ObserverPromotionAnalysisError("evaluated_count mismatch")
        if self.confirmation_ratio_numerator != self.confirmed_count:
            raise ObserverPromotionAnalysisError("ratio numerator mismatch")
        if self.confirmation_ratio_denominator != self.evaluated_count:
            raise ObserverPromotionAnalysisError("ratio denominator mismatch")
        if self.status not in STABILITY_STATUSES:
            raise ObserverPromotionAnalysisError("unsupported stability status")
        if set(self.reason_codes) - STABILITY_REASON_CODES:
            raise ObserverPromotionAnalysisError("unsupported stability reason")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.stability_id != expected_id:
            raise ObserverPromotionAnalysisError("stability_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            confirmation_ratio_denominator=self.confirmation_ratio_denominator,
            confirmation_ratio_numerator=self.confirmation_ratio_numerator,
            confirmed_count=self.confirmed_count,
            confirmed_occurrence_ids=list(self.confirmed_occurrence_ids),
            contradicted_count=self.contradicted_count,
            contradicted_occurrence_ids=list(self.contradicted_occurrence_ids),
            evaluated_count=self.evaluated_count,
            inconclusive_count=self.inconclusive_count,
            inconclusive_occurrence_ids=list(self.inconclusive_occurrence_ids),
            reason_codes=list(self.reason_codes),
            recurrence_id=self.recurrence_id,
            status=self.status,
            transition_key_id=self.transition_key_id,
        )
        if include_id:
            payload["stability_id"] = self.stability_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverTransitionStabilityDTO":
        payload = _payload_with_version(OBSERVER_TRANSITION_STABILITY_VERSION, **values)
        return cls(stability_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverEvidenceIndependenceDTO:
    independence_id: str
    transition_key_id: str
    episode_ids: tuple[str, ...]
    source_observation_artifact_ids: tuple[str, ...]
    source_state_class_ids: tuple[str, ...]
    rule_regime_ids: tuple[str, ...]
    independent_episode_count: int
    distinct_source_observation_count: int
    distinct_rule_regime_count: int
    status: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_EVIDENCE_INDEPENDENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_EVIDENCE_INDEPENDENCE_VERSION:
            raise ObserverPromotionAnalysisError("unsupported independence version")
        for field_name in (
            "episode_ids",
            "source_observation_artifact_ids",
            "source_state_class_ids",
            "rule_regime_ids",
            "reason_codes",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        if self.independent_episode_count != len(self.episode_ids):
            raise ObserverPromotionAnalysisError("independent episode count mismatch")
        if self.distinct_source_observation_count != len(
            self.source_observation_artifact_ids
        ):
            raise ObserverPromotionAnalysisError("source observation count mismatch")
        if self.distinct_rule_regime_count != len(self.rule_regime_ids):
            raise ObserverPromotionAnalysisError("rule regime count mismatch")
        if self.status not in INDEPENDENCE_STATUSES:
            raise ObserverPromotionAnalysisError("unsupported independence status")
        if set(self.reason_codes) - INDEPENDENCE_REASON_CODES:
            raise ObserverPromotionAnalysisError("unsupported independence reason")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.independence_id != expected_id:
            raise ObserverPromotionAnalysisError("independence_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            distinct_rule_regime_count=self.distinct_rule_regime_count,
            distinct_source_observation_count=self.distinct_source_observation_count,
            episode_ids=list(self.episode_ids),
            independent_episode_count=self.independent_episode_count,
            reason_codes=list(self.reason_codes),
            rule_regime_ids=list(self.rule_regime_ids),
            source_observation_artifact_ids=list(self.source_observation_artifact_ids),
            source_state_class_ids=list(self.source_state_class_ids),
            status=self.status,
            transition_key_id=self.transition_key_id,
        )
        if include_id:
            payload["independence_id"] = self.independence_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverEvidenceIndependenceDTO":
        payload = _payload_with_version(
            OBSERVER_EVIDENCE_INDEPENDENCE_VERSION, **values
        )
        return cls(independence_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverRuleChangeSurvivalDTO:
    rule_change_survival_id: str
    transition_key_id: str
    pre_change_occurrence_ids: tuple[str, ...]
    post_change_occurrence_ids: tuple[str, ...]
    post_change_confirmed_occurrence_ids: tuple[str, ...]
    post_change_contradicted_occurrence_ids: tuple[str, ...]
    survived_rule_change: bool
    status: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_RULE_CHANGE_SURVIVAL_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_RULE_CHANGE_SURVIVAL_VERSION:
            raise ObserverPromotionAnalysisError("unsupported rule-change version")
        for field_name in (
            "pre_change_occurrence_ids",
            "post_change_occurrence_ids",
            "post_change_confirmed_occurrence_ids",
            "post_change_contradicted_occurrence_ids",
            "reason_codes",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        if self.status not in RULE_CHANGE_STATUSES:
            raise ObserverPromotionAnalysisError("unsupported rule-change status")
        if set(self.reason_codes) - RULE_CHANGE_REASON_CODES:
            raise ObserverPromotionAnalysisError("unsupported rule-change reason")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.rule_change_survival_id != expected_id:
            raise ObserverPromotionAnalysisError("rule_change_survival_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            post_change_confirmed_occurrence_ids=list(
                self.post_change_confirmed_occurrence_ids
            ),
            post_change_contradicted_occurrence_ids=list(
                self.post_change_contradicted_occurrence_ids
            ),
            post_change_occurrence_ids=list(self.post_change_occurrence_ids),
            pre_change_occurrence_ids=list(self.pre_change_occurrence_ids),
            reason_codes=list(self.reason_codes),
            status=self.status,
            survived_rule_change=self.survived_rule_change,
            transition_key_id=self.transition_key_id,
        )
        if include_id:
            payload["rule_change_survival_id"] = self.rule_change_survival_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverRuleChangeSurvivalDTO":
        payload = _payload_with_version(OBSERVER_RULE_CHANGE_SURVIVAL_VERSION, **values)
        return cls(rule_change_survival_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverPromotionCandidateDTO:
    promotion_candidate_id: str
    promotion_recipe_id: str
    ledger_snapshot_id: str
    observation_graph_id: str
    transition_key_id: str
    graph_edge_id: str
    recurrence_id: str
    stability_id: str
    independence_id: str
    rule_change_survival_id: str | None
    disposition: str
    eligible_for_compilation: bool
    reason_codes: tuple[str, ...]
    supporting_occurrence_ids: tuple[str, ...]
    supporting_ledger_entry_ids: tuple[str, ...]
    version: str = OBSERVER_PROMOTION_CANDIDATE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_PROMOTION_CANDIDATE_VERSION:
            raise ObserverPromotionAnalysisError("unsupported candidate version")
        for field_name in (
            "promotion_recipe_id",
            "ledger_snapshot_id",
            "observation_graph_id",
            "transition_key_id",
            "graph_edge_id",
            "recurrence_id",
            "stability_id",
            "independence_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.disposition not in PROMOTION_DISPOSITIONS:
            raise ObserverPromotionAnalysisError("unsupported disposition")
        if self.eligible_for_compilation != (self.disposition == "eligible"):
            raise ObserverPromotionAnalysisError("eligibility/disposition mismatch")
        for field_name in (
            "reason_codes",
            "supporting_occurrence_ids",
            "supporting_ledger_entry_ids",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        if set(self.reason_codes) - PROMOTION_REASON_CODES:
            raise ObserverPromotionAnalysisError("unsupported candidate reason")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.promotion_candidate_id != expected_id:
            raise ObserverPromotionAnalysisError("candidate id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            disposition=self.disposition,
            eligible_for_compilation=self.eligible_for_compilation,
            graph_edge_id=self.graph_edge_id,
            independence_id=self.independence_id,
            ledger_snapshot_id=self.ledger_snapshot_id,
            observation_graph_id=self.observation_graph_id,
            promotion_recipe_id=self.promotion_recipe_id,
            reason_codes=list(self.reason_codes),
            recurrence_id=self.recurrence_id,
            rule_change_survival_id=self.rule_change_survival_id,
            stability_id=self.stability_id,
            supporting_ledger_entry_ids=list(self.supporting_ledger_entry_ids),
            supporting_occurrence_ids=list(self.supporting_occurrence_ids),
            transition_key_id=self.transition_key_id,
        )
        if include_id:
            payload["promotion_candidate_id"] = self.promotion_candidate_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverPromotionCandidateDTO":
        payload = _payload_with_version(OBSERVER_PROMOTION_CANDIDATE_VERSION, **values)
        return cls(promotion_candidate_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverPromotionAnalysisDTO:
    promotion_analysis_id: str
    ledger_snapshot_id: str
    observation_graph_id: str
    grouping_recipe_id: str
    promotion_recipe_id: str
    novelty_evidence: tuple[ObserverNoveltyEvidenceDTO, ...]
    occurrences: tuple[ObserverTransitionOccurrenceDTO, ...]
    recurrences: tuple[ObserverTransitionRecurrenceDTO, ...]
    stabilities: tuple[ObserverTransitionStabilityDTO, ...]
    independence_results: tuple[ObserverEvidenceIndependenceDTO, ...]
    rule_change_results: tuple[ObserverRuleChangeSurvivalDTO, ...]
    promotion_candidates: tuple[ObserverPromotionCandidateDTO, ...]
    eligible_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    status: str
    failure_codes: tuple[str, ...]
    version: str = OBSERVER_PROMOTION_ANALYSIS_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_PROMOTION_ANALYSIS_VERSION:
            raise ObserverPromotionAnalysisError("unsupported analysis version")
        if self.status not in PROMOTION_ANALYSIS_STATUSES:
            raise ObserverPromotionAnalysisError("unsupported analysis status")
        _ensure_sorted_unique(self.eligible_candidate_ids, "eligible_candidate_ids")
        _ensure_sorted_unique(self.rejected_candidate_ids, "rejected_candidate_ids")
        _ensure_sorted_unique(self.failure_codes, "failure_codes")
        if set(self.failure_codes) - PROMOTION_ANALYSIS_FAILURE_CODES:
            raise ObserverPromotionAnalysisError("unsupported analysis failure code")
        candidate_ids = tuple(
            sorted(item.promotion_candidate_id for item in self.promotion_candidates)
        )
        if (
            tuple(sorted(self.eligible_candidate_ids + self.rejected_candidate_ids))
            != candidate_ids
        ):
            raise ObserverPromotionAnalysisError("candidate id partition mismatch")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.promotion_analysis_id != expected_id:
            raise ObserverPromotionAnalysisError("analysis id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            eligible_candidate_ids=list(self.eligible_candidate_ids),
            failure_codes=list(self.failure_codes),
            grouping_recipe_id=self.grouping_recipe_id,
            independence_results=_canonical_tuple(self.independence_results),
            ledger_snapshot_id=self.ledger_snapshot_id,
            novelty_evidence=_canonical_tuple(self.novelty_evidence),
            observation_graph_id=self.observation_graph_id,
            occurrences=_canonical_tuple(self.occurrences),
            promotion_candidates=_canonical_tuple(self.promotion_candidates),
            promotion_recipe_id=self.promotion_recipe_id,
            recurrences=_canonical_tuple(self.recurrences),
            rejected_candidate_ids=list(self.rejected_candidate_ids),
            rule_change_results=_canonical_tuple(self.rule_change_results),
            stabilities=_canonical_tuple(self.stabilities),
            status=self.status,
        )
        if include_id:
            payload["promotion_analysis_id"] = self.promotion_analysis_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverPromotionAnalysisDTO":
        eligible_candidate_ids = cast(Sequence[str], values["eligible_candidate_ids"])
        failure_codes = cast(Sequence[str], values["failure_codes"])
        rejected_candidate_ids = cast(Sequence[str], values["rejected_candidate_ids"])
        payload = _payload_with_version(
            OBSERVER_PROMOTION_ANALYSIS_VERSION,
            eligible_candidate_ids=list(eligible_candidate_ids),
            failure_codes=list(failure_codes),
            grouping_recipe_id=values["grouping_recipe_id"],
            independence_results=_canonical_tuple(values["independence_results"]),  # type: ignore[arg-type]
            ledger_snapshot_id=values["ledger_snapshot_id"],
            novelty_evidence=_canonical_tuple(values["novelty_evidence"]),  # type: ignore[arg-type]
            observation_graph_id=values["observation_graph_id"],
            occurrences=_canonical_tuple(values["occurrences"]),  # type: ignore[arg-type]
            promotion_candidates=_canonical_tuple(values["promotion_candidates"]),  # type: ignore[arg-type]
            promotion_recipe_id=values["promotion_recipe_id"],
            recurrences=_canonical_tuple(values["recurrences"]),  # type: ignore[arg-type]
            rejected_candidate_ids=list(rejected_candidate_ids),
            rule_change_results=_canonical_tuple(values["rule_change_results"]),  # type: ignore[arg-type]
            stabilities=_canonical_tuple(values["stabilities"]),  # type: ignore[arg-type]
            status=values["status"],
        )
        return cls(promotion_analysis_id=canonical_id(payload), **values)  # type: ignore[arg-type]
