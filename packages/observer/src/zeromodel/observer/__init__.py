"""ZeroModel Observer demonstration public API.

This package owns bounded Observer-level DTOs for transition comparison,
contradiction artifacts, and replacement lineage. It is a consumer package and
does not widen the core VPM artifact contract.
"""

from zeromodel.observer.artifacts import (
    ObserverContradictionArtifactDTO,
    ObserverFeatureDefinitionDTO,
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
    ObserverReplacementPolicyArtifactDTO,
    ObserverTransitionRecordDTO,
    build_contradiction_artifact,
    build_replacement_policy_artifact,
    build_transition_record,
)
from zeromodel.observer.comparison import (
    ObserverComparisonRecipeDTO,
    ObserverComparisonResultDTO,
    ObserverFeatureComparisonDTO,
    ObserverFeatureComparisonResultDTO,
    ObserverHiddenStateHypothesisDTO,
    ObserverHiddenStateHypothesisSetDTO,
    ObserverPolicyConsequenceEvidenceDTO,
    compare_observer_transition,
)
from zeromodel.observer.fixture import (
    ObserverExecutedFixtureStepDTO,
    ObserverFixtureActionDTO,
    ObserverFixtureError,
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
)
from zeromodel.observer.fixture_predictor import (
    ObserverPredictedTransitionDTO,
    build_observer_fixture_comparison_recipe,
    build_observer_fixture_observation_schema,
    execute_observer_fixture_step,
    predict_observer_fixture_transition,
)
from zeromodel.observer.fixture_runtime import (
    ObserverFixtureEpisodeResultDTO,
    ObserverFixtureRuleScheduleEntryDTO,
    active_rule_for_step,
    run_observer_fixture_episode,
)
from zeromodel.observer.ledger import (
    InMemoryObserverTransitionLedger,
    ObserverLedgerReplayResultDTO,
    ObserverTransitionLedgerEntryDTO,
    ObserverTransitionLedgerSnapshotDTO,
    build_observer_transition_ledger_snapshot,
    replay_observer_transition_ledger,
)
from zeromodel.observer.repair import (
    ObserverProposedChangeDTO,
    ObserverRepairConstraintDTO,
    ObserverRepairProposalDTO,
    ObserverRepairProposalError,
)
from zeromodel.observer.repair_service import propose_observer_repair
from zeromodel.observer.transition_service import (
    ObserverTransitionVerificationDTO,
    ObserverTransitionVerificationError,
    verify_observer_transition,
)
from zeromodel.observer.wake import (
    ObserverWakeEvaluationDTO,
    ObserverWakePolicyAblationDTO,
    ObserverWakePolicyDTO,
    ObserverWakePolicyReplayDTO,
    build_wake_policy_ablation,
    evaluate_wake_policy_for_entry,
    evaluate_wake_policy_over_ledger,
)

__all__ = [
    "ObserverComparisonRecipeDTO",
    "ObserverComparisonResultDTO",
    "ObserverContradictionArtifactDTO",
    "ObserverExecutedFixtureStepDTO",
    "ObserverFeatureComparisonDTO",
    "ObserverFeatureComparisonResultDTO",
    "ObserverFeatureDefinitionDTO",
    "ObserverFixtureActionDTO",
    "ObserverFixtureEpisodeResultDTO",
    "ObserverFixtureError",
    "ObserverFixtureRuleScheduleEntryDTO",
    "ObserverFixtureRuleSetDTO",
    "ObserverFixtureStateDTO",
    "ObserverHiddenStateHypothesisDTO",
    "ObserverHiddenStateHypothesisSetDTO",
    "ObserverLedgerReplayResultDTO",
    "ObserverObservationArtifactDTO",
    "ObserverObservationSchemaDTO",
    "ObserverPolicyConsequenceEvidenceDTO",
    "ObserverPredictedTransitionDTO",
    "ObserverProposedChangeDTO",
    "ObserverRepairConstraintDTO",
    "ObserverRepairProposalDTO",
    "ObserverRepairProposalError",
    "ObserverReplacementPolicyArtifactDTO",
    "ObserverTransitionLedgerEntryDTO",
    "ObserverTransitionLedgerSnapshotDTO",
    "ObserverTransitionRecordDTO",
    "ObserverTransitionVerificationDTO",
    "ObserverTransitionVerificationError",
    "ObserverWakeEvaluationDTO",
    "ObserverWakePolicyAblationDTO",
    "ObserverWakePolicyDTO",
    "ObserverWakePolicyReplayDTO",
    "InMemoryObserverTransitionLedger",
    "build_contradiction_artifact",
    "build_observer_fixture_comparison_recipe",
    "build_observer_fixture_observation_schema",
    "build_observer_transition_ledger_snapshot",
    "build_replacement_policy_artifact",
    "build_transition_record",
    "build_wake_policy_ablation",
    "compare_observer_transition",
    "evaluate_wake_policy_for_entry",
    "evaluate_wake_policy_over_ledger",
    "execute_observer_fixture_step",
    "predict_observer_fixture_transition",
    "propose_observer_repair",
    "replay_observer_transition_ledger",
    "run_observer_fixture_episode",
    "verify_observer_transition",
    "active_rule_for_step",
]
