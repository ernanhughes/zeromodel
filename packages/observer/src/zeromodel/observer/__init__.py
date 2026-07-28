"""ZeroModel Observer demonstration public API.

This package owns bounded Observer-level DTOs for transition comparison,
contradiction artifacts, and replacement lineage. It is a consumer package and
does not widen the core VPM artifact contract.
"""

from zeromodel.observer.artifacts import (
    ObserverContradictionArtifactDTO,
    ObserverObservationArtifactDTO,
    ObserverReplacementPolicyArtifactDTO,
    ObserverTransitionRecordDTO,
    build_contradiction_artifact,
    build_replacement_policy_artifact,
    build_transition_record,
)
from zeromodel.observer.comparison import (
    ObserverComparisonRecipeDTO,
    ObserverComparisonResultDTO,
    compare_observer_transition,
)

__all__ = [
    "ObserverComparisonRecipeDTO",
    "ObserverComparisonResultDTO",
    "ObserverContradictionArtifactDTO",
    "ObserverObservationArtifactDTO",
    "ObserverReplacementPolicyArtifactDTO",
    "ObserverTransitionRecordDTO",
    "build_contradiction_artifact",
    "build_replacement_policy_artifact",
    "build_transition_record",
    "compare_observer_transition",
]
