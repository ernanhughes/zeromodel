from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CriticPromotionPolicyDTO:
    min_candidate_auroc: float = 0.5
    max_candidate_ece: float = 0.2
    min_auroc_gain: float = 0.0
    max_ece_regression: float = 0.05


@dataclass(frozen=True, slots=True)
class CriticPromotionDecisionDTO:
    recommended: bool
    reasons: tuple[str, ...]


def evaluate_promotion(
    *,
    current_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    policy: CriticPromotionPolicyDTO,
) -> CriticPromotionDecisionDTO:
    reasons = []
    if candidate_metrics.get("auroc", 0.0) < policy.min_candidate_auroc:
        reasons.append("candidate AUROC below floor")
    if candidate_metrics.get("ece", 1.0) > policy.max_candidate_ece:
        reasons.append("candidate ECE above ceiling")
    if (
        candidate_metrics.get("auroc", 0.0) - current_metrics.get("auroc", 0.0)
        < policy.min_auroc_gain
    ):
        reasons.append("candidate AUROC gain below required improvement")
    if (
        candidate_metrics.get("ece", 1.0) - current_metrics.get("ece", 1.0)
        > policy.max_ece_regression
    ):
        reasons.append("candidate ECE regression exceeds limit")
    if not reasons:
        reasons.append("candidate satisfies promotion policy")
    return CriticPromotionDecisionDTO(
        recommended=len(reasons) == 1 and reasons[0].startswith("candidate satisfies"),
        reasons=tuple(reasons),
    )
