from __future__ import annotations

from dataclasses import dataclass

from zeromodel.critic.dto import CriticScoreResultDTO


def rank_by_critic(result: CriticScoreResultDTO) -> tuple[str, ...]:
    ordered = sorted(
        result.items,
        key=lambda item: (-item.score, item.artifact_ref.artifact_id),
    )
    return tuple(item.artifact_ref.artifact_id for item in ordered)


@dataclass(frozen=True, slots=True)
class CriticTriageResultDTO:
    selected_artifact_ids: tuple[str, ...]
    budget_count: int
    ordering_contract: str = "critic_score_desc;artifact_id_asc"


def triage_by_budget(
    result: CriticScoreResultDTO, *, budget_count: int
) -> CriticTriageResultDTO:
    if budget_count < 0:
        raise ValueError("budget_count must be non-negative")
    return CriticTriageResultDTO(
        selected_artifact_ids=rank_by_critic(result)[:budget_count],
        budget_count=budget_count,
    )
