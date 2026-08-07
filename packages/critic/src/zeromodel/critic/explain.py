from __future__ import annotations

from zeromodel.critic.dto import CriticFeatureContributionDTO


def top_positive_contributions(
    contributions: tuple[CriticFeatureContributionDTO, ...], *, limit: int
) -> tuple[CriticFeatureContributionDTO, ...]:
    return tuple(
        sorted(
            (item for item in contributions if item.contribution > 0.0),
            key=lambda item: (-item.contribution, item.feature_id),
        )[:limit]
    )


def top_negative_contributions(
    contributions: tuple[CriticFeatureContributionDTO, ...], *, limit: int
) -> tuple[CriticFeatureContributionDTO, ...]:
    return tuple(
        sorted(
            (item for item in contributions if item.contribution < 0.0),
            key=lambda item: (item.contribution, item.feature_id),
        )[:limit]
    )
