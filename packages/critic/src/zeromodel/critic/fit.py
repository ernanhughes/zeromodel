from __future__ import annotations

from zeromodel.critic.dto import (
    CriticFitSpecDTO,
    CriticContractDTO,
    CriticFeatureSpecDTO,
)
from zeromodel.critic.linear import CompiledCriticReadout


def fit_compiled_critic(
    features: object,
    labels: object,
    *,
    feature_spec: CriticFeatureSpecDTO,
    contract: CriticContractDTO,
    fit_spec: CriticFitSpecDTO | None = None,
) -> CompiledCriticReadout:
    spec = fit_spec or CriticFitSpecDTO()
    return CompiledCriticReadout.fit(
        features,
        labels,
        feature_spec=feature_spec,
        contract_id=contract.critic_contract_id,
        l2_penalty=spec.l2_penalty,
        max_iterations=spec.max_iterations,
        tolerance=spec.tolerance,
        class_weighting=spec.class_weighting,
    )
