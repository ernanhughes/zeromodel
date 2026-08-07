from __future__ import annotations

import numpy as np

from zeromodel.artifacts import ArtifactRef, ArtifactStore

from zeromodel.critic.dto import (
    CriticItemScoreDTO,
    CriticReadoutArtifactDTO,
    CriticScoreRequestDTO,
    CriticScoreResultDTO,
)
from zeromodel.critic.linear import CompiledCriticReadout, stable_sigmoid
from zeromodel.critic.persistence import (
    ResolvedCriticReadoutAggregate,
    load_critic_feature_batch_aggregate,
    load_critic_readout_aggregate,
    store_critic_score_request,
    store_critic_score_result,
)

ORDERING_CONTRACT = "source_order;rank_by_critic_available_separately"


def compiled_from_aggregate(
    aggregate: ResolvedCriticReadoutAggregate,
) -> CompiledCriticReadout:
    calibration = None
    if aggregate.calibration is not None:
        calibration = aggregate.calibration.to_dict()
    return CompiledCriticReadout(
        feature_ids=aggregate.feature_spec.feature_ids,
        directionality=aggregate.feature_spec.directionality,
        center=np.asarray(aggregate.center_blob.to_array(), dtype=np.float64),
        scale=np.asarray(aggregate.scale_blob.to_array(), dtype=np.float64),
        coefficients=np.asarray(
            aggregate.coefficients_blob.to_array(), dtype=np.float64
        ),
        intercept=float(
            np.asarray(aggregate.intercept_blob.to_array(), dtype=np.float64)[0]
        ),
        contract_id=aggregate.contract.critic_contract_id,
        feature_spec_id=aggregate.feature_spec.feature_spec_id,
        calibration=calibration,
    )


def _calibrated(
    runtime: CompiledCriticReadout, logits: np.ndarray
) -> np.ndarray | None:
    if not runtime.calibration or runtime.calibration.get("method") == "none":
        return None
    params = runtime.calibration.get("parameters") or {}
    a = float(params.get("a", 1.0))
    b = float(params.get("b", 0.0))
    return stable_sigmoid(a * logits + b)


def _verdict(score: float, request: CriticScoreRequestDTO) -> tuple[str | None, float]:
    threshold = request.threshold_contract
    if threshold is None:
        return None, 0.0
    if threshold.reject_below is not None and score < threshold.reject_below:
        return "REJECT", float(threshold.reject_below - score)
    if (
        threshold.accept_at_or_above is not None
        and score >= threshold.accept_at_or_above
    ):
        return "ACCEPT", float(score - threshold.accept_at_or_above)
    margins = []
    if threshold.reject_below is not None:
        margins.append(abs(score - threshold.reject_below))
    if threshold.accept_at_or_above is not None:
        margins.append(abs(score - threshold.accept_at_or_above))
    return "REVIEW", float(min(margins) if margins else 0.0)


def score_critic(
    *,
    store: ArtifactStore,
    request: CriticScoreRequestDTO,
    persist_result: bool = True,
) -> tuple[CriticScoreResultDTO, ArtifactRef | None, ArtifactRef]:
    request_ref = store_critic_score_request(store, request)
    readout = load_critic_readout_aggregate(request.readout_ref, store)
    batch = load_critic_feature_batch_aggregate(request.feature_batch_ref, store)
    runtime = compiled_from_aggregate(readout)
    logits = runtime.logit_many(
        batch.values, feature_spec_id=batch.spec.feature_spec_id
    )
    scores = stable_sigmoid(logits)
    calibrated = _calibrated(runtime, logits)
    items = []
    for index, ref in enumerate(batch.batch.item_refs):
        contributions = runtime.contributions_one(
            batch.values[index], feature_spec_id=batch.spec.feature_spec_id
        )
        positive = sum(max(0.0, item.contribution) for item in contributions)
        negative = sum(abs(min(0.0, item.contribution)) for item in contributions)
        shown = ()
        if request.explanation_depth:
            ordered = sorted(
                contributions,
                key=lambda item: (-abs(item.contribution), item.feature_id),
            )
            shown = tuple(ordered[: request.explanation_depth])
        verdict, margin = _verdict(float(scores[index]), request)
        items.append(
            CriticItemScoreDTO(
                artifact_ref=ref,
                logit=float(logits[index]),
                score=float(scores[index]),
                calibrated_probability=None
                if calibrated is None
                else float(calibrated[index]),
                verdict=verdict,
                decision_margin=margin,
                feature_coverage=1.0,
                positive_contribution_strength=float(positive),
                negative_contribution_strength=float(negative),
                contributions=shown,
            )
        )
    result = CriticScoreResultDTO(
        request_ref=request_ref,
        readout_ref=request.readout_ref,
        feature_batch_ref=request.feature_batch_ref,
        critic_contract_ref=readout.readout.critic_contract_ref,
        feature_spec_ref=readout.readout.feature_spec_ref,
        items=tuple(items),
        ordering_contract=ORDERING_CONTRACT,
        metadata={"score_semantics": readout.contract.score_semantics},
    )
    return (
        result,
        store_critic_score_result(store, result) if persist_result else None,
        request_ref,
    )


def readout_from_runtime(
    *,
    runtime: CompiledCriticReadout,
    store: ArtifactStore,
    readout: CriticReadoutArtifactDTO,
) -> tuple[CriticReadoutArtifactDTO, ArtifactRef]:
    from zeromodel.critic.persistence import store_critic_readout

    return readout, store_critic_readout(store, readout)
