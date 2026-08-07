from __future__ import annotations

from zeromodel.artifacts import ArtifactRef, ArtifactStore

from zeromodel.critic.dto import CriticScoreReceiptDTO
from zeromodel.critic.errors import CriticReplayMismatchError
from zeromodel.critic.persistence import (
    load_critic_score_receipt,
    load_critic_score_request,
    load_critic_score_result,
    store_critic_score_receipt,
)
from zeromodel.critic.scoring import score_critic


def build_critic_score_receipt(
    *,
    store: ArtifactStore,
    request_ref: ArtifactRef,
    result_ref: ArtifactRef,
) -> tuple[CriticScoreReceiptDTO, ArtifactRef]:
    request = load_critic_score_request(store, request_ref)
    result = load_critic_score_result(store, result_ref)
    receipt = CriticScoreReceiptDTO(
        request_ref=request_ref,
        result_ref=result_ref,
        readout_ref=request.readout_ref,
        feature_batch_ref=request.feature_batch_ref,
        required_checks=CriticScoreReceiptDTO.REQUIRED,
        result_id=result.result_id,
    )
    return receipt, store_critic_score_receipt(store, receipt)


def replay_critic_score(*, store: ArtifactStore, receipt_ref: ArtifactRef) -> object:
    receipt = load_critic_score_receipt(store, receipt_ref)
    request = load_critic_score_request(store, receipt.request_ref)
    fresh, fresh_ref, _ = score_critic(
        store=store, request=request, persist_result=True
    )
    if fresh.result_id != receipt.result_id:
        raise CriticReplayMismatchError("replayed critic result diverged from receipt")
    if fresh_ref is None or fresh_ref.artifact_id != receipt.result_ref.artifact_id:
        raise CriticReplayMismatchError("replayed result ref diverged from receipt")
    return fresh
