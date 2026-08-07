from __future__ import annotations

import json

import pytest

from zeromodel.critic import (
    CriticScoreRequestDTO,
    CriticThresholdContractDTO,
    build_critic_score_receipt,
    build_critic_score_vpm,
    compile_critic_readout,
    export_portable_critic,
    load_critic_readout_aggregate,
    rank_by_critic,
    replay_critic_score,
    score_critic,
    score_portable,
)


def test_compile_score_portable_vpm_and_replay(critic_fixture) -> None:
    store = critic_fixture["store"]
    readout, readout_ref = compile_critic_readout(
        store=store,
        features=critic_fixture["features"],
        labels=critic_fixture["labels"],
        fit_spec=critic_fixture["fit_spec"],
    )
    aggregate = load_critic_readout_aggregate(readout_ref, store)
    payload = export_portable_critic(aggregate)
    assert len(payload.encode("utf-8")) < 50 * 1024
    request = CriticScoreRequestDTO(
        readout_ref=readout_ref,
        feature_batch_ref=readout.training_feature_batch_ref,
        threshold_contract=CriticThresholdContractDTO(
            reject_below=0.4, accept_at_or_above=0.6
        ),
        explanation_depth=2,
    )
    result, result_ref, request_ref = score_critic(store=store, request=request)
    assert result_ref is not None
    assert request_ref.artifact_id == result.request_ref.artifact_id
    portable = score_portable(payload, critic_fixture["values"][0].tolist())
    assert portable["logit"] == pytest.approx(result.items[0].logit)
    assert portable["score"] == pytest.approx(result.items[0].score)
    positive_frontier = {
        critic_fixture["features"].batch.item_refs[0].artifact_id,
        critic_fixture["features"].batch.item_refs[4].artifact_id,
    }
    assert rank_by_critic(result)[0] in positive_frontier
    vpm = build_critic_score_vpm(result=result)
    assert vpm.provenance["readout_ref"] == readout_ref.artifact_id
    _, receipt_ref = build_critic_score_receipt(
        store=store, request_ref=request_ref, result_ref=result_ref
    )
    replayed = replay_critic_score(store=store, receipt_ref=receipt_ref)
    assert replayed.result_id == result.result_id
    assert json.loads(payload)["score_contract"]["calibration_input"] == "logit"
