from __future__ import annotations

import json

import pytest

from examples.tiny_critic import (
    TRAINING_RECORDS,
    independent_portable_score,
    run_tiny_critic_fixture,
    store_fixture_batch,
)
from zeromodel.critic import (
    CriticFeatureBatchDTO,
    CriticFeatureDTO,
    CriticFeatureSpecDTO,
    CriticFeatureSchemaMismatchError,
    CriticScoreRequestDTO,
    build_critic_score_vpm,
    load_critic_evaluation_result_aggregate,
    load_critic_readout_aggregate,
    replay_critic_score,
    score_critic,
    score_portable,
)
from zeromodel.critic.dto import CriticEvaluationSetDTO
from zeromodel.critic.persistence import (
    load_critic_feature_batch,
    load_critic_score_result,
    store_critic_evaluation_set,
    store_critic_feature_batch,
    store_critic_feature_spec,
)


def test_tiny_critic_uses_disjoint_held_out_evidence() -> None:
    run = run_tiny_critic_fixture()
    training_ids = {ref.artifact_id for ref in run.training.item_refs}
    evaluation_ids = {ref.artifact_id for ref in run.evaluation.item_refs}
    assert training_ids.isdisjoint(evaluation_ids)
    result = load_critic_score_result(run.store, run.score_result_ref)
    assert (
        result.feature_batch_ref.artifact_id == run.evaluation.feature_ref.artifact_id
    )
    assert result.feature_batch_ref.artifact_id != run.training.feature_ref.artifact_id


def test_tiny_critic_explanation_and_portable_closure() -> None:
    run = run_tiny_critic_fixture()
    aggregate = load_critic_readout_aggregate(run.readout_ref, run.store)
    runtime = __import__(
        "zeromodel.critic.scoring"
    ).critic.scoring.compiled_from_aggregate(aggregate)
    result = load_critic_score_result(run.store, run.score_result_ref)
    row = run.evaluation.values[0]
    contributions = runtime.contributions_one(
        row, feature_spec_id=run.feature_spec.feature_spec_id
    )
    assert sum(
        item.contribution for item in contributions
    ) + runtime.intercept == pytest.approx(result.items[0].logit)
    assert runtime.score_one(
        row, feature_spec_id=run.feature_spec.feature_spec_id
    ) == pytest.approx(result.items[0].score)
    portable = score_portable(run.portable_payload, row.tolist())
    reference = independent_portable_score(run.portable_payload, row)
    assert portable["score"] == pytest.approx(result.items[0].score)
    assert reference["score"] == pytest.approx(result.items[0].score)
    assert (
        len(run.portable_payload.encode("utf-8"))
        < run.fit_spec.portable_payload_limit_bytes
    )


def test_tiny_critic_schema_mismatch_rejected() -> None:
    run = run_tiny_critic_fixture()
    original = load_critic_feature_batch(run.store, run.evaluation.feature_ref)
    changed_spec = CriticFeatureSpecDTO(
        features=(
            CriticFeatureDTO("coverage", "changed order"),
            CriticFeatureDTO("stability", "changed order"),
            CriticFeatureDTO("uncertainty", "changed order", directionality=-1),
            CriticFeatureDTO("consistency", "changed order"),
        )
    )
    changed_spec_ref = store_critic_feature_spec(run.store, changed_spec)
    incompatible = CriticFeatureBatchDTO(
        feature_spec_ref=changed_spec_ref,
        values_blob_ref=original.values_blob_ref,
        item_refs=original.item_refs,
        values_shape=original.values_shape,
        values_dtype=original.values_dtype,
    )
    incompatible_ref = store_critic_feature_batch(run.store, incompatible)
    with pytest.raises(CriticFeatureSchemaMismatchError):
        score_critic(
            store=run.store,
            request=CriticScoreRequestDTO(
                readout_ref=run.readout_ref,
                feature_batch_ref=incompatible_ref,
            ),
        )


def test_tiny_critic_mutation_changes_identities_and_evaluation_is_frozen() -> None:
    run = run_tiny_critic_fixture()
    mutated_records = list(TRAINING_RECORDS)
    mutated = dict(mutated_records[0])
    mutated["label"] = 0
    mutated_records[0] = mutated
    mutated_training = store_fixture_batch(
        run.store,
        split="training-mutated",
        records=tuple(mutated_records),
        feature_spec_ref=load_critic_feature_batch(
            run.store, run.training.feature_ref
        ).feature_spec_ref,
        contract_ref=__import__("zeromodel.critic.persistence")
        .critic.persistence.load_critic_label_batch(run.store, run.training.label_ref)
        .critic_contract_ref,
    )
    assert mutated_training.label_ref.artifact_id != run.training.label_ref.artifact_id
    from zeromodel.critic import CriticFitSpecDTO, compile_critic_readout
    from zeromodel.critic.persistence import (
        load_critic_feature_batch_aggregate,
        load_critic_label_batch_aggregate,
    )

    _, mutated_readout_ref = compile_critic_readout(
        store=run.store,
        features=load_critic_feature_batch_aggregate(
            mutated_training.feature_ref, run.store
        ),
        labels=load_critic_label_batch_aggregate(mutated_training.label_ref, run.store),
        fit_spec=CriticFitSpecDTO(l2_penalty=0.1, max_iterations=80),
    )
    assert mutated_readout_ref.artifact_id != run.readout_ref.artifact_id
    evaluation_aggregate = load_critic_evaluation_result_aggregate(
        run.evaluation_result_ref, run.store
    )
    assert (
        evaluation_aggregate.evaluation_set.evaluation_set.feature_batch_ref.artifact_id
        == run.evaluation.feature_ref.artifact_id
    )


def test_tiny_critic_vpm_replay_and_evaluation_identity() -> None:
    run = run_tiny_critic_fixture()
    result = load_critic_score_result(run.store, run.score_result_ref)
    first = build_critic_score_vpm(result=result)
    second = build_critic_score_vpm(result=result)
    assert first.artifact_id == second.artifact_id
    assert first.provenance["readout_ref"] == run.readout_ref.artifact_id
    assert (
        first.provenance["feature_batch_ref"] == run.evaluation.feature_ref.artifact_id
    )
    replayed = replay_critic_score(store=run.store, receipt_ref=run.receipt_ref)
    assert replayed.result_id == result.result_id
    evaluation = load_critic_evaluation_result_aggregate(
        run.evaluation_result_ref, run.store
    )
    assert (
        evaluation.evaluation_result.readout_ref.artifact_id
        == run.readout_ref.artifact_id
    )
    assert (
        evaluation.evaluation_set.evaluation_set.split_id == "tiny-critic-held-out-v1"
    )
    changed_set = CriticEvaluationSetDTO(
        feature_batch_ref=run.evaluation.feature_ref,
        label_batch_ref=run.evaluation.label_ref,
        split_id="tiny-critic-held-out-v2",
    )
    changed_ref = store_critic_evaluation_set(run.store, changed_set)
    assert changed_ref.artifact_id != run.evaluation_set_ref.artifact_id
    assert len(json.dumps(run.metrics, sort_keys=True)) > 0
