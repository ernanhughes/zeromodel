from __future__ import annotations

import numpy as np

from zeromodel.artifacts import InMemoryArtifactStore, canonical_json_bytes
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.critic import (
    CriticContractDTO,
    CriticFeatureBatchDTO,
    CriticFeatureDTO,
    CriticFeatureSpecDTO,
    CriticFitSpecDTO,
    CriticLabelBatchDTO,
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
)
from zeromodel.critic.persistence import (
    load_critic_feature_batch_aggregate,
    load_critic_label_batch_aggregate,
    store_critic_contract,
    store_critic_feature_batch,
    store_critic_feature_spec,
    store_critic_label_batch,
    store_matrix_blob,
)


def main() -> None:
    store = InMemoryArtifactStore()
    feature_spec = CriticFeatureSpecDTO(
        features=(
            CriticFeatureDTO("stability", "deterministic fixture stability"),
            CriticFeatureDTO("coverage", "deterministic fixture coverage"),
            CriticFeatureDTO(
                "uncertainty", "deterministic fixture uncertainty", directionality=-1
            ),
            CriticFeatureDTO("consistency", "deterministic fixture consistency"),
        )
    )
    contract = CriticContractDTO(
        critic_id="tiny-fixture-success",
        version="v1",
        target_id="synthetic-success",
        positive_label="successful",
        negative_label="failed",
        score_semantics="similarity to this deterministic capability fixture's successful rows",
        intended_uses=("ranking", "triage"),
        prohibited_uses=("semantic truth", "universal quality"),
    )
    spec_ref = store_critic_feature_spec(store, feature_spec)
    contract_ref = store_critic_contract(store, contract)
    values = np.asarray(
        [
            [0.90, 0.85, 0.10, 0.80],
            [0.75, 0.70, 0.20, 0.70],
            [0.20, 0.30, 0.80, 0.35],
            [0.15, 0.20, 0.90, 0.25],
            [0.88, 0.92, 0.15, 0.86],
            [0.30, 0.25, 0.75, 0.40],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([1, 1, 0, 0, 1, 0], dtype=np.float64)
    item_refs = tuple(
        store.put("example.tiny_critic.item", canonical_json_bytes({"row": index}), {})
        for index in range(values.shape[0])
    )
    feature_batch = CriticFeatureBatchDTO(
        feature_spec_ref=spec_ref,
        values_blob_ref=store_matrix_blob(
            store, MatrixBlob.from_array(values, dtype="float64")
        ),
        item_refs=item_refs,
        values_shape=values.shape,
        values_dtype="float64",
    )
    label_batch = CriticLabelBatchDTO(
        critic_contract_ref=contract_ref,
        item_refs=item_refs,
        labels_blob_ref=store_matrix_blob(
            store, MatrixBlob.from_array(labels, dtype="float64")
        ),
        labels_shape=labels.shape,
    )
    feature_ref = store_critic_feature_batch(store, feature_batch)
    label_ref = store_critic_label_batch(store, label_batch)
    readout, readout_ref = compile_critic_readout(
        store=store,
        features=load_critic_feature_batch_aggregate(feature_ref, store),
        labels=load_critic_label_batch_aggregate(label_ref, store),
        fit_spec=CriticFitSpecDTO(l2_penalty=0.1),
    )
    request = CriticScoreRequestDTO(
        readout_ref=readout_ref,
        feature_batch_ref=readout.training_feature_batch_ref,
        threshold_contract=CriticThresholdContractDTO(
            reject_below=0.4, accept_at_or_above=0.6
        ),
        explanation_depth=2,
    )
    result, result_ref, request_ref = score_critic(store=store, request=request)
    if result_ref is None:
        raise RuntimeError("result was not persisted")
    portable = export_portable_critic(load_critic_readout_aggregate(readout_ref, store))
    vpm = build_critic_score_vpm(result=result)
    _, receipt_ref = build_critic_score_receipt(
        store=store, request_ref=request_ref, result_ref=result_ref
    )
    replayed = replay_critic_score(store=store, receipt_ref=receipt_ref)
    print("ranked:", rank_by_critic(result)[:3])
    print("portable-bytes:", len(portable.encode("utf-8")))
    print("vpm:", vpm.artifact_id)
    print("replay:", replayed.result_id)


if __name__ == "__main__":
    main()
