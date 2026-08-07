from __future__ import annotations

import numpy as np
import pytest

from zeromodel.artifacts import InMemoryArtifactStore, canonical_json_bytes
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.critic.dto import (
    CriticContractDTO,
    CriticFeatureBatchDTO,
    CriticFeatureDTO,
    CriticFeatureSpecDTO,
    CriticFitSpecDTO,
    CriticLabelBatchDTO,
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


@pytest.fixture()
def critic_fixture():
    store = InMemoryArtifactStore()
    spec = CriticFeatureSpecDTO(
        features=(
            CriticFeatureDTO("stability", "stable numeric signal"),
            CriticFeatureDTO("coverage", "coverage signal"),
            CriticFeatureDTO("uncertainty", "uncertainty signal", directionality=-1),
            CriticFeatureDTO("constant", "zero variance fixture"),
        )
    )
    contract = CriticContractDTO(
        critic_id="fixture-success",
        version="v1",
        target_id="synthetic-success",
        positive_label="successful",
        negative_label="failed",
        score_semantics="similarity to deterministic synthetic successful examples",
        intended_uses=("ranking", "triage"),
        prohibited_uses=("semantic truth",),
    )
    spec_ref = store_critic_feature_spec(store, spec)
    contract_ref = store_critic_contract(store, contract)
    values = np.asarray(
        [
            [0.9, 0.8, 0.1, 1.0],
            [0.8, 0.7, 0.2, 1.0],
            [0.2, 0.4, 0.8, 1.0],
            [0.1, 0.3, 0.9, 1.0],
            [0.85, 0.9, 0.15, 1.0],
            [0.15, 0.2, 0.75, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([1, 1, 0, 0, 1, 0], dtype=np.float64)
    item_refs = tuple(
        store.put(
            "fixture.item", canonical_json_bytes({"item": index}), {"fixture": "critic"}
        )
        for index in range(values.shape[0])
    )
    values_ref = store_matrix_blob(
        store,
        MatrixBlob.from_array(
            values, dtype="float64", metadata={"role": "critic_features"}
        ),
    )
    labels_ref = store_matrix_blob(
        store,
        MatrixBlob.from_array(
            labels, dtype="float64", metadata={"role": "critic_labels"}
        ),
    )
    feature_batch = CriticFeatureBatchDTO(
        feature_spec_ref=spec_ref,
        values_blob_ref=values_ref,
        item_refs=item_refs,
        values_shape=values.shape,
        values_dtype="float64",
    )
    label_batch = CriticLabelBatchDTO(
        critic_contract_ref=contract_ref,
        item_refs=item_refs,
        labels_blob_ref=labels_ref,
        labels_shape=labels.shape,
    )
    return {
        "store": store,
        "spec": spec,
        "contract": contract,
        "features": load_critic_feature_batch_aggregate(
            store_critic_feature_batch(store, feature_batch), store
        ),
        "labels": load_critic_label_batch_aggregate(
            store_critic_label_batch(store, label_batch), store
        ),
        "fit_spec": CriticFitSpecDTO(l2_penalty=0.1, max_iterations=80),
        "values": values,
        "labels_array": labels,
    }
