from __future__ import annotations

import numpy as np

from zeromodel.artifacts import ArtifactStore
from zeromodel.core.matrix_blob import MatrixBlob

from zeromodel.critic.dto import CriticFitSpecDTO, CriticReadoutArtifactDTO
from zeromodel.critic.errors import CriticValidationError
from zeromodel.critic.linear import CompiledCriticReadout
from zeromodel.critic.persistence import (
    ResolvedCriticFeatureBatchAggregate,
    ResolvedCriticLabelBatchAggregate,
    store_critic_contract,
    store_critic_feature_batch,
    store_critic_feature_spec,
    store_critic_fit_spec,
    store_critic_label_batch,
    store_critic_readout,
    store_matrix_blob,
)
from zeromodel.critic.portable import export_portable_critic


def compile_critic_readout(
    *,
    store: ArtifactStore,
    features: ResolvedCriticFeatureBatchAggregate,
    labels: ResolvedCriticLabelBatchAggregate,
    fit_spec: CriticFitSpecDTO,
    metadata: dict | None = None,
) -> tuple[CriticReadoutArtifactDTO, object]:
    if features.batch.item_refs != labels.batch.item_refs:
        raise CriticValidationError(
            "training feature and label rows must align exactly"
        )
    runtime = CompiledCriticReadout.fit(
        features.values,
        labels.labels,
        feature_spec=features.spec,
        contract_id=labels.contract.critic_contract_id,
        l2_penalty=fit_spec.l2_penalty,
        max_iterations=fit_spec.max_iterations,
        tolerance=fit_spec.tolerance,
        class_weighting=fit_spec.class_weighting,
    )
    spec_ref = store_critic_feature_spec(store, features.spec)
    contract_ref = store_critic_contract(store, labels.contract)
    fit_ref = store_critic_fit_spec(store, fit_spec)
    center_ref = store_matrix_blob(
        store,
        MatrixBlob.from_array(
            runtime.center, dtype="float64", metadata={"role": "critic_center"}
        ),
    )
    scale_ref = store_matrix_blob(
        store,
        MatrixBlob.from_array(
            runtime.scale, dtype="float64", metadata={"role": "critic_scale"}
        ),
    )
    coefficients_ref = store_matrix_blob(
        store,
        MatrixBlob.from_array(
            runtime.coefficients,
            dtype="float64",
            metadata={"role": "critic_coefficients"},
        ),
    )
    intercept_ref = store_matrix_blob(
        store,
        MatrixBlob.from_array(
            np.asarray([runtime.intercept]),
            dtype="float64",
            metadata={"role": "critic_intercept"},
        ),
    )
    training_feature_ref = store_critic_feature_batch(store, features.batch)
    training_label_ref = store_critic_label_batch(store, labels.batch)
    readout = CriticReadoutArtifactDTO(
        critic_contract_ref=contract_ref,
        feature_spec_ref=spec_ref,
        fit_spec_ref=fit_ref,
        center_blob_ref=center_ref,
        scale_blob_ref=scale_ref,
        coefficients_blob_ref=coefficients_ref,
        intercept_blob_ref=intercept_ref,
        training_feature_batch_ref=training_feature_ref,
        training_label_batch_ref=training_label_ref,
        metadata=metadata or {},
    )
    readout_ref = store_critic_readout(store, readout)
    from zeromodel.critic.persistence import load_critic_readout_aggregate

    aggregate = load_critic_readout_aggregate(readout_ref, store)
    export_portable_critic(aggregate)
    return readout, readout_ref
