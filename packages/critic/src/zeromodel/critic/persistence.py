from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypeVar

import numpy as np

from zeromodel.artifacts import ArtifactRef, ArtifactResolver, ArtifactStore
from zeromodel.core.matrix_blob import MatrixBlob

from zeromodel.critic.dto import (
    CriticCalibrationDTO,
    CriticContractDTO,
    CriticFeatureBatchDTO,
    CriticFeatureSpecDTO,
    CriticFitSpecDTO,
    CriticLabelBatchDTO,
    CriticReadoutArtifactDTO,
    CriticScoreReceiptDTO,
    CriticScoreRequestDTO,
    CriticScoreResultDTO,
    canonical_dto_bytes,
)
from zeromodel.critic.errors import (
    CriticContractMismatchError,
    CriticFeatureSchemaMismatchError,
    CriticReadoutIntegrityError,
)
from zeromodel.critic.kinds import (
    CRITIC_CALIBRATION_ARTIFACT_KIND,
    CRITIC_CONTRACT_ARTIFACT_KIND,
    CRITIC_FEATURE_BATCH_ARTIFACT_KIND,
    CRITIC_FEATURE_SPEC_ARTIFACT_KIND,
    CRITIC_FIT_SPEC_ARTIFACT_KIND,
    CRITIC_LABEL_BATCH_ARTIFACT_KIND,
    CRITIC_READOUT_ARTIFACT_KIND,
    CRITIC_SCORE_RECEIPT_ARTIFACT_KIND,
    CRITIC_SCORE_REQUEST_ARTIFACT_KIND,
    CRITIC_SCORE_RESULT_ARTIFACT_KIND,
    MATRIX_BLOB_ARTIFACT_KIND,
)

T = TypeVar("T")


def _require_kind(ref: ArtifactRef, expected: str) -> None:
    if ref.artifact_kind != expected:
        raise CriticReadoutIntegrityError(
            f"expected artifact kind {expected!r}, got {ref.artifact_kind!r}"
        )


def store_dto(store: ArtifactStore, artifact_kind: str, dto: Any) -> ArtifactRef:
    return store.put(artifact_kind, canonical_dto_bytes(dto), {"dto": artifact_kind})


def _load_dto(
    resolver: ArtifactResolver,
    ref: ArtifactRef,
    artifact_kind: str,
    decoder: Callable[[Mapping[str, Any]], T],
) -> T:
    _require_kind(ref, artifact_kind)
    import json

    return decoder(json.loads(resolver.resolve_canonical_bytes(ref).decode("utf-8")))


def store_matrix_blob(store: ArtifactStore, blob: MatrixBlob) -> ArtifactRef:
    return store.put(
        MATRIX_BLOB_ARTIFACT_KIND,
        canonical_dto_bytes(blob),
        {"blob_id": blob.blob_id, "shape": list(blob.shape), "dtype": blob.dtype},
    )


def load_matrix_blob(resolver: ArtifactResolver, ref: ArtifactRef) -> MatrixBlob:
    _require_kind(ref, MATRIX_BLOB_ARTIFACT_KIND)
    import json

    return MatrixBlob.from_dict(
        json.loads(resolver.resolve_canonical_bytes(ref).decode("utf-8"))
    )


def store_critic_feature_spec(
    store: ArtifactStore, dto: CriticFeatureSpecDTO
) -> ArtifactRef:
    return store_dto(store, CRITIC_FEATURE_SPEC_ARTIFACT_KIND, dto)


def load_critic_feature_spec(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> CriticFeatureSpecDTO:
    return _load_dto(
        resolver, ref, CRITIC_FEATURE_SPEC_ARTIFACT_KIND, CriticFeatureSpecDTO.from_dict
    )


def store_critic_feature_batch(
    store: ArtifactStore, dto: CriticFeatureBatchDTO
) -> ArtifactRef:
    return store_dto(store, CRITIC_FEATURE_BATCH_ARTIFACT_KIND, dto)


def load_critic_feature_batch(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> CriticFeatureBatchDTO:
    return _load_dto(
        resolver,
        ref,
        CRITIC_FEATURE_BATCH_ARTIFACT_KIND,
        CriticFeatureBatchDTO.from_dict,
    )


def store_critic_contract(store: ArtifactStore, dto: CriticContractDTO) -> ArtifactRef:
    return store_dto(store, CRITIC_CONTRACT_ARTIFACT_KIND, dto)


def load_critic_contract(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> CriticContractDTO:
    return _load_dto(
        resolver, ref, CRITIC_CONTRACT_ARTIFACT_KIND, CriticContractDTO.from_dict
    )


def store_critic_label_batch(
    store: ArtifactStore, dto: CriticLabelBatchDTO
) -> ArtifactRef:
    return store_dto(store, CRITIC_LABEL_BATCH_ARTIFACT_KIND, dto)


def load_critic_label_batch(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> CriticLabelBatchDTO:
    return _load_dto(
        resolver, ref, CRITIC_LABEL_BATCH_ARTIFACT_KIND, CriticLabelBatchDTO.from_dict
    )


def store_critic_fit_spec(store: ArtifactStore, dto: CriticFitSpecDTO) -> ArtifactRef:
    return store_dto(store, CRITIC_FIT_SPEC_ARTIFACT_KIND, dto)


def load_critic_fit_spec(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> CriticFitSpecDTO:
    return _load_dto(
        resolver, ref, CRITIC_FIT_SPEC_ARTIFACT_KIND, CriticFitSpecDTO.from_dict
    )


def store_critic_calibration(
    store: ArtifactStore, dto: CriticCalibrationDTO
) -> ArtifactRef:
    return store_dto(store, CRITIC_CALIBRATION_ARTIFACT_KIND, dto)


def load_critic_calibration(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> CriticCalibrationDTO:
    return _load_dto(
        resolver, ref, CRITIC_CALIBRATION_ARTIFACT_KIND, CriticCalibrationDTO.from_dict
    )


def store_critic_readout(
    store: ArtifactStore, dto: CriticReadoutArtifactDTO
) -> ArtifactRef:
    return store_dto(store, CRITIC_READOUT_ARTIFACT_KIND, dto)


def load_critic_readout(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> CriticReadoutArtifactDTO:
    return _load_dto(
        resolver, ref, CRITIC_READOUT_ARTIFACT_KIND, CriticReadoutArtifactDTO.from_dict
    )


def store_critic_score_request(
    store: ArtifactStore, dto: CriticScoreRequestDTO
) -> ArtifactRef:
    return store_dto(store, CRITIC_SCORE_REQUEST_ARTIFACT_KIND, dto)


def load_critic_score_request(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> CriticScoreRequestDTO:
    return _load_dto(
        resolver,
        ref,
        CRITIC_SCORE_REQUEST_ARTIFACT_KIND,
        CriticScoreRequestDTO.from_dict,
    )


def store_critic_score_result(
    store: ArtifactStore, dto: CriticScoreResultDTO
) -> ArtifactRef:
    return store_dto(store, CRITIC_SCORE_RESULT_ARTIFACT_KIND, dto)


def load_critic_score_result(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> CriticScoreResultDTO:
    return _load_dto(
        resolver, ref, CRITIC_SCORE_RESULT_ARTIFACT_KIND, CriticScoreResultDTO.from_dict
    )


def store_critic_score_receipt(
    store: ArtifactStore, dto: CriticScoreReceiptDTO
) -> ArtifactRef:
    return store_dto(store, CRITIC_SCORE_RECEIPT_ARTIFACT_KIND, dto)


def load_critic_score_receipt(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> CriticScoreReceiptDTO:
    return _load_dto(
        resolver,
        ref,
        CRITIC_SCORE_RECEIPT_ARTIFACT_KIND,
        CriticScoreReceiptDTO.from_dict,
    )


@dataclass(frozen=True, slots=True)
class ResolvedCriticFeatureBatchAggregate:
    batch: CriticFeatureBatchDTO
    spec: CriticFeatureSpecDTO
    values_blob: MatrixBlob

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.values_blob.to_array(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ResolvedCriticLabelBatchAggregate:
    batch: CriticLabelBatchDTO
    contract: CriticContractDTO
    labels_blob: MatrixBlob

    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self.labels_blob.to_array(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ResolvedCriticReadoutAggregate:
    readout: CriticReadoutArtifactDTO
    contract: CriticContractDTO
    feature_spec: CriticFeatureSpecDTO
    fit_spec: CriticFitSpecDTO
    center_blob: MatrixBlob
    scale_blob: MatrixBlob
    coefficients_blob: MatrixBlob
    intercept_blob: MatrixBlob
    training_features: ResolvedCriticFeatureBatchAggregate
    training_labels: ResolvedCriticLabelBatchAggregate
    calibration: CriticCalibrationDTO | None = None


def load_critic_feature_batch_aggregate(
    ref: ArtifactRef, resolver: ArtifactResolver
) -> ResolvedCriticFeatureBatchAggregate:
    batch = load_critic_feature_batch(resolver, ref)
    spec = load_critic_feature_spec(resolver, batch.feature_spec_ref)
    blob = load_matrix_blob(resolver, batch.values_blob_ref)
    if blob.shape != batch.values_shape or blob.dtype != batch.values_dtype:
        raise CriticFeatureSchemaMismatchError("feature batch matrix metadata mismatch")
    if blob.ndim != 2 or blob.shape[1] != len(spec.features):
        raise CriticFeatureSchemaMismatchError("feature dimensions do not match spec")
    values = blob.to_array()
    if not np.isfinite(values).all():
        raise CriticFeatureSchemaMismatchError("feature values must be finite")
    return ResolvedCriticFeatureBatchAggregate(batch=batch, spec=spec, values_blob=blob)


def load_critic_label_batch_aggregate(
    ref: ArtifactRef, resolver: ArtifactResolver
) -> ResolvedCriticLabelBatchAggregate:
    batch = load_critic_label_batch(resolver, ref)
    contract = load_critic_contract(resolver, batch.critic_contract_ref)
    blob = load_matrix_blob(resolver, batch.labels_blob_ref)
    if blob.shape != batch.labels_shape:
        raise CriticContractMismatchError("label batch metadata mismatch")
    labels = np.asarray(blob.to_array(), dtype=np.float64)
    if labels.ndim != 1 or not np.isin(labels, [0.0, 1.0]).all():
        raise CriticContractMismatchError("labels must be a binary 1D vector")
    return ResolvedCriticLabelBatchAggregate(
        batch=batch, contract=contract, labels_blob=blob
    )


def load_critic_readout_aggregate(
    ref: ArtifactRef, resolver: ArtifactResolver
) -> ResolvedCriticReadoutAggregate:
    readout = load_critic_readout(resolver, ref)
    aggregate = ResolvedCriticReadoutAggregate(
        readout=readout,
        contract=load_critic_contract(resolver, readout.critic_contract_ref),
        feature_spec=load_critic_feature_spec(resolver, readout.feature_spec_ref),
        fit_spec=load_critic_fit_spec(resolver, readout.fit_spec_ref),
        center_blob=load_matrix_blob(resolver, readout.center_blob_ref),
        scale_blob=load_matrix_blob(resolver, readout.scale_blob_ref),
        coefficients_blob=load_matrix_blob(resolver, readout.coefficients_blob_ref),
        intercept_blob=load_matrix_blob(resolver, readout.intercept_blob_ref),
        training_features=load_critic_feature_batch_aggregate(
            readout.training_feature_batch_ref, resolver
        ),
        training_labels=load_critic_label_batch_aggregate(
            readout.training_label_batch_ref, resolver
        ),
        calibration=None
        if readout.calibration_ref is None
        else load_critic_calibration(resolver, readout.calibration_ref),
    )
    validate_critic_readout_aggregate(aggregate)
    return aggregate


def validate_critic_readout_aggregate(
    aggregate: ResolvedCriticReadoutAggregate,
) -> None:
    readout = aggregate.readout
    if (
        readout.feature_spec_ref.artifact_id
        != aggregate.training_features.batch.feature_spec_ref.artifact_id
    ):
        raise CriticReadoutIntegrityError(
            "training feature spec does not match readout"
        )
    if (
        readout.critic_contract_ref.artifact_id
        != aggregate.training_labels.batch.critic_contract_ref.artifact_id
    ):
        raise CriticReadoutIntegrityError(
            "training label contract does not match readout"
        )
    if (
        aggregate.training_features.batch.item_refs
        != aggregate.training_labels.batch.item_refs
    ):
        raise CriticReadoutIntegrityError("training rows are not exactly aligned")
    width = len(aggregate.feature_spec.features)
    center = aggregate.center_blob.to_array()
    scale = aggregate.scale_blob.to_array()
    coefficients = aggregate.coefficients_blob.to_array()
    intercept = aggregate.intercept_blob.to_array()
    if (
        center.shape != (width,)
        or scale.shape != (width,)
        or coefficients.shape != (width,)
    ):
        raise CriticReadoutIntegrityError("readout vector dimensions do not close")
    if intercept.shape != (1,):
        raise CriticReadoutIntegrityError("intercept dimensions do not close")
    if (
        not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(coefficients).all()
        or not np.isfinite(intercept).all()
    ):
        raise CriticReadoutIntegrityError("readout arrays must be finite")
    if np.any(scale <= 0.0):
        raise CriticReadoutIntegrityError("scale must be positive")
