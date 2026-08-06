from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypeVar

import numpy as np

from zeromodel.artifacts import ArtifactRef, ArtifactResolver, ArtifactStore
from zeromodel.core.matrix_blob import MatrixBlob

from zeromodel.search.dto import (
    RelationContractDTO,
    RelationCoordinateBatchDTO,
    RelationFitSpecDTO,
    RelationReadoutArtifactDTO,
    RelationSearchReceiptDTO,
    RelationSearchRequestDTO,
    RelationSearchResultDTO,
    RepresentationBatchDTO,
    RepresentationSpecDTO,
    canonical_dto_bytes,
)
from zeromodel.search.errors import (
    RelationContractMismatchError,
    RelationReadoutIntegrityError,
    RepresentationMismatchError,
)
from zeromodel.search.kinds import (
    MATRIX_BLOB_ARTIFACT_KIND,
    RELATION_CONTRACT_ARTIFACT_KIND,
    RELATION_COORDINATE_BATCH_ARTIFACT_KIND,
    RELATION_FIT_SPEC_ARTIFACT_KIND,
    RELATION_READOUT_ARTIFACT_KIND,
    RELATION_SEARCH_RECEIPT_ARTIFACT_KIND,
    RELATION_SEARCH_REQUEST_ARTIFACT_KIND,
    RELATION_SEARCH_RESULT_ARTIFACT_KIND,
    REPRESENTATION_BATCH_ARTIFACT_KIND,
    REPRESENTATION_SPEC_ARTIFACT_KIND,
)

T = TypeVar("T")


def _require_kind(ref: ArtifactRef, expected: str) -> None:
    if ref.artifact_kind != expected:
        raise RelationReadoutIntegrityError(
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


def store_representation_spec(
    store: ArtifactStore, dto: RepresentationSpecDTO
) -> ArtifactRef:
    return store_dto(store, REPRESENTATION_SPEC_ARTIFACT_KIND, dto)


def load_representation_spec(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> RepresentationSpecDTO:
    return _load_dto(
        resolver,
        ref,
        REPRESENTATION_SPEC_ARTIFACT_KIND,
        RepresentationSpecDTO.from_dict,
    )


def store_representation_batch(
    store: ArtifactStore, dto: RepresentationBatchDTO
) -> ArtifactRef:
    return store_dto(store, REPRESENTATION_BATCH_ARTIFACT_KIND, dto)


def load_representation_batch(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> RepresentationBatchDTO:
    return _load_dto(
        resolver,
        ref,
        REPRESENTATION_BATCH_ARTIFACT_KIND,
        RepresentationBatchDTO.from_dict,
    )


def store_relation_contract(
    store: ArtifactStore, dto: RelationContractDTO
) -> ArtifactRef:
    return store_dto(store, RELATION_CONTRACT_ARTIFACT_KIND, dto)


def load_relation_contract(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> RelationContractDTO:
    return _load_dto(
        resolver, ref, RELATION_CONTRACT_ARTIFACT_KIND, RelationContractDTO.from_dict
    )


def store_relation_coordinate_batch(
    store: ArtifactStore, dto: RelationCoordinateBatchDTO
) -> ArtifactRef:
    return store_dto(store, RELATION_COORDINATE_BATCH_ARTIFACT_KIND, dto)


def load_relation_coordinate_batch(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> RelationCoordinateBatchDTO:
    return _load_dto(
        resolver,
        ref,
        RELATION_COORDINATE_BATCH_ARTIFACT_KIND,
        RelationCoordinateBatchDTO.from_dict,
    )


def store_relation_fit_spec(
    store: ArtifactStore, dto: RelationFitSpecDTO
) -> ArtifactRef:
    return store_dto(store, RELATION_FIT_SPEC_ARTIFACT_KIND, dto)


def load_relation_fit_spec(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> RelationFitSpecDTO:
    return _load_dto(
        resolver, ref, RELATION_FIT_SPEC_ARTIFACT_KIND, RelationFitSpecDTO.from_dict
    )


def store_relation_readout(
    store: ArtifactStore, dto: RelationReadoutArtifactDTO
) -> ArtifactRef:
    return store_dto(store, RELATION_READOUT_ARTIFACT_KIND, dto)


def load_relation_readout(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> RelationReadoutArtifactDTO:
    return _load_dto(
        resolver,
        ref,
        RELATION_READOUT_ARTIFACT_KIND,
        RelationReadoutArtifactDTO.from_dict,
    )


def store_relation_search_request(
    store: ArtifactStore, dto: RelationSearchRequestDTO
) -> ArtifactRef:
    return store_dto(store, RELATION_SEARCH_REQUEST_ARTIFACT_KIND, dto)


def load_relation_search_request(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> RelationSearchRequestDTO:
    return _load_dto(
        resolver,
        ref,
        RELATION_SEARCH_REQUEST_ARTIFACT_KIND,
        RelationSearchRequestDTO.from_dict,
    )


def store_relation_search_result(
    store: ArtifactStore, dto: RelationSearchResultDTO
) -> ArtifactRef:
    return store_dto(store, RELATION_SEARCH_RESULT_ARTIFACT_KIND, dto)


def load_relation_search_result(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> RelationSearchResultDTO:
    return _load_dto(
        resolver,
        ref,
        RELATION_SEARCH_RESULT_ARTIFACT_KIND,
        RelationSearchResultDTO.from_dict,
    )


def store_relation_search_receipt(
    store: ArtifactStore, dto: RelationSearchReceiptDTO
) -> ArtifactRef:
    return store_dto(store, RELATION_SEARCH_RECEIPT_ARTIFACT_KIND, dto)


def load_relation_search_receipt(
    resolver: ArtifactResolver, ref: ArtifactRef
) -> RelationSearchReceiptDTO:
    return _load_dto(
        resolver,
        ref,
        RELATION_SEARCH_RECEIPT_ARTIFACT_KIND,
        RelationSearchReceiptDTO.from_dict,
    )


@dataclass(frozen=True, slots=True)
class ResolvedRepresentationBatchAggregate:
    batch: RepresentationBatchDTO
    spec: RepresentationSpecDTO
    matrix_blob: MatrixBlob

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.matrix_blob.to_array(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ResolvedRelationCoordinateBatchAggregate:
    batch: RelationCoordinateBatchDTO
    contract: RelationContractDTO
    values_blob: MatrixBlob

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.values_blob.to_array(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ResolvedRelationReadoutAggregate:
    readout: RelationReadoutArtifactDTO
    representation_spec: RepresentationSpecDTO
    relation_contract: RelationContractDTO
    fit_spec: RelationFitSpecDTO
    coefficients_blob: MatrixBlob
    intercept_blob: MatrixBlob
    relation_median_blob: MatrixBlob
    relation_scale_blob: MatrixBlob
    embedding_mean_blob: MatrixBlob
    training_representations: ResolvedRepresentationBatchAggregate
    training_coordinates: ResolvedRelationCoordinateBatchAggregate


def load_representation_batch_aggregate(
    ref: ArtifactRef, resolver: ArtifactResolver
) -> ResolvedRepresentationBatchAggregate:
    batch = load_representation_batch(resolver, ref)
    spec = load_representation_spec(resolver, batch.representation_spec_ref)
    blob = load_matrix_blob(resolver, batch.matrix_blob_ref)
    if blob.shape != batch.matrix_shape or blob.dtype != batch.matrix_dtype:
        raise RepresentationMismatchError(
            "representation batch matrix metadata mismatch"
        )
    if blob.shape[1] != spec.dimensions:
        raise RepresentationMismatchError("representation dimensions do not match spec")
    return ResolvedRepresentationBatchAggregate(
        batch=batch, spec=spec, matrix_blob=blob
    )


def load_relation_coordinate_batch_aggregate(
    ref: ArtifactRef, resolver: ArtifactResolver
) -> ResolvedRelationCoordinateBatchAggregate:
    batch = load_relation_coordinate_batch(resolver, ref)
    contract = load_relation_contract(resolver, batch.relation_contract_ref)
    blob = load_matrix_blob(resolver, batch.values_blob_ref)
    if blob.shape != batch.values_shape or blob.dtype != batch.values_dtype:
        raise RelationContractMismatchError(
            "relation coordinate matrix metadata mismatch"
        )
    if blob.shape[1] != len(contract.coordinates):
        raise RelationContractMismatchError("relation coordinate count mismatch")
    return ResolvedRelationCoordinateBatchAggregate(
        batch=batch, contract=contract, values_blob=blob
    )


def load_relation_readout_aggregate(
    ref: ArtifactRef, resolver: ArtifactResolver
) -> ResolvedRelationReadoutAggregate:
    readout = load_relation_readout(resolver, ref)
    aggregate = ResolvedRelationReadoutAggregate(
        readout=readout,
        representation_spec=load_representation_spec(
            resolver, readout.representation_spec_ref
        ),
        relation_contract=load_relation_contract(
            resolver, readout.relation_contract_ref
        ),
        fit_spec=load_relation_fit_spec(resolver, readout.fit_spec_ref),
        coefficients_blob=load_matrix_blob(resolver, readout.coefficients_blob_ref),
        intercept_blob=load_matrix_blob(resolver, readout.intercept_blob_ref),
        relation_median_blob=load_matrix_blob(
            resolver, readout.relation_median_blob_ref
        ),
        relation_scale_blob=load_matrix_blob(resolver, readout.relation_scale_blob_ref),
        embedding_mean_blob=load_matrix_blob(resolver, readout.embedding_mean_blob_ref),
        training_representations=load_representation_batch_aggregate(
            readout.training_representation_batch_ref, resolver
        ),
        training_coordinates=load_relation_coordinate_batch_aggregate(
            readout.training_coordinate_batch_ref, resolver
        ),
    )
    validate_relation_readout_aggregate(aggregate)
    return aggregate


def validate_relation_readout_aggregate(
    aggregate: ResolvedRelationReadoutAggregate,
) -> None:
    readout = aggregate.readout
    if (
        readout.representation_spec_ref.artifact_id
        != aggregate.training_representations.batch.representation_spec_ref.artifact_id
    ):
        raise RelationReadoutIntegrityError(
            "training representation spec does not match readout"
        )
    if (
        readout.relation_contract_ref.artifact_id
        != aggregate.training_coordinates.batch.relation_contract_ref.artifact_id
    ):
        raise RelationReadoutIntegrityError(
            "training relation contract does not match readout"
        )
    if (
        aggregate.training_representations.batch.item_refs
        != aggregate.training_coordinates.batch.item_refs
    ):
        raise RelationReadoutIntegrityError("training rows are not exactly aligned")

    coefficients = aggregate.coefficients_blob.to_array()
    intercept = aggregate.intercept_blob.to_array()
    median = aggregate.relation_median_blob.to_array()
    scale = aggregate.relation_scale_blob.to_array()
    embedding_mean = aggregate.embedding_mean_blob.to_array()
    relation_count = len(aggregate.relation_contract.coordinates)
    dimensions = aggregate.representation_spec.dimensions
    if coefficients.shape != (dimensions, relation_count):
        raise RelationReadoutIntegrityError("coefficient dimensions do not close")
    if (
        intercept.shape != (relation_count,)
        or median.shape != (relation_count,)
        or scale.shape != (relation_count,)
    ):
        raise RelationReadoutIntegrityError("relation vector dimensions do not close")
    if embedding_mean.shape != (dimensions,):
        raise RelationReadoutIntegrityError("embedding_mean dimensions do not close")
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
        raise RelationReadoutIntegrityError("projection arrays must be finite")
    if (
        not np.isfinite(median).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise RelationReadoutIntegrityError(
            "relation scale arrays must be finite and positive"
        )
