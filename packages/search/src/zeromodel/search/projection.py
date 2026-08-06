from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from zeromodel.artifacts import ArtifactStore
from zeromodel.core.matrix_blob import MatrixBlob

from zeromodel.search.dto import RelationFitSpecDTO, RelationReadoutArtifactDTO
from zeromodel.search.errors import RepresentationMismatchError, SearchValidationError
from zeromodel.search.persistence import (
    ResolvedRelationReadoutAggregate,
    ResolvedRelationCoordinateBatchAggregate,
    ResolvedRepresentationBatchAggregate,
    store_matrix_blob,
    store_relation_coordinate_batch,
    store_relation_fit_spec,
    store_relation_readout,
    store_representation_batch,
)


FloatArray = npt.NDArray[np.float64]


def _matrix(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise SearchValidationError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise SearchValidationError(f"{name} must contain only finite values")
    return array


def _vector(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise SearchValidationError(f"{name} must be a one-dimensional vector")
    if not np.isfinite(array).all():
        raise SearchValidationError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class CompiledRelationReadout:
    coefficients: FloatArray
    intercept: FloatArray
    relation_median: FloatArray
    relation_scale: FloatArray
    embedding_mean: FloatArray
    relation_names: tuple[str, ...]
    representation_dimensions: int
    representation_spec_id: str

    def __post_init__(self) -> None:
        coefficients = _matrix("coefficients", self.coefficients)
        intercept = _vector("intercept", self.intercept)
        median = _vector("relation_median", self.relation_median)
        scale = _vector("relation_scale", self.relation_scale)
        embedding_mean = _vector("embedding_mean", self.embedding_mean)
        if coefficients.shape != (
            self.representation_dimensions,
            len(self.relation_names),
        ):
            raise SearchValidationError(
                "coefficient dimensions do not match readout contract"
            )
        if any(
            value.size != len(self.relation_names)
            for value in (intercept, median, scale)
        ):
            raise SearchValidationError("relation parameter dimensions differ")
        if embedding_mean.size != self.representation_dimensions:
            raise SearchValidationError("embedding_mean dimensions differ")
        if np.any(scale <= 0.0):
            raise SearchValidationError("relation_scale must be positive")

    @property
    def relation_dimensions(self) -> int:
        return len(self.relation_names)

    @classmethod
    def fit(
        cls,
        embeddings: npt.ArrayLike,
        relation_coordinates: npt.ArrayLike,
        *,
        alpha: float,
        minimum_relation_scale: float,
        relation_names: Sequence[str],
        representation_spec_id: str,
    ) -> "CompiledRelationReadout":
        x = _matrix("embeddings", embeddings)
        y = _matrix("relation_coordinates", relation_coordinates)
        if x.shape[0] != y.shape[0]:
            raise SearchValidationError("training matrices need the same rows")
        if x.shape[0] < 2:
            raise SearchValidationError("at least two training rows are required")
        median = np.median(y, axis=0)
        q25, q75 = np.percentile(y, (25.0, 75.0), axis=0)
        scale = np.maximum(q75 - q25, float(minimum_relation_scale))
        scaled_y = (y - median) / scale
        embedding_mean = x.mean(axis=0)
        output_mean = scaled_y.mean(axis=0)
        centered_x = x - embedding_mean
        centered_y = scaled_y - output_mean
        gram = centered_x.T @ centered_x
        regularized = gram.copy()
        regularized.flat[:: regularized.shape[0] + 1] += float(alpha)
        cross = centered_x.T @ centered_y
        try:
            coefficients = np.linalg.solve(regularized, cross)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(regularized) @ cross
        intercept = output_mean - embedding_mean @ coefficients
        return cls(
            coefficients=np.asarray(coefficients, dtype=np.float64),
            intercept=np.asarray(intercept, dtype=np.float64),
            relation_median=np.asarray(median, dtype=np.float64),
            relation_scale=np.asarray(scale, dtype=np.float64),
            embedding_mean=np.asarray(embedding_mean, dtype=np.float64),
            relation_names=tuple(relation_names),
            representation_dimensions=x.shape[1],
            representation_spec_id=representation_spec_id,
        )

    def project_many(
        self, embeddings: npt.ArrayLike, *, representation_spec_id: str
    ) -> FloatArray:
        if representation_spec_id != self.representation_spec_id:
            raise RepresentationMismatchError("representation spec identity mismatch")
        matrix = _matrix("embeddings", embeddings)
        if matrix.shape[1] != self.representation_dimensions:
            raise RepresentationMismatchError(
                "embedding dimensions do not match readout"
            )
        return matrix @ self.coefficients + self.intercept

    def project_one(
        self, embedding: npt.ArrayLike, *, representation_spec_id: str
    ) -> FloatArray:
        vector = _vector("embedding", embedding)
        return self.project_many(
            vector[None, :], representation_spec_id=representation_spec_id
        )[0]

    def restore_natural_coordinates(self, scaled: npt.ArrayLike) -> FloatArray:
        return _matrix("scaled", scaled) * self.relation_scale + self.relation_median


def compiled_from_aggregate(
    aggregate: ResolvedRelationReadoutAggregate,
) -> CompiledRelationReadout:
    return CompiledRelationReadout(
        coefficients=np.asarray(
            aggregate.coefficients_blob.to_array(), dtype=np.float64
        ),
        intercept=np.asarray(aggregate.intercept_blob.to_array(), dtype=np.float64),
        relation_median=np.asarray(
            aggregate.relation_median_blob.to_array(), dtype=np.float64
        ),
        relation_scale=np.asarray(
            aggregate.relation_scale_blob.to_array(), dtype=np.float64
        ),
        embedding_mean=np.asarray(
            aggregate.embedding_mean_blob.to_array(), dtype=np.float64
        ),
        relation_names=aggregate.relation_contract.coordinate_ids,
        representation_dimensions=aggregate.representation_spec.dimensions,
        representation_spec_id=aggregate.representation_spec.representation_spec_id,
    )


def compile_relation_readout(
    *,
    store: ArtifactStore,
    representations: ResolvedRepresentationBatchAggregate,
    coordinates: ResolvedRelationCoordinateBatchAggregate,
    fit_spec: RelationFitSpecDTO,
    metadata: dict | None = None,
) -> tuple[RelationReadoutArtifactDTO, object]:
    if representations.batch.item_refs != coordinates.batch.item_refs:
        raise SearchValidationError(
            "training representation and coordinate rows must align exactly"
        )
    runtime = CompiledRelationReadout.fit(
        representations.matrix,
        coordinates.values,
        alpha=fit_spec.alpha,
        minimum_relation_scale=fit_spec.minimum_relation_scale,
        relation_names=coordinates.contract.coordinate_ids,
        representation_spec_id=representations.spec.representation_spec_id,
    )
    fit_ref = store_relation_fit_spec(store, fit_spec)
    coeff_ref = store_matrix_blob(
        store,
        MatrixBlob.from_array(
            runtime.coefficients,
            dtype="float64",
            metadata={"role": "relation_readout_coefficients"},
        ),
    )
    intercept_ref = store_matrix_blob(
        store,
        MatrixBlob.from_array(
            runtime.intercept,
            dtype="float64",
            metadata={"role": "relation_readout_intercept"},
        ),
    )
    median_ref = store_matrix_blob(
        store,
        MatrixBlob.from_array(
            runtime.relation_median,
            dtype="float64",
            metadata={"role": "relation_median"},
        ),
    )
    scale_ref = store_matrix_blob(
        store,
        MatrixBlob.from_array(
            runtime.relation_scale, dtype="float64", metadata={"role": "relation_scale"}
        ),
    )
    mean_ref = store_matrix_blob(
        store,
        MatrixBlob.from_array(
            runtime.embedding_mean, dtype="float64", metadata={"role": "embedding_mean"}
        ),
    )
    readout = RelationReadoutArtifactDTO(
        representation_spec_ref=representations.batch.representation_spec_ref,
        relation_contract_ref=coordinates.batch.relation_contract_ref,
        fit_spec_ref=fit_ref,
        coefficients_blob_ref=coeff_ref,
        intercept_blob_ref=intercept_ref,
        relation_median_blob_ref=median_ref,
        relation_scale_blob_ref=scale_ref,
        embedding_mean_blob_ref=mean_ref,
        training_representation_batch_ref=store_representation_batch(
            store, representations.batch
        ),
        training_coordinate_batch_ref=store_relation_coordinate_batch(
            store, coordinates.batch
        ),
        metadata=metadata or {},
    )
    readout_ref = store_relation_readout(store, readout)
    return readout, readout_ref
