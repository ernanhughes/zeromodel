from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from zeromodel.artifacts import ArtifactRef, canonical_json_bytes
from zeromodel.core.artifact import VPMValidationError

from zeromodel.search._canonical import (
    digest_payload,
    freeze_json,
    ref_from_dict,
    ref_payload,
    refs_from_dict,
    refs_payload,
    require_finite,
    require_nonempty,
    require_unique,
    thaw_json,
)

SEARCH_SPEC_VERSION = "zeromodel-search/v1"


def _check_id(actual: str, expected: str, field: str) -> None:
    if actual != expected:
        raise VPMValidationError(f"{field} does not match canonical content")


@dataclass(frozen=True, slots=True)
class RepresentationSpecDTO:
    provider_id: str
    model_id: str
    model_revision: str
    dimensions: int
    dtype: str
    pooling_policy: str
    normalization_policy: str
    preprocessing_contract_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    representation_spec_id: str = ""
    spec_version: str = SEARCH_SPEC_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "provider_id",
            "model_id",
            "model_revision",
            "dtype",
            "pooling_policy",
            "normalization_policy",
            "preprocessing_contract_id",
        ):
            object.__setattr__(
                self,
                field_name,
                require_nonempty(getattr(self, field_name), field_name),
            )
        if int(self.dimensions) <= 0:
            raise VPMValidationError("dimensions must be positive")
        object.__setattr__(self, "dimensions", int(self.dimensions))
        if self.dtype not in {"float32", "float64"}:
            raise VPMValidationError("representation dtype must be float32 or float64")
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_representation_spec_id(self)
        object.__setattr__(
            self, "representation_spec_id", self.representation_spec_id or expected
        )
        _check_id(self.representation_spec_id, expected, "representation_spec_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dimensions": self.dimensions,
            "dtype": self.dtype,
            "pooling_policy": self.pooling_policy,
            "normalization_policy": self.normalization_policy,
            "preprocessing_contract_id": self.preprocessing_contract_id,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["representation_spec_id"] = self.representation_spec_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepresentationSpecDTO":
        return cls(
            provider_id=str(data["provider_id"]),
            model_id=str(data["model_id"]),
            model_revision=str(data["model_revision"]),
            dimensions=int(data["dimensions"]),
            dtype=str(data["dtype"]),
            pooling_policy=str(data["pooling_policy"]),
            normalization_policy=str(data["normalization_policy"]),
            preprocessing_contract_id=str(data["preprocessing_contract_id"]),
            metadata=data.get("metadata") or {},
            representation_spec_id=str(data.get("representation_spec_id") or ""),
            spec_version=str(data.get("spec_version", SEARCH_SPEC_VERSION)),
        )


def compute_representation_spec_id(spec: RepresentationSpecDTO) -> str:
    return digest_payload(
        "zeromodel.search.representation_spec.v1", spec.identity_payload()
    )


@dataclass(frozen=True, slots=True)
class RepresentationBatchDTO:
    representation_spec_ref: ArtifactRef
    matrix_blob_ref: ArtifactRef
    item_refs: Tuple[ArtifactRef, ...]
    matrix_shape: Tuple[int, int]
    matrix_dtype: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    batch_id: str = ""
    spec_version: str = SEARCH_SPEC_VERSION

    def __post_init__(self) -> None:
        if len(self.matrix_shape) != 2 or any(int(v) <= 0 for v in self.matrix_shape):
            raise VPMValidationError("matrix_shape must be positive 2D")
        object.__setattr__(
            self, "matrix_shape", tuple(int(v) for v in self.matrix_shape)
        )
        if len(self.item_refs) != self.matrix_shape[0]:
            raise VPMValidationError("item_refs must match matrix row count")
        require_unique([ref.artifact_id for ref in self.item_refs], "item_refs")
        object.__setattr__(self, "item_refs", tuple(self.item_refs))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_representation_batch_id(self)
        object.__setattr__(self, "batch_id", self.batch_id or expected)
        _check_id(self.batch_id, expected, "batch_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "representation_spec_ref": ref_payload(self.representation_spec_ref),
            "matrix_blob_ref": ref_payload(self.matrix_blob_ref),
            "item_refs": refs_payload(self.item_refs),
            "matrix_shape": list(self.matrix_shape),
            "matrix_dtype": self.matrix_dtype,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["batch_id"] = self.batch_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepresentationBatchDTO":
        return cls(
            representation_spec_ref=ref_from_dict(data["representation_spec_ref"]),
            matrix_blob_ref=ref_from_dict(data["matrix_blob_ref"]),
            item_refs=refs_from_dict(data["item_refs"]),
            matrix_shape=tuple(data["matrix_shape"]),
            matrix_dtype=str(data["matrix_dtype"]),
            metadata=data.get("metadata") or {},
            batch_id=str(data.get("batch_id") or ""),
            spec_version=str(data.get("spec_version", SEARCH_SPEC_VERSION)),
        )


def compute_representation_batch_id(batch: RepresentationBatchDTO) -> str:
    return digest_payload(
        "zeromodel.search.representation_batch.v1", batch.identity_payload()
    )


@dataclass(frozen=True, slots=True)
class RelationCoordinateSpecDTO:
    coordinate_id: str
    description: str
    measurement_contract_id: str
    units: Optional[str] = None
    orientation: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coordinate_id", require_nonempty(self.coordinate_id, "coordinate_id")
        )
        object.__setattr__(
            self,
            "measurement_contract_id",
            require_nonempty(self.measurement_contract_id, "measurement_contract_id"),
        )
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_id": self.coordinate_id,
            "description": self.description,
            "measurement_contract_id": self.measurement_contract_id,
            "units": self.units,
            "orientation": self.orientation,
            "metadata": thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationCoordinateSpecDTO":
        return cls(
            coordinate_id=str(data["coordinate_id"]),
            description=str(data.get("description", "")),
            measurement_contract_id=str(data["measurement_contract_id"]),
            units=data.get("units"),
            orientation=data.get("orientation"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True, slots=True)
class RelationContractDTO:
    relation_id: str
    version: str
    subject_kind: str
    coordinates: Tuple[RelationCoordinateSpecDTO, ...]
    scaling_kind: str = "robust_median_iqr"
    distance_kind: str = "chebyshev"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    relation_contract_id: str = ""
    spec_version: str = SEARCH_SPEC_VERSION

    def __post_init__(self) -> None:
        for field_name in ("relation_id", "version", "subject_kind"):
            object.__setattr__(
                self,
                field_name,
                require_nonempty(getattr(self, field_name), field_name),
            )
        if not self.coordinates:
            raise VPMValidationError("relation contract requires coordinates")
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        require_unique(
            [coord.coordinate_id for coord in self.coordinates], "coordinates"
        )
        if self.scaling_kind != "robust_median_iqr":
            raise VPMValidationError("only robust_median_iqr scaling is supported")
        if self.distance_kind != "chebyshev":
            raise VPMValidationError("only chebyshev distance is supported")
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_relation_contract_id(self)
        object.__setattr__(
            self, "relation_contract_id", self.relation_contract_id or expected
        )
        _check_id(self.relation_contract_id, expected, "relation_contract_id")

    @property
    def coordinate_ids(self) -> tuple[str, ...]:
        return tuple(coord.coordinate_id for coord in self.coordinates)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "relation_id": self.relation_id,
            "version": self.version,
            "subject_kind": self.subject_kind,
            "coordinates": [coord.to_dict() for coord in self.coordinates],
            "scaling_kind": self.scaling_kind,
            "distance_kind": self.distance_kind,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["relation_contract_id"] = self.relation_contract_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationContractDTO":
        return cls(
            relation_id=str(data["relation_id"]),
            version=str(data["version"]),
            subject_kind=str(data["subject_kind"]),
            coordinates=tuple(
                RelationCoordinateSpecDTO.from_dict(item)
                for item in data["coordinates"]
            ),
            scaling_kind=str(data.get("scaling_kind", "robust_median_iqr")),
            distance_kind=str(data.get("distance_kind", "chebyshev")),
            metadata=data.get("metadata") or {},
            relation_contract_id=str(data.get("relation_contract_id") or ""),
            spec_version=str(data.get("spec_version", SEARCH_SPEC_VERSION)),
        )


def compute_relation_contract_id(contract: RelationContractDTO) -> str:
    return digest_payload(
        "zeromodel.search.relation_contract.v1", contract.identity_payload()
    )


@dataclass(frozen=True, slots=True)
class RelationCoordinateBatchDTO:
    relation_contract_ref: ArtifactRef
    values_blob_ref: ArtifactRef
    item_refs: Tuple[ArtifactRef, ...]
    values_shape: Tuple[int, int]
    values_dtype: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    coordinate_batch_id: str = ""
    spec_version: str = SEARCH_SPEC_VERSION

    def __post_init__(self) -> None:
        if len(self.values_shape) != 2 or any(int(v) <= 0 for v in self.values_shape):
            raise VPMValidationError("values_shape must be positive 2D")
        object.__setattr__(
            self, "values_shape", tuple(int(v) for v in self.values_shape)
        )
        if len(self.item_refs) != self.values_shape[0]:
            raise VPMValidationError("item_refs must match values row count")
        require_unique([ref.artifact_id for ref in self.item_refs], "item_refs")
        object.__setattr__(self, "item_refs", tuple(self.item_refs))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_relation_coordinate_batch_id(self)
        object.__setattr__(
            self, "coordinate_batch_id", self.coordinate_batch_id or expected
        )
        _check_id(self.coordinate_batch_id, expected, "coordinate_batch_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "relation_contract_ref": ref_payload(self.relation_contract_ref),
            "values_blob_ref": ref_payload(self.values_blob_ref),
            "item_refs": refs_payload(self.item_refs),
            "values_shape": list(self.values_shape),
            "values_dtype": self.values_dtype,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["coordinate_batch_id"] = self.coordinate_batch_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationCoordinateBatchDTO":
        return cls(
            relation_contract_ref=ref_from_dict(data["relation_contract_ref"]),
            values_blob_ref=ref_from_dict(data["values_blob_ref"]),
            item_refs=refs_from_dict(data["item_refs"]),
            values_shape=tuple(data["values_shape"]),
            values_dtype=str(data["values_dtype"]),
            metadata=data.get("metadata") or {},
            coordinate_batch_id=str(data.get("coordinate_batch_id") or ""),
            spec_version=str(data.get("spec_version", SEARCH_SPEC_VERSION)),
        )


def compute_relation_coordinate_batch_id(batch: RelationCoordinateBatchDTO) -> str:
    return digest_payload(
        "zeromodel.search.relation_coordinate_batch.v1", batch.identity_payload()
    )


@dataclass(frozen=True, slots=True)
class RelationFitSpecDTO:
    algorithm: str = "independent_ridge_projection"
    alpha: float = 1.0
    minimum_relation_scale: float = 1.0
    fit_version: str = "v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fit_spec_id: str = ""
    spec_version: str = SEARCH_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.algorithm != "independent_ridge_projection":
            raise VPMValidationError("only independent_ridge_projection is supported")
        object.__setattr__(
            self, "fit_version", require_nonempty(self.fit_version, "fit_version")
        )
        alpha = require_finite(self.alpha, "alpha")
        if alpha < 0.0:
            raise VPMValidationError("alpha must be non-negative")
        scale = require_finite(self.minimum_relation_scale, "minimum_relation_scale")
        if scale <= 0.0:
            raise VPMValidationError("minimum_relation_scale must be positive")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "minimum_relation_scale", scale)
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_relation_fit_spec_id(self)
        object.__setattr__(self, "fit_spec_id", self.fit_spec_id or expected)
        _check_id(self.fit_spec_id, expected, "fit_spec_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "algorithm": self.algorithm,
            "alpha": self.alpha,
            "minimum_relation_scale": self.minimum_relation_scale,
            "fit_version": self.fit_version,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["fit_spec_id"] = self.fit_spec_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationFitSpecDTO":
        return cls(
            algorithm=str(data.get("algorithm", "independent_ridge_projection")),
            alpha=float(data.get("alpha", 1.0)),
            minimum_relation_scale=float(data.get("minimum_relation_scale", 1.0)),
            fit_version=str(data.get("fit_version", "v1")),
            metadata=data.get("metadata") or {},
            fit_spec_id=str(data.get("fit_spec_id") or ""),
            spec_version=str(data.get("spec_version", SEARCH_SPEC_VERSION)),
        )


def compute_relation_fit_spec_id(spec: RelationFitSpecDTO) -> str:
    return digest_payload(
        "zeromodel.search.relation_fit_spec.v1", spec.identity_payload()
    )


@dataclass(frozen=True, slots=True)
class RelationReadoutArtifactDTO:
    representation_spec_ref: ArtifactRef
    relation_contract_ref: ArtifactRef
    fit_spec_ref: ArtifactRef
    coefficients_blob_ref: ArtifactRef
    intercept_blob_ref: ArtifactRef
    relation_median_blob_ref: ArtifactRef
    relation_scale_blob_ref: ArtifactRef
    embedding_mean_blob_ref: ArtifactRef
    training_representation_batch_ref: ArtifactRef
    training_coordinate_batch_ref: ArtifactRef
    metadata: Mapping[str, Any] = field(default_factory=dict)
    readout_id: str = ""
    spec_version: str = SEARCH_SPEC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_relation_readout_id(self)
        object.__setattr__(self, "readout_id", self.readout_id or expected)
        _check_id(self.readout_id, expected, "readout_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "representation_spec_ref": ref_payload(self.representation_spec_ref),
            "relation_contract_ref": ref_payload(self.relation_contract_ref),
            "fit_spec_ref": ref_payload(self.fit_spec_ref),
            "coefficients_blob_ref": ref_payload(self.coefficients_blob_ref),
            "intercept_blob_ref": ref_payload(self.intercept_blob_ref),
            "relation_median_blob_ref": ref_payload(self.relation_median_blob_ref),
            "relation_scale_blob_ref": ref_payload(self.relation_scale_blob_ref),
            "embedding_mean_blob_ref": ref_payload(self.embedding_mean_blob_ref),
            "training_representation_batch_ref": ref_payload(
                self.training_representation_batch_ref
            ),
            "training_coordinate_batch_ref": ref_payload(
                self.training_coordinate_batch_ref
            ),
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["readout_id"] = self.readout_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationReadoutArtifactDTO":
        return cls(
            representation_spec_ref=ref_from_dict(data["representation_spec_ref"]),
            relation_contract_ref=ref_from_dict(data["relation_contract_ref"]),
            fit_spec_ref=ref_from_dict(data["fit_spec_ref"]),
            coefficients_blob_ref=ref_from_dict(data["coefficients_blob_ref"]),
            intercept_blob_ref=ref_from_dict(data["intercept_blob_ref"]),
            relation_median_blob_ref=ref_from_dict(data["relation_median_blob_ref"]),
            relation_scale_blob_ref=ref_from_dict(data["relation_scale_blob_ref"]),
            embedding_mean_blob_ref=ref_from_dict(data["embedding_mean_blob_ref"]),
            training_representation_batch_ref=ref_from_dict(
                data["training_representation_batch_ref"]
            ),
            training_coordinate_batch_ref=ref_from_dict(
                data["training_coordinate_batch_ref"]
            ),
            metadata=data.get("metadata") or {},
            readout_id=str(data.get("readout_id") or ""),
            spec_version=str(data.get("spec_version", SEARCH_SPEC_VERSION)),
        )


def compute_relation_readout_id(readout: RelationReadoutArtifactDTO) -> str:
    return digest_payload(
        "zeromodel.search.relation_readout.v1", readout.identity_payload()
    )


@dataclass(frozen=True, slots=True)
class RelationSearchRequestDTO:
    readout_ref: ArtifactRef
    corpus_ref: ArtifactRef
    query_representation_blob_ref: ArtifactRef
    k: int
    query_artifact_ref: Optional[ArtifactRef] = None
    exclude_refs: Tuple[ArtifactRef, ...] = field(default_factory=tuple)
    include_cosine_comparison: bool = False
    candidate_generation_spec_ref: Optional[ArtifactRef] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = ""
    spec_version: str = SEARCH_SPEC_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k <= 0:
            raise VPMValidationError("k must be a positive integer")
        object.__setattr__(self, "exclude_refs", tuple(self.exclude_refs))
        require_unique([ref.artifact_id for ref in self.exclude_refs], "exclude_refs")
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_relation_search_request_id(self)
        object.__setattr__(self, "request_id", self.request_id or expected)
        _check_id(self.request_id, expected, "request_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "readout_ref": ref_payload(self.readout_ref),
            "corpus_ref": ref_payload(self.corpus_ref),
            "query_artifact_ref": None
            if self.query_artifact_ref is None
            else ref_payload(self.query_artifact_ref),
            "query_representation_blob_ref": ref_payload(
                self.query_representation_blob_ref
            ),
            "k": self.k,
            "exclude_refs": refs_payload(self.exclude_refs),
            "include_cosine_comparison": self.include_cosine_comparison,
            "candidate_generation_spec_ref": None
            if self.candidate_generation_spec_ref is None
            else ref_payload(self.candidate_generation_spec_ref),
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["request_id"] = self.request_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationSearchRequestDTO":
        query = data.get("query_artifact_ref")
        candidate = data.get("candidate_generation_spec_ref")
        return cls(
            readout_ref=ref_from_dict(data["readout_ref"]),
            corpus_ref=ref_from_dict(data["corpus_ref"]),
            query_artifact_ref=None if query is None else ref_from_dict(query),
            query_representation_blob_ref=ref_from_dict(
                data["query_representation_blob_ref"]
            ),
            k=int(data["k"]),
            exclude_refs=refs_from_dict(data.get("exclude_refs") or []),
            include_cosine_comparison=bool(
                data.get("include_cosine_comparison", False)
            ),
            candidate_generation_spec_ref=None
            if candidate is None
            else ref_from_dict(candidate),
            metadata=data.get("metadata") or {},
            request_id=str(data.get("request_id") or ""),
            spec_version=str(data.get("spec_version", SEARCH_SPEC_VERSION)),
        )


def compute_relation_search_request_id(request: RelationSearchRequestDTO) -> str:
    return digest_payload("zeromodel.search.request.v1", request.identity_payload())


@dataclass(frozen=True, slots=True)
class RelationSearchHitDTO:
    artifact_ref: ArtifactRef
    rank: int
    relation_distance: float
    predicted_coordinates: Tuple[float, ...]
    coordinate_deltas: Tuple[float, ...]
    cosine_distance: Optional[float] = None
    cosine_rank: Optional[int] = None
    relation_rank: int = 0
    rank_shift: Optional[int] = None
    dominant_coordinate_id: str = ""
    dominant_coordinate_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise VPMValidationError("rank must begin at one")
        if self.relation_rank < 1:
            object.__setattr__(self, "relation_rank", self.rank)
        object.__setattr__(
            self,
            "relation_distance",
            require_finite(self.relation_distance, "relation_distance"),
        )
        predicted = tuple(
            require_finite(v, "predicted_coordinates")
            for v in self.predicted_coordinates
        )
        deltas = tuple(
            require_finite(v, "coordinate_deltas") for v in self.coordinate_deltas
        )
        if len(predicted) != len(deltas):
            raise VPMValidationError("coordinate lengths must match")
        object.__setattr__(self, "predicted_coordinates", predicted)
        object.__setattr__(self, "coordinate_deltas", deltas)
        if self.cosine_distance is not None:
            object.__setattr__(
                self,
                "cosine_distance",
                require_finite(self.cosine_distance, "cosine_distance"),
            )
        if self.cosine_rank is not None and self.cosine_rank < 1:
            raise VPMValidationError("cosine_rank must begin at one")
        object.__setattr__(
            self,
            "dominant_coordinate_delta",
            require_finite(self.dominant_coordinate_delta, "dominant_coordinate_delta"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_ref": ref_payload(self.artifact_ref),
            "rank": self.rank,
            "relation_distance": self.relation_distance,
            "predicted_coordinates": list(self.predicted_coordinates),
            "coordinate_deltas": list(self.coordinate_deltas),
            "cosine_distance": self.cosine_distance,
            "cosine_rank": self.cosine_rank,
            "relation_rank": self.relation_rank,
            "rank_shift": self.rank_shift,
            "dominant_coordinate_id": self.dominant_coordinate_id,
            "dominant_coordinate_delta": self.dominant_coordinate_delta,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationSearchHitDTO":
        return cls(
            artifact_ref=ref_from_dict(data["artifact_ref"]),
            rank=int(data["rank"]),
            relation_distance=float(data["relation_distance"]),
            predicted_coordinates=tuple(
                float(v) for v in data["predicted_coordinates"]
            ),
            coordinate_deltas=tuple(float(v) for v in data["coordinate_deltas"]),
            cosine_distance=None
            if data.get("cosine_distance") is None
            else float(data["cosine_distance"]),
            cosine_rank=None
            if data.get("cosine_rank") is None
            else int(data["cosine_rank"]),
            relation_rank=int(data.get("relation_rank", data["rank"])),
            rank_shift=None
            if data.get("rank_shift") is None
            else int(data["rank_shift"]),
            dominant_coordinate_id=str(data.get("dominant_coordinate_id", "")),
            dominant_coordinate_delta=float(data.get("dominant_coordinate_delta", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class RelationSearchResultDTO:
    request_ref: ArtifactRef
    readout_ref: ArtifactRef
    corpus_ref: ArtifactRef
    relation_contract_ref: ArtifactRef
    representation_spec_ref: ArtifactRef
    query_projected_coordinates: Tuple[float, ...]
    hits: Tuple[RelationSearchHitDTO, ...]
    total_candidates: int
    evaluated_candidates: int
    ordering_contract: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result_id: str = ""
    spec_version: str = SEARCH_SPEC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_projected_coordinates",
            tuple(
                require_finite(v, "query_projected_coordinates")
                for v in self.query_projected_coordinates
            ),
        )
        object.__setattr__(self, "hits", tuple(self.hits))
        ranks = [hit.rank for hit in self.hits]
        if ranks != list(range(1, len(ranks) + 1)):
            raise VPMValidationError("result ranks must be contiguous")
        if self.total_candidates < 0 or self.evaluated_candidates < 0:
            raise VPMValidationError("candidate counts must be non-negative")
        object.__setattr__(
            self,
            "ordering_contract",
            require_nonempty(self.ordering_contract, "ordering_contract"),
        )
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_relation_search_result_id(self)
        object.__setattr__(self, "result_id", self.result_id or expected)
        _check_id(self.result_id, expected, "result_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "request_ref": ref_payload(self.request_ref),
            "readout_ref": ref_payload(self.readout_ref),
            "corpus_ref": ref_payload(self.corpus_ref),
            "relation_contract_ref": ref_payload(self.relation_contract_ref),
            "representation_spec_ref": ref_payload(self.representation_spec_ref),
            "query_projected_coordinates": list(self.query_projected_coordinates),
            "hits": [hit.to_dict() for hit in self.hits],
            "total_candidates": self.total_candidates,
            "evaluated_candidates": self.evaluated_candidates,
            "ordering_contract": self.ordering_contract,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["result_id"] = self.result_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationSearchResultDTO":
        return cls(
            request_ref=ref_from_dict(data["request_ref"]),
            readout_ref=ref_from_dict(data["readout_ref"]),
            corpus_ref=ref_from_dict(data["corpus_ref"]),
            relation_contract_ref=ref_from_dict(data["relation_contract_ref"]),
            representation_spec_ref=ref_from_dict(data["representation_spec_ref"]),
            query_projected_coordinates=tuple(
                float(v) for v in data["query_projected_coordinates"]
            ),
            hits=tuple(RelationSearchHitDTO.from_dict(item) for item in data["hits"]),
            total_candidates=int(data["total_candidates"]),
            evaluated_candidates=int(data["evaluated_candidates"]),
            ordering_contract=str(data["ordering_contract"]),
            metadata=data.get("metadata") or {},
            result_id=str(data.get("result_id") or ""),
            spec_version=str(data.get("spec_version", SEARCH_SPEC_VERSION)),
        )


def compute_relation_search_result_id(result: RelationSearchResultDTO) -> str:
    return digest_payload("zeromodel.search.result.v1", result.identity_payload())


@dataclass(frozen=True, slots=True)
class RelationSearchReceiptDTO:
    request_ref: ArtifactRef
    result_ref: ArtifactRef
    readout_ref: ArtifactRef
    corpus_ref: ArtifactRef
    required_checks: Tuple[str, ...]
    result_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    receipt_id: str = ""
    spec_version: str = SEARCH_SPEC_VERSION

    REQUIRED = (
        "request_resolved",
        "readout_aggregate_validated",
        "corpus_aggregate_validated",
        "query_representation_resolved",
        "deterministic_ranking_recomputed",
        "result_ref_matches_recomputed_result",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_checks", tuple(self.required_checks))
        if self.required_checks != self.REQUIRED:
            raise VPMValidationError(
                "search receipt must include exactly the required checks"
            )
        object.__setattr__(
            self, "result_id", require_nonempty(self.result_id, "result_id")
        )
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_relation_search_receipt_id(self)
        object.__setattr__(self, "receipt_id", self.receipt_id or expected)
        _check_id(self.receipt_id, expected, "receipt_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "request_ref": ref_payload(self.request_ref),
            "result_ref": ref_payload(self.result_ref),
            "readout_ref": ref_payload(self.readout_ref),
            "corpus_ref": ref_payload(self.corpus_ref),
            "required_checks": list(self.required_checks),
            "result_id": self.result_id,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationSearchReceiptDTO":
        return cls(
            request_ref=ref_from_dict(data["request_ref"]),
            result_ref=ref_from_dict(data["result_ref"]),
            readout_ref=ref_from_dict(data["readout_ref"]),
            corpus_ref=ref_from_dict(data["corpus_ref"]),
            required_checks=tuple(str(v) for v in data["required_checks"]),
            result_id=str(data["result_id"]),
            metadata=data.get("metadata") or {},
            receipt_id=str(data.get("receipt_id") or ""),
            spec_version=str(data.get("spec_version", SEARCH_SPEC_VERSION)),
        )


def compute_relation_search_receipt_id(receipt: RelationSearchReceiptDTO) -> str:
    return digest_payload("zeromodel.search.receipt.v1", receipt.identity_payload())


def canonical_dto_bytes(dto: Any) -> bytes:
    return canonical_json_bytes(dto.to_dict())
