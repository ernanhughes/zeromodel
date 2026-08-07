from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from zeromodel.artifacts import ArtifactRef, canonical_json_bytes
from zeromodel.core.artifact import VPMValidationError

from zeromodel.critic._canonical import (
    check_id,
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

CRITIC_SPEC_VERSION = "zeromodel-critic/v1"


@dataclass(frozen=True, slots=True)
class CriticFeatureDTO:
    feature_id: str
    description: str
    units: Optional[str] = None
    directionality: int = 1
    required: bool = True
    missing_policy: str = "error"
    missing_value: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "feature_id", require_nonempty(self.feature_id, "feature_id")
        )
        object.__setattr__(self, "description", str(self.description))
        if int(self.directionality) not in {-1, 1}:
            raise VPMValidationError("directionality must be +1 or -1")
        object.__setattr__(self, "directionality", int(self.directionality))
        if self.missing_policy not in {"error", "constant"}:
            raise VPMValidationError("missing_policy must be error or constant")
        if self.required and self.missing_policy != "error":
            raise VPMValidationError(
                "required features must use missing_policy='error'"
            )
        if not self.required and self.missing_policy == "constant":
            if self.missing_value is None:
                raise VPMValidationError(
                    "constant missing policy requires missing_value"
                )
            object.__setattr__(
                self,
                "missing_value",
                require_finite(float(self.missing_value), "missing_value"),
            )
        elif self.missing_value is not None:
            object.__setattr__(
                self,
                "missing_value",
                require_finite(float(self.missing_value), "missing_value"),
            )
        object.__setattr__(self, "metadata", freeze_json(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "description": self.description,
            "units": self.units,
            "directionality": self.directionality,
            "required": self.required,
            "missing_policy": self.missing_policy,
            "missing_value": self.missing_value,
            "metadata": thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticFeatureDTO":
        return cls(
            feature_id=str(data["feature_id"]),
            description=str(data.get("description", "")),
            units=data.get("units"),
            directionality=int(data.get("directionality", 1)),
            required=bool(data.get("required", True)),
            missing_policy=str(data.get("missing_policy", "error")),
            missing_value=data.get("missing_value"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True, slots=True)
class CriticFeatureSpecDTO:
    features: Tuple[CriticFeatureDTO, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    feature_spec_id: str = ""
    spec_version: str = CRITIC_SPEC_VERSION

    def __post_init__(self) -> None:
        if not self.features:
            raise VPMValidationError("feature spec requires at least one feature")
        object.__setattr__(self, "features", tuple(self.features))
        require_unique([feature.feature_id for feature in self.features], "features")
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_critic_feature_spec_id(self)
        object.__setattr__(self, "feature_spec_id", self.feature_spec_id or expected)
        check_id(self.feature_spec_id, expected, "feature_spec_id")

    @property
    def feature_ids(self) -> tuple[str, ...]:
        return tuple(feature.feature_id for feature in self.features)

    @property
    def directionality(self) -> tuple[int, ...]:
        return tuple(feature.directionality for feature in self.features)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "features": [feature.to_dict() for feature in self.features],
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["feature_spec_id"] = self.feature_spec_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticFeatureSpecDTO":
        return cls(
            features=tuple(
                CriticFeatureDTO.from_dict(item) for item in data["features"]
            ),
            metadata=data.get("metadata") or {},
            feature_spec_id=str(data.get("feature_spec_id") or ""),
            spec_version=str(data.get("spec_version", CRITIC_SPEC_VERSION)),
        )


def compute_critic_feature_spec_id(spec: CriticFeatureSpecDTO) -> str:
    return digest_payload("zeromodel.critic.feature_spec.v1", spec.identity_payload())


@dataclass(frozen=True, slots=True)
class CriticFeatureBatchDTO:
    feature_spec_ref: ArtifactRef
    values_blob_ref: ArtifactRef
    item_refs: Tuple[ArtifactRef, ...]
    values_shape: Tuple[int, int]
    values_dtype: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    batch_id: str = ""
    spec_version: str = CRITIC_SPEC_VERSION

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
        expected = compute_critic_feature_batch_id(self)
        object.__setattr__(self, "batch_id", self.batch_id or expected)
        check_id(self.batch_id, expected, "batch_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "feature_spec_ref": ref_payload(self.feature_spec_ref),
            "values_blob_ref": ref_payload(self.values_blob_ref),
            "item_refs": refs_payload(self.item_refs),
            "values_shape": list(self.values_shape),
            "values_dtype": self.values_dtype,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["batch_id"] = self.batch_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticFeatureBatchDTO":
        return cls(
            feature_spec_ref=ref_from_dict(data["feature_spec_ref"]),
            values_blob_ref=ref_from_dict(data["values_blob_ref"]),
            item_refs=refs_from_dict(data["item_refs"]),
            values_shape=tuple(data["values_shape"]),
            values_dtype=str(data["values_dtype"]),
            metadata=data.get("metadata") or {},
            batch_id=str(data.get("batch_id") or ""),
            spec_version=str(data.get("spec_version", CRITIC_SPEC_VERSION)),
        )


def compute_critic_feature_batch_id(batch: CriticFeatureBatchDTO) -> str:
    return digest_payload("zeromodel.critic.feature_batch.v1", batch.identity_payload())


@dataclass(frozen=True, slots=True)
class CriticContractDTO:
    critic_id: str
    version: str
    target_id: str
    positive_label: str
    negative_label: str
    score_semantics: str
    intended_uses: Tuple[str, ...] = field(default_factory=tuple)
    prohibited_uses: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    critic_contract_id: str = ""
    spec_version: str = CRITIC_SPEC_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "critic_id",
            "version",
            "target_id",
            "positive_label",
            "negative_label",
            "score_semantics",
        ):
            object.__setattr__(
                self,
                field_name,
                require_nonempty(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self, "intended_uses", tuple(str(v) for v in self.intended_uses)
        )
        object.__setattr__(
            self, "prohibited_uses", tuple(str(v) for v in self.prohibited_uses)
        )
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_critic_contract_id(self)
        object.__setattr__(
            self, "critic_contract_id", self.critic_contract_id or expected
        )
        check_id(self.critic_contract_id, expected, "critic_contract_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "critic_id": self.critic_id,
            "version": self.version,
            "target_id": self.target_id,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "score_semantics": self.score_semantics,
            "intended_uses": list(self.intended_uses),
            "prohibited_uses": list(self.prohibited_uses),
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["critic_contract_id"] = self.critic_contract_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticContractDTO":
        return cls(
            critic_id=str(data["critic_id"]),
            version=str(data["version"]),
            target_id=str(data["target_id"]),
            positive_label=str(data["positive_label"]),
            negative_label=str(data["negative_label"]),
            score_semantics=str(data["score_semantics"]),
            intended_uses=tuple(str(v) for v in data.get("intended_uses") or ()),
            prohibited_uses=tuple(str(v) for v in data.get("prohibited_uses") or ()),
            metadata=data.get("metadata") or {},
            critic_contract_id=str(data.get("critic_contract_id") or ""),
            spec_version=str(data.get("spec_version", CRITIC_SPEC_VERSION)),
        )


def compute_critic_contract_id(contract: CriticContractDTO) -> str:
    return digest_payload("zeromodel.critic.contract.v1", contract.identity_payload())


@dataclass(frozen=True, slots=True)
class CriticLabelBatchDTO:
    critic_contract_ref: ArtifactRef
    item_refs: Tuple[ArtifactRef, ...]
    labels_blob_ref: ArtifactRef
    labels_shape: Tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    label_batch_id: str = ""
    spec_version: str = CRITIC_SPEC_VERSION

    def __post_init__(self) -> None:
        if len(self.labels_shape) != 1 or int(self.labels_shape[0]) <= 0:
            raise VPMValidationError("labels_shape must be positive 1D")
        object.__setattr__(
            self, "labels_shape", tuple(int(v) for v in self.labels_shape)
        )
        if len(self.item_refs) != self.labels_shape[0]:
            raise VPMValidationError("item_refs must match label row count")
        require_unique([ref.artifact_id for ref in self.item_refs], "item_refs")
        object.__setattr__(self, "item_refs", tuple(self.item_refs))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_critic_label_batch_id(self)
        object.__setattr__(self, "label_batch_id", self.label_batch_id or expected)
        check_id(self.label_batch_id, expected, "label_batch_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "critic_contract_ref": ref_payload(self.critic_contract_ref),
            "item_refs": refs_payload(self.item_refs),
            "labels_blob_ref": ref_payload(self.labels_blob_ref),
            "labels_shape": list(self.labels_shape),
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["label_batch_id"] = self.label_batch_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticLabelBatchDTO":
        return cls(
            critic_contract_ref=ref_from_dict(data["critic_contract_ref"]),
            item_refs=refs_from_dict(data["item_refs"]),
            labels_blob_ref=ref_from_dict(data["labels_blob_ref"]),
            labels_shape=tuple(data["labels_shape"]),
            metadata=data.get("metadata") or {},
            label_batch_id=str(data.get("label_batch_id") or ""),
            spec_version=str(data.get("spec_version", CRITIC_SPEC_VERSION)),
        )


def compute_critic_label_batch_id(batch: CriticLabelBatchDTO) -> str:
    return digest_payload("zeromodel.critic.label_batch.v1", batch.identity_payload())


@dataclass(frozen=True, slots=True)
class CriticFitSpecDTO:
    algorithm: str = "standardized_l2_logistic_v1"
    l2_penalty: float = 1.0
    max_iterations: int = 100
    tolerance: float = 1e-8
    class_weighting: str = "none"
    fit_version: str = "v1"
    portable_payload_limit_bytes: int = 50 * 1024
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fit_spec_id: str = ""
    spec_version: str = CRITIC_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.algorithm != "standardized_l2_logistic_v1":
            raise VPMValidationError("only standardized_l2_logistic_v1 is supported")
        penalty = require_finite(self.l2_penalty, "l2_penalty")
        if penalty < 0:
            raise VPMValidationError("l2_penalty must be non-negative")
        tolerance = require_finite(self.tolerance, "tolerance")
        if tolerance <= 0:
            raise VPMValidationError("tolerance must be positive")
        if int(self.max_iterations) <= 0:
            raise VPMValidationError("max_iterations must be positive")
        if self.class_weighting not in {"none", "balanced"}:
            raise VPMValidationError("class_weighting must be none or balanced")
        if int(self.portable_payload_limit_bytes) <= 0:
            raise VPMValidationError("portable_payload_limit_bytes must be positive")
        object.__setattr__(self, "l2_penalty", penalty)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        object.__setattr__(
            self, "portable_payload_limit_bytes", int(self.portable_payload_limit_bytes)
        )
        object.__setattr__(
            self, "fit_version", require_nonempty(self.fit_version, "fit_version")
        )
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_critic_fit_spec_id(self)
        object.__setattr__(self, "fit_spec_id", self.fit_spec_id or expected)
        check_id(self.fit_spec_id, expected, "fit_spec_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "algorithm": self.algorithm,
            "l2_penalty": self.l2_penalty,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "class_weighting": self.class_weighting,
            "fit_version": self.fit_version,
            "portable_payload_limit_bytes": self.portable_payload_limit_bytes,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["fit_spec_id"] = self.fit_spec_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticFitSpecDTO":
        return cls(
            algorithm=str(data.get("algorithm", "standardized_l2_logistic_v1")),
            l2_penalty=float(data.get("l2_penalty", 1.0)),
            max_iterations=int(data.get("max_iterations", 100)),
            tolerance=float(data.get("tolerance", 1e-8)),
            class_weighting=str(data.get("class_weighting", "none")),
            fit_version=str(data.get("fit_version", "v1")),
            portable_payload_limit_bytes=int(
                data.get("portable_payload_limit_bytes", 50 * 1024)
            ),
            metadata=data.get("metadata") or {},
            fit_spec_id=str(data.get("fit_spec_id") or ""),
            spec_version=str(data.get("spec_version", CRITIC_SPEC_VERSION)),
        )


def compute_critic_fit_spec_id(spec: CriticFitSpecDTO) -> str:
    return digest_payload("zeromodel.critic.fit_spec.v1", spec.identity_payload())


@dataclass(frozen=True, slots=True)
class CriticCalibrationDTO:
    method: str = "none"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    calibration_set_ref: Optional[ArtifactRef] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    calibration_id: str = ""
    spec_version: str = CRITIC_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.method not in {"none", "platt"}:
            raise VPMValidationError("calibration method must be none or platt")
        object.__setattr__(self, "parameters", freeze_json(self.parameters))
        object.__setattr__(self, "metrics", freeze_json(self.metrics))
        expected = compute_critic_calibration_id(self)
        object.__setattr__(self, "calibration_id", self.calibration_id or expected)
        check_id(self.calibration_id, expected, "calibration_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "method": self.method,
            "parameters": thaw_json(self.parameters),
            "calibration_set_ref": None
            if self.calibration_set_ref is None
            else ref_payload(self.calibration_set_ref),
            "metrics": thaw_json(self.metrics),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["calibration_id"] = self.calibration_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticCalibrationDTO":
        calibration_set = data.get("calibration_set_ref")
        return cls(
            method=str(data.get("method", "none")),
            parameters=data.get("parameters") or {},
            calibration_set_ref=None
            if calibration_set is None
            else ref_from_dict(calibration_set),
            metrics=data.get("metrics") or {},
            calibration_id=str(data.get("calibration_id") or ""),
            spec_version=str(data.get("spec_version", CRITIC_SPEC_VERSION)),
        )


def compute_critic_calibration_id(calibration: CriticCalibrationDTO) -> str:
    return digest_payload(
        "zeromodel.critic.calibration.v1", calibration.identity_payload()
    )


@dataclass(frozen=True, slots=True)
class CriticReadoutArtifactDTO:
    critic_contract_ref: ArtifactRef
    feature_spec_ref: ArtifactRef
    fit_spec_ref: ArtifactRef
    center_blob_ref: ArtifactRef
    scale_blob_ref: ArtifactRef
    coefficients_blob_ref: ArtifactRef
    intercept_blob_ref: ArtifactRef
    training_feature_batch_ref: ArtifactRef
    training_label_batch_ref: ArtifactRef
    calibration_ref: Optional[ArtifactRef] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    readout_id: str = ""
    spec_version: str = CRITIC_SPEC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_critic_readout_id(self)
        object.__setattr__(self, "readout_id", self.readout_id or expected)
        check_id(self.readout_id, expected, "readout_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "critic_contract_ref": ref_payload(self.critic_contract_ref),
            "feature_spec_ref": ref_payload(self.feature_spec_ref),
            "fit_spec_ref": ref_payload(self.fit_spec_ref),
            "center_blob_ref": ref_payload(self.center_blob_ref),
            "scale_blob_ref": ref_payload(self.scale_blob_ref),
            "coefficients_blob_ref": ref_payload(self.coefficients_blob_ref),
            "intercept_blob_ref": ref_payload(self.intercept_blob_ref),
            "training_feature_batch_ref": ref_payload(self.training_feature_batch_ref),
            "training_label_batch_ref": ref_payload(self.training_label_batch_ref),
            "calibration_ref": None
            if self.calibration_ref is None
            else ref_payload(self.calibration_ref),
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["readout_id"] = self.readout_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticReadoutArtifactDTO":
        calibration = data.get("calibration_ref")
        return cls(
            critic_contract_ref=ref_from_dict(data["critic_contract_ref"]),
            feature_spec_ref=ref_from_dict(data["feature_spec_ref"]),
            fit_spec_ref=ref_from_dict(data["fit_spec_ref"]),
            center_blob_ref=ref_from_dict(data["center_blob_ref"]),
            scale_blob_ref=ref_from_dict(data["scale_blob_ref"]),
            coefficients_blob_ref=ref_from_dict(data["coefficients_blob_ref"]),
            intercept_blob_ref=ref_from_dict(data["intercept_blob_ref"]),
            training_feature_batch_ref=ref_from_dict(
                data["training_feature_batch_ref"]
            ),
            training_label_batch_ref=ref_from_dict(data["training_label_batch_ref"]),
            calibration_ref=None if calibration is None else ref_from_dict(calibration),
            metadata=data.get("metadata") or {},
            readout_id=str(data.get("readout_id") or ""),
            spec_version=str(data.get("spec_version", CRITIC_SPEC_VERSION)),
        )


def compute_critic_readout_id(readout: CriticReadoutArtifactDTO) -> str:
    return digest_payload("zeromodel.critic.readout.v1", readout.identity_payload())


@dataclass(frozen=True, slots=True)
class CriticThresholdContractDTO:
    reject_below: Optional[float] = None
    accept_at_or_above: Optional[float] = None
    threshold_contract_id: str = ""
    spec_version: str = CRITIC_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.reject_below is not None:
            object.__setattr__(
                self, "reject_below", require_finite(self.reject_below, "reject_below")
            )
        if self.accept_at_or_above is not None:
            object.__setattr__(
                self,
                "accept_at_or_above",
                require_finite(self.accept_at_or_above, "accept_at_or_above"),
            )
        if (
            self.reject_below is not None
            and self.accept_at_or_above is not None
            and self.reject_below > self.accept_at_or_above
        ):
            raise VPMValidationError("reject_below must be <= accept_at_or_above")
        expected = digest_payload(
            "zeromodel.critic.threshold_contract.v1", self.identity_payload()
        )
        object.__setattr__(
            self, "threshold_contract_id", self.threshold_contract_id or expected
        )
        check_id(self.threshold_contract_id, expected, "threshold_contract_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "reject_below": self.reject_below,
            "accept_at_or_above": self.accept_at_or_above,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["threshold_contract_id"] = self.threshold_contract_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticThresholdContractDTO":
        return cls(
            reject_below=data.get("reject_below"),
            accept_at_or_above=data.get("accept_at_or_above"),
            threshold_contract_id=str(data.get("threshold_contract_id") or ""),
            spec_version=str(data.get("spec_version", CRITIC_SPEC_VERSION)),
        )


@dataclass(frozen=True, slots=True)
class CriticScoreRequestDTO:
    readout_ref: ArtifactRef
    feature_batch_ref: ArtifactRef
    threshold_contract: Optional[CriticThresholdContractDTO] = None
    explanation_depth: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = ""
    spec_version: str = CRITIC_SPEC_VERSION

    def __post_init__(self) -> None:
        if int(self.explanation_depth) < 0:
            raise VPMValidationError("explanation_depth must be non-negative")
        object.__setattr__(self, "explanation_depth", int(self.explanation_depth))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_critic_score_request_id(self)
        object.__setattr__(self, "request_id", self.request_id or expected)
        check_id(self.request_id, expected, "request_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "readout_ref": ref_payload(self.readout_ref),
            "feature_batch_ref": ref_payload(self.feature_batch_ref),
            "threshold_contract": None
            if self.threshold_contract is None
            else self.threshold_contract.to_dict(),
            "explanation_depth": self.explanation_depth,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["request_id"] = self.request_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticScoreRequestDTO":
        threshold = data.get("threshold_contract")
        return cls(
            readout_ref=ref_from_dict(data["readout_ref"]),
            feature_batch_ref=ref_from_dict(data["feature_batch_ref"]),
            threshold_contract=None
            if threshold is None
            else CriticThresholdContractDTO.from_dict(threshold),
            explanation_depth=int(data.get("explanation_depth", 0)),
            metadata=data.get("metadata") or {},
            request_id=str(data.get("request_id") or ""),
            spec_version=str(data.get("spec_version", CRITIC_SPEC_VERSION)),
        )


def compute_critic_score_request_id(request: CriticScoreRequestDTO) -> str:
    return digest_payload(
        "zeromodel.critic.score_request.v1", request.identity_payload()
    )


@dataclass(frozen=True, slots=True)
class CriticFeatureContributionDTO:
    feature_id: str
    raw_value: float
    directed_value: float
    standardized_value: float
    coefficient: float
    contribution: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "feature_id", require_nonempty(self.feature_id, "feature_id")
        )
        for field_name in (
            "raw_value",
            "directed_value",
            "standardized_value",
            "coefficient",
            "contribution",
        ):
            object.__setattr__(
                self, field_name, require_finite(getattr(self, field_name), field_name)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "raw_value": self.raw_value,
            "directed_value": self.directed_value,
            "standardized_value": self.standardized_value,
            "coefficient": self.coefficient,
            "contribution": self.contribution,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticFeatureContributionDTO":
        return cls(
            feature_id=str(data["feature_id"]),
            raw_value=float(data["raw_value"]),
            directed_value=float(data["directed_value"]),
            standardized_value=float(data["standardized_value"]),
            coefficient=float(data["coefficient"]),
            contribution=float(data["contribution"]),
        )


@dataclass(frozen=True, slots=True)
class CriticItemScoreDTO:
    artifact_ref: ArtifactRef
    logit: float
    score: float
    calibrated_probability: Optional[float] = None
    verdict: Optional[str] = None
    decision_margin: float = 0.0
    feature_coverage: float = 1.0
    positive_contribution_strength: float = 0.0
    negative_contribution_strength: float = 0.0
    contributions: Tuple[CriticFeatureContributionDTO, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        for field_name in (
            "logit",
            "score",
            "decision_margin",
            "feature_coverage",
            "positive_contribution_strength",
            "negative_contribution_strength",
        ):
            object.__setattr__(
                self, field_name, require_finite(getattr(self, field_name), field_name)
            )
        if self.calibrated_probability is not None:
            object.__setattr__(
                self,
                "calibrated_probability",
                require_finite(self.calibrated_probability, "calibrated_probability"),
            )
        object.__setattr__(self, "contributions", tuple(self.contributions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_ref": ref_payload(self.artifact_ref),
            "logit": self.logit,
            "score": self.score,
            "calibrated_probability": self.calibrated_probability,
            "verdict": self.verdict,
            "decision_margin": self.decision_margin,
            "feature_coverage": self.feature_coverage,
            "positive_contribution_strength": self.positive_contribution_strength,
            "negative_contribution_strength": self.negative_contribution_strength,
            "contributions": [item.to_dict() for item in self.contributions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticItemScoreDTO":
        return cls(
            artifact_ref=ref_from_dict(data["artifact_ref"]),
            logit=float(data["logit"]),
            score=float(data["score"]),
            calibrated_probability=None
            if data.get("calibrated_probability") is None
            else float(data["calibrated_probability"]),
            verdict=data.get("verdict"),
            decision_margin=float(data.get("decision_margin", 0.0)),
            feature_coverage=float(data.get("feature_coverage", 1.0)),
            positive_contribution_strength=float(
                data.get("positive_contribution_strength", 0.0)
            ),
            negative_contribution_strength=float(
                data.get("negative_contribution_strength", 0.0)
            ),
            contributions=tuple(
                CriticFeatureContributionDTO.from_dict(item)
                for item in data.get("contributions") or ()
            ),
        )


@dataclass(frozen=True, slots=True)
class CriticScoreResultDTO:
    request_ref: ArtifactRef
    readout_ref: ArtifactRef
    feature_batch_ref: ArtifactRef
    critic_contract_ref: ArtifactRef
    feature_spec_ref: ArtifactRef
    items: Tuple[CriticItemScoreDTO, ...]
    ordering_contract: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result_id: str = ""
    spec_version: str = CRITIC_SPEC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(
            self,
            "ordering_contract",
            require_nonempty(self.ordering_contract, "ordering_contract"),
        )
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_critic_score_result_id(self)
        object.__setattr__(self, "result_id", self.result_id or expected)
        check_id(self.result_id, expected, "result_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "request_ref": ref_payload(self.request_ref),
            "readout_ref": ref_payload(self.readout_ref),
            "feature_batch_ref": ref_payload(self.feature_batch_ref),
            "critic_contract_ref": ref_payload(self.critic_contract_ref),
            "feature_spec_ref": ref_payload(self.feature_spec_ref),
            "items": [item.to_dict() for item in self.items],
            "ordering_contract": self.ordering_contract,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["result_id"] = self.result_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticScoreResultDTO":
        return cls(
            request_ref=ref_from_dict(data["request_ref"]),
            readout_ref=ref_from_dict(data["readout_ref"]),
            feature_batch_ref=ref_from_dict(data["feature_batch_ref"]),
            critic_contract_ref=ref_from_dict(data["critic_contract_ref"]),
            feature_spec_ref=ref_from_dict(data["feature_spec_ref"]),
            items=tuple(CriticItemScoreDTO.from_dict(item) for item in data["items"]),
            ordering_contract=str(data["ordering_contract"]),
            metadata=data.get("metadata") or {},
            result_id=str(data.get("result_id") or ""),
            spec_version=str(data.get("spec_version", CRITIC_SPEC_VERSION)),
        )


def compute_critic_score_result_id(result: CriticScoreResultDTO) -> str:
    return digest_payload("zeromodel.critic.score_result.v1", result.identity_payload())


@dataclass(frozen=True, slots=True)
class CriticScoreReceiptDTO:
    request_ref: ArtifactRef
    result_ref: ArtifactRef
    readout_ref: ArtifactRef
    feature_batch_ref: ArtifactRef
    required_checks: Tuple[str, ...]
    result_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    receipt_id: str = ""
    spec_version: str = CRITIC_SPEC_VERSION

    REQUIRED = (
        "request_resolved",
        "readout_aggregate_validated",
        "feature_batch_validated",
        "deterministic_scores_recomputed",
        "result_ref_matches_recomputed_result",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_checks", tuple(self.required_checks))
        if self.required_checks != self.REQUIRED:
            raise VPMValidationError(
                "critic receipt must include exactly the required checks"
            )
        object.__setattr__(
            self, "result_id", require_nonempty(self.result_id, "result_id")
        )
        object.__setattr__(self, "metadata", freeze_json(self.metadata))
        expected = compute_critic_score_receipt_id(self)
        object.__setattr__(self, "receipt_id", self.receipt_id or expected)
        check_id(self.receipt_id, expected, "receipt_id")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "request_ref": ref_payload(self.request_ref),
            "result_ref": ref_payload(self.result_ref),
            "readout_ref": ref_payload(self.readout_ref),
            "feature_batch_ref": ref_payload(self.feature_batch_ref),
            "required_checks": list(self.required_checks),
            "result_id": self.result_id,
            "metadata": thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticScoreReceiptDTO":
        return cls(
            request_ref=ref_from_dict(data["request_ref"]),
            result_ref=ref_from_dict(data["result_ref"]),
            readout_ref=ref_from_dict(data["readout_ref"]),
            feature_batch_ref=ref_from_dict(data["feature_batch_ref"]),
            required_checks=tuple(str(v) for v in data["required_checks"]),
            result_id=str(data["result_id"]),
            metadata=data.get("metadata") or {},
            receipt_id=str(data.get("receipt_id") or ""),
            spec_version=str(data.get("spec_version", CRITIC_SPEC_VERSION)),
        )


def compute_critic_score_receipt_id(receipt: CriticScoreReceiptDTO) -> str:
    return digest_payload(
        "zeromodel.critic.score_receipt.v1", receipt.identity_payload()
    )


def canonical_dto_bytes(dto: Any) -> bytes:
    return canonical_json_bytes(dto.to_dict())
