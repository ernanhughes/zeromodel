from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.typing as npt

from zeromodel.critic.dto import CriticFeatureContributionDTO, CriticFeatureSpecDTO
from zeromodel.critic.errors import (
    CriticFeatureSchemaMismatchError,
    CriticValidationError,
)

FloatArray = npt.NDArray[np.float64]


def stable_sigmoid(value: npt.ArrayLike) -> FloatArray:
    x = np.asarray(value, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _matrix(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise CriticValidationError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise CriticValidationError(f"{name} must contain only finite values")
    return array


def _vector(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise CriticValidationError(f"{name} must be a one-dimensional vector")
    if not np.isfinite(array).all():
        raise CriticValidationError(f"{name} must contain only finite values")
    return array


def features_from_mapping(
    values: Mapping[str, float | None], spec: CriticFeatureSpecDTO
) -> np.ndarray:
    row = []
    for feature in spec.features:
        raw = values.get(feature.feature_id)
        if raw is None:
            if (
                feature.missing_policy == "constant"
                and feature.missing_value is not None
            ):
                raw = feature.missing_value
            else:
                raise CriticFeatureSchemaMismatchError(
                    f"missing required feature: {feature.feature_id}"
                )
        row.append(float(raw))
    array = np.asarray(row, dtype=np.float64)
    if not np.isfinite(array).all():
        raise CriticValidationError("feature values must be finite")
    return array


def _fit_logistic_irls(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2_penalty: float,
    max_iterations: int,
    tolerance: float,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, float]:
    n_rows, n_cols = x.shape
    design = np.column_stack([np.ones(n_rows, dtype=np.float64), x])
    beta = np.zeros(n_cols + 1, dtype=np.float64)
    regularization = np.diag(np.r_[0.0, np.full(n_cols, float(l2_penalty))])
    for _ in range(max_iterations):
        logits = design @ beta
        p = stable_sigmoid(logits)
        w = np.maximum(p * (1.0 - p) * sample_weight, 1e-12)
        gradient = design.T @ ((p - y) * sample_weight) + regularization @ beta
        hessian = (design.T * w) @ design + regularization
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta_next = beta - step
        if float(np.max(np.abs(step))) <= tolerance:
            beta = beta_next
            break
        beta = beta_next
    return np.asarray(beta[1:], dtype=np.float64), float(beta[0])


@dataclass(frozen=True, slots=True)
class CompiledCriticReadout:
    feature_ids: tuple[str, ...]
    directionality: tuple[int, ...]
    center: FloatArray
    scale: FloatArray
    coefficients: FloatArray
    intercept: float
    contract_id: str
    feature_spec_id: str
    calibration: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        width = len(self.feature_ids)
        if width == 0:
            raise CriticValidationError("readout requires features")
        if len(self.directionality) != width:
            raise CriticValidationError("directionality length differs from features")
        for direction in self.directionality:
            if int(direction) not in {-1, 1}:
                raise CriticValidationError("directionality must be +1 or -1")
        center = _vector("center", self.center)
        scale = _vector("scale", self.scale)
        coefficients = _vector("coefficients", self.coefficients)
        if center.size != width or scale.size != width or coefficients.size != width:
            raise CriticValidationError(
                "readout vector dimensions do not match features"
            )
        if np.any(scale <= 0.0):
            raise CriticValidationError("scale must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", float(self.intercept))

    @classmethod
    def fit(
        cls,
        features: npt.ArrayLike,
        labels: npt.ArrayLike,
        *,
        feature_spec: CriticFeatureSpecDTO,
        contract_id: str,
        l2_penalty: float,
        max_iterations: int,
        tolerance: float,
        class_weighting: str,
    ) -> "CompiledCriticReadout":
        raw = _matrix("features", features)
        y = _vector("labels", labels)
        if raw.shape[0] != y.size:
            raise CriticValidationError(
                "training features and labels need the same rows"
            )
        if raw.shape[1] != len(feature_spec.features):
            raise CriticFeatureSchemaMismatchError("feature width does not match spec")
        if not np.isin(y, [0.0, 1.0]).all():
            raise CriticValidationError("labels must be binary 0/1")
        if np.unique(y).size != 2:
            raise CriticValidationError("training labels must contain both classes")
        direction = np.asarray(feature_spec.directionality, dtype=np.float64)
        directed = raw * direction
        center = directed.mean(axis=0)
        scale = directed.std(axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        standardized = (directed - center) / scale
        weights = np.ones(y.size, dtype=np.float64)
        if class_weighting == "balanced":
            positives = float(np.sum(y == 1.0))
            negatives = float(np.sum(y == 0.0))
            weights[y == 1.0] = y.size / (2.0 * positives)
            weights[y == 0.0] = y.size / (2.0 * negatives)
        coefficients, intercept = _fit_logistic_irls(
            standardized,
            y,
            l2_penalty=l2_penalty,
            max_iterations=max_iterations,
            tolerance=tolerance,
            sample_weight=weights,
        )
        return cls(
            feature_ids=feature_spec.feature_ids,
            directionality=feature_spec.directionality,
            center=center,
            scale=scale,
            coefficients=coefficients,
            intercept=intercept,
            contract_id=contract_id,
            feature_spec_id=feature_spec.feature_spec_id,
        )

    def _standardize_many(
        self, values: npt.ArrayLike, *, feature_spec_id: str
    ) -> FloatArray:
        if feature_spec_id != self.feature_spec_id:
            raise CriticFeatureSchemaMismatchError("feature spec identity mismatch")
        matrix = _matrix("features", values)
        if matrix.shape[1] != len(self.feature_ids):
            raise CriticFeatureSchemaMismatchError(
                "feature width does not match readout"
            )
        direction = np.asarray(self.directionality, dtype=np.float64)
        return (matrix * direction - self.center) / self.scale

    def logit_many(self, values: npt.ArrayLike, *, feature_spec_id: str) -> FloatArray:
        z = self._standardize_many(values, feature_spec_id=feature_spec_id)
        return np.asarray(z @ self.coefficients + self.intercept, dtype=np.float64)

    def logit_one(self, values: npt.ArrayLike, *, feature_spec_id: str) -> float:
        vector = _vector("features", values)
        return float(
            self.logit_many(vector[None, :], feature_spec_id=feature_spec_id)[0]
        )

    def score_many(self, values: npt.ArrayLike, *, feature_spec_id: str) -> FloatArray:
        return stable_sigmoid(self.logit_many(values, feature_spec_id=feature_spec_id))

    def score_one(self, values: npt.ArrayLike, *, feature_spec_id: str) -> float:
        return float(
            stable_sigmoid(
                np.asarray([self.logit_one(values, feature_spec_id=feature_spec_id)])
            )[0]
        )

    def contributions_one(
        self, values: npt.ArrayLike, *, feature_spec_id: str
    ) -> tuple[CriticFeatureContributionDTO, ...]:
        raw = _vector("features", values)
        z = self._standardize_many(raw[None, :], feature_spec_id=feature_spec_id)[0]
        direction = np.asarray(self.directionality, dtype=np.float64)
        directed = raw * direction
        contributions = z * self.coefficients
        return tuple(
            CriticFeatureContributionDTO(
                feature_id=self.feature_ids[index],
                raw_value=float(raw[index]),
                directed_value=float(directed[index]),
                standardized_value=float(z[index]),
                coefficient=float(self.coefficients[index]),
                contribution=float(contributions[index]),
            )
            for index in range(len(self.feature_ids))
        )
