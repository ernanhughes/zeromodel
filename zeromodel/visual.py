"""Observation-addressed policy lookup for bounded visual worlds.

The visual index is separate from the policy artifact:

- index rows contain deterministic visual feature vectors;
- policy rows contain action/evidence values;
- both artifacts share the same stable row ids;
- the index declares an ``addresses`` parent relation to the policy.

This module does not perform open-world perception or learned generalization.
It matches an observation against a complete, calibrated visual codebook and
either returns the addressed policy decision or a first-class rejection trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import (
    LayoutRecipe,
    ScoreTable,
    VPMArtifact,
    VPMValidationError,
    build_vpm,
)
from .policy_lookup import VPMPolicyLookup

VISUAL_FEATURE_VERSION = "zeromodel-visual-feature/v1"
VISUAL_INDEX_VERSION = "zeromodel-visual-index/v1"
VISUAL_READER_VERSION = "zeromodel-visual-sign-reader/v1"
DISTANCE_METRIC = "euclidean"


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("visual metadata must be JSON-serializable") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class VisualFeatureSpec:
    """Identity-bearing deterministic integer feature contract.

    Frames must have the declared input dimensions and either one grayscale
    channel or three/four uint8-compatible channels. RGB conversion, box
    pooling, and quantization use integer arithmetic.
    """

    input_height: int
    input_width: int
    target_height: int
    target_width: int
    quantization_levels: int = 16
    version: str = VISUAL_FEATURE_VERSION
    grayscale: str = "bt601-integer"
    pooling: str = "box-integer"
    quantization: str = "uniform-uint8"

    def validate(self) -> None:
        if self.version != VISUAL_FEATURE_VERSION:
            raise VPMValidationError("Unsupported visual feature version: %r" % self.version)
        if self.grayscale != "bt601-integer":
            raise VPMValidationError("Unsupported grayscale contract: %r" % self.grayscale)
        if self.pooling != "box-integer":
            raise VPMValidationError("Unsupported pooling contract: %r" % self.pooling)
        if self.quantization != "uniform-uint8":
            raise VPMValidationError(
                "Unsupported visual quantization contract: %r" % self.quantization
            )
        for name, value in (
            ("input_height", self.input_height),
            ("input_width", self.input_width),
            ("target_height", self.target_height),
            ("target_width", self.target_width),
        ):
            if int(value) <= 0:
                raise VPMValidationError("%s must be positive" % name)
        if self.input_height % self.target_height != 0:
            raise VPMValidationError("target_height must divide input_height exactly")
        if self.input_width % self.target_width != 0:
            raise VPMValidationError("target_width must divide input_width exactly")
        if not (2 <= int(self.quantization_levels) <= 256):
            raise VPMValidationError("quantization_levels must be in [2, 256]")

    @property
    def feature_count(self) -> int:
        return int(self.target_height) * int(self.target_width)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "input_height": int(self.input_height),
            "input_width": int(self.input_width),
            "target_height": int(self.target_height),
            "target_width": int(self.target_width),
            "quantization_levels": int(self.quantization_levels),
            "grayscale": self.grayscale,
            "pooling": self.pooling,
            "quantization": self.quantization,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisualFeatureSpec":
        spec = cls(
            input_height=int(data["input_height"]),
            input_width=int(data["input_width"]),
            target_height=int(data["target_height"]),
            target_width=int(data["target_width"]),
            quantization_levels=int(data.get("quantization_levels", 16)),
            version=str(data.get("version", VISUAL_FEATURE_VERSION)),
            grayscale=str(data.get("grayscale", "bt601-integer")),
            pooling=str(data.get("pooling", "box-integer")),
            quantization=str(data.get("quantization", "uniform-uint8")),
        )
        spec.validate()
        return spec

    @property
    def digest(self) -> str:
        self.validate()
        return _sha256(_canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class VisualIndexCalibration:
    """Separation audit and compiled acceptance contract."""

    state_count: int
    feature_count: int
    min_between_distance: float
    closest_pair_row_ids: Tuple[str, str]
    threshold_fraction: float
    acceptance_threshold: float
    margin_fraction: float
    required_margin: float
    distance_metric: str = DISTANCE_METRIC

    def validate(self) -> None:
        if self.distance_metric != DISTANCE_METRIC:
            raise VPMValidationError(
                "Unsupported visual distance metric: %r" % self.distance_metric
            )
        if int(self.state_count) < 2:
            raise VPMValidationError("visual index requires at least two states")
        if int(self.feature_count) < 1:
            raise VPMValidationError("visual index requires at least one feature")
        if len(self.closest_pair_row_ids) != 2:
            raise VPMValidationError("closest_pair_row_ids must contain two row ids")
        if self.closest_pair_row_ids[0] == self.closest_pair_row_ids[1]:
            raise VPMValidationError("closest pair must contain distinct row ids")
        for name, value in (
            ("min_between_distance", self.min_between_distance),
            ("threshold_fraction", self.threshold_fraction),
            ("acceptance_threshold", self.acceptance_threshold),
            ("margin_fraction", self.margin_fraction),
            ("required_margin", self.required_margin),
        ):
            if not np.isfinite(float(value)):
                raise VPMValidationError("%s must be finite" % name)
        if float(self.min_between_distance) <= 0.0:
            raise VPMValidationError(
                "visual states are not separable: min_between_distance must be positive"
            )
        if not (0.0 < float(self.threshold_fraction) < 0.5):
            raise VPMValidationError("threshold_fraction must be between 0 and 0.5")
        if not (0.0 < float(self.margin_fraction) <= 1.0):
            raise VPMValidationError("margin_fraction must be in (0, 1]")
        expected_threshold = float(self.min_between_distance) * float(
            self.threshold_fraction
        )
        expected_margin = float(self.min_between_distance) * float(self.margin_fraction)
        if not np.isclose(float(self.acceptance_threshold), expected_threshold):
            raise VPMValidationError(
                "acceptance_threshold does not match the declared separation audit"
            )
        if not np.isclose(float(self.required_margin), expected_margin):
            raise VPMValidationError(
                "required_margin does not match the declared separation audit"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distance_metric": self.distance_metric,
            "state_count": int(self.state_count),
            "feature_count": int(self.feature_count),
            "min_between_distance": float(self.min_between_distance),
            "closest_pair_row_ids": list(self.closest_pair_row_ids),
            "threshold_fraction": float(self.threshold_fraction),
            "acceptance_threshold": float(self.acceptance_threshold),
            "margin_fraction": float(self.margin_fraction),
            "required_margin": float(self.required_margin),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisualIndexCalibration":
        calibration = cls(
            state_count=int(data["state_count"]),
            feature_count=int(data["feature_count"]),
            min_between_distance=float(data["min_between_distance"]),
            closest_pair_row_ids=tuple(
                str(value) for value in data["closest_pair_row_ids"]
            ),
            threshold_fraction=float(data["threshold_fraction"]),
            acceptance_threshold=float(data["acceptance_threshold"]),
            margin_fraction=float(data["margin_fraction"]),
            required_margin=float(data["required_margin"]),
            distance_metric=str(data.get("distance_metric", DISTANCE_METRIC)),
        )
        calibration.validate()
        return calibration

    @property
    def digest(self) -> str:
        self.validate()
        return _sha256(_canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class VisualIndexBuild:
    artifact: VPMArtifact
    feature_spec: VisualFeatureSpec
    calibration: VisualIndexCalibration


@dataclass(frozen=True)
class VisualDecision:
    """Accepted policy decision or evidence-bearing visual rejection."""

    accepted: bool
    reason: str
    input_digest: str
    feature_digest: str
    reader_version: str
    visual_index_artifact_id: str
    policy_artifact_id: str
    feature_spec_digest: str
    calibration_digest: str
    nearest_row_id: str
    nearest_distance: float
    second_nearest_row_id: str
    second_nearest_distance: float
    distance_margin: float
    acceptance_threshold: float
    required_margin: float
    exact_feature_match: bool
    matched_row_id: Optional[str] = None
    action: Optional[str] = None
    value: Optional[float] = None
    source_row_index: Optional[int] = None
    source_metric_index: Optional[int] = None
    view_row: Optional[int] = None
    view_column: Optional[int] = None
    candidates: Mapping[str, float] = field(default_factory=dict)
    evidence: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "input_digest": self.input_digest,
            "feature_digest": self.feature_digest,
            "reader_version": self.reader_version,
            "visual_index_artifact_id": self.visual_index_artifact_id,
            "policy_artifact_id": self.policy_artifact_id,
            "feature_spec_digest": self.feature_spec_digest,
            "calibration_digest": self.calibration_digest,
            "nearest_row_id": self.nearest_row_id,
            "nearest_distance": float(self.nearest_distance),
            "second_nearest_row_id": self.second_nearest_row_id,
            "second_nearest_distance": float(self.second_nearest_distance),
            "distance_margin": float(self.distance_margin),
            "acceptance_threshold": float(self.acceptance_threshold),
            "required_margin": float(self.required_margin),
            "exact_feature_match": bool(self.exact_feature_match),
            "matched_row_id": self.matched_row_id,
            "action": self.action,
            "value": None if self.value is None else float(self.value),
            "source_row_index": self.source_row_index,
            "source_metric_index": self.source_metric_index,
            "view_row": self.view_row,
            "view_column": self.view_column,
            "candidates": {str(k): float(v) for k, v in self.candidates.items()},
            "evidence": {str(k): float(v) for k, v in self.evidence.items()},
        }


def _grayscale_uint8(frame: Any, spec: VisualFeatureSpec) -> np.ndarray:
    spec.validate()
    array = np.asarray(frame)
    if array.dtype.kind not in {"u", "i"}:
        raise VPMValidationError(
            "visual frames must contain integer samples in the [0, 255] range"
        )
    if array.size == 0:
        raise VPMValidationError("visual frame cannot be empty")
    if int(array.min()) < 0 or int(array.max()) > 255:
        raise VPMValidationError("visual frame samples must be in the [0, 255] range")
    if array.ndim == 2:
        gray = np.asarray(array, dtype=np.uint8)
    elif array.ndim == 3 and array.shape[2] in {3, 4}:
        rgb = np.asarray(array[:, :, :3], dtype=np.uint16)
        gray = (
            77 * rgb[:, :, 0]
            + 150 * rgb[:, :, 1]
            + 29 * rgb[:, :, 2]
            + 128
        ) // 256
        gray = gray.astype(np.uint8)
    else:
        raise VPMValidationError(
            "visual frame must be HxW grayscale or HxWx3/4 RGB(A)"
        )
    if gray.shape != (spec.input_height, spec.input_width):
        raise VPMValidationError(
            "visual frame shape must be (%d, %d); got %s"
            % (spec.input_height, spec.input_width, gray.shape)
        )
    gray = np.ascontiguousarray(gray)
    gray.flags.writeable = False
    return gray


def extract_visual_features(frame: Any, spec: VisualFeatureSpec) -> np.ndarray:
    """Extract the canonical quantized feature vector for one frame."""

    gray = _grayscale_uint8(frame, spec)
    block_height = spec.input_height // spec.target_height
    block_width = spec.input_width // spec.target_width
    area = block_height * block_width
    pooled_sum = gray.astype(np.uint32).reshape(
        spec.target_height,
        block_height,
        spec.target_width,
        block_width,
    ).sum(axis=(1, 3))
    pooled = (pooled_sum + area // 2) // area
    levels = int(spec.quantization_levels)
    quantized = (pooled * (levels - 1) + 127) // 255
    features = np.ascontiguousarray(quantized.astype(np.uint8).reshape(-1))
    features.flags.writeable = False
    return features


def visual_input_digest(frame: Any, spec: VisualFeatureSpec) -> str:
    """Digest the canonical grayscale observation under the feature contract."""

    gray = _grayscale_uint8(frame, spec)
    payload = b"".join(
        (
            b"zeromodel.visual-input.v1\0",
            bytes.fromhex(spec.digest),
            int(spec.input_height).to_bytes(4, "big"),
            int(spec.input_width).to_bytes(4, "big"),
            gray.tobytes(order="C"),
        )
    )
    return "sha256:" + _sha256(payload)


def visual_feature_digest(features: np.ndarray, spec: VisualFeatureSpec) -> str:
    vector = np.asarray(features, dtype=np.uint8).reshape(-1)
    if vector.size != spec.feature_count:
        raise VPMValidationError("feature vector does not match the feature spec")
    payload = b"".join(
        (
            b"zeromodel.visual-feature-vector.v1\0",
            bytes.fromhex(spec.digest),
            vector.tobytes(order="C"),
        )
    )
    return "sha256:" + _sha256(payload)


def _pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    squared = np.sum(values**2, axis=1)
    distances_squared = squared[:, None] + squared[None, :] - 2.0 * (
        values @ values.T
    )
    np.maximum(distances_squared, 0.0, out=distances_squared)
    return np.sqrt(distances_squared)


def build_visual_index(
    policy_artifact: VPMArtifact,
    frames_by_row_id: Mapping[str, Any],
    feature_spec: VisualFeatureSpec,
    *,
    threshold_fraction: float = 0.25,
    margin_fraction: float = 0.25,
    name: str = "visual-index-source-order",
) -> VisualIndexBuild:
    """Compile a complete calibrated visual codebook for a bounded policy."""

    policy_artifact.validate()
    feature_spec.validate()
    policy_row_ids = tuple(policy_artifact.source.row_ids)
    frame_row_ids = {str(row_id) for row_id in frames_by_row_id}
    missing = sorted(set(policy_row_ids) - frame_row_ids)
    unknown = sorted(frame_row_ids - set(policy_row_ids))
    if missing or unknown:
        raise VPMValidationError(
            "visual frames must cover policy rows exactly "
            "(missing=%s, unknown=%s)" % (missing, unknown)
        )

    feature_rows = [
        extract_visual_features(frames_by_row_id[row_id], feature_spec)
        for row_id in policy_row_ids
    ]
    matrix = np.asarray(feature_rows, dtype=np.float64)
    if matrix.shape != (len(policy_row_ids), feature_spec.feature_count):
        raise VPMValidationError("visual feature matrix shape mismatch")

    distances = _pairwise_distances(matrix)
    np.fill_diagonal(distances, np.inf)
    closest_flat = int(np.argmin(distances))
    left_index, right_index = np.unravel_index(closest_flat, distances.shape)
    min_between = float(distances[left_index, right_index])
    calibration = VisualIndexCalibration(
        state_count=len(policy_row_ids),
        feature_count=feature_spec.feature_count,
        min_between_distance=min_between,
        closest_pair_row_ids=(
            policy_row_ids[int(left_index)],
            policy_row_ids[int(right_index)],
        ),
        threshold_fraction=float(threshold_fraction),
        acceptance_threshold=min_between * float(threshold_fraction),
        margin_fraction=float(margin_fraction),
        required_margin=min_between * float(margin_fraction),
    )
    calibration.validate()

    metric_ids = tuple(
        "visual:%04d" % index for index in range(feature_spec.feature_count)
    )
    table = ScoreTable(
        values=matrix,
        row_ids=policy_row_ids,
        metric_ids=metric_ids,
        metadata={
            "kind": "visual_index",
            "visual_index_version": VISUAL_INDEX_VERSION,
            "addresses_policy_artifact_id": policy_artifact.artifact_id,
            "feature_spec": feature_spec.to_dict(),
            "feature_spec_digest": feature_spec.digest,
            "calibration": calibration.to_dict(),
            "calibration_digest": calibration.digest,
        },
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": str(name),
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    artifact = build_vpm(
        table,
        recipe,
        provenance={
            "kind": "visual_index",
            "feature_spec_digest": feature_spec.digest,
            "calibration_digest": calibration.digest,
            "parents": [
                {
                    "artifact_id": policy_artifact.artifact_id,
                    "relation": "addresses",
                }
            ],
        },
    )
    return VisualIndexBuild(
        artifact=artifact,
        feature_spec=feature_spec,
        calibration=calibration,
    )


class VisualSignReader:
    """Address a finite policy from a calibrated visual observation codebook."""

    def __init__(
        self,
        visual_index_artifact: VPMArtifact,
        policy_artifact: VPMArtifact,
        *,
        action_metric_ids: Optional[Sequence[str]] = None,
        evidence_metric_ids: Optional[Sequence[str]] = None,
        value_source: str = "raw",
        evidence_value_source: str = "raw",
        tie_break: str = "metric_order",
    ) -> None:
        visual_index_artifact.validate()
        policy_artifact.validate()
        metadata = visual_index_artifact.source.metadata
        if metadata.get("kind") != "visual_index":
            raise VPMValidationError("VisualSignReader requires a visual_index artifact")
        if metadata.get("visual_index_version") != VISUAL_INDEX_VERSION:
            raise VPMValidationError("Unsupported visual index version")
        addressed_policy = str(metadata.get("addresses_policy_artifact_id") or "")
        if addressed_policy != policy_artifact.artifact_id:
            raise VPMValidationError(
                "visual index addresses policy %s, not %s"
                % (addressed_policy, policy_artifact.artifact_id)
            )
        if tuple(visual_index_artifact.source.row_ids) != tuple(
            policy_artifact.source.row_ids
        ):
            raise VPMValidationError(
                "visual index and policy must have identical ordered row ids"
            )

        parents = tuple(visual_index_artifact.provenance.get("parents") or ())
        if not any(
            isinstance(parent, Mapping)
            and parent.get("relation") == "addresses"
            and parent.get("artifact_id") == policy_artifact.artifact_id
            for parent in parents
        ):
            raise VPMValidationError(
                "visual index provenance must address the supplied policy artifact"
            )

        feature_spec_data = metadata.get("feature_spec")
        calibration_data = metadata.get("calibration")
        if not isinstance(feature_spec_data, Mapping):
            raise VPMValidationError("visual index metadata requires feature_spec")
        if not isinstance(calibration_data, Mapping):
            raise VPMValidationError("visual index metadata requires calibration")
        feature_spec = VisualFeatureSpec.from_dict(feature_spec_data)
        calibration = VisualIndexCalibration.from_dict(calibration_data)
        if str(metadata.get("feature_spec_digest")) != feature_spec.digest:
            raise VPMValidationError("visual feature spec digest mismatch")
        if str(metadata.get("calibration_digest")) != calibration.digest:
            raise VPMValidationError("visual calibration digest mismatch")
        if calibration.state_count != len(visual_index_artifact.source.row_ids):
            raise VPMValidationError("visual calibration state count mismatch")
        if calibration.feature_count != len(visual_index_artifact.source.metric_ids):
            raise VPMValidationError("visual calibration feature count mismatch")

        raw_matrix = np.asarray(
            visual_index_artifact.source.values, dtype=np.float64
        )
        rounded = np.rint(raw_matrix)
        if not np.array_equal(raw_matrix, rounded):
            raise VPMValidationError("visual index features must be integral")
        if raw_matrix.min() < 0 or raw_matrix.max() >= feature_spec.quantization_levels:
            raise VPMValidationError("visual index features exceed quantization range")
        matrix = np.ascontiguousarray(raw_matrix, dtype=np.float64)
        matrix.flags.writeable = False

        self.visual_index_artifact = visual_index_artifact
        self.policy_artifact = policy_artifact
        self.feature_spec = feature_spec
        self.calibration = calibration
        self._matrix = matrix
        self._row_ids = tuple(visual_index_artifact.source.row_ids)
        self._row_norms = np.sum(matrix**2, axis=1)
        self._exact_rows = {
            np.asarray(matrix[index], dtype=np.uint8).tobytes(order="C"): index
            for index in range(matrix.shape[0])
        }
        if len(self._exact_rows) != len(self._row_ids):
            raise VPMValidationError(
                "visual index contains duplicate feature vectors"
            )
        pairwise = _pairwise_distances(matrix)
        np.fill_diagonal(pairwise, np.inf)
        self._nearest_other_distance = pairwise.min(axis=1)
        self._nearest_other_index = pairwise.argmin(axis=1)
        self.policy_lookup = VPMPolicyLookup(
            policy_artifact,
            action_metric_ids=action_metric_ids,
            evidence_metric_ids=evidence_metric_ids,
            value_source=value_source,
            evidence_value_source=evidence_value_source,
            tie_break=tie_break,
        )

    def _candidate_distances(
        self, features: np.ndarray
    ) -> Tuple[int, int, float, float, bool]:
        key = features.tobytes(order="C")
        exact_index = self._exact_rows.get(key)
        if exact_index is not None:
            second_index = int(self._nearest_other_index[exact_index])
            return (
                int(exact_index),
                second_index,
                0.0,
                float(self._nearest_other_distance[exact_index]),
                True,
            )

        query = np.asarray(features, dtype=np.float64)
        distances_squared = self._row_norms + float(np.sum(query**2)) - 2.0 * (
            self._matrix @ query
        )
        np.maximum(distances_squared, 0.0, out=distances_squared)
        distances = np.sqrt(distances_squared)
        ranking = sorted(
            range(len(self._row_ids)),
            key=lambda index: (float(distances[index]), self._row_ids[index]),
        )
        first, second = int(ranking[0]), int(ranking[1])
        return first, second, float(distances[first]), float(distances[second]), False

    def read(self, frame: Any) -> VisualDecision:
        features = extract_visual_features(frame, self.feature_spec)
        input_digest = visual_input_digest(frame, self.feature_spec)
        feature_digest = visual_feature_digest(features, self.feature_spec)
        first, second, nearest, second_nearest, exact = self._candidate_distances(
            features
        )
        margin = second_nearest - nearest
        nearest_row_id = self._row_ids[first]
        second_row_id = self._row_ids[second]

        distance_ok = nearest <= self.calibration.acceptance_threshold + 1e-12
        margin_ok = margin + 1e-12 >= self.calibration.required_margin
        if not distance_ok or not margin_ok:
            reason = (
                "visual_distance_above_threshold"
                if not distance_ok
                else "ambiguous_visual_address"
            )
            return VisualDecision(
                accepted=False,
                reason=reason,
                input_digest=input_digest,
                feature_digest=feature_digest,
                reader_version=VISUAL_READER_VERSION,
                visual_index_artifact_id=self.visual_index_artifact.artifact_id,
                policy_artifact_id=self.policy_artifact.artifact_id,
                feature_spec_digest=self.feature_spec.digest,
                calibration_digest=self.calibration.digest,
                nearest_row_id=nearest_row_id,
                nearest_distance=nearest,
                second_nearest_row_id=second_row_id,
                second_nearest_distance=second_nearest,
                distance_margin=margin,
                acceptance_threshold=self.calibration.acceptance_threshold,
                required_margin=self.calibration.required_margin,
                exact_feature_match=exact,
            )

        policy_decision = self.policy_lookup.read(nearest_row_id)
        return VisualDecision(
            accepted=True,
            reason="accepted",
            input_digest=input_digest,
            feature_digest=feature_digest,
            reader_version=VISUAL_READER_VERSION,
            visual_index_artifact_id=self.visual_index_artifact.artifact_id,
            policy_artifact_id=self.policy_artifact.artifact_id,
            feature_spec_digest=self.feature_spec.digest,
            calibration_digest=self.calibration.digest,
            nearest_row_id=nearest_row_id,
            nearest_distance=nearest,
            second_nearest_row_id=second_row_id,
            second_nearest_distance=second_nearest,
            distance_margin=margin,
            acceptance_threshold=self.calibration.acceptance_threshold,
            required_margin=self.calibration.required_margin,
            exact_feature_match=exact,
            matched_row_id=nearest_row_id,
            action=policy_decision.action,
            value=policy_decision.value,
            source_row_index=policy_decision.source_row_index,
            source_metric_index=policy_decision.source_metric_index,
            view_row=policy_decision.view_row,
            view_column=policy_decision.view_column,
            candidates=dict(policy_decision.candidates),
            evidence=dict(policy_decision.evidence),
        )
