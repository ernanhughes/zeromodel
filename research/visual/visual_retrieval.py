"""Calibrated vector-address baselines for the Phase 1 visual benchmark.

This module separates representation extraction from address matching. It can run
with normalized pixels, precomputed frozen embeddings, or any future encoder that
implements ``FrozenVisualEncoder``. Policy selection remains in
``VPMPolicyLookup`` after a row is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.observation.visual_address import (
    ImageObservation,
    VisualAddressContract,
    VisualAddressDecision,
)
from zeromodel.observation.visual_address_manifest import (
    PrototypeBinding,
    VisualAddressManifest,
)
from research.visual.visual_encoder import EncoderManifest, FrozenVisualEncoder


VECTOR_ADDRESS_READER_VERSION = "zeromodel-vector-address-reader/v2"
VECTOR_CALIBRATION_VERSION = "zeromodel-vector-calibration/v2"
LEGACY_VECTOR_CALIBRATION_VERSION = "zeromodel-vector-calibration/v1"
LINEAR_PROBE_VERSION = "zeromodel-linear-probe/v2"


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, np.generic):
        raise VPMValidationError("vector-address JSON must use plain scalar types")
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError(
            "vector-address values must be JSON-serializable"
        ) from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _as_matrix(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise VPMValidationError("%s must be a non-empty 2D matrix" % name)
    if not np.isfinite(matrix).all():
        raise VPMValidationError("%s must contain only finite values" % name)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def l2_normalize_rows(values: Any, *, allow_zero: bool = False) -> np.ndarray:
    matrix = _as_matrix(values, name="vector matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if not allow_zero and np.any(norms <= 0.0):
        raise VPMValidationError("vector rows must have non-zero L2 norm")
    result = np.zeros_like(matrix, dtype=np.float32)
    nonzero = norms > 0.0
    result[nonzero] = matrix[nonzero] / norms[nonzero, None]
    result.flags.writeable = False
    return result


def _lower_quantile(values: Sequence[float], quantile: float) -> float:
    if not np.isfinite(quantile) or not (0.0 <= quantile <= 1.0):
        raise VPMValidationError("calibration quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise VPMValidationError("calibration quantile requires at least one value")
    index = int(np.floor(quantile * float(len(ordered) - 1)))
    return ordered[index]


def _representation_digest(
    vector: np.ndarray,
    representation_spec_digest: str,
    execution_scope_digest: Optional[str] = None,
) -> str:
    """Identify one exact float32 representation under a declared runtime scope.

    This is deliberately not a claim that the same logical observation will
    produce byte-identical vectors across devices, framework versions, or
    hardware backends. The execution scope is normally the encoder manifest ID,
    which includes the declared device and framework versions.
    """

    item = np.ascontiguousarray(vector, dtype=">f4")
    scope = str(execution_scope_digest or representation_spec_digest)
    payload = (
        b"zeromodel.vector-representation.v2\0"
        + str(representation_spec_digest).encode("utf-8")
        + b"\0"
        + scope.encode("utf-8")
        + b"\0"
        + item.tobytes(order="C")
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class VectorCalibration:
    """Per-row similarity and conflicting-action margin thresholds.

    ``conflict_contract_complete`` distinguishes new calibrations, which record
    exactly which rows have a conflicting-action candidate, from v1 artifacts,
    where an empty set meant "unknown" rather than "none".
    """

    acceptance_thresholds: Mapping[str, float]
    ambiguity_margins: Mapping[str, float]
    calibration_quantile: float
    calibration_count: int
    calibration_counts: Mapping[str, int] = field(default_factory=dict)
    conflicting_action_rows: Tuple[str, ...] = ()
    conflict_contract_complete: bool = False
    method: str = "empirical-lower-quantile"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VECTOR_CALIBRATION_VERSION

    def __post_init__(self) -> None:
        thresholds = {
            str(key): float(value) for key, value in self.acceptance_thresholds.items()
        }
        margins = {
            str(key): float(value) for key, value in self.ambiguity_margins.items()
        }
        counts = {
            str(key): int(value) for key, value in self.calibration_counts.items()
        }
        conflict_rows = tuple(
            sorted(set(str(value) for value in self.conflicting_action_rows))
        )
        object.__setattr__(
            self,
            "acceptance_thresholds",
            MappingProxyType(thresholds),
        )
        object.__setattr__(
            self,
            "ambiguity_margins",
            MappingProxyType(margins),
        )
        object.__setattr__(
            self,
            "calibration_counts",
            MappingProxyType(counts),
        )
        object.__setattr__(
            self,
            "conflicting_action_rows",
            conflict_rows,
        )
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.version not in {
            VECTOR_CALIBRATION_VERSION,
            LEGACY_VECTOR_CALIBRATION_VERSION,
        }:
            raise VPMValidationError("unsupported vector calibration version")
        if not thresholds or set(thresholds) != set(margins):
            raise VPMValidationError(
                "calibration thresholds and margins require identical rows"
            )
        if any(
            not np.isfinite(value)
            for value in tuple(thresholds.values()) + tuple(margins.values())
        ):
            raise VPMValidationError(
                "calibration thresholds and margins must be finite"
            )
        if not np.isfinite(self.calibration_quantile) or not (
            0.0 <= self.calibration_quantile <= 1.0
        ):
            raise VPMValidationError("calibration_quantile must be in [0, 1]")
        if self.calibration_count <= 0:
            raise VPMValidationError("calibration_count must be positive")
        if counts:
            if set(counts) != set(thresholds):
                raise VPMValidationError(
                    "calibration_counts must cover every calibrated row"
                )
            if any(value <= 0 for value in counts.values()):
                raise VPMValidationError("per-row calibration counts must be positive")
            if sum(counts.values()) != int(self.calibration_count):
                raise VPMValidationError(
                    "per-row calibration counts must sum to calibration_count"
                )
        if not set(conflict_rows).issubset(set(thresholds)):
            raise VPMValidationError(
                "conflicting_action_rows must reference calibrated rows"
            )
        if not self.method:
            raise VPMValidationError("calibration method cannot be empty")
        _json_bytes(self.metadata)

    def has_conflicting_action(self, row_id: str) -> bool:
        key = str(row_id)
        if key not in self.acceptance_thresholds:
            raise VPMValidationError("unknown calibrated row: %s" % key)
        if self.conflict_contract_complete:
            return key in set(self.conflicting_action_rows)
        # Legacy v1 calibrations did not preserve this distinction.
        return True

    @property
    def digest(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "acceptance_thresholds": dict(self.acceptance_thresholds),
            "ambiguity_margins": dict(self.ambiguity_margins),
            "calibration_quantile": float(self.calibration_quantile),
            "calibration_count": int(self.calibration_count),
            "calibration_counts": dict(self.calibration_counts),
            "conflicting_action_rows": list(self.conflicting_action_rows),
            "conflict_contract_complete": bool(self.conflict_contract_complete),
            "method": self.method,
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VectorCalibration":
        payload = dict(data)
        version = str(payload.get("version", LEGACY_VECTOR_CALIBRATION_VERSION))
        if version == LEGACY_VECTOR_CALIBRATION_VERSION:
            payload.setdefault("calibration_counts", {})
            payload.setdefault("conflicting_action_rows", ())
            payload.setdefault("conflict_contract_complete", False)
        return cls(**payload)


@dataclass(frozen=True)
class VectorAddressBuild:
    matrix_blob: MatrixBlob
    manifest: VisualAddressManifest
    calibration: VectorCalibration
    prototype_action_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prototype_action_ids",
            tuple(str(value) for value in self.prototype_action_ids),
        )
        if self.matrix_blob.shape[0] != len(self.prototype_action_ids):
            raise VPMValidationError("prototype action ids must cover matrix rows")
        if self.matrix_blob.blob_id != self.manifest.matrix_blob_id:
            raise VPMValidationError(
                "address manifest does not reference supplied matrix blob"
            )
        if self.calibration.digest != self.manifest.calibration_artifact_id:
            raise VPMValidationError(
                "address manifest does not reference supplied calibration"
            )


def _validate_labels(
    vectors: np.ndarray,
    row_ids: Sequence[str],
    action_ids: Sequence[str],
    observation_ids: Sequence[str],
    *,
    label: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    rows = tuple(str(value) for value in row_ids)
    actions = tuple(str(value) for value in action_ids)
    observations = tuple(str(value) for value in observation_ids)
    if not (len(rows) == len(actions) == len(observations) == vectors.shape[0]):
        raise VPMValidationError(
            "%s vectors and labels must have identical length" % label
        )
    if any(not value for value in rows + actions + observations):
        raise VPMValidationError("%s labels cannot be empty" % label)
    if len(set(observations)) != len(observations):
        raise VPMValidationError("%s observation ids must be unique" % label)
    return rows, actions, observations


def _select_medoid_indices(
    vectors: np.ndarray,
    row_ids: Sequence[str],
) -> Tuple[int, ...]:
    selected = []
    for row_id in sorted(set(row_ids)):
        indices = [index for index, value in enumerate(row_ids) if value == row_id]
        local = vectors[indices]
        similarity = local @ local.T
        totals = similarity.sum(axis=1)
        best_local = max(
            range(len(indices)),
            key=lambda index: (float(totals[index]), -indices[index]),
        )
        selected.append(indices[best_local])
    return tuple(selected)


def _calibrate_prototypes(
    prototypes: np.ndarray,
    prototype_rows: Sequence[str],
    prototype_actions: Sequence[str],
    calibration_vectors: np.ndarray,
    calibration_rows: Sequence[str],
    calibration_actions: Sequence[str],
    *,
    quantile: float,
    strategy: str,
) -> VectorCalibration:
    values_by_row: Dict[str, list[float]] = {}
    margins_by_row: Dict[str, list[float]] = {}
    prototype_rows_tuple = tuple(prototype_rows)
    prototype_actions_tuple = tuple(prototype_actions)
    conflict_rows = set()

    for vector, row_id, action_id in zip(
        calibration_vectors,
        calibration_rows,
        calibration_actions,
    ):
        similarities = prototypes @ vector
        same = [
            index
            for index, candidate_row in enumerate(prototype_rows_tuple)
            if candidate_row == row_id
        ]
        if not same:
            raise VPMValidationError(
                "calibration row has no matching prototype: %s" % row_id
            )
        correct_score = max(float(similarities[index]) for index in same)
        conflicts = [
            index
            for index, candidate_action in enumerate(prototype_actions_tuple)
            if candidate_action != action_id
        ]
        values_by_row.setdefault(row_id, []).append(correct_score)
        if conflicts:
            conflict_rows.add(row_id)
            conflict_score = max(float(similarities[index]) for index in conflicts)
            margins_by_row.setdefault(row_id, []).append(correct_score - conflict_score)

    expected_rows = set(prototype_rows_tuple)
    if set(values_by_row) != expected_rows:
        raise VPMValidationError("calibration split must cover every prototype row")
    thresholds = {
        row_id: _lower_quantile(values_by_row[row_id], quantile)
        for row_id in sorted(expected_rows)
    }
    margins = {
        row_id: (
            _lower_quantile(margins_by_row[row_id], quantile)
            if row_id in margins_by_row
            else 0.0
        )
        for row_id in sorted(expected_rows)
    }
    calibration_counts = {
        row_id: len(values_by_row[row_id]) for row_id in sorted(expected_rows)
    }
    return VectorCalibration(
        acceptance_thresholds=thresholds,
        ambiguity_margins=margins,
        calibration_quantile=float(quantile),
        calibration_count=int(calibration_vectors.shape[0]),
        calibration_counts=calibration_counts,
        conflicting_action_rows=tuple(sorted(conflict_rows)),
        conflict_contract_complete=True,
        metadata={
            "prototype_strategy": strategy,
            "score": "cosine_similarity",
            "no_conflicting_action_rows": sorted(expected_rows - conflict_rows),
        },
    )


def build_vector_address(
    *,
    prototype_vectors: Any,
    prototype_row_ids: Sequence[str],
    prototype_action_ids: Sequence[str],
    prototype_observation_ids: Sequence[str],
    calibration_vectors: Any,
    calibration_row_ids: Sequence[str],
    calibration_action_ids: Sequence[str],
    calibration_observation_ids: Sequence[str],
    policy_artifact_id: str,
    source_scope: str,
    representation_spec_digest: str,
    encoder_manifest_id: Optional[str],
    strategy: str = "medoid",
    calibration_quantile: float = 0.0,
    deployment_status: str = "research",
) -> VectorAddressBuild:
    """Build medoid or all-example cosine retrieval with independent calibration."""

    prototype_matrix = l2_normalize_rows(prototype_vectors)
    calibration_matrix = l2_normalize_rows(calibration_vectors)
    prototype_rows, prototype_actions, prototype_ids = _validate_labels(
        prototype_matrix,
        prototype_row_ids,
        prototype_action_ids,
        prototype_observation_ids,
        label="prototype",
    )
    calibration_rows, calibration_actions, _ = _validate_labels(
        calibration_matrix,
        calibration_row_ids,
        calibration_action_ids,
        calibration_observation_ids,
        label="calibration",
    )
    if set(prototype_rows) != set(calibration_rows):
        raise VPMValidationError("prototype and calibration rows must match")

    if strategy == "medoid":
        selected = _select_medoid_indices(
            prototype_matrix,
            prototype_rows,
        )
    elif strategy == "all":
        selected = tuple(range(prototype_matrix.shape[0]))
    else:
        raise VPMValidationError("prototype strategy must be 'medoid' or 'all'")

    vectors = np.ascontiguousarray(
        prototype_matrix[list(selected)],
        dtype=np.float32,
    )
    rows = tuple(prototype_rows[index] for index in selected)
    actions = tuple(prototype_actions[index] for index in selected)
    source_ids = tuple(prototype_ids[index] for index in selected)
    calibration = _calibrate_prototypes(
        vectors,
        rows,
        actions,
        calibration_matrix,
        calibration_rows,
        calibration_actions,
        quantile=calibration_quantile,
        strategy=strategy,
    )
    blob = MatrixBlob.from_array(
        vectors,
        dtype="float32",
        metadata={
            "kind": "visual_address_prototypes",
            "strategy": strategy,
            "representation_spec_digest": representation_spec_digest,
            "representation_identity_scope": ("exact_float32_under_encoder_manifest"),
            "execution_scope_digest": (
                encoder_manifest_id or representation_spec_digest
            ),
        },
    )
    bindings = tuple(
        PrototypeBinding(
            prototype_id="%s:%04d" % (strategy, index),
            vector_index=index,
            policy_row_id=row_id,
            metadata={
                "action_id": action_id,
                "source_observation_id": source_id,
            },
        )
        for index, (row_id, action_id, source_id) in enumerate(
            zip(rows, actions, source_ids)
        )
    )
    manifest = VisualAddressManifest(
        address_kind="frozen_embedding_%s" % strategy,
        policy_artifact_id=str(policy_artifact_id),
        matrix_blob_id=blob.blob_id,
        matrix_row_count=blob.shape[0],
        representation_spec_digest=str(representation_spec_digest),
        calibration_artifact_id=calibration.digest,
        score_semantics="similarity",
        source_scope=str(source_scope),
        prototype_bindings=bindings,
        encoder_manifest_id=encoder_manifest_id,
        deployment_status=str(deployment_status),
        metadata={
            "calibration_quantile": float(calibration_quantile),
            "representation_identity_scope": ("exact_float32_under_encoder_manifest"),
            "execution_scope_digest": (
                encoder_manifest_id or representation_spec_digest
            ),
        },
    )
    return VectorAddressBuild(
        matrix_blob=blob,
        manifest=manifest,
        calibration=calibration,
        prototype_action_ids=actions,
    )


class VectorAddressIndex:
    """Runtime cosine matcher over an identified prototype matrix."""

    def __init__(self, build: VectorAddressBuild) -> None:
        self.build = build
        matrix = build.matrix_blob.to_array()
        self._matrix = l2_normalize_rows(matrix)
        self._row_ids = tuple(
            binding.policy_row_id for binding in build.manifest.prototype_bindings
        )
        self._prototype_ids = tuple(
            binding.prototype_id for binding in build.manifest.prototype_bindings
        )
        self._action_ids = tuple(build.prototype_action_ids)

    def contract(self) -> VisualAddressContract:
        return VisualAddressContract(
            provider_kind=self.build.manifest.address_kind,
            provider_version=VECTOR_ADDRESS_READER_VERSION,
            score_semantics="similarity",
            observation_spec_digest=(self.build.manifest.representation_spec_digest),
            representation_spec_digest=(self.build.manifest.representation_spec_digest),
            address_artifact_id=str(self.build.manifest.manifest_id),
            calibration_artifact_id=self.build.calibration.digest,
            policy_artifact_id=self.build.manifest.policy_artifact_id,
            source_scope=self.build.manifest.source_scope,
            replay_contract="exact_decision",
            metadata={
                "encoder_manifest_id": (self.build.manifest.encoder_manifest_id),
                "prototype_count": len(self._row_ids),
                "representation_identity_scope": (
                    "exact_float32_under_encoder_manifest"
                ),
                "execution_scope_digest": (
                    self.build.manifest.encoder_manifest_id
                    or self.build.manifest.representation_spec_digest
                ),
            },
        )

    def match_vector(
        self,
        vector: Any,
        *,
        observation_digest: str,
        trace: Optional[Mapping[str, Any]] = None,
    ) -> VisualAddressDecision:
        query_matrix = _as_matrix(
            np.asarray(vector, dtype=np.float32).reshape(1, -1),
            name="query vector",
        )
        if query_matrix.shape[1] != self._matrix.shape[1]:
            raise VPMValidationError(
                "query vector dimension does not match address index"
            )
        norm = float(np.linalg.norm(query_matrix[0]))
        contract = self.contract()
        execution_scope = str(
            contract.metadata.get(
                "execution_scope_digest",
                contract.representation_spec_digest,
            )
        )
        representation_digest = _representation_digest(
            query_matrix[0],
            contract.representation_spec_digest,
            execution_scope,
        )
        if norm <= 0.0:
            return VisualAddressDecision(
                accepted=False,
                reason="zero_visual_representation",
                observation_digest=str(observation_digest),
                representation_digest=representation_digest,
                provider_kind=contract.provider_kind,
                provider_version=contract.provider_version,
                score_semantics="similarity",
                address_artifact_id=contract.address_artifact_id,
                calibration_artifact_id=contract.calibration_artifact_id,
                policy_artifact_id=contract.policy_artifact_id,
                nearest_row_id=None,
                nearest_score=None,
                second_row_id=None,
                second_score=None,
                ambiguity_measure=None,
                trace=dict(trace or {}),
            )

        query = query_matrix[0] / norm
        similarities = self._matrix @ query
        ranking = sorted(
            range(len(self._row_ids)),
            key=lambda index: (
                -float(similarities[index]),
                self._row_ids[index],
                self._prototype_ids[index],
            ),
        )
        first = int(ranking[0])
        nearest_row = self._row_ids[first]
        nearest_action = self._action_ids[first]
        conflict_candidates = [
            index for index in ranking[1:] if self._action_ids[index] != nearest_action
        ]
        has_conflict = self.build.calibration.has_conflicting_action(nearest_row)
        if has_conflict and not conflict_candidates:
            raise VPMValidationError(
                "calibration requires a conflicting action candidate that "
                "runtime cannot reproduce"
            )
        second = int(conflict_candidates[0]) if conflict_candidates else None
        nearest_score = float(similarities[first])
        second_score = float(similarities[second]) if second is not None else None
        margin = nearest_score - second_score if second_score is not None else None
        threshold = float(self.build.calibration.acceptance_thresholds[nearest_row])
        required_margin = (
            float(self.build.calibration.ambiguity_margins[nearest_row])
            if has_conflict
            else None
        )
        similarity_ok = nearest_score + 1e-12 >= threshold
        margin_ok = (
            True
            if not has_conflict
            else (
                margin is not None
                and required_margin is not None
                and margin + 1e-12 >= required_margin
            )
        )
        accepted = similarity_ok and margin_ok
        reason = (
            "accepted"
            if accepted
            else (
                "visual_similarity_below_threshold"
                if not similarity_ok
                else "ambiguous_visual_address"
            )
        )
        checks = []
        if similarity_ok:
            checks.append("similarity_threshold")
        if has_conflict and margin_ok:
            checks.append("conflicting_action_margin")
        elif not has_conflict:
            checks.append("no_conflicting_action_candidates")
        decision_trace = {
            "prototype_id": self._prototype_ids[first],
            "nearest_action_id": nearest_action,
            "acceptance_threshold": threshold,
            "required_conflicting_action_margin": required_margin,
            "has_conflicting_action_candidate": has_conflict,
            "representation_identity_scope": ("exact_float32_under_encoder_manifest"),
            "execution_scope_digest": execution_scope,
        }
        decision_trace.update(dict(trace or {}))
        return VisualAddressDecision(
            accepted=accepted,
            reason=reason,
            observation_digest=str(observation_digest),
            representation_digest=representation_digest,
            provider_kind=contract.provider_kind,
            provider_version=contract.provider_version,
            score_semantics="similarity",
            address_artifact_id=contract.address_artifact_id,
            calibration_artifact_id=contract.calibration_artifact_id,
            policy_artifact_id=contract.policy_artifact_id,
            nearest_row_id=nearest_row,
            nearest_score=nearest_score,
            second_row_id=(self._row_ids[second] if second is not None else None),
            second_score=second_score,
            ambiguity_measure=margin,
            matched_row_id=nearest_row if accepted else None,
            exact_match=False,
            accepted_by=tuple(checks) if accepted else (),
            trace=decision_trace,
        )


class FrozenVectorAddressProvider:
    """VisualAddressProvider that combines a frozen encoder and vector index."""

    def __init__(
        self,
        encoder: FrozenVisualEncoder,
        index: VectorAddressIndex,
    ) -> None:
        manifest = encoder.manifest()
        expected = index.build.manifest.encoder_manifest_id
        if expected is not None and expected != manifest.manifest_id:
            raise VPMValidationError("encoder manifest does not match address index")
        if manifest.output_dimension != index._matrix.shape[1]:
            raise VPMValidationError(
                "encoder output dimension does not match address index"
            )
        self.encoder = encoder
        self.index = index

    def contract(self) -> VisualAddressContract:
        return self.index.contract()

    def read(self, observation: ImageObservation) -> VisualAddressDecision:
        vectors = self.encoder.encode_batch((observation,))
        return self.index.match_vector(
            vectors[0],
            observation_digest=observation.raw_digest,
            trace={"observation": observation.to_descriptor()},
        )


class NormalizedPixelEncoder:
    """No-model normalized-pixel baseline for same-shape observations."""

    def __init__(self, *, height: int, width: int) -> None:
        if height <= 0 or width <= 0:
            raise VPMValidationError("pixel encoder dimensions must be positive")
        self.height = int(height)
        self.width = int(width)
        spec = {
            "kind": "normalized_grayscale_pixels",
            "height": self.height,
            "width": self.width,
            "normalization": "mean_center_then_l2",
        }
        digest = _sha256_json(spec)
        self._manifest = EncoderManifest(
            provider_kind="normalized_pixel_baseline",
            model_id="none",
            revision="v1",
            architecture="flattened_grayscale",
            weights_digest="none",
            preprocessing_digest=digest,
            output_dimension=self.height * self.width,
            normalization="mean_center_then_l2",
            framework="numpy",
            framework_version=np.__version__,
            license_id="not-applicable",
            source_record=("zeromodel.visual_retrieval.NormalizedPixelEncoder"),
            metadata=spec,
        )

    def manifest(self) -> EncoderManifest:
        return self._manifest

    def encode_batch(
        self,
        observations: Sequence[ImageObservation],
    ) -> np.ndarray:
        items = tuple(observations)
        if not items:
            raise VPMValidationError("pixel encoder batch cannot be empty")
        rows = []
        for observation in items:
            array = observation.pixels
            if array.shape[:2] != (self.height, self.width):
                raise VPMValidationError(
                    "pixel observation shape violates encoder contract"
                )
            if array.ndim == 3:
                rgb = array[:, :, :3].astype(np.uint16)
                gray = (
                    77 * rgb[:, :, 0] + 150 * rgb[:, :, 1] + 29 * rgb[:, :, 2] + 128
                ) // 256
            else:
                gray = array
            vector = gray.astype(np.float32).reshape(-1) / 255.0
            vector = vector - float(vector.mean())
            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                vector = vector / norm
            rows.append(vector)
        matrix = np.ascontiguousarray(rows, dtype=np.float32)
        matrix.flags.writeable = False
        return matrix


@dataclass(frozen=True)
class LinearProbeBuild:
    weights_blob: MatrixBlob
    row_ids: Tuple[str, ...]
    action_ids: Tuple[str, ...]
    calibration: VectorCalibration
    policy_artifact_id: str
    source_scope: str
    representation_spec_digest: str
    encoder_manifest_id: str
    model_id: str


class LinearProbeIndex:
    """Ridge least-squares row classifier with calibrated rejection."""

    def __init__(self, build: LinearProbeBuild) -> None:
        self.build = build
        self._weights = build.weights_blob.to_array()
        if self._weights.shape[1] != len(build.row_ids):
            raise VPMValidationError("linear probe class count does not match row ids")

    def contract(self) -> VisualAddressContract:
        return VisualAddressContract(
            provider_kind="rejection_equipped_linear_probe",
            provider_version=LINEAR_PROBE_VERSION,
            score_semantics="similarity",
            observation_spec_digest=self.build.representation_spec_digest,
            representation_spec_digest=self.build.representation_spec_digest,
            address_artifact_id=self.build.model_id,
            calibration_artifact_id=self.build.calibration.digest,
            policy_artifact_id=self.build.policy_artifact_id,
            source_scope=self.build.source_scope,
            replay_contract="exact_decision",
            metadata={
                "encoder_manifest_id": self.build.encoder_manifest_id,
                "representation_identity_scope": (
                    "exact_float32_under_encoder_manifest"
                ),
                "execution_scope_digest": self.build.encoder_manifest_id,
            },
        )

    def match_vector(
        self,
        vector: Any,
        *,
        observation_digest: str,
    ) -> VisualAddressDecision:
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.size + 1 != self._weights.shape[0] or not np.isfinite(query).all():
            raise VPMValidationError("linear probe query violates fitted dimension")
        scores = np.concatenate((query, np.ones(1, dtype=np.float32))) @ self._weights
        ranking = sorted(
            range(len(self.build.row_ids)),
            key=lambda index: (
                -float(scores[index]),
                self.build.row_ids[index],
            ),
        )
        first = ranking[0]
        action = self.build.action_ids[first]
        conflicts = [
            index for index in ranking[1:] if self.build.action_ids[index] != action
        ]
        row_id = self.build.row_ids[first]
        has_conflict = self.build.calibration.has_conflicting_action(row_id)
        if has_conflict and not conflicts:
            raise VPMValidationError(
                "linear-probe calibration requires a conflicting action "
                "candidate that runtime cannot reproduce"
            )
        second = conflicts[0] if conflicts else None
        nearest = float(scores[first])
        second_score = float(scores[second]) if second is not None else None
        margin = nearest - second_score if second_score is not None else None
        threshold = float(self.build.calibration.acceptance_thresholds[row_id])
        required_margin = (
            float(self.build.calibration.ambiguity_margins[row_id])
            if has_conflict
            else None
        )
        similarity_ok = nearest + 1e-12 >= threshold
        margin_ok = (
            True
            if not has_conflict
            else (
                margin is not None
                and required_margin is not None
                and margin + 1e-12 >= required_margin
            )
        )
        accepted = similarity_ok and margin_ok
        contract = self.contract()
        checks = []
        if similarity_ok:
            checks.append("class_score_threshold")
        if has_conflict and margin_ok:
            checks.append("conflicting_action_margin")
        elif not has_conflict:
            checks.append("no_conflicting_action_candidates")
        return VisualAddressDecision(
            accepted=accepted,
            reason=(
                "accepted"
                if accepted
                else (
                    "visual_similarity_below_threshold"
                    if not similarity_ok
                    else "ambiguous_visual_address"
                )
            ),
            observation_digest=str(observation_digest),
            representation_digest=_representation_digest(
                query,
                contract.representation_spec_digest,
                self.build.encoder_manifest_id,
            ),
            provider_kind=contract.provider_kind,
            provider_version=contract.provider_version,
            score_semantics="similarity",
            address_artifact_id=contract.address_artifact_id,
            calibration_artifact_id=contract.calibration_artifact_id,
            policy_artifact_id=contract.policy_artifact_id,
            nearest_row_id=row_id,
            nearest_score=nearest,
            second_row_id=(self.build.row_ids[second] if second is not None else None),
            second_score=second_score,
            ambiguity_measure=margin,
            matched_row_id=row_id if accepted else None,
            accepted_by=tuple(checks) if accepted else (),
            trace={
                "acceptance_threshold": threshold,
                "required_conflicting_action_margin": required_margin,
                "has_conflicting_action_candidate": has_conflict,
                "nearest_action_id": action,
                "representation_identity_scope": (
                    "exact_float32_under_encoder_manifest"
                ),
                "execution_scope_digest": self.build.encoder_manifest_id,
            },
        )


def build_linear_probe(
    *,
    prototype_vectors: Any,
    prototype_row_ids: Sequence[str],
    prototype_action_ids: Sequence[str],
    calibration_vectors: Any,
    calibration_row_ids: Sequence[str],
    calibration_action_ids: Sequence[str],
    policy_artifact_id: str,
    source_scope: str,
    representation_spec_digest: str,
    encoder_manifest_id: str,
    ridge: float = 1e-3,
    calibration_quantile: float = 0.0,
) -> LinearProbeBuild:
    x = l2_normalize_rows(prototype_vectors)
    calibration_x = l2_normalize_rows(calibration_vectors)
    rows = tuple(sorted(set(str(value) for value in prototype_row_ids)))
    if set(rows) != set(str(value) for value in calibration_row_ids):
        raise VPMValidationError(
            "linear probe prototype and calibration rows must match"
        )
    action_by_row: Dict[str, str] = {}
    for row_id, action_id in zip(
        prototype_row_ids,
        prototype_action_ids,
    ):
        key = str(row_id)
        value = str(action_id)
        if key in action_by_row and action_by_row[key] != value:
            raise VPMValidationError("one policy row cannot map to multiple actions")
        action_by_row[key] = value
    class_index = {row_id: index for index, row_id in enumerate(rows)}
    y = np.zeros((x.shape[0], len(rows)), dtype=np.float32)
    for sample, row_id in enumerate(prototype_row_ids):
        y[sample, class_index[str(row_id)]] = 1.0
    x_aug = np.concatenate(
        (
            x,
            np.ones((x.shape[0], 1), dtype=np.float32),
        ),
        axis=1,
    )
    regularizer = np.eye(x_aug.shape[1], dtype=np.float32) * float(ridge)
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(
        x_aug.T @ x_aug + regularizer,
        x_aug.T @ y,
    ).astype(np.float32)

    calibration_scores = (
        np.concatenate(
            (
                calibration_x,
                np.ones(
                    (calibration_x.shape[0], 1),
                    dtype=np.float32,
                ),
            ),
            axis=1,
        )
        @ weights
    )
    score_rows = np.eye(len(rows), dtype=np.float32)
    calibration = _calibrate_prototypes(
        score_rows,
        rows,
        tuple(action_by_row[row_id] for row_id in rows),
        calibration_scores,
        tuple(str(value) for value in calibration_row_ids),
        tuple(str(value) for value in calibration_action_ids),
        quantile=calibration_quantile,
        strategy="linear_probe",
    )
    blob = MatrixBlob.from_array(
        weights,
        dtype="float32",
        metadata={
            "kind": "visual_linear_probe",
            "ridge": float(ridge),
            "representation_identity_scope": ("exact_float32_under_encoder_manifest"),
            "execution_scope_digest": encoder_manifest_id,
        },
    )
    model_id = hashlib.sha256(
        b"zeromodel.linear-probe.identity.v2\0"
        + blob.blob_id.encode("utf-8")
        + calibration.digest.encode("utf-8")
        + _json_bytes(rows)
        + str(encoder_manifest_id).encode("utf-8")
    ).hexdigest()
    return LinearProbeBuild(
        weights_blob=blob,
        row_ids=rows,
        action_ids=tuple(action_by_row[row_id] for row_id in rows),
        calibration=calibration,
        policy_artifact_id=str(policy_artifact_id),
        source_scope=str(source_scope),
        representation_spec_digest=str(representation_spec_digest),
        encoder_manifest_id=str(encoder_manifest_id),
        model_id=model_id,
    )
