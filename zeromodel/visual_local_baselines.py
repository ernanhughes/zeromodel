"""Deterministic local visual baselines built on bounded integer registration."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import VPMValidationError
from .visual_address import (
    ImageObservation,
    VisualAddressContract,
    VisualAddressDecision,
)
from .visual_dataset import VisualDatasetManifest, VisualExampleRecord
from .visual_benchmark import BenchmarkSystemResult, VisualBenchmarkMetrics
from .visual_experiment import EXPECTED_ACCEPT, EXPECTED_REJECT, evaluate_visual_provider
from .visual_registration import (
    RegistrationConfig,
    RegistrationResult,
    _grayscale,
    _displacement_order,
    register_integer_translation,
)


REGISTERED_PIXEL_PROVIDER_VERSION = "zeromodel-registered-pixel-provider/v1"
LOCAL_BASELINE_SELECTION_VERSION = "zeromodel-local-baseline-selection/v1"


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("local-baseline values must be JSON-serializable") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_split(records: Sequence[VisualExampleRecord], split: str) -> Tuple[VisualExampleRecord, ...]:
    selected = tuple(record for record in records if record.split == split)
    if not selected:
        raise VPMValidationError("required split is empty: %s" % split)
    return selected


def _records_digest(records: Sequence[VisualExampleRecord]) -> str:
    return _sha256_json([record.to_dict() for record in records])


def _conservative_distance_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise VPMValidationError("distance quantile requires at least one value")
    if not np.isfinite(quantile) or not (0.0 <= quantile <= 1.0):
        raise VPMValidationError("distance quantile must be in [0, 1]")
    index = int(np.floor((1.0 - float(quantile)) * float(len(ordered) - 1)))
    return ordered[index]


@dataclass(frozen=True)
class RegisteredPixelPrototype:
    prototype_observation_id: str
    row_id: str
    action_id: str
    observation_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prototype_observation_id": self.prototype_observation_id,
            "row_id": self.row_id,
            "action_id": self.action_id,
            "observation_digest": self.observation_digest,
        }


@dataclass(frozen=True)
class RegisteredPixelCandidate:
    row_id: str
    action_id: str
    prototype_observation_id: str
    observation_digest: str
    registration: RegistrationResult

    @property
    def distance(self) -> float:
        return float(self.registration.distance_after)

    @property
    def displacement_magnitude(self) -> int:
        return abs(int(self.registration.dx)) + abs(int(self.registration.dy))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "action_id": self.action_id,
            "prototype_observation_id": self.prototype_observation_id,
            "observation_digest": self.observation_digest,
            "registration": self.registration.to_dict(),
        }


@dataclass(frozen=True)
class RegisteredPixelObservationRanking:
    observation_id: str
    best: RegisteredPixelCandidate
    second: Optional[RegisteredPixelCandidate]
    raw_best: RegisteredPixelCandidate
    expected_row_id: Optional[str]
    expected_action_id: Optional[str]
    expected_disposition: str


@dataclass(frozen=True)
class RegisteredPixelCalibration:
    threshold: float
    ambiguity_margin: float
    quantile: float
    registration_config_digest: str
    prototype_digest: str
    benign_calibration_digest: str
    rejection_calibration_digest: str
    source_scope: str
    policy_artifact_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = REGISTERED_PIXEL_PROVIDER_VERSION

    def __post_init__(self) -> None:
        if self.version != REGISTERED_PIXEL_PROVIDER_VERSION:
            raise VPMValidationError("unsupported registered-pixel calibration version")
        if not np.isfinite(float(self.threshold)):
            raise VPMValidationError("threshold must be finite")
        if not np.isfinite(float(self.ambiguity_margin)):
            raise VPMValidationError("ambiguity_margin must be finite")
        if not np.isfinite(float(self.quantile)) or not (0.0 <= float(self.quantile) <= 1.0):
            raise VPMValidationError("quantile must be in [0, 1]")

    @property
    def digest(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "threshold": float(self.threshold),
            "ambiguity_margin": float(self.ambiguity_margin),
            "quantile": float(self.quantile),
            "registration_config_digest": self.registration_config_digest,
            "prototype_digest": self.prototype_digest,
            "benign_calibration_digest": self.benign_calibration_digest,
            "rejection_calibration_digest": self.rejection_calibration_digest,
            "source_scope": self.source_scope,
            "policy_artifact_id": self.policy_artifact_id,
            "metadata": _freeze(self.metadata),
        }


@dataclass(frozen=True)
class RegisteredPixelCandidateEvaluation:
    quantile: float
    threshold: float
    ambiguity_margin: float
    feasible: bool
    infeasible_reasons: Tuple[str, ...]
    calibration: RegisteredPixelCalibration
    benign_result: Any
    rejection_result: Any

    def to_dict(self) -> Dict[str, Any]:
        benign_metrics = self.benign_result.metrics
        rejection_metrics = self.rejection_result.metrics
        return {
            "quantile": float(self.quantile),
            "threshold": float(self.threshold),
            "ambiguity_margin": float(self.ambiguity_margin),
            "feasible": bool(self.feasible),
            "infeasible_reasons": list(self.infeasible_reasons),
            "accepted_benign_count": int(benign_metrics.accepted_benign_count),
            "accepted_exact_row_precision": (
                None
                if benign_metrics.accepted_benign_count == 0
                else float(benign_metrics.accepted_benign_row_correctness)
            ),
            "accepted_action_precision": (
                None
                if benign_metrics.accepted_benign_count == 0
                else float(benign_metrics.accepted_benign_action_correctness)
            ),
            "benign_coverage": float(
                benign_metrics.accepted_benign_count
                / float(benign_metrics.false_reject_opportunities or 1)
            ),
            "accepted_exact_row_recall": float(benign_metrics.benign_row_accuracy),
            "raw_benign_exact_row_accuracy": float(benign_metrics.top1_benign_row_accuracy),
            "false_accepts": int(rejection_metrics.false_accept_count),
            "conflicting_action_accepts": int(benign_metrics.conflicting_action_error_count),
            "false_rejects": int(benign_metrics.false_reject_count),
            "calibration_artifact_id": self.calibration.digest,
            "benign_result": self.benign_result.to_dict(),
            "rejection_result": self.rejection_result.to_dict(),
        }


@dataclass(frozen=True)
class RegisteredPixelSelectionArtifact:
    registration_config_digest: str
    prototype_digest: str
    benign_calibration_digest: str
    rejection_calibration_digest: str
    candidate_grid_digest: str
    selection_status: str
    selected_quantile: Optional[float]
    selected_threshold: Optional[float]
    selected_ambiguity_margin: Optional[float]
    selected_calibration_digest: Optional[str]
    selection_rule: str
    source_scope: str
    policy_artifact_id: str
    candidates: Tuple[RegisteredPixelCandidateEvaluation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = LOCAL_BASELINE_SELECTION_VERSION

    @property
    def digest(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "registration_config_digest": self.registration_config_digest,
            "prototype_digest": self.prototype_digest,
            "benign_calibration_digest": self.benign_calibration_digest,
            "rejection_calibration_digest": self.rejection_calibration_digest,
            "candidate_grid_digest": self.candidate_grid_digest,
            "selection_status": self.selection_status,
            "selected_quantile": self.selected_quantile,
            "selected_threshold": self.selected_threshold,
            "selected_ambiguity_margin": self.selected_ambiguity_margin,
            "selected_calibration_digest": self.selected_calibration_digest,
            "selection_rule": self.selection_rule,
            "source_scope": self.source_scope,
            "policy_artifact_id": self.policy_artifact_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": _freeze(self.metadata),
        }


class RegisteredPixelAddressProvider:
    def __init__(
        self,
        *,
        prototypes: Mapping[str, Tuple[RegisteredPixelPrototype, ImageObservation]],
        calibration: RegisteredPixelCalibration,
        registration_config: RegistrationConfig,
        provider_id: str,
    ) -> None:
        items = []
        for key in sorted(prototypes):
            prototype, observation = prototypes[key]
            items.append((prototype, observation, _grayscale(observation.pixels)))
        if not items:
            raise VPMValidationError("registered local baseline requires prototypes")
        self._items = tuple(items)
        self._prototype_pixels = np.stack([gray for _prototype, _observation, gray in self._items], axis=0)
        self._prototype_rows = tuple(item[0].row_id for item in self._items)
        self._prototype_actions = tuple(item[0].action_id for item in self._items)
        self._prototype_observation_ids = tuple(item[0].prototype_observation_id for item in self._items)
        self._prototype_digests = tuple(item[0].observation_digest for item in self._items)
        self._shift_grid = tuple(
            sorted(
                (
                    (dx, dy)
                    for dy in range(-int(registration_config.max_dy), int(registration_config.max_dy) + 1)
                    for dx in range(-int(registration_config.max_dx), int(registration_config.max_dx) + 1)
                ),
                key=lambda displacement: _displacement_order(displacement[0], displacement[1]),
            )
        )
        self._calibration = calibration
        self._registration_config = registration_config
        self._provider_id = str(provider_id)

    def contract(self) -> VisualAddressContract:
        return VisualAddressContract(
            provider_kind="registered_local_normalized_pixels",
            provider_version=REGISTERED_PIXEL_PROVIDER_VERSION,
            score_semantics="distance",
            observation_spec_digest=self._registration_config.digest,
            representation_spec_digest=self._registration_config.digest,
            address_artifact_id=self._provider_id,
            calibration_artifact_id=self._calibration.digest,
            policy_artifact_id=self._calibration.policy_artifact_id,
            source_scope=self._calibration.source_scope,
            replay_contract="exact_decision",
            metadata={
                "registration_config_digest": self._registration_config.digest,
                "threshold": float(self._calibration.threshold),
                "ambiguity_margin": float(self._calibration.ambiguity_margin),
            },
        )

    def _sort_key(
        self,
        candidate: RegisteredPixelCandidate,
        *,
        use_registered_distance: bool,
    ) -> Tuple[float, float, int, str, str]:
        target_distance = (
            float(candidate.registration.distance_after)
            if use_registered_distance
            else float(candidate.registration.distance_before)
        )
        return (
            target_distance,
            -float(candidate.registration.overlap_fraction),
            candidate.displacement_magnitude,
            candidate.row_id,
            candidate.prototype_observation_id,
        )

    def _rank(
        self,
        observation: ImageObservation,
    ) -> Tuple[RegisteredPixelCandidate, Optional[RegisteredPixelCandidate], RegisteredPixelCandidate]:
        observation_pixels = _grayscale(observation.pixels)
        prototype_pixels = self._prototype_pixels
        prototype_count, height, width = prototype_pixels.shape
        best_distances = np.full(prototype_count, np.inf, dtype=np.float64)
        best_dx = np.zeros(prototype_count, dtype=np.int32)
        best_dy = np.zeros(prototype_count, dtype=np.int32)
        best_overlap = np.zeros(prototype_count, dtype=np.float64)
        best_valid = np.zeros(prototype_count, dtype=np.int32)
        raw_distances: Optional[np.ndarray] = None
        raw_overlap = 0.0
        raw_valid = 0

        for dx, dy in self._shift_grid:
            x0 = max(0, dx)
            x1 = min(width, width + dx)
            y0 = max(0, dy)
            y1 = min(height, height + dy)
            valid_width = x1 - x0
            valid_height = y1 - y0
            if valid_width <= 0 or valid_height <= 0:
                continue
            valid_count = int(valid_width * valid_height)
            overlap_fraction = float(valid_count) / float(height * width)
            if overlap_fraction + 1e-12 < float(self._registration_config.minimum_overlap_fraction):
                continue

            prototype_region = prototype_pixels[:, y0:y1, x0:x1]
            observation_region = observation_pixels[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
            observation_sum = float(observation_region.sum(dtype=np.float64))
            observation_sum_sq = float(np.square(observation_region, dtype=np.float32).sum(dtype=np.float64))
            prototype_sum = prototype_region.sum(axis=(1, 2), dtype=np.float64)
            prototype_sum_sq = np.square(prototype_region, dtype=np.float32).sum(axis=(1, 2), dtype=np.float64)
            cross_sum = np.multiply(prototype_region, observation_region, dtype=np.float32).sum(
                axis=(1, 2),
                dtype=np.float64,
            )

            pixel_count = float(valid_count)
            prototype_centered_sum_sq = np.maximum(0.0, prototype_sum_sq - np.square(prototype_sum) / pixel_count)
            observation_centered_sum_sq = max(0.0, observation_sum_sq - (observation_sum * observation_sum) / pixel_count)
            distances = np.empty(prototype_count, dtype=np.float64)
            both_constant = (prototype_centered_sum_sq <= 1e-12) & (observation_centered_sum_sq <= 1e-12)
            one_constant = (prototype_centered_sum_sq <= 1e-12) ^ (observation_centered_sum_sq <= 1e-12)
            regular = ~(both_constant | one_constant)
            distances[both_constant] = 0.0
            distances[one_constant] = 1.0
            if np.any(regular):
                centered_cross_sum = cross_sum[regular] - (prototype_sum[regular] * observation_sum) / pixel_count
                cosine = centered_cross_sum / np.sqrt(
                    prototype_centered_sum_sq[regular] * observation_centered_sum_sq
                )
                cosine = np.clip(cosine, -1.0, 1.0)
                distances[regular] = np.sqrt(np.maximum(0.0, 2.0 - (2.0 * cosine)))

            if dx == 0 and dy == 0:
                raw_distances = distances.copy()
                raw_overlap = overlap_fraction
                raw_valid = valid_count
            improved = distances < best_distances
            if np.any(improved):
                best_distances[improved] = distances[improved]
                best_dx[improved] = dx
                best_dy[improved] = dy
                best_overlap[improved] = overlap_fraction
                best_valid[improved] = valid_count

        if raw_distances is None:
            raise VPMValidationError("registration grid must include zero displacement")

        candidates = []
        for index in range(prototype_count):
            registration = RegistrationResult(
                dx=int(best_dx[index]),
                dy=int(best_dy[index]),
                distance_before=float(raw_distances[index]),
                distance_after=float(best_distances[index]),
                distance_improvement=float(raw_distances[index] - best_distances[index]),
                overlap_fraction=float(best_overlap[index]),
                valid_pixel_count=int(best_valid[index]),
                score_before=-float(raw_distances[index]),
                score_after=-float(best_distances[index]),
                registration_succeeded=True,
                rejection_reason=None,
            )
            candidates.append(
                RegisteredPixelCandidate(
                    row_id=self._prototype_rows[index],
                    action_id=self._prototype_actions[index],
                    prototype_observation_id=self._prototype_observation_ids[index],
                    observation_digest=self._prototype_digests[index],
                    registration=registration,
                )
            )
        ranking = sorted(candidates, key=lambda candidate: self._sort_key(candidate, use_registered_distance=True))
        raw_ranking = sorted(candidates, key=lambda candidate: self._sort_key(candidate, use_registered_distance=False))
        best = ranking[0]
        raw_best = raw_ranking[0]
        second = next(
            (
                candidate
                for candidate in ranking[1:]
                if candidate.action_id != best.action_id
            ),
            None,
        )
        return best, second, raw_best

    def read(self, observation: ImageObservation) -> VisualAddressDecision:
        best, second, raw_best = self._rank(observation)
        margin = (
            float(second.distance - best.distance)
            if second is not None
            else float("inf")
        )
        distance_ok = float(best.distance) <= float(self._calibration.threshold) + 1e-12
        margin_ok = margin + 1e-12 >= float(self._calibration.ambiguity_margin)
        accepted = distance_ok and margin_ok
        if not distance_ok:
            reason = "registered_distance_above_threshold"
        elif not margin_ok:
            reason = "ambiguous_registered_local_address"
        else:
            reason = "accepted"
        return VisualAddressDecision(
            accepted=accepted,
            reason=reason,
            observation_digest=observation.raw_digest,
            representation_digest=observation.raw_digest,
            provider_kind="registered_local_normalized_pixels",
            provider_version=REGISTERED_PIXEL_PROVIDER_VERSION,
            score_semantics="distance",
            address_artifact_id=self._provider_id,
            calibration_artifact_id=self._calibration.digest,
            policy_artifact_id=self._calibration.policy_artifact_id,
            nearest_row_id=best.row_id,
            nearest_score=float(best.distance),
            second_row_id=(None if second is None else second.row_id),
            second_score=(None if second is None else float(second.distance)),
            ambiguity_measure=float(margin),
            local_evidence_score=float(best.registration.distance_improvement),
            visible_evidence_fraction=float(best.registration.overlap_fraction),
            critical_evidence_present=None,
            matched_row_id=(best.row_id if accepted else None),
            accepted_by=(
                tuple(
                    check
                    for check, ok in (
                        ("distance_threshold", distance_ok),
                        ("conflicting_action_margin", margin_ok),
                    )
                    if ok
                )
            ),
            trace={
                "raw_top1_row_id": raw_best.row_id,
                "raw_top1_action_id": raw_best.action_id,
                "raw_top1_prototype_observation_id": raw_best.prototype_observation_id,
                "raw_top1_distance": float(raw_best.registration.distance_before),
                "raw_top1_overlap_fraction": float(raw_best.registration.overlap_fraction),
                "registration": best.registration.to_dict(),
                "candidate_row_id": best.row_id,
                "candidate_action_id": best.action_id,
                "prototype_observation_id": best.prototype_observation_id,
                "distance_threshold": float(self._calibration.threshold),
                "required_conflicting_action_margin": float(self._calibration.ambiguity_margin),
                "distance_before": float(best.registration.distance_before),
                "distance_after": float(best.registration.distance_after),
                "distance_improvement": float(best.registration.distance_improvement),
                "overlap_fraction": float(best.registration.overlap_fraction),
                "valid_pixel_count": int(best.registration.valid_pixel_count),
                "dx": int(best.registration.dx),
                "dy": int(best.registration.dy),
                "registration_succeeded": bool(best.registration.registration_succeeded),
                "rejection_reason": best.registration.rejection_reason,
                "second_candidate_row_id": (None if second is None else second.row_id),
                "second_candidate_action_id": (None if second is None else second.action_id),
                "second_candidate_distance": (None if second is None else float(second.distance)),
            },
        )


def build_registered_pixel_prototypes(
    *,
    dataset_manifest: VisualDatasetManifest,
    observations: Mapping[str, ImageObservation],
) -> Mapping[str, Tuple[RegisteredPixelPrototype, ImageObservation]]:
    prototypes = {}
    for record in _require_split(dataset_manifest.records, "prototype"):
        if record.row_id is None or record.action_id is None:
            raise VPMValidationError("prototype rows require row and action ids")
        prototypes[record.observation_id] = (
            RegisteredPixelPrototype(
                prototype_observation_id=record.observation_id,
                row_id=str(record.row_id),
                action_id=str(record.action_id),
                observation_digest=record.observation_digest,
            ),
            observations[record.observation_id],
        )
    return prototypes


def _provider_id(
    *,
    registration_config: RegistrationConfig,
    calibration: RegisteredPixelCalibration,
) -> str:
    return _sha256_json(
        {
            "version": REGISTERED_PIXEL_PROVIDER_VERSION,
            "registration_config_digest": registration_config.digest,
            "calibration_digest": calibration.digest,
            "source_scope": calibration.source_scope,
            "policy_artifact_id": calibration.policy_artifact_id,
        }
    )


def build_registered_pixel_candidates(
    *,
    dataset_manifest: VisualDatasetManifest,
    observations: Mapping[str, ImageObservation],
    policy_lookup: Any,
    registration_config: RegistrationConfig,
    quantiles: Sequence[float],
    source_scope: str,
    capture_ids: Optional[set[str]] = None,
) -> Tuple[RegisteredPixelCandidateEvaluation, ...]:
    prototype_records = _require_split(dataset_manifest.records, "prototype")
    benign_records = _require_split(dataset_manifest.records, "benign_calibration")
    rejection_records = _require_split(dataset_manifest.records, "rejection_calibration")
    prototype_map = build_registered_pixel_prototypes(
        dataset_manifest=dataset_manifest,
        observations=observations,
    )
    provisional = RegisteredPixelCalibration(
        threshold=2.0,
        ambiguity_margin=0.0,
        quantile=0.0,
        registration_config_digest=registration_config.digest,
        prototype_digest=_records_digest(prototype_records),
        benign_calibration_digest=_records_digest(benign_records),
        rejection_calibration_digest=_records_digest(rejection_records),
        source_scope=source_scope,
        policy_artifact_id=dataset_manifest.policy_artifact_id,
    )
    rank_provider = RegisteredPixelAddressProvider(
        prototypes=prototype_map,
        calibration=provisional,
        registration_config=registration_config,
        provider_id=_provider_id(registration_config=registration_config, calibration=provisional),
    )

    benign_distances = []
    benign_margins = []
    benign_rankings = []
    for record in benign_records:
        if capture_ids is not None:
            capture_ids.add(record.observation_id)
        best, second, raw_best = rank_provider._rank(observations[record.observation_id])
        benign_rankings.append(
            RegisteredPixelObservationRanking(
                observation_id=record.observation_id,
                best=best,
                second=second,
                raw_best=raw_best,
                expected_row_id=record.row_id,
                expected_action_id=record.action_id,
                expected_disposition=EXPECTED_ACCEPT,
            )
        )
        if best.row_id == str(record.row_id):
            benign_distances.append(float(best.distance))
            benign_margins.append(
                float("inf") if second is None else float(second.distance - best.distance)
            )
    rejection_rankings = []
    for record in rejection_records:
        if capture_ids is not None:
            capture_ids.add(record.observation_id)
        best, second, raw_best = rank_provider._rank(observations[record.observation_id])
        rejection_rankings.append(
            RegisteredPixelObservationRanking(
                observation_id=record.observation_id,
                best=best,
                second=second,
                raw_best=raw_best,
                expected_row_id=record.row_id,
                expected_action_id=record.action_id,
                expected_disposition=EXPECTED_REJECT,
            )
        )
    if not benign_distances:
        raise VPMValidationError("registered local baseline calibration found no correct benign rankings")

    evaluations = []
    for quantile in tuple(float(value) for value in quantiles):
        threshold = _conservative_distance_quantile(benign_distances, quantile)
        finite_margins = [value for value in benign_margins if np.isfinite(value)]
        ambiguity_margin = (
            _conservative_distance_quantile(finite_margins, quantile)
            if finite_margins
            else 0.0
        )
        calibration = RegisteredPixelCalibration(
            threshold=float(threshold),
            ambiguity_margin=float(ambiguity_margin),
            quantile=float(quantile),
            registration_config_digest=registration_config.digest,
            prototype_digest=_records_digest(prototype_records),
            benign_calibration_digest=_records_digest(benign_records),
            rejection_calibration_digest=_records_digest(rejection_records),
            source_scope=source_scope,
            policy_artifact_id=dataset_manifest.policy_artifact_id,
        )
        provider = RegisteredPixelAddressProvider(
            prototypes=prototype_map,
            calibration=calibration,
            registration_config=registration_config,
            provider_id=_provider_id(registration_config=registration_config, calibration=calibration),
        )
        benign_result = _evaluate_cached_rankings(
            rankings=benign_rankings,
            calibration=calibration,
            provider=provider,
            split_name="benign_calibration",
        )
        rejection_result = _evaluate_cached_rankings(
            rankings=rejection_rankings,
            calibration=calibration,
            provider=provider,
            split_name="rejection_calibration",
        )
        reasons = []
        if rejection_result.metrics.false_accept_count > 0:
            reasons.append("distinguishable_false_acceptance")
        if benign_result.metrics.conflicting_action_error_count > 0:
            reasons.append("conflicting_action_acceptance")
        evaluations.append(
            RegisteredPixelCandidateEvaluation(
                quantile=float(quantile),
                threshold=float(threshold),
                ambiguity_margin=float(ambiguity_margin),
                feasible=(len(reasons) == 0),
                infeasible_reasons=tuple(reasons),
                calibration=calibration,
                benign_result=benign_result,
                rejection_result=rejection_result,
            )
        )
    return tuple(evaluations)


def select_registered_pixel_candidate(
    *,
    dataset_manifest: VisualDatasetManifest,
    registration_config: RegistrationConfig,
    candidates: Sequence[RegisteredPixelCandidateEvaluation],
    source_scope: str,
) -> RegisteredPixelSelectionArtifact:
    candidate_items = tuple(candidates)
    if not candidate_items:
        raise VPMValidationError("registered local baseline selection requires candidates")
    feasible = tuple(candidate for candidate in candidate_items if candidate.feasible)
    def key(candidate: RegisteredPixelCandidateEvaluation) -> Tuple[float, float, float, float, float, float]:
        benign = candidate.benign_result.metrics
        accepted_precision = (
            -1.0
            if benign.accepted_benign_row_correctness is None
            else float(benign.accepted_benign_row_correctness)
        )
        coverage = float(benign.accepted_benign_count) / float(benign.false_reject_opportunities or 1)
        recall = float(benign.correct_row_count) / float(benign.false_reject_opportunities or 1)
        raw_rank = float(benign.top1_benign_row_accuracy)
        return (
            accepted_precision,
            coverage,
            recall,
            raw_rank,
            -float(candidate.threshold),
            -float(candidate.quantile),
        )
    selected = max(feasible, key=key) if feasible else None
    prototype_records = _require_split(dataset_manifest.records, "prototype")
    benign_records = _require_split(dataset_manifest.records, "benign_calibration")
    rejection_records = _require_split(dataset_manifest.records, "rejection_calibration")
    return RegisteredPixelSelectionArtifact(
        registration_config_digest=registration_config.digest,
        prototype_digest=_records_digest(prototype_records),
        benign_calibration_digest=_records_digest(benign_records),
        rejection_calibration_digest=_records_digest(rejection_records),
        candidate_grid_digest=_sha256_json([candidate.to_dict() for candidate in candidate_items]),
        selection_status=("selected_operating_point" if selected is not None else "no_feasible_operating_point"),
        selected_quantile=(None if selected is None else float(selected.quantile)),
        selected_threshold=(None if selected is None else float(selected.threshold)),
        selected_ambiguity_margin=(None if selected is None else float(selected.ambiguity_margin)),
        selected_calibration_digest=(None if selected is None else selected.calibration.digest),
        selection_rule=(
            "Among feasible candidates maximize accepted exact-row precision, "
            "then benign accepted coverage, then accepted exact-row recall, "
            "then raw benign exact-row ranking accuracy, then prefer the more "
            "conservative distance threshold, then deterministic quantile ordering."
        ),
        source_scope=source_scope,
        policy_artifact_id=dataset_manifest.policy_artifact_id,
        candidates=candidate_items,
        metadata={
            "candidate_quantiles": [float(candidate.quantile) for candidate in candidate_items],
        },
    )


def build_registered_pixel_provider(
    *,
    dataset_manifest: VisualDatasetManifest,
    observations: Mapping[str, ImageObservation],
    registration_config: RegistrationConfig,
    calibration: RegisteredPixelCalibration,
) -> RegisteredPixelAddressProvider:
    prototype_map = build_registered_pixel_prototypes(
        dataset_manifest=dataset_manifest,
        observations=observations,
    )
    return RegisteredPixelAddressProvider(
        prototypes=prototype_map,
        calibration=calibration,
        registration_config=registration_config,
        provider_id=_provider_id(registration_config=registration_config, calibration=calibration),
    )


def _evaluate_cached_rankings(
    *,
    rankings: Sequence[RegisteredPixelObservationRanking],
    calibration: RegisteredPixelCalibration,
    provider: RegisteredPixelAddressProvider,
    split_name: str,
) -> BenchmarkSystemResult:
    accepted_count = 0
    rejected_count = 0
    correct_row_count = 0
    correct_action_count = 0
    conflicting_action_error_count = 0
    false_accept_count = 0
    false_accept_opportunities = 0
    false_reject_count = 0
    false_reject_opportunities = 0
    top1_correct_row_count = 0
    top1_correct_action_count = 0
    for ranking in rankings:
        margin = (
            float("inf")
            if ranking.second is None
            else float(ranking.second.distance - ranking.best.distance)
        )
        accepted = (
            float(ranking.best.distance) <= float(calibration.threshold) + 1e-12
            and margin + 1e-12 >= float(calibration.ambiguity_margin)
        )
        if ranking.expected_disposition == EXPECTED_ACCEPT:
            false_reject_opportunities += 1
            top1_correct_row_count += int(ranking.best.row_id == ranking.expected_row_id)
            top1_correct_action_count += int(ranking.best.action_id == ranking.expected_action_id)
            if accepted:
                accepted_count += 1
                correct_row_count += int(ranking.best.row_id == ranking.expected_row_id)
                correct_action_count += int(ranking.best.action_id == ranking.expected_action_id)
                conflicting_action_error_count += int(ranking.best.action_id != ranking.expected_action_id)
            else:
                rejected_count += 1
                false_reject_count += 1
        else:
            false_accept_opportunities += 1
            if accepted:
                accepted_count += 1
                false_accept_count += 1
            else:
                rejected_count += 1
    metrics = VisualBenchmarkMetrics(
        evaluation_count=len(rankings),
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        correct_row_count=correct_row_count,
        correct_action_count=correct_action_count,
        conflicting_action_error_count=conflicting_action_error_count,
        false_accept_count=false_accept_count,
        false_accept_opportunities=false_accept_opportunities,
        false_reject_count=false_reject_count,
        false_reject_opportunities=false_reject_opportunities,
        top1_correct_row_count=top1_correct_row_count,
        top1_correct_action_count=top1_correct_action_count,
    )
    return BenchmarkSystemResult(
        system_id="R1",
        system_name="registered_local_normalized_pixels",
        contract_digest=provider.contract().digest,
        metrics=metrics,
        notes={"evaluated_split": split_name, "cached_rankings": len(rankings)},
    )
