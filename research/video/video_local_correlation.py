"""Deterministic local-correlation visual addressing for bounded video research."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.observation.visual_address import (
    ImageObservation,
    VisualAddressContract,
    VisualAddressDecision,
)
from zeromodel.vision.visual_registration import RegistrationConfig, register_integer_translation


VIDEO_LOCAL_CORRELATION_PROVIDER_VERSION = "zeromodel-video-local-correlation-provider/v1"
VIDEO_LOCAL_CORRELATION_CALIBRATION_VERSION = "zeromodel-video-local-correlation-calibration/v1"
VIDEO_LOCAL_CORRELATION_REGION_SPEC_VERSION = "zeromodel-video-local-correlation-region-spec/v1"
VIDEO_LOCAL_CORRELATION_EVIDENCE_VERSION = "zeromodel-video-local-correlation-evidence/v1"
VIDEO_LOCAL_CORRELATION_SELECTION_VERSION = "zeromodel-video-local-correlation-selection/v1"


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
        raise VPMValidationError("local-correlation values must be JSON-serializable") from exc


def _json_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _freeze(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _crop(array: np.ndarray, *, top: int, left: int, height: int, width: int) -> np.ndarray:
    return np.ascontiguousarray(array[top : top + height, left : left + width], dtype=np.uint8)


@dataclass(frozen=True)
class LocalRegionSpec:
    region_id: str
    top: int
    left: int
    height: int
    width: int
    weight: float
    registration_config: RegistrationConfig
    minimum_visible_fraction: float = 1.0
    critical: bool = False
    version: str = VIDEO_LOCAL_CORRELATION_REGION_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_LOCAL_CORRELATION_REGION_SPEC_VERSION:
            raise VPMValidationError("unsupported local-correlation region spec version")
        if not str(self.region_id):
            raise VPMValidationError("region_id cannot be empty")
        for name in ("top", "left", "height", "width"):
            if int(getattr(self, name)) < 0:
                raise VPMValidationError("%s must be non-negative" % name)
        if int(self.height) <= 0 or int(self.width) <= 0:
            raise VPMValidationError("region height and width must be positive")
        if not np.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise VPMValidationError("region weight must be finite and positive")
        if not np.isfinite(float(self.minimum_visible_fraction)) or not (
            0.0 < float(self.minimum_visible_fraction) <= 1.0
        ):
            raise VPMValidationError("minimum_visible_fraction must be in (0, 1]")

    @property
    def digest(self) -> str:
        return _json_digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "region_id": self.region_id,
            "top": int(self.top),
            "left": int(self.left),
            "height": int(self.height),
            "width": int(self.width),
            "weight": float(self.weight),
            "registration_config": self.registration_config.to_dict(),
            "minimum_visible_fraction": float(self.minimum_visible_fraction),
            "critical": bool(self.critical),
        }


@dataclass(frozen=True)
class LocalCorrelationCalibration:
    winner_threshold: float
    runner_up_margin: float
    conflicting_action_margin: float
    minimum_visible_fraction: float
    region_spec_digest: str
    prototype_digest: str
    benign_calibration_digest: str
    rejection_calibration_digest: str
    policy_artifact_id: str
    source_scope: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VIDEO_LOCAL_CORRELATION_CALIBRATION_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_LOCAL_CORRELATION_CALIBRATION_VERSION:
            raise VPMValidationError("unsupported local-correlation calibration version")
        for name in ("winner_threshold", "runner_up_margin", "conflicting_action_margin", "minimum_visible_fraction"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise VPMValidationError("%s must be finite" % name)
        if self.winner_threshold < 0.0:
            raise VPMValidationError("winner_threshold must be non-negative")
        if self.runner_up_margin < 0.0 or self.conflicting_action_margin < 0.0:
            raise VPMValidationError("margins must be non-negative")
        if not (0.0 < self.minimum_visible_fraction <= 1.0):
            raise VPMValidationError("minimum_visible_fraction must be in (0, 1]")

    @property
    def digest(self) -> str:
        return _json_digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "winner_threshold": float(self.winner_threshold),
            "runner_up_margin": float(self.runner_up_margin),
            "conflicting_action_margin": float(self.conflicting_action_margin),
            "minimum_visible_fraction": float(self.minimum_visible_fraction),
            "region_spec_digest": self.region_spec_digest,
            "prototype_digest": self.prototype_digest,
            "benign_calibration_digest": self.benign_calibration_digest,
            "rejection_calibration_digest": self.rejection_calibration_digest,
            "policy_artifact_id": self.policy_artifact_id,
            "source_scope": self.source_scope,
            "metadata": _freeze(self.metadata),
        }


@dataclass(frozen=True)
class RegionCorrelationEvidence:
    region_id: str
    weight: float
    distance: float
    score: float
    dx: int
    dy: int
    overlap_fraction: float
    valid_pixel_count: int
    registration_succeeded: bool
    rejection_reason: Optional[str]
    visible_fraction: float
    critical: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "weight": float(self.weight),
            "distance": float(self.distance),
            "score": float(self.score),
            "dx": int(self.dx),
            "dy": int(self.dy),
            "overlap_fraction": float(self.overlap_fraction),
            "valid_pixel_count": int(self.valid_pixel_count),
            "registration_succeeded": bool(self.registration_succeeded),
            "rejection_reason": self.rejection_reason,
            "visible_fraction": float(self.visible_fraction),
            "critical": bool(self.critical),
        }


@dataclass(frozen=True)
class LocalCorrelationCandidate:
    row_id: str
    action_id: str
    prototype_observation_id: str
    observation_digest: str
    total_distance: float
    visible_fraction: float
    critical_evidence_present: bool
    region_evidence: Tuple[RegionCorrelationEvidence, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "action_id": self.action_id,
            "prototype_observation_id": self.prototype_observation_id,
            "observation_digest": self.observation_digest,
            "total_distance": float(self.total_distance),
            "visible_fraction": float(self.visible_fraction),
            "critical_evidence_present": bool(self.critical_evidence_present),
            "region_evidence": [item.to_dict() for item in self.region_evidence],
        }


@dataclass(frozen=True)
class LocalCorrelationEvaluation:
    observation_id: str
    best: LocalCorrelationCandidate
    second: Optional[LocalCorrelationCandidate]
    nearest_conflicting_action: Optional[LocalCorrelationCandidate]
    expected_row_id: Optional[str]
    expected_action_id: Optional[str]
    expected_disposition: str


@dataclass(frozen=True)
class LocalCorrelationCandidateEvaluation:
    winner_threshold: float
    runner_up_margin: float
    conflicting_action_margin: float
    minimum_visible_fraction: float
    feasible: bool
    infeasible_reasons: Tuple[str, ...]
    calibration: LocalCorrelationCalibration
    benign_result: Any
    rejection_result: Any

    def to_dict(self) -> Dict[str, Any]:
        benign_metrics = self.benign_result.metrics
        rejection_metrics = self.rejection_result.metrics
        return {
            "winner_threshold": float(self.winner_threshold),
            "runner_up_margin": float(self.runner_up_margin),
            "conflicting_action_margin": float(self.conflicting_action_margin),
            "minimum_visible_fraction": float(self.minimum_visible_fraction),
            "feasible": bool(self.feasible),
            "infeasible_reasons": list(self.infeasible_reasons),
            "accepted_benign_count": int(benign_metrics.accepted_benign_count),
            "accepted_exact_row_precision": (
                None if benign_metrics.accepted_benign_count == 0 else float(benign_metrics.accepted_benign_row_correctness)
            ),
            "accepted_action_precision": (
                None if benign_metrics.accepted_benign_count == 0 else float(benign_metrics.accepted_benign_action_correctness)
            ),
            "benign_coverage": float(benign_metrics.accepted_benign_count / float(benign_metrics.false_reject_opportunities or 1)),
            "accepted_exact_row_recall": float(benign_metrics.benign_row_accuracy),
            "raw_exact_row_accuracy": float(benign_metrics.top1_benign_row_accuracy),
            "false_accepts": int(rejection_metrics.false_accept_count),
            "conflicting_action_accepts": int(benign_metrics.conflicting_action_error_count),
            "false_rejects": int(benign_metrics.false_reject_count),
            "calibration_artifact_id": self.calibration.digest,
            "benign_result": self.benign_result.to_dict(),
            "rejection_result": self.rejection_result.to_dict(),
        }


@dataclass(frozen=True)
class LocalCorrelationSelectionArtifact:
    selection_status: str
    selected_calibration_digest: Optional[str]
    selected_winner_threshold: Optional[float]
    selected_runner_up_margin: Optional[float]
    selected_conflicting_action_margin: Optional[float]
    selected_minimum_visible_fraction: Optional[float]
    selection_rule: str
    candidate_grid_digest: str
    policy_artifact_id: str
    source_scope: str
    candidates: Tuple[LocalCorrelationCandidateEvaluation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VIDEO_LOCAL_CORRELATION_SELECTION_VERSION

    @property
    def digest(self) -> str:
        return _json_digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "selection_status": self.selection_status,
            "selected_calibration_digest": self.selected_calibration_digest,
            "selected_winner_threshold": self.selected_winner_threshold,
            "selected_runner_up_margin": self.selected_runner_up_margin,
            "selected_conflicting_action_margin": self.selected_conflicting_action_margin,
            "selected_minimum_visible_fraction": self.selected_minimum_visible_fraction,
            "selection_rule": self.selection_rule,
            "candidate_grid_digest": self.candidate_grid_digest,
            "policy_artifact_id": self.policy_artifact_id,
            "source_scope": self.source_scope,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": _freeze(self.metadata),
        }


class LocalCorrelationVideoAddressProvider:
    def __init__(
        self,
        *,
        prototypes: Mapping[str, Tuple[str, str, str, ImageObservation]],
        calibration: LocalCorrelationCalibration,
        regions: Sequence[LocalRegionSpec],
    ) -> None:
        items = []
        for observation_id in sorted(prototypes):
            row_id, action_id, observation_digest, observation = prototypes[observation_id]
            items.append((str(row_id), str(action_id), str(observation_id), str(observation_digest), observation))
        if not items:
            raise VPMValidationError("local-correlation provider requires prototypes")
        self._items = tuple(items)
        self._calibration = calibration
        self._regions = tuple(regions)
        self._region_by_id = {region.region_id: region for region in self._regions}
        self._provider_id = _json_digest(
            {
                "provider_version": VIDEO_LOCAL_CORRELATION_PROVIDER_VERSION,
                "calibration_digest": calibration.digest,
                "regions": [region.to_dict() for region in self._regions],
            }
        )
        first_shape = items[0][4].pixels.shape
        if len(first_shape) != 2:
            raise VPMValidationError("local-correlation provider currently requires grayscale observations")
        for _row_id, _action_id, _observation_id, _digest, observation in self._items:
            if observation.pixels.shape != first_shape:
                raise VPMValidationError("prototype geometry must remain constant")
        self._shape = first_shape
        for region in self._regions:
            if region.top + region.height > self._shape[0] or region.left + region.width > self._shape[1]:
                raise VPMValidationError("region extends beyond prototype geometry")
        self._cache: Dict[str, Tuple[LocalCorrelationCandidate, ...]] = {}

    def contract(self) -> VisualAddressContract:
        return VisualAddressContract(
            provider_kind="deterministic_local_correlation",
            provider_version=VIDEO_LOCAL_CORRELATION_PROVIDER_VERSION,
            score_semantics="distance",
            observation_spec_digest=self._calibration.region_spec_digest,
            representation_spec_digest=self._calibration.region_spec_digest,
            address_artifact_id=self._provider_id,
            calibration_artifact_id=self._calibration.digest,
            policy_artifact_id=self._calibration.policy_artifact_id,
            source_scope=self._calibration.source_scope,
            replay_contract="exact_decision",
            metadata={
                "winner_threshold": float(self._calibration.winner_threshold),
                "runner_up_margin": float(self._calibration.runner_up_margin),
                "conflicting_action_margin": float(self._calibration.conflicting_action_margin),
                "minimum_visible_fraction": float(self._calibration.minimum_visible_fraction),
                "region_ids": [region.region_id for region in self._regions],
            },
        )

    def _cache_key(self, observation: ImageObservation) -> str:
        return _json_digest(
            {
                "observation_digest": observation.raw_digest,
                "source_id": observation.source_id,
                "shape": list(observation.pixels.shape),
                "provider_id": self._provider_id,
                "calibration_digest": self._calibration.digest,
                "region_spec_digest": self._calibration.region_spec_digest,
            }
        )

    def _region_evidence(
        self,
        *,
        prototype_pixels: np.ndarray,
        observation_pixels: np.ndarray,
        region: LocalRegionSpec,
    ) -> RegionCorrelationEvidence:
        prototype_crop = _crop(
            prototype_pixels,
            top=region.top,
            left=region.left,
            height=region.height,
            width=region.width,
        )
        observation_crop = _crop(
            observation_pixels,
            top=region.top,
            left=region.left,
            height=region.height,
            width=region.width,
        )
        registration = register_integer_translation(
            prototype_crop,
            observation_crop,
            config=region.registration_config,
        )
        visible_fraction = float(registration.overlap_fraction)
        return RegionCorrelationEvidence(
            region_id=region.region_id,
            weight=float(region.weight),
            distance=float(registration.distance_after),
            score=float(registration.score_after),
            dx=int(registration.dx),
            dy=int(registration.dy),
            overlap_fraction=float(registration.overlap_fraction),
            valid_pixel_count=int(registration.valid_pixel_count),
            registration_succeeded=bool(registration.registration_succeeded),
            rejection_reason=registration.rejection_reason,
            visible_fraction=visible_fraction,
            critical=bool(region.critical),
        )

    def _rank(self, observation: ImageObservation) -> Tuple[LocalCorrelationCandidate, ...]:
        if observation.pixels.shape != self._shape:
            raise VPMValidationError("local-correlation observation geometry does not match prototypes")
        key = self._cache_key(observation)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        candidates = []
        for row_id, action_id, prototype_observation_id, observation_digest, prototype_observation in self._items:
            region_evidence = tuple(
                self._region_evidence(
                    prototype_pixels=prototype_observation.pixels,
                    observation_pixels=observation.pixels,
                    region=region,
                )
                for region in self._regions
            )
            total_weight = sum(item.weight for item in region_evidence)
            total_distance = sum(item.weight * item.distance for item in region_evidence) / float(total_weight or 1.0)
            visible_fraction = min(item.visible_fraction for item in region_evidence)
            critical_evidence_present = all(
                (not item.critical) or item.visible_fraction + 1e-12 >= self._region_by_id[item.region_id].minimum_visible_fraction
                for item in region_evidence
            )
            candidates.append(
                LocalCorrelationCandidate(
                    row_id=row_id,
                    action_id=action_id,
                    prototype_observation_id=prototype_observation_id,
                    observation_digest=observation_digest,
                    total_distance=float(total_distance),
                    visible_fraction=float(visible_fraction),
                    critical_evidence_present=critical_evidence_present,
                    region_evidence=region_evidence,
                )
            )
        ranked = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    float(candidate.total_distance),
                    -float(candidate.visible_fraction),
                    candidate.row_id,
                    candidate.prototype_observation_id,
                ),
            )
        )
        self._cache[key] = ranked
        return ranked

    def read(self, observation: ImageObservation) -> VisualAddressDecision:
        ranked = self._rank(observation)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        nearest_conflicting = next(
            (candidate for candidate in ranked[1:] if candidate.action_id != best.action_id),
            None,
        )
        margin = float("inf") if second is None else float(second.total_distance - best.total_distance)
        conflicting_margin = (
            float("inf")
            if nearest_conflicting is None
            else float(nearest_conflicting.total_distance - best.total_distance)
        )
        accepted_by = []
        reasons = []
        if best.total_distance > self._calibration.winner_threshold + 1e-12:
            reasons.append("winner_threshold")
        else:
            accepted_by.append("winner_threshold")
        if best.visible_fraction + 1e-12 < self._calibration.minimum_visible_fraction:
            reasons.append("minimum_visible_fraction")
        else:
            accepted_by.append("minimum_visible_fraction")
        if not best.critical_evidence_present:
            reasons.append("critical_region_occluded")
        else:
            accepted_by.append("critical_region_occluded")
        if margin + 1e-12 < self._calibration.runner_up_margin:
            reasons.append("runner_up_margin")
        else:
            accepted_by.append("runner_up_margin")
        if conflicting_margin + 1e-12 < self._calibration.conflicting_action_margin:
            reasons.append("conflicting_action_margin")
        else:
            accepted_by.append("conflicting_action_margin")
        accepted = len(reasons) == 0
        reason = "accepted" if accepted else reasons[0]
        trace = {
            "version": VIDEO_LOCAL_CORRELATION_EVIDENCE_VERSION,
            "provider_id": self._provider_id,
            "region_spec_digest": self._calibration.region_spec_digest,
            "winner_threshold": float(self._calibration.winner_threshold),
            "runner_up_margin_threshold": float(self._calibration.runner_up_margin),
            "conflicting_action_margin_threshold": float(self._calibration.conflicting_action_margin),
            "minimum_visible_fraction_threshold": float(self._calibration.minimum_visible_fraction),
            "best_candidate": best.to_dict(),
            "second_candidate": None if second is None else second.to_dict(),
            "nearest_conflicting_action_candidate": (
                None if nearest_conflicting is None else nearest_conflicting.to_dict()
            ),
            "winner_margin": None if second is None else float(margin),
            "conflicting_action_margin": None if nearest_conflicting is None else float(conflicting_margin),
            "rejection_reasons": list(reasons),
            "raw_top1_row_id": best.row_id,
            "raw_top1_action_id": best.action_id,
            "candidate_count": len(ranked),
        }
        return VisualAddressDecision(
            accepted=accepted,
            reason=reason,
            observation_digest=observation.raw_digest,
            representation_digest=key if (key := self._cache_key(observation)) else self._cache_key(observation),
            provider_kind="deterministic_local_correlation",
            provider_version=VIDEO_LOCAL_CORRELATION_PROVIDER_VERSION,
            score_semantics="distance",
            address_artifact_id=self._provider_id,
            calibration_artifact_id=self._calibration.digest,
            policy_artifact_id=self._calibration.policy_artifact_id,
            nearest_row_id=best.row_id,
            nearest_score=float(-best.total_distance),
            second_row_id=None if second is None else second.row_id,
            second_score=None if second is None else float(-second.total_distance),
            ambiguity_measure=None if second is None else float(margin),
            local_evidence_score=float(-best.total_distance),
            visible_evidence_fraction=float(best.visible_fraction),
            critical_evidence_present=bool(best.critical_evidence_present),
            matched_row_id=(best.row_id if accepted else None),
            exact_match=False,
            accepted_by=tuple(accepted_by),
            trace=trace,
        )


def local_region_digest(regions: Sequence[LocalRegionSpec]) -> str:
    return _json_digest([region.to_dict() for region in regions])


def build_local_correlation_prototypes(
    *,
    dataset_manifest: Any,
    observations: Mapping[str, ImageObservation],
) -> Mapping[str, Tuple[str, str, str, ImageObservation]]:
    prototypes = {}
    for record in dataset_manifest.records:
        if record.split != "prototype":
            continue
        if record.row_id is None or record.action_id is None:
            raise VPMValidationError("prototype records require row and action ids")
        prototypes[record.observation_id] = (
            str(record.row_id),
            str(record.action_id),
            str(record.observation_digest),
            observations[record.observation_id],
        )
    if not prototypes:
        raise VPMValidationError("local-correlation prototypes are empty")
    return prototypes


def _records_digest(records: Iterable[Any]) -> str:
    return _json_digest([record.to_dict() for record in records])


def _require_split(records: Sequence[Any], split: str) -> Tuple[Any, ...]:
    values = tuple(record for record in records if record.split == split)
    if not values:
        raise VPMValidationError("required split is empty: %s" % split)
    return values


def build_local_correlation_candidates(
    *,
    dataset_manifest: Any,
    observations: Mapping[str, ImageObservation],
    policy_lookup: Any,
    regions: Sequence[LocalRegionSpec],
    winner_thresholds: Sequence[float],
    runner_up_margins: Sequence[float],
    conflicting_action_margins: Sequence[float],
    minimum_visible_fractions: Sequence[float],
    source_scope: str,
) -> Tuple[LocalCorrelationCandidateEvaluation, ...]:
    prototype_records = _require_split(dataset_manifest.records, "prototype")
    benign_records = _require_split(dataset_manifest.records, "benign_calibration")
    rejection_records = _require_split(dataset_manifest.records, "rejection_calibration")
    prototypes = build_local_correlation_prototypes(
        dataset_manifest=dataset_manifest,
        observations=observations,
    )
    region_digest = local_region_digest(regions)
    provider_cache = {}
    benign_rankings = []
    rejection_rankings = []
    provisional = LocalCorrelationCalibration(
        winner_threshold=max(float(value) for value in winner_thresholds),
        runner_up_margin=min(float(value) for value in runner_up_margins),
        conflicting_action_margin=min(float(value) for value in conflicting_action_margins),
        minimum_visible_fraction=min(float(value) for value in minimum_visible_fractions),
        region_spec_digest=region_digest,
        prototype_digest=_records_digest(prototype_records),
        benign_calibration_digest=_records_digest(benign_records),
        rejection_calibration_digest=_records_digest(rejection_records),
        policy_artifact_id=dataset_manifest.policy_artifact_id,
        source_scope=source_scope,
    )
    ranking_provider = LocalCorrelationVideoAddressProvider(
        prototypes=prototypes,
        calibration=provisional,
        regions=regions,
    )
    for record in benign_records:
        ranked = ranking_provider._rank(observations[record.observation_id])
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        nearest_conflicting = next((candidate for candidate in ranked[1:] if candidate.action_id != best.action_id), None)
        benign_rankings.append(
            LocalCorrelationEvaluation(
                observation_id=record.observation_id,
                best=best,
                second=second,
                nearest_conflicting_action=nearest_conflicting,
                expected_row_id=record.row_id,
                expected_action_id=record.action_id,
                expected_disposition=str(record.evaluation_role or record.metadata.get("expected_disposition")),
            )
        )
    for record in rejection_records:
        ranked = ranking_provider._rank(observations[record.observation_id])
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        nearest_conflicting = next((candidate for candidate in ranked[1:] if candidate.action_id != best.action_id), None)
        rejection_rankings.append(
            LocalCorrelationEvaluation(
                observation_id=record.observation_id,
                best=best,
                second=second,
                nearest_conflicting_action=nearest_conflicting,
                expected_row_id=record.row_id,
                expected_action_id=record.action_id,
                expected_disposition=str(record.evaluation_role or record.metadata.get("expected_disposition")),
            )
        )
    evaluations = []
    for winner_threshold in tuple(float(value) for value in winner_thresholds):
        for runner_up_margin in tuple(float(value) for value in runner_up_margins):
            for conflicting_action_margin in tuple(float(value) for value in conflicting_action_margins):
                for minimum_visible_fraction in tuple(float(value) for value in minimum_visible_fractions):
                    calibration = LocalCorrelationCalibration(
                        winner_threshold=winner_threshold,
                        runner_up_margin=runner_up_margin,
                        conflicting_action_margin=conflicting_action_margin,
                        minimum_visible_fraction=minimum_visible_fraction,
                        region_spec_digest=region_digest,
                        prototype_digest=_records_digest(prototype_records),
                        benign_calibration_digest=_records_digest(benign_records),
                        rejection_calibration_digest=_records_digest(rejection_records),
                        policy_artifact_id=dataset_manifest.policy_artifact_id,
                        source_scope=source_scope,
                    )
                    provider = provider_cache.get(calibration.digest)
                    if provider is None:
                        provider = LocalCorrelationVideoAddressProvider(
                            prototypes=prototypes,
                            calibration=calibration,
                            regions=regions,
                        )
                        provider_cache[calibration.digest] = provider
                    benign_result = _evaluate_rankings(
                        rankings=tuple(benign_rankings),
                        provider=provider,
                        split_name="benign_calibration",
                    )
                    rejection_result = _evaluate_rankings(
                        rankings=tuple(rejection_rankings),
                        provider=provider,
                        split_name="rejection_calibration",
                    )
                    reasons = []
                    if rejection_result.metrics.false_accept_count > 0:
                        reasons.append("distinguishable_false_acceptance")
                    if benign_result.metrics.conflicting_action_error_count > 0:
                        reasons.append("conflicting_action_acceptance")
                    if benign_result.metrics.accepted_benign_count <= 0:
                        reasons.append("zero_benign_coverage")
                    evaluations.append(
                        LocalCorrelationCandidateEvaluation(
                            winner_threshold=winner_threshold,
                            runner_up_margin=runner_up_margin,
                            conflicting_action_margin=conflicting_action_margin,
                            minimum_visible_fraction=minimum_visible_fraction,
                            feasible=len(reasons) == 0,
                            infeasible_reasons=tuple(reasons),
                            calibration=calibration,
                            benign_result=benign_result,
                            rejection_result=rejection_result,
                        )
                    )
    return tuple(evaluations)


def _evaluate_rankings(
    *,
    rankings: Sequence[LocalCorrelationEvaluation],
    provider: LocalCorrelationVideoAddressProvider,
    split_name: str,
) -> Any:
    from research.benchmarks.visual_benchmark import BenchmarkSystemResult, VisualBenchmarkMetrics

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
        expected_accept = ranking.expected_disposition == "expected_accept"
        top1_correct_row_count += int(expected_accept and ranking.best.row_id == ranking.expected_row_id)
        top1_correct_action_count += int(expected_accept and ranking.best.action_id == ranking.expected_action_id)
        second_margin = float("inf") if ranking.second is None else float(ranking.second.total_distance - ranking.best.total_distance)
        conflicting_margin = (
            float("inf")
            if ranking.nearest_conflicting_action is None
            else float(ranking.nearest_conflicting_action.total_distance - ranking.best.total_distance)
        )
        accepted = (
            ranking.best.total_distance <= provider._calibration.winner_threshold + 1e-12
            and second_margin + 1e-12 >= provider._calibration.runner_up_margin
            and conflicting_margin + 1e-12 >= provider._calibration.conflicting_action_margin
            and ranking.best.visible_fraction + 1e-12 >= provider._calibration.minimum_visible_fraction
            and ranking.best.critical_evidence_present
        )
        if expected_accept:
            false_reject_opportunities += 1
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
        system_id="V2",
        system_name="deterministic_local_correlation",
        contract_digest=provider.contract().digest,
        metrics=metrics,
        notes={"evaluated_split": split_name, "cached_rankings": len(rankings)},
    )


def select_local_correlation_candidate(
    *,
    dataset_manifest: Any,
    candidates: Sequence[LocalCorrelationCandidateEvaluation],
    source_scope: str,
) -> LocalCorrelationSelectionArtifact:
    candidate_items = tuple(candidates)
    feasible = tuple(candidate for candidate in candidate_items if candidate.feasible)

    def key(candidate: LocalCorrelationCandidateEvaluation) -> Tuple[float, float, float, float, float, float, float]:
        benign = candidate.benign_result.metrics
        accepted_precision = (
            -1.0 if benign.accepted_benign_row_correctness is None else float(benign.accepted_benign_row_correctness)
        )
        coverage = float(benign.accepted_benign_count) / float(benign.false_reject_opportunities or 1)
        recall = float(benign.correct_row_count) / float(benign.false_reject_opportunities or 1)
        accepted_action_precision = (
            -1.0 if benign.accepted_benign_action_correctness is None else float(benign.accepted_benign_action_correctness)
        )
        raw_rank = float(benign.top1_benign_row_accuracy)
        return (
            accepted_precision,
            coverage,
            recall,
            accepted_action_precision,
            raw_rank,
            -float(candidate.winner_threshold),
            float(candidate.runner_up_margin + candidate.conflicting_action_margin + candidate.minimum_visible_fraction),
        )

    selected = max(feasible, key=key) if feasible else None
    return LocalCorrelationSelectionArtifact(
        selection_status="selected_operating_point" if selected is not None else "no_feasible_operating_point",
        selected_calibration_digest=None if selected is None else selected.calibration.digest,
        selected_winner_threshold=None if selected is None else float(selected.winner_threshold),
        selected_runner_up_margin=None if selected is None else float(selected.runner_up_margin),
        selected_conflicting_action_margin=None if selected is None else float(selected.conflicting_action_margin),
        selected_minimum_visible_fraction=None if selected is None else float(selected.minimum_visible_fraction),
        selection_rule=(
            "Among feasible candidates maximize accepted exact-row precision, then benign accepted coverage, "
            "then accepted exact-row recall, then accepted action precision, then raw benign exact-row ranking accuracy, "
            "then prefer the stricter winner threshold, then the stricter combined margin and visibility thresholds."
        ),
        candidate_grid_digest=_json_digest([candidate.to_dict() for candidate in candidate_items]),
        policy_artifact_id=dataset_manifest.policy_artifact_id,
        source_scope=source_scope,
        candidates=candidate_items,
        metadata={},
    )


__all__ = [
    "LocalCorrelationCalibration",
    "LocalCorrelationCandidateEvaluation",
    "LocalCorrelationSelectionArtifact",
    "LocalCorrelationVideoAddressProvider",
    "LocalRegionSpec",
    "VIDEO_LOCAL_CORRELATION_CALIBRATION_VERSION",
    "VIDEO_LOCAL_CORRELATION_PROVIDER_VERSION",
    "build_local_correlation_candidates",
    "build_local_correlation_prototypes",
    "local_region_digest",
    "select_local_correlation_candidate",
]
