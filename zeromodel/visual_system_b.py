"""Deterministic System B adjudication over calibration-only operating points."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import VPMValidationError
from .visual_address import ImageObservation
from .visual_benchmark import BenchmarkSystemResult
from .visual_dataset import VisualDatasetManifest, VisualExampleRecord
from .visual_experiment import evaluate_visual_provider, records_for_split
from .visual_retrieval import (
    FrozenVectorAddressProvider,
    NormalizedPixelEncoder,
    VectorAddressIndex,
    VectorAddressBuild,
    build_vector_address,
)


SYSTEM_B_PROTOCOL_VERSION = "zeromodel-system-b-protocol/v1"
SYSTEM_B_SELECTION_VERSION = "zeromodel-system-b-selection/v1"


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
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
        raise VPMValidationError("system B values must be JSON-serializable") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def system_b_candidate_quantiles() -> Tuple[float, ...]:
    return (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _require_split(records: Sequence[VisualExampleRecord], split: str) -> Tuple[VisualExampleRecord, ...]:
    items = tuple(record for record in records if record.split == split)
    if not items:
        raise VPMValidationError("required split is empty: %s" % split)
    return items


def validate_selection_records(records: Sequence[VisualExampleRecord]) -> None:
    items = tuple(records)
    if not items:
        raise VPMValidationError("selection requires records")
    forbidden = sorted({record.observation_id for record in items if record.split == "final_evaluation"})
    if forbidden:
        raise VPMValidationError(
            "final_evaluation records cannot enter System B calibration selection: %s"
            % ", ".join(forbidden[:5])
        )


@dataclass(frozen=True)
class SystemBCandidateResult:
    quantile: float
    calibration_artifact_id: str
    benign_result: BenchmarkSystemResult
    rejection_result: BenchmarkSystemResult
    feasible: bool
    infeasible_reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantile": float(self.quantile),
            "calibration_artifact_id": self.calibration_artifact_id,
            "feasible": self.feasible,
            "infeasible_reasons": list(self.infeasible_reasons),
            "benign_result": self.benign_result.to_dict(),
            "rejection_result": self.rejection_result.to_dict(),
        }


@dataclass(frozen=True)
class SystemBSelectionArtifact:
    protocol_version: str
    system_id: str
    source_scope: str
    policy_artifact_id: str
    prototype_dataset_digest: str
    benign_calibration_digest: str
    rejection_calibration_digest: str
    candidate_grid_digest: str
    selection_rule: str
    selection_status: str
    selected_quantile: Optional[float]
    selected_calibration_artifact_id: Optional[str]
    candidates: Tuple[SystemBCandidateResult, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = SYSTEM_B_SELECTION_VERSION

    @property
    def digest(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "protocol_version": self.protocol_version,
            "system_id": self.system_id,
            "source_scope": self.source_scope,
            "policy_artifact_id": self.policy_artifact_id,
            "prototype_dataset_digest": self.prototype_dataset_digest,
            "benign_calibration_digest": self.benign_calibration_digest,
            "rejection_calibration_digest": self.rejection_calibration_digest,
            "candidate_grid_digest": self.candidate_grid_digest,
            "selection_rule": self.selection_rule,
            "selection_status": self.selection_status,
            "selected_quantile": self.selected_quantile,
            "selected_calibration_artifact_id": self.selected_calibration_artifact_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": _thaw(self.metadata),
        }


def _records_digest(records: Sequence[VisualExampleRecord]) -> str:
    payload = [record.to_dict() for record in records]
    return _sha256_json(payload)


def _candidate_sort_key(candidate: SystemBCandidateResult) -> Tuple[float, float, float, float]:
    benign = candidate.benign_result.metrics
    return (
        benign.accepted_benign_row_correctness,
        float(benign.accepted_benign_count) / float(benign.false_reject_opportunities or 1),
        benign.benign_row_accuracy,
        -float(candidate.quantile),
    )


def _candidate_feasibility(candidate: SystemBCandidateResult) -> Tuple[bool, Tuple[str, ...]]:
    reasons = []
    if candidate.rejection_result.metrics.false_accept_count > 0:
        reasons.append("distinguishable_false_acceptance")
    if candidate.benign_result.metrics.conflicting_action_error_count > 0:
        reasons.append("conflicting_action_acceptance")
    return (len(reasons) == 0, tuple(reasons))


def build_system_b_candidates(
    *,
    dataset_manifest: VisualDatasetManifest,
    observations: Mapping[str, ImageObservation],
    policy_lookup: Any,
    source_scope: str,
    quantiles: Sequence[float] = (),
) -> Tuple[SystemBCandidateResult, ...]:
    prototype_records = _require_split(dataset_manifest.records, "prototype")
    benign_records = _require_split(dataset_manifest.records, "benign_calibration")
    rejection_records = _require_split(dataset_manifest.records, "rejection_calibration")
    validate_selection_records(benign_records + rejection_records)

    encoder = NormalizedPixelEncoder(
        height=next(iter(observations.values())).pixels.shape[0],
        width=next(iter(observations.values())).pixels.shape[1],
    )
    all_ids = tuple(record.observation_id for record in dataset_manifest.records)
    from .visual_experiment import encode_observations, vectors_for_records

    vectors = encode_observations(encoder, all_ids, observations, batch_size=128)
    prototype = vectors_for_records(prototype_records, vectors)
    benign = vectors_for_records(benign_records, vectors)
    quantile_values = tuple(float(value) for value in (quantiles or system_b_candidate_quantiles()))
    results = []
    for quantile in quantile_values:
        build = build_vector_address(
            prototype_vectors=prototype[0],
            prototype_row_ids=prototype[1],
            prototype_action_ids=prototype[2],
            prototype_observation_ids=prototype[3],
            calibration_vectors=benign[0],
            calibration_row_ids=benign[1],
            calibration_action_ids=benign[2],
            calibration_observation_ids=benign[3],
            policy_artifact_id=dataset_manifest.policy_artifact_id,
            source_scope=source_scope,
            representation_spec_digest=encoder.manifest().manifest_id,
            encoder_manifest_id=encoder.manifest().manifest_id,
            strategy="medoid",
            calibration_quantile=quantile,
            deployment_status="research",
        )
        provider = FrozenVectorAddressProvider(encoder, VectorAddressIndex(build))
        benign_result, _ = evaluate_visual_provider(
            provider=provider,
            dataset_manifest=dataset_manifest,
            observations=observations,
            policy_lookup=policy_lookup,
            system_id="B",
            system_name="normalized_template_matching",
            splits=("benign_calibration",),
            include_traces=False,
        )
        rejection_result, _ = evaluate_visual_provider(
            provider=provider,
            dataset_manifest=dataset_manifest,
            observations=observations,
            policy_lookup=policy_lookup,
            system_id="B",
            system_name="normalized_template_matching",
            splits=("rejection_calibration",),
            include_traces=False,
        )
        feasible, reasons = _candidate_feasibility(
            SystemBCandidateResult(
                quantile=quantile,
                calibration_artifact_id=build.calibration.digest,
                benign_result=benign_result,
                rejection_result=rejection_result,
                feasible=False,
            )
        )
        results.append(
            SystemBCandidateResult(
                quantile=quantile,
                calibration_artifact_id=build.calibration.digest,
                benign_result=benign_result,
                rejection_result=rejection_result,
                feasible=feasible,
                infeasible_reasons=reasons,
            )
        )
    return tuple(results)


def select_system_b_operating_point(
    *,
    dataset_manifest: VisualDatasetManifest,
    candidates: Sequence[SystemBCandidateResult],
    metadata: Optional[Mapping[str, Any]] = None,
) -> SystemBSelectionArtifact:
    prototype_records = _require_split(dataset_manifest.records, "prototype")
    benign_records = _require_split(dataset_manifest.records, "benign_calibration")
    rejection_records = _require_split(dataset_manifest.records, "rejection_calibration")
    validate_selection_records(benign_records + rejection_records)
    candidate_items = tuple(candidates)
    if not candidate_items:
        raise VPMValidationError("System B selection requires candidates")
    feasible = tuple(candidate for candidate in candidate_items if candidate.feasible)
    selected = max(feasible, key=_candidate_sort_key) if feasible else None
    meta = dict(metadata or {})
    meta.setdefault("quantiles", list(system_b_candidate_quantiles()))
    meta.setdefault("selection_order", [
        "zero distinguishable false accepts",
        "zero conflicting-action accepts",
        "maximize accepted exact-row precision",
        "maximize benign coverage",
        "maximize exact-row recall",
        "deterministic quantile ordering",
    ])
    return SystemBSelectionArtifact(
        protocol_version=SYSTEM_B_PROTOCOL_VERSION,
        system_id="B",
        source_scope=dataset_manifest.source_scope,
        policy_artifact_id=dataset_manifest.policy_artifact_id,
        prototype_dataset_digest=_records_digest(prototype_records),
        benign_calibration_digest=_records_digest(benign_records),
        rejection_calibration_digest=_records_digest(rejection_records),
        candidate_grid_digest=_sha256_json([candidate.to_dict() for candidate in candidate_items]),
        selection_rule=(
            "Reject candidates with any distinguishable false acceptance or any "
            "conflicting-action acceptance; among remaining candidates maximize "
            "accepted exact-row precision, then benign coverage, then exact-row recall, "
            "then deterministic quantile ordering."
        ),
        selection_status=(
            "selected_operating_point"
            if selected is not None
            else "no_feasible_operating_point"
        ),
        selected_quantile=(None if selected is None else selected.quantile),
        selected_calibration_artifact_id=(
            None if selected is None else selected.calibration_artifact_id
        ),
        candidates=candidate_items,
        metadata=meta,
    )


def build_system_b_provider(
    *,
    dataset_manifest: VisualDatasetManifest,
    observations: Mapping[str, ImageObservation],
    quantile: float,
) -> Tuple[VectorAddressBuild, FrozenVectorAddressProvider, NormalizedPixelEncoder]:
    encoder = NormalizedPixelEncoder(
        height=next(iter(observations.values())).pixels.shape[0],
        width=next(iter(observations.values())).pixels.shape[1],
    )
    from .visual_experiment import encode_observations, vectors_for_records

    all_ids = tuple(record.observation_id for record in dataset_manifest.records)
    vectors = encode_observations(encoder, all_ids, observations, batch_size=128)
    prototype = vectors_for_records(records_for_split(dataset_manifest, "prototype"), vectors)
    benign = vectors_for_records(records_for_split(dataset_manifest, "benign_calibration"), vectors)
    build = build_vector_address(
        prototype_vectors=prototype[0],
        prototype_row_ids=prototype[1],
        prototype_action_ids=prototype[2],
        prototype_observation_ids=prototype[3],
        calibration_vectors=benign[0],
        calibration_row_ids=benign[1],
        calibration_action_ids=benign[2],
        calibration_observation_ids=benign[3],
        policy_artifact_id=dataset_manifest.policy_artifact_id,
        source_scope=dataset_manifest.source_scope,
        representation_spec_digest=encoder.manifest().manifest_id,
        encoder_manifest_id=encoder.manifest().manifest_id,
        strategy="medoid",
        calibration_quantile=float(quantile),
        deployment_status="research",
    )
    return build, FrozenVectorAddressProvider(encoder, VectorAddressIndex(build)), encoder
