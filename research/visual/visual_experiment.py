"""Executable evaluation helpers for held-out visual-address benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.observation.visual_address import ImageObservation, VisualAddressDecision, VisualAddressProvider
from research.benchmarks.visual_benchmark import (
    BenchmarkSystemResult,
    VisualBenchmarkMetrics,
    VisualBenchmarkReport,
)
from zeromodel.vision.visual_dataset import VisualDatasetManifest, VisualExampleRecord
from zeromodel.vision.visual_encoder import FrozenVisualEncoder


EXPECTED_ACCEPT = "expected_accept"
EXPECTED_REJECT = "expected_reject"
IMPOSSIBILITY_CONTROL = "information_theoretic_control"
_EXPECTED_DISPOSITIONS = {
    EXPECTED_ACCEPT,
    EXPECTED_REJECT,
    IMPOSSIBILITY_CONTROL,
}


@dataclass(frozen=True)
class VisualEvaluationTrace:
    observation_id: str
    family_id: str
    split: str
    expected_disposition: str
    expected_accept: Optional[bool]
    expected_row_id: Optional[str]
    expected_action_id: Optional[str]
    predicted_row_id: Optional[str]
    predicted_action_id: Optional[str]
    decision: VisualAddressDecision
    top1_row_id: Optional[str] = None
    top1_action_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.expected_disposition not in _EXPECTED_DISPOSITIONS:
            raise VPMValidationError("unsupported expected visual disposition")
        required = {
            EXPECTED_ACCEPT: True,
            EXPECTED_REJECT: False,
            IMPOSSIBILITY_CONTROL: None,
        }[self.expected_disposition]
        if self.expected_accept is not required:
            raise VPMValidationError(
                "expected_accept does not match expected_disposition"
            )
        if self.predicted_row_id is not None and not self.decision.accepted:
            raise VPMValidationError(
                "predicted_row_id requires an accepted visual decision"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "family_id": self.family_id,
            "split": self.split,
            "expected_disposition": self.expected_disposition,
            "expected_accept": self.expected_accept,
            "expected_row_id": self.expected_row_id,
            "expected_action_id": self.expected_action_id,
            "predicted_row_id": self.predicted_row_id,
            "predicted_action_id": self.predicted_action_id,
            "top1_row_id": self.top1_row_id,
            "top1_action_id": self.top1_action_id,
            "decision": self.decision.to_dict(),
        }


def encode_observations(
    encoder: FrozenVisualEncoder,
    observation_ids: Sequence[str],
    observations: Mapping[str, ImageObservation],
    *,
    batch_size: int = 32,
) -> Mapping[str, np.ndarray]:
    """Encode a declared observation order into immutable float32 vectors."""

    ids = tuple(str(value) for value in observation_ids)
    if not ids:
        raise VPMValidationError("embedding extraction requires observation ids")
    if len(set(ids)) != len(ids):
        raise VPMValidationError("embedding extraction observation ids must be unique")
    if batch_size <= 0:
        raise VPMValidationError("embedding batch_size must be positive")
    missing = sorted(set(ids) - set(observations))
    if missing:
        raise VPMValidationError(
            "embedding extraction is missing observations: %s" % ", ".join(missing)
        )

    result: Dict[str, np.ndarray] = {}
    expected_dimension = encoder.manifest().output_dimension
    for start in range(0, len(ids), int(batch_size)):
        batch_ids = ids[start : start + int(batch_size)]
        matrix = np.asarray(
            encoder.encode_batch(tuple(observations[item] for item in batch_ids)),
            dtype=np.float32,
        )
        if matrix.shape != (len(batch_ids), expected_dimension):
            raise VPMValidationError("encoder output shape violates its manifest")
        if not np.isfinite(matrix).all():
            raise VPMValidationError("encoder output must be finite")
        for observation_id, vector in zip(batch_ids, matrix):
            owned = np.ascontiguousarray(vector, dtype=np.float32)
            owned.flags.writeable = False
            result[observation_id] = owned
    return result


def records_for_split(
    manifest: VisualDatasetManifest,
    split: str,
) -> Tuple[VisualExampleRecord, ...]:
    return tuple(record for record in manifest.records if record.split == str(split))


def vectors_for_records(
    records: Sequence[VisualExampleRecord],
    vectors_by_observation_id: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    items = tuple(records)
    if not items:
        raise VPMValidationError("vector selection requires at least one record")
    missing = sorted(
        {record.observation_id for record in items} - set(vectors_by_observation_id)
    )
    if missing:
        raise VPMValidationError(
            "vector selection is missing observations: %s" % ", ".join(missing)
        )
    if any(record.row_id is None or record.action_id is None for record in items):
        raise VPMValidationError("prototype/calibration vector records require row and action")
    matrix = np.ascontiguousarray(
        [vectors_by_observation_id[record.observation_id] for record in items],
        dtype=np.float32,
    )
    return (
        matrix,
        tuple(str(record.row_id) for record in items),
        tuple(str(record.action_id) for record in items),
        tuple(record.observation_id for record in items),
    )


def _expected_disposition(
    record: VisualExampleRecord,
    manifest: VisualDatasetManifest,
) -> str:
    explicit = record.evaluation_role
    if explicit is None:
        explicit = record.metadata.get("evaluation_role")
    if explicit is not None:
        value = str(explicit)
        if value not in _EXPECTED_DISPOSITIONS:
            raise VPMValidationError(
                "unsupported evaluation_role for %s: %s"
                % (record.observation_id, value)
            )
        return value
    if record.split == "ood":
        return EXPECTED_REJECT
    if record.split == "rejection_calibration":
        return EXPECTED_REJECT
    if record.split == "final_evaluation":
        if record.evaluation_role is None:
            raise VPMValidationError(
                "final_evaluation records must declare evaluation_role"
            )
        return str(record.evaluation_role)
    if record.split == "benign_calibration":
        return EXPECTED_ACCEPT
    family_by_id = {family.family_id: family for family in manifest.families}
    return (
        EXPECTED_REJECT
        if family_by_id[record.family_id].critical_evidence_removed
        else EXPECTED_ACCEPT
    )


def _family_counter() -> Dict[str, int]:
    return {
        "observation_count": 0,
        "scored_evaluation_count": 0,
        "expected_accept_count": 0,
        "expected_reject_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "scored_accepted_count": 0,
        "scored_rejected_count": 0,
        "correct_row_count": 0,
        "correct_action_count": 0,
        "top1_correct_row_count": 0,
        "top1_correct_action_count": 0,
        "conflicting_action_error_count": 0,
        "false_accept_count": 0,
        "false_reject_count": 0,
        "correct_disposition_count": 0,
        "impossibility_control_count": 0,
        "impossibility_control_accepted_count": 0,
        "impossibility_control_rejected_count": 0,
    }


def _increment(
    target: Dict[str, int],
    *,
    disposition: str,
    accepted: bool,
    correct_row: bool,
    correct_action: bool,
    top1_correct_row: bool,
    top1_correct_action: bool,
    conflicting_error: bool,
) -> None:
    target["observation_count"] += 1
    target["accepted_count"] += int(accepted)
    target["rejected_count"] += int(not accepted)
    if disposition == IMPOSSIBILITY_CONTROL:
        target["impossibility_control_count"] += 1
        target["impossibility_control_accepted_count"] += int(accepted)
        target["impossibility_control_rejected_count"] += int(not accepted)
        return

    expected_accept = disposition == EXPECTED_ACCEPT
    target["scored_evaluation_count"] += 1
    target["expected_accept_count"] += int(expected_accept)
    target["expected_reject_count"] += int(not expected_accept)
    target["scored_accepted_count"] += int(accepted)
    target["scored_rejected_count"] += int(not accepted)
    target["correct_row_count"] += int(correct_row)
    target["correct_action_count"] += int(correct_action)
    target["top1_correct_row_count"] += int(top1_correct_row)
    target["top1_correct_action_count"] += int(top1_correct_action)
    target["conflicting_action_error_count"] += int(conflicting_error)
    target["false_accept_count"] += int(not expected_accept and accepted)
    target["false_reject_count"] += int(expected_accept and not accepted)
    target["correct_disposition_count"] += int(
        (expected_accept and accepted) or (not expected_accept and not accepted)
    )


def evaluate_visual_provider(
    *,
    provider: VisualAddressProvider,
    dataset_manifest: VisualDatasetManifest,
    observations: Mapping[str, ImageObservation],
    policy_lookup: VPMPolicyLookup,
    system_id: str,
    system_name: str,
    splits: Sequence[str] = ("test", "ood"),
    include_traces: bool = False,
) -> Tuple[BenchmarkSystemResult, Tuple[VisualEvaluationTrace, ...]]:
    """Evaluate benign, rejection, and information-theoretic control cases.

    The evaluator records both the accepted prediction and the provider's raw
    top-1 row before rejection. This separates ranking quality from calibration.
    """

    contract = provider.contract()
    if contract.policy_artifact_id != dataset_manifest.policy_artifact_id:
        raise VPMValidationError("provider and dataset target different policy artifacts")
    if policy_lookup.artifact.artifact_id != dataset_manifest.policy_artifact_id:
        raise VPMValidationError("policy lookup and dataset target different artifacts")

    selected_splits = {str(value) for value in splits}
    records = tuple(
        record for record in dataset_manifest.records if record.split in selected_splits
    )
    if not records:
        raise VPMValidationError("benchmark evaluation selected no records")
    missing = sorted({record.observation_id for record in records} - set(observations))
    if missing:
        raise VPMValidationError(
            "benchmark observations are missing: %s" % ", ".join(missing)
        )

    family_counts: Dict[str, Dict[str, int]] = {}
    traces = []
    totals = _family_counter()

    for record in records:
        disposition = _expected_disposition(record, dataset_manifest)
        expected_accept = (
            True
            if disposition == EXPECTED_ACCEPT
            else False if disposition == EXPECTED_REJECT else None
        )
        decision = provider.read(observations[record.observation_id])

        top1_row = decision.nearest_row_id
        top1_action = (
            policy_lookup.choose(str(top1_row)) if top1_row is not None else None
        )
        predicted_row = decision.matched_row_id if decision.accepted else None
        predicted_action = (
            policy_lookup.choose(str(predicted_row)) if predicted_row is not None else None
        )

        correct_row = bool(
            disposition == EXPECTED_ACCEPT
            and predicted_row is not None
            and predicted_row == record.row_id
        )
        correct_action = bool(
            disposition == EXPECTED_ACCEPT
            and predicted_action is not None
            and predicted_action == record.action_id
        )
        top1_correct_row = bool(
            disposition == EXPECTED_ACCEPT
            and top1_row is not None
            and top1_row == record.row_id
        )
        top1_correct_action = bool(
            disposition == EXPECTED_ACCEPT
            and top1_action is not None
            and top1_action == record.action_id
        )
        conflicting_error = bool(
            disposition == EXPECTED_ACCEPT
            and decision.accepted
            and predicted_action is not None
            and predicted_action != record.action_id
        )

        counters = family_counts.setdefault(record.family_id, _family_counter())
        for target in (totals, counters):
            _increment(
                target,
                disposition=disposition,
                accepted=decision.accepted,
                correct_row=correct_row,
                correct_action=correct_action,
                top1_correct_row=top1_correct_row,
                top1_correct_action=top1_correct_action,
                conflicting_error=conflicting_error,
            )

        if include_traces:
            traces.append(
                VisualEvaluationTrace(
                    observation_id=record.observation_id,
                    family_id=record.family_id,
                    split=record.split,
                    expected_disposition=disposition,
                    expected_accept=expected_accept,
                    expected_row_id=record.row_id,
                    expected_action_id=record.action_id,
                    predicted_row_id=predicted_row,
                    predicted_action_id=predicted_action,
                    top1_row_id=top1_row,
                    top1_action_id=top1_action,
                    decision=decision,
                )
            )

    if totals["scored_evaluation_count"] <= 0:
        raise VPMValidationError("benchmark evaluation contains no scored observations")
    metrics = VisualBenchmarkMetrics(
        evaluation_count=totals["scored_evaluation_count"],
        accepted_count=totals["scored_accepted_count"],
        rejected_count=totals["scored_rejected_count"],
        correct_row_count=totals["correct_row_count"],
        correct_action_count=totals["correct_action_count"],
        conflicting_action_error_count=totals["conflicting_action_error_count"],
        false_accept_count=totals["false_accept_count"],
        false_accept_opportunities=totals["expected_reject_count"],
        false_reject_count=totals["false_reject_count"],
        false_reject_opportunities=totals["expected_accept_count"],
        top1_correct_row_count=totals["top1_correct_row_count"],
        top1_correct_action_count=totals["top1_correct_action_count"],
    )
    notes = {
        "observation_count_including_controls": totals["observation_count"],
        "scored_evaluation_count": totals["scored_evaluation_count"],
        "expected_accept_count": totals["expected_accept_count"],
        "expected_reject_count": totals["expected_reject_count"],
        "correct_disposition_count": totals["correct_disposition_count"],
        "correct_disposition_rate": metrics.correct_disposition_rate,
        # Compatibility copies. These values are now also first-class metrics.
        "benign_row_accuracy": metrics.benign_row_accuracy,
        "benign_action_accuracy": metrics.benign_action_accuracy,
        "top1_benign_row_accuracy": metrics.top1_benign_row_accuracy,
        "top1_benign_action_accuracy": metrics.top1_benign_action_accuracy,
        "accepted_benign_row_correctness": (
            metrics.accepted_benign_row_correctness
        ),
        "accepted_benign_action_correctness": (
            metrics.accepted_benign_action_correctness
        ),
        "impossibility_control_count": totals["impossibility_control_count"],
        "impossibility_control_accepted_count": totals[
            "impossibility_control_accepted_count"
        ],
        "impossibility_control_rejected_count": totals[
            "impossibility_control_rejected_count"
        ],
        "impossibility_control_acceptance_rate": (
            float(totals["impossibility_control_accepted_count"])
            / float(totals["impossibility_control_count"])
            if totals["impossibility_control_count"]
            else 0.0
        ),
        "family_counts": family_counts,
        "evaluated_splits": sorted(selected_splits),
        "metric_semantics": {
            "accepted_prediction": (
                "predicted_row_id and predicted_action_id exist only when the "
                "provider accepts"
            ),
            "top1_prediction": (
                "top1_row_id and top1_action_id are recorded before rejection "
                "when the provider exposes a nearest row"
            ),
        },
    }
    result = BenchmarkSystemResult(
        system_id=str(system_id),
        system_name=str(system_name),
        contract_digest=contract.digest,
        metrics=metrics,
        notes=notes,
    )
    return result, tuple(traces)


def build_research_report(
    *,
    dataset_manifest: VisualDatasetManifest,
    system_results: Sequence[BenchmarkSystemResult],
    metadata: Optional[Mapping[str, Any]] = None,
) -> VisualBenchmarkReport:
    present_splits = {record.split for record in dataset_manifest.records}
    default_metadata = {
        "dataset_splits": sorted(present_splits),
        "dataset_digest": dataset_manifest.digest,
    }
    default_metadata.update(dict(metadata or {}))
    return VisualBenchmarkReport(
        dataset_manifest_digest=dataset_manifest.digest,
        systems=tuple(system_results),
        validation_status="research",
        metadata=default_metadata,
    )
