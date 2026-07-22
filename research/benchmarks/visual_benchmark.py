"""Result contracts for rejection-sensitive visual-address benchmarks."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.vision.visual_dataset import (
    VISUAL_BENCHMARK_REPORT_VERSION,
    _canonical_json_bytes,
    _freeze_json,
    _sha256_json,
    _thaw_json,
    _validate_rate,
)


WILSON_Z_95 = 1.959963984540054


def wilson_score_interval(
    successes: int,
    opportunities: int,
    *,
    z: float = WILSON_Z_95,
) -> Tuple[float, float]:
    """Return a two-sided Wilson score interval for one binomial rate.

    The interval is an observation-level descriptive interval. Benchmark
    observations can share source states and corruption families, so research
    reports should use paired and clustered analyses for system comparisons.
    """

    count = int(successes)
    total = int(opportunities)
    if count < 0 or total < 0 or count > total:
        raise VPMValidationError("Wilson interval counts are inconsistent")
    if total == 0:
        return (0.0, 0.0)
    z_value = float(z)
    if not np.isfinite(z_value) or z_value <= 0.0:
        raise VPMValidationError("Wilson interval z must be positive and finite")
    probability = float(count) / float(total)
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / float(total)
    centre = (
        probability + z_squared / (2.0 * float(total))
    ) / denominator
    half_width = (
        z_value
        * sqrt(
            probability * (1.0 - probability) / float(total)
            + z_squared / (4.0 * float(total) * float(total))
        )
        / denominator
    )
    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


@dataclass(frozen=True)
class VisualBenchmarkMetrics:
    """Counts with explicit denominators for rejection-sensitive metrics.

    ``row_accuracy`` and ``action_accuracy`` retain the original whole-scored-
    evaluation denominator for backward compatibility. New research should use
    the explicit benign, accepted-benign, and top-1 metrics below.
    """

    evaluation_count: int
    accepted_count: int
    rejected_count: int
    correct_row_count: int
    correct_action_count: int
    conflicting_action_error_count: int
    false_accept_count: int
    false_accept_opportunities: int
    false_reject_count: int
    false_reject_opportunities: int
    top1_correct_row_count: int = 0
    top1_correct_action_count: int = 0

    def validate(self) -> None:
        values = {name: int(value) for name, value in self.to_counts_dict().items()}
        if any(value < 0 for value in values.values()):
            raise VPMValidationError("visual benchmark counts cannot be negative")
        if self.accepted_count + self.rejected_count != self.evaluation_count:
            raise VPMValidationError(
                "accepted_count + rejected_count must equal evaluation_count"
            )
        if (
            self.false_accept_opportunities + self.false_reject_opportunities
            != self.evaluation_count
        ):
            raise VPMValidationError(
                "false-accept and false-reject opportunities must partition "
                "evaluation_count"
            )
        for name in (
            "correct_row_count",
            "correct_action_count",
            "conflicting_action_error_count",
        ):
            if values[name] > self.false_reject_opportunities:
                raise VPMValidationError(
                    "%s cannot exceed benign opportunities" % name
                )
        for name in (
            "top1_correct_row_count",
            "top1_correct_action_count",
        ):
            if values[name] > self.false_reject_opportunities:
                raise VPMValidationError(
                    "%s cannot exceed benign opportunities" % name
                )
        if self.false_accept_count > self.false_accept_opportunities:
            raise VPMValidationError(
                "false_accept_count cannot exceed its opportunities"
            )
        if self.false_reject_count > self.false_reject_opportunities:
            raise VPMValidationError(
                "false_reject_count cannot exceed its opportunities"
            )
        if self.accepted_count != self.accepted_benign_count + self.false_accept_count:
            raise VPMValidationError(
                "accepted_count must equal accepted benign plus false accepts"
            )
        if self.rejected_count != self.false_reject_count + self.correct_reject_count:
            raise VPMValidationError(
                "rejected_count must equal false rejects plus correct rejects"
            )
        if self.correct_row_count > self.accepted_benign_count:
            raise VPMValidationError(
                "correct_row_count cannot exceed accepted benign observations"
            )
        if self.correct_action_count > self.accepted_benign_count:
            raise VPMValidationError(
                "correct_action_count cannot exceed accepted benign observations"
            )
        if self.conflicting_action_error_count > self.accepted_benign_count:
            raise VPMValidationError(
                "conflicting_action_error_count cannot exceed accepted benign "
                "observations"
            )

    @property
    def row_accuracy(self) -> float:
        """Legacy whole-scored-evaluation row rate."""

        return self._rate(self.correct_row_count, self.evaluation_count)

    @property
    def action_accuracy(self) -> float:
        """Legacy whole-scored-evaluation action rate."""

        return self._rate(self.correct_action_count, self.evaluation_count)

    @property
    def benign_row_accuracy(self) -> float:
        return self._rate(
            self.correct_row_count,
            self.false_reject_opportunities,
        )

    @property
    def benign_action_accuracy(self) -> float:
        return self._rate(
            self.correct_action_count,
            self.false_reject_opportunities,
        )

    @property
    def top1_benign_row_accuracy(self) -> float:
        return self._rate(
            self.top1_correct_row_count,
            self.false_reject_opportunities,
        )

    @property
    def top1_benign_action_accuracy(self) -> float:
        return self._rate(
            self.top1_correct_action_count,
            self.false_reject_opportunities,
        )

    @property
    def accepted_benign_count(self) -> int:
        return int(self.false_reject_opportunities - self.false_reject_count)

    @property
    def accepted_benign_row_correctness(self) -> float:
        return self._rate(self.correct_row_count, self.accepted_benign_count)

    @property
    def accepted_benign_action_correctness(self) -> float:
        return self._rate(self.correct_action_count, self.accepted_benign_count)

    @property
    def correct_reject_count(self) -> int:
        return int(self.false_accept_opportunities - self.false_accept_count)

    @property
    def false_acceptance_rate(self) -> float:
        return self._rate(self.false_accept_count, self.false_accept_opportunities)

    @property
    def false_rejection_rate(self) -> float:
        return self._rate(self.false_reject_count, self.false_reject_opportunities)

    @property
    def correct_disposition_rate(self) -> float:
        return self._rate(
            self.accepted_benign_count + self.correct_reject_count,
            self.evaluation_count,
        )

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return 0.0 if denominator == 0 else float(numerator) / float(denominator)

    def to_counts_dict(self) -> Dict[str, int]:
        return {
            "evaluation_count": int(self.evaluation_count),
            "accepted_count": int(self.accepted_count),
            "rejected_count": int(self.rejected_count),
            "correct_row_count": int(self.correct_row_count),
            "correct_action_count": int(self.correct_action_count),
            "conflicting_action_error_count": int(
                self.conflicting_action_error_count
            ),
            "false_accept_count": int(self.false_accept_count),
            "false_accept_opportunities": int(
                self.false_accept_opportunities
            ),
            "false_reject_count": int(self.false_reject_count),
            "false_reject_opportunities": int(
                self.false_reject_opportunities
            ),
            "top1_correct_row_count": int(self.top1_correct_row_count),
            "top1_correct_action_count": int(self.top1_correct_action_count),
        }

    def confidence_intervals_95(self) -> Dict[str, Optional[Tuple[float, float]]]:
        accepted_row_interval: Optional[Tuple[float, float]]
        accepted_action_interval: Optional[Tuple[float, float]]
        if self.accepted_benign_count == 0:
            accepted_row_interval = None
            accepted_action_interval = None
        else:
            accepted_row_interval = wilson_score_interval(
                self.correct_row_count,
                self.accepted_benign_count,
            )
            accepted_action_interval = wilson_score_interval(
                self.correct_action_count,
                self.accepted_benign_count,
            )
        return {
            "benign_row_accuracy": wilson_score_interval(
                self.correct_row_count,
                self.false_reject_opportunities,
            ),
            "benign_action_accuracy": wilson_score_interval(
                self.correct_action_count,
                self.false_reject_opportunities,
            ),
            "top1_benign_row_accuracy": wilson_score_interval(
                self.top1_correct_row_count,
                self.false_reject_opportunities,
            ),
            "top1_benign_action_accuracy": wilson_score_interval(
                self.top1_correct_action_count,
                self.false_reject_opportunities,
            ),
            "accepted_benign_row_correctness": accepted_row_interval,
            "accepted_benign_action_correctness": accepted_action_interval,
            "false_acceptance_rate": wilson_score_interval(
                self.false_accept_count,
                self.false_accept_opportunities,
            ),
            "false_rejection_rate": wilson_score_interval(
                self.false_reject_count,
                self.false_reject_opportunities,
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        result: Dict[str, Any] = self.to_counts_dict()
        result.update(
            {
                "row_accuracy": self.row_accuracy,
                "action_accuracy": self.action_accuracy,
                "benign_row_accuracy": self.benign_row_accuracy,
                "benign_action_accuracy": self.benign_action_accuracy,
                "top1_benign_row_accuracy": self.top1_benign_row_accuracy,
                "top1_benign_action_accuracy": self.top1_benign_action_accuracy,
                "accepted_benign_count": self.accepted_benign_count,
                "accepted_benign_row_correctness": (
                    None
                    if self.accepted_benign_count == 0
                    else self.accepted_benign_row_correctness
                ),
                "accepted_benign_action_correctness": (
                    None
                    if self.accepted_benign_count == 0
                    else self.accepted_benign_action_correctness
                ),
                "correct_reject_count": self.correct_reject_count,
                "false_acceptance_rate": self.false_acceptance_rate,
                "false_rejection_rate": self.false_rejection_rate,
                "correct_disposition_rate": self.correct_disposition_rate,
                "confidence_intervals_95": {
                    name: (
                        None
                        if bounds is None
                        else [float(bounds[0]), float(bounds[1])]
                    )
                    for name, bounds in self.confidence_intervals_95().items()
                },
                "confidence_interval_note": (
                    "Observation-level Wilson intervals are descriptive. "
                    "Use paired and state/family-clustered analyses for system "
                    "comparisons."
                ),
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisualBenchmarkMetrics":
        metrics = cls(
            evaluation_count=int(data["evaluation_count"]),
            accepted_count=int(data["accepted_count"]),
            rejected_count=int(data["rejected_count"]),
            correct_row_count=int(data["correct_row_count"]),
            correct_action_count=int(data["correct_action_count"]),
            conflicting_action_error_count=int(
                data["conflicting_action_error_count"]
            ),
            false_accept_count=int(data["false_accept_count"]),
            false_accept_opportunities=int(
                data["false_accept_opportunities"]
            ),
            false_reject_count=int(data["false_reject_count"]),
            false_reject_opportunities=int(
                data["false_reject_opportunities"]
            ),
            # Legacy reports did not preserve pre-rejection top-1 outcomes.
            # The accepted-correct counts are the only defensible lower-bound
            # fallback during deserialization.
            top1_correct_row_count=int(
                data.get("top1_correct_row_count", data["correct_row_count"])
            ),
            top1_correct_action_count=int(
                data.get("top1_correct_action_count", data["correct_action_count"])
            ),
        )
        metrics.validate()
        return metrics


@dataclass(frozen=True)
class BenchmarkSystemResult:
    system_id: str
    system_name: str
    contract_digest: str
    metrics: VisualBenchmarkMetrics
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        system_id: str,
        system_name: str,
        contract_digest: str,
        metrics: VisualBenchmarkMetrics,
        notes: Optional[Mapping[str, Any]] = None,
    ) -> None:
        object.__setattr__(self, "system_id", str(system_id))
        object.__setattr__(self, "system_name", str(system_name))
        object.__setattr__(self, "contract_digest", str(contract_digest))
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "notes", _freeze_json(notes or {}))
        self.validate()

    def validate(self) -> None:
        if not self.system_id or not self.system_name or not self.contract_digest:
            raise VPMValidationError(
                "benchmark system id, name, and contract digest are required"
            )
        self.metrics.validate()
        _canonical_json_bytes(self.notes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "system_name": self.system_name,
            "contract_digest": self.contract_digest,
            "metrics": self.metrics.to_dict(),
            "notes": _thaw_json(self.notes),
        }


@dataclass(frozen=True)
class GovernanceAuditResult:
    system_id: str
    question_id: str
    answered: bool
    fidelity_score: float
    effort_minutes: float
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        system_id: str,
        question_id: str,
        answered: bool,
        fidelity_score: float,
        effort_minutes: float,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        object.__setattr__(self, "system_id", str(system_id))
        object.__setattr__(self, "question_id", str(question_id))
        object.__setattr__(self, "answered", bool(answered))
        object.__setattr__(self, "fidelity_score", float(fidelity_score))
        object.__setattr__(self, "effort_minutes", float(effort_minutes))
        object.__setattr__(self, "evidence", _freeze_json(evidence or {}))
        self.validate()

    def validate(self) -> None:
        if not self.system_id or not self.question_id:
            raise VPMValidationError(
                "governance audit system_id and question_id are required"
            )
        _validate_rate("fidelity_score", self.fidelity_score)
        if not np.isfinite(self.effort_minutes) or self.effort_minutes < 0.0:
            raise VPMValidationError(
                "governance audit effort_minutes must be finite and non-negative"
            )
        _canonical_json_bytes(self.evidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "question_id": self.question_id,
            "answered": self.answered,
            "fidelity_score": self.fidelity_score,
            "effort_minutes": self.effort_minutes,
            "evidence": _thaw_json(self.evidence),
        }


@dataclass(frozen=True)
class VisualBenchmarkReport:
    """Research or validated report over declared conventional baselines."""

    dataset_manifest_digest: str
    systems: Tuple[BenchmarkSystemResult, ...]
    governance_audit: Tuple[GovernanceAuditResult, ...] = ()
    declared_false_acceptance_target: Optional[float] = None
    validation_status: str = "research"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VISUAL_BENCHMARK_REPORT_VERSION

    def __init__(
        self,
        *,
        dataset_manifest_digest: str,
        systems: Sequence[BenchmarkSystemResult],
        governance_audit: Sequence[GovernanceAuditResult] = (),
        declared_false_acceptance_target: Optional[float] = None,
        validation_status: str = "research",
        metadata: Optional[Mapping[str, Any]] = None,
        version: str = VISUAL_BENCHMARK_REPORT_VERSION,
    ) -> None:
        object.__setattr__(
            self,
            "dataset_manifest_digest",
            str(dataset_manifest_digest),
        )
        object.__setattr__(self, "systems", tuple(systems))
        object.__setattr__(self, "governance_audit", tuple(governance_audit))
        object.__setattr__(
            self,
            "declared_false_acceptance_target",
            None
            if declared_false_acceptance_target is None
            else float(declared_false_acceptance_target),
        )
        object.__setattr__(self, "validation_status", str(validation_status))
        object.__setattr__(self, "metadata", _freeze_json(metadata or {}))
        object.__setattr__(self, "version", str(version))
        self.validate()

    def validate(self) -> None:
        if self.version != VISUAL_BENCHMARK_REPORT_VERSION:
            raise VPMValidationError(
                "unsupported visual benchmark report version: %r" % self.version
            )
        if not self.dataset_manifest_digest:
            raise VPMValidationError(
                "benchmark report requires dataset_manifest_digest"
            )
        if not self.systems:
            raise VPMValidationError(
                "benchmark report requires at least one system result"
            )
        system_ids = [result.system_id for result in self.systems]
        if len(set(system_ids)) != len(system_ids):
            raise VPMValidationError(
                "benchmark report system ids must be unique"
            )
        for result in self.systems:
            result.validate()
        known = set(system_ids)
        for audit in self.governance_audit:
            audit.validate()
            if audit.system_id not in known:
                raise VPMValidationError(
                    "governance audit references unknown system_id: %s"
                    % audit.system_id
                )
        if self.declared_false_acceptance_target is not None:
            _validate_rate(
                "declared_false_acceptance_target",
                self.declared_false_acceptance_target,
            )
        if self.validation_status not in {"research", "validated"}:
            raise VPMValidationError(
                "validation_status must be 'research' or 'validated'"
            )
        _canonical_json_bytes(self.metadata)

    @property
    def digest(self) -> str:
        return _sha256_json(self.to_dict())

    @property
    def deployment_permitted(self) -> bool:
        return self.validation_status == "validated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "systems": [result.to_dict() for result in self.systems],
            "governance_audit": [
                result.to_dict() for result in self.governance_audit
            ],
            "declared_false_acceptance_target": (
                self.declared_false_acceptance_target
            ),
            "validation_status": self.validation_status,
            "deployment_permitted": self.deployment_permitted,
            "metadata": _thaw_json(self.metadata),
        }
