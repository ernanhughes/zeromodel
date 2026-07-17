"""Result contracts for rejection-sensitive visual-address benchmarks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import VPMValidationError
from .visual_dataset import (
    VISUAL_BENCHMARK_REPORT_VERSION,
    _canonical_json_bytes,
    _freeze_json,
    _sha256_json,
    _thaw_json,
    _validate_rate,
)


@dataclass(frozen=True)
class VisualBenchmarkMetrics:
    """Counts with explicit denominators for rejection-sensitive metrics."""

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

    def validate(self) -> None:
        values = {name: int(value) for name, value in self.to_counts_dict().items()}
        if any(value < 0 for value in values.values()):
            raise VPMValidationError("visual benchmark counts cannot be negative")
        if self.accepted_count + self.rejected_count != self.evaluation_count:
            raise VPMValidationError(
                "accepted_count + rejected_count must equal evaluation_count"
            )
        for name in (
            "correct_row_count",
            "correct_action_count",
            "conflicting_action_error_count",
        ):
            if values[name] > self.evaluation_count:
                raise VPMValidationError("%s cannot exceed evaluation_count" % name)
        if self.false_accept_count > self.false_accept_opportunities:
            raise VPMValidationError(
                "false_accept_count cannot exceed its opportunities"
            )
        if self.false_reject_count > self.false_reject_opportunities:
            raise VPMValidationError(
                "false_reject_count cannot exceed its opportunities"
            )

    @property
    def row_accuracy(self) -> float:
        return self._rate(self.correct_row_count, self.evaluation_count)

    @property
    def action_accuracy(self) -> float:
        return self._rate(self.correct_action_count, self.evaluation_count)

    @property
    def false_acceptance_rate(self) -> float:
        return self._rate(self.false_accept_count, self.false_accept_opportunities)

    @property
    def false_rejection_rate(self) -> float:
        return self._rate(self.false_reject_count, self.false_reject_opportunities)

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
        }

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        result: Dict[str, Any] = self.to_counts_dict()
        result.update(
            {
                "row_accuracy": self.row_accuracy,
                "action_accuracy": self.action_accuracy,
                "false_acceptance_rate": self.false_acceptance_rate,
                "false_rejection_rate": self.false_rejection_rate,
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
