"""Wake-policy replay over stored Observer ledger evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.fixture import ObserverFixtureError
from zeromodel.observer.ledger import (
    ObserverTransitionLedgerEntryDTO,
    ObserverTransitionLedgerSnapshotDTO,
)

OBSERVER_WAKE_POLICY_VERSION: Final = "observer-wake-policy/1"
OBSERVER_WAKE_EVALUATION_VERSION: Final = "observer-wake-evaluation/1"
OBSERVER_WAKE_POLICY_REPLAY_VERSION: Final = "observer-wake-policy-replay/1"
OBSERVER_WAKE_POLICY_ABLATION_VERSION: Final = "observer-wake-policy-ablation/1"

TRIGGER_CODES: Final = frozenset(
    {
        "always",
        "comparison_wake",
        "contradiction",
        "inconclusive",
        "schema_mismatch",
        "missing_policy_evidence",
        "missing_hidden_state_evidence",
        "critical_action",
    }
)


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverFixtureError(f"{field_name} must be unique and sorted")


@dataclass(frozen=True)
class ObserverWakePolicyDTO:
    """Declarative wake policy evaluated over stored transition evidence."""

    wake_policy_id: str
    policy_name: str
    wake_on_every_transition: bool = False
    wake_on_comparison_wake: bool = False
    wake_on_contradiction: bool = False
    wake_on_inconclusive: bool = False
    wake_on_schema_mismatch: bool = False
    wake_on_missing_policy_evidence: bool = False
    wake_on_missing_hidden_state_evidence: bool = False
    wake_on_critical_action: bool = False
    critical_actions: tuple[str, ...] = ()
    version: str = OBSERVER_WAKE_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_WAKE_POLICY_VERSION:
            raise ObserverFixtureError("unsupported wake policy version")
        if not self.policy_name:
            raise ObserverFixtureError("policy_name must be non-empty")
        _ensure_sorted_unique(self.critical_actions, "critical_actions")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.wake_policy_id != expected_id:
            raise ObserverFixtureError(
                "wake_policy_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "critical_actions": list(self.critical_actions),
            "policy_name": self.policy_name,
            "version": self.version,
            "wake_on_comparison_wake": self.wake_on_comparison_wake,
            "wake_on_contradiction": self.wake_on_contradiction,
            "wake_on_critical_action": self.wake_on_critical_action,
            "wake_on_every_transition": self.wake_on_every_transition,
            "wake_on_inconclusive": self.wake_on_inconclusive,
            "wake_on_missing_hidden_state_evidence": (
                self.wake_on_missing_hidden_state_evidence
            ),
            "wake_on_missing_policy_evidence": self.wake_on_missing_policy_evidence,
            "wake_on_schema_mismatch": self.wake_on_schema_mismatch,
        }
        if include_id:
            payload["wake_policy_id"] = self.wake_policy_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        policy_name: str,
        wake_on_every_transition: bool = False,
        wake_on_comparison_wake: bool = False,
        wake_on_contradiction: bool = False,
        wake_on_inconclusive: bool = False,
        wake_on_schema_mismatch: bool = False,
        wake_on_missing_policy_evidence: bool = False,
        wake_on_missing_hidden_state_evidence: bool = False,
        wake_on_critical_action: bool = False,
        critical_actions: tuple[str, ...] = (),
    ) -> "ObserverWakePolicyDTO":
        critical_actions = tuple(sorted(critical_actions))
        payload = {
            "critical_actions": list(critical_actions),
            "policy_name": policy_name,
            "version": OBSERVER_WAKE_POLICY_VERSION,
            "wake_on_comparison_wake": wake_on_comparison_wake,
            "wake_on_contradiction": wake_on_contradiction,
            "wake_on_critical_action": wake_on_critical_action,
            "wake_on_every_transition": wake_on_every_transition,
            "wake_on_inconclusive": wake_on_inconclusive,
            "wake_on_missing_hidden_state_evidence": (
                wake_on_missing_hidden_state_evidence
            ),
            "wake_on_missing_policy_evidence": wake_on_missing_policy_evidence,
            "wake_on_schema_mismatch": wake_on_schema_mismatch,
        }
        return cls(
            wake_policy_id=canonical_id(payload),
            policy_name=policy_name,
            wake_on_every_transition=wake_on_every_transition,
            wake_on_comparison_wake=wake_on_comparison_wake,
            wake_on_contradiction=wake_on_contradiction,
            wake_on_inconclusive=wake_on_inconclusive,
            wake_on_schema_mismatch=wake_on_schema_mismatch,
            wake_on_missing_policy_evidence=wake_on_missing_policy_evidence,
            wake_on_missing_hidden_state_evidence=(
                wake_on_missing_hidden_state_evidence
            ),
            wake_on_critical_action=wake_on_critical_action,
            critical_actions=critical_actions,
        )


@dataclass(frozen=True)
class ObserverWakeEvaluationDTO:
    wake_evaluation_id: str
    wake_policy_id: str
    ledger_entry_id: str
    should_wake: bool
    trigger_codes: tuple[str, ...]
    version: str = OBSERVER_WAKE_EVALUATION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_WAKE_EVALUATION_VERSION:
            raise ObserverFixtureError("unsupported wake evaluation version")
        _ensure_sorted_unique(self.trigger_codes, "trigger_codes")
        if set(self.trigger_codes) - TRIGGER_CODES:
            raise ObserverFixtureError("unsupported trigger code")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.wake_evaluation_id != expected_id:
            raise ObserverFixtureError(
                "wake_evaluation_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "ledger_entry_id": self.ledger_entry_id,
            "should_wake": self.should_wake,
            "trigger_codes": list(self.trigger_codes),
            "version": self.version,
            "wake_policy_id": self.wake_policy_id,
        }
        if include_id:
            payload["wake_evaluation_id"] = self.wake_evaluation_id
        return payload


@dataclass(frozen=True)
class ObserverWakePolicyReplayDTO:
    wake_policy_replay_id: str
    wake_policy_id: str
    ledger_snapshot_id: str
    evaluations: tuple[ObserverWakeEvaluationDTO, ...]
    wake_count: int
    non_wake_count: int
    contradiction_wake_count: int
    inconclusive_wake_count: int
    missed_contradiction_entry_ids: tuple[str, ...]
    version: str = OBSERVER_WAKE_POLICY_REPLAY_VERSION

    @property
    def wake_rate(self) -> float:
        total = self.wake_count + self.non_wake_count
        return 0.0 if total == 0 else self.wake_count / total

    def __post_init__(self) -> None:
        if self.version != OBSERVER_WAKE_POLICY_REPLAY_VERSION:
            raise ObserverFixtureError("unsupported wake replay version")
        _ensure_sorted_unique(
            self.missed_contradiction_entry_ids, "missed_contradiction_entry_ids"
        )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.wake_policy_replay_id != expected_id:
            raise ObserverFixtureError("wake_policy_replay_id disagrees with payload")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "contradiction_wake_count": self.contradiction_wake_count,
            "evaluations": [item.canonical_payload() for item in self.evaluations],
            "inconclusive_wake_count": self.inconclusive_wake_count,
            "ledger_snapshot_id": self.ledger_snapshot_id,
            "missed_contradiction_entry_ids": list(self.missed_contradiction_entry_ids),
            "non_wake_count": self.non_wake_count,
            "version": self.version,
            "wake_count": self.wake_count,
            "wake_policy_id": self.wake_policy_id,
        }
        if include_id:
            payload["wake_policy_replay_id"] = self.wake_policy_replay_id
        return payload


@dataclass(frozen=True)
class ObserverWakePolicyAblationDTO:
    wake_ablation_id: str
    ledger_snapshot_id: str
    wake_policy_replays: tuple[ObserverWakePolicyReplayDTO, ...]
    baseline_policy_id: str
    policy_ids_by_wake_count: tuple[str, ...]
    policy_ids_missing_contradictions: tuple[str, ...]
    version: str = OBSERVER_WAKE_POLICY_ABLATION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_WAKE_POLICY_ABLATION_VERSION:
            raise ObserverFixtureError("unsupported wake ablation version")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.wake_ablation_id != expected_id:
            raise ObserverFixtureError(
                "wake_ablation_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "baseline_policy_id": self.baseline_policy_id,
            "ledger_snapshot_id": self.ledger_snapshot_id,
            "policy_ids_by_wake_count": list(self.policy_ids_by_wake_count),
            "policy_ids_missing_contradictions": list(
                self.policy_ids_missing_contradictions
            ),
            "version": self.version,
            "wake_policy_replays": [
                item.canonical_payload() for item in self.wake_policy_replays
            ],
        }
        if include_id:
            payload["wake_ablation_id"] = self.wake_ablation_id
        return payload


def evaluate_wake_policy_for_entry(
    *,
    entry: ObserverTransitionLedgerEntryDTO,
    wake_policy: ObserverWakePolicyDTO,
) -> ObserverWakeEvaluationDTO:
    comparison = entry.transition_verification.comparison_result
    triggers: set[str] = set()
    if wake_policy.wake_on_every_transition:
        triggers.add("always")
    if wake_policy.wake_on_comparison_wake and comparison.wake_required:
        triggers.add("comparison_wake")
    if wake_policy.wake_on_contradiction and comparison.contradiction:
        triggers.add("contradiction")
    if wake_policy.wake_on_inconclusive and comparison.inconclusive_reasons:
        triggers.add("inconclusive")
    if (
        wake_policy.wake_on_schema_mismatch
        and "schema_mismatch" in comparison.inconclusive_reasons
    ):
        triggers.add("schema_mismatch")
    if (
        wake_policy.wake_on_missing_policy_evidence
        and "missing_policy_consequence_evidence" in comparison.inconclusive_reasons
    ):
        triggers.add("missing_policy_evidence")
    if (
        wake_policy.wake_on_missing_hidden_state_evidence
        and "missing_hidden_state_hypothesis_set" in comparison.inconclusive_reasons
    ):
        triggers.add("missing_hidden_state_evidence")
    action = entry.transition_verification.transition_record.action
    if wake_policy.wake_on_critical_action and action in wake_policy.critical_actions:
        triggers.add("critical_action")
    trigger_codes = tuple(sorted(triggers))
    payload = {
        "ledger_entry_id": entry.ledger_entry_id,
        "should_wake": bool(trigger_codes),
        "trigger_codes": list(trigger_codes),
        "version": OBSERVER_WAKE_EVALUATION_VERSION,
        "wake_policy_id": wake_policy.wake_policy_id,
    }
    return ObserverWakeEvaluationDTO(
        wake_evaluation_id=canonical_id(payload),
        wake_policy_id=wake_policy.wake_policy_id,
        ledger_entry_id=entry.ledger_entry_id,
        should_wake=bool(trigger_codes),
        trigger_codes=trigger_codes,
    )


def evaluate_wake_policy_over_ledger(
    *,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
    wake_policy: ObserverWakePolicyDTO,
) -> ObserverWakePolicyReplayDTO:
    evaluations = tuple(
        evaluate_wake_policy_for_entry(entry=entry, wake_policy=wake_policy)
        for entry in entries
    )
    wake_count = sum(1 for item in evaluations if item.should_wake)
    contradiction_wake_count = sum(
        1
        for entry, evaluation in zip(entries, evaluations, strict=True)
        if entry.transition_verification.comparison_result.contradiction
        and evaluation.should_wake
    )
    inconclusive_wake_count = sum(
        1
        for entry, evaluation in zip(entries, evaluations, strict=True)
        if entry.transition_verification.comparison_result.inconclusive_reasons
        and evaluation.should_wake
    )
    missed = tuple(
        sorted(
            entry.ledger_entry_id
            for entry, evaluation in zip(entries, evaluations, strict=True)
            if entry.transition_verification.comparison_result.contradiction
            and not evaluation.should_wake
        )
    )
    payload = {
        "contradiction_wake_count": contradiction_wake_count,
        "evaluations": [item.canonical_payload() for item in evaluations],
        "inconclusive_wake_count": inconclusive_wake_count,
        "ledger_snapshot_id": ledger_snapshot.ledger_snapshot_id,
        "missed_contradiction_entry_ids": list(missed),
        "non_wake_count": len(entries) - wake_count,
        "version": OBSERVER_WAKE_POLICY_REPLAY_VERSION,
        "wake_count": wake_count,
        "wake_policy_id": wake_policy.wake_policy_id,
    }
    return ObserverWakePolicyReplayDTO(
        wake_policy_replay_id=canonical_id(payload),
        wake_policy_id=wake_policy.wake_policy_id,
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        evaluations=evaluations,
        wake_count=wake_count,
        non_wake_count=len(entries) - wake_count,
        contradiction_wake_count=contradiction_wake_count,
        inconclusive_wake_count=inconclusive_wake_count,
        missed_contradiction_entry_ids=missed,
    )


def build_wake_policy_ablation(
    *,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    wake_policy_replays: tuple[ObserverWakePolicyReplayDTO, ...],
    baseline_policy_id: str,
) -> ObserverWakePolicyAblationDTO:
    replays = tuple(sorted(wake_policy_replays, key=lambda item: item.wake_policy_id))
    by_wake = tuple(
        item.wake_policy_id
        for item in sorted(
            replays, key=lambda item: (item.wake_count, item.wake_policy_id)
        )
    )
    missing = tuple(
        sorted(
            item.wake_policy_id
            for item in replays
            if item.missed_contradiction_entry_ids
        )
    )
    payload = {
        "baseline_policy_id": baseline_policy_id,
        "ledger_snapshot_id": ledger_snapshot.ledger_snapshot_id,
        "policy_ids_by_wake_count": list(by_wake),
        "policy_ids_missing_contradictions": list(missing),
        "version": OBSERVER_WAKE_POLICY_ABLATION_VERSION,
        "wake_policy_replays": [item.canonical_payload() for item in replays],
    }
    return ObserverWakePolicyAblationDTO(
        wake_ablation_id=canonical_id(payload),
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        wake_policy_replays=replays,
        baseline_policy_id=baseline_policy_id,
        policy_ids_by_wake_count=by_wake,
        policy_ids_missing_contradictions=missing,
    )
