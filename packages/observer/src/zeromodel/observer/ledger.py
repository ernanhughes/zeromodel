"""Append-only Observer transition ledger for Stage O3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.fixture import (
    ObserverExecutedFixtureStepDTO,
    ObserverFixtureError,
)
from zeromodel.observer.fixture_predictor import ObserverPredictedTransitionDTO
from zeromodel.observer.transition_service import ObserverTransitionVerificationDTO

OBSERVER_TRANSITION_LEDGER_ENTRY_VERSION: Final = "observer-transition-ledger-entry/1"
OBSERVER_TRANSITION_LEDGER_SNAPSHOT_VERSION: Final = (
    "observer-transition-ledger-snapshot/1"
)
OBSERVER_LEDGER_REPLAY_RESULT_VERSION: Final = "observer-ledger-replay-result/1"

LEDGER_REPLAY_STATUSES: Final = frozenset({"verified", "failed", "inconclusive"})
LEDGER_FAILURE_CODES: Final = frozenset(
    {
        "sequence_gap",
        "previous_link_mismatch",
        "source_state_mismatch",
        "action_identity_mismatch",
        "prediction_identity_mismatch",
        "execution_identity_mismatch",
        "verification_identity_mismatch",
        "terminal_sequence_violation",
    }
)


@dataclass(frozen=True)
class ObserverTransitionLedgerEntryDTO:
    """Immutable source-of-truth record for one executed fixture action."""

    ledger_entry_id: str
    ledger_sequence: int
    episode_id: str
    fixture_id: str
    source_state_id: str
    action_id: str
    predictor_rule_set_id: str
    environment_rule_set_id: str
    predicted_transition: ObserverPredictedTransitionDTO
    executed_step: ObserverExecutedFixtureStepDTO
    transition_verification: ObserverTransitionVerificationDTO
    previous_ledger_entry_id: str | None
    recorded_at_logical_step: int
    version: str = OBSERVER_TRANSITION_LEDGER_ENTRY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_TRANSITION_LEDGER_ENTRY_VERSION:
            raise ObserverFixtureError("unsupported ledger entry version")
        if self.ledger_sequence < 0:
            raise ObserverFixtureError("ledger_sequence must be non-negative")
        if self.recorded_at_logical_step < 0:
            raise ObserverFixtureError("recorded_at_logical_step must be non-negative")
        for field_name in (
            "episode_id",
            "fixture_id",
            "source_state_id",
            "action_id",
            "predictor_rule_set_id",
            "environment_rule_set_id",
        ):
            if not getattr(self, field_name):
                raise ObserverFixtureError(f"{field_name} must be non-empty")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.ledger_entry_id != expected_id:
            raise ObserverFixtureError(
                "ledger_entry_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "action_id": self.action_id,
            "environment_rule_set_id": self.environment_rule_set_id,
            "episode_id": self.episode_id,
            "executed_step": self.executed_step.canonical_payload(),
            "fixture_id": self.fixture_id,
            "ledger_sequence": self.ledger_sequence,
            "predicted_transition": self.predicted_transition.canonical_payload(),
            "predictor_rule_set_id": self.predictor_rule_set_id,
            "previous_ledger_entry_id": self.previous_ledger_entry_id,
            "recorded_at_logical_step": self.recorded_at_logical_step,
            "source_state_id": self.source_state_id,
            "transition_verification": self.transition_verification.canonical_payload(),
            "version": self.version,
        }
        if include_id:
            payload["ledger_entry_id"] = self.ledger_entry_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        ledger_sequence: int,
        episode_id: str,
        fixture_id: str,
        source_state_id: str,
        action_id: str,
        predictor_rule_set_id: str,
        environment_rule_set_id: str,
        predicted_transition: ObserverPredictedTransitionDTO,
        executed_step: ObserverExecutedFixtureStepDTO,
        transition_verification: ObserverTransitionVerificationDTO,
        previous_ledger_entry_id: str | None,
        recorded_at_logical_step: int,
    ) -> "ObserverTransitionLedgerEntryDTO":
        payload = {
            "action_id": action_id,
            "environment_rule_set_id": environment_rule_set_id,
            "episode_id": episode_id,
            "executed_step": executed_step.canonical_payload(),
            "fixture_id": fixture_id,
            "ledger_sequence": ledger_sequence,
            "predicted_transition": predicted_transition.canonical_payload(),
            "predictor_rule_set_id": predictor_rule_set_id,
            "previous_ledger_entry_id": previous_ledger_entry_id,
            "recorded_at_logical_step": recorded_at_logical_step,
            "source_state_id": source_state_id,
            "transition_verification": transition_verification.canonical_payload(),
            "version": OBSERVER_TRANSITION_LEDGER_ENTRY_VERSION,
        }
        return cls(
            ledger_entry_id=canonical_id(payload),
            ledger_sequence=ledger_sequence,
            episode_id=episode_id,
            fixture_id=fixture_id,
            source_state_id=source_state_id,
            action_id=action_id,
            predictor_rule_set_id=predictor_rule_set_id,
            environment_rule_set_id=environment_rule_set_id,
            predicted_transition=predicted_transition,
            executed_step=executed_step,
            transition_verification=transition_verification,
            previous_ledger_entry_id=previous_ledger_entry_id,
            recorded_at_logical_step=recorded_at_logical_step,
        )


@dataclass(frozen=True)
class ObserverTransitionLedgerSnapshotDTO:
    """Canonical snapshot of a ledger entry sequence."""

    ledger_snapshot_id: str
    fixture_id: str
    episode_ids: tuple[str, ...]
    entry_ids: tuple[str, ...]
    head_entry_id: str | None
    entry_count: int
    version: str = OBSERVER_TRANSITION_LEDGER_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_TRANSITION_LEDGER_SNAPSHOT_VERSION:
            raise ObserverFixtureError("unsupported ledger snapshot version")
        if self.entry_count != len(self.entry_ids):
            raise ObserverFixtureError("entry_count must match entry_ids length")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.ledger_snapshot_id != expected_id:
            raise ObserverFixtureError(
                "ledger_snapshot_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "entry_count": self.entry_count,
            "entry_ids": list(self.entry_ids),
            "episode_ids": list(self.episode_ids),
            "fixture_id": self.fixture_id,
            "head_entry_id": self.head_entry_id,
            "version": self.version,
        }
        if include_id:
            payload["ledger_snapshot_id"] = self.ledger_snapshot_id
        return payload


@dataclass(frozen=True)
class ObserverLedgerReplayResultDTO:
    """Canonical ledger replay integrity result."""

    ledger_replay_result_id: str
    ledger_snapshot_id: str
    status: str
    replayed_entry_ids: tuple[str, ...]
    failed_entry_ids: tuple[str, ...]
    failure_codes: tuple[str, ...]
    version: str = OBSERVER_LEDGER_REPLAY_RESULT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_LEDGER_REPLAY_RESULT_VERSION:
            raise ObserverFixtureError("unsupported ledger replay result version")
        if self.status not in LEDGER_REPLAY_STATUSES:
            raise ObserverFixtureError("unsupported ledger replay status")
        if self.failure_codes != tuple(sorted(set(self.failure_codes))):
            raise ObserverFixtureError("failure_codes must be unique and sorted")
        unknown = set(self.failure_codes) - LEDGER_FAILURE_CODES
        if unknown:
            raise ObserverFixtureError(f"unsupported failure codes: {sorted(unknown)}")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.ledger_replay_result_id != expected_id:
            raise ObserverFixtureError(
                "ledger_replay_result_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "failed_entry_ids": list(self.failed_entry_ids),
            "failure_codes": list(self.failure_codes),
            "ledger_snapshot_id": self.ledger_snapshot_id,
            "replayed_entry_ids": list(self.replayed_entry_ids),
            "status": self.status,
            "version": self.version,
        }
        if include_id:
            payload["ledger_replay_result_id"] = self.ledger_replay_result_id
        return payload


class InMemoryObserverTransitionLedger:
    """Episode-scoped append-only in-memory transition ledger."""

    def __init__(self, *, fixture_id: str, episode_id: str) -> None:
        self._fixture_id = fixture_id
        self._episode_id = episode_id
        self._entries: list[ObserverTransitionLedgerEntryDTO] = []

    def append(self, entry: ObserverTransitionLedgerEntryDTO) -> None:
        if entry.fixture_id != self._fixture_id:
            raise ObserverFixtureError("entry fixture_id does not match ledger")
        if entry.episode_id != self._episode_id:
            raise ObserverFixtureError("entry episode_id does not match ledger")
        if entry.ledger_sequence != len(self._entries):
            raise ObserverFixtureError("sequence gap or duplicate sequence")
        previous_id = None if not self._entries else self._entries[-1].ledger_entry_id
        if entry.previous_ledger_entry_id != previous_id:
            raise ObserverFixtureError("incorrect previous-entry linkage")
        if self._entries and self._entries[-1].executed_step.actual_state.terminal:
            raise ObserverFixtureError(
                "cannot append after terminal same-episode entry"
            )
        if any(
            existing.ledger_entry_id == entry.ledger_entry_id
            for existing in self._entries
        ):
            raise ObserverFixtureError("duplicate ledger entry")
        self._entries.append(entry)

    def get(self, entry_id: str) -> ObserverTransitionLedgerEntryDTO:
        for entry in self._entries:
            if entry.ledger_entry_id == entry_id:
                return entry
        raise ObserverFixtureError("unknown ledger entry")

    def get_by_sequence(self, sequence: int) -> ObserverTransitionLedgerEntryDTO:
        return self._entries[sequence]

    def entries(self) -> tuple[ObserverTransitionLedgerEntryDTO, ...]:
        return tuple(self._entries)

    def head(self) -> ObserverTransitionLedgerEntryDTO | None:
        return None if not self._entries else self._entries[-1]

    def snapshot(self) -> ObserverTransitionLedgerSnapshotDTO:
        return build_observer_transition_ledger_snapshot(entries=self.entries())

    def verify_integrity(self) -> ObserverLedgerReplayResultDTO:
        return replay_observer_transition_ledger(
            ledger_snapshot=self.snapshot(), entries=self.entries()
        )


def build_observer_transition_ledger_snapshot(
    *, entries: tuple[ObserverTransitionLedgerEntryDTO, ...]
) -> ObserverTransitionLedgerSnapshotDTO:
    fixture_id = "" if not entries else entries[0].fixture_id
    episode_ids = tuple(sorted({entry.episode_id for entry in entries}))
    entry_ids = tuple(entry.ledger_entry_id for entry in entries)
    payload = {
        "entry_count": len(entries),
        "entry_ids": list(entry_ids),
        "episode_ids": list(episode_ids),
        "fixture_id": fixture_id,
        "head_entry_id": None if not entries else entries[-1].ledger_entry_id,
        "version": OBSERVER_TRANSITION_LEDGER_SNAPSHOT_VERSION,
    }
    return ObserverTransitionLedgerSnapshotDTO(
        ledger_snapshot_id=canonical_id(payload),
        fixture_id=fixture_id,
        episode_ids=episode_ids,
        entry_ids=entry_ids,
        head_entry_id=None if not entries else entries[-1].ledger_entry_id,
        entry_count=len(entries),
    )


def replay_observer_transition_ledger(
    *,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
) -> ObserverLedgerReplayResultDTO:
    failures: set[str] = set()
    failed_entries: set[str] = set()
    replayed: list[str] = []
    previous_id: str | None = None
    for expected_sequence, entry in enumerate(entries):
        replayed.append(entry.ledger_entry_id)
        if entry.ledger_sequence != expected_sequence:
            failures.add("sequence_gap")
            failed_entries.add(entry.ledger_entry_id)
        if entry.previous_ledger_entry_id != previous_id:
            failures.add("previous_link_mismatch")
            failed_entries.add(entry.ledger_entry_id)
        if entry.source_state_id != entry.predicted_transition.source_state_id:
            failures.add("source_state_mismatch")
            failed_entries.add(entry.ledger_entry_id)
        if (
            entry.action_id != entry.predicted_transition.action_id
            or entry.action_id != entry.executed_step.action_id
        ):
            failures.add("action_identity_mismatch")
            failed_entries.add(entry.ledger_entry_id)
        if entry.predicted_transition.predicted_transition_id != canonical_id(
            entry.predicted_transition.canonical_payload(include_id=False)
        ):
            failures.add("prediction_identity_mismatch")
            failed_entries.add(entry.ledger_entry_id)
        if entry.executed_step.executed_step_id != canonical_id(
            entry.executed_step.canonical_payload(include_id=False)
        ):
            failures.add("execution_identity_mismatch")
            failed_entries.add(entry.ledger_entry_id)
        if entry.transition_verification.verification_id != canonical_id(
            entry.transition_verification.canonical_payload(include_id=False)
        ):
            failures.add("verification_identity_mismatch")
            failed_entries.add(entry.ledger_entry_id)
        if (
            previous_id is not None
            and entries[expected_sequence - 1].executed_step.actual_state.terminal
        ):
            failures.add("terminal_sequence_violation")
            failed_entries.add(entry.ledger_entry_id)
        previous_id = entry.ledger_entry_id
    if ledger_snapshot.entry_ids != tuple(entry.ledger_entry_id for entry in entries):
        failures.add("sequence_gap")
    status = "verified" if not failures else "failed"
    payload = {
        "failed_entry_ids": sorted(failed_entries),
        "failure_codes": sorted(failures),
        "ledger_snapshot_id": ledger_snapshot.ledger_snapshot_id,
        "replayed_entry_ids": replayed,
        "status": status,
        "version": OBSERVER_LEDGER_REPLAY_RESULT_VERSION,
    }
    return ObserverLedgerReplayResultDTO(
        ledger_replay_result_id=canonical_id(payload),
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        status=status,
        replayed_entry_ids=tuple(replayed),
        failed_entry_ids=tuple(sorted(failed_entries)),
        failure_codes=tuple(sorted(failures)),
    )
