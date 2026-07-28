"""Append-only Observer transition ledger for Stage O3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.fixture import (
    ObserverExecutedFixtureStepDTO,
    ObserverFixtureActionDTO,
    ObserverFixtureError,
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
)
from zeromodel.observer.fixture_predictor import (
    ObserverPredictedTransitionDTO,
    execute_observer_fixture_step,
    predict_observer_fixture_transition,
)
from zeromodel.observer.artifacts import ObserverObservationSchemaDTO
from zeromodel.observer.comparison import ObserverComparisonRecipeDTO
from zeromodel.observer.transition_service import (
    ObserverTransitionVerificationDTO,
    verify_observer_transition,
)

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
        "snapshot_identity_mismatch",
        "prediction_replay_mismatch",
        "execution_replay_mismatch",
        "verification_replay_mismatch",
        "missing_predictor_rule_set",
        "missing_environment_rule_set",
        "missing_source_state",
    }
)


@dataclass(frozen=True)
class ObserverTransitionLedgerEntryDTO:
    """Immutable source-of-truth record for one executed fixture action."""

    ledger_entry_id: str
    ledger_sequence: int
    episode_id: str
    fixture_id: str
    source_state: ObserverFixtureStateDTO
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
        self._validate_embedded_object_invariants()
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.ledger_entry_id != expected_id:
            raise ObserverFixtureError(
                "ledger_entry_id disagrees with canonical payload"
            )

    def _validate_embedded_object_invariants(self) -> None:
        if self.source_state.fixture_state_id != self.source_state_id:
            raise ObserverFixtureError("source_state_id does not match source_state")
        if self.fixture_id != self.source_state.fixture_id:
            raise ObserverFixtureError("fixture_id does not match source_state")
        if self.episode_id != self.source_state.episode_id:
            raise ObserverFixtureError("episode_id does not match source_state")
        if self.source_state_id != self.predicted_transition.source_state_id:
            raise ObserverFixtureError(
                "source_state_id does not match predicted transition"
            )
        if self.source_state_id != self.executed_step.source_state_id:
            raise ObserverFixtureError("source_state_id does not match executed step")
        if self.action_id != self.predicted_transition.action_id:
            raise ObserverFixtureError("action_id does not match predicted transition")
        if self.action_id != self.executed_step.action_id:
            raise ObserverFixtureError("action_id does not match executed step")
        if (
            self.predictor_rule_set_id
            != self.predicted_transition.predictor_rule_set_id
        ):
            raise ObserverFixtureError(
                "predictor_rule_set_id does not match predicted transition"
            )
        if self.environment_rule_set_id != self.executed_step.environment_rule_set_id:
            raise ObserverFixtureError(
                "environment_rule_set_id does not match executed step"
            )
        if (
            self.transition_verification.predicted_observation_artifact_id
            != self.predicted_transition.predicted_observation.observation_artifact_id
        ):
            raise ObserverFixtureError(
                "verification predicted observation does not match prediction"
            )
        if (
            self.transition_verification.observed_observation_artifact_id
            != self.executed_step.actual_observation_id
        ):
            raise ObserverFixtureError(
                "verification observed observation does not match execution"
            )
        record = self.transition_verification.transition_record
        if record.state_before_id != self.source_state_id:
            raise ObserverFixtureError(
                "transition record state_before_id does not match source_state_id"
            )
        if (
            record.predicted_state_after_id
            != self.predicted_transition.predicted_observation.observation_artifact_id
        ):
            raise ObserverFixtureError(
                "transition record predicted state id does not match prediction"
            )
        if record.observed_state_after_id != self.executed_step.actual_observation_id:
            raise ObserverFixtureError(
                "transition record observed state id does not match execution"
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
            "source_state": self.source_state.canonical_payload(),
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
        source_state: ObserverFixtureStateDTO,
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
            "source_state": source_state.canonical_payload(),
            "source_state_id": source_state_id,
            "transition_verification": transition_verification.canonical_payload(),
            "version": OBSERVER_TRANSITION_LEDGER_ENTRY_VERSION,
        }
        return cls(
            ledger_entry_id=canonical_id(payload),
            ledger_sequence=ledger_sequence,
            episode_id=episode_id,
            fixture_id=fixture_id,
            source_state=source_state,
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
    episode_id: str
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
            "episode_id": self.episode_id,
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
        return verify_observer_transition_ledger_integrity(
            ledger_snapshot=self.snapshot(), entries=self.entries()
        )


def build_observer_transition_ledger_snapshot(
    *, entries: tuple[ObserverTransitionLedgerEntryDTO, ...]
) -> ObserverTransitionLedgerSnapshotDTO:
    fixture_id = "" if not entries else entries[0].fixture_id
    episode_id = "" if not entries else entries[0].episode_id
    if len({entry.fixture_id for entry in entries}) > 1:
        raise ObserverFixtureError("snapshot entries must share one fixture_id")
    if len({entry.episode_id for entry in entries}) > 1:
        raise ObserverFixtureError("snapshot entries must share one episode_id")
    entry_ids = tuple(entry.ledger_entry_id for entry in entries)
    payload = {
        "entry_count": len(entries),
        "entry_ids": list(entry_ids),
        "episode_id": episode_id,
        "fixture_id": fixture_id,
        "head_entry_id": None if not entries else entries[-1].ledger_entry_id,
        "version": OBSERVER_TRANSITION_LEDGER_SNAPSHOT_VERSION,
    }
    return ObserverTransitionLedgerSnapshotDTO(
        ledger_snapshot_id=canonical_id(payload),
        fixture_id=fixture_id,
        episode_id=episode_id,
        entry_ids=entry_ids,
        head_entry_id=None if not entries else entries[-1].ledger_entry_id,
        entry_count=len(entries),
    )


def verify_observer_transition_ledger_integrity(
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
        if entry.source_state.fixture_state_id != entry.source_state_id:
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
    expected_snapshot = build_observer_transition_ledger_snapshot(entries=entries)
    if ledger_snapshot.ledger_snapshot_id != expected_snapshot.ledger_snapshot_id:
        failures.add("snapshot_identity_mismatch")
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


def replay_observer_fixture_ledger(
    *,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
    observation_schema: ObserverObservationSchemaDTO,
    comparison_recipe: ObserverComparisonRecipeDTO,
    predictor_rule_sets: Mapping[str, ObserverFixtureRuleSetDTO],
    environment_rule_sets: Mapping[str, ObserverFixtureRuleSetDTO],
) -> ObserverLedgerReplayResultDTO:
    integrity = verify_observer_transition_ledger_integrity(
        ledger_snapshot=ledger_snapshot, entries=entries
    )
    failures = set(integrity.failure_codes)
    failed_entries = set(integrity.failed_entry_ids)
    replayed: list[str] = []
    expected_source_state: ObserverFixtureStateDTO | None = None
    for entry in entries:
        replayed.append(entry.ledger_entry_id)
        source_state = entry.source_state
        if expected_source_state is not None:
            source_state = expected_source_state
            if source_state.fixture_state_id != entry.source_state_id:
                failures.add("missing_source_state")
                failed_entries.add(entry.ledger_entry_id)
                expected_source_state = entry.executed_step.actual_state
                continue
        predictor_rule_set = predictor_rule_sets.get(entry.predictor_rule_set_id)
        if predictor_rule_set is None:
            failures.add("missing_predictor_rule_set")
            failed_entries.add(entry.ledger_entry_id)
            expected_source_state = entry.executed_step.actual_state
            continue
        environment_rule_set = environment_rule_sets.get(entry.environment_rule_set_id)
        if environment_rule_set is None:
            failures.add("missing_environment_rule_set")
            failed_entries.add(entry.ledger_entry_id)
            expected_source_state = entry.executed_step.actual_state
            continue
        action = ObserverFixtureActionDTO.create(
            action_name=entry.transition_verification.transition_record.action
        )
        if action.fixture_action_id != entry.action_id:
            failures.add("action_identity_mismatch")
            failed_entries.add(entry.ledger_entry_id)
            expected_source_state = entry.executed_step.actual_state
            continue
        replayed_prediction = predict_observer_fixture_transition(
            source_state=source_state,
            action=action,
            predictor_rule_set=predictor_rule_set,
            observation_schema=observation_schema,
        )
        replayed_execution, replayed_observation = execute_observer_fixture_step(
            source_state=source_state,
            action=action,
            environment_rule_set=environment_rule_set,
            observation_schema=observation_schema,
        )
        record = entry.transition_verification.transition_record
        missing_hidden = (
            "missing_hidden_state_hypothesis_set"
            in entry.transition_verification.comparison_result.inconclusive_reasons
        )
        contradiction = entry.transition_verification.contradiction_artifact
        replayed_verification = verify_observer_transition(
            recipe=comparison_recipe,
            predicted_observation=replayed_prediction.predicted_observation,
            observed_observation=replayed_observation,
            policy_artifact_id=record.policy_artifact_id,
            state_before_id=record.state_before_id,
            action=record.action,
            affected_policy_row_id=record.affected_policy_row_id,
            hidden_state_hypothesis_set=(
                None
                if missing_hidden
                else replayed_prediction.hidden_state_hypothesis_set
            ),
            reproduction=(
                {} if contradiction is None else dict(contradiction.reproduction)
            ),
            relevant_context_keys=(
                () if contradiction is None else contradiction.relevant_context_keys
            ),
        )
        if (
            replayed_prediction.predicted_transition_id
            != entry.predicted_transition.predicted_transition_id
        ):
            failures.add("prediction_replay_mismatch")
            failed_entries.add(entry.ledger_entry_id)
        if replayed_execution.executed_step_id != entry.executed_step.executed_step_id:
            failures.add("execution_replay_mismatch")
            failed_entries.add(entry.ledger_entry_id)
        if (
            replayed_verification.verification_id
            != entry.transition_verification.verification_id
        ):
            failures.add("verification_replay_mismatch")
            failed_entries.add(entry.ledger_entry_id)
        expected_source_state = replayed_execution.actual_state
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
