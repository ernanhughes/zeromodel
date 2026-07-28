"""Immutable in-memory registry for admitted Observer habits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence, cast

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.habit import ObserverHabitError, ObserverHabitSpecificationDTO
from zeromodel.observer.habit_admission import ObserverHabitAdmissionDecisionDTO

OBSERVER_HABIT_REGISTRY_ENTRY_VERSION: Final = "observer-habit-registry-entry/1"
OBSERVER_HABIT_REGISTRY_EVENT_VERSION: Final = "observer-habit-registry-event/1"
OBSERVER_HABIT_REGISTRY_SNAPSHOT_VERSION: Final = "observer-habit-registry-snapshot/1"
OBSERVER_HABIT_ROLLBACK_REQUEST_VERSION: Final = "observer-habit-rollback-request/1"
OBSERVER_HABIT_ROLLBACK_RESULT_VERSION: Final = "observer-habit-rollback-result/1"
OBSERVER_HABIT_REGISTRY_REPLAY_VERSION: Final = "observer-habit-registry-replay/1"

REGISTRY_STATUSES: Final = frozenset(
    {"admitted_inactive", "active", "suspended", "retired"}
)
REGISTRY_EVENTS: Final = frozenset(
    {
        "admitted",
        "activated",
        "deactivated",
        "suspended",
        "resumed",
        "retired",
        "rolled_back",
    }
)
ROLLBACK_DISPOSITIONS: Final = frozenset(
    {
        "rolled_back",
        "target_not_found",
        "target_not_ancestor",
        "stale_registry_snapshot",
        "registry_invalid",
        "unsupported",
    }
)
REPLAY_STATUSES: Final = frozenset({"verified", "failed", "inconclusive"})


class ObserverHabitRegistryError(ObserverHabitError):
    """Raised when registry state or events are malformed."""


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverHabitRegistryError(f"{field_name} must be non-empty")


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverHabitRegistryError(f"{field_name} must be unique and sorted")


@dataclass(frozen=True)
class ObserverHabitRegistryEntryDTO:
    habit_registry_entry_id: str
    habit_specification_id: str
    habit_admission_decision_id: str
    status: str
    activation_generation: int
    active_since_registry_sequence: int | None
    suspended_since_registry_sequence: int | None
    retired_since_registry_sequence: int | None
    status_reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_REGISTRY_ENTRY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_REGISTRY_ENTRY_VERSION:
            raise ObserverHabitRegistryError("unsupported registry entry version")
        _require_non_empty(self.habit_specification_id, "habit_specification_id")
        _require_non_empty(
            self.habit_admission_decision_id, "habit_admission_decision_id"
        )
        if self.status not in REGISTRY_STATUSES:
            raise ObserverHabitRegistryError("unsupported registry status")
        if self.activation_generation < 0:
            raise ObserverHabitRegistryError(
                "activation_generation must be non-negative"
            )
        if self.status == "admitted_inactive":
            if (
                self.active_since_registry_sequence is not None
                or self.suspended_since_registry_sequence is not None
                or self.retired_since_registry_sequence is not None
            ):
                raise ObserverHabitRegistryError("inactive entry has status timestamps")
        elif self.status == "active":
            if (
                self.active_since_registry_sequence is None
                or self.suspended_since_registry_sequence is not None
                or self.retired_since_registry_sequence is not None
                or self.activation_generation < 1
            ):
                raise ObserverHabitRegistryError("active entry timestamps are invalid")
        elif self.status == "suspended":
            if (
                self.active_since_registry_sequence is not None
                or self.suspended_since_registry_sequence is None
                or self.retired_since_registry_sequence is not None
            ):
                raise ObserverHabitRegistryError(
                    "suspended entry timestamps are invalid"
                )
        elif self.status == "retired" and (
            self.active_since_registry_sequence is not None
            or self.retired_since_registry_sequence is None
        ):
            raise ObserverHabitRegistryError("retired entry timestamps are invalid")
        _ensure_sorted_unique(self.status_reason_codes, "status_reason_codes")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_registry_entry_id != expected_id:
            raise ObserverHabitRegistryError("habit_registry_entry_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "activation_generation": self.activation_generation,
            "active_since_registry_sequence": self.active_since_registry_sequence,
            "habit_admission_decision_id": self.habit_admission_decision_id,
            "habit_specification_id": self.habit_specification_id,
            "retired_since_registry_sequence": self.retired_since_registry_sequence,
            "status": self.status,
            "status_reason_codes": list(self.status_reason_codes),
            "suspended_since_registry_sequence": self.suspended_since_registry_sequence,
            "version": self.version,
        }
        if include_id:
            payload["habit_registry_entry_id"] = self.habit_registry_entry_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitRegistryEntryDTO":
        values["status_reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("status_reason_codes", ()))))
        )
        payload = {
            **values,
            "status_reason_codes": list(
                cast(tuple[str, ...], values["status_reason_codes"])
            ),
            "version": OBSERVER_HABIT_REGISTRY_ENTRY_VERSION,
        }
        return cls(
            habit_registry_entry_id=canonical_id(payload),
            version=OBSERVER_HABIT_REGISTRY_ENTRY_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitRegistryEventDTO:
    habit_registry_event_id: str
    registry_sequence: int
    event_type: str
    habit_specification_id: str
    habit_registry_entry_id: str | None
    admission_decision_id: str | None
    previous_registry_snapshot_id: str | None
    source_registry_snapshot_id: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_REGISTRY_EVENT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_REGISTRY_EVENT_VERSION:
            raise ObserverHabitRegistryError("unsupported registry event version")
        if self.registry_sequence < 0:
            raise ObserverHabitRegistryError("registry_sequence must be non-negative")
        if self.event_type not in REGISTRY_EVENTS:
            raise ObserverHabitRegistryError("unsupported registry event type")
        _require_non_empty(self.habit_specification_id, "habit_specification_id")
        _require_non_empty(
            self.source_registry_snapshot_id, "source_registry_snapshot_id"
        )
        if self.event_type == "admitted" and self.admission_decision_id is None:
            raise ObserverHabitRegistryError("admitted event requires admission")
        if (
            self.event_type != "rolled_back"
            and self.previous_registry_snapshot_id is not None
        ):
            raise ObserverHabitRegistryError(
                "non-rollback event cannot carry rollback target"
            )
        if (
            self.event_type == "rolled_back"
            and self.previous_registry_snapshot_id is None
        ):
            raise ObserverHabitRegistryError("rollback event requires target snapshot")
        if self.event_type == "rolled_back":
            if self.habit_specification_id != "registry":
                raise ObserverHabitRegistryError("rollback event must target registry")
            if (
                self.habit_registry_entry_id is not None
                or self.admission_decision_id is not None
            ):
                raise ObserverHabitRegistryError(
                    "rollback event cannot carry habit lineage"
                )
        elif self.habit_registry_entry_id is None:
            raise ObserverHabitRegistryError("habit event requires registry entry ID")
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_registry_event_id != expected_id:
            raise ObserverHabitRegistryError("habit_registry_event_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "admission_decision_id": self.admission_decision_id,
            "event_type": self.event_type,
            "habit_registry_entry_id": self.habit_registry_entry_id,
            "habit_specification_id": self.habit_specification_id,
            "previous_registry_snapshot_id": self.previous_registry_snapshot_id,
            "reason_codes": list(self.reason_codes),
            "registry_sequence": self.registry_sequence,
            "source_registry_snapshot_id": self.source_registry_snapshot_id,
            "version": self.version,
        }
        if include_id:
            payload["habit_registry_event_id"] = self.habit_registry_event_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitRegistryEventDTO":
        values["reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("reason_codes", ()))))
        )
        payload = {
            **values,
            "reason_codes": list(cast(tuple[str, ...], values["reason_codes"])),
            "version": OBSERVER_HABIT_REGISTRY_EVENT_VERSION,
        }
        return cls(
            habit_registry_event_id=canonical_id(payload),
            version=OBSERVER_HABIT_REGISTRY_EVENT_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitRegistrySnapshotDTO:
    habit_registry_snapshot_id: str
    registry_sequence: int
    entries: tuple[ObserverHabitRegistryEntryDTO, ...]
    active_habit_ids: tuple[str, ...]
    previous_registry_snapshot_id: str | None
    version: str = OBSERVER_HABIT_REGISTRY_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_REGISTRY_SNAPSHOT_VERSION:
            raise ObserverHabitRegistryError("unsupported registry snapshot version")
        if self.registry_sequence < 0:
            raise ObserverHabitRegistryError("registry_sequence must be non-negative")
        habit_ids = tuple(entry.habit_specification_id for entry in self.entries)
        _ensure_sorted_unique(habit_ids, "entry habit_specification_ids")
        entry_ids = tuple(entry.habit_registry_entry_id for entry in self.entries)
        _ensure_sorted_unique(tuple(sorted(entry_ids)), "entry IDs")
        active = tuple(
            sorted(
                entry.habit_specification_id
                for entry in self.entries
                if entry.status == "active"
            )
        )
        if self.active_habit_ids != active:
            raise ObserverHabitRegistryError("active_habit_ids mismatch")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_registry_snapshot_id != expected_id:
            raise ObserverHabitRegistryError("habit_registry_snapshot_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "active_habit_ids": list(self.active_habit_ids),
            "entries": [entry.canonical_payload() for entry in self.entries],
            "previous_registry_snapshot_id": self.previous_registry_snapshot_id,
            "registry_sequence": self.registry_sequence,
            "version": self.version,
        }
        if include_id:
            payload["habit_registry_snapshot_id"] = self.habit_registry_snapshot_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        registry_sequence: int,
        entries: tuple[ObserverHabitRegistryEntryDTO, ...],
        previous_registry_snapshot_id: str | None,
    ) -> "ObserverHabitRegistrySnapshotDTO":
        entries = tuple(sorted(entries, key=lambda item: item.habit_specification_id))
        active = tuple(
            sorted(
                entry.habit_specification_id
                for entry in entries
                if entry.status == "active"
            )
        )
        payload = {
            "active_habit_ids": list(active),
            "entries": [entry.canonical_payload() for entry in entries],
            "previous_registry_snapshot_id": previous_registry_snapshot_id,
            "registry_sequence": registry_sequence,
            "version": OBSERVER_HABIT_REGISTRY_SNAPSHOT_VERSION,
        }
        return cls(
            habit_registry_snapshot_id=canonical_id(payload),
            registry_sequence=registry_sequence,
            entries=entries,
            active_habit_ids=active,
            previous_registry_snapshot_id=previous_registry_snapshot_id,
        )


@dataclass(frozen=True)
class ObserverHabitRollbackRequestDTO:
    habit_rollback_request_id: str
    current_registry_snapshot_id: str
    target_registry_snapshot_id: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_ROLLBACK_REQUEST_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ROLLBACK_REQUEST_VERSION:
            raise ObserverHabitRegistryError("unsupported rollback request version")
        _require_non_empty(
            self.current_registry_snapshot_id, "current_registry_snapshot_id"
        )
        _require_non_empty(
            self.target_registry_snapshot_id, "target_registry_snapshot_id"
        )
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_rollback_request_id != expected_id:
            raise ObserverHabitRegistryError("habit_rollback_request_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "current_registry_snapshot_id": self.current_registry_snapshot_id,
            "reason_codes": list(self.reason_codes),
            "target_registry_snapshot_id": self.target_registry_snapshot_id,
            "version": self.version,
        }
        if include_id:
            payload["habit_rollback_request_id"] = self.habit_rollback_request_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitRollbackRequestDTO":
        values["reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("reason_codes", ()))))
        )
        payload = {
            **values,
            "reason_codes": list(cast(tuple[str, ...], values["reason_codes"])),
            "version": OBSERVER_HABIT_ROLLBACK_REQUEST_VERSION,
        }
        return cls(
            habit_rollback_request_id=canonical_id(payload),
            version=OBSERVER_HABIT_ROLLBACK_REQUEST_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitRollbackResultDTO:
    habit_rollback_result_id: str
    rollback_request_id: str
    source_registry_snapshot_id: str
    target_registry_snapshot_id: str
    result_registry_snapshot_id: str | None
    registry_event_id: str | None
    disposition: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_ROLLBACK_RESULT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ROLLBACK_RESULT_VERSION:
            raise ObserverHabitRegistryError("unsupported rollback result version")
        if self.disposition not in ROLLBACK_DISPOSITIONS:
            raise ObserverHabitRegistryError("unsupported rollback disposition")
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_rollback_result_id != expected_id:
            raise ObserverHabitRegistryError("habit_rollback_result_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
            "registry_event_id": self.registry_event_id,
            "result_registry_snapshot_id": self.result_registry_snapshot_id,
            "rollback_request_id": self.rollback_request_id,
            "source_registry_snapshot_id": self.source_registry_snapshot_id,
            "target_registry_snapshot_id": self.target_registry_snapshot_id,
            "version": self.version,
        }
        if include_id:
            payload["habit_rollback_result_id"] = self.habit_rollback_result_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitRollbackResultDTO":
        values["reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("reason_codes", ()))))
        )
        payload = {
            **values,
            "reason_codes": list(cast(tuple[str, ...], values["reason_codes"])),
            "version": OBSERVER_HABIT_ROLLBACK_RESULT_VERSION,
        }
        return cls(
            habit_rollback_result_id=canonical_id(payload),
            version=OBSERVER_HABIT_ROLLBACK_RESULT_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitRegistryReplayDTO:
    habit_registry_replay_id: str
    initial_snapshot_id: str
    final_snapshot_id: str
    replayed_event_ids: tuple[str, ...]
    status: str
    failure_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_REGISTRY_REPLAY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_REGISTRY_REPLAY_VERSION:
            raise ObserverHabitRegistryError("unsupported registry replay version")
        if self.status not in REPLAY_STATUSES:
            raise ObserverHabitRegistryError("unsupported registry replay status")
        _ensure_sorted_unique(self.failure_codes, "failure_codes")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_registry_replay_id != expected_id:
            raise ObserverHabitRegistryError("habit_registry_replay_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "failure_codes": list(self.failure_codes),
            "final_snapshot_id": self.final_snapshot_id,
            "initial_snapshot_id": self.initial_snapshot_id,
            "replayed_event_ids": list(self.replayed_event_ids),
            "status": self.status,
            "version": self.version,
        }
        if include_id:
            payload["habit_registry_replay_id"] = self.habit_registry_replay_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitRegistryReplayDTO":
        values["failure_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("failure_codes", ()))))
        )
        payload = {
            **values,
            "failure_codes": list(cast(tuple[str, ...], values["failure_codes"])),
            "replayed_event_ids": list(
                cast(tuple[str, ...], values["replayed_event_ids"])
            ),
            "version": OBSERVER_HABIT_REGISTRY_REPLAY_VERSION,
        }
        return cls(
            habit_registry_replay_id=canonical_id(payload),
            version=OBSERVER_HABIT_REGISTRY_REPLAY_VERSION,
            **values,  # type: ignore[arg-type]
        )


def empty_observer_habit_registry_snapshot() -> ObserverHabitRegistrySnapshotDTO:
    return ObserverHabitRegistrySnapshotDTO.create(
        registry_sequence=0,
        entries=(),
        previous_registry_snapshot_id=None,
    )


class InMemoryObserverHabitRegistry:
    def __init__(self) -> None:
        initial = empty_observer_habit_registry_snapshot()
        self._snapshots: dict[str, ObserverHabitRegistrySnapshotDTO] = {
            initial.habit_registry_snapshot_id: initial
        }
        self._current_id = initial.habit_registry_snapshot_id
        self._events: list[ObserverHabitRegistryEventDTO] = []

    def current_snapshot(self) -> ObserverHabitRegistrySnapshotDTO:
        return self._snapshots[self._current_id]

    def events(self) -> tuple[ObserverHabitRegistryEventDTO, ...]:
        return tuple(self._events)

    def snapshots(self) -> tuple[ObserverHabitRegistrySnapshotDTO, ...]:
        return tuple(self._snapshots.values())

    def get_entry(
        self, habit_specification_id: str
    ) -> ObserverHabitRegistryEntryDTO | None:
        return _entry_for(self.current_snapshot(), habit_specification_id)

    def register_admission(
        self,
        *,
        habit_specification: ObserverHabitSpecificationDTO,
        admission_decision: ObserverHabitAdmissionDecisionDTO,
    ) -> ObserverHabitRegistryEventDTO | None:
        if admission_decision.decision != "admit":
            return None
        if (
            admission_decision.habit_specification_id
            != habit_specification.habit_specification_id
        ):
            raise ObserverHabitRegistryError("admission decision habit mismatch")
        source = self.current_snapshot()
        if _entry_for(source, habit_specification.habit_specification_id) is not None:
            raise ObserverHabitRegistryError("habit already registered")
        entry = ObserverHabitRegistryEntryDTO.create(
            habit_specification_id=habit_specification.habit_specification_id,
            habit_admission_decision_id=admission_decision.habit_admission_decision_id,
            status="admitted_inactive",
            activation_generation=0,
            active_since_registry_sequence=None,
            suspended_since_registry_sequence=None,
            retired_since_registry_sequence=None,
            status_reason_codes=("admitted",),
        )
        return self._append_event(
            source=source,
            event_type="admitted",
            habit_specification_id=habit_specification.habit_specification_id,
            admission_decision_id=admission_decision.habit_admission_decision_id,
            new_entries=source.entries + (entry,),
            reason_codes=("admitted",),
        )

    def activate(
        self,
        *,
        habit_specification_id: str,
        expected_source_registry_snapshot_id: str,
        reason_codes: tuple[str, ...] = ("activated",),
    ) -> ObserverHabitRegistryEventDTO:
        source = self.current_snapshot()
        if source.habit_registry_snapshot_id != expected_source_registry_snapshot_id:
            raise ObserverHabitRegistryError("stale registry snapshot")
        entry = _require_entry(source, habit_specification_id)
        if entry.status != "admitted_inactive":
            raise ObserverHabitRegistryError("habit is not admitted inactive")
        if source.active_habit_ids:
            raise ObserverHabitRegistryError("active habit conflict")
        sequence = source.registry_sequence + 1
        activated = _entry_with_status(
            entry,
            status="active",
            sequence=sequence,
            reason_codes=reason_codes,
            activation_generation=entry.activation_generation + 1,
        )
        event = self._append_event(
            source=source,
            event_type="activated",
            habit_specification_id=habit_specification_id,
            admission_decision_id=entry.habit_admission_decision_id,
            new_entries=_replace_entry(source.entries, activated),
            reason_codes=reason_codes,
        )
        return event

    def deactivate(
        self,
        *,
        habit_specification_id: str,
        reason_codes: tuple[str, ...] = ("deactivated",),
    ) -> ObserverHabitRegistryEventDTO:
        return self._transition(
            habit_specification_id,
            "active",
            "admitted_inactive",
            "deactivated",
            reason_codes,
        )

    def suspend(
        self,
        *,
        habit_specification_id: str,
        reason_codes: tuple[str, ...] = ("suspended",),
    ) -> ObserverHabitRegistryEventDTO:
        return self._transition(
            habit_specification_id, "active", "suspended", "suspended", reason_codes
        )

    def resume(
        self,
        *,
        habit_specification_id: str,
        reason_codes: tuple[str, ...] = ("resumed",),
    ) -> ObserverHabitRegistryEventDTO:
        return self._transition(
            habit_specification_id,
            "suspended",
            "admitted_inactive",
            "resumed",
            reason_codes,
        )

    def retire(
        self,
        *,
        habit_specification_id: str,
        reason_codes: tuple[str, ...] = ("retired",),
    ) -> ObserverHabitRegistryEventDTO:
        source = self.current_snapshot()
        entry = _require_entry(source, habit_specification_id)
        if entry.status not in {"admitted_inactive", "suspended"}:
            raise ObserverHabitRegistryError(
                "habit must be inactive or suspended to retire"
            )
        return self._transition(
            habit_specification_id, entry.status, "retired", "retired", reason_codes
        )

    def rollback_to(
        self, request: ObserverHabitRollbackRequestDTO
    ) -> ObserverHabitRollbackResultDTO:
        source = self.current_snapshot()
        if source.habit_registry_snapshot_id != request.current_registry_snapshot_id:
            return _rollback_result(
                request,
                source,
                None,
                None,
                "stale_registry_snapshot",
                ("stale_registry_snapshot",),
            )
        target = self._snapshots.get(request.target_registry_snapshot_id)
        if target is None:
            return _rollback_result(
                request, source, None, None, "target_not_found", ("target_not_found",)
            )
        if not _is_ancestor(target.habit_registry_snapshot_id, source, self._snapshots):
            return _rollback_result(
                request,
                source,
                target,
                None,
                "target_not_ancestor",
                ("target_not_ancestor",),
            )
        event = self._append_event(
            source=source,
            event_type="rolled_back",
            habit_specification_id="registry",
            admission_decision_id=None,
            previous_registry_snapshot_id=target.habit_registry_snapshot_id,
            new_entries=target.entries,
            reason_codes=request.reason_codes or ("rolled_back",),
        )
        return _rollback_result(
            request,
            source,
            target,
            self.current_snapshot(),
            "rolled_back",
            event.reason_codes,
            event.habit_registry_event_id,
        )

    def verify_integrity(self) -> ObserverHabitRegistryReplayDTO:
        return replay_observer_habit_registry(
            initial_snapshot=empty_observer_habit_registry_snapshot(),
            events=self.events(),
            final_snapshot=self.current_snapshot(),
            snapshots=self.snapshots(),
        )

    def replay(self) -> ObserverHabitRegistryReplayDTO:
        return self.verify_integrity()

    def _transition(
        self,
        habit_id: str,
        from_status: str,
        to_status: str,
        event_type: str,
        reason_codes: tuple[str, ...],
    ) -> ObserverHabitRegistryEventDTO:
        source = self.current_snapshot()
        entry = _require_entry(source, habit_id)
        if entry.status != from_status:
            raise ObserverHabitRegistryError("illegal registry status transition")
        sequence = source.registry_sequence + 1
        updated = _entry_with_status(
            entry, status=to_status, sequence=sequence, reason_codes=reason_codes
        )
        return self._append_event(
            source=source,
            event_type=event_type,
            habit_specification_id=habit_id,
            admission_decision_id=entry.habit_admission_decision_id,
            new_entries=_replace_entry(source.entries, updated),
            reason_codes=reason_codes,
        )

    def _append_event(
        self,
        *,
        source: ObserverHabitRegistrySnapshotDTO,
        event_type: str,
        habit_specification_id: str,
        admission_decision_id: str | None,
        new_entries: tuple[ObserverHabitRegistryEntryDTO, ...],
        reason_codes: tuple[str, ...],
        previous_registry_snapshot_id: str | None = None,
    ) -> ObserverHabitRegistryEventDTO:
        provisional = ObserverHabitRegistrySnapshotDTO.create(
            registry_sequence=source.registry_sequence + 1,
            entries=new_entries,
            previous_registry_snapshot_id=source.habit_registry_snapshot_id,
        )
        event = ObserverHabitRegistryEventDTO.create(
            registry_sequence=source.registry_sequence + 1,
            event_type=event_type,
            habit_specification_id=habit_specification_id,
            habit_registry_entry_id=(
                None
                if habit_specification_id == "registry"
                else _require_entry(
                    provisional, habit_specification_id
                ).habit_registry_entry_id
            ),
            admission_decision_id=admission_decision_id,
            previous_registry_snapshot_id=previous_registry_snapshot_id,
            source_registry_snapshot_id=source.habit_registry_snapshot_id,
            reason_codes=reason_codes,
        )
        self._events.append(event)
        self._snapshots[provisional.habit_registry_snapshot_id] = provisional
        self._current_id = provisional.habit_registry_snapshot_id
        return event


def replay_observer_habit_registry(
    *,
    initial_snapshot: ObserverHabitRegistrySnapshotDTO,
    events: tuple[ObserverHabitRegistryEventDTO, ...],
    final_snapshot: ObserverHabitRegistrySnapshotDTO,
    snapshots: tuple[ObserverHabitRegistrySnapshotDTO, ...] = (),
) -> ObserverHabitRegistryReplayDTO:
    failures: set[str] = set()
    replayed: list[str] = []
    current = initial_snapshot
    known = {initial_snapshot.habit_registry_snapshot_id: initial_snapshot}
    supplied = {snapshot.habit_registry_snapshot_id: snapshot for snapshot in snapshots}
    for expected_sequence, event in enumerate(events, start=1):
        replayed.append(event.habit_registry_event_id)
        if event.registry_sequence != expected_sequence:
            failures.add("sequence_gap")
        if event.source_registry_snapshot_id != current.habit_registry_snapshot_id:
            failures.add("event_source_snapshot_mismatch")
        result, event_failures = _apply_registry_event(
            source=current,
            event=event,
            known_snapshots={**known, **supplied},
        )
        failures.update(event_failures)
        supplied_result = _supplied_result_for_event(
            source=current, event=event, supplied=supplied
        )
        if supplied_result is None:
            failures.add("result_snapshot_missing")
        else:
            if (
                supplied_result.habit_registry_snapshot_id
                != result.habit_registry_snapshot_id
            ):
                failures.add("event_application_mismatch")
            if supplied_result.entries != result.entries:
                failures.add("event_application_mismatch")
                failures.update(
                    _entry_difference_failures(
                        expected=result.entries,
                        observed=supplied_result.entries,
                        event=event,
                    )
                )
        known[result.habit_registry_snapshot_id] = result
        current = result
    if current.habit_registry_snapshot_id != final_snapshot.habit_registry_snapshot_id:
        failures.add("final_snapshot_mismatch")
    if current.entries != final_snapshot.entries:
        failures.add("final_snapshot_mismatch")
    status = "verified" if not failures else "failed"
    return ObserverHabitRegistryReplayDTO.create(
        initial_snapshot_id=initial_snapshot.habit_registry_snapshot_id,
        final_snapshot_id=final_snapshot.habit_registry_snapshot_id,
        replayed_event_ids=tuple(replayed),
        status=status,
        failure_codes=tuple(sorted(failures)),
    )


def _apply_registry_event(
    *,
    source: ObserverHabitRegistrySnapshotDTO,
    event: ObserverHabitRegistryEventDTO,
    known_snapshots: Mapping[str, ObserverHabitRegistrySnapshotDTO],
) -> tuple[ObserverHabitRegistrySnapshotDTO, set[str]]:
    failures: set[str] = set()
    entries = source.entries
    habit_id = event.habit_specification_id
    entry = None if habit_id == "registry" else _entry_for(source, habit_id)
    affected_entry: ObserverHabitRegistryEntryDTO | None = None

    if (
        event.event_type != "rolled_back"
        and event.previous_registry_snapshot_id is not None
    ):
        failures.add("unexpected_rollback_target")
    if event.event_type == "rolled_back":
        if event.habit_specification_id != "registry":
            failures.add("event_application_mismatch")
        if event.habit_registry_entry_id is not None:
            failures.add("event_entry_id_mismatch")
        if event.admission_decision_id is not None:
            failures.add("admission_decision_mismatch")
    elif entry is not None and event.admission_decision_id != (
        entry.habit_admission_decision_id
    ):
        failures.add("admission_decision_mismatch")

    if event.event_type == "admitted":
        if entry is not None:
            failures.add("unexpected_entry_added")
            result_entries = entries
        elif event.admission_decision_id is None:
            failures.add("event_application_mismatch")
            result_entries = entries
        else:
            added = ObserverHabitRegistryEntryDTO.create(
                habit_specification_id=habit_id,
                habit_admission_decision_id=event.admission_decision_id,
                status="admitted_inactive",
                activation_generation=0,
                active_since_registry_sequence=None,
                suspended_since_registry_sequence=None,
                retired_since_registry_sequence=None,
                status_reason_codes=event.reason_codes,
            )
            affected_entry = added
            result_entries = entries + (added,)
    elif event.event_type == "activated":
        if entry is None:
            failures.add("unexpected_entry_removed")
            result_entries = entries
        elif entry.status != "admitted_inactive":
            failures.add("illegal_status_transition")
            result_entries = entries
        elif source.active_habit_ids:
            failures.add("event_application_mismatch")
            result_entries = entries
        else:
            affected_entry = _entry_with_status(
                entry,
                status="active",
                sequence=event.registry_sequence,
                reason_codes=event.reason_codes,
                activation_generation=entry.activation_generation + 1,
            )
            result_entries = _replace_entry(
                entries,
                affected_entry,
            )
    elif event.event_type == "deactivated":
        result_entries, affected_entry = _apply_status_transition(
            entries, entry, "active", "admitted_inactive", event, failures
        )
    elif event.event_type == "suspended":
        result_entries, affected_entry = _apply_status_transition(
            entries, entry, "active", "suspended", event, failures
        )
    elif event.event_type == "resumed":
        result_entries, affected_entry = _apply_status_transition(
            entries, entry, "suspended", "admitted_inactive", event, failures
        )
    elif event.event_type == "retired":
        if entry is None:
            failures.add("unexpected_entry_removed")
            result_entries = entries
        elif entry.status not in {"admitted_inactive", "suspended"}:
            failures.add("illegal_status_transition")
            result_entries = entries
        else:
            affected_entry = _entry_with_status(
                entry,
                status="retired",
                sequence=event.registry_sequence,
                reason_codes=event.reason_codes,
            )
            result_entries = _replace_entry(
                entries,
                affected_entry,
            )
    elif event.event_type == "rolled_back":
        target_id = event.previous_registry_snapshot_id
        target = None if target_id is None else known_snapshots.get(target_id)
        if target is None:
            failures.add("rollback_content_mismatch")
            result_entries = entries
        elif not _is_ancestor(
            target.habit_registry_snapshot_id, source, known_snapshots
        ):
            failures.add("rollback_content_mismatch")
            result_entries = entries
        else:
            result_entries = target.entries
    else:
        failures.add("event_application_mismatch")
        result_entries = entries

    result = ObserverHabitRegistrySnapshotDTO.create(
        registry_sequence=event.registry_sequence,
        entries=result_entries,
        previous_registry_snapshot_id=source.habit_registry_snapshot_id,
    )
    if affected_entry is not None:
        if event.habit_registry_entry_id != affected_entry.habit_registry_entry_id:
            failures.add("event_entry_id_mismatch")
        if event.admission_decision_id != affected_entry.habit_admission_decision_id:
            failures.add("admission_decision_mismatch")
    return result, failures


def _apply_status_transition(
    entries: tuple[ObserverHabitRegistryEntryDTO, ...],
    entry: ObserverHabitRegistryEntryDTO | None,
    from_status: str,
    to_status: str,
    event: ObserverHabitRegistryEventDTO,
    failures: set[str],
) -> tuple[
    tuple[ObserverHabitRegistryEntryDTO, ...], ObserverHabitRegistryEntryDTO | None
]:
    if entry is None:
        failures.add("unexpected_entry_removed")
        return entries, None
    if entry.status != from_status:
        failures.add("illegal_status_transition")
        return entries, None
    updated = _entry_with_status(
        entry,
        status=to_status,
        sequence=event.registry_sequence,
        reason_codes=event.reason_codes,
    )
    return _replace_entry(entries, updated), updated


def _supplied_result_for_event(
    *,
    source: ObserverHabitRegistrySnapshotDTO,
    event: ObserverHabitRegistryEventDTO,
    supplied: Mapping[str, ObserverHabitRegistrySnapshotDTO],
) -> ObserverHabitRegistrySnapshotDTO | None:
    matches = [
        snapshot
        for snapshot in supplied.values()
        if snapshot.registry_sequence == event.registry_sequence
        and snapshot.previous_registry_snapshot_id == source.habit_registry_snapshot_id
    ]
    return matches[0] if len(matches) == 1 else None


def _entry_difference_failures(
    *,
    expected: tuple[ObserverHabitRegistryEntryDTO, ...],
    observed: tuple[ObserverHabitRegistryEntryDTO, ...],
    event: ObserverHabitRegistryEventDTO,
) -> set[str]:
    failures: set[str] = set()
    expected_by_id = {item.habit_specification_id: item for item in expected}
    observed_by_id = {item.habit_specification_id: item for item in observed}
    if set(observed_by_id) - set(expected_by_id):
        failures.add("unexpected_entry_added")
    if set(expected_by_id) - set(observed_by_id):
        failures.add("unexpected_entry_removed")
    for habit_id in set(expected_by_id) & set(observed_by_id):
        expected_entry = expected_by_id[habit_id]
        observed_entry = observed_by_id[habit_id]
        if expected_entry.activation_generation != observed_entry.activation_generation:
            failures.add("activation_generation_mismatch")
        if (
            expected_entry != observed_entry
            and habit_id != event.habit_specification_id
        ):
            failures.add("unrelated_entry_changed")
        if expected_entry.status != observed_entry.status:
            failures.add("illegal_status_transition")
    if event.event_type == "rolled_back" and expected != observed:
        failures.add("rollback_content_mismatch")
    return failures


def _entry_for(
    snapshot: ObserverHabitRegistrySnapshotDTO, habit_id: str
) -> ObserverHabitRegistryEntryDTO | None:
    for entry in snapshot.entries:
        if entry.habit_specification_id == habit_id:
            return entry
    return None


def _require_entry(
    snapshot: ObserverHabitRegistrySnapshotDTO, habit_id: str
) -> ObserverHabitRegistryEntryDTO:
    entry = _entry_for(snapshot, habit_id)
    if entry is None:
        raise ObserverHabitRegistryError("habit is not registered")
    return entry


def _replace_entry(
    entries: tuple[ObserverHabitRegistryEntryDTO, ...],
    updated: ObserverHabitRegistryEntryDTO,
) -> tuple[ObserverHabitRegistryEntryDTO, ...]:
    return tuple(
        updated
        if item.habit_specification_id == updated.habit_specification_id
        else item
        for item in entries
    )


def _entry_with_status(
    entry: ObserverHabitRegistryEntryDTO,
    *,
    status: str,
    sequence: int,
    reason_codes: tuple[str, ...],
    activation_generation: int | None = None,
) -> ObserverHabitRegistryEntryDTO:
    return ObserverHabitRegistryEntryDTO.create(
        habit_specification_id=entry.habit_specification_id,
        habit_admission_decision_id=entry.habit_admission_decision_id,
        status=status,
        activation_generation=entry.activation_generation
        if activation_generation is None
        else activation_generation,
        active_since_registry_sequence=sequence if status == "active" else None,
        suspended_since_registry_sequence=sequence if status == "suspended" else None,
        retired_since_registry_sequence=sequence
        if status == "retired"
        else entry.retired_since_registry_sequence,
        status_reason_codes=reason_codes,
    )


def _is_ancestor(
    target_id: str,
    source: ObserverHabitRegistrySnapshotDTO,
    snapshots: Mapping[str, ObserverHabitRegistrySnapshotDTO],
) -> bool:
    current: ObserverHabitRegistrySnapshotDTO | None = source
    while current is not None:
        if current.habit_registry_snapshot_id == target_id:
            return True
        previous_id = current.previous_registry_snapshot_id
        current = None if previous_id is None else snapshots.get(previous_id)
    return False


def _rollback_result(
    request: ObserverHabitRollbackRequestDTO,
    source: ObserverHabitRegistrySnapshotDTO,
    target: ObserverHabitRegistrySnapshotDTO | None,
    result: ObserverHabitRegistrySnapshotDTO | None,
    disposition: str,
    reasons: tuple[str, ...],
    event_id: str | None = None,
) -> ObserverHabitRollbackResultDTO:
    return ObserverHabitRollbackResultDTO.create(
        rollback_request_id=request.habit_rollback_request_id,
        source_registry_snapshot_id=source.habit_registry_snapshot_id,
        target_registry_snapshot_id=(
            request.target_registry_snapshot_id
            if target is None
            else target.habit_registry_snapshot_id
        ),
        result_registry_snapshot_id=None
        if result is None
        else result.habit_registry_snapshot_id,
        registry_event_id=event_id,
        disposition=disposition,
        reason_codes=reasons,
    )
