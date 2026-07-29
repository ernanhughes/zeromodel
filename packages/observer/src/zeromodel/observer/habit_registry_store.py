"""DTO-only Store boundary for durable Observer habit registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, Sequence, cast

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.habit import ObserverHabitError
from zeromodel.observer.habit_registry import (
    ObserverHabitRegistryEntryDTO,
    ObserverHabitRegistryEventDTO,
    ObserverHabitRegistrySnapshotDTO,
)

OBSERVER_HABIT_REGISTRY_STORE_COMMIT_VERSION: Final = (
    "observer-habit-registry-store-commit/1"
)
OBSERVER_HABIT_REGISTRY_STORE_VERIFICATION_VERSION: Final = (
    "observer-habit-registry-store-verification/1"
)
OBSERVER_HABIT_REGISTRY_RECOVERY_VERSION: Final = "observer-habit-registry-recovery/1"

STORE_COMMIT_DISPOSITIONS: Final = frozenset(
    {
        "committed",
        "stale_source_snapshot",
        "source_snapshot_missing",
        "event_invalid",
        "semantic_replay_failed",
        "snapshot_mismatch",
        "database_locked",
        "database_integrity_failure",
        "unsupported",
    }
)
STORE_VERIFICATION_STATUSES: Final = frozenset({"verified", "failed", "inconclusive"})
STORE_RECOVERY_DISPOSITIONS: Final = frozenset(
    {"recovered", "not_needed", "unsafe_to_recover", "unsupported", "failed"}
)


class ObserverHabitRegistryStoreError(ObserverHabitError):
    """Raised when durable registry storage cannot satisfy the DTO contract."""


class ObserverHabitRegistryDatabaseError(ObserverHabitRegistryStoreError):
    """Raised for unexpected SQLite failures."""


class ObserverHabitRegistrySchemaError(ObserverHabitRegistryStoreError):
    """Raised when the SQLite schema is unsupported or contradictory."""


def _sorted_codes(codes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(codes)))


@dataclass(frozen=True)
class ObserverHabitRegistryStoreCommitDTO:
    store_commit_id: str
    store_id: str
    source_snapshot_id: str
    result_snapshot_id: str | None
    event_id: str | None
    registry_sequence: int | None
    disposition: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_REGISTRY_STORE_COMMIT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_REGISTRY_STORE_COMMIT_VERSION:
            raise ObserverHabitRegistryStoreError("unsupported store commit version")
        if self.disposition not in STORE_COMMIT_DISPOSITIONS:
            raise ObserverHabitRegistryStoreError("unsupported store disposition")
        if self.disposition == "committed" and (
            self.result_snapshot_id is None
            or self.event_id is None
            or self.registry_sequence is None
        ):
            raise ObserverHabitRegistryStoreError("committed result is incomplete")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.store_commit_id != expected:
            raise ObserverHabitRegistryStoreError("store_commit_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "disposition": self.disposition,
            "event_id": self.event_id,
            "reason_codes": list(self.reason_codes),
            "registry_sequence": self.registry_sequence,
            "result_snapshot_id": self.result_snapshot_id,
            "source_snapshot_id": self.source_snapshot_id,
            "store_id": self.store_id,
            "version": self.version,
        }
        if include_id:
            payload["store_commit_id"] = self.store_commit_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitRegistryStoreCommitDTO":
        values["reason_codes"] = _sorted_codes(
            values.get("reason_codes", ())  # type: ignore[arg-type]
        )
        payload = {
            **values,
            "reason_codes": list(cast(Sequence[str], values["reason_codes"])),
            "version": OBSERVER_HABIT_REGISTRY_STORE_COMMIT_VERSION,
        }
        return cls(
            store_commit_id=canonical_id(payload),
            version=OBSERVER_HABIT_REGISTRY_STORE_COMMIT_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitRegistryStoreVerificationDTO:
    store_verification_id: str
    store_id: str
    head_snapshot_id: str | None
    head_registry_sequence: int | None
    event_count: int
    snapshot_count: int
    entry_row_count: int
    semantic_replay_id: str | None
    status: str
    failure_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_REGISTRY_STORE_VERIFICATION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_REGISTRY_STORE_VERIFICATION_VERSION:
            raise ObserverHabitRegistryStoreError(
                "unsupported store verification version"
            )
        if self.status not in STORE_VERIFICATION_STATUSES:
            raise ObserverHabitRegistryStoreError("unsupported verification status")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.store_verification_id != expected:
            raise ObserverHabitRegistryStoreError("store_verification_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "entry_row_count": self.entry_row_count,
            "event_count": self.event_count,
            "failure_codes": list(self.failure_codes),
            "head_registry_sequence": self.head_registry_sequence,
            "head_snapshot_id": self.head_snapshot_id,
            "semantic_replay_id": self.semantic_replay_id,
            "snapshot_count": self.snapshot_count,
            "status": self.status,
            "store_id": self.store_id,
            "version": self.version,
        }
        if include_id:
            payload["store_verification_id"] = self.store_verification_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitRegistryStoreVerificationDTO":
        values["failure_codes"] = _sorted_codes(
            values.get("failure_codes", ())  # type: ignore[arg-type]
        )
        payload = {
            **values,
            "failure_codes": list(cast(Sequence[str], values["failure_codes"])),
            "version": OBSERVER_HABIT_REGISTRY_STORE_VERIFICATION_VERSION,
        }
        return cls(
            store_verification_id=canonical_id(payload),
            version=OBSERVER_HABIT_REGISTRY_STORE_VERIFICATION_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitRegistryRecoveryDTO:
    habit_registry_recovery_id: str
    store_id: str
    recovered_head_snapshot_id: str | None
    recovered_registry_sequence: int | None
    disposition: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_REGISTRY_RECOVERY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_REGISTRY_RECOVERY_VERSION:
            raise ObserverHabitRegistryStoreError("unsupported recovery version")
        if self.disposition not in STORE_RECOVERY_DISPOSITIONS:
            raise ObserverHabitRegistryStoreError("unsupported recovery disposition")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_registry_recovery_id != expected:
            raise ObserverHabitRegistryStoreError("habit_registry_recovery_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
            "recovered_head_snapshot_id": self.recovered_head_snapshot_id,
            "recovered_registry_sequence": self.recovered_registry_sequence,
            "store_id": self.store_id,
            "version": self.version,
        }
        if include_id:
            payload["habit_registry_recovery_id"] = self.habit_registry_recovery_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitRegistryRecoveryDTO":
        values["reason_codes"] = _sorted_codes(
            values.get("reason_codes", ())  # type: ignore[arg-type]
        )
        payload = {
            **values,
            "reason_codes": list(cast(Sequence[str], values["reason_codes"])),
            "version": OBSERVER_HABIT_REGISTRY_RECOVERY_VERSION,
        }
        return cls(
            habit_registry_recovery_id=canonical_id(payload),
            version=OBSERVER_HABIT_REGISTRY_RECOVERY_VERSION,
            **values,  # type: ignore[arg-type]
        )


class ObserverHabitRegistryStore(Protocol):
    def initialize(self) -> None: ...

    def load_current_snapshot(self) -> ObserverHabitRegistrySnapshotDTO: ...

    def load_snapshot(
        self, snapshot_id: str
    ) -> ObserverHabitRegistrySnapshotDTO | None: ...

    def load_snapshots(self) -> tuple[ObserverHabitRegistrySnapshotDTO, ...]: ...

    def load_events(self) -> tuple[ObserverHabitRegistryEventDTO, ...]: ...

    def load_event(self, event_id: str) -> ObserverHabitRegistryEventDTO | None: ...

    def load_entry(
        self, habit_specification_id: str, *, snapshot_id: str | None = None
    ) -> ObserverHabitRegistryEntryDTO | None: ...

    def append_transition(
        self,
        *,
        expected_source_snapshot_id: str,
        event: ObserverHabitRegistryEventDTO,
        result_snapshot: ObserverHabitRegistrySnapshotDTO,
    ) -> ObserverHabitRegistryStoreCommitDTO: ...

    def verify_integrity(self) -> ObserverHabitRegistryStoreVerificationDTO: ...

    def recover(self) -> ObserverHabitRegistryRecoveryDTO: ...
