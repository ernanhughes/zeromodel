"""SQLite-backed durable Observer habit registry."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

from zeromodel.observer._canonical import canonical_json
from zeromodel.observer._habit_registry_transition import (
    apply_observer_habit_registry_event,
)
from zeromodel.observer.habit import ObserverHabitSpecificationDTO
from zeromodel.observer.habit_admission import ObserverHabitAdmissionDecisionDTO
from zeromodel.observer.habit_registry import (
    ObserverHabitRegistryEntryDTO,
    ObserverHabitRegistryError,
    ObserverHabitRegistryEventDTO,
    ObserverHabitRegistryReplayDTO,
    ObserverHabitRegistrySnapshotDTO,
    ObserverHabitRollbackRequestDTO,
    ObserverHabitRollbackResultDTO,
    _entry_for,
    _entry_with_status,
    _is_ancestor,
    _replace_entry,
    _require_entry,
    _rollback_result,
    empty_observer_habit_registry_snapshot,
    replay_observer_habit_registry,
)
from zeromodel.observer.habit_registry_store import (
    ObserverHabitRegistryDatabaseError,
    ObserverHabitRegistryRecoveryDTO,
    ObserverHabitRegistrySchemaError,
    ObserverHabitRegistryStoreCommitDTO,
    ObserverHabitRegistryStoreError,
    ObserverHabitRegistryStoreVerificationDTO,
)

SCHEMA_VERSION = "observer-habit-registry-sqlite/1"


def _encode_payload(payload: dict[str, object]) -> str:
    return canonical_json(payload).decode("utf-8")


def _decode_payload(payload: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(payload))


def _entry_from_payload(payload: dict[str, Any]) -> ObserverHabitRegistryEntryDTO:
    return ObserverHabitRegistryEntryDTO(
        habit_registry_entry_id=payload["habit_registry_entry_id"],
        habit_specification_id=payload["habit_specification_id"],
        habit_admission_decision_id=payload["habit_admission_decision_id"],
        status=payload["status"],
        activation_generation=payload["activation_generation"],
        active_since_registry_sequence=payload["active_since_registry_sequence"],
        suspended_since_registry_sequence=payload["suspended_since_registry_sequence"],
        retired_since_registry_sequence=payload["retired_since_registry_sequence"],
        status_reason_codes=tuple(payload["status_reason_codes"]),
        version=payload["version"],
    )


def _event_from_payload(payload: dict[str, Any]) -> ObserverHabitRegistryEventDTO:
    return ObserverHabitRegistryEventDTO(
        habit_registry_event_id=payload["habit_registry_event_id"],
        registry_sequence=payload["registry_sequence"],
        event_type=payload["event_type"],
        habit_specification_id=payload["habit_specification_id"],
        habit_registry_entry_id=payload["habit_registry_entry_id"],
        admission_decision_id=payload["admission_decision_id"],
        previous_registry_snapshot_id=payload["previous_registry_snapshot_id"],
        source_registry_snapshot_id=payload["source_registry_snapshot_id"],
        reason_codes=tuple(payload["reason_codes"]),
        version=payload["version"],
    )


def _snapshot_from_payload(payload: dict[str, Any]) -> ObserverHabitRegistrySnapshotDTO:
    return ObserverHabitRegistrySnapshotDTO(
        habit_registry_snapshot_id=payload["habit_registry_snapshot_id"],
        registry_sequence=payload["registry_sequence"],
        entries=tuple(_entry_from_payload(item) for item in payload["entries"]),
        active_habit_ids=tuple(payload["active_habit_ids"]),
        previous_registry_snapshot_id=payload["previous_registry_snapshot_id"],
        version=payload["version"],
    )


class SqliteObserverHabitRegistryStore:
    """Durable DTO Store for one SQLite habit registry database."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 250,
        journal_mode: Literal["wal", "delete"] = "wal",
        read_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        self.journal_mode = journal_mode
        self.read_only = read_only
        self._verification_read_started_hook: Callable[[], None] | None = None
        if read_only:
            uri = f"file:{self.path.as_posix()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._setup_connection()

    @classmethod
    def open_read_only(
        cls, path: str | Path, *, busy_timeout_ms: int = 250
    ) -> "SqliteObserverHabitRegistryStore":
        return cls(path, busy_timeout_ms=busy_timeout_ms, read_only=True)

    def close(self) -> None:
        self._conn.close()

    def _setup_connection(self) -> None:
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
        if not self.read_only and self.journal_mode == "wal":
            self._conn.execute("PRAGMA journal_mode = WAL")

    def initialize(self) -> None:
        if self.read_only:
            raise ObserverHabitRegistryStoreError("read-only registry store")
        try:
            self._create_schema()
            self._conn.execute("BEGIN IMMEDIATE")
            store_id = self._metadata("store_id")
            if store_id is None:
                self._conn.execute(
                    "INSERT INTO observer_habit_registry_metadata VALUES (?, ?)",
                    ("store_id", f"sqlite-store:{uuid.uuid4().hex}"),
                )
            version = self._metadata("schema_version")
            if version is None:
                self._conn.execute(
                    "INSERT INTO observer_habit_registry_metadata VALUES (?, ?)",
                    ("schema_version", SCHEMA_VERSION),
                )
            elif version != SCHEMA_VERSION:
                raise ObserverHabitRegistrySchemaError("unsupported registry schema")
            empty = empty_observer_habit_registry_snapshot()
            existing_empty = self._load_snapshot_by_sequence_in_transaction(0)
            if existing_empty is None:
                self._insert_snapshot(empty)
            elif existing_empty != empty:
                raise ObserverHabitRegistrySchemaError(
                    "conflicting empty registry snapshot"
                )
            self._conn.execute(
                """
                INSERT OR IGNORE INTO observer_habit_registry_head
                    (singleton_key, current_snapshot_id, current_registry_sequence, generation)
                VALUES (1, ?, 0, 0)
                """,
                (empty.habit_registry_snapshot_id,),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._rollback_quietly()
            raise

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS observer_habit_registry_metadata (
                metadata_key TEXT PRIMARY KEY,
                metadata_value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observer_habit_registry_events (
                registry_sequence INTEGER PRIMARY KEY,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                habit_specification_id TEXT NOT NULL,
                habit_registry_entry_id TEXT,
                admission_decision_id TEXT,
                rollback_target_snapshot_id TEXT,
                source_snapshot_id TEXT NOT NULL,
                canonical_payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observer_habit_registry_snapshots (
                registry_sequence INTEGER PRIMARY KEY,
                snapshot_id TEXT NOT NULL UNIQUE,
                previous_snapshot_id TEXT,
                canonical_payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observer_habit_registry_snapshot_entries (
                snapshot_id TEXT NOT NULL,
                habit_specification_id TEXT NOT NULL,
                registry_entry_id TEXT NOT NULL,
                status TEXT NOT NULL,
                activation_generation INTEGER NOT NULL,
                canonical_payload_json TEXT NOT NULL,
                PRIMARY KEY (snapshot_id, habit_specification_id),
                UNIQUE (snapshot_id, registry_entry_id),
                FOREIGN KEY (snapshot_id)
                    REFERENCES observer_habit_registry_snapshots(snapshot_id)
            );
            CREATE TABLE IF NOT EXISTS observer_habit_registry_head (
                singleton_key INTEGER PRIMARY KEY CHECK (singleton_key = 1),
                current_snapshot_id TEXT NOT NULL,
                current_registry_sequence INTEGER NOT NULL,
                generation INTEGER NOT NULL
            );
            """
        )

    def store_id(self) -> str:
        store_id = self._metadata("store_id")
        if store_id is None:
            raise ObserverHabitRegistrySchemaError("missing store ID")
        return store_id

    def _metadata(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT metadata_value FROM observer_habit_registry_metadata WHERE metadata_key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row["metadata_value"])

    def _check_schema(self) -> None:
        if self._metadata("schema_version") != SCHEMA_VERSION:
            raise ObserverHabitRegistrySchemaError("unsupported registry schema")

    def load_current_snapshot(self) -> ObserverHabitRegistrySnapshotDTO:
        self._check_schema()
        self._conn.execute("BEGIN")
        try:
            row = self._conn.execute(
                "SELECT current_snapshot_id, current_registry_sequence FROM observer_habit_registry_head WHERE singleton_key = 1"
            ).fetchone()
            if row is None:
                raise ObserverHabitRegistrySchemaError("missing registry head")
            snapshot = self._load_snapshot_in_transaction(row["current_snapshot_id"])
            if snapshot is None:
                raise ObserverHabitRegistrySchemaError("head snapshot missing")
            if snapshot.registry_sequence != row["current_registry_sequence"]:
                raise ObserverHabitRegistrySchemaError("head sequence mismatch")
            self._conn.execute("COMMIT")
            return snapshot
        except Exception:
            self._rollback_quietly()
            raise

    def load_snapshot(
        self, snapshot_id: str
    ) -> ObserverHabitRegistrySnapshotDTO | None:
        self._check_schema()
        self._conn.execute("BEGIN")
        try:
            snapshot = self._load_snapshot_in_transaction(snapshot_id)
            self._conn.execute("COMMIT")
            return snapshot
        except Exception:
            self._rollback_quietly()
            raise

    def _load_snapshot_in_transaction(
        self, snapshot_id: str
    ) -> ObserverHabitRegistrySnapshotDTO | None:
        row = self._conn.execute(
            """
            SELECT registry_sequence, snapshot_id, previous_snapshot_id, canonical_payload_json
            FROM observer_habit_registry_snapshots
            WHERE snapshot_id = ?
            """,
            (snapshot_id,),
        ).fetchone()
        if row is None:
            return None
        snapshot = _snapshot_from_payload(
            _decode_payload(row["canonical_payload_json"])
        )
        if (
            snapshot.habit_registry_snapshot_id != row["snapshot_id"]
            or snapshot.registry_sequence != row["registry_sequence"]
            or snapshot.previous_registry_snapshot_id != row["previous_snapshot_id"]
        ):
            raise ObserverHabitRegistrySchemaError("snapshot projection mismatch")
        entry_rows = self._conn.execute(
            """
            SELECT habit_specification_id, registry_entry_id, status,
                   activation_generation, canonical_payload_json
            FROM observer_habit_registry_snapshot_entries
            WHERE snapshot_id = ?
            ORDER BY habit_specification_id
            """,
            (snapshot_id,),
        ).fetchall()
        entries = tuple(
            _entry_from_payload(_decode_payload(item["canonical_payload_json"]))
            for item in entry_rows
        )
        for item, entry in zip(entry_rows, entries):
            if (
                item["habit_specification_id"] != entry.habit_specification_id
                or item["registry_entry_id"] != entry.habit_registry_entry_id
                or item["status"] != entry.status
                or item["activation_generation"] != entry.activation_generation
            ):
                raise ObserverHabitRegistrySchemaError("entry projection mismatch")
        if entries != snapshot.entries:
            raise ObserverHabitRegistrySchemaError("snapshot entry mismatch")
        return snapshot

    def _load_snapshot_by_sequence_in_transaction(
        self, registry_sequence: int
    ) -> ObserverHabitRegistrySnapshotDTO | None:
        row = self._conn.execute(
            """
            SELECT snapshot_id
            FROM observer_habit_registry_snapshots
            WHERE registry_sequence = ?
            """,
            (registry_sequence,),
        ).fetchone()
        if row is None:
            return None
        return self._load_snapshot_in_transaction(row["snapshot_id"])

    def load_snapshots(self) -> tuple[ObserverHabitRegistrySnapshotDTO, ...]:
        self._check_schema()
        self._conn.execute("BEGIN")
        try:
            snapshots = self._load_snapshots_in_transaction()
            self._conn.execute("COMMIT")
            return snapshots
        except Exception:
            self._rollback_quietly()
            raise

    def load_event(self, event_id: str) -> ObserverHabitRegistryEventDTO | None:
        row = self._conn.execute(
            """
            SELECT registry_sequence, event_id, event_type, habit_specification_id,
                   habit_registry_entry_id, admission_decision_id,
                   rollback_target_snapshot_id, source_snapshot_id,
                   canonical_payload_json
            FROM observer_habit_registry_events
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        event = _event_from_payload(_decode_payload(row["canonical_payload_json"]))
        if (
            event.registry_sequence != row["registry_sequence"]
            or event.habit_registry_event_id != row["event_id"]
            or event.event_type != row["event_type"]
            or event.habit_specification_id != row["habit_specification_id"]
            or event.habit_registry_entry_id != row["habit_registry_entry_id"]
            or event.admission_decision_id != row["admission_decision_id"]
            or event.previous_registry_snapshot_id != row["rollback_target_snapshot_id"]
            or event.source_registry_snapshot_id != row["source_snapshot_id"]
        ):
            raise ObserverHabitRegistrySchemaError("event projection mismatch")
        return event

    def load_events(self) -> tuple[ObserverHabitRegistryEventDTO, ...]:
        self._check_schema()
        self._conn.execute("BEGIN")
        try:
            events = self._load_events_in_transaction()
            self._conn.execute("COMMIT")
            return events
        except Exception:
            self._rollback_quietly()
            raise

    def load_entry(
        self, habit_specification_id: str, *, snapshot_id: str | None = None
    ) -> ObserverHabitRegistryEntryDTO | None:
        snapshot = (
            self.load_current_snapshot()
            if snapshot_id is None
            else self.load_snapshot(snapshot_id)
        )
        return (
            None if snapshot is None else _entry_for(snapshot, habit_specification_id)
        )

    def append_transition(
        self,
        *,
        expected_source_snapshot_id: str,
        event: ObserverHabitRegistryEventDTO,
        result_snapshot: ObserverHabitRegistrySnapshotDTO,
    ) -> ObserverHabitRegistryStoreCommitDTO:
        if self.read_only:
            return self._commit_result(
                expected_source_snapshot_id, None, None, "unsupported", ("read_only",)
            )
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            head = self._conn.execute(
                "SELECT current_snapshot_id, current_registry_sequence FROM observer_habit_registry_head WHERE singleton_key = 1"
            ).fetchone()
            if head is None:
                self._conn.execute("ROLLBACK")
                return self._commit_result(
                    expected_source_snapshot_id,
                    None,
                    None,
                    "database_integrity_failure",
                    ("missing_head",),
                )
            if head["current_snapshot_id"] != expected_source_snapshot_id:
                self._conn.execute("ROLLBACK")
                return self._commit_result(
                    expected_source_snapshot_id,
                    None,
                    None,
                    "stale_source_snapshot",
                    ("stale_source_snapshot",),
                )
            if event.source_registry_snapshot_id != expected_source_snapshot_id:
                self._conn.execute("ROLLBACK")
                return self._commit_result(
                    expected_source_snapshot_id,
                    None,
                    None,
                    "event_invalid",
                    ("event_source_snapshot_mismatch",),
                )
            if event.registry_sequence != head["current_registry_sequence"] + 1:
                self._conn.execute("ROLLBACK")
                return self._commit_result(
                    expected_source_snapshot_id,
                    None,
                    None,
                    "event_invalid",
                    ("sequence_gap",),
                )
            source = self._load_snapshot_in_transaction(expected_source_snapshot_id)
            if source is None:
                self._conn.execute("ROLLBACK")
                return self._commit_result(
                    expected_source_snapshot_id,
                    None,
                    None,
                    "source_snapshot_missing",
                    ("source_snapshot_missing",),
                )
            snapshots = {
                item.habit_registry_snapshot_id: item
                for item in self._load_snapshots_in_transaction()
            }
            replayed, failures = apply_observer_habit_registry_event(
                source=source, event=event, known_snapshots=snapshots
            )
            if failures:
                self._conn.execute("ROLLBACK")
                return self._commit_result(
                    expected_source_snapshot_id,
                    None,
                    None,
                    "semantic_replay_failed",
                    failures,
                )
            if replayed != result_snapshot:
                self._conn.execute("ROLLBACK")
                return self._commit_result(
                    expected_source_snapshot_id,
                    None,
                    None,
                    "snapshot_mismatch",
                    ("snapshot_mismatch",),
                )
            self._insert_event(event)
            self._insert_snapshot(result_snapshot)
            persisted = self._load_snapshot_in_transaction(
                result_snapshot.habit_registry_snapshot_id
            )
            entry_count = self._snapshot_entry_row_count(
                result_snapshot.habit_registry_snapshot_id
            )
            if persisted != result_snapshot or entry_count != len(
                result_snapshot.entries
            ):
                self._conn.execute("ROLLBACK")
                return self._commit_result(
                    expected_source_snapshot_id,
                    None,
                    None,
                    "database_integrity_failure",
                    ("persisted_snapshot_mismatch",),
                )
            cursor = self._conn.execute(
                """
                UPDATE observer_habit_registry_head
                SET current_snapshot_id = ?,
                    current_registry_sequence = ?,
                    generation = generation + 1
                WHERE singleton_key = 1
                  AND current_snapshot_id = ?
                  AND current_registry_sequence = ?
                """,
                (
                    result_snapshot.habit_registry_snapshot_id,
                    result_snapshot.registry_sequence,
                    expected_source_snapshot_id,
                    source.registry_sequence,
                ),
            )
            if cursor.rowcount != 1:
                self._conn.execute("ROLLBACK")
                return self._commit_result(
                    expected_source_snapshot_id,
                    None,
                    None,
                    "stale_source_snapshot",
                    ("stale_source_snapshot",),
                )
            self._conn.execute("COMMIT")
            return self._commit_result(
                expected_source_snapshot_id,
                result_snapshot,
                event,
                "committed",
                ("committed",),
            )
        except sqlite3.OperationalError as exc:
            self._rollback_quietly()
            if "locked" in str(exc).lower():
                return self._commit_result(
                    expected_source_snapshot_id,
                    None,
                    None,
                    "database_locked",
                    ("database_locked",),
                )
            raise ObserverHabitRegistryDatabaseError("sqlite registry failure") from exc
        except sqlite3.IntegrityError:
            self._rollback_quietly()
            return self._commit_result(
                expected_source_snapshot_id,
                None,
                None,
                "database_integrity_failure",
                ("database_integrity_failure",),
            )

    def _insert_event(self, event: ObserverHabitRegistryEventDTO) -> None:
        self._conn.execute(
            """
            INSERT INTO observer_habit_registry_events
                (registry_sequence, event_id, event_type, habit_specification_id,
                 habit_registry_entry_id, admission_decision_id,
                 rollback_target_snapshot_id, source_snapshot_id,
                 canonical_payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.registry_sequence,
                event.habit_registry_event_id,
                event.event_type,
                event.habit_specification_id,
                event.habit_registry_entry_id,
                event.admission_decision_id,
                event.previous_registry_snapshot_id,
                event.source_registry_snapshot_id,
                _encode_payload(dict(event.canonical_payload())),
            ),
        )

    def _insert_snapshot(self, snapshot: ObserverHabitRegistrySnapshotDTO) -> None:
        self._conn.execute(
            """
            INSERT INTO observer_habit_registry_snapshots
                (registry_sequence, snapshot_id, previous_snapshot_id, canonical_payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                snapshot.registry_sequence,
                snapshot.habit_registry_snapshot_id,
                snapshot.previous_registry_snapshot_id,
                _encode_payload(dict(snapshot.canonical_payload())),
            ),
        )
        for entry in snapshot.entries:
            self._conn.execute(
                """
                INSERT INTO observer_habit_registry_snapshot_entries
                    (snapshot_id, habit_specification_id, registry_entry_id, status,
                     activation_generation, canonical_payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.habit_registry_snapshot_id,
                    entry.habit_specification_id,
                    entry.habit_registry_entry_id,
                    entry.status,
                    entry.activation_generation,
                    _encode_payload(dict(entry.canonical_payload())),
                ),
            )

    def _snapshot_entry_row_count(self, snapshot_id: str) -> int:
        return int(
            self._conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM observer_habit_registry_snapshot_entries
                WHERE snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchone()["n"]
        )

    def _load_snapshots_in_transaction(
        self,
    ) -> tuple[ObserverHabitRegistrySnapshotDTO, ...]:
        ids = [
            row["snapshot_id"]
            for row in self._conn.execute(
                "SELECT snapshot_id FROM observer_habit_registry_snapshots ORDER BY registry_sequence"
            )
        ]
        return tuple(
            cast(
                ObserverHabitRegistrySnapshotDTO,
                self._load_snapshot_in_transaction(id_),
            )
            for id_ in ids
        )

    def _load_events_in_transaction(self) -> tuple[ObserverHabitRegistryEventDTO, ...]:
        ids = [
            row["event_id"]
            for row in self._conn.execute(
                "SELECT event_id FROM observer_habit_registry_events ORDER BY registry_sequence"
            )
        ]
        return tuple(
            cast(ObserverHabitRegistryEventDTO, self.load_event(id_)) for id_ in ids
        )

    def verify_integrity(self) -> ObserverHabitRegistryStoreVerificationDTO:
        failures: set[str] = set()
        replay_id: str | None = None
        head_id: str | None = None
        head_sequence: int | None = None
        event_count = snapshot_count = entry_count = 0
        try:
            self._check_schema()
            self._conn.execute("BEGIN")
            if self._verification_read_started_hook is not None:
                self._verification_read_started_hook()
            head_rows = self._conn.execute(
                "SELECT current_snapshot_id, current_registry_sequence FROM observer_habit_registry_head"
            ).fetchall()
            if len(head_rows) != 1:
                failures.add("head_row_count")
            else:
                head_id = head_rows[0]["current_snapshot_id"]
                head_sequence = head_rows[0]["current_registry_sequence"]
            event_count = self._count("observer_habit_registry_events")
            snapshot_count = self._count("observer_habit_registry_snapshots")
            entry_count = self._count("observer_habit_registry_snapshot_entries")
            snapshots = self._load_snapshots_in_transaction()
            events = self._load_events_in_transaction()
            empty = empty_observer_habit_registry_snapshot()
            if not any(
                item.habit_registry_snapshot_id == empty.habit_registry_snapshot_id
                for item in snapshots
            ):
                failures.add("empty_snapshot_missing")
            sequences = tuple(item.registry_sequence for item in snapshots)
            if sequences != tuple(range(len(snapshots))):
                failures.add("snapshot_sequence_gap")
            event_sequences = tuple(item.registry_sequence for item in events)
            if event_sequences != tuple(range(1, len(events) + 1)):
                failures.add("event_sequence_gap")
            by_id = {item.habit_registry_snapshot_id: item for item in snapshots}
            for snapshot in snapshots:
                if (
                    snapshot.previous_registry_snapshot_id is not None
                    and snapshot.previous_registry_snapshot_id not in by_id
                ):
                    failures.add("snapshot_parent_missing")
            if head_id is None or head_id not in by_id:
                failures.add("head_snapshot_missing")
                final = empty
            else:
                final = by_id[head_id]
                if final.registry_sequence != head_sequence:
                    failures.add("head_sequence_mismatch")
            replay = replay_observer_habit_registry(
                initial_snapshot=empty,
                events=events,
                final_snapshot=final,
                snapshots=snapshots,
            )
            replay_id = replay.habit_registry_replay_id
            if replay.status != "verified":
                failures.update(replay.failure_codes)
            if len(snapshots) != len(events) + 1:
                failures.add("orphan_snapshot")
            self._conn.execute("COMMIT")
        except Exception as exc:
            self._rollback_quietly()
            failures.add(type(exc).__name__)
        return ObserverHabitRegistryStoreVerificationDTO.create(
            store_id=self._safe_store_id(),
            head_snapshot_id=head_id,
            head_registry_sequence=head_sequence,
            event_count=event_count,
            snapshot_count=snapshot_count,
            entry_row_count=entry_count,
            semantic_replay_id=replay_id,
            status="verified" if not failures else "failed",
            failure_codes=tuple(sorted(failures)),
        )

    def recover(self) -> ObserverHabitRegistryRecoveryDTO:
        verification = self.verify_integrity()
        if verification.status == "verified":
            return ObserverHabitRegistryRecoveryDTO.create(
                store_id=self._safe_store_id(),
                recovered_head_snapshot_id=verification.head_snapshot_id,
                recovered_registry_sequence=verification.head_registry_sequence,
                disposition="not_needed",
                reason_codes=("not_needed",),
            )
        if "head_row_count" not in verification.failure_codes:
            return ObserverHabitRegistryRecoveryDTO.create(
                store_id=self._safe_store_id(),
                recovered_head_snapshot_id=None,
                recovered_registry_sequence=None,
                disposition="unsafe_to_recover",
                reason_codes=verification.failure_codes,
            )
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            head_rows = self._conn.execute(
                "SELECT current_snapshot_id, current_registry_sequence FROM observer_habit_registry_head"
            ).fetchall()
            if head_rows:
                self._conn.execute("COMMIT")
                return ObserverHabitRegistryRecoveryDTO.create(
                    store_id=self._safe_store_id(),
                    recovered_head_snapshot_id=head_rows[0]["current_snapshot_id"],
                    recovered_registry_sequence=head_rows[0][
                        "current_registry_sequence"
                    ],
                    disposition="not_needed",
                    reason_codes=("recovery_raced",),
                )
            snapshots = self._load_snapshots_in_transaction()
            events = self._load_events_in_transaction()
            candidate = max(snapshots, key=lambda item: item.registry_sequence)
            replay = replay_observer_habit_registry(
                initial_snapshot=empty_observer_habit_registry_snapshot(),
                events=events,
                final_snapshot=candidate,
                snapshots=snapshots,
            )
            if replay.status != "verified":
                self._conn.execute("ROLLBACK")
                return ObserverHabitRegistryRecoveryDTO.create(
                    store_id=self._safe_store_id(),
                    recovered_head_snapshot_id=None,
                    recovered_registry_sequence=None,
                    disposition="unsafe_to_recover",
                    reason_codes=replay.failure_codes,
                )
            cursor = self._conn.execute(
                """
                INSERT INTO observer_habit_registry_head
                    (singleton_key, current_snapshot_id, current_registry_sequence, generation)
                SELECT 1, ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM observer_habit_registry_head
                )
                """,
                (
                    candidate.habit_registry_snapshot_id,
                    candidate.registry_sequence,
                    candidate.registry_sequence,
                ),
            )
            if cursor.rowcount != 1:
                self._conn.execute("ROLLBACK")
                return ObserverHabitRegistryRecoveryDTO.create(
                    store_id=self._safe_store_id(),
                    recovered_head_snapshot_id=None,
                    recovered_registry_sequence=None,
                    disposition="not_needed",
                    reason_codes=("recovery_raced",),
                )
            self._conn.execute("COMMIT")
        except Exception as exc:
            self._rollback_quietly()
            return ObserverHabitRegistryRecoveryDTO.create(
                store_id=self._safe_store_id(),
                recovered_head_snapshot_id=None,
                recovered_registry_sequence=None,
                disposition="unsafe_to_recover",
                reason_codes=(type(exc).__name__,),
            )
        return ObserverHabitRegistryRecoveryDTO.create(
            store_id=self._safe_store_id(),
            recovered_head_snapshot_id=candidate.habit_registry_snapshot_id,
            recovered_registry_sequence=candidate.registry_sequence,
            disposition="recovered",
            reason_codes=("missing_head_recovered",),
        )

    def _count(self, table: str) -> int:
        return int(
            self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        )

    def _safe_store_id(self) -> str:
        try:
            return self.store_id()
        except Exception:
            return "unknown"

    def _commit_result(
        self,
        source_id: str,
        result: ObserverHabitRegistrySnapshotDTO | None,
        event: ObserverHabitRegistryEventDTO | None,
        disposition: str,
        reasons: Sequence[str],
    ) -> ObserverHabitRegistryStoreCommitDTO:
        return ObserverHabitRegistryStoreCommitDTO.create(
            store_id=self._safe_store_id(),
            source_snapshot_id=source_id,
            result_snapshot_id=None
            if result is None
            else result.habit_registry_snapshot_id,
            event_id=None if event is None else event.habit_registry_event_id,
            registry_sequence=None if event is None else event.registry_sequence,
            disposition=disposition,
            reason_codes=tuple(reasons),
        )

    def _rollback_quietly(self) -> None:
        try:
            self._conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass


class SqliteObserverHabitRegistry:
    """Facade exposing registry operations over a durable SQLite Store."""

    def __init__(self, store: SqliteObserverHabitRegistryStore) -> None:
        self.store = store
        self.store.initialize()

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        busy_timeout_ms: int = 250,
        journal_mode: Literal["wal", "delete"] = "wal",
    ) -> "SqliteObserverHabitRegistry":
        return cls(
            SqliteObserverHabitRegistryStore(
                path,
                busy_timeout_ms=busy_timeout_ms,
                journal_mode=journal_mode,
            )
        )

    def close(self) -> None:
        self.store.close()

    def current_snapshot(self) -> ObserverHabitRegistrySnapshotDTO:
        return self.store.load_current_snapshot()

    def events(self) -> tuple[ObserverHabitRegistryEventDTO, ...]:
        return self.store.load_events()

    def snapshots(self) -> tuple[ObserverHabitRegistrySnapshotDTO, ...]:
        return self.store.load_snapshots()

    def get_entry(
        self, habit_specification_id: str
    ) -> ObserverHabitRegistryEntryDTO | None:
        return self.store.load_entry(habit_specification_id)

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
        event = ObserverHabitRegistryEventDTO.create(
            registry_sequence=source.registry_sequence + 1,
            event_type="admitted",
            habit_specification_id=habit_specification.habit_specification_id,
            habit_registry_entry_id=ObserverHabitRegistryEntryDTO.create(
                habit_specification_id=habit_specification.habit_specification_id,
                habit_admission_decision_id=admission_decision.habit_admission_decision_id,
                status="admitted_inactive",
                activation_generation=0,
                active_since_registry_sequence=None,
                suspended_since_registry_sequence=None,
                retired_since_registry_sequence=None,
                status_reason_codes=("admitted",),
            ).habit_registry_entry_id,
            admission_decision_id=admission_decision.habit_admission_decision_id,
            previous_registry_snapshot_id=None,
            source_registry_snapshot_id=source.habit_registry_snapshot_id,
            reason_codes=("admitted",),
        )
        return self._commit_event(source, event)

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
        activated = _entry_with_status(
            entry,
            status="active",
            sequence=source.registry_sequence + 1,
            reason_codes=reason_codes,
            activation_generation=entry.activation_generation + 1,
        )
        event = self._event_for_entries(
            source=source,
            event_type="activated",
            habit_specification_id=habit_specification_id,
            admission_decision_id=entry.habit_admission_decision_id,
            entries=_replace_entry(source.entries, activated),
            reason_codes=reason_codes,
        )
        return self._commit_event(source, event)

    def deactivate(
        self,
        *,
        habit_specification_id: str,
        reason_codes: tuple[str, ...] = ("deactivated",),
    ) -> ObserverHabitRegistryEventDTO:
        return self._status_transition(
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
        return self._status_transition(
            habit_specification_id, "active", "suspended", "suspended", reason_codes
        )

    def resume(
        self,
        *,
        habit_specification_id: str,
        reason_codes: tuple[str, ...] = ("resumed",),
    ) -> ObserverHabitRegistryEventDTO:
        return self._status_transition(
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
        return self._status_transition(
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
        snapshots = {item.habit_registry_snapshot_id: item for item in self.snapshots()}
        target = snapshots.get(request.target_registry_snapshot_id)
        if target is None:
            return _rollback_result(
                request, source, None, None, "target_not_found", ("target_not_found",)
            )
        if not _is_ancestor(target.habit_registry_snapshot_id, source, snapshots):
            return _rollback_result(
                request,
                source,
                target,
                None,
                "target_not_ancestor",
                ("target_not_ancestor",),
            )
        event = self._event_for_entries(
            source=source,
            event_type="rolled_back",
            habit_specification_id="registry",
            admission_decision_id=None,
            entries=target.entries,
            reason_codes=request.reason_codes or ("rolled_back",),
            previous_registry_snapshot_id=target.habit_registry_snapshot_id,
        )
        committed = self._commit_event(source, event)
        return _rollback_result(
            request,
            source,
            target,
            self.current_snapshot(),
            "rolled_back",
            committed.reason_codes,
            committed.habit_registry_event_id,
        )

    def verify_integrity(self) -> ObserverHabitRegistryStoreVerificationDTO:
        return self.store.verify_integrity()

    def recover(self) -> ObserverHabitRegistryRecoveryDTO:
        return self.store.recover()

    def replay(self) -> ObserverHabitRegistryReplayDTO:
        return replay_observer_habit_registry(
            initial_snapshot=empty_observer_habit_registry_snapshot(),
            events=self.events(),
            final_snapshot=self.current_snapshot(),
            snapshots=self.snapshots(),
        )

    def _status_transition(
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
        updated = _entry_with_status(
            entry,
            status=to_status,
            sequence=source.registry_sequence + 1,
            reason_codes=reason_codes,
        )
        event = self._event_for_entries(
            source=source,
            event_type=event_type,
            habit_specification_id=habit_id,
            admission_decision_id=entry.habit_admission_decision_id,
            entries=_replace_entry(source.entries, updated),
            reason_codes=reason_codes,
        )
        return self._commit_event(source, event)

    def _event_for_entries(
        self,
        *,
        source: ObserverHabitRegistrySnapshotDTO,
        event_type: str,
        habit_specification_id: str,
        admission_decision_id: str | None,
        entries: tuple[ObserverHabitRegistryEntryDTO, ...],
        reason_codes: tuple[str, ...],
        previous_registry_snapshot_id: str | None = None,
    ) -> ObserverHabitRegistryEventDTO:
        provisional = ObserverHabitRegistrySnapshotDTO.create(
            registry_sequence=source.registry_sequence + 1,
            entries=entries,
            previous_registry_snapshot_id=source.habit_registry_snapshot_id,
        )
        return ObserverHabitRegistryEventDTO.create(
            registry_sequence=provisional.registry_sequence,
            event_type=event_type,
            habit_specification_id=habit_specification_id,
            habit_registry_entry_id=None
            if habit_specification_id == "registry"
            else _require_entry(
                provisional, habit_specification_id
            ).habit_registry_entry_id,
            admission_decision_id=admission_decision_id,
            previous_registry_snapshot_id=previous_registry_snapshot_id,
            source_registry_snapshot_id=source.habit_registry_snapshot_id,
            reason_codes=reason_codes,
        )

    def _commit_event(
        self,
        source: ObserverHabitRegistrySnapshotDTO,
        event: ObserverHabitRegistryEventDTO,
    ) -> ObserverHabitRegistryEventDTO:
        result, failures = apply_observer_habit_registry_event(
            source=source,
            event=event,
            known_snapshots={
                item.habit_registry_snapshot_id: item for item in self.snapshots()
            },
        )
        if failures:
            raise ObserverHabitRegistryError("semantic registry transition failed")
        commit = self.store.append_transition(
            expected_source_snapshot_id=source.habit_registry_snapshot_id,
            event=event,
            result_snapshot=result,
        )
        if commit.disposition != "committed":
            raise ObserverHabitRegistryError(commit.disposition)
        return event
