from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from video_final_test_support import approved_protocol, authorization
from zeromodel.core.artifact import VPMValidationError
from zeromodel.persistence.sqlalchemy.db.runtime import build_finalization_sqlite_runtime
from zeromodel.persistence.sqlalchemy.db.session import sqlite_database_url
from zeromodel.video.domains.video_action_set.final_access_service import FinalAccessService


pytestmark = pytest.mark.integration


def _database_url(path: Path) -> str:
    return sqlite_database_url(path)


def _services(
    tmp_path: Path,
    initial_state: str,
) -> tuple[FinalAccessService, FinalAccessService, object]:
    path = tmp_path / "concurrent-finalization.sqlite3"
    first_runtime = build_finalization_sqlite_runtime(
        _database_url(path),
        initialize_authority=True,
    )
    second_runtime = build_finalization_sqlite_runtime(_database_url(path))
    first = first_runtime.video_action_set.engine.final_access_service
    second = second_runtime.video_action_set.engine.final_access_service
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    record = first.create_authorization(auth, protocol, process_identity="setup")
    if initial_state in {"reserved", "running"}:
        record = first.reserve(record.access_id, process_identity="setup")
    if initial_state == "running":
        record = first.mark_running(record.access_id, process_identity="setup")
    return first, second, record


def _compete(
    first: Callable[[], object],
    second: Callable[[], object],
) -> tuple[Exception | None, Exception | None]:
    barrier = Barrier(2)

    def run(operation: Callable[[], object]) -> Exception | None:
        barrier.wait(timeout=10)
        try:
            operation()
        except Exception as exc:  # the exact losing path is asserted by state
            return exc
        return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(run, first), pool.submit(run, second))
        outcomes = tuple(future.result(timeout=20) for future in futures)
        return outcomes[0], outcomes[1]


@pytest.mark.parametrize(
    ("scenario", "initial_state", "terminal"),
    [
        ("reservation_vs_reservation", "authorized", False),
        ("running_vs_running", "reserved", False),
        ("failure_vs_interruption", "running", True),
        ("completion_vs_interruption", "running", True),
        ("completion_vs_completion", "running", True),
    ],
)
def test_file_sqlite_separate_connections_have_exactly_one_cas_winner(
    tmp_path: Path,
    scenario: str,
    initial_state: str,
    terminal: bool,
) -> None:
    first, second, initial = _services(tmp_path, initial_state)
    before_events = first.list_events(initial.access_id)

    if scenario == "reservation_vs_reservation":
        one = first._next_event(
            initial.access_id,
            "reserved",
            process_identity="one",
            utc="2026-07-21T01:00:00Z",
            event_payload={"kind": "reserved"},
        )
        two = second._next_event(
            initial.access_id,
            "reserved",
            process_identity="two",
            utc="2026-07-21T01:00:01Z",
            event_payload={"kind": "reserved"},
        )
        operations = (
            lambda: first.store.reserve_final_access(*one),
            lambda: second.store.reserve_final_access(*two),
        )
    elif scenario == "running_vs_running":
        one = first._next_event(
            initial.access_id,
            "running",
            process_identity="one",
            utc="2026-07-21T01:00:00Z",
            event_payload={"kind": "running"},
        )
        two = second._next_event(
            initial.access_id,
            "running",
            process_identity="two",
            utc="2026-07-21T01:00:01Z",
            event_payload={"kind": "running"},
        )
        operations = (
            lambda: first.store.mark_final_access_running(*one),
            lambda: second.store.mark_final_access_running(*two),
        )
    elif scenario == "failure_vs_interruption":
        one = first._failure_transition(
            initial.access_id,
            "failed",
            failure_kind="one",
            error_code="one",
            error_message="one",
            process_identity="one",
            utc="2026-07-21T01:00:00Z",
        )
        two = second._failure_transition(
            initial.access_id,
            "interrupted",
            failure_kind="two",
            error_code="two",
            error_message="two",
            process_identity="two",
            utc="2026-07-21T01:00:01Z",
        )
        operations = (
            lambda: first.store.fail_final_access(one[1], one[2], one[0]),
            lambda: second.store.interrupt_final_access(two[1], two[2], two[0]),
        )
    elif scenario == "completion_vs_interruption":
        one = first._next_event(
            initial.access_id,
            "completed",
            process_identity="one",
            utc="2026-07-21T01:00:00Z",
            event_payload={"kind": "completed"},
        )
        two = second._failure_transition(
            initial.access_id,
            "interrupted",
            failure_kind="two",
            error_code="two",
            error_message="two",
            process_identity="two",
            utc="2026-07-21T01:00:01Z",
        )
        operations = (
            lambda: first.store.complete_final_access(*one),
            lambda: second.store.interrupt_final_access(two[1], two[2], two[0]),
        )
    else:
        one = first._next_event(
            initial.access_id,
            "completed",
            process_identity="one",
            utc="2026-07-21T01:00:00Z",
            event_payload={"kind": "completed"},
        )
        two = second._next_event(
            initial.access_id,
            "completed",
            process_identity="two",
            utc="2026-07-21T01:00:01Z",
            event_payload={"kind": "completed"},
        )
        operations = (
            lambda: first.store.complete_final_access(*one),
            lambda: second.store.complete_final_access(*two),
        )

    outcomes = _compete(*operations)
    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(outcome is not None for outcome in outcomes) == 1

    final_record = first.load_record(initial.access_id)
    final_events = first.list_events(initial.access_id)
    assert final_record is not None
    assert len(final_events) == len(before_events) + 1
    assert final_record.current_event_ordinal == len(final_events) - 1
    assert final_record.last_event_digest == final_events[-1].event_digest
    assert [event.ordinal for event in final_events] == list(range(len(final_events)))
    if terminal:
        assert final_record.state in {"failed", "interrupted", "completed"}
        with pytest.raises(VPMValidationError):
            first.fail(
                final_record.access_id,
                failure_kind="must-not-append",
                error_code="terminal",
                error_message="terminal",
            )
        assert first.list_events(initial.access_id) == final_events
