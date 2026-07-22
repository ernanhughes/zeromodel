from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Callable

import pytest

from video_final_test_support import approved_protocol, authorization
from zeromodel.artifact import VPMValidationError
from zeromodel.db.runtime import build_finalization_sqlite_runtime
from zeromodel.db.session import sqlite_database_url
from zeromodel.domains.video_action_set.final_access_service import FinalAccessService
from zeromodel.stores.video_action_set_memory import InMemoryVideoActionSetStore


StoreFactory = Callable[[Path], object]


def _memory_store(_tmp_path: Path) -> InMemoryVideoActionSetStore:
    return InMemoryVideoActionSetStore()


def _sqlite_store(tmp_path: Path) -> object:
    database_path = tmp_path / "finalization.sqlite3"
    runtime = build_finalization_sqlite_runtime(
        sqlite_database_url(database_path),
        initialize_authority=True,
    )
    return runtime.video_action_set.engine.final_access_service.store


def _service_with_state(
    tmp_path: Path,
    store_factory: StoreFactory,
    state: str,
) -> tuple[FinalAccessService, object]:
    store = store_factory(tmp_path)
    service = FinalAccessService(store=store)  # type: ignore[arg-type]
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    record = service.create_authorization(auth, protocol, process_identity="test")
    if state in {"reserved", "running"}:
        record = service.reserve(record.access_id, process_identity="test")
    if state == "running":
        record = service.mark_running(record.access_id, process_identity="test")
    return service, record


def _compete(first: Callable[[], object], second: Callable[[], object]) -> int:
    barrier = Barrier(2)

    def run(operation: Callable[[], object]) -> bool:
        barrier.wait()
        try:
            operation()
        except VPMValidationError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(run, (first, second)))
    return sum(outcomes)


@pytest.mark.parametrize("store_factory", [_memory_store, _sqlite_store])
def test_two_stale_reservations_have_one_winner(
    tmp_path: Path,
    store_factory: StoreFactory,
) -> None:
    service, authorized = _service_with_state(tmp_path, store_factory, "authorized")
    first = service._next_event(
        authorized.access_id,
        "reserved",
        process_identity="one",
        utc="2026-07-21T01:00:00Z",
        event_payload={"kind": "reserved"},
    )
    second = service._next_event(
        authorized.access_id,
        "reserved",
        process_identity="two",
        utc="2026-07-21T01:00:01Z",
        event_payload={"kind": "reserved"},
    )
    assert _compete(
        lambda: service.store.reserve_final_access(*first),
        lambda: service.store.reserve_final_access(*second),
    ) == 1


@pytest.mark.parametrize("store_factory", [_memory_store, _sqlite_store])
def test_two_stale_running_transitions_have_one_winner(
    tmp_path: Path,
    store_factory: StoreFactory,
) -> None:
    service, reserved = _service_with_state(tmp_path, store_factory, "reserved")
    first = service._next_event(
        reserved.access_id,
        "running",
        process_identity="one",
        utc="2026-07-21T01:00:00Z",
        event_payload={"kind": "running"},
    )
    second = service._next_event(
        reserved.access_id,
        "running",
        process_identity="two",
        utc="2026-07-21T01:00:01Z",
        event_payload={"kind": "running"},
    )
    assert _compete(
        lambda: service.store.mark_final_access_running(*first),
        lambda: service.store.mark_final_access_running(*second),
    ) == 1


@pytest.mark.parametrize("store_factory", [_memory_store, _sqlite_store])
def test_running_versus_failure_has_one_winner(
    tmp_path: Path,
    store_factory: StoreFactory,
) -> None:
    service, reserved = _service_with_state(tmp_path, store_factory, "reserved")
    running = service._next_event(
        reserved.access_id,
        "running",
        process_identity="runner",
        utc="2026-07-21T01:00:00Z",
        event_payload={"kind": "running"},
    )
    failure, failed_record, failed_event = service._failure_transition(
        reserved.access_id,
        "failed",
        failure_kind="synthetic",
        error_code="synthetic",
        error_message="synthetic",
        process_identity="failure",
        utc="2026-07-21T01:00:01Z",
    )
    assert _compete(
        lambda: service.store.mark_final_access_running(*running),
        lambda: service.store.fail_final_access(
            failed_record,
            failed_event,
            failure,
        ),
    ) == 1


@pytest.mark.parametrize("store_factory", [_memory_store, _sqlite_store])
def test_completion_versus_interruption_has_one_winner(
    tmp_path: Path,
    store_factory: StoreFactory,
) -> None:
    service, running = _service_with_state(tmp_path, store_factory, "running")
    completed = service._next_event(
        running.access_id,
        "completed",
        process_identity="completion",
        utc="2026-07-21T01:00:00Z",
        event_payload={"kind": "completed"},
    )
    failure, interrupted_record, interrupted_event = service._failure_transition(
        running.access_id,
        "interrupted",
        failure_kind="synthetic",
        error_code="synthetic",
        error_message="synthetic",
        process_identity="interruption",
        utc="2026-07-21T01:00:01Z",
    )
    assert _compete(
        lambda: service.store.complete_final_access(*completed),
        lambda: service.store.interrupt_final_access(
            interrupted_record,
            interrupted_event,
            failure,
        ),
    ) == 1


@pytest.mark.parametrize("store_factory", [_memory_store, _sqlite_store])
def test_two_terminal_transitions_have_one_winner(
    tmp_path: Path,
    store_factory: StoreFactory,
) -> None:
    service, running = _service_with_state(tmp_path, store_factory, "running")
    first = service._failure_transition(
        running.access_id,
        "failed",
        failure_kind="one",
        error_code="one",
        error_message="one",
        process_identity="one",
        utc="2026-07-21T01:00:00Z",
    )
    second = service._failure_transition(
        running.access_id,
        "interrupted",
        failure_kind="two",
        error_code="two",
        error_message="two",
        process_identity="two",
        utc="2026-07-21T01:00:01Z",
    )
    assert _compete(
        lambda: service.store.fail_final_access(first[1], first[2], first[0]),
        lambda: service.store.interrupt_final_access(
            second[1],
            second[2],
            second[0],
        ),
    ) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "event_deleted",
        "event_inserted",
        "event_reordered",
        "ordinal_changed",
        "previous_digest_changed",
        "payload_changed",
        "state_changed",
        "cross_access_substituted",
    ],
)
def test_every_event_chain_mutation_blocks_terminal_transition(
    tmp_path: Path,
    mutation: str,
) -> None:
    service, running = _service_with_state(tmp_path, _memory_store, "running")
    store = service.store
    events = store._final_events[running.access_id]  # type: ignore[attr-defined]
    if mutation == "event_deleted":
        events.pop(1)
    elif mutation == "event_inserted":
        events.insert(1, events[1])
    elif mutation == "event_reordered":
        events[1], events[2] = events[2], events[1]
    elif mutation == "ordinal_changed":
        object.__setattr__(events[1], "ordinal", 99)
    elif mutation == "previous_digest_changed":
        object.__setattr__(events[2], "previous_event_digest", "sha256:" + "9" * 64)
    elif mutation == "payload_changed":
        from zeromodel.domains.video_action_set.final_access_dto import FinalJsonDTO

        object.__setattr__(
            events[1],
            "event_payload",
            FinalJsonDTO.from_value({"kind": "reserved", "tampered": True}),
        )
    elif mutation == "state_changed":
        object.__setattr__(events[2], "new_state", "failed")
    else:
        other_root = tmp_path / "other"
        other_root.mkdir()
        other_service, other = _service_with_state(
            other_root,
            _memory_store,
            "running",
        )
        events[1] = other_service.store.list_final_access_events(other.access_id)[1]

    with pytest.raises(VPMValidationError, match="event|digest|state|transition"):
        service.fail(
            running.access_id,
            failure_kind="must-not-commit",
            error_code="tampered_chain",
            error_message="tampered chain",
        )
