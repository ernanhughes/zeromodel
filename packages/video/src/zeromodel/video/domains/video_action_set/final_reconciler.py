from __future__ import annotations

from collections.abc import Callable

from zeromodel.video.domains.video_action_set.final_access_dto import FinalAccessRecordDTO
from zeromodel.video.domains.video_action_set.final_access_service import FinalAccessService
from zeromodel.video.domains.video_action_set.store import VideoActionSetStore


ProcessAlive = Callable[[str], bool]


def interrupt_abandoned_running_access(
    store: VideoActionSetStore,
    access_id: str,
    *,
    process_is_alive: ProcessAlive,
    reconciler_identity: str,
) -> FinalAccessRecordDTO | None:
    record = store.load_final_access_record(access_id)
    if record is None or record.state != "running":
        return None
    if process_is_alive(record.process_identity):
        return None
    return FinalAccessService(store=store).interrupt(
        access_id,
        failure_kind="abandoned_running_process",
        error_code="process_not_alive",
        error_message="running final access process was not alive during reconciliation",
        process_identity=reconciler_identity,
    )


__all__ = [
    "ProcessAlive",
    "interrupt_abandoned_running_access",
]
