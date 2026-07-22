from __future__ import annotations

from typing import Protocol

from zeromodel.video.domains.video_action_set.final_access_dto import (
    FinalAccessEventDTO,
    FinalAccessRecordDTO,
    FinalExecutionAuthorizationDTO,
)


class ExternalFinalAccessRegistry(Protocol):
    """Unintegrated extension point for cross-machine reservation authority.

    The current orchestration never calls this protocol. Adopting it requires an
    explicit reservation-flow change and review; implementing the interface alone
    does not extend local SQLite exactly-once semantics across machines.
    """

    def assert_authorization_available(
        self,
        authorization: FinalExecutionAuthorizationDTO,
    ) -> None: ...

    def record_reservation(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> None: ...


__all__ = ["ExternalFinalAccessRegistry"]
