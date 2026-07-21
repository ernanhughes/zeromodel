from __future__ import annotations

from typing import Protocol

from .final_access_dto import (
    FinalAccessEventDTO,
    FinalAccessRecordDTO,
    FinalExecutionAuthorizationDTO,
)


class ExternalFinalAccessRegistry(Protocol):
    """Optional future hook for cross-machine final-access coordination.

    The repository implements local exactly-once semantics through the durable
    store. A deployment that needs global coordination can provide this protocol
    at the boundary without changing scientific execution code.
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
