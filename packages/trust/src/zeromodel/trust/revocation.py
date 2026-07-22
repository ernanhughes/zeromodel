"""Bounded, in-memory revocation resolution.

This is explicitly the first slice: a process-local resolver, not a network
PKI or hosted revocation service. `RevocationResolver` is a protocol so a
later stage can substitute a real backend without changing
`verify_artifact_for_scope`'s contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Protocol, runtime_checkable

from zeromodel.trust.dto import RevocationRecordDTO


class RevocationStatus(str, Enum):
    CLEAR = "clear"
    REVOKED = "revoked"
    INDETERMINATE = "indeterminate"


@runtime_checkable
class RevocationResolver(Protocol):
    def status(self, target_kind: str, target_id: str) -> RevocationStatus: ...


class InMemoryRevocationResolver:
    """Bounded in-memory resolver over an explicit, auditable record set."""

    def __init__(self, records: Iterable[RevocationRecordDTO] = ()) -> None:
        self._records: dict[tuple[str, str], RevocationRecordDTO] = {
            (record.target_kind, record.target_id): record for record in records
        }

    @property
    def records(self) -> tuple[RevocationRecordDTO, ...]:
        return tuple(self._records.values())

    def status(self, target_kind: str, target_id: str) -> RevocationStatus:
        if (target_kind, target_id) in self._records:
            return RevocationStatus.REVOKED
        return RevocationStatus.CLEAR


class IndeterminateRevocationResolver:
    """Test/edge-case resolver: reports every check as unknown.

    Models a revocation-check backend that could not be reached or could not
    return a confident answer - the "indeterminate" case the trust decision
    must fail closed on, distinct from a definite "revoked".
    """

    def status(self, target_kind: str, target_id: str) -> RevocationStatus:
        return RevocationStatus.INDETERMINATE
