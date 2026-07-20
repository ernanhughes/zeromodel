from __future__ import annotations

from typing import NoReturn, Protocol

from ...artifact import VPMValidationError
from .dto import BenchmarkIdentityDTO


BENCHMARK_IDENTITY_CONFLICT_MESSAGE = "benchmark identity conflict for seed digest"


class VideoActionSetStore(Protocol):
    def save_identity(
        self,
        identity: BenchmarkIdentityDTO,
    ) -> BenchmarkIdentityDTO: ...

    def get_identity(
        self,
        seed_digest: str,
    ) -> BenchmarkIdentityDTO | None: ...


def raise_identity_conflict() -> NoReturn:
    raise VPMValidationError(BENCHMARK_IDENTITY_CONFLICT_MESSAGE)


__all__ = [
    "BENCHMARK_IDENTITY_CONFLICT_MESSAGE",
    "VideoActionSetStore",
    "raise_identity_conflict",
]
