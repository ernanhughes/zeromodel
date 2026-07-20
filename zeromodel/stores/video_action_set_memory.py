from __future__ import annotations

from ..domains.video_action_set.dto import BenchmarkIdentityDTO
from ..domains.video_action_set.store import (
    VideoActionSetStore,
    raise_identity_conflict,
)


class InMemoryVideoActionSetStore(VideoActionSetStore):
    def __init__(self) -> None:
        self._identities: dict[str, BenchmarkIdentityDTO] = {}

    def save_identity(self, identity: BenchmarkIdentityDTO) -> BenchmarkIdentityDTO:
        existing = self._identities.get(identity.seed_digest)
        if existing is not None:
            if existing != identity:
                raise_identity_conflict()
            return existing
        self._identities[identity.seed_digest] = identity
        return identity

    def get_identity(self, seed_digest: str) -> BenchmarkIdentityDTO | None:
        return self._identities.get(seed_digest)


__all__ = ["InMemoryVideoActionSetStore"]
