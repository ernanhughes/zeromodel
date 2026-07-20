from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dto import BenchmarkIdentityDTO
from .identity_service import IdentityService


@dataclass(frozen=True, slots=True)
class VideoActionSetEngine:
    identity_service: IdentityService

    def load_identity(self, repo_root: Path) -> BenchmarkIdentityDTO:
        return self.identity_service.load_identity(repo_root)

    def get_identity(self, seed_digest: str) -> BenchmarkIdentityDTO | None:
        return self.identity_service.get_identity(seed_digest)

    def save_identity(self, identity: BenchmarkIdentityDTO) -> BenchmarkIdentityDTO:
        return self.identity_service.save_identity(identity)


__all__ = ["VideoActionSetEngine"]
