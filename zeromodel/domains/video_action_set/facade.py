from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dto import BenchmarkIdentityDTO
from .engine import VideoActionSetEngine


@dataclass(frozen=True, slots=True)
class VideoActionSetFacade:
    engine: VideoActionSetEngine

    def load_identity(self, repo_root: Path) -> BenchmarkIdentityDTO:
        return self.engine.load_identity(repo_root)

    def get_identity(self, seed_digest: str) -> BenchmarkIdentityDTO | None:
        return self.engine.get_identity(seed_digest)

    def save_identity(self, identity: BenchmarkIdentityDTO) -> BenchmarkIdentityDTO:
        return self.engine.save_identity(identity)


__all__ = ["VideoActionSetFacade"]
