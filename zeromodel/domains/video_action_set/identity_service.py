from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dto import BenchmarkIdentityDTO
from .store import VideoActionSetStore


IDENTITY_DOCUMENT_PATH = (
    "docs/research/video-action-set-reachability-benchmark-identity-v1.md"
)


@dataclass(frozen=True, slots=True)
class IdentityService:
    store: VideoActionSetStore

    def load_identity(self, repo_root: Path) -> BenchmarkIdentityDTO:
        values = parse_identity_document(repo_root / IDENTITY_DOCUMENT_PATH)
        identity = BenchmarkIdentityDTO(
            contract_commit=values["contract commit SHA"],
            seed_material=values["seed material"],
            seed_digest=values["seed digest"],
            policy_artifact_id=values["policy artifact ID"],
            parent_audit_sha=values["parent audit SHA"],
            parent_v3_sha=values["parent v3 SHA"],
        )
        return self.store.save_identity(identity)

    def get_identity(self, seed_digest: str) -> BenchmarkIdentityDTO | None:
        return self.store.get_identity(seed_digest)

    def save_identity(self, identity: BenchmarkIdentityDTO) -> BenchmarkIdentityDTO:
        return self.store.save_identity(identity)


def parse_identity_document(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        values[left.strip("- ").strip()] = right.strip().strip("`")
    return values


__all__ = ["IDENTITY_DOCUMENT_PATH", "IdentityService", "parse_identity_document"]
