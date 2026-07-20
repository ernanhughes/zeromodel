from __future__ import annotations

from dataclasses import dataclass
import hashlib

from ...artifact import VPMValidationError
from .contracts import (
    BENCHMARK_VERSION,
    GENERATOR_VERSION,
    REACHABILITY_TILE_DIGEST,
    REACHABILITY_TILE_VERSION,
)


@dataclass(frozen=True, slots=True)
class BenchmarkIdentityDTO:
    contract_commit: str
    seed_material: str
    seed_digest: str
    policy_artifact_id: str
    parent_audit_sha: str
    parent_v3_sha: str

    def __post_init__(self) -> None:
        expected = (
            "sha256:" + hashlib.sha256(self.seed_material.encode("utf-8")).hexdigest()
        )
        if self.seed_digest != expected:
            raise VPMValidationError(
                "benchmark seed digest is inconsistent with frozen seed material"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "benchmark_version": BENCHMARK_VERSION,
            "generator_version": GENERATOR_VERSION,
            "contract_commit": self.contract_commit,
            "seed_material": self.seed_material,
            "seed_digest": self.seed_digest,
            "policy_artifact_id": self.policy_artifact_id,
            "parent_audit_sha": self.parent_audit_sha,
            "parent_v3_sha": self.parent_v3_sha,
            "reachability_tile_version": REACHABILITY_TILE_VERSION,
            "reachability_tile_digest": REACHABILITY_TILE_DIGEST,
        }


__all__ = ["BenchmarkIdentityDTO"]
