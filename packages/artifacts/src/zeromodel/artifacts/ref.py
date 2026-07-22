from __future__ import annotations

import re
from dataclasses import dataclass

from zeromodel.core.artifact import VPMValidationError

ARTIFACT_REF_VERSION = "zeromodel-artifact-ref/v1"

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def is_sha256_digest(value: str) -> bool:
    return isinstance(value, str) and bool(_SHA256_PATTERN.match(value))


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A stable, resolvable reference to an artifact stored through this package.

    `artifact_id` is always a `sha256:` content digest of the artifact's own
    canonical bytes - never a filename, a display name, or a mutable label.
    `artifact_kind` is a namespaced string (e.g. "zeromodel.core.vpm",
    "zeromodel.navigation.tile") owned by whichever package defines that
    kind of artifact; this package does not enumerate kinds itself.
    """

    artifact_kind: str
    artifact_id: str
    spec_version: str = ARTIFACT_REF_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_kind, str) or not self.artifact_kind:
            raise VPMValidationError(
                "ArtifactRef.artifact_kind must be a non-empty string"
            )
        if not is_sha256_digest(self.artifact_id):
            raise VPMValidationError(
                "ArtifactRef.artifact_id must be a 'sha256:<64 hex>' content digest"
            )
        if self.spec_version != ARTIFACT_REF_VERSION:
            raise VPMValidationError(
                f"unsupported ArtifactRef spec_version: {self.spec_version!r}"
            )
