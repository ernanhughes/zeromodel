"""Lossless bundle serialization for VPM artifacts."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from zeromodel.core.artifact import VPMArtifact

BUNDLE_VERSION = "vpm-bundle/0"
MANIFEST_NAME = "manifest.json"


def bundle_manifest(artifact: VPMArtifact) -> dict[str, Any]:
    return {
        "bundle_version": BUNDLE_VERSION,
        "artifact": artifact.to_dict(),
    }


def to_bundle(artifact: VPMArtifact, path: str | Path) -> Path:
    """Write a lossless `.vpm` zip bundle for an artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle_manifest(artifact), sort_keys=True, indent=2, allow_nan=False)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, payload)
    return target


def from_bundle(path: str | Path) -> VPMArtifact:
    """Read and validate a `.vpm` zip bundle."""
    source = Path(path)
    with zipfile.ZipFile(source, "r") as zf:
        payload = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
    if payload.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("Unsupported bundle_version: %r" % payload.get("bundle_version"))
    return VPMArtifact.from_dict(payload["artifact"])
