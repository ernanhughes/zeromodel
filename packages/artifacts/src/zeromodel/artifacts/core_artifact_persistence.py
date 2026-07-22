"""Persist and reload the Core `ScoreTable`/`LayoutRecipe`/`VPMArtifact`
objects a compiled report references, through the same injected
`ArtifactStore` used for the compiled report record itself.

Without this module, a `CompiledReportArtifactDTO` could only name these
three Core objects by digest string - a claim that some object with that
digest once existed in a discarded local Python variable during
`compile_report()`, not a resolvable artifact. After process restart there
would be no way to load a compiled report, obtain its VPM, and render it.
These functions make each Core object a real, independently-loadable
artifact addressed by `ArtifactRef`, using the same decode-and-verify
pattern as `report_loading.load_compiled_report_artifact`: resolve
canonical bytes, recompute the digest, require it to equal the requested
ref's `artifact_id`, decode, and reconstruct via the type's own
`from_dict()` - never trusting a store's manifest as authoritative.

Note: Core's own identity scheme (`ScoreTable.digest`,
`VPMArtifact.artifact_id`, via `_compute_identity_bytes()` +
`hashlib.sha256`) is independent of the `ArtifactRef.artifact_id` values
`store.put()` computes here (`sha256_digest(canonical_json_bytes(...))`).
The two digests for the same object will not be numerically equal; both
are valid, self-consistent identities for their own layer and this module
does not attempt to reconcile them.
"""

from __future__ import annotations

import json

from zeromodel.artifacts.canonicalization import canonical_json_bytes, sha256_digest
from zeromodel.artifacts.ref import ArtifactRef
from zeromodel.artifacts.report_errors import ReportCompilationError
from zeromodel.artifacts.store import ArtifactResolver, ArtifactStore
from zeromodel.core.artifact import LayoutRecipe, ScoreTable, VPMArtifact

SCORE_TABLE_ARTIFACT_KIND = "zeromodel.core.score-table/v1"
LAYOUT_RECIPE_ARTIFACT_KIND = "zeromodel.core.layout-recipe/v1"
VPM_ARTIFACT_ARTIFACT_KIND = "zeromodel.core.vpm-artifact/v1"


def _resolve_and_verify(
    ref: ArtifactRef, *, resolver: ArtifactResolver, expected_kind: str
) -> dict:
    if ref.artifact_kind != expected_kind:
        raise ReportCompilationError(
            f"expected artifact_kind={expected_kind!r}, got {ref.artifact_kind!r}"
        )
    canonical_bytes = resolver.resolve_canonical_bytes(ref)
    actual_digest = sha256_digest(canonical_bytes)
    if actual_digest != ref.artifact_id:
        raise ReportCompilationError(
            f"resolved canonical bytes for {ref.artifact_id} do not hash to the requested id "
            f"(got {actual_digest})"
        )
    return json.loads(canonical_bytes)


def store_score_table(score_table: ScoreTable, *, store: ArtifactStore) -> ArtifactRef:
    canonical_bytes = canonical_json_bytes(score_table.to_identity_payload())
    return store.put(SCORE_TABLE_ARTIFACT_KIND, canonical_bytes, manifest=None)


def load_score_table(ref: ArtifactRef, *, resolver: ArtifactResolver) -> ScoreTable:
    payload = _resolve_and_verify(
        ref, resolver=resolver, expected_kind=SCORE_TABLE_ARTIFACT_KIND
    )
    return ScoreTable.from_dict(payload)


def store_layout_recipe(
    layout_recipe: LayoutRecipe, *, store: ArtifactStore
) -> ArtifactRef:
    canonical_bytes = canonical_json_bytes(layout_recipe.to_dict())
    return store.put(LAYOUT_RECIPE_ARTIFACT_KIND, canonical_bytes, manifest=None)


def load_layout_recipe(ref: ArtifactRef, *, resolver: ArtifactResolver) -> LayoutRecipe:
    payload = _resolve_and_verify(
        ref, resolver=resolver, expected_kind=LAYOUT_RECIPE_ARTIFACT_KIND
    )
    return LayoutRecipe.from_dict(payload)


def store_vpm_artifact(
    vpm_artifact: VPMArtifact, *, store: ArtifactStore
) -> ArtifactRef:
    canonical_bytes = canonical_json_bytes(vpm_artifact.to_dict())
    return store.put(VPM_ARTIFACT_ARTIFACT_KIND, canonical_bytes, manifest=None)


def load_vpm_artifact(ref: ArtifactRef, *, resolver: ArtifactResolver) -> VPMArtifact:
    payload = _resolve_and_verify(
        ref, resolver=resolver, expected_kind=VPM_ARTIFACT_ARTIFACT_KIND
    )
    return VPMArtifact.from_dict(payload)
