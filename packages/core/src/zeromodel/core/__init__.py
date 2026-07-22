"""ZeroModel core public API."""

from __future__ import annotations

from .artifact import (
    LAYOUT_VERSION,
    LayoutRecipe,
    SPEC_VERSION,
    ScoreTable,
    VPMArtifact,
    VPMCell,
    VPMRegion,
    VPMValidationError,
    build_vpm,
)
from .bundle import (
    BUNDLE_VERSION,
    MANIFEST_NAME,
    bundle_manifest,
    from_bundle,
    to_bundle,
)
from .content_identity import (
    PROTOTYPE_UNIVERSE_IDENTITY_VERSION,
    PrototypeUniverseIdentity,
    UnresolvedArtifactIdentity,
    array_content_digest,
    canonical_float64_bytes,
    canonical_json_bytes,
    prototype_universe_identity,
    sha256_digest,
)
from .lua import (
    POLICY_LUA_FORMAT,
    compiled_plan_id,
    lua_policy_source,
    write_lua_policy,
)
from .matrix_blob import (
    MATRIX_BLOB_VERSION,
    MatrixBlob,
)
from .metrics import (
    CANONICAL_METRICS,
    metric_ids_for_rows,
    pack_metrics,
    score_table_from_metric_rows,
)
from .policy_lookup import (
    POLICY_PLAN_VERSION,
    PolicyLookupDecision,
    SignReader,
    VPMPolicyLookup,
)
from .policy_transitions import (
    POLICY_TRANSITION_EVIDENCE_VERSION,
    POLICY_TRANSITION_SPEC_VERSION,
    PolicyTransitionEvidence,
    PolicyTransitionSpec,
    ROW_UNION_TRANSITION_SCOPE,
)
from .render import (
    PNG_SIGNATURE,
    as_field,
    png_bytes,
    svg_text,
    to_uint8,
    write_png,
    write_svg,
)
from .views import (
    ViewProfile,
    ViewSet,
    build_view,
    build_views,
)

__all__ = [
    "BUNDLE_VERSION",
    "CANONICAL_METRICS",
    "LAYOUT_VERSION",
    "LayoutRecipe",
    "MANIFEST_NAME",
    "MATRIX_BLOB_VERSION",
    "MatrixBlob",
    "PNG_SIGNATURE",
    "POLICY_LUA_FORMAT",
    "POLICY_PLAN_VERSION",
    "POLICY_TRANSITION_EVIDENCE_VERSION",
    "POLICY_TRANSITION_SPEC_VERSION",
    "PROTOTYPE_UNIVERSE_IDENTITY_VERSION",
    "PolicyLookupDecision",
    "PolicyTransitionEvidence",
    "PolicyTransitionSpec",
    "PrototypeUniverseIdentity",
    "ROW_UNION_TRANSITION_SCOPE",
    "SPEC_VERSION",
    "ScoreTable",
    "SignReader",
    "UnresolvedArtifactIdentity",
    "VPMArtifact",
    "VPMCell",
    "VPMPolicyLookup",
    "VPMRegion",
    "VPMValidationError",
    "ViewProfile",
    "ViewSet",
    "array_content_digest",
    "as_field",
    "build_view",
    "build_views",
    "build_vpm",
    "bundle_manifest",
    "canonical_float64_bytes",
    "canonical_json_bytes",
    "compiled_plan_id",
    "from_bundle",
    "lua_policy_source",
    "metric_ids_for_rows",
    "pack_metrics",
    "png_bytes",
    "prototype_universe_identity",
    "score_table_from_metric_rows",
    "sha256_digest",
    "svg_text",
    "to_bundle",
    "to_uint8",
    "write_lua_policy",
    "write_png",
    "write_svg",
]
