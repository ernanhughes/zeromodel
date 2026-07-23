"""Content-derived compatibility schema identity.

`ReportAdapterContractDTO.compatibility_id` is an opaque, caller-chosen
string - two adapters can declare the same `compatibility_id` while
actually producing incompatible dimension schemas (different dimension
sets, different ordering, different score semantics, different value
ranges, different missing-value policy). Search or comparison code that
only checks `compatibility_id` string equality would then risk treating
structurally incompatible artifacts as compatible.

`compute_compatibility_schema_id` binds the human-readable name to an
actual content digest of the coordinate schema it names: dimension ids in
declared order, each dimension's score semantics and value/target ranges,
and the report's missing-value policy. Two compiled reports are only
truly schema-compatible if both `compatibility_id` (the human label) and
`compatibility_schema_id` (the content digest) agree.

Deliberately out of scope for this digest: the layout recipe's
normalization contract (`per_metric_minmax`, clipping, row/column
ordering, ...). That is a per-render concern owned by `LayoutRecipe`, not
a per-report-schema concern - conflating the two would make the same
report schema-incompatible with itself under two equally valid layouts.

`compute_compatibility_schema_id` alone is still not the complete
compatibility story: it only covers *dimensions*. Two adapters could
declare an identical dimension schema while describing entirely different
kinds of subjects (a report over sentences vs. a report over claims), or
different report kinds, or different duplicate-value policy - none of
which show up in a dimension-only digest. `compute_report_semantics_id`
closes that separate, additive layer (Stage C's "Design B": kept
independent of `compute_compatibility_schema_id` rather than folded into
it, so the already-shipped `compatibility_schema_id` identity/kind never
needs to be reinterpreted or versioned). Full compatibility between two
compiled reports requires all three to agree: `compatibility_id` (the
human label), `compatibility_schema_id` (the dimension schema), and
`report_semantics_id` (the report/subject schema).
"""

from __future__ import annotations

from typing import Tuple

from zeromodel.artifacts.canonicalization import canonical_json_bytes, sha256_digest
from zeromodel.artifacts.report_dto import AdaptedDimensionDTO


def compute_compatibility_schema_id(
    *, dimensions: Tuple[AdaptedDimensionDTO, ...], missing_value_semantics: str
) -> str:
    """Content digest of a report's coordinate schema.

    Dimension order is significant and preserved as declared - it is part
    of the schema two reports must share to be genuinely comparable.
    """
    payload = {
        "missing_value_semantics": missing_value_semantics,
        "dimensions": [
            {
                "dimension_id": dimension.dimension_id,
                "score_semantics": dimension.score_semantics.value,
                "value_min": dimension.value_min,
                "value_max": dimension.value_max,
                "target_min": dimension.target_min,
                "target_max": dimension.target_max,
            }
            for dimension in dimensions
        ],
    }
    return sha256_digest(canonical_json_bytes(payload))


def compute_report_semantics_id(
    *,
    report_kind: str,
    subject_kind: str,
    dimension_namespace: str,
    duplicate_value_semantics: str,
) -> str:
    """Content digest of a report's report/subject-level compatibility
    claim - the layer `compute_compatibility_schema_id` does not cover.

    Two compiled reports are only genuinely comparable if both this digest
    and `compatibility_schema_id` agree; two structurally identical
    dimension schemas over different subject kinds (sentences vs. claims)
    or report kinds must not be treated as compatible.
    """
    payload = {
        "report_kind": report_kind,
        "subject_kind": subject_kind,
        "dimension_namespace": dimension_namespace,
        "duplicate_value_semantics": duplicate_value_semantics,
    }
    return sha256_digest(canonical_json_bytes(payload))
