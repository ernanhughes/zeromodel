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
