"""Domain-neutral evidence contract model.

Nothing here mentions arcade or warehouse component names -- domains declare
requirements against this shared vocabulary via ``compiler_adapters/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, get_args

EvidenceKind = Literal[
    "presence",
    "numeric_value",
    "categorical_state",
    "spatial_position",
    "signed_delta",
    "exact_magnitude",
    "relation",
    "visible_identity",
]

AggregationKind = Literal[
    "mean",
    "max",
    "min",
    "sum",
    "count_nonzero",
    "centroid",
    "template",
    "exact_pattern",
]

ComparisonKind = Literal[
    "changed",
    "unchanged",
    "equal",
    "not_equal",
    "increase",
    "decrease",
    "signed_delta",
    "exact_delta",
    "categorical_transition",
    "relation_holds",
    "identity_equal",
]

_EVIDENCE_KINDS = frozenset(get_args(EvidenceKind))
_AGGREGATION_KINDS = frozenset(get_args(AggregationKind))
_COMPARISON_KINDS = frozenset(get_args(ComparisonKind))


class EvidenceContractError(ValueError):
    pass


def _canonical_json(payload) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest(payload) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()[:24]}"


@dataclass(frozen=True)
class VisualEvidenceRequirement:
    domain_name: str
    component_type: str
    property_name: str
    evidence_kind: EvidenceKind

    candidate_region_id: str
    expected_value_domain: Tuple[object, ...] = ()

    required_resolution: Optional[Tuple[int, int]] = None
    required_precision: Optional[float] = None
    comparison: ComparisonKind = "changed"

    permits_temporal_pair: bool = True
    permits_local_template: bool = False
    permits_identity_marker: bool = False

    requirement_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.evidence_kind not in _EVIDENCE_KINDS:
            raise EvidenceContractError(
                f"unsupported evidence_kind: {self.evidence_kind}"
            )
        if self.comparison not in _COMPARISON_KINDS:
            raise EvidenceContractError(f"unsupported comparison: {self.comparison}")
        if (
            not self.domain_name
            or not self.component_type
            or not self.property_name
            or not self.candidate_region_id
        ):
            raise EvidenceContractError("requirement identities must be non-empty")
        if (
            self.evidence_kind == "numeric_value"
            and self.required_precision is not None
            and not self.expected_value_domain
        ):
            raise EvidenceContractError(
                "exact-value requirements (numeric_value with a declared required_precision) "
                "must declare expected_value_domain"
            )
        if (
            self.evidence_kind == "visible_identity"
            and not self.permits_identity_marker
        ):
            raise EvidenceContractError(
                "visible_identity requirements must set permits_identity_marker=True"
            )
        payload = {
            "domain_name": self.domain_name,
            "component_type": self.component_type,
            "property_name": self.property_name,
            "evidence_kind": self.evidence_kind,
            "candidate_region_id": self.candidate_region_id,
            "expected_value_domain": [str(v) for v in self.expected_value_domain],
            "required_resolution": list(self.required_resolution)
            if self.required_resolution
            else None,
            "required_precision": self.required_precision,
            "comparison": self.comparison,
            "permits_temporal_pair": self.permits_temporal_pair,
            "permits_local_template": self.permits_local_template,
            "permits_identity_marker": self.permits_identity_marker,
        }
        object.__setattr__(self, "requirement_id", _digest(payload))
