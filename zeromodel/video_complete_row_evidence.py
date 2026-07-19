from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .artifact import VPMValidationError
from .content_identity import canonical_float64_bytes, sha256_digest


VIDEO_COMPLETE_ROW_EVIDENCE_VERSION = "zeromodel-video-complete-row-evidence/v2"
VIDEO_COMPLETE_RANKING_VERSION = "zeromodel-video-complete-ranking/v2"
VIDEO_POLICY_ROW_ORDER_VERSION = "zeromodel-video-policy-row-order/v1"
VIDEO_QUANTIZED_SCORE_VECTOR_VERSION = "zeromodel-video-quantized-score-vector/v2"
VIDEO_RAW_SCORE_DIAGNOSTIC_VERSION = "zeromodel-video-raw-score-diagnostic/v1"
VIDEO_SCORE_QUANTIZER_VERSION = "zeromodel-video-score-quantizer/v1"
QUANTIZATION_SCALE = 1_000_000


def quantize_similarity(value: float) -> int:
    if not math.isfinite(float(value)):
        raise VPMValidationError("score must be finite")
    clamped = min(1.0, max(0.0, float(value)))
    quantized = int(math.floor(clamped * QUANTIZATION_SCALE + 0.5))
    if not (0 <= quantized <= QUANTIZATION_SCALE):
        raise VPMValidationError("quantized score out of range")
    return quantized


def _canonical_policy_row_ids(
    row_ids: Sequence[str],
    *,
    policy_row_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    actual = tuple(str(row_id) for row_id in row_ids)
    if len(set(actual)) != len(actual):
        raise VPMValidationError("row ids must be unique")
    if policy_row_ids is not None:
        canonical = tuple(str(row_id) for row_id in policy_row_ids)
        if set(canonical) != set(actual):
            raise VPMValidationError("policy row universe does not match row score ids")
        return canonical
    return tuple(sorted(actual))


def _policy_row_universe_digest(*, policy_artifact_id: str, row_ids: Sequence[str]) -> str:
    return sha256_digest(
        {
            "version": VIDEO_POLICY_ROW_ORDER_VERSION,
            "policy_artifact_id": policy_artifact_id,
            "row_ids": list(row_ids),
        }
    )


@dataclass(frozen=True)
class RowScore:
    row_id: str
    raw_score: float
    quantized_score: int

    def __post_init__(self) -> None:
        if not str(self.row_id):
            raise VPMValidationError("row_id cannot be empty")
        if not math.isfinite(float(self.raw_score)):
            raise VPMValidationError("raw_score must be finite")
        if not (0 <= int(self.quantized_score) <= QUANTIZATION_SCALE):
            raise VPMValidationError("quantized_score out of range")

    def to_dict(self) -> dict[str, Any]:
        return {"row_id": self.row_id, "raw_score": float(self.raw_score), "quantized_score": int(self.quantized_score)}


@dataclass(frozen=True)
class TieGroup:
    tie_group_index: int
    quantized_score: int
    row_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if int(self.tie_group_index) < 0:
            raise VPMValidationError("tie_group_index must be non-negative")
        if not self.row_ids:
            raise VPMValidationError("tie groups cannot be empty")
        if len(set(self.row_ids)) != len(self.row_ids):
            raise VPMValidationError("tie group row_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {"tie_group_index": int(self.tie_group_index), "quantized_score": int(self.quantized_score), "row_ids": list(self.row_ids)}


def _ranking_digest_payload(
    *,
    ranked_rows: Sequence[tuple[str, int]],
    tie_groups: Sequence[TieGroup],
) -> dict[str, Any]:
    return {
        "version": VIDEO_COMPLETE_RANKING_VERSION,
        "ranked_rows": [{"row_id": row_id, "quantized_score": quantized} for row_id, quantized in ranked_rows],
        "tie_groups": [item.to_dict() for item in tie_groups],
    }


@dataclass(frozen=True)
class CompleteRanking:
    ranked_row_ids: tuple[str, ...]
    tie_groups: tuple[TieGroup, ...]
    ranking_digest: str
    version: str = VIDEO_COMPLETE_RANKING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "ranked_row_ids": list(self.ranked_row_ids),
            "tie_groups": [item.to_dict() for item in self.tie_groups],
            "ranking_digest": self.ranking_digest,
        }


@dataclass(frozen=True)
class CompleteRowEvidence:
    row_scores: tuple[RowScore, ...]
    ranking: CompleteRanking
    policy_artifact_id: str
    provider_id: str
    provider_version: str
    canonical_row_ids: tuple[str, ...]
    policy_row_universe_digest: str
    quantized_score_vector_digest: str
    raw_score_diagnostic_digest: str
    version: str = VIDEO_COMPLETE_ROW_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if len(self.row_scores) != 112:
            raise VPMValidationError("exactly 112 row scores required")
        row_by_id = {item.row_id: item for item in self.row_scores}
        if len(row_by_id) != 112:
            raise VPMValidationError("row ids must be unique and complete")
        if tuple(self.canonical_row_ids) != _canonical_policy_row_ids(row_by_id.keys(), policy_row_ids=self.canonical_row_ids):
            raise VPMValidationError("canonical row ids must be unique and stable")
        if self.policy_row_universe_digest != _policy_row_universe_digest(
            policy_artifact_id=self.policy_artifact_id,
            row_ids=self.canonical_row_ids,
        ):
            raise VPMValidationError("foreign policy row-universe digest")
        canonical_scores = tuple(row_by_id[row_id] for row_id in self.canonical_row_ids)
        if self.quantized_score_vector_digest != _quantized_score_vector_digest(
            policy_artifact_id=self.policy_artifact_id,
            policy_row_universe_digest=self.policy_row_universe_digest,
            row_scores=canonical_scores,
        ):
            raise VPMValidationError("stored quantized digest does not match recomputed digest")
        if self.raw_score_diagnostic_digest != _raw_score_diagnostic_digest(canonical_scores):
            raise VPMValidationError("stored raw diagnostic digest does not match recomputed digest")
        expected_ranking = build_complete_ranking(canonical_scores)
        if self.ranking.ranked_row_ids != expected_ranking.ranked_row_ids or tuple(group.to_dict() for group in self.ranking.tie_groups) != tuple(
            group.to_dict() for group in expected_ranking.tie_groups
        ):
            raise VPMValidationError("ranking must reconstruct from quantized scores")
        if self.ranking.ranking_digest != expected_ranking.ranking_digest:
            raise VPMValidationError("stored ranking digest does not match recomputed digest")

    @property
    def score_vector_digest(self) -> str:
        return self.quantized_score_vector_digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policy_artifact_id": self.policy_artifact_id,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "canonical_row_order_version": VIDEO_POLICY_ROW_ORDER_VERSION,
            "policy_row_ids": list(self.canonical_row_ids),
            "policy_row_universe_digest": self.policy_row_universe_digest,
            "row_scores": [item.to_dict() for item in self.row_scores],
            "quantized_score_vector_digest": self.quantized_score_vector_digest,
            "raw_score_diagnostic_digest": self.raw_score_diagnostic_digest,
            "score_vector_digest": self.quantized_score_vector_digest,
            "ranking": self.ranking.to_dict(),
        }


def _quantized_score_vector_digest(
    *,
    policy_artifact_id: str,
    policy_row_universe_digest: str,
    row_scores: Sequence[RowScore],
) -> str:
    return sha256_digest(
        {
            "version": VIDEO_QUANTIZED_SCORE_VECTOR_VERSION,
            "policy_artifact_id": policy_artifact_id,
            "policy_row_universe_digest": policy_row_universe_digest,
            "quantizer_identity": {"version": VIDEO_SCORE_QUANTIZER_VERSION, "scale": QUANTIZATION_SCALE},
            "rows": [{"row_id": item.row_id, "quantized_score": int(item.quantized_score)} for item in row_scores],
        }
    )


def _raw_score_diagnostic_digest(row_scores: Sequence[RowScore]) -> str:
    return sha256_digest(
        {
            "version": VIDEO_RAW_SCORE_DIAGNOSTIC_VERSION,
            "rows": [{"row_id": item.row_id, "raw_score_binary64": canonical_float64_bytes(item.raw_score).hex()} for item in row_scores],
        }
    )


def build_complete_ranking(row_scores: Sequence[RowScore]) -> CompleteRanking:
    ranked = sorted(row_scores, key=lambda item: (-item.quantized_score, item.row_id))
    tie_groups = []
    current_rows: list[str] = []
    current_score: int | None = None
    index = -1
    for item in ranked:
        if current_score != item.quantized_score:
            if current_rows:
                tie_groups.append(TieGroup(index, int(current_score), tuple(current_rows)))
            index += 1
            current_score = item.quantized_score
            current_rows = [item.row_id]
        else:
            current_rows.append(item.row_id)
    if current_rows:
        tie_groups.append(TieGroup(index, int(current_score), tuple(current_rows)))
    digest = sha256_digest(_ranking_digest_payload(ranked_rows=[(item.row_id, item.quantized_score) for item in ranked], tie_groups=tie_groups))
    return CompleteRanking(ranked_row_ids=tuple(item.row_id for item in ranked), tie_groups=tuple(tie_groups), ranking_digest=digest)


def build_complete_row_evidence(
    *,
    row_scores: Sequence[tuple[str, float]],
    policy_artifact_id: str,
    provider_id: str,
    provider_version: str,
    policy_row_ids: Sequence[str] | None = None,
) -> CompleteRowEvidence:
    if len(row_scores) != 112:
        raise VPMValidationError("exactly 112 scores required")
    raw_by_id = {str(row_id): float(score) for row_id, score in row_scores}
    if len(raw_by_id) != len(row_scores):
        raise VPMValidationError("row ids must be unique")
    canonical_row_ids = _canonical_policy_row_ids(raw_by_id.keys(), policy_row_ids=policy_row_ids)
    scores = tuple(
        RowScore(row_id=row_id, raw_score=raw_by_id[row_id], quantized_score=quantize_similarity(raw_by_id[row_id]))
        for row_id in canonical_row_ids
    )
    ranking = build_complete_ranking(scores)
    policy_row_universe_digest = _policy_row_universe_digest(policy_artifact_id=policy_artifact_id, row_ids=canonical_row_ids)
    return CompleteRowEvidence(
        row_scores=scores,
        ranking=ranking,
        policy_artifact_id=policy_artifact_id,
        provider_id=provider_id,
        provider_version=provider_version,
        canonical_row_ids=canonical_row_ids,
        policy_row_universe_digest=policy_row_universe_digest,
        quantized_score_vector_digest=_quantized_score_vector_digest(
            policy_artifact_id=policy_artifact_id,
            policy_row_universe_digest=policy_row_universe_digest,
            row_scores=scores,
        ),
        raw_score_diagnostic_digest=_raw_score_diagnostic_digest(scores),
    )


__all__ = [
    "CompleteRanking",
    "CompleteRowEvidence",
    "QUANTIZATION_SCALE",
    "RowScore",
    "TieGroup",
    "VIDEO_COMPLETE_RANKING_VERSION",
    "VIDEO_COMPLETE_ROW_EVIDENCE_VERSION",
    "VIDEO_POLICY_ROW_ORDER_VERSION",
    "VIDEO_QUANTIZED_SCORE_VECTOR_VERSION",
    "VIDEO_RAW_SCORE_DIAGNOSTIC_VERSION",
    "VIDEO_SCORE_QUANTIZER_VERSION",
    "build_complete_ranking",
    "build_complete_row_evidence",
    "quantize_similarity",
]
