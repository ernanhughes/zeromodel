from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .artifact import VPMValidationError


VIDEO_COMPLETE_ROW_EVIDENCE_VERSION = "zeromodel-video-complete-row-evidence/v1"
VIDEO_COMPLETE_RANKING_VERSION = "zeromodel-video-complete-ranking/v1"
VIDEO_SCORE_QUANTIZER_VERSION = "zeromodel-video-score-quantizer/v1"
QUANTIZATION_SCALE = 1_000_000


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def quantize_similarity(value: float) -> int:
    if not math.isfinite(float(value)):
        raise VPMValidationError("score must be finite")
    clamped = min(1.0, max(0.0, float(value)))
    quantized = int(math.floor(clamped * QUANTIZATION_SCALE + 0.5))
    if not (0 <= quantized <= QUANTIZATION_SCALE):
        raise VPMValidationError("quantized score out of range")
    return quantized


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
        return {
            "row_id": self.row_id,
            "raw_score": float(self.raw_score),
            "quantized_score": int(self.quantized_score),
        }


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
        return {
            "tie_group_index": int(self.tie_group_index),
            "quantized_score": int(self.quantized_score),
            "row_ids": list(self.row_ids),
        }


@dataclass(frozen=True)
class CompleteRanking:
    ranked_row_ids: tuple[str, ...]
    tie_groups: tuple[TieGroup, ...]
    version: str = VIDEO_COMPLETE_RANKING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "ranked_row_ids": list(self.ranked_row_ids),
            "tie_groups": [item.to_dict() for item in self.tie_groups],
            "ranking_digest": _sha256(
                {
                    "version": self.version,
                    "ranked_row_ids": list(self.ranked_row_ids),
                    "tie_groups": [item.to_dict() for item in self.tie_groups],
                }
            ),
        }


@dataclass(frozen=True)
class CompleteRowEvidence:
    row_scores: tuple[RowScore, ...]
    ranking: CompleteRanking
    policy_artifact_id: str
    provider_id: str
    provider_version: str
    version: str = VIDEO_COMPLETE_ROW_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if len(self.row_scores) != 112:
            raise VPMValidationError("exactly 112 row scores required")
        row_ids = [item.row_id for item in self.row_scores]
        if len(set(row_ids)) != 112:
            raise VPMValidationError("row ids must be unique and complete")
        if tuple(self.ranking.ranked_row_ids) != tuple(item.row_id for item in sorted(self.row_scores, key=lambda item: (-item.quantized_score, item.row_id))):
            raise VPMValidationError("ranking must reconstruct from quantized scores")

    @property
    def score_vector_digest(self) -> str:
        return _sha256([item.to_dict() for item in self.row_scores])

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policy_artifact_id": self.policy_artifact_id,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "row_scores": [item.to_dict() for item in self.row_scores],
            "score_vector_digest": self.score_vector_digest,
            "ranking": self.ranking.to_dict(),
        }


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
    return CompleteRanking(
        ranked_row_ids=tuple(item.row_id for item in ranked),
        tie_groups=tuple(tie_groups),
    )


def build_complete_row_evidence(
    *,
    row_scores: Sequence[tuple[str, float]],
    policy_artifact_id: str,
    provider_id: str,
    provider_version: str,
) -> CompleteRowEvidence:
    if len(row_scores) != 112:
        raise VPMValidationError("exactly 112 scores required")
    scores = tuple(
        RowScore(row_id=str(row_id), raw_score=float(score), quantized_score=quantize_similarity(float(score)))
        for row_id, score in row_scores
    )
    ranking = build_complete_ranking(scores)
    return CompleteRowEvidence(
        row_scores=scores,
        ranking=ranking,
        policy_artifact_id=policy_artifact_id,
        provider_id=provider_id,
        provider_version=provider_version,
    )


__all__ = [
    "CompleteRanking",
    "CompleteRowEvidence",
    "QUANTIZATION_SCALE",
    "RowScore",
    "TieGroup",
    "VIDEO_COMPLETE_RANKING_VERSION",
    "VIDEO_COMPLETE_ROW_EVIDENCE_VERSION",
    "VIDEO_SCORE_QUANTIZER_VERSION",
    "build_complete_ranking",
    "build_complete_row_evidence",
    "quantize_similarity",
]
