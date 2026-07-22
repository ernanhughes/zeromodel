from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.content_identity import canonical_float64_bytes, sha256_digest


VIDEO_COMPLETE_ROW_EVIDENCE_VERSION = "zeromodel-video-complete-row-evidence/v2"
VIDEO_COMPLETE_RANKING_VERSION = "zeromodel-video-complete-ranking/v2"
VIDEO_POLICY_ROW_ORDER_VERSION = "zeromodel-video-policy-row-order/v1"
VIDEO_QUANTIZED_SCORE_VECTOR_VERSION = "zeromodel-video-quantized-score-vector/v2"
VIDEO_RAW_SCORE_DIAGNOSTIC_VERSION = "zeromodel-video-raw-score-diagnostic/v1"
VIDEO_SCORE_QUANTIZER_VERSION = "zeromodel-video-score-quantizer/v1"
VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION = "zeromodel-video-semantic-top-set-outcome/v1"
QUANTIZATION_SCALE = 1_000_000
SEMANTIC_TOP_SET_STATUSES = frozenset(
    {
        "unique_row",
        "action_unanimous_tie",
        "conflicting_action_tie",
        "unresolved",
    }
)


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


def _semantic_outcome_digest_payload(
    *,
    policy_artifact_id: str,
    provider_id: str,
    provider_version: str,
    policy_row_universe_digest: str,
    quantized_score_vector_digest: str,
    top_quantized_score: int | None,
    top_row_ids: Sequence[str],
    top_row_actions: Sequence[tuple[str, str]],
    top_action_ids: Sequence[str],
    status: str,
    resolved_row_id: str | None,
    resolved_action_id: str | None,
    rejection_reason: str | None,
    unresolved_reason: str | None,
) -> dict[str, Any]:
    return {
        "version": VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION,
        "policy_artifact_id": policy_artifact_id,
        "provider_id": provider_id,
        "provider_version": provider_version,
        "policy_row_universe_digest": policy_row_universe_digest,
        "quantized_score_vector_digest": quantized_score_vector_digest,
        "top_quantized_score": top_quantized_score,
        "top_row_ids": list(top_row_ids),
        "top_row_actions": [{"row_id": row_id, "action_id": action_id} for row_id, action_id in top_row_actions],
        "top_action_ids": list(top_action_ids),
        "status": status,
        "resolved_row_id": resolved_row_id,
        "resolved_action_id": resolved_action_id,
        "rejection_reason": rejection_reason,
        "unresolved_reason": unresolved_reason,
    }


@dataclass(frozen=True)
class SemanticTopSetOutcome:
    policy_artifact_id: str
    provider_id: str
    provider_version: str
    policy_row_ids: tuple[str, ...]
    policy_row_universe_digest: str
    quantized_score_vector_digest: str
    top_quantized_score: int | None
    top_row_ids: tuple[str, ...]
    top_row_actions: tuple[tuple[str, str], ...]
    top_action_ids: tuple[str, ...]
    status: str
    resolved_row_id: str | None
    resolved_action_id: str | None
    rejection_reason: str | None
    unresolved_reason: str | None
    semantic_outcome_digest: str = ""
    version: str = VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION:
            raise VPMValidationError("unsupported semantic top-set outcome version")
        policy_row_ids = tuple(str(row_id) for row_id in self.policy_row_ids)
        if not policy_row_ids or len(set(policy_row_ids)) != len(policy_row_ids):
            raise VPMValidationError("policy row ids must be unique and non-empty")
        if self.policy_row_universe_digest != _policy_row_universe_digest(
            policy_artifact_id=self.policy_artifact_id,
            row_ids=policy_row_ids,
        ):
            raise VPMValidationError("foreign policy row-universe digest")

        top_row_ids = tuple(str(row_id) for row_id in self.top_row_ids)
        if len(set(top_row_ids)) != len(top_row_ids):
            raise VPMValidationError("top row ids must be unique")
        missing = set(top_row_ids) - set(policy_row_ids)
        if missing:
            raise VPMValidationError("top row absent from policy universe")

        action_pairs = []
        for item in self.top_row_actions:
            if isinstance(item, Mapping):
                row_id = str(item["row_id"])
                action_id = str(item["action_id"])
            else:
                row_id = str(item[0])
                action_id = str(item[1])
            action_pairs.append((row_id, action_id))
        top_row_actions = tuple(sorted(action_pairs))
        if tuple(row_id for row_id, _action in top_row_actions) != tuple(sorted(top_row_ids)):
            raise VPMValidationError("top row/action mapping must exactly cover the top set")
        top_action_ids = tuple(sorted(set(str(action_id) for _row_id, action_id in top_row_actions)))
        if tuple(str(action_id) for action_id in self.top_action_ids) != top_action_ids:
            raise VPMValidationError("top action set is inconsistent with top row/action mapping")

        status = str(self.status)
        if status not in SEMANTIC_TOP_SET_STATUSES:
            raise VPMValidationError("unsupported semantic top-set status")
        if self.top_quantized_score is not None and not (0 <= int(self.top_quantized_score) <= QUANTIZATION_SCALE):
            raise VPMValidationError("top quantized score out of range")
        top_quantized_score = None if self.top_quantized_score is None else int(self.top_quantized_score)
        resolved_row_id = None if self.resolved_row_id is None else str(self.resolved_row_id)
        resolved_action_id = None if self.resolved_action_id is None else str(self.resolved_action_id)
        rejection_reason = None if self.rejection_reason is None else str(self.rejection_reason)
        unresolved_reason = None if self.unresolved_reason is None else str(self.unresolved_reason)

        if status == "unique_row":
            if len(top_row_ids) != 1 or len(top_action_ids) != 1:
                raise VPMValidationError("unique_row status requires exactly one top row and action")
            if resolved_row_id != top_row_ids[0] or resolved_action_id != top_action_ids[0]:
                raise VPMValidationError("unique_row status must resolve the top row and action")
            if rejection_reason is not None or unresolved_reason is not None:
                raise VPMValidationError("unique_row status must not carry rejection or unresolved reasons")
        elif status == "action_unanimous_tie":
            if len(top_row_ids) <= 1 or len(top_action_ids) != 1:
                raise VPMValidationError("action_unanimous_tie status requires multiple top rows with one action")
            if resolved_row_id is not None or resolved_action_id != top_action_ids[0]:
                raise VPMValidationError("action_unanimous_tie must resolve only the action")
            if rejection_reason is not None or unresolved_reason is not None:
                raise VPMValidationError("action_unanimous_tie must not carry rejection or unresolved reasons")
        elif status == "conflicting_action_tie":
            if len(top_row_ids) <= 1 or len(top_action_ids) <= 1:
                raise VPMValidationError("conflicting_action_tie status requires multiple top rows and actions")
            if resolved_row_id is not None or resolved_action_id is not None:
                raise VPMValidationError("conflicting_action_tie must not resolve a row or action")
            if not rejection_reason:
                raise VPMValidationError("conflicting_action_tie requires a rejection reason")
        else:
            if top_row_ids:
                raise VPMValidationError("unresolved status must not carry a manufactured top row")
            if resolved_row_id is not None or resolved_action_id is not None:
                raise VPMValidationError("unresolved status must not resolve a row or action")
            if not unresolved_reason:
                raise VPMValidationError("unresolved status requires an unresolved reason")

        object.__setattr__(self, "policy_row_ids", policy_row_ids)
        object.__setattr__(self, "top_quantized_score", top_quantized_score)
        object.__setattr__(self, "top_row_ids", tuple(sorted(top_row_ids)))
        object.__setattr__(self, "top_row_actions", top_row_actions)
        object.__setattr__(self, "top_action_ids", top_action_ids)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "resolved_row_id", resolved_row_id)
        object.__setattr__(self, "resolved_action_id", resolved_action_id)
        object.__setattr__(self, "rejection_reason", rejection_reason)
        object.__setattr__(self, "unresolved_reason", unresolved_reason)

        digest = sha256_digest(
            _semantic_outcome_digest_payload(
                policy_artifact_id=self.policy_artifact_id,
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                policy_row_universe_digest=self.policy_row_universe_digest,
                quantized_score_vector_digest=self.quantized_score_vector_digest,
                top_quantized_score=top_quantized_score,
                top_row_ids=tuple(sorted(top_row_ids)),
                top_row_actions=top_row_actions,
                top_action_ids=top_action_ids,
                status=status,
                resolved_row_id=resolved_row_id,
                resolved_action_id=resolved_action_id,
                rejection_reason=rejection_reason,
                unresolved_reason=unresolved_reason,
            )
        )
        if self.semantic_outcome_digest and self.semantic_outcome_digest != digest:
            raise VPMValidationError("foreign semantic top-set outcome digest")
        object.__setattr__(self, "semantic_outcome_digest", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "policy_artifact_id": self.policy_artifact_id,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "policy_row_ids": list(self.policy_row_ids),
            "policy_row_universe_digest": self.policy_row_universe_digest,
            "quantized_score_vector_digest": self.quantized_score_vector_digest,
            "top_quantized_score": self.top_quantized_score,
            "top_row_ids": list(self.top_row_ids),
            "top_row_actions": [{"row_id": row_id, "action_id": action_id} for row_id, action_id in self.top_row_actions],
            "top_action_ids": list(self.top_action_ids),
            "status": self.status,
            "resolved_row_id": self.resolved_row_id,
            "resolved_action_id": self.resolved_action_id,
            "rejection_reason": self.rejection_reason,
            "unresolved_reason": self.unresolved_reason,
            "semantic_outcome_digest": self.semantic_outcome_digest,
        }


def build_semantic_top_set_outcome(
    *,
    evidence: CompleteRowEvidence,
    row_action: Mapping[str, str],
) -> SemanticTopSetOutcome:
    action_by_row = {str(row_id): str(action_id) for row_id, action_id in row_action.items()}
    if set(action_by_row) != set(evidence.canonical_row_ids):
        raise VPMValidationError("row/action mapping must cover exactly the policy universe")
    if not evidence.ranking.tie_groups:
        return SemanticTopSetOutcome(
            policy_artifact_id=evidence.policy_artifact_id,
            provider_id=evidence.provider_id,
            provider_version=evidence.provider_version,
            policy_row_ids=evidence.canonical_row_ids,
            policy_row_universe_digest=evidence.policy_row_universe_digest,
            quantized_score_vector_digest=evidence.quantized_score_vector_digest,
            top_quantized_score=None,
            top_row_ids=(),
            top_row_actions=(),
            top_action_ids=(),
            status="unresolved",
            resolved_row_id=None,
            resolved_action_id=None,
            rejection_reason=None,
            unresolved_reason="complete ranking contains no tie groups",
        )
    top_group = evidence.ranking.tie_groups[0]
    top_rows = tuple(sorted(str(row_id) for row_id in top_group.row_ids))
    top_row_actions = tuple((row_id, action_by_row[row_id]) for row_id in top_rows)
    top_action_ids = tuple(sorted(set(action_id for _row_id, action_id in top_row_actions)))
    if len(top_rows) == 1:
        status = "unique_row"
        resolved_row_id = top_rows[0]
        resolved_action_id = top_action_ids[0]
        rejection_reason = None
    elif len(top_action_ids) == 1:
        status = "action_unanimous_tie"
        resolved_row_id = None
        resolved_action_id = top_action_ids[0]
        rejection_reason = None
    else:
        status = "conflicting_action_tie"
        resolved_row_id = None
        resolved_action_id = None
        rejection_reason = "top quantized score is shared by rows governed by multiple actions"
    return SemanticTopSetOutcome(
        policy_artifact_id=evidence.policy_artifact_id,
        provider_id=evidence.provider_id,
        provider_version=evidence.provider_version,
        policy_row_ids=evidence.canonical_row_ids,
        policy_row_universe_digest=evidence.policy_row_universe_digest,
        quantized_score_vector_digest=evidence.quantized_score_vector_digest,
        top_quantized_score=top_group.quantized_score,
        top_row_ids=top_rows,
        top_row_actions=top_row_actions,
        top_action_ids=top_action_ids,
        status=status,
        resolved_row_id=resolved_row_id,
        resolved_action_id=resolved_action_id,
        rejection_reason=rejection_reason,
        unresolved_reason=None,
    )


def semantic_top_set_outcome_from_dict(
    payload: Mapping[str, Any],
    *,
    evidence: CompleteRowEvidence,
    row_action: Mapping[str, str],
) -> SemanticTopSetOutcome:
    expected = build_semantic_top_set_outcome(evidence=evidence, row_action=row_action)
    outcome = SemanticTopSetOutcome(
        policy_artifact_id=str(payload["policy_artifact_id"]),
        provider_id=str(payload["provider_id"]),
        provider_version=str(payload["provider_version"]),
        policy_row_ids=tuple(str(row_id) for row_id in payload["policy_row_ids"]),
        policy_row_universe_digest=str(payload["policy_row_universe_digest"]),
        quantized_score_vector_digest=str(payload["quantized_score_vector_digest"]),
        top_quantized_score=payload.get("top_quantized_score"),
        top_row_ids=tuple(str(row_id) for row_id in payload["top_row_ids"]),
        top_row_actions=tuple(payload["top_row_actions"]),
        top_action_ids=tuple(str(action_id) for action_id in payload["top_action_ids"]),
        status=str(payload["status"]),
        resolved_row_id=payload.get("resolved_row_id"),
        resolved_action_id=payload.get("resolved_action_id"),
        rejection_reason=payload.get("rejection_reason"),
        unresolved_reason=payload.get("unresolved_reason"),
        semantic_outcome_digest=str(payload["semantic_outcome_digest"]),
        version=str(payload.get("version", VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION)),
    )
    if outcome.to_dict() != expected.to_dict():
        raise VPMValidationError("semantic top-set outcome is inconsistent with evidence or policy mapping")
    return outcome


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
    "SEMANTIC_TOP_SET_STATUSES",
    "SemanticTopSetOutcome",
    "TieGroup",
    "VIDEO_COMPLETE_RANKING_VERSION",
    "VIDEO_COMPLETE_ROW_EVIDENCE_VERSION",
    "VIDEO_POLICY_ROW_ORDER_VERSION",
    "VIDEO_QUANTIZED_SCORE_VECTOR_VERSION",
    "VIDEO_RAW_SCORE_DIAGNOSTIC_VERSION",
    "VIDEO_SCORE_QUANTIZER_VERSION",
    "VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION",
    "build_complete_ranking",
    "build_complete_row_evidence",
    "build_semantic_top_set_outcome",
    "quantize_similarity",
    "semantic_top_set_outcome_from_dict",
]
