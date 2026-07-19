from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .artifact import VPMValidationError


VIDEO_ACTION_EQUIVALENCE_AUDIT_VERSION = "zeromodel-video-action-equivalence-audit/v1"
VIDEO_ACTION_SET_DECISION_VERSION = "zeromodel-video-action-set-decision/v1"
VIDEO_RETROSPECTIVE_EVIDENCE_INVENTORY_VERSION = "zeromodel-video-retrospective-evidence-inventory/v1"
VIDEO_ROW_SET_CONFORMAL_VERSION = "zeromodel-video-row-set-conformal/v1"
VIDEO_POLICY_REACHABILITY_TILE_VERSION = "zeromodel-video-policy-reachability-tile/v1"
VIDEO_REACHABILITY_REPLAY_VERSION = "zeromodel-video-reachability-replay/v1"
VIDEO_TILE_COMPOSITION_TRACE_VERSION = "zeromodel-video-tile-composition-trace/v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _json_ready(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("action-equivalence values must be JSON serializable") from exc


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True, ensure_ascii=False)
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fieldnames})


def _require(value: Optional[str], name: str) -> str:
    text = "" if value is None else str(value)
    if not text:
        raise VPMValidationError(f"{name} cannot be empty")
    return text


@dataclass(frozen=True)
class ScoreVectorAvailability:
    status: str
    detail: str

    def __post_init__(self) -> None:
        allowed = {
            "stored_original_scores",
            "recomputed_from_frozen_committed_artifacts",
            "aggregate_only",
            "top1_only",
            "missing",
        }
        if self.status not in allowed:
            raise VPMValidationError("unsupported score vector availability status")

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class SequenceReplayAvailability:
    clip_ids_available: bool
    episode_ids_available: bool
    frame_order_available: bool
    executed_actions_available: bool
    recommended_actions_available: bool
    expected_actions_available: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_ids_available": bool(self.clip_ids_available),
            "episode_ids_available": bool(self.episode_ids_available),
            "frame_order_available": bool(self.frame_order_available),
            "executed_actions_available": bool(self.executed_actions_available),
            "recommended_actions_available": bool(self.recommended_actions_available),
            "expected_actions_available": bool(self.expected_actions_available),
        }


@dataclass(frozen=True)
class HistoricalProviderEvidence:
    system_id: str
    system_version: str
    provider_family: str
    source_commit: str
    source_result_directory: str
    benchmark_version: str
    benchmark_digest: str
    policy_artifact_id: str
    provider_contract_digest: str
    score_semantics: str
    higher_or_lower_is_better: str
    score_range: str
    full_112_score_vector_available: str
    ordered_full_ranking_available: str
    top_k_available: str
    top_1_only: bool
    raw_observation_pixels_available: bool
    provider_reproducible_from_committed_code: bool
    reproduction_command: str
    calibration_split_available: bool
    evaluation_split_available: bool
    negative_split_available: bool
    clip_ids_available: bool
    episode_ids_available: bool
    frame_order_available: bool
    executed_actions_available: bool
    recommended_actions_available: bool
    expected_actions_available: bool
    row_to_action_map_available: bool
    information_theoretic_controls_identified: bool
    historical_final_data_already_unblinded: bool
    eligible_for_top1_action_rescore: bool
    eligible_for_top_k_rescore: bool
    eligible_for_score_gap_rescore: bool
    eligible_for_conformal_rescore: bool
    eligible_for_reachability_replay: bool
    ineligibility_reasons: Tuple[str, ...] = field(default_factory=tuple)
    notes: Mapping[str, Any] = field(default_factory=dict)
    version: str = VIDEO_RETROSPECTIVE_EVIDENCE_INVENTORY_VERSION

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "version": self.version,
            "system_id": self.system_id,
            "system_version": self.system_version,
            "provider_family": self.provider_family,
            "source_commit": self.source_commit,
            "source_result_directory": self.source_result_directory,
            "benchmark_version": self.benchmark_version,
            "benchmark_digest": self.benchmark_digest,
            "policy_artifact_id": self.policy_artifact_id,
            "provider_contract_digest": self.provider_contract_digest,
            "score_semantics": self.score_semantics,
            "higher_or_lower_is_better": self.higher_or_lower_is_better,
            "score_range": self.score_range,
            "full_112_score_vector_available": self.full_112_score_vector_available,
            "ordered_full_ranking_available": self.ordered_full_ranking_available,
            "top_k_available": self.top_k_available,
            "top_1_only": bool(self.top_1_only),
            "raw_observation_pixels_available": bool(self.raw_observation_pixels_available),
            "provider_reproducible_from_committed_code": bool(self.provider_reproducible_from_committed_code),
            "reproduction_command": self.reproduction_command,
            "calibration_split_available": bool(self.calibration_split_available),
            "evaluation_split_available": bool(self.evaluation_split_available),
            "negative_split_available": bool(self.negative_split_available),
            "clip_ids_available": bool(self.clip_ids_available),
            "episode_ids_available": bool(self.episode_ids_available),
            "frame_order_available": bool(self.frame_order_available),
            "executed_actions_available": bool(self.executed_actions_available),
            "recommended_actions_available": bool(self.recommended_actions_available),
            "expected_actions_available": bool(self.expected_actions_available),
            "row_to_action_map_available": bool(self.row_to_action_map_available),
            "information_theoretic_controls_identified": bool(self.information_theoretic_controls_identified),
            "historical_final_data_already_unblinded": bool(self.historical_final_data_already_unblinded),
            "eligible_for_top1_action_rescore": bool(self.eligible_for_top1_action_rescore),
            "eligible_for_top_k_rescore": bool(self.eligible_for_top_k_rescore),
            "eligible_for_score_gap_rescore": bool(self.eligible_for_score_gap_rescore),
            "eligible_for_conformal_rescore": bool(self.eligible_for_conformal_rescore),
            "eligible_for_reachability_replay": bool(self.eligible_for_reachability_replay),
            "ineligibility_reasons": list(self.ineligibility_reasons),
            "notes": _json_ready(self.notes),
        }
        payload["evidence_digest"] = _sha256(payload)
        return payload


@dataclass(frozen=True)
class MetricVerificationRecord:
    reported_claim_id: str
    provider_id: str
    metric_name: str
    reported_value: Optional[float]
    repository_derived_value: Optional[float]
    unit: str
    source_artifact: str
    calculation_method: str
    match_status: str
    difference: Optional[float]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PolicyRowActionMapEntry:
    policy_artifact_id: str
    policy_version: str
    policy_source_digest: str
    row_id: str
    action_id: str
    source_mapping_digest: str
    lookup_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class EvidenceInventory:
    audit_version: str
    inventory_version: str
    providers: Tuple[HistoricalProviderEvidence, ...]
    verification_records: Tuple[MetricVerificationRecord, ...]
    row_action_map: Tuple[PolicyRowActionMapEntry, ...]
    frozen_v3_manifest: Mapping[str, str]
    phase_access_audits: Mapping[str, int]
    superseded_experiment: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "audit_version": self.audit_version,
            "inventory_version": self.inventory_version,
            "providers": [item.to_dict() for item in self.providers],
            "verification_records": [item.to_dict() for item in self.verification_records],
            "row_action_map": [item.to_dict() for item in self.row_action_map],
            "frozen_v3_manifest": dict(self.frozen_v3_manifest),
            "phase_access_audits": dict(self.phase_access_audits),
            "superseded_experiment": _json_ready(self.superseded_experiment),
        }
        payload["inventory_digest"] = _sha256(payload)
        return payload


def build_policy_row_action_map(*, policy_artifact_id: str) -> Tuple[PolicyRowActionMapEntry, ...]:
    from examples.arcade_shooter_policy import ACTIONS, compile_policy_artifact
    from .policy_lookup import VPMPolicyLookup

    artifact = compile_policy_artifact()
    if artifact.artifact_id != policy_artifact_id:
        raise VPMValidationError("provider cannot be rescored with a mismatched policy version")
    lookup = VPMPolicyLookup(artifact, action_metric_ids=ACTIONS)
    source_digest = _sha256(
        {
            "artifact_id": artifact.artifact_id,
            "row_ids": list(artifact.source.row_ids),
            "metric_ids": list(artifact.source.metric_ids),
        }
    )
    rows = []
    for row_id in artifact.source.row_ids:
        action_id = lookup.choose(row_id)
        source_mapping_digest = _sha256({"row_id": row_id, "action_id": action_id, "policy_artifact_id": artifact.artifact_id})
        rows.append(
            PolicyRowActionMapEntry(
                policy_artifact_id=artifact.artifact_id,
                policy_version="arcade_shooter_policy/v1",
                policy_source_digest=source_digest,
                row_id=str(row_id),
                action_id=str(action_id),
                source_mapping_digest=source_mapping_digest,
                lookup_digest=_sha256({"row_id": row_id, "winner": action_id}),
            )
        )
    return tuple(rows)


def policy_action_for_row(
    row_id: str,
    *,
    row_action_map: Sequence[PolicyRowActionMapEntry],
    policy_artifact_id: str,
) -> str:
    for entry in row_action_map:
        if entry.policy_artifact_id == policy_artifact_id and entry.row_id == row_id:
            return entry.action_id
    raise VPMValidationError(f"unknown policy row: {row_id}")


def collect_v3_preservation_manifest(repo_root: Path) -> Dict[str, str]:
    target = repo_root / "docs" / "results" / "video-discriminative-local-evidence-v3"
    rows: Dict[str, str] = {}
    for path in sorted(target.rglob("*")):
        if path.is_file():
            rel = path.relative_to(repo_root).as_posix()
            rows[rel] = _file_sha256(path)
    return rows


def verify_v3_preservation(repo_root: Path, manifest: Mapping[str, str]) -> Dict[str, Any]:
    current = collect_v3_preservation_manifest(repo_root)
    mismatches = []
    for path, digest in sorted(manifest.items()):
        if current.get(path) != digest:
            mismatches.append({"path": path, "expected": digest, "actual": current.get(path)})
    extra = sorted(set(current) - set(manifest))
    return {
        "verified": not mismatches and not extra,
        "mismatches": mismatches,
        "extra_files": extra,
        "current_digest": _sha256(current),
    }


def classify_score_evidence(
    *,
    full_vector: bool,
    full_ranking: bool,
    top_k: bool,
    top1_only: bool,
    aggregate_only: bool,
    reproducible: bool,
) -> Tuple[str, str, str]:
    if full_vector:
        return ("stored_original_scores", "stored_original_scores", "stored_original_scores")
    if full_ranking:
        return ("missing", "stored_original_scores", "stored_original_scores")
    if top_k:
        return ("missing", "missing", "stored_original_scores")
    if top1_only:
        return ("top1_only", "top1_only", "top1_only")
    if aggregate_only:
        return ("aggregate_only", "aggregate_only", "aggregate_only")
    if reproducible:
        return (
            "recomputed_from_frozen_committed_artifacts",
            "recomputed_from_frozen_committed_artifacts",
            "recomputed_from_frozen_committed_artifacts",
        )
    return ("missing", "missing", "missing")


def replay_eligibility(
    *,
    frame_order_available: bool,
    executed_actions_available: bool,
    recommended_actions_available: bool,
) -> Tuple[bool, Tuple[str, ...]]:
    reasons = []
    if not frame_order_available:
        reasons.append("missing_sequence_order")
    if not executed_actions_available:
        reasons.append("missing_executed_actions")
    if recommended_actions_available and not executed_actions_available:
        reasons.append("recommended_action_is_not_executed_action")
    return (len(reasons) == 0, tuple(reasons))

