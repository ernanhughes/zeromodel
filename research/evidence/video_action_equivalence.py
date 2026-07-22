from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from zeromodel.core.artifact import VPMValidationError


VIDEO_ACTION_EQUIVALENCE_AUDIT_VERSION = "zeromodel-video-action-equivalence-audit/v1"
VIDEO_ACTION_SET_DECISION_VERSION = "zeromodel-video-action-set-decision/v1"
VIDEO_REACHABILITY_REPLAY_VERSION = "zeromodel-video-reachability-replay/v1"
VIDEO_RETROSPECTIVE_EVIDENCE_CLOSURE_VERSION = "zeromodel-video-retrospective-evidence-closure/v1"
VIDEO_RETROSPECTIVE_EVIDENCE_INVENTORY_VERSION = "zeromodel-video-retrospective-evidence-inventory/v2"
VIDEO_POLICY_REACHABILITY_TILE_VERSION = "zeromodel-video-policy-reachability-tile/v1"
VIDEO_ROW_SET_CONFORMAL_VERSION = "zeromodel-video-row-set-conformal/v1"
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_policy_row_action_map(*, policy_artifact_id: str) -> tuple[dict[str, str], ...]:
    from zeromodel.video.arcade_policy import ACTIONS, compile_policy_artifact
    from zeromodel.core.policy_lookup import VPMPolicyLookup

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
        action_id = lookup.choose(str(row_id))
        rows.append(
            {
                "policy_artifact_id": artifact.artifact_id,
                "policy_version": "arcade_shooter_policy/v1",
                "policy_source_digest": source_digest,
                "row_id": str(row_id),
                "action_id": str(action_id),
                "source_mapping_digest": _sha256(
                    {"row_id": str(row_id), "action_id": str(action_id), "policy_artifact_id": artifact.artifact_id}
                ),
                "lookup_digest": _sha256({"row_id": str(row_id), "winner": str(action_id)}),
            }
        )
    return tuple(rows)


def policy_action_for_row(
    row_id: str,
    *,
    row_action_map: Sequence[Mapping[str, str]],
    policy_artifact_id: str,
) -> str:
    for entry in row_action_map:
        if entry["policy_artifact_id"] == policy_artifact_id and entry["row_id"] == row_id:
            return entry["action_id"]
    raise VPMValidationError(f"unknown policy row: {row_id}")


def collect_v3_preservation_manifest(repo_root: Path) -> dict[str, str]:
    target = repo_root / "docs" / "results" / "video-discriminative-local-evidence-v3"
    rows: dict[str, str] = {}
    for path in sorted(target.rglob("*")):
        if path.is_file():
            rows[path.relative_to(repo_root).as_posix()] = _file_sha256(path)
    return rows


def verify_v3_preservation(repo_root: Path, manifest: Mapping[str, str]) -> dict[str, Any]:
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


def summarize_top1_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_row_field: str,
    predicted_row_field: str,
    predicted_action_field: str,
    expected_action_field: str,
) -> dict[str, Any]:
    total = 0
    row_correct = 0
    action_correct = 0
    same_action_wrong_row = 0
    for row in rows:
        expected_row = row.get(expected_row_field)
        predicted_row = row.get(predicted_row_field)
        expected_action = row.get(expected_action_field)
        predicted_action = row.get(predicted_action_field)
        if not expected_row or not predicted_row or not expected_action or not predicted_action:
            continue
        total += 1
        row_ok = predicted_row == expected_row
        action_ok = predicted_action == expected_action
        row_correct += int(row_ok)
        action_correct += int(action_ok)
        same_action_wrong_row += int((not row_ok) and action_ok)
    return {
        "observation_count": total,
        "row_top1_accuracy": None if total == 0 else row_correct / float(total),
        "action_top1_accuracy": None if total == 0 else action_correct / float(total),
        "same_action_wrong_row_count": same_action_wrong_row,
        "raw_action_gap": None if total == 0 else (action_correct - row_correct) / float(total),
    }


def classify_score_evidence(
    *,
    full_vector: bool,
    full_ranking: bool,
    top_k: bool,
    top1_only: bool,
    aggregate_only: bool,
    reproducible: bool,
) -> tuple[str, str, str]:
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
) -> tuple[bool, tuple[str, ...]]:
    reasons = []
    if not frame_order_available:
        reasons.append("missing_sequence_order")
    if not executed_actions_available:
        reasons.append("missing_executed_actions")
    if recommended_actions_available and not executed_actions_available:
        reasons.append("recommended_action_is_not_executed_action")
    return (len(reasons) == 0, tuple(reasons))


class ScoreVectorAvailability(dict):
    pass


class SequenceReplayAvailability(dict):
    pass


class HistoricalProviderEvidence(dict):
    pass


class MetricVerificationRecord(dict):
    pass


class PolicyRowActionMapEntry(dict):
    pass


class EvidenceInventory(dict):
    pass


__all__ = [
    "VIDEO_ACTION_EQUIVALENCE_AUDIT_VERSION",
    "VIDEO_ACTION_SET_DECISION_VERSION",
    "VIDEO_POLICY_REACHABILITY_TILE_VERSION",
    "VIDEO_REACHABILITY_REPLAY_VERSION",
    "VIDEO_RETROSPECTIVE_EVIDENCE_CLOSURE_VERSION",
    "VIDEO_RETROSPECTIVE_EVIDENCE_INVENTORY_VERSION",
    "VIDEO_ROW_SET_CONFORMAL_VERSION",
    "VIDEO_TILE_COMPOSITION_TRACE_VERSION",
    "EvidenceInventory",
    "HistoricalProviderEvidence",
    "MetricVerificationRecord",
    "PolicyRowActionMapEntry",
    "ScoreVectorAvailability",
    "SequenceReplayAvailability",
    "_file_sha256",
    "_json_ready",
    "_load_csv",
    "_load_json",
    "_load_jsonl",
    "_sha256",
    "_write_csv",
    "_write_json",
    "_write_markdown",
    "build_policy_row_action_map",
    "classify_score_evidence",
    "collect_v3_preservation_manifest",
    "policy_action_for_row",
    "replay_eligibility",
    "summarize_top1_records",
    "verify_v3_preservation",
]
