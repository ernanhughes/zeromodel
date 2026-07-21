from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

from zeromodel.domains.video_action_set.final_access_dto import (
    FINAL_EVIDENCE_BUNDLE_VERSION,
    FINAL_EVALUATION_PROTOCOL_VERSION,
    FINAL_EXECUTION_AUTHORIZATION_VERSION,
    FINAL_EXECUTION_REQUEST_VERSION,
    FinalAccessRecordDTO,
    FinalEvaluationProtocolDTO,
    FinalEvaluationResultDTO,
    FinalEvidenceBundleDTO,
    FinalExecutionAuthorizationDTO,
    FinalExecutionRequestDTO,
)
from zeromodel.domains.video_action_set.canonical_json import canonical_json_bytes
from zeromodel.domains.video_action_set.final_historical_authority import (
    HISTORICAL_EVIDENCE_MANIFEST_VERSION,
    HistoricalEvidenceManifestDTO,
)


def digest(char: str) -> str:
    return "sha256:" + char * 64


def final_rows(
    scores: tuple[str, ...] = ("0.8", "0.9"),
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "family_id": f"family-{index}",
            "episode_id": f"episode-{index}",
            "frame_ordinal": 0,
            "frame_id": f"frame-{index}",
            "provider_id": "P1",
            "split": "final",
            "metrics": {"score": score},
        }
        for index, score in enumerate(scores, start=1)
    )


def approved_protocol(
    *,
    status: str = "approved",
    threshold: object = "0.75",
    expected_row_count: int = 2,
    decision_rule: Mapping[str, object] | None = None,
    protocol_id: str = "synthetic-protocol",
) -> FinalEvaluationProtocolDTO:
    rule = (
        {
            "kind": "fixed_metric_threshold",
            "aggregate": "mean",
            "metric_id": "score",
            "operator": "gte",
            "threshold": threshold,
        }
        if decision_rule is None
        else dict(decision_rule)
    )
    return FinalEvaluationProtocolDTO.create(
        {
            "version": FINAL_EVALUATION_PROTOCOL_VERSION,
            "protocol_id": protocol_id,
            "protocol_status": status,
            "created_utc": "2026-07-21T00:00:00Z",
            "approved_utc": "2026-07-21T00:01:00Z" if status == "approved" else None,
            "approved_by": "reviewer" if status == "approved" else None,
            "benchmark_seed_digest": digest("a"),
            "sealed_plan_digest": digest("b"),
            "policy_artifact_id": "policy",
            "candidate_set_id": "candidate-set",
            "selected_provider_id": "P1",
            "decision_rule": rule,
            "required_evidence": {
                "provider_id": "P1",
                "expected_row_count": expected_row_count,
            },
            "claim_rule_set": {"claims": [{"claim_id": "synthetic-claim"}]},
            "review_notes": {"scope": "synthetic-test"},
        }
    )


def authorization(
    tmp_path: Path,
    protocol: FinalEvaluationProtocolDTO,
    *,
    authorization_id: str = "auth-1",
    expected_counts: Mapping[str, int] | None = None,
    expected_artifacts: Sequence[str] = ("final-summary.json",),
    unattended_permitted: bool = True,
) -> FinalExecutionAuthorizationDTO:
    protocol_file = tmp_path / f"{authorization_id}-protocol.json"
    protocol_file.write_text(
        json.dumps(protocol.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    counts = dict(
        expected_counts
        or {
            "evidence_row_count": 2,
            "episode_count": 2,
            "frame_count": 2,
            "provider_count": 1,
        }
    )
    historical = synthetic_historical_authority(tmp_path, authorization_id)
    return FinalExecutionAuthorizationDTO.create(
        {
            "version": FINAL_EXECUTION_AUTHORIZATION_VERSION,
            "authorization_id": authorization_id,
            "authorization_status": "authorized",
            "created_utc": "2026-07-21T00:02:00Z",
            "created_by": "reviewer",
            "protocol_digest": protocol.protocol_digest,
            "expected_benchmark_seed_digest": protocol.benchmark_seed_digest,
            "expected_sealed_plan_digest": protocol.sealed_plan_digest,
            "expected_policy_artifact_id": protocol.policy_artifact_id,
            "output_dir": str(tmp_path / f"{authorization_id}-final-output"),
            "database_path": str(tmp_path / f"{authorization_id}-final.sqlite3"),
            "unattended_permitted": unattended_permitted,
            "operator_confirmation_text": "CONFIRM FINAL ACCESS",
            "authorization_payload": {
                "protocol_file": str(protocol_file),
                "execution_commit": "synthetic-execution-commit",
                "provider_order": ["P1"],
                "provider_versions": {"P1": "synthetic-v1"},
                "expected_counts": counts,
                "expected_episode_ids": [
                    f"episode-{index}"
                    for index in range(1, counts["episode_count"] + 1)
                ],
                "expected_artifacts": list(expected_artifacts),
                "historical_authority": historical,
            },
        }
    )


def synthetic_historical_authority(
    tmp_path: Path,
    authorization_id: str = "auth-1",
) -> dict[str, object]:
    database_path = (tmp_path / "stage8.sqlite3").resolve()
    if not database_path.exists():
        database_path.write_bytes(b"synthetic Stage 8 authority\n")
    database_sha256 = "sha256:" + hashlib.sha256(database_path.read_bytes()).hexdigest()
    authority_id = "synthetic-stage8-authority"
    stage8_commit = "stage8-synthetic-commit"
    manifest = HistoricalEvidenceManifestDTO.create(
        {
            "version": HISTORICAL_EVIDENCE_MANIFEST_VERSION,
            "historical_authority_id": authority_id,
            "historical_database_path": str(database_path),
            "historical_database_sha256": database_sha256,
            "stage8_commit": stage8_commit,
        }
    )
    manifest_path = (
        tmp_path / f"{authorization_id}-historical-evidence-manifest.json"
    ).resolve()
    manifest_path.write_bytes(canonical_json_bytes(manifest.to_dict()))
    return {
        "version": "zeromodel-stage8-authority/v1",
        "historical_authority_id": authority_id,
        "historical_database_path": str(database_path),
        "historical_database_sha256": database_sha256,
        "evidence_manifest_path": str(manifest_path),
        "evidence_manifest_digest": manifest.evidence_manifest_digest,
        "stage8_commit": stage8_commit,
    }


def request(
    tmp_path: Path,
    auth: FinalExecutionAuthorizationDTO,
    *,
    preflight_only: bool = False,
) -> FinalExecutionRequestDTO:
    authorization_file = tmp_path / f"{auth.authorization_id}-authorization.json"
    authorization_file.write_text(
        json.dumps(auth.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    return FinalExecutionRequestDTO.create(
        {
            "version": FINAL_EXECUTION_REQUEST_VERSION,
            "output_dir": auth.output_dir,
            "authorization_file": str(authorization_file),
            "expected_authorization_digest": auth.authorization_digest,
            "expected_sealed_plan_digest": auth.expected_sealed_plan_digest,
            "database_path": auth.database_path,
            "preflight_only": preflight_only,
            "operator_identity": "synthetic-operator",
            "unattended": True,
            "request_payload": {"scope": "bounded-synthetic-test"},
        }
    )


def evidence_bundle(
    protocol: FinalEvaluationProtocolDTO,
    *,
    rows: Sequence[Mapping[str, object]] | None = None,
    expected_counts: Mapping[str, int] | None = None,
    access_id: str = "final-access:auth-1",
) -> FinalEvidenceBundleDTO:
    return FinalEvidenceBundleDTO.create(
        {
            "version": FINAL_EVIDENCE_BUNDLE_VERSION,
            "access_id": access_id,
            "authorization_digest": digest("c"),
            "protocol_digest": protocol.protocol_digest,
            "benchmark_seed_digest": protocol.benchmark_seed_digest,
            "sealed_plan_digest": protocol.sealed_plan_digest,
            "execution_commit": "synthetic-execution-commit",
            "provider_order": ["P1"],
            "provider_versions": {"P1": "synthetic-v1"},
            "expected_counts": dict(
                expected_counts
                or {
                    "evidence_row_count": 2,
                    "episode_count": 2,
                    "frame_count": 2,
                    "provider_count": 1,
                }
            ),
            "rows": [dict(row) for row in (rows or final_rows())],
        }
    )


class SyntheticFinalExecutor:
    def __init__(
        self,
        *,
        rows: Sequence[Mapping[str, object]] | None = None,
        artifacts: Mapping[str, bytes] | None = None,
    ) -> None:
        self.rows = tuple(rows or final_rows())
        self.artifacts = dict(
            artifacts or {"final-summary.json": b'{"synthetic":true}'}
        )
        self.calls: list[str] = []

    def materialize(
        self,
        access: FinalAccessRecordDTO,
        request_dto: FinalExecutionRequestDTO,
    ) -> object:
        self.calls.append("materialize")
        return {"access_id": access.access_id, "request": request_dto.request_digest}

    def score_providers(
        self,
        access: FinalAccessRecordDTO,
        materialized: object,
    ) -> object:
        self.calls.append("score_providers")
        return {"access_id": access.access_id, "materialized": materialized}

    def assess_reachability(
        self,
        access: FinalAccessRecordDTO,
        scored: object,
    ) -> Sequence[Mapping[str, object]]:
        self.calls.append("assess_reachability")
        return self.rows

    def build_artifacts(
        self,
        access: FinalAccessRecordDTO,
        evidence: FinalEvidenceBundleDTO,
        evaluation: FinalEvaluationResultDTO,
    ) -> Mapping[str, bytes]:
        self.calls.append("build_artifacts")
        return self.artifacts
