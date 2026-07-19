from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.video_complete_row_evidence import build_complete_row_evidence, build_semantic_top_set_outcome


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("reference-verification-fixture")
    benchmark.freeze_benchmark(output_dir, REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    rows_by_action: dict[str, list[str]] = {}
    for row_id, action_id in row_actions.items():
        rows_by_action.setdefault(action_id, []).append(row_id)
    conflicting_pair = next(
        (left_rows[0], right_rows[0])
        for left_action, left_rows in rows_by_action.items()
        for right_action, right_rows in rows_by_action.items()
        if left_action != right_action
    )
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    evidence_cache: dict[tuple[str, tuple[str, ...]], object] = {}

    def semantic_payload(provider_id: str, top_rows: list[str]) -> tuple[object, object]:
        key = (provider_id, tuple(top_rows))
        if key not in evidence_cache:
            scores = {row_id: 0.1 for row_id in row_ids}
            for row_id in top_rows:
                scores[row_id] = 1.0
            evidence = build_complete_row_evidence(
                row_scores=[(row_id, scores[row_id]) for row_id in row_ids],
                policy_artifact_id=policy.artifact_id,
                provider_id=provider_id,
                provider_version=benchmark._provider_version(provider_id),
                policy_row_ids=row_ids,
            )
            outcome = build_semantic_top_set_outcome(evidence=evidence, row_action=row_actions)
            evidence_cache[key] = (evidence, outcome)
        return evidence_cache[key]  # type: ignore[return-value]

    def provider_row(frame: dict[str, object], provider_id: str, state: dict[str, object]) -> dict[str, object]:
        action_id = str(frame.get("expected_action") or row_actions[row_ids[0]])
        if provider_id == "P1":
            top_rows = [str(frame.get("expected_row")) if frame.get("expected_row") in row_actions else row_ids[0]]
        elif provider_id == "P2":
            top_rows = rows_by_action[action_id][:2]
        else:
            top_rows = list(conflicting_pair)
        evidence, outcome = semantic_payload(provider_id, top_rows)
        trace = benchmark.compose_reachability_trace(
            frame_id=str(frame["frame_id"]),
            semantic_outcome=outcome.to_dict(),
            previous_state=state[provider_id],
            reachability_tile=tile,
            row_actions=row_actions,
        )
        state[provider_id] = benchmark._state_from_trace(trace)
        row_scores = evidence.to_dict()["row_scores"]
        runner_up = evidence.ranking.ranked_row_ids[1]
        runner_up_index = [item.row_id for item in evidence.row_scores].index(runner_up)
        return {
            **frame,
            "provider_id": provider_id,
            "provider_version": benchmark._provider_version(provider_id),
            "policy_artifact_id": policy.artifact_id,
            "reachability_tile_digest": benchmark.REACHABILITY_TILE_DIGEST,
            "all_112_row_ids": [row["row_id"] for row in row_scores],
            "all_112_raw_scores": [row["raw_score"] for row in row_scores],
            "all_112_quantized_scores": [row["quantized_score"] for row in row_scores],
            "complete_ordered_ranking": list(evidence.ranking.ranked_row_ids),
            "tie_groups": [group.to_dict() for group in evidence.ranking.tie_groups],
            "semantic_top_set_outcome": outcome.to_dict(),
            "semantic_status": outcome.status,
            "resolved_row": outcome.resolved_row_id,
            "resolved_action": outcome.resolved_action_id,
            "top_quantized_score": outcome.top_quantized_score,
            "top_row_ids": list(outcome.top_row_ids),
            "top_action_ids": list(outcome.top_action_ids),
            "semantic_outcome_digest": outcome.semantic_outcome_digest,
            "reachability_composition_trace": trace,
            "winner_row": outcome.resolved_row_id,
            "winner_action": outcome.resolved_action_id,
            "winner_quantized_score": outcome.top_quantized_score if outcome.resolved_row_id else None,
            "runner_up_row": runner_up,
            "runner_up_quantized_score": evidence.row_scores[runner_up_index].quantized_score,
            "policy_row_universe_digest": evidence.policy_row_universe_digest,
            "quantized_score_vector_digest": evidence.quantized_score_vector_digest,
            "raw_score_diagnostic_digest": evidence.raw_score_diagnostic_digest,
            "score_vector_digest": evidence.score_vector_digest,
            "ranking_digest": evidence.ranking.to_dict()["ranking_digest"],
            "observation_digest": frame["observation_pixel_digest"],
            "episode_seed": frame["metadata"]["episode_seed"],  # type: ignore[index]
            "generator_identity": {
                "generator_version": benchmark.GENERATOR_VERSION,
                "seed_digest": frame["metadata"]["seed_digest"],  # type: ignore[index]
            },
            "provider_diagnostics": {},
        }

    for split in ("development", "calibration", "selection"):
        frame_rows = [{key: value for key, value in row.items() if key != "pixels"} for row in benchmark._materialize_records(split, REPO_ROOT)]
        provider_rows = []
        reachability_state: dict[str, object] = {"P1": None, "P2": None, "P3": None}
        for frame in frame_rows:
            if frame.get("event_type") == "gap_unknown" or frame.get("observation_pixel_digest") is None:
                for provider_id in reachability_state:
                    reachability_state[provider_id] = benchmark._gap_reachability_state(frame)
                continue
            for provider_id in ("P1", "P2", "P3"):
                provider_rows.append(provider_row(frame, provider_id, reachability_state))
        benchmark._write_jsonl(output_dir / split / "frame-metadata.jsonl", frame_rows)
        benchmark._write_jsonl(output_dir / split / "provider-evidence.jsonl", provider_rows)
        benchmark._write_json(
            output_dir / f"{split}-manifest.json",
            {
                "split": split,
                "observation_count": len(frame_rows),
                "provider_frame_record_count": len(provider_rows),
                "frame_digest": benchmark._sha256(frame_rows),
                "provider_evidence_digest": benchmark._sha256(provider_rows),
            },
        )
    benchmark._write_observation_identity_manifest(output_dir)
    benchmark._write_split_overlap_audit(output_dir)
    benchmark._write_json(output_dir / "phase-access-audits.json", benchmark._measured_phase_access_counts(output_dir))
    return output_dir


def test_reference_verifier_schema_gates_and_counts(reference_fixture: Path) -> None:
    report = benchmark.verify_reference_instrument(reference_fixture, REPO_ROOT)
    assert report["version"] == benchmark.REFERENCE_VERIFICATION_VERSION
    assert report["verified"] is True
    assert {gate["gate"] for gate in report["gates"]} >= set(benchmark._REQUIRED_VERIFICATION_GATES)
    assert all(gate["status"] == "passed" for gate in report["gates"])
    assert report["final_access_measurements"]["final_observation_materialization_count"] == 0
    assert report["final_access_measurements"]["final_provider_score_access_count"] == 0
    assert report["final_access_measurements"]["final_reachability_execution_count"] == 0


def test_reference_verifier_read_only_and_deterministic(reference_fixture: Path) -> None:
    before = benchmark._directory_snapshot(reference_fixture)
    first = benchmark.verify_reference_instrument(reference_fixture, REPO_ROOT, enabled_gates=("structural_identity", "access_prohibition"))
    second = benchmark.verify_reference_instrument(reference_fixture, REPO_ROOT, enabled_gates=("structural_identity", "access_prohibition"))
    after = benchmark._directory_snapshot(reference_fixture)
    assert before == after
    assert first == second
    assert first["verification_digest"] == second["verification_digest"]


@pytest.mark.parametrize(
    ("mutation_name", "expected_code", "gate_scope"),
    [
        ("evidence_quantized_score_changed", "quantized_score_vector_mismatch", ("structural_identity", "semantic_outcome")),
        ("semantic_resolved_row_for_action_unanimous_tie", "resolved_row_not_permitted", ("structural_identity", "semantic_outcome")),
        ("seed_alter_final_sealed_identity", "sealed_episode_identity_mismatch", ("structural_identity", "seed_and_plan")),
        ("access_add_final_observation_artifact", "forbidden_final_materialization", ("structural_identity", "access_prohibition")),
        ("semantic_lexically_reorder_tied_rows", None, ("structural_identity", "semantic_outcome")),
    ],
)
def test_selected_adversarial_mutations_have_expected_primary_codes(
    reference_fixture: Path,
    tmp_path: Path,
    mutation_name: str,
    expected_code: str | None,
    gate_scope: tuple[str, ...],
) -> None:
    case_dir = tmp_path / mutation_name
    shutil.copytree(reference_fixture, case_dir)
    benchmark._apply_reference_mutation(case_dir, mutation_name)
    report = benchmark.verify_reference_instrument(case_dir, REPO_ROOT, enabled_gates=gate_scope)
    assert report["primary_failure_code"] == expected_code


def test_mutation_audit_schema_declares_required_matrix() -> None:
    cases = {case["name"]: case for case in benchmark._MUTATION_CASES}
    for required in (
        "evidence_quantized_score_changed",
        "semantic_alter_outcome_digest",
        "seed_alter_final_sealed_identity",
        "observation_change_pixels_and_recompute_digest",
        "family_clipping_quantization_noop",
        "reachability_change_tile_identity",
        "access_increment_forbidden_access_counter",
    ):
        assert required in cases
    laundering_classes = {case["artifact_class"] for case in cases.values() if case.get("digest_laundering")}
    assert {"evidence", "semantic", "episode_plan", "observation", "family_output", "reachability_trace", "access_status"} <= laundering_classes
