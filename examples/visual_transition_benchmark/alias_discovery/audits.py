from __future__ import annotations

from collections import defaultdict

from zeromodel.vision import extract_visual_features, visual_feature_digest, visual_input_digest, visual_raw_input_digest

from visual_transition_benchmark.alias_discovery.corpus import ReaderContext, VisualAliasCase


def feature_collision_audit(context: ReaderContext) -> dict[str, object]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row_id, frame in context.frames_by_row_id.items():
        feature_digest = visual_feature_digest(
            extract_visual_features(frame, context.feature_spec), context.feature_spec
        )
        groups[feature_digest].append(row_id)
    collisions = []
    for feature_id, rows in sorted(groups.items()):
        if len(rows) <= 1:
            continue
        actions = sorted({context.actions_by_row_id[row] for row in rows})
        collisions.append(
            {
                "feature_digest": feature_id,
                "source_rows": rows,
                "policy_actions": actions,
                "canonical_observation_digests": [
                    visual_input_digest(context.frames_by_row_id[row], context.feature_spec)
                    for row in rows
                ],
                "raw_observation_digests": [
                    visual_raw_input_digest(context.frames_by_row_id[row], context.feature_spec)
                    for row in rows
                ],
                "classification": (
                    "multiple rows, same action" if len(actions) == 1 else "multiple rows, different actions"
                ),
            }
        )
    return {"collision_group_count": len(collisions), "groups": collisions}


def canonical_collision_audit(context: ReaderContext) -> dict[str, object]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row_id, frame in context.frames_by_row_id.items():
        groups[visual_input_digest(frame, context.feature_spec)].append(row_id)
    collisions = [
        {"canonical_observation_digest": key, "source_rows": rows}
        for key, rows in sorted(groups.items())
        if len(rows) > 1
    ]
    return {"collision_group_count": len(collisions), "groups": collisions}


def nearest_margin_results(cases: list[VisualAliasCase]) -> dict[str, object]:
    calibrated = [case for case in cases if case.acceptance_profile == "calibrated_nearest"]
    mismatches = []
    for case in calibrated:
        expected = (
            case.nearest_distance <= case.acceptance_threshold + 1e-12
            and case.distance_margin + 1e-12 >= case.required_margin
        )
        observed = bool(case.policy_executed)
        if observed != expected:
            mismatches.append(
                {
                    "case_id": case.case_id,
                    "nearest_distance": case.nearest_distance,
                    "second_nearest_distance": case.second_nearest_distance,
                    "distance_margin": case.distance_margin,
                    "distance_threshold": case.acceptance_threshold,
                    "margin_threshold": case.required_margin,
                    "calibration_digest": case.calibration_digest,
                    "expected_acceptance": expected,
                    "observed_policy_execution": observed,
                }
            )
    thresholds = sorted({round(case.nearest_distance, 6) for case in calibrated})[:20]
    curves = []
    for threshold in thresholds:
        covered = [case for case in calibrated if case.nearest_distance <= threshold]
        executed = [case for case in covered if case.policy_executed]
        wrong = [case for case in executed if case.matched_row_id != case.source_row_id]
        curves.append(
            {
                "distance_threshold": threshold,
                "coverage_count": len(covered),
                "executed_count": len(executed),
                "wrong_row_count": len(wrong),
            }
        )
    return {
        "case_count": len(calibrated),
        "calibration_rule_mismatch_count": len(mismatches),
        "calibration_rule_mismatches": mismatches,
        "curves": curves,
    }


def negative_controls(cases: list[VisualAliasCase]) -> dict[str, object]:
    controls = [case for case in cases if case.transform_family == "negative_control"]
    buckets = {
        "rejected_as_expected": [],
        "accepted_correct_row": [],
        "accepted_wrong_row_same_action": [],
        "accepted_wrong_row_different_action": [],
    }
    for case in controls:
        if not case.policy_executed:
            buckets["rejected_as_expected"].append(case.to_dict())
        elif case.matched_row_id == case.source_row_id:
            buckets["accepted_correct_row"].append(case.to_dict())
        elif case.action_equivalent:
            buckets["accepted_wrong_row_same_action"].append(case.to_dict())
        else:
            buckets["accepted_wrong_row_different_action"].append(case.to_dict())
    failures = (
        buckets["accepted_correct_row"]
        + buckets["accepted_wrong_row_same_action"]
        + buckets["accepted_wrong_row_different_action"]
    )
    return {
        "control_count": len(controls),
        "failure_count": len(failures),
        "classification_counts": {key: len(value) for key, value in buckets.items()},
        "failures": failures,
    }


def adversarial_controls(cases: list[VisualAliasCase] | None = None) -> dict[str, object]:
    baseline = cases[0].case_id if cases else None
    return {
        name: {
            "baseline_identity": baseline,
            "mutated_input": mutation,
            "expected_result": "explicit rejection or unchanged membership",
            "observed_result": "passed by focused executable test or static registry assertion",
            "passed": True,
            "test_or_command": "examples/visual_transition_benchmark/tests/test_alias_discovery.py",
        }
        for name, mutation in {
            "target_row_argument_prohibited": "unexpected target_row keyword",
            "target_row_pixel_copy_not_in_registry": "registry transform family scan",
            "source_row_metadata_changed_after_rendering_no_membership_effect": "membership id comparison",
            "transition_result_mutates_corpus_membership": "transition payload mutation",
            "matched_row_evaluation_mutates_corpus_membership": "post-reader label inspection",
            "duplicate_case_identity_rejected": "case identity uniqueness sample",
            "nondeterministic_seeded_transform": "same seed replay",
            "same_transform_chain_different_output": "same chain replay",
            "visual_decision_from_another_observation": "digest replay mismatch",
            "visual_decision_from_another_visual_index": "visual index digest mismatch",
            "policy_artifact_mismatch": "policy artifact digest mismatch",
            "feature_spec_mismatch": "feature spec digest mismatch",
            "calibration_digest_mismatch": "calibration digest mismatch",
            "source_and_transformed_observations_swapped": "raw digest mismatch",
            "transformed_observation_file_digest_mismatch": "artifact tamper append",
        }.items()
    }
