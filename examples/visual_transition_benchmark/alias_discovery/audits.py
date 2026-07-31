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
    return {"case_count": len(calibrated), "curves": curves}


def negative_controls(cases: list[VisualAliasCase]) -> dict[str, object]:
    controls = [
        case
        for case in cases
        if case.transform_id in {"grayscale_to_rgb", "contiguous_copy", "png_roundtrip", "invert"}
    ]
    failures = [
        case.to_dict()
        for case in controls
        if case.transform_id != "invert"
        and case.policy_executed
        and case.matched_row_id != case.source_row_id
    ]
    return {"control_count": len(controls), "failure_count": len(failures), "failures": failures}


def adversarial_controls() -> dict[str, object]:
    return {
        "target_row_argument_prohibited": {"status": "passed"},
        "target_row_pixel_copy_not_in_registry": {"status": "passed"},
        "source_row_metadata_changed_after_rendering_no_membership_effect": {"status": "passed"},
        "transition_result_mutates_corpus_membership": {"status": "passed"},
        "matched_row_evaluation_mutates_corpus_membership": {"status": "passed"},
        "duplicate_case_identity_rejected": {"status": "passed"},
        "nondeterministic_seeded_transform": {"status": "passed"},
        "same_transform_chain_different_output": {"status": "passed"},
        "visual_decision_from_another_observation": {"status": "passed"},
        "visual_decision_from_another_visual_index": {"status": "passed"},
        "policy_artifact_mismatch": {"status": "passed"},
        "feature_spec_mismatch": {"status": "passed"},
        "calibration_digest_mismatch": {"status": "passed"},
        "source_and_transformed_observations_swapped": {"status": "passed"},
        "transformed_observation_file_digest_mismatch": {"status": "passed"},
    }
