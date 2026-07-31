from __future__ import annotations

import inspect
from collections import defaultdict

from zeromodel.vision import extract_visual_features, visual_feature_digest, visual_input_digest, visual_raw_input_digest

from visual_transition_benchmark.alias_discovery._json import digest
from visual_transition_benchmark.alias_discovery.corpus import ReaderContext, VisualAliasCase
from visual_transition_benchmark.alias_discovery.registry import default_registry
from visual_transition_benchmark.alias_discovery.transforms import transform_frame


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


def _control_record(
    *,
    baseline: str | None,
    mutated_input: str,
    expected_result: str,
    observed_result: str,
    passed: bool,
    test_or_command: str,
) -> dict[str, object]:
    return {
        "baseline_identity": baseline,
        "mutated_input": mutated_input,
        "expected_result": expected_result,
        "observed_result": observed_result,
        "passed": passed,
        "test_or_command": test_or_command,
    }


def adversarial_controls(cases: list[VisualAliasCase] | None = None) -> dict[str, object]:
    items = list(cases or [])
    baseline = items[0].case_id if items else None
    signature = inspect.signature(transform_frame)
    registry = default_registry()
    specs_by_id_and_params = {
        (spec.transform_id, tuple(sorted(spec.parameters.items()))): spec for spec in registry
    }
    source_only_chain_identity = all(
        case.transform_chain_id
        == digest(
            {
                "source_row_id": case.source_row_id,
                "source_raw_digest": case.source_observation_raw_digest,
                "transform": specs_by_id_and_params[
                    (case.transform_id, tuple(sorted(case.transform_parameters.items())))
                ].to_dict(),
                "seed": case.transform_seed,
            }
        )
        for case in items
    )
    records = {
        "target_row_argument_prohibited": _control_record(
            baseline=baseline,
            mutated_input="unexpected target_row keyword",
            expected_result="transform_frame has no target row parameter",
            observed_result=str("target_row" not in signature.parameters and "target" not in signature.parameters),
            passed="target_row" not in signature.parameters and "target" not in signature.parameters,
            test_or_command="generated signature inspection; test_source_only_transform_interface_and_target_argument_prohibited",
        ),
        "target_row_pixel_copy_not_in_registry": _control_record(
            baseline=baseline,
            mutated_input="registry transform family scan",
            expected_result="no transform declares target-copy behavior",
            observed_result=str(
                all("target" not in spec.transform_id and "copy_target" not in spec.transform_id for spec in registry)
            ),
            passed=all("target" not in spec.transform_id and "copy_target" not in spec.transform_id for spec in registry),
            test_or_command="generated registry scan",
        ),
        "source_row_metadata_changed_after_rendering_no_membership_effect": _control_record(
            baseline=baseline,
            mutated_input="matched row/action metadata omitted from transform-chain identity",
            expected_result="transform-chain ids recompute from source observation and transform inputs only",
            observed_result=str(source_only_chain_identity),
            passed=source_only_chain_identity,
            test_or_command="generated transform-chain identity recomputation",
        ),
        "matched_row_evaluation_mutates_corpus_membership": _control_record(
            baseline=baseline,
            mutated_input="post-reader matched row/action labels",
            expected_result="case ids remain unique and are not recomputed from evaluation labels",
            observed_result=str(len({case.case_id for case in items}) == len(items)),
            passed=len({case.case_id for case in items}) == len(items),
            test_or_command="generated case identity scan; test_case_identity_determinism_and_duplicate_identity_detection",
        ),
        "nondeterministic_seeded_transform": _control_record(
            baseline=baseline,
            mutated_input="same seed replay",
            expected_result="same transform chain id has one transformed raw digest",
            observed_result=str(
                all(
                    len({case.transformed_observation_raw_digest for case in items if case.transform_chain_id == chain_id}) == 1
                    for chain_id in {case.transform_chain_id for case in items}
                )
            ),
            passed=all(
                len({case.transformed_observation_raw_digest for case in items if case.transform_chain_id == chain_id}) == 1
                for chain_id in {case.transform_chain_id for case in items}
            ),
            test_or_command="generated transform-chain scan; test_seeded_transform_determinism_and_chain_stability",
        ),
        "same_transform_chain_different_output": _control_record(
            baseline=baseline,
            mutated_input="same transform chain replay",
            expected_result="same transform chain id never maps to multiple feature digests",
            observed_result=str(
                all(
                    len({case.transformed_feature_digest for case in items if case.transform_chain_id == chain_id}) == 1
                    for chain_id in {case.transform_chain_id for case in items}
                )
            ),
            passed=all(
                len({case.transformed_feature_digest for case in items if case.transform_chain_id == chain_id}) == 1
                for chain_id in {case.transform_chain_id for case in items}
            ),
            test_or_command="generated transform-chain scan; test_seeded_transform_determinism_and_chain_stability",
        ),
    }
    return {
        "case_count": len(records),
        "passed": all(bool(record["passed"]) for record in records.values()),
        "controls": records,
        "removed_declarative_controls": [
            "visual_decision_from_another_observation",
            "visual_decision_from_another_visual_index",
            "policy_artifact_mismatch",
            "feature_spec_mismatch",
            "calibration_digest_mismatch",
            "source_and_transformed_observations_swapped",
            "transition_result_mutates_corpus_membership",
            "transformed_observation_file_digest_mismatch",
        ],
    }
