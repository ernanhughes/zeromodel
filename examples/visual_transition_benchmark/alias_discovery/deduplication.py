from __future__ import annotations

from collections import defaultdict

from visual_transition_benchmark.alias_discovery._json import digest
from visual_transition_benchmark.alias_discovery.corpus import VisualAliasCase


def visual_alias_key(case: VisualAliasCase) -> tuple[str, str, str, str, str | None]:
    return (
        case.source_row_id,
        case.transformed_observation_raw_digest,
        case.transformed_observation_canonical_digest,
        case.transformed_feature_digest,
        case.matched_row_id,
    )


def visual_alias_identity_payload(case: VisualAliasCase) -> dict[str, str | None]:
    return {
        "source_row_id": case.source_row_id,
        "transformed_observation_raw_digest": case.transformed_observation_raw_digest,
        "transformed_observation_canonical_digest": case.transformed_observation_canonical_digest,
        "transformed_feature_digest": case.transformed_feature_digest,
        "matched_row_id": case.matched_row_id,
    }


def unique_wrong_row_aliases(cases: list[VisualAliasCase]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str, str | None], list[VisualAliasCase]] = defaultdict(list)
    for case in cases:
        if case.policy_executed and case.matched_row_id != case.source_row_id:
            groups[visual_alias_key(case)].append(case)
    aliases = []
    for group in groups.values():
        representative = sorted(group, key=lambda item: item.case_id)[0]
        aliases.append(
            {
                "visual_alias_id": digest(visual_alias_identity_payload(representative)),
                "representative_profile_case_id": representative.case_id,
                "source_row_id": representative.source_row_id,
                "source_action": representative.source_action,
                "matched_row_id": representative.matched_row_id,
                "matched_action": representative.matched_action,
                "action_equivalent": representative.action_equivalent,
                "transformed_observation_raw_digest": representative.transformed_observation_raw_digest,
                "transformed_observation_canonical_digest": representative.transformed_observation_canonical_digest,
                "transformed_feature_digest": representative.transformed_feature_digest,
                "accepting_profiles": sorted({item.acceptance_profile for item in group}),
                "profile_case_ids": sorted(item.case_id for item in group),
                "transform_chain_ids": sorted({item.transform_chain_id for item in group}),
                "transform_families": sorted({item.transform_family for item in group}),
            }
        )
    return sorted(aliases, key=lambda item: str(item["visual_alias_id"]))


def deduplicate(cases: list[VisualAliasCase]) -> dict[str, object]:
    groups: dict[tuple[str, str, str, str, str, str | None], list[VisualAliasCase]] = defaultdict(list)
    for case in cases:
        key = (
            case.transformed_observation_raw_digest,
            case.transformed_observation_canonical_digest,
            case.transformed_feature_digest,
            case.source_row_id,
            case.acceptance_profile,
            case.matched_row_id,
        )
        groups[key].append(case)
    duplicate_groups = [
        {
            "representative_case_id": group[0].case_id,
            "duplicate_case_ids": [item.case_id for item in group[1:]],
            "all_transform_chain_ids": [item.transform_chain_id for item in group],
        }
        for group in groups.values()
        if len(group) > 1
    ]
    return {
        "generated_case_count": len(cases),
        "unique_transformed_observation_count": len(groups),
        "duplicate_count": sum(len(group) - 1 for group in groups.values()),
        "duplicate_groups": duplicate_groups,
        "unique_wrong_row_visual_alias_count": len(unique_wrong_row_aliases(cases)),
    }
