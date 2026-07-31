from __future__ import annotations

from collections import defaultdict

from visual_transition_benchmark.alias_discovery.corpus import VisualAliasCase


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
    }
