from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from typing import Iterable

from visual_transition_benchmark.alias_discovery.corpus import VisualAliasCase


def rate(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def wilson(num: int, den: int) -> dict[str, float | int]:
    if den == 0:
        return {"count": num, "denominator": den, "low": 0.0, "high": 0.0}
    z = 1.96
    phat = num / den
    denom = 1 + z * z / den
    centre = (phat + z * z / (2 * den)) / denom
    margin = z * sqrt((phat * (1 - phat) + z * z / (4 * den)) / den) / denom
    return {"count": num, "denominator": den, "low": centre - margin, "high": centre + margin}


def summarize(cases: Iterable[VisualAliasCase]) -> dict[str, object]:
    items = list(cases)
    accepted = [case for case in items if case.accepted]
    executed = [case for case in accepted if case.policy_executed]
    wrong = [case for case in executed if case.matched_row_id != case.source_row_id]
    same = [case for case in wrong if case.action_equivalent]
    different = [case for case in wrong if case.action_equivalent is False]
    return {
        "generated_case_count": len(items),
        "executed_case_count": len(items),
        "failed_transformation_count": 0,
        "accepted_case_count": len(accepted),
        "acceptance_rate": rate(len(accepted), len(items)),
        "rejection_rate": rate(len(items) - len(accepted), len(items)),
        "exact_row_accuracy_among_accepted": rate(
            sum(case.matched_row_id == case.source_row_id for case in executed), len(executed)
        ),
        "policy_action_accuracy_among_accepted": rate(
            sum(case.matched_action == case.source_action for case in executed), len(executed)
        ),
        "accepted_wrong_row_count": len(wrong),
        "wrong_row_accepted_rate": rate(len(wrong), len(executed)),
        "wrong_row_same_action_count": len(same),
        "wrong_row_different_action_count": len(different),
        "action_equivalent_alias_rate": rate(len(same), len(executed)),
        "action_changing_alias_rate": rate(len(different), len(executed)),
        "accepted_wrong_row_per_1000_transformations": rate(len(wrong) * 1000, len(items)),
        "unique_source_rows_producing_aliases": len({case.source_row_id for case in wrong}),
        "unique_matched_rows_receiving_aliases": len({case.matched_row_id for case in wrong}),
        "unique_source_matched_row_pairs": len({(case.source_row_id, case.matched_row_id) for case in wrong}),
        "wrong_row_wilson_interval": wilson(len(wrong), len(executed)),
    }


def breakdown(cases: Iterable[VisualAliasCase], field: str) -> dict[str, object]:
    groups: dict[str, list[VisualAliasCase]] = defaultdict(list)
    for case in cases:
        groups[str(getattr(case, field))].append(case)
    return {
        key: {"count": len(group), "summary": summarize(group)}
        for key, group in sorted(groups.items())
    }


def row_pair_results(cases: Iterable[VisualAliasCase]) -> dict[str, object]:
    wrong = [
        case
        for case in cases
        if case.policy_executed and case.matched_row_id != case.source_row_id
    ]
    counts = Counter(f"{case.source_row_id} -> {case.matched_row_id}" for case in wrong)
    return {"pairs": dict(sorted(counts.items())), "pair_count": len(counts)}
