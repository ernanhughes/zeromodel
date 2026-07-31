from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping


def rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def summarize(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    items = list(rows)
    accepted = [row for row in items if row["policy_executed"]]
    wrong = [row for row in accepted if not row["exact_address"]]
    correct = [row for row in accepted if row["exact_address"]]
    initially_covered = [
        row for row in accepted if bool(row.get("true_row_initially_present"))
    ]
    initially_absent = [
        row for row in accepted if not bool(row.get("true_row_initially_present"))
    ]
    unresolved = [
        row
        for row in items
        if str(row["runtime_adjudication_status"]).endswith("unresolved")
        or row["runtime_adjudication_status"] in {"transition_signature_collision"}
    ]
    return {
        "case_count": len(items),
        "accepted_case_count": len(accepted),
        "exact_row_accuracy": rate(
            sum(bool(row["exact_address"]) for row in accepted), len(accepted)
        ),
        "policy_action_accuracy": rate(
            sum(bool(row["same_action"]) for row in accepted), len(accepted)
        ),
        "wrong_row_accepted_rate": rate(len(wrong), len(accepted)),
        "wrong_row_same_action_rate": rate(
            sum(bool(row["same_action"]) for row in wrong), len(wrong)
        ),
        "wrong_row_different_action_rate": rate(
            sum(not bool(row["same_action"]) for row in wrong), len(wrong)
        ),
        "rejection_rate": rate(
            sum(not bool(row["policy_executed"]) for row in items), len(items)
        ),
        "wrong_address_contradiction_rate": rate(
            sum(row["addressed_candidate_status"] == "contradicted" for row in wrong),
            len(wrong),
        ),
        "correct_address_false_contradiction_rate": rate(
            sum(row["addressed_candidate_status"] == "contradicted" for row in correct),
            len(correct),
        ),
        "true_row_retention_rate": rate(
            sum(bool(row["true_row_retained"]) for row in accepted), len(accepted)
        ),
        "true_row_elimination_rate": rate(
            sum(not bool(row["true_row_retained"]) for row in accepted), len(accepted)
        ),
        "addressed_row_retention_rate": rate(
            sum(row["addressed_candidate_status"] == "retained" for row in accepted),
            len(accepted),
        ),
        "candidate_set_reduction_rate": rate(
            sum(bool(row["candidate_reduction"]) for row in accepted), len(accepted)
        ),
        "mean_candidate_count_before": sum(
            int(row["candidate_count_before"]) for row in accepted
        )
        / len(accepted)
        if accepted
        else 0.0,
        "mean_candidate_count_after": sum(
            int(row["candidate_count_after"]) for row in accepted
        )
        / len(accepted)
        if accepted
        else 0.0,
        "unique_transition_consistent_candidate_rate": rate(
            sum(
                row["runtime_adjudication_status"]
                == "unique_transition_consistent_candidate"
                for row in accepted
            ),
            len(accepted),
        ),
        "unique_correction_to_true_row_rate": rate(
            sum(bool(row["unique_correction_to_true_row"]) for row in accepted),
            len(accepted),
        ),
        "unique_resolution_to_wrong_row_rate": rate(
            sum(bool(row["unique_resolution_to_wrong_row"]) for row in accepted),
            len(accepted),
        ),
        "false_confirmation_rate": rate(
            sum(bool(row["false_confirmation"]) for row in accepted), len(accepted)
        ),
        "initial_true_row_coverage_rate": rate(len(initially_covered), len(accepted)),
        "true_row_retention_given_initial_coverage": rate(
            sum(bool(row["true_row_retained"]) for row in initially_covered),
            len(initially_covered),
        ),
        "true_row_elimination_given_initial_coverage": rate(
            sum(not bool(row["true_row_retained"]) for row in initially_covered),
            len(initially_covered),
        ),
        "false_confirmation_when_truth_initially_present": rate(
            sum(bool(row["false_confirmation"]) for row in initially_covered),
            len(initially_covered),
        ),
        "false_confirmation_when_truth_initially_absent": rate(
            sum(bool(row["false_confirmation"]) for row in initially_absent),
            len(initially_absent),
        ),
        "candidate_reduction_with_truth_preserved": rate(
            sum(
                bool(row["candidate_reduction"]) and bool(row["true_row_retained"])
                for row in accepted
            ),
            len(accepted),
        ),
        "candidate_reduction_with_truth_removed": rate(
            sum(
                bool(row["candidate_reduction"]) and not bool(row["true_row_retained"])
                for row in accepted
            ),
            len(accepted),
        ),
        "unresolved_rate": rate(len(unresolved), len(items)),
        "transition_signature_collision_rate": rate(
            sum(
                row["runtime_adjudication_status"] == "transition_signature_collision"
                for row in items
            ),
            len(items),
        ),
        "status_counts": dict(
            Counter(str(row["runtime_adjudication_status"]) for row in items)
        ),
    }


def breakdown(rows: Iterable[Mapping[str, object]], key: str) -> dict[str, object]:
    buckets: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(row)
    return {name: summarize(values) for name, values in sorted(buckets.items())}
