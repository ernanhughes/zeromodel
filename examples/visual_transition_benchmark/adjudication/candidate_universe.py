from __future__ import annotations

from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.video.arcade_policy import ACTIONS, compile_policy_artifact
from zeromodel.vision import VisualDecision


def action_for_row(row_id: str) -> str:
    lookup = VPMPolicyLookup(compile_policy_artifact(), action_metric_ids=ACTIONS)
    return str(lookup.read(row_id).action)


def reader_local_candidates(decision: VisualDecision) -> tuple[str, ...]:
    rows = {str(decision.nearest_row_id), str(decision.second_nearest_row_id)}
    if decision.matched_row_id is not None:
        rows.add(str(decision.matched_row_id))
    return tuple(sorted(rows))


def policy_action_candidates(action: str) -> tuple[str, ...]:
    policy = compile_policy_artifact()
    return tuple(
        sorted(
            row_id
            for row_id in policy.source.row_ids
            if action_for_row(str(row_id)) == action
        )
    )


def candidate_rows(mode: str, decision: VisualDecision) -> tuple[str, ...]:
    if not decision.policy_executed or decision.action is None:
        return ()
    if mode == "reader_local":
        return reader_local_candidates(decision)
    if mode == "policy_action":
        return policy_action_candidates(str(decision.action))
    raise ValueError(f"unsupported candidate universe: {mode}")
