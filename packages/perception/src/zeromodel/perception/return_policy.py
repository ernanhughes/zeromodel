"""Bounded Return policy: compiled recovery after authority failure.

When the ordinary coupled path cannot safely commit (`accepted` false),
the Observer addresses a tiny compiled Return VPM that selects one
recovery operation (REOBSERVE / FALLBACK / STOP) from a finite,
hand-scored situation vocabulary. Terminal disposition (ABSTAIN vs
ESCALATE) is a deterministic reading of trajectory path history, not a
second policy: STOP with an empty path reports ABSTAIN (no path was
available); STOP with a used path reports ESCALATE (attempted and
failed).

The Return policy is deliberately terminal in the meta-policy
hierarchy: tiny, frozen, ungated. Its mistakes surface as return harm;
no authority layer watches the authority layer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Final, Mapping

from zeromodel.core.artifact import LayoutRecipe, ScoreTable, build_vpm
from zeromodel.core.policy_lookup import VPMPolicyLookup

from .inference import (
    BaselineNearestNeighborModelDTO,
    BaselinePredictionDTO,
    predict_baseline_action,
)
from .memory_authority import MemoryAuthorityContextDTO
from .representation import SourceVPMDTO
from .world_action import (
    CoupledActionPredictionDTO,
    CoupledLoopInputsDTO,
    predict_action_with_future_memory,
)

RETURN_POLICY_VERSION: Final = "perception-return-policy/1"
RETURN_DECISION_VERSION: Final = "perception-return-decision/1"
RETURN_TRAJECTORY_VERSION: Final = "perception-return-trajectory/1"
RETURN_OPERATIONS: Final = ("REOBSERVE", "FALLBACK", "STOP")
RETURN_TRIGGERS: Final = (
    "PREDICTOR_REJECTED",
    "OOD",
    "CONTRADICTED",
    "INSUFFICIENT",
)
RETURN_DECIDED_BY: Final = ("PRIMARY", "REOBSERVE", "FALLBACK", "NONE")
RETURN_DISPOSITIONS: Final = ("ABSTAIN", "ESCALATE")

# Frozen scores: (trigger, reobserve, fallback) -> winning operation.
# Re-observation serves support/recognition failures only, never a
# confident contradiction; fallback serves contradiction and unsupported
# cases; STOP ends trajectories the table cannot recover.
_RETURN_SCORES: Final = {
    ("OOD", True, True): "REOBSERVE",
    ("OOD", True, False): "REOBSERVE",
    ("OOD", False, True): "FALLBACK",
    ("OOD", False, False): "STOP",
    ("CONTRADICTED", True, True): "FALLBACK",
    ("CONTRADICTED", True, False): "STOP",
    ("CONTRADICTED", False, True): "FALLBACK",
    ("CONTRADICTED", False, False): "STOP",
    ("INSUFFICIENT", True, True): "REOBSERVE",
    ("INSUFFICIENT", True, False): "REOBSERVE",
    ("INSUFFICIENT", False, True): "FALLBACK",
    ("INSUFFICIENT", False, False): "STOP",
    ("PREDICTOR_REJECTED", True, True): "REOBSERVE",
    ("PREDICTOR_REJECTED", True, False): "REOBSERVE",
    ("PREDICTOR_REJECTED", False, True): "FALLBACK",
    ("PREDICTOR_REJECTED", False, False): "STOP",
}


class PerceptionReturnPolicyError(ValueError):
    """Raised when Return policy compilation or driving is ill-formed."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


def return_situation_row_id(
    trigger: str, reobserve_available: bool, fallback_decisive: bool
) -> str:
    """Deterministic situation address for the Return policy VPM."""
    if trigger not in RETURN_TRIGGERS:
        raise PerceptionReturnPolicyError(f"unknown Return trigger: {trigger!r}")
    reobserve = "yes" if reobserve_available else "no"
    fallback = "yes" if fallback_decisive else "no"
    return f"trigger={trigger}|reobserve={reobserve}|fallback={fallback}"


def compile_return_policy() -> tuple[object, VPMPolicyLookup]:
    """Compile the frozen 16-row Return policy into a VPM artifact + lookup."""
    row_ids: list[str] = []
    values: list[tuple[float, ...]] = []
    for trigger in ("PREDICTOR_REJECTED", "OOD", "CONTRADICTED", "INSUFFICIENT"):
        for reobserve in (True, False):
            for fallback in (True, False):
                row_ids.append(return_situation_row_id(trigger, reobserve, fallback))
                winner = _RETURN_SCORES[(trigger, reobserve, fallback)]
                values.append(
                    tuple(
                        1.0 if operation == winner else 0.0
                        for operation in RETURN_OPERATIONS
                    )
                )
    table = ScoreTable(
        values=values,
        row_ids=row_ids,
        metric_ids=list(RETURN_OPERATIONS),
        metadata={
            "kind": "compiled_return_policy",
            "world": "zeromodel-return",
            "addressing": "trigger,reobserve_available,fallback_decisive",
        },
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "return-policy-source-order",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    artifact = build_vpm(
        table,
        recipe,
        provenance={
            "kind": "compiled_policy",
            "consumer": "ReturnDriver",
            "compile_time_intelligence": "hand_scored_closed_world_return_policy",
        },
    )
    lookup = VPMPolicyLookup(artifact, action_metric_ids=list(RETURN_OPERATIONS))
    for row_id in row_ids:
        decision = lookup.read(row_id)
        trigger, reobserve, fallback = _parse_row_id(row_id)
        expected = _RETURN_SCORES[(trigger, reobserve, fallback)]
        if decision.action != expected:
            raise PerceptionReturnPolicyError(
                f"compiled Return row {row_id} selects {decision.action}, "
                f"expected {expected}"
            )
    return artifact, lookup


def _parse_row_id(row_id: str) -> tuple[str, bool, bool]:
    parts: dict[str, str] = {}
    for part in str(row_id).split("|"):
        key, value = part.split("=", 1)
        parts[key] = value
    return (
        parts["trigger"],
        parts["reobserve"] == "yes",
        parts["fallback"] == "yes",
    )


def classify_return_trigger(coupled: CoupledActionPredictionDTO) -> str:
    """Derive the Return trigger from a terminally failed coupled result."""
    if coupled.accepted:
        raise PerceptionReturnPolicyError(
            "Return requires terminal failure: coupled prediction was accepted"
        )
    if coupled.baseline.status != "accepted":
        return "PREDICTOR_REJECTED"
    rank_zero = coupled.candidates[0]
    if rank_zero.status == "contradicted_by_transition_expectation":
        return "CONTRADICTED"
    if rank_zero.status == "future_out_of_distribution":
        return "OOD"
    if rank_zero.status == "insufficient_future_evidence":
        return "INSUFFICIENT"
    raise PerceptionReturnPolicyError(
        f"untriggerable coupled failure: rank-0 status {rank_zero.status!r}"
    )


@dataclass(frozen=True)
class ReturnDecisionDTO:
    """Transient Observer decision recording one Return policy address."""

    decision_id: str
    source_vpm_id: str
    trigger: str
    trigger_authority: str
    trigger_prediction_id: str
    operation: str
    disposition: str | None
    return_policy_id: str
    recovery_memory_id: str | None
    observation_profile_id: str | None
    path: tuple[str, ...]
    decided_by: str
    budget_reobserve_spent: bool
    budget_fallback_spent: bool
    budget_trajectory_depth: int
    reasons: tuple[str, ...] = ()
    version: str = RETURN_DECISION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.decision_id,
                self.source_vpm_id,
                self.trigger_prediction_id,
                self.return_policy_id,
            )
        ):
            raise PerceptionReturnPolicyError(
                "return decision identities must be non-empty"
            )
        if self.trigger not in RETURN_TRIGGERS:
            raise PerceptionReturnPolicyError(
                f"unsupported return trigger: {self.trigger!r}"
            )
        if self.operation not in RETURN_OPERATIONS:
            raise PerceptionReturnPolicyError(
                f"unsupported return operation: {self.operation!r}"
            )
        if self.operation != "STOP" and self.disposition is not None:
            raise PerceptionReturnPolicyError(
                "non-terminal return operation must carry no disposition"
            )
        if self.operation == "STOP" and not self.path:
            if self.disposition != "ABSTAIN":
                raise PerceptionReturnPolicyError(
                    "STOP with an empty path must report ABSTAIN"
                )
        if self.operation == "STOP" and self.path:
            if self.disposition != "ESCALATE":
                raise PerceptionReturnPolicyError(
                    "STOP with a used path must report ESCALATE"
                )
        if self.decided_by not in ("PRIMARY", "REOBSERVE", "FALLBACK", "STOP"):
            raise PerceptionReturnPolicyError(
                f"unsupported return decided_by: {self.decided_by!r}"
            )
        if self.budget_trajectory_depth < 0:
            raise PerceptionReturnPolicyError("trajectory depth must be >= 0")
        if self.version != RETURN_DECISION_VERSION:
            raise PerceptionReturnPolicyError("unsupported return decision version")


@dataclass(frozen=True)
class ReturnTrajectoryDTO:
    """Terminal outcome of one bounded Return trajectory."""

    trajectory_id: str
    source_vpm_id: str
    return_policy_id: str
    decisions: tuple[ReturnDecisionDTO, ...]
    final_action: str | None
    accepted: bool
    decided_by: str
    disposition: str | None
    version: str = RETURN_TRAJECTORY_VERSION

    def __post_init__(self) -> None:
        if not all((self.trajectory_id, self.source_vpm_id, self.return_policy_id)):
            raise PerceptionReturnPolicyError(
                "return trajectory identities must be non-empty"
            )
        if self.decided_by not in RETURN_DECIDED_BY:
            raise PerceptionReturnPolicyError(
                f"unsupported trajectory decided_by: {self.decided_by!r}"
            )
        if self.accepted and (self.final_action is None or self.decided_by == "NONE"):
            raise PerceptionReturnPolicyError(
                "accepted trajectory requires an action and a deciding step"
            )
        if not self.accepted and self.final_action is not None:
            raise PerceptionReturnPolicyError("refused trajectory must carry no action")
        if self.accepted and self.disposition is not None:
            raise PerceptionReturnPolicyError(
                "committed trajectory carries no disposition"
            )
        if not self.accepted and self.disposition not in RETURN_DISPOSITIONS:
            raise PerceptionReturnPolicyError(
                "refused trajectory requires ABSTAIN or ESCALATE"
            )
        if self.version != RETURN_TRAJECTORY_VERSION:
            raise PerceptionReturnPolicyError("unsupported return trajectory version")


@dataclass(frozen=True)
class _DecisionContext:
    source_vpm_id: str
    trigger: str
    trigger_authority: str
    trigger_prediction_id: str
    return_policy_id: str


def _return_decision(
    context: _DecisionContext,
    *,
    operation: str,
    recovery_memory_id: str | None,
    observation_profile_id: str | None,
    path: tuple[str, ...],
    decided_by: str,
    reobserve_spent: bool,
    fallback_spent: bool,
    depth: int,
    reasons: tuple[str, ...] = (),
) -> ReturnDecisionDTO:
    if operation != "STOP":
        disposition = None
    else:
        disposition = "ESCALATE" if path else "ABSTAIN"
    payload = {
        "budget_fallback_spent": fallback_spent,
        "budget_reobserve_spent": reobserve_spent,
        "budget_trajectory_depth": depth,
        "decided_by": decided_by,
        "disposition": disposition,
        "observation_profile_id": observation_profile_id,
        "operation": operation,
        "path": list(path),
        "recovery_memory_id": recovery_memory_id,
        "return_policy_id": context.return_policy_id,
        "source_vpm_id": context.source_vpm_id,
        "trigger": context.trigger,
        "trigger_authority": context.trigger_authority,
        "trigger_prediction_id": context.trigger_prediction_id,
        "version": RETURN_DECISION_VERSION,
    }
    return ReturnDecisionDTO(
        decision_id=_digest(_canonical_json(payload)),
        source_vpm_id=context.source_vpm_id,
        trigger=context.trigger,
        trigger_authority=context.trigger_authority,
        trigger_prediction_id=context.trigger_prediction_id,
        operation=operation,
        disposition=disposition,
        return_policy_id=context.return_policy_id,
        recovery_memory_id=recovery_memory_id,
        observation_profile_id=observation_profile_id,
        path=path,
        decided_by=decided_by,
        budget_reobserve_spent=reobserve_spent,
        budget_fallback_spent=fallback_spent,
        budget_trajectory_depth=depth,
        reasons=reasons,
    )


def _rerun_coarse_loop(
    coarse: CoupledLoopInputsDTO,
    coarse_source: SourceVPMDTO,
    coarse_authority: MemoryAuthorityContextDTO | None,
) -> CoupledActionPredictionDTO:
    return predict_action_with_future_memory(
        coarse.predictor_model,
        coarse.transition_model,
        coarse_source,
        coarse.field_schema,
        declarations=coarse.declarations,
        policy=coarse.policy,
        field_weights=coarse.weights_dict(),
        baseline_override=coarse.override_for(coarse_source),
        authority=coarse_authority,
    )


def _refuse_trajectory(
    trajectory_id: str,
    source: SourceVPMDTO,
    return_artifact_id: str,
    decisions: list[ReturnDecisionDTO],
    path: list[str],
) -> ReturnTrajectoryDTO:
    disposition = "ESCALATE" if path else "ABSTAIN"
    return ReturnTrajectoryDTO(
        trajectory_id=trajectory_id,
        source_vpm_id=source.source_vpm_id,
        return_policy_id=return_artifact_id,
        decisions=tuple(decisions),
        final_action=None,
        accepted=False,
        decided_by="NONE",
        disposition=disposition,
    )


def _fallback_decisive(
    fallback_predictor: BaselineNearestNeighborModelDTO | None,
    source: SourceVPMDTO,
) -> tuple[bool, BaselinePredictionDTO | None]:
    if fallback_predictor is None:
        return False, None
    prediction = predict_baseline_action(fallback_predictor, source)
    return prediction.status == "accepted", prediction


def _open_trajectory(
    loop: CoupledLoopInputsDTO,
    source: SourceVPMDTO,
    authority: MemoryAuthorityContextDTO | None,
    return_artifact_id: str,
) -> tuple[str, CoupledActionPredictionDTO]:
    trajectory_id = _digest(
        _canonical_json(
            {
                "return_policy_id": return_artifact_id,
                "source_vpm_id": source.source_vpm_id,
                "version": RETURN_TRAJECTORY_VERSION,
            }
        )
    )
    coupled = predict_action_with_future_memory(
        loop.predictor_model,
        loop.transition_model,
        source,
        loop.field_schema,
        declarations=loop.declarations,
        policy=loop.policy,
        field_weights=loop.weights_dict(),
        baseline_override=loop.override_for(source),
        authority=authority,
    )
    return trajectory_id, coupled


def _take_reobserve(
    context: _DecisionContext,
    coarse: CoupledLoopInputsDTO,
    reobserve_source: Callable[[SourceVPMDTO], SourceVPMDTO],
    coarse_authority: MemoryAuthorityContextDTO | None,
    source: SourceVPMDTO,
    path: list[str],
    decisions: list[ReturnDecisionDTO],
    fallback_spent: bool,
) -> CoupledActionPredictionDTO:
    path.append("REOBSERVE")
    decisions.append(
        _return_decision(
            context,
            operation="REOBSERVE",
            recovery_memory_id=coarse.transition_model.model_id,
            observation_profile_id=coarse.profile_id,
            path=tuple(path),
            decided_by="REOBSERVE",
            reobserve_spent=True,
            fallback_spent=fallback_spent,
            depth=len(path),
            reasons=("re-observing under an alternative profile",),
        )
    )
    coarse_source = reobserve_source(source)
    return _rerun_coarse_loop(coarse, coarse_source, coarse_authority)


def _take_fallback(
    context: _DecisionContext,
    fallback_predictor: BaselineNearestNeighborModelDTO,
    fallback_prediction: BaselinePredictionDTO,
    path: list[str],
    decisions: list[ReturnDecisionDTO],
    reobserve_spent: bool,
) -> None:
    path.append("FALLBACK")
    decisions.append(
        _return_decision(
            context,
            operation="FALLBACK",
            recovery_memory_id=fallback_predictor.model_id,
            observation_profile_id=None,
            path=tuple(path),
            decided_by="FALLBACK",
            reobserve_spent=reobserve_spent,
            fallback_spent=True,
            depth=len(path),
            reasons=("addressing the identified fallback ranking",),
        )
    )


def _consult_operation(
    return_lookup: VPMPolicyLookup,
    trigger: str,
    reobserve_available: bool,
    fallback_ok: bool,
) -> str:
    operation = return_lookup.choose(
        return_situation_row_id(trigger, reobserve_available, fallback_ok)
    )
    # Budget is an execution invariant: scores never override it.
    if operation == "REOBSERVE" and not reobserve_available:
        return "STOP"
    if operation == "FALLBACK" and not fallback_ok:
        return "STOP"
    return operation


def _refuse_step(
    context: _DecisionContext,
    path: list[str],
    decisions: list[ReturnDecisionDTO],
    reobserve_spent: bool,
    fallback_spent: bool,
) -> None:
    decisions.append(
        _return_decision(
            context,
            operation="STOP",
            recovery_memory_id=None,
            observation_profile_id=None,
            path=tuple(path),
            decided_by="STOP",
            reobserve_spent=reobserve_spent,
            fallback_spent=fallback_spent,
            depth=len(path),
            reasons=("no further recovery operation authorized",),
        )
    )


def _commit_primary(
    trajectory_id: str,
    source: SourceVPMDTO,
    return_artifact_id: str,
    coupled: CoupledActionPredictionDTO,
) -> ReturnTrajectoryDTO | None:
    if not coupled.accepted:
        return None
    assert coupled.selected_action is not None
    return _commit_trajectory(
        trajectory_id,
        source,
        return_artifact_id,
        [],
        coupled.selected_action,
        "PRIMARY",
    )


def _failure_context(
    source: SourceVPMDTO,
    return_artifact_id: str,
    coupled: CoupledActionPredictionDTO,
) -> _DecisionContext:
    return _DecisionContext(
        source_vpm_id=source.source_vpm_id,
        trigger=classify_return_trigger(coupled),
        trigger_authority=_trigger_authority(coupled),
        trigger_prediction_id=coupled.prediction_id,
        return_policy_id=return_artifact_id,
    )


def decide_with_return(
    loop: CoupledLoopInputsDTO,
    source: SourceVPMDTO,
    *,
    authority: MemoryAuthorityContextDTO | None = None,
    return_lookup: VPMPolicyLookup,
    return_artifact_id: str,
    fallback_predictor: BaselineNearestNeighborModelDTO | None = None,
    coarse: CoupledLoopInputsDTO | None = None,
    coarse_authority: MemoryAuthorityContextDTO | None = None,
    reobserve_source: Callable[[SourceVPMDTO], SourceVPMDTO] | None = None,
) -> ReturnTrajectoryDTO:
    """Run the primary coupled loop, recovering through a bounded trajectory.

    Commits directly when the primary path authorizes. Otherwise consults
    the compiled Return VPM (at most one REOBSERVE and one FALLBACK;
    budget enforced independently of table scores, so even a malformed
    artifact cannot overspend recovery).
    """
    decisions: list[ReturnDecisionDTO] = []
    path: list[str] = []
    reobserve_spent = False
    fallback_spent = False
    trajectory_id, coupled = _open_trajectory(
        loop, source, authority, return_artifact_id
    )
    primary = _commit_primary(trajectory_id, source, return_artifact_id, coupled)
    if primary is not None:
        return primary
    trigger = classify_return_trigger(coupled)
    context = _failure_context(source, return_artifact_id, coupled)
    while True:
        reobserve_available = (
            reobserve_source is not None and coarse is not None and not reobserve_spent
        )
        fallback_ok, fallback_prediction = _fallback_decisive(
            fallback_predictor, source
        )
        operation = _consult_operation(
            return_lookup, trigger, reobserve_available, fallback_ok
        )
        if operation == "REOBSERVE":
            assert coarse is not None and reobserve_source is not None
            reobserve_spent = True
            coupled = _take_reobserve(
                context,
                coarse,
                reobserve_source,
                coarse_authority,
                source,
                path,
                decisions,
                fallback_spent,
            )
            if coupled.accepted:
                assert coupled.selected_action is not None
                return _commit_trajectory(
                    trajectory_id,
                    source,
                    return_artifact_id,
                    decisions,
                    coupled.selected_action,
                    "REOBSERVE",
                )
            trigger = classify_return_trigger(coupled)
            context = _failure_context(source, return_artifact_id, coupled)
            continue
        if operation == "FALLBACK":
            assert fallback_predictor is not None
            assert fallback_prediction is not None
            assert fallback_ok and fallback_prediction.status == "accepted"
            fallback_spent = True
            _take_fallback(
                context,
                fallback_predictor,
                fallback_prediction,
                path,
                decisions,
                reobserve_spent,
            )
            assert fallback_prediction.selected_action is not None
            return _commit_trajectory(
                trajectory_id,
                source,
                return_artifact_id,
                decisions,
                fallback_prediction.selected_action,
                "FALLBACK",
            )
        _refuse_step(context, path, decisions, reobserve_spent, fallback_spent)
        return _refuse_trajectory(
            trajectory_id, source, return_artifact_id, decisions, path
        )


def _trigger_authority(coupled: CoupledActionPredictionDTO) -> str:
    rank_zero = coupled.candidates[0]
    return rank_zero.memory_authority or "unknown"


def _commit_trajectory(
    trajectory_id: str,
    source: SourceVPMDTO,
    return_artifact_id: str,
    decisions: list[ReturnDecisionDTO],
    action: str,
    decided_by: str,
) -> ReturnTrajectoryDTO:
    return ReturnTrajectoryDTO(
        trajectory_id=trajectory_id,
        source_vpm_id=source.source_vpm_id,
        return_policy_id=return_artifact_id,
        decisions=tuple(decisions),
        final_action=action,
        accepted=True,
        decided_by=decided_by,
        disposition=None,
    )
