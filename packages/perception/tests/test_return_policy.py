"""Unit tests for the compiled Return policy and bounded driver."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from zeromodel.perception import (
    build_dataset_manifest,
    build_grid_field_schema,
    encode_discrete_action,
    encode_source_array,
    fit_baseline_nearest_neighbor,
)
from zeromodel.perception.dataset import RecordedInteractionDTO
from zeromodel.perception.representation import (
    DiscreteActionSchemaDTO,
    SourceImageEncoderSpecDTO,
)
from zeromodel.perception.return_policy import (
    RETURN_TRIGGERS,
    PerceptionReturnPolicyError,
    ReturnDecisionDTO,
    classify_return_trigger,
    compile_return_policy,
    decide_with_return,
    return_situation_row_id,
)
from zeromodel.perception.transition_model import (
    fit_action_conditioned_transition_model,
)
from zeromodel.perception.world_action import (
    CoupledActionPredictionDTO,
    CoupledLoopInputsDTO,
    DeclarationScopeDTO,
    WorldActionCandidateDTO,
    WorldActionPolicyDTO,
    predict_action_with_future_memory,
)

_SPEC = SourceImageEncoderSpecDTO(color_space="L")
_WIDTH, _HEIGHT = 12, 8
_ACTION_SCHEMA = DiscreteActionSchemaDTO.from_labels(["left", "right"])

EXPECTED_WINNERS = {
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


def _pattern(marker: int, action: str):
    before_array = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    before_array[marker % _HEIGHT, (marker * 5) % _WIDTH] = 200
    after_array = before_array.copy()
    if action == "left":
        after_array[:, 0:4] = 60
    else:
        after_array[:, 8:12] = 60
    return encode_source_array(before_array, _SPEC), encode_source_array(
        after_array, _SPEC
    )


def _build(count_per_action: int = 6):
    interactions = []
    sources = {}
    step = 0
    marker = 0
    for action in ("left", "right"):
        for _ in range(count_per_action):
            before, after = _pattern(marker, action)
            sources[before.source_vpm_id] = before
            sources[after.source_vpm_id] = after
            interactions.append(
                RecordedInteractionDTO.from_vpms(
                    sequence_id=f"seq-{action}",
                    step_index=step,
                    source=before,
                    target=encode_discrete_action(action, _ACTION_SCHEMA),
                    next_source=after,
                )
            )
            step += 1
            marker += 1
    manifest = build_dataset_manifest(
        interactions, source_encoder_spec_ids=[_SPEC.encoder_spec_id]
    )
    schema = build_grid_field_schema(
        next(iter(sources.values())), tile_width=4, tile_height=4, channel_mode="joint"
    )
    return manifest, sources, schema


def _loop(count_per_action: int = 6):
    manifest, sources, schema = _build(count_per_action)
    predictor = fit_baseline_nearest_neighbor(manifest, sources, training_split="all")
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    scope = DeclarationScopeDTO.create({}, ())
    policy = WorldActionPolicyDTO()
    loop = CoupledLoopInputsDTO(
        profile_id="native-L",
        predictor_model=predictor,
        transition_model=model,
        field_schema=schema,
        declarations=scope,
        policy=policy,
    )
    query = next(
        sources[interaction.source_vpm_id]
        for interaction in manifest.interactions
        if interaction.action_label == "left"
    )
    artifact, lookup = compile_return_policy()
    artifact_id = artifact.artifact_id
    return manifest, sources, schema, predictor, model, query, loop, lookup, artifact_id


def test_return_table_selects_frozen_winners() -> None:
    _, lookup = compile_return_policy()
    assert set(lookup.action_metric_ids) == {"REOBSERVE", "FALLBACK", "STOP"}
    for (trigger, reobserve, fallback), winner in EXPECTED_WINNERS.items():
        row_id = return_situation_row_id(trigger, reobserve, fallback)
        assert lookup.choose(row_id) == winner
    assert RETURN_TRIGGERS == (
        "PREDICTOR_REJECTED",
        "OOD",
        "CONTRADICTED",
        "INSUFFICIENT",
    )
    with pytest.raises(PerceptionReturnPolicyError):
        return_situation_row_id("BOGUS", True, True)


def test_compiled_policy_is_deterministic_and_identified() -> None:
    first_artifact, _ = compile_return_policy()
    second_artifact, _ = compile_return_policy()
    assert first_artifact.artifact_id == second_artifact.artifact_id


def test_primary_commit_needs_no_return() -> None:
    _, _, _, _, _, query, loop, lookup, artifact_id = _loop()
    trajectory = decide_with_return(
        loop,
        query,
        return_lookup=lookup,
        return_artifact_id=artifact_id,
    )
    assert trajectory.accepted
    assert trajectory.decided_by == "PRIMARY"
    assert trajectory.disposition is None
    assert trajectory.decisions == ()
    assert trajectory.final_action == "left"


def test_trigger_requires_terminal_failure() -> None:
    _, _, schema, predictor, model, query, _, _, _ = _loop()
    out = predict_action_with_future_memory(predictor, model, query, schema)
    assert out.accepted
    with pytest.raises(PerceptionReturnPolicyError, match="terminal failure"):
        classify_return_trigger(out)


def test_trigger_maps_rank_zero_fate() -> None:
    _, _, _, _, _, query, _, _, _ = _loop()
    _, _, _, predictor, _, _, _, _, _ = _loop()
    from zeromodel.perception import predict_baseline_action as _predict

    baseline = _predict(predictor, query)

    def _coupled_with(rank_zero_status: str) -> CoupledActionPredictionDTO:
        candidate = WorldActionCandidateDTO(
            action_label="left",
            base_score=0.9,
            base_rank=0,
            expected_transition_id="expected",
            transition_support=0.5,
            transition_confidence=0.5,
            expectation_conformance="conformant",
            status=rank_zero_status,
            reasons=("probe",),
            memory_authority="SUPPORT_ONLY",
        )
        return CoupledActionPredictionDTO(
            prediction_id="prediction",
            source_vpm_id=query.source_vpm_id,
            baseline=baseline,
            candidates=(candidate,),
            selected_action=None,
            accepted=False,
            policy_id="policy",
            baseline_model_id=predictor.model_id,
            transition_model_id="transition",
        )

    assert (
        classify_return_trigger(_coupled_with("contradicted_by_transition_expectation"))
        == "CONTRADICTED"
    )
    assert classify_return_trigger(_coupled_with("future_out_of_distribution")) == "OOD"
    assert (
        classify_return_trigger(_coupled_with("insufficient_future_evidence"))
        == "INSUFFICIENT"
    )
    rejected = dataclasses.replace(baseline, status="rejected_ambiguous")
    probe = _coupled_with("insufficient_future_evidence")
    probe_rejected = dataclasses.replace(probe, baseline=rejected)
    assert classify_return_trigger(probe_rejected) == "PREDICTOR_REJECTED"


def test_abstain_reports_without_recovery_path() -> None:
    _, _, _, _, _, query, loop, lookup, artifact_id = _loop()
    strict = dataclasses.replace(
        loop, policy=WorldActionPolicyDTO(min_support=500, reject_on_insufficient=True)
    )
    trajectory = decide_with_return(
        strict,
        query,
        return_lookup=lookup,
        return_artifact_id=artifact_id,
    )
    assert not trajectory.accepted
    assert trajectory.final_action is None
    assert trajectory.decided_by == "NONE"
    assert trajectory.disposition == "ABSTAIN"
    assert len(trajectory.decisions) == 1
    assert trajectory.decisions[0].operation == "STOP"
    assert trajectory.decisions[0].path == ()


def test_fallback_commits_identified_ranking() -> None:
    _, _, _, predictor, _, query, loop, lookup, artifact_id = _loop()
    strict = dataclasses.replace(
        loop, policy=WorldActionPolicyDTO(min_support=500, reject_on_insufficient=True)
    )
    trajectory = decide_with_return(
        strict,
        query,
        return_lookup=lookup,
        return_artifact_id=artifact_id,
        fallback_predictor=predictor,
    )
    assert trajectory.accepted
    assert trajectory.decided_by == "FALLBACK"
    assert trajectory.disposition is None
    assert trajectory.decisions[-1].operation == "FALLBACK"


def test_fallback_rejection_escalates() -> None:
    _, _, _, predictor, _, _, loop, lookup, artifact_id = _loop()
    strict = dataclasses.replace(
        loop, policy=WorldActionPolicyDTO(min_support=500, reject_on_insufficient=True)
    )
    far = encode_source_array(np.full((_HEIGHT, _WIDTH), 255, dtype=np.uint8), _SPEC)
    trajectory = decide_with_return(
        strict,
        far,
        return_lookup=lookup,
        return_artifact_id=artifact_id,
        fallback_predictor=predictor,
        coarse=dataclasses.replace(loop, profile_id="coarse-L"),
        reobserve_source=lambda source: source,
    )
    assert not trajectory.accepted
    assert trajectory.disposition == "ESCALATE"
    operations = [item.operation for item in trajectory.decisions]
    assert operations[0] == "REOBSERVE"
    assert operations[-1] == "STOP"


def test_budget_survives_malformed_scores() -> None:
    from zeromodel.core.artifact import LayoutRecipe, ScoreTable, build_vpm
    from zeromodel.core.policy_lookup import VPMPolicyLookup

    _, _, _, _, _, query, loop, _, artifact_id = _loop()
    strict = dataclasses.replace(
        loop, policy=WorldActionPolicyDTO(min_support=500, reject_on_insufficient=True)
    )
    row_ids = [
        f"trigger={trigger}|reobserve={reobserve}|fallback={fallback}"
        for trigger in RETURN_TRIGGERS
        for reobserve in ("yes", "no")
        for fallback in ("yes", "no")
    ]
    table = ScoreTable(
        values=[(1.0, 0.0, 0.0)] * len(row_ids),
        row_ids=row_ids,
        metric_ids=["REOBSERVE", "FALLBACK", "STOP"],
        metadata={"kind": "malformed-return-policy"},
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "malformed",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    malformed = VPMPolicyLookup(
        build_vpm(table, recipe, provenance={"kind": "malformed"}),
        action_metric_ids=["REOBSERVE", "FALLBACK", "STOP"],
    )
    calls = {"count": 0}

    def _counting(source):
        calls["count"] += 1
        return source

    trajectory = decide_with_return(
        strict,
        query,
        return_lookup=malformed,
        return_artifact_id=artifact_id,
        coarse=dataclasses.replace(strict, profile_id="coarse-L"),
        reobserve_source=_counting,
    )
    assert calls["count"] == 1
    assert trajectory.disposition == "ESCALATE"
    assert [item.operation for item in trajectory.decisions] == [
        "REOBSERVE",
        "STOP",
    ]


def test_disposition_invariant_is_pinned() -> None:
    _, _, _, _, _, _, _, _, _ = _loop()
    base = {
        "source_vpm_id": "source",
        "trigger": "OOD",
        "trigger_authority": "SUPPORT_ONLY",
        "trigger_prediction_id": "prediction",
        "return_policy_id": "policy",
        "recovery_memory_id": None,
        "observation_profile_id": None,
        "reasons": (),
    }
    live = ReturnDecisionDTO(
        decision_id="one",
        operation="REOBSERVE",
        disposition=None,
        path=("REOBSERVE",),
        decided_by="REOBSERVE",
        budget_reobserve_spent=True,
        budget_fallback_spent=False,
        budget_trajectory_depth=1,
        **base,  # type: ignore[arg-type]
    )
    assert live.disposition is None
    with pytest.raises(PerceptionReturnPolicyError):
        ReturnDecisionDTO(
            decision_id="bad",
            operation="REOBSERVE",
            disposition="ABSTAIN",
            path=("REOBSERVE",),
            decided_by="REOBSERVE",
            budget_reobserve_spent=True,
            budget_fallback_spent=False,
            budget_trajectory_depth=1,
            **base,  # type: ignore[arg-type]
        )
    abstain = ReturnDecisionDTO(
        decision_id="two",
        operation="STOP",
        disposition="ABSTAIN",
        path=(),
        decided_by="STOP",
        budget_reobserve_spent=False,
        budget_fallback_spent=False,
        budget_trajectory_depth=0,
        **base,  # type: ignore[arg-type]
    )
    assert abstain.disposition == "ABSTAIN"
    with pytest.raises(PerceptionReturnPolicyError):
        ReturnDecisionDTO(
            decision_id="three",
            operation="STOP",
            disposition="ABSTAIN",
            path=("REOBSERVE",),
            decided_by="STOP",
            budget_reobserve_spent=True,
            budget_fallback_spent=False,
            budget_trajectory_depth=1,
            **base,  # type: ignore[arg-type]
        )
    escalate = ReturnDecisionDTO(
        decision_id="four",
        operation="STOP",
        disposition="ESCALATE",
        path=("REOBSERVE",),
        decided_by="STOP",
        budget_reobserve_spent=True,
        budget_fallback_spent=False,
        budget_trajectory_depth=1,
        **base,  # type: ignore[arg-type]
    )
    assert escalate.disposition == "ESCALATE"
    with pytest.raises(PerceptionReturnPolicyError):
        ReturnDecisionDTO(
            decision_id="five",
            operation="STOP",
            disposition="ESCALATE",
            path=(),
            decided_by="STOP",
            budget_reobserve_spent=False,
            budget_fallback_spent=False,
            budget_trajectory_depth=0,
            **base,  # type: ignore[arg-type]
        )
