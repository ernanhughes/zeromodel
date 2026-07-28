import pytest

from zeromodel.observer import (
    ObserverComparisonRecipeDTO,
    ObserverFeatureComparisonDTO,
    ObserverFeatureDefinitionDTO,
    ObserverHiddenStateHypothesisDTO,
    ObserverHiddenStateHypothesisSetDTO,
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
    ObserverProposedChangeDTO,
    ObserverRepairConstraintDTO,
    ObserverRepairProposalError,
    propose_observer_repair,
    verify_observer_transition,
)


ROW_ID = "row:cooldown-sensitive"
CELL_ID = f"{ROW_ID}/action:move_right"


def observer_schema() -> ObserverObservationSchemaDTO:
    return ObserverObservationSchemaDTO.create(
        schema_name="stage-o3-repair",
        features=(
            ObserverFeatureDefinitionDTO.create(
                qualified_key="hidden.cooldown", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.action_effect", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.agent_x", value_type="int", required=True
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.next_action", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.target_x", value_type="int", required=True
            ),
        ),
    )


SCHEMA = observer_schema()


def observation(
    *,
    sequence_index: int = 1,
    visible: dict[str, object],
    hidden: dict[str, object] | None = None,
) -> ObserverObservationArtifactDTO:
    return ObserverObservationArtifactDTO.create(
        observation_schema=SCHEMA,
        visible_state_features=visible,
        hidden_state_uncertainty=hidden or {},
        provenance={"fixture": "stage-o2"},
        sequence_index=sequence_index,
    )


def cooldown_recipe() -> ObserverComparisonRecipeDTO:
    feature_comparisons = tuple(
        ObserverFeatureComparisonDTO.create(feature_key=key, mode="exact")
        for key in (
            "hidden.cooldown",
            "visible.action_effect",
            "visible.agent_x",
            "visible.target_x",
        )
    )
    return ObserverComparisonRecipeDTO.create(
        feature_comparisons=feature_comparisons,
        observable_feature_keys=("visible.agent_x", "visible.target_x"),
        action_effect_keys=("visible.action_effect",),
        hidden_state_keys=("hidden.cooldown",),
        wake_on_policy_consequence_mismatch=True,
    )


def hypothesis_set(*, possible: bool = True) -> ObserverHiddenStateHypothesisSetDTO:
    return ObserverHiddenStateHypothesisSetDTO.create(
        observation_schema_id=SCHEMA.schema_id,
        hypotheses=(
            ObserverHiddenStateHypothesisDTO.create(
                state_key="hidden.cooldown",
                state_value="clear",
                status="possible" if possible else "eliminated",
            ),
        ),
    )


def contradicted_verification():
    predicted = observation(
        visible={
            "agent_x": 5,
            "target_x": 9,
            "action_effect": "moved_right",
            "next_action": "move_right",
        },
        hidden={"cooldown": "clear"},
    )
    observed = observation(
        visible={
            "agent_x": 4,
            "target_x": 9,
            "action_effect": "blocked_by_cooldown",
            "next_action": "wait",
        },
        hidden={"cooldown": "active"},
    )
    return verify_observer_transition(
        recipe=cooldown_recipe(),
        predicted_observation=predicted,
        observed_observation=observed,
        policy_artifact_id="policy:A",
        state_before_id="state:before",
        action="move_right",
        affected_policy_row_id=ROW_ID,
        hidden_state_hypothesis_set=hypothesis_set(possible=True),
        reproduction={"episode_id": "episode:1", "step": 7},
        relevant_context_keys=("hidden.cooldown",),
    )


def confirmed_verification():
    predicted = observation(
        visible={
            "agent_x": 5,
            "target_x": 9,
            "action_effect": "moved_right",
            "next_action": "move_right",
        },
        hidden={"cooldown": "clear"},
    )
    observed = observation(
        visible={
            "agent_x": 5,
            "target_x": 9,
            "action_effect": "moved_right",
            "next_action": "move_right",
        },
        hidden={"cooldown": "clear"},
    )
    return verify_observer_transition(
        recipe=cooldown_recipe(),
        predicted_observation=predicted,
        observed_observation=observed,
        policy_artifact_id="policy:A",
        state_before_id="state:before",
        action="move_right",
        affected_policy_row_id=ROW_ID,
        hidden_state_hypothesis_set=hypothesis_set(possible=True),
    )


def inconclusive_verification():
    predicted = observation(
        visible={"agent_x": 5, "target_x": 9, "action_effect": "moved_right"},
        hidden={"cooldown": "clear"},
    )
    observed = observation(
        visible={"agent_x": 5, "target_x": 9},
        hidden={"cooldown": "clear"},
    )
    return verify_observer_transition(
        recipe=cooldown_recipe(),
        predicted_observation=predicted,
        observed_observation=observed,
        policy_artifact_id="policy:A",
        state_before_id="state:before",
        action="move_right",
        affected_policy_row_id=ROW_ID,
        hidden_state_hypothesis_set=hypothesis_set(possible=True),
    )


def repair_constraint(
    *,
    allowed_rows: tuple[str, ...] = (ROW_ID,),
    allowed_cells: tuple[str, ...] = (CELL_ID,),
    forbidden_rows: tuple[str, ...] = (),
    max_rows: int = 1,
    max_cells: int = 1,
    allow_schema: bool = True,
) -> ObserverRepairConstraintDTO:
    return ObserverRepairConstraintDTO.create(
        allowed_row_ids=allowed_rows,
        allowed_cell_ids=allowed_cells,
        allowed_context_keys=("hidden.actuator_mode", "hidden.cooldown"),
        forbidden_row_ids=forbidden_rows,
        max_changed_rows=max_rows,
        max_changed_cells=max_cells,
        allow_action_value_change=True,
        allow_new_context_precondition=True,
        allow_schema_extension_request=allow_schema,
    )


def cooldown_change(
    *,
    target_id: str = CELL_ID,
    condition_keys: tuple[str, ...] = ("hidden.cooldown",),
    proposed_value: object = "wait",
    evidence_ids: tuple[str, ...] = (),
) -> ObserverProposedChangeDTO:
    return ObserverProposedChangeDTO.create(
        target_kind="policy_cell",
        target_id=target_id,
        operation="replace",
        field_name="action_value",
        old_value="move_right",
        proposed_value=proposed_value,
        condition_keys=condition_keys,
        evidence_ids=evidence_ids,
    )


def propose(
    *,
    verification=None,
    constraint=None,
    changes: tuple[ObserverProposedChangeDTO, ...] = (),
    represented_context_keys: tuple[str, ...] = (
        "hidden.cooldown",
        "visible.agent_x",
        "visible.target_x",
    ),
    missing_schema_keys: tuple[str, ...] = (),
    rationale_codes: tuple[str, ...] = (
        "action_effect_mismatch",
        "affected_row_localised",
        "repair_scope_bounded",
    ),
    evidence_ids: tuple[str, ...] = (),
):
    return propose_observer_repair(
        verification=verification or contradicted_verification(),
        constraint=constraint or repair_constraint(),
        available_policy_row_ids=(ROW_ID, "row:other"),
        available_policy_cell_ids=(
            CELL_ID,
            f"{ROW_ID}/action:wait",
            f"{ROW_ID}/transition:move_right",
            "row:other/action:move_left",
        ),
        represented_context_keys=represented_context_keys,
        requested_changes=changes,
        missing_schema_keys=missing_schema_keys,
        rationale_codes=rationale_codes,
        evidence_ids=evidence_ids,
    )


def test_locally_repairable_cooldown_proposal_replays() -> None:
    verification = contradicted_verification()
    change = cooldown_change(
        evidence_ids=(verification.contradiction_artifact.contradiction_artifact_id,)
    )

    first = propose(verification=verification, changes=(change,))
    second = propose(verification=verification, changes=(change,))

    assert first.disposition == "repairable"
    assert first.source_policy_artifact_id == "policy:A"
    assert first.transition_verification_id == verification.verification_id
    assert first.contradiction_artifact_id == (
        verification.contradiction_artifact.contradiction_artifact_id
    )
    assert first.affected_row_ids == (ROW_ID,)
    assert first.affected_cell_ids == (CELL_ID,)
    assert first.required_context_keys == ("hidden.cooldown",)
    assert first.missing_schema_keys == ()
    assert first.requested_changes == (change,)
    assert len(first.proposed_changes) == 1
    assert not hasattr(first, "replacement_policy_artifact_id")
    assert first.repair_proposal_id == second.repair_proposal_id
    assert first.proposed_changes[0].change_id == second.proposed_changes[0].change_id


def test_requires_schema_extension_preserves_blocking_key() -> None:
    change = cooldown_change(condition_keys=("hidden.actuator_mode",))

    proposal = propose(
        changes=(change,),
        represented_context_keys=("hidden.cooldown", "visible.agent_x"),
        rationale_codes=(
            "action_effect_mismatch",
            "missing_required_context",
            "repair_scope_bounded",
        ),
    )

    assert proposal.disposition == "requires_schema_extension"
    assert proposal.missing_schema_keys == ("hidden.actuator_mode",)
    assert proposal.requested_changes == (change,)
    assert proposal.proposed_changes == ()
    assert not hasattr(proposal, "replacement_policy_artifact_id")


def test_insufficient_evidence_has_no_executable_changes() -> None:
    verification = contradicted_verification()

    proposal = propose(
        verification=verification,
        changes=(),
        rationale_codes=("affected_row_localised",),
    )

    assert proposal.disposition == "insufficient_evidence"
    assert proposal.requested_changes == ()
    assert proposal.proposed_changes == ()
    assert verification.contradiction_artifact.contradiction_artifact_id in (
        proposal.evidence_ids
    )
    assert not hasattr(proposal, "replacement_policy_artifact_id")


def test_confirmed_verification_is_rejected() -> None:
    with pytest.raises(ObserverRepairProposalError, match="contradicted"):
        propose(verification=confirmed_verification(), changes=(cooldown_change(),))


def test_inconclusive_verification_is_rejected() -> None:
    with pytest.raises(ObserverRepairProposalError, match="contradicted"):
        propose(verification=inconclusive_verification(), changes=(cooldown_change(),))


def test_scope_violation_outside_allowed_rows_is_rejected() -> None:
    change = cooldown_change(target_id="row:other/action:move_left")

    with pytest.raises(ObserverRepairProposalError, match="affected policy row"):
        propose(changes=(change,))


def test_row_limit_violation_is_rejected() -> None:
    constraint = repair_constraint(max_rows=0)

    with pytest.raises(ObserverRepairProposalError, match="max_changed_rows"):
        propose(constraint=constraint, changes=(cooldown_change(),))


def test_cell_limit_violation_is_rejected() -> None:
    other = ObserverProposedChangeDTO.create(
        target_kind="policy_cell",
        target_id=f"{ROW_ID}/transition:move_right",
        operation="replace",
        field_name="action_value",
        old_value="move_right",
        proposed_value="wait",
        condition_keys=("hidden.cooldown",),
    )
    constraint = repair_constraint(
        allowed_cells=(CELL_ID, f"{ROW_ID}/transition:move_right"),
        max_rows=1,
        max_cells=1,
    )

    with pytest.raises(ObserverRepairProposalError, match="max_changed_cells"):
        propose(constraint=constraint, changes=(cooldown_change(), other))


def test_forbidden_region_is_rejected() -> None:
    with pytest.raises(ObserverRepairProposalError, match="forbidden"):
        propose(
            constraint=repair_constraint(forbidden_rows=(ROW_ID,)),
            changes=(cooldown_change(),),
        )


def test_unsupported_repair_type_is_reported_without_changes() -> None:
    proposal = propose(
        changes=(),
        rationale_codes=("unsupported_repair_type",),
    )

    assert proposal.disposition == "unsupported"
    assert proposal.requested_changes == ()
    assert proposal.proposed_changes == ()


def test_unsupported_change_operation_is_rejected() -> None:
    with pytest.raises(ObserverRepairProposalError, match="unsupported"):
        ObserverProposedChangeDTO.create(
            target_kind="policy_cell",
            target_id=CELL_ID,
            operation="merge",
            field_name="action_value",
        )


def test_missing_namespace_is_rejected() -> None:
    with pytest.raises(ObserverRepairProposalError, match="non-namespaced"):
        cooldown_change(condition_keys=("cooldown",))


@pytest.mark.parametrize(
    "make_proposal",
    [
        lambda: propose(
            constraint=repair_constraint(max_cells=2),
            changes=(cooldown_change(),),
        ),
        lambda: propose(
            changes=(
                ObserverProposedChangeDTO.create(
                    target_kind="policy_row",
                    target_id=ROW_ID,
                    operation="suspend",
                    field_name="row_status",
                    old_value="active",
                    proposed_value="suspended",
                ),
            ),
        ),
        lambda: propose(
            constraint=repair_constraint(
                allowed_cells=(CELL_ID, f"{ROW_ID}/action:wait"),
                max_cells=2,
            ),
            changes=(cooldown_change(target_id=f"{ROW_ID}/action:wait"),),
        ),
        lambda: propose(changes=(cooldown_change(proposed_value="move_left"),)),
        lambda: propose(
            changes=(cooldown_change(condition_keys=("hidden.actuator_mode",)),),
            represented_context_keys=("hidden.cooldown",),
        ),
        lambda: propose(
            changes=(cooldown_change(),),
            rationale_codes=("action_effect_mismatch", "repair_scope_bounded"),
        ),
        lambda: propose(
            changes=(cooldown_change(),),
            evidence_ids=("manual:evidence",),
        ),
    ],
)
def test_canonical_identity_changes_on_relevant_mutation(make_proposal) -> None:
    baseline = propose(changes=(cooldown_change(),))

    mutated = make_proposal()

    assert mutated.repair_proposal_id != baseline.repair_proposal_id


def test_replay_identity_canonicalizes_change_order() -> None:
    first = cooldown_change(evidence_ids=("evidence:a",))
    second = ObserverProposedChangeDTO.create(
        target_kind="context_precondition",
        target_id=ROW_ID,
        operation="add_precondition",
        field_name="context_precondition",
        proposed_value="hidden.cooldown == active",
        condition_keys=("hidden.cooldown",),
        evidence_ids=("evidence:b",),
    )
    constraint = repair_constraint(max_rows=1, max_cells=1)

    left = propose(constraint=constraint, changes=(first, second))
    right = propose(constraint=constraint, changes=(second, first))

    assert left.repair_constraint_id == right.repair_constraint_id
    assert tuple(change.change_id for change in left.proposed_changes) == tuple(
        change.change_id for change in right.proposed_changes
    )
    assert tuple(change.change_id for change in left.requested_changes) == tuple(
        change.change_id for change in right.requested_changes
    )
    assert left.repair_proposal_id == right.repair_proposal_id


def test_repair_public_api() -> None:
    import zeromodel.observer as observer

    for name in (
        "ObserverRepairConstraintDTO",
        "ObserverProposedChangeDTO",
        "ObserverRepairProposalDTO",
        "ObserverRepairProposalError",
        "propose_observer_repair",
    ):
        assert name in observer.__all__
        assert hasattr(observer, name)
