from __future__ import annotations

from zeromodel.perception import (
    INVALID_OBSERVATION_DECISION,
    REJECT_AMBIGUOUS_DECISION,
    FieldEvidence,
    StateSpecification,
    build_policy_compatibility_report,
    build_state_claim_set,
)


def _states() -> tuple[StateSpecification, ...]:
    return (
        StateSpecification(
            state_id="state-001",
            fields={
                "power": "green",
                "mode": "auto",
                "temperature": "normal",
                "door": "closed",
                "alarm": "inactive",
            },
            action_id="CONTINUE",
        ),
        StateSpecification(
            state_id="state-002",
            fields={
                "power": "green",
                "mode": "auto",
                "temperature": "normal",
                "door": "open",
                "alarm": "inactive",
            },
            action_id="CONTINUE",
        ),
        StateSpecification(
            state_id="state-011",
            fields={
                "power": "off",
                "mode": "maintenance",
                "temperature": "normal",
                "door": "closed",
                "alarm": "inactive",
            },
            action_id="WAIT",
        ),
        StateSpecification(
            state_id="state-012",
            fields={
                "power": "off",
                "mode": "auto",
                "temperature": "normal",
                "door": "open",
                "alarm": "inactive",
            },
            action_id="INSPECT",
        ),
    )


def _evidence(
    field_id: str,
    status: str,
    *,
    supported: tuple[str, ...] = (),
    contradicted: tuple[str, ...] = (),
    unresolved: tuple[str, ...] = (),
    reason: str = "",
) -> FieldEvidence:
    return FieldEvidence(
        field_id=field_id,
        status=status,
        supported_values=supported,
        contradicted_values=contradicted,
        unresolved_values=unresolved,
        source_region=f"{field_id}_region",
        observation_id="obs-001",
        compiler_id="fixture-compiler",
        reason=reason,
    )


def test_action_equivalent_ambiguity_executes_common_policy() -> None:
    evidence = (
        _evidence("power", "supported", supported=("green",)),
        _evidence("mode", "supported", supported=("auto",)),
        _evidence("temperature", "supported", supported=("normal",)),
        _evidence("door", "unresolved", unresolved=("closed", "open")),
        _evidence("alarm", "supported", supported=("inactive",)),
    )

    claim_set = build_state_claim_set("obs-001", _states(), evidence)
    report = build_policy_compatibility_report(claim_set, _states())

    assert claim_set.compatible_state_ids == ("state-001", "state-002")
    assert claim_set.unresolved_fields == ("door",)
    assert report.unanimous is True
    assert report.decision == "CONTINUE"


def test_action_changing_ambiguity_rejects_and_names_missing_field() -> None:
    evidence = (
        _evidence("power", "supported", supported=("off",)),
        _evidence("mode", "unresolved", unresolved=("auto", "maintenance")),
        _evidence("temperature", "supported", supported=("normal",)),
        _evidence("door", "unresolved", unresolved=("closed", "open")),
        _evidence("alarm", "supported", supported=("inactive",)),
    )

    claim_set = build_state_claim_set("obs-001", _states(), evidence)
    report = build_policy_compatibility_report(claim_set, _states())

    assert claim_set.compatible_state_ids == ("state-011", "state-012")
    assert claim_set.unresolved_fields == ("door", "mode")
    assert report.unanimous is False
    assert report.decision == REJECT_AMBIGUOUS_DECISION
    assert report.action_ids == ("INSPECT", "WAIT")
    assert report.conflicting_actions == {
        "INSPECT": ("state-012",),
        "WAIT": ("state-011",),
    }


def test_invalid_observation_short_circuits_state_claims() -> None:
    evidence = (
        _evidence(
            "panel",
            "invalid_observation",
            reason="panel_missing",
        ),
    )

    claim_set = build_state_claim_set("obs-001", _states(), evidence)
    report = build_policy_compatibility_report(claim_set, _states())

    assert claim_set.invalid_observation is True
    assert claim_set.compatible_state_ids == ()
    assert report.decision == INVALID_OBSERVATION_DECISION
