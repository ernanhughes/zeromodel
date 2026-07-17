from __future__ import annotations

from zeromodel import (
    LayoutRecipe,
    PolicyPropertyChecker,
    PolicyPropertySpec,
    ScoreTable,
    VPMPolicyLookup,
    build_vpm,
    from_bundle,
    to_bundle,
    with_q_diagnostics,
)

ACTIONS = ("LEFT", "RIGHT", "STAY", "FIRE")

FIRE_REQUIRES_ALIGNMENT = PolicyPropertySpec.from_dict(
    {
        "id": "fire_requires_alignment",
        "version": "1",
        "description": "FIRE wins only when aligned and ready.",
        "assert": {
            "implies": [
                {"eq": [{"var": "winner"}, "FIRE"]},
                {
                    "all": [
                        {
                            "eq": [
                                {"var": "state.tank"},
                                {"var": "state.target"},
                            ]
                        },
                        {"eq": [{"var": "state.cooldown"}, 0]},
                    ]
                },
            ]
        },
    }
)


def _artifact(*, unsafe: bool = False, parent_id: str | None = None):
    rows = [
        "tank=0|target=0|cooldown=0",
        "tank=0|target=1|cooldown=0",
        "tank=0|target=none|cooldown=0",
    ]
    values = [
        [0.0, 0.0, 0.0, 5.0],
        [0.0, 2.0, 0.5, 4.0 if unsafe else -1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    table = with_q_diagnostics(
        ScoreTable(values, rows, ACTIONS, {"kind": "q_policy"}),
        action_metric_ids=ACTIONS,
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "source",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    parents = (
        []
        if parent_id is None
        else [{"artifact_id": parent_id, "relation": "repaired_from"}]
    )
    return build_vpm(
        table,
        recipe,
        provenance={"kind": "compiled_policy", "parents": parents},
    )


def _checker(artifact):
    return PolicyPropertyChecker(
        artifact,
        action_metric_ids=ACTIONS,
        evidence_metric_ids=("criticality", "decision_margin"),
    )


def test_property_checker_passes_and_builds_linked_deterministic_artifact(
    tmp_path,
) -> None:
    policy = _artifact()
    report = _checker(policy).check([FIRE_REQUIRES_ALIGNMENT])

    assert report.passed is True
    assert report.results[0].rows_checked == 3
    assert report.results[0].violations == ()

    first = report.to_vpm()
    second = report.to_vpm()

    assert first.artifact_id == second.artifact_id
    assert first.provenance["parents"] == (
        {
            "artifact_id": policy.artifact_id,
            "relation": "verifies",
        },
    )

    loaded = from_bundle(to_bundle(first, tmp_path / "verification.vpm"))
    assert loaded.artifact_id == first.artifact_id


def test_counterexample_is_localized_then_repair_passes() -> None:
    safe = _artifact()
    unsafe = _artifact(unsafe=True, parent_id=safe.artifact_id)

    failed = _checker(unsafe).check([FIRE_REQUIRES_ALIGNMENT])

    assert failed.passed is False
    violation = failed.results[0].violations[0]
    assert violation.row_id == "tank=0|target=1|cooldown=0"
    assert violation.action == "FIRE"
    assert violation.source_metric_index == ACTIONS.index("FIRE")

    failed_artifact = failed.to_vpm()
    repaired = _artifact(unsafe=False, parent_id=unsafe.artifact_id)
    passed = _checker(repaired).check([FIRE_REQUIRES_ALIGNMENT])
    passed_artifact = passed.to_vpm()

    assert safe.artifact_id != unsafe.artifact_id
    assert unsafe.artifact_id != repaired.artifact_id
    assert failed_artifact.artifact_id != passed_artifact.artifact_id
    assert passed.passed is True


def test_evidence_columns_never_participate_in_action_selection() -> None:
    artifact = _artifact()
    decision = VPMPolicyLookup(
        artifact,
        action_metric_ids=ACTIONS,
        evidence_metric_ids=("criticality", "decision_margin"),
    ).read("tank=0|target=1|cooldown=0")

    assert decision.action == "RIGHT"
    assert set(decision.candidates) == set(ACTIONS)
    assert set(decision.evidence) == {"criticality", "decision_margin"}
