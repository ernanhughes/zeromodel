"""Tests for the provider-evaluation report adapter (composition seam in
`examples/`).

The report compiler requires a dense subject x dimension matrix
(`missing_value_semantics="error"` - its `"absent"` sparse path is not
implemented). So the adapter fills every cell for every case, including
cells that do not apply to a given case (a predicted-state factor on a
rejected case, or a factor key absent from that case's `expected_state`).
These tests pin down the exact convention the adapter must follow for those
cells - `raw_value=0.0, importance=0.0`, tagged `applicable=false` on the
source binding - and that it is never confused with a measured `False`
(`raw_value=0.0, importance=1.0`) or a measured `True`
(`raw_value=1.0, importance=1.0`).
"""

from __future__ import annotations

from typing import Any

from examples.provider_evaluation_report_adapter import (
    ProviderEvaluationReportAdapter,
    compile_provider_evaluation_report,
)
from test_video_provider_evaluation_rmdto import (
    POLICY_ARTIFACT_ID,
    sample_case,
    sample_configuration,
)

from zeromodel.artifacts import AdaptedValueDTO, InMemoryArtifactStore
from zeromodel.artifacts.adapted_report_persistence import load_adapted_report
from zeromodel.core.artifact import LayoutRecipe
from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
    CASE_OUTCOME_ACTION_CHANGING,
    CASE_OUTCOME_ACTION_EQUIVALENT,
    CASE_OUTCOME_EXACT,
    CASE_OUTCOME_REJECTED,
    ProviderEvaluationCaseContext,
    ProviderEvaluationCaseDTO,
    ProviderResponseEvidence,
    build_provider_evaluation_run,
)

_LAYOUT = LayoutRecipe(
    {
        "version": "vpm-layout/0",
        "row_order": {"kind": "source", "tie_break": "row_id"},
        "column_order": {"kind": "source"},
        "normalization": {"kind": "per_metric_minmax", "clip": True},
    }
)

# `sample_case`'s `expected_state` only ever carries `tank_column`/`cooldown`
# (see `test_video_provider_evaluation_rmdto.sample_case`), so
# `target_present_match`/`target_column_match` are always an inapplicable
# placeholder for every fixture case built here - useful for pinning down
# "absent factor key" behavior without a bespoke fixture.
_ALWAYS_INAPPLICABLE_FACTORS = ("target_present_match", "target_column_match")
_SOMETIMES_APPLICABLE_FACTORS = ("tank_column_match", "cooldown_match")


def _frame_ids(count: int) -> tuple[str, ...]:
    return tuple(f"development:x:frame-{i:02d}" for i in range(count))


def _imperfect_fixture_with_rejection() -> list[Any]:
    """3 exact / 1 action_equivalent / 1 action_changing / 1 rejected."""
    frame_ids = _frame_ids(5)
    configuration_id = sample_configuration().provider_configuration_id
    outcomes = [
        CASE_OUTCOME_EXACT,
        CASE_OUTCOME_EXACT,
        CASE_OUTCOME_EXACT,
        CASE_OUTCOME_ACTION_EQUIVALENT,
        CASE_OUTCOME_ACTION_CHANGING,
    ]
    cases = [
        sample_case(
            case_ordinal=index,
            frame_id=frame_id,
            provider_configuration_id=configuration_id,
            outcome=outcome,
            provider_latency_us=1000 + index * 10,
        )
        for index, (frame_id, outcome) in enumerate(zip(frame_ids, outcomes))
    ]
    # Add one rejected case last, at the next ordinal.
    cases.append(
        sample_case(
            case_ordinal=len(cases),
            frame_id=_frame_ids(len(cases) + 1)[-1],
            provider_configuration_id=configuration_id,
            outcome=CASE_OUTCOME_REJECTED,
        )
    )
    return cases


def _build_run():
    cases = _imperfect_fixture_with_rejection()
    return build_provider_evaluation_run(
        fixture_identity="report-adapter-test",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="unlabelled",
        cases=cases,
    )


def _values_by_case_and_dimension(
    values: tuple[AdaptedValueDTO, ...],
) -> dict[tuple[str, str], AdaptedValueDTO]:
    return {(value.subject_id, value.dimension_id): value for value in values}


def test_inapplicable_factor_cells_are_zero_raw_zero_importance() -> None:
    run = _build_run()
    adapted = ProviderEvaluationReportAdapter().adapt(run)
    by_cell = _values_by_case_and_dimension(adapted.values)

    for case in run.cases:
        for dimension_id in _ALWAYS_INAPPLICABLE_FACTORS:
            value = by_cell[(case.case_id, dimension_id)]
            assert value.raw_value == 0.0
            assert value.importance == 0.0
            assert value.source_binding.attributes == (
                ("applicable", "false"),
                ("placeholder", "true"),
            )


def test_applicable_factor_cells_use_full_importance() -> None:
    run = _build_run()
    adapted = ProviderEvaluationReportAdapter().adapt(run)
    by_cell = _values_by_case_and_dimension(adapted.values)

    for case in run.cases:
        if not case.accepted:
            continue
        for dimension_id in _SOMETIMES_APPLICABLE_FACTORS:
            value = by_cell[(case.case_id, dimension_id)]
            assert value.importance == 1.0
            assert value.raw_value in (0.0, 1.0)
            assert value.source_binding.attributes == (("applicable", "true"),)


def test_rejected_case_predicted_factor_cells_are_inapplicable_placeholders() -> None:
    run = _build_run()
    adapted = ProviderEvaluationReportAdapter().adapt(run)
    by_cell = _values_by_case_and_dimension(adapted.values)

    rejected_cases = [case for case in run.cases if not case.accepted]
    assert rejected_cases
    for case in rejected_cases:
        for dimension_id in _SOMETIMES_APPLICABLE_FACTORS:
            value = by_cell[(case.case_id, dimension_id)]
            assert value.raw_value == 0.0
            assert value.importance == 0.0
            assert value.source_binding.attributes == (
                ("applicable", "false"),
                ("placeholder", "true"),
            )


def test_confidence_absence_is_an_inapplicable_placeholder() -> None:
    # `sample_case` always supplies a confidence value; build a case
    # explicitly with no confidence at all to exercise the None path.
    configuration_id = sample_configuration().provider_configuration_id
    context = ProviderEvaluationCaseContext(
        policy_artifact_id=POLICY_ARTIFACT_ID,
        provider_configuration_id=configuration_id,
    )
    case_no_confidence = ProviderEvaluationCaseDTO.build(
        case_ordinal=0,
        frame_id=_frame_ids(1)[0],
        context=context,
        expected_state={"tank_column": 0, "cooldown": 0},
        expected_decision={
            "artifact_id": POLICY_ARTIFACT_ID,
            "row_id": "r0",
            "action": "STAY",
            "metric_id": "STAY",
            "value": 1.0,
            "source_row_index": 0,
            "source_metric_index": 0,
            "view_row": 0,
            "view_column": 0,
            "candidates": {"STAY": 1.0},
            "evidence": {},
        },
        accepted=False,
        evidence=ProviderResponseEvidence(
            rejection_reason="confidence_below_threshold"
        ),
    )
    run = build_provider_evaluation_run(
        fixture_identity="confidence-absent",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="unlabelled",
        cases=[case_no_confidence],
    )
    adapted = ProviderEvaluationReportAdapter().adapt(run)
    by_cell = _values_by_case_and_dimension(adapted.values)
    value = by_cell[(case_no_confidence.case_id, "confidence")]
    assert value.raw_value == 0.0
    assert value.importance == 0.0
    assert value.source_binding.attributes == (
        ("applicable", "false"),
        ("placeholder", "true"),
    )


def test_latency_absence_is_an_inapplicable_placeholder() -> None:
    configuration_id = sample_configuration().provider_configuration_id
    case = sample_case(
        case_ordinal=0,
        frame_id=_frame_ids(1)[0],
        provider_configuration_id=configuration_id,
        outcome=CASE_OUTCOME_EXACT,
        provider_latency_us=None,
    )
    run = build_provider_evaluation_run(
        fixture_identity="latency-absent",
        provider_configuration=sample_configuration(),
        policy_artifact_id=POLICY_ARTIFACT_ID,
        case_mode="smoke",
        representation_mode="unlabelled",
        cases=[case],
    )
    adapted = ProviderEvaluationReportAdapter().adapt(run)
    by_cell = _values_by_case_and_dimension(adapted.values)
    value = by_cell[(case.case_id, "latency")]
    assert value.raw_value == 0.0
    assert value.importance == 0.0
    assert value.source_binding.attributes == (
        ("applicable", "false"),
        ("placeholder", "true"),
    )


def test_compilation_and_reload_preserve_importance_values() -> None:
    run = _build_run()
    adapter = ProviderEvaluationReportAdapter()
    adapted_direct = adapter.adapt(run)

    store = InMemoryArtifactStore()
    compiled = compile_provider_evaluation_report(
        run, layout_recipe=_LAYOUT, store=store
    )
    reloaded = load_adapted_report(compiled.adapted_report_ref, resolver=store)

    direct_by_cell = _values_by_case_and_dimension(adapted_direct.values)
    reloaded_by_cell = _values_by_case_and_dimension(reloaded.values)
    assert direct_by_cell.keys() == reloaded_by_cell.keys()
    for key, direct_value in direct_by_cell.items():
        reloaded_value = reloaded_by_cell[key]
        assert reloaded_value.importance == direct_value.importance
        assert reloaded_value.raw_value == direct_value.raw_value
