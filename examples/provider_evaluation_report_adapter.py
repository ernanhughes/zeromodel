"""Report adapter: provider-evaluation run -> Artifacts report compilation.

This module is the composition seam between `zeromodel.video` (the provider
evaluation aggregate) and `zeromodel.artifacts` (report compilation and VPM
rendering). Neither production package may import the other
(`package-boundaries.toml`); `examples/` is exempt from that mechanical check
and is exactly where the architecture doc says a concrete `ReportAdapter`
belongs (see `writer_report_adapters_demo.py` for the same pattern in another
domain). It does not reimplement report persistence, VPM persistence, or
aggregate closure validation - it only translates one materialized provider
evaluation run into the neutral `AdaptedReportDTO` shape and then calls the
existing `compile_report`.

One row per evaluation case. Case order is priority-sorted so the most
consequential failures land first: action-changing errors, then rejections,
then accepted-but-not-exact cases, then exact cases.
"""

from __future__ import annotations

from dataclasses import dataclass

from zeromodel.artifacts import (
    AdaptedDimensionDTO,
    AdaptedReportDTO,
    AdaptedSubjectDTO,
    AdaptedValueDTO,
    ArtifactStore,
    CompiledReportArtifactDTO,
    ReportAdapterContractDTO,
    ReportFindingRefDTO,
    ScoreSemantics,
    SourceBindingDTO,
    compile_report,
)
from zeromodel.artifacts.report_dto import (
    compute_adapted_report_id,
    compute_report_adapter_contract_id,
)
from zeromodel.core.artifact import LayoutRecipe
from zeromodel.video.domains.video_action_set.provider_evaluation_dto import (
    CASE_OUTCOME_ACTION_CHANGING,
    CASE_OUTCOME_ACTION_EQUIVALENT,
    CASE_OUTCOME_REJECTED,
    MaterializedProviderEvaluationRunDTO,
    ProviderEvaluationCaseDTO,
)


ADAPTER_ID = "zeromodel.examples.provider_evaluation_report_adapter"
ADAPTER_VERSION = "provider-evaluation-report-adapter/v1"
REPORT_KIND = "provider_policy_impact_evaluation"
SUBJECT_KIND = "provider_observation_case"
DIMENSION_NAMESPACE = "provider_evaluation/v1"
COMPATIBILITY_ID = "provider-evaluation-report/v1"

# Dimension identifiers whose value is read generically from a case's
# `factor_matches` mapping rather than a dedicated DTO field. Declared here as
# the adapter's known factor set. The report compiler requires a dense
# subject x dimension matrix (`missing_value_semantics="error"` - its
# `"absent"` sparse path is not implemented, see
# `report_compiler._build_score_table`). So a factor key absent from a given
# case's `expected_state` is NOT omitted: it is filled with an explicit
# inapplicable placeholder (`raw_value=0.0, importance=0.0`, see `_values_for_case`).
# That zero is not a synthesized "false" - it must never be interpreted as
# measured negative evidence, only as "this cell does not apply here."
FACTOR_DIMENSIONS = {
    "tank_column_match": "tank_column",
    "target_present_match": "target_present",
    "target_column_match": "target_column",
    "cooldown_match": "cooldown",
}

DIMENSIONS = (
    AdaptedDimensionDTO(
        dimension_id="accepted",
        label="Accepted",
        score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        value_min=0.0,
        value_max=1.0,
    ),
    AdaptedDimensionDTO(
        dimension_id="exact_state_match",
        label="Exact State Match",
        score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        value_min=0.0,
        value_max=1.0,
    ),
    AdaptedDimensionDTO(
        dimension_id="action_match",
        label="Action Match",
        score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        value_min=0.0,
        value_max=1.0,
    ),
    AdaptedDimensionDTO(
        dimension_id="action_equivalent",
        label="Action Equivalent (state wrong, action still correct)",
        score_semantics=ScoreSemantics.DESCRIPTIVE,
        value_min=0.0,
        value_max=1.0,
    ),
    AdaptedDimensionDTO(
        dimension_id="action_changing",
        label="Action Changing (policy-boundary error)",
        score_semantics=ScoreSemantics.HIGHER_IS_WORSE,
        value_min=0.0,
        value_max=1.0,
    ),
    AdaptedDimensionDTO(
        dimension_id="tank_column_match",
        label="Tank Column Match",
        score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        value_min=0.0,
        value_max=1.0,
    ),
    AdaptedDimensionDTO(
        dimension_id="target_present_match",
        label="Target Present Match",
        score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        value_min=0.0,
        value_max=1.0,
    ),
    AdaptedDimensionDTO(
        dimension_id="target_column_match",
        label="Target Column Match",
        score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        value_min=0.0,
        value_max=1.0,
    ),
    AdaptedDimensionDTO(
        dimension_id="cooldown_match",
        label="Cooldown Match",
        score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        value_min=0.0,
        value_max=1.0,
    ),
    AdaptedDimensionDTO(
        dimension_id="confidence",
        label="Provider Confidence",
        score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        value_min=0.0,
        value_max=1.0,
    ),
    AdaptedDimensionDTO(
        dimension_id="latency",
        label="Provider Latency (microseconds)",
        score_semantics=ScoreSemantics.HIGHER_IS_WORSE,
        value_min=0.0,
    ),
)


def _priority_rank(case: ProviderEvaluationCaseDTO) -> int:
    if case.outcome == CASE_OUTCOME_ACTION_CHANGING:
        return 0
    if case.outcome == CASE_OUTCOME_REJECTED:
        return 1
    if case.outcome == CASE_OUTCOME_ACTION_EQUIVALENT:
        return 2
    return 3


def _values_for_case(
    case: ProviderEvaluationCaseDTO, *, run_id: str
) -> list[AdaptedValueDTO]:
    """Build every dimension's value for one case.

    The compiler currently requires a dense subject x dimension matrix
    (``missing_value_semantics="error"`` - its ``"absent"`` sparse path is
    not implemented yet, see ``report_compiler._build_score_table``). So
    every dimension always gets a value here; for a dimension that does not
    apply to this case (a predicted-state factor on a rejected case, or a
    factor key absent from this case's ``expected_state``), the cell is an
    explicit inapplicable placeholder: ``raw_value=0.0`` with
    ``importance=0.0``. That zero is never measured negative evidence and
    must not be interpreted independently of ``importance`` - it must not be
    described or read as "omitted". Applicable cells always carry
    ``importance=1.0`` (``raw_value=1.0`` for a true match, ``0.0`` for a
    measured false).

    Each cell's ``source_binding.attributes`` also carries an explicit
    ``("applicable", "true"|"false")`` tag (plus ``("placeholder", "true")``
    when inapplicable), so a consumer inspecting the compiled report's
    source bindings does not have to infer applicability from
    ``raw_value``/``importance`` alone.
    """
    values: list[AdaptedValueDTO] = []

    def add(dimension_id: str, raw_value: float | None) -> None:
        applicable = raw_value is not None
        finding = ReportFindingRefDTO(
            report_id=run_id, finding_id=f"{case.case_id}:{dimension_id}"
        )
        attributes = (
            (("applicable", "true"),)
            if applicable
            else (("applicable", "false"), ("placeholder", "true"))
        )
        binding = SourceBindingDTO(
            subject_id=case.case_id,
            dimension_id=dimension_id,
            finding_ref=finding,
            source_uri=case.frame_id,
            attributes=attributes,
        )
        values.append(
            AdaptedValueDTO(
                subject_id=case.case_id,
                dimension_id=dimension_id,
                raw_value=0.0 if raw_value is None else float(raw_value),
                importance=1.0 if applicable else 0.0,
                source_binding=binding,
            )
        )

    add("accepted", 1.0 if case.accepted else 0.0)
    add("exact_state_match", 1.0 if case.exact_state_match else 0.0)
    add("action_match", 1.0 if case.action_match else 0.0)
    add(
        "action_equivalent",
        1.0 if case.outcome == CASE_OUTCOME_ACTION_EQUIVALENT else 0.0,
    )
    add(
        "action_changing",
        1.0 if case.outcome == CASE_OUTCOME_ACTION_CHANGING else 0.0,
    )
    factor_matches = case.factor_matches.to_value()
    factor_matches = factor_matches if isinstance(factor_matches, dict) else {}
    for dimension_id, factor_key in FACTOR_DIMENSIONS.items():
        if factor_key in factor_matches:
            add(dimension_id, 1.0 if factor_matches[factor_key] else 0.0)
        else:
            add(dimension_id, None)
    # `case.provider_confidence` is the DTO's derived 0.0..1.0 presentation
    # property (recomputed from the canonical `provider_confidence_basis_points`
    # integer on every access) - never expose the raw basis-points integer
    # (0..10000) directly as a report score whose declared maximum is 1.0.
    add("confidence", case.provider_confidence)
    add(
        "latency",
        None if case.provider_latency_us is None else float(case.provider_latency_us),
    )
    return values


@dataclass(frozen=True, slots=True)
class ProviderEvaluationReportAdapter:
    """Translates one `MaterializedProviderEvaluationRunDTO` into a neutral report."""

    def contract(self) -> ReportAdapterContractDTO:
        contract_id = compute_report_adapter_contract_id(
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
            report_kind=REPORT_KIND,
            subject_kind=SUBJECT_KIND,
            dimension_namespace=DIMENSION_NAMESPACE,
            compatibility_id=COMPATIBILITY_ID,
            missing_value_semantics="error",
        )
        return ReportAdapterContractDTO(
            contract_id=contract_id,
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
            report_kind=REPORT_KIND,
            subject_kind=SUBJECT_KIND,
            dimension_namespace=DIMENSION_NAMESPACE,
            compatibility_id=COMPATIBILITY_ID,
            missing_value_semantics="error",
        )

    def adapt(self, report: MaterializedProviderEvaluationRunDTO) -> AdaptedReportDTO:
        contract = self.contract()
        run_id = report.run.run_id
        ordered_cases = sorted(
            report.cases,
            key=lambda case: (_priority_rank(case), case.case_ordinal),
        )
        subjects = tuple(
            AdaptedSubjectDTO(
                subject_id=case.case_id,
                label=f"case-{case.case_ordinal:03d}",
                ordinal=index,
                source_ref=case.frame_id,
            )
            for index, case in enumerate(ordered_cases)
        )
        values: list[AdaptedValueDTO] = []
        for case in ordered_cases:
            values.extend(_values_for_case(case, run_id=run_id))
        values_tuple = tuple(values)
        adapted_report_id = compute_adapted_report_id(
            report_id=run_id,
            report_kind=contract.report_kind,
            adapter_contract_id=contract.contract_id,
            compatibility_id=contract.compatibility_id,
            subjects=subjects,
            dimensions=DIMENSIONS,
            values=values_tuple,
        )
        return AdaptedReportDTO(
            adapted_report_id=adapted_report_id,
            report_id=run_id,
            report_kind=contract.report_kind,
            adapter_contract_id=contract.contract_id,
            compatibility_id=contract.compatibility_id,
            subjects=subjects,
            dimensions=DIMENSIONS,
            values=values_tuple,
        )


def compile_provider_evaluation_report(
    run: MaterializedProviderEvaluationRunDTO,
    *,
    layout_recipe: LayoutRecipe,
    store: ArtifactStore,
) -> CompiledReportArtifactDTO:
    """Compile a materialized provider-evaluation run through the existing
    Artifacts report-compilation pipeline. Does not reimplement persistence
    or aggregate-closure validation - see `zeromodel.artifacts.report_compiler`."""
    adapter = ProviderEvaluationReportAdapter()
    return compile_report(
        adapter=adapter, report=run, layout_recipe=layout_recipe, store=store
    )


__all__ = [
    "ProviderEvaluationReportAdapter",
    "compile_provider_evaluation_report",
]
