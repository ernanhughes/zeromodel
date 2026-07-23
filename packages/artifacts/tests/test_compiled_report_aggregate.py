"""Normal-path tests for the compiled-report aggregate (Stage C, sections
8-9, 15): resolving all five artifacts a compiled report references,
proving they form one coherent report, and issuing an auditable closure
receipt for a genuinely valid aggregate.

See `test_compiled_report_aggregate_adversarial.py` for the adversarial
counterpart - individually valid but mutually inconsistent artifacts
constructed by hand to prove `validate_compiled_report_aggregate`
(aggregate closure), not a digest check, is what catches inconsistency.
"""

from __future__ import annotations

import pytest

from zeromodel.artifacts import (
    InMemoryArtifactStore,
    ScoreSemantics,
    build_compiled_report_closure_receipt,
    compile_report,
    load_compiled_report_aggregate,
    validate_compiled_report_aggregate,
)
from zeromodel.artifacts.report_dto import AdaptedReportDTO
from zeromodel.artifacts.report_errors import ReportCompilationError


# --- Normal path -----------------------------------------------------------


def test_aggregate_loader_resolves_all_five_objects(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    contract, adapted = ai_artifact_family
    adapter = FakeAdapter(contract, adapted)
    store = InMemoryArtifactStore()
    compiled = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )

    aggregate = load_compiled_report_aggregate(
        ref=compiled.artifact_ref, resolver=store
    )

    assert aggregate.compiled_report == compiled
    assert aggregate.adapted_report.adapted_report_id == adapted.adapted_report_id
    assert aggregate.adapter_contract.contract_id == contract.contract_id
    assert aggregate.score_table.shape == (
        len(adapted.subjects),
        len(adapted.dimensions),
    )
    assert aggregate.layout_recipe.digest == source_layout_recipe.digest
    assert aggregate.vpm_artifact.shape == aggregate.score_table.shape


def test_aggregate_loader_succeeds_after_original_objects_discarded(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    """Proves the aggregate is recoverable from the store alone - no
    reliance on any in-memory object from the compiling process."""
    contract, adapted = ai_artifact_family
    adapter = FakeAdapter(contract, adapted)
    store = InMemoryArtifactStore()
    compiled_ref = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    ).artifact_ref
    del contract, adapted, adapter

    aggregate = load_compiled_report_aggregate(ref=compiled_ref, resolver=store)
    assert aggregate.compiled_report.artifact_ref == compiled_ref
    validate_compiled_report_aggregate(aggregate)


def test_aggregate_closure_succeeds_for_compile_report_output(
    quality_family, source_layout_recipe, FakeAdapter
):
    contract, adapted = quality_family
    adapter = FakeAdapter(contract, adapted)
    store = InMemoryArtifactStore()
    compiled = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    aggregate = load_compiled_report_aggregate(
        ref=compiled.artifact_ref, resolver=store
    )
    validate_compiled_report_aggregate(aggregate)  # must not raise


def test_aggregate_closure_is_deterministic(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    contract, adapted = ai_artifact_family
    adapter = FakeAdapter(contract, adapted)
    store = InMemoryArtifactStore()
    compiled = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    aggregate_a = load_compiled_report_aggregate(
        ref=compiled.artifact_ref, resolver=store
    )
    aggregate_b = load_compiled_report_aggregate(
        ref=compiled.artifact_ref, resolver=store
    )
    # ScoreTable/VPMArtifact hold numpy arrays, so dataclass `==` on the
    # aggregate itself is not meaningful - compare by each member's own
    # content identity instead.
    assert aggregate_a.compiled_report == aggregate_b.compiled_report
    assert aggregate_a.adapted_report == aggregate_b.adapted_report
    assert aggregate_a.adapter_contract == aggregate_b.adapter_contract
    assert aggregate_a.score_table.digest == aggregate_b.score_table.digest
    assert aggregate_a.layout_recipe.digest == aggregate_b.layout_recipe.digest
    assert aggregate_a.vpm_artifact.artifact_id == aggregate_b.vpm_artifact.artifact_id


def test_vpm_cells_map_to_correctly_bound_adapted_values(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    contract, adapted = ai_artifact_family
    adapter = FakeAdapter(contract, adapted)
    store = InMemoryArtifactStore()
    compiled = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    aggregate = load_compiled_report_aggregate(
        ref=compiled.artifact_ref, resolver=store
    )

    for cell in aggregate.compiled_report.cell_bindings:
        vpm_cell = aggregate.vpm_artifact.cell(cell.view_row, cell.view_column)
        value = aggregate.adapted_report.values[cell.value_index]
        assert value.subject_id == vpm_cell.row_id
        assert value.dimension_id == vpm_cell.metric_id
        assert value.raw_value == vpm_cell.raw_value


def test_confidence_importance_parents_and_attributes_survive_compilation_and_reload(
    make_contract, source_layout_recipe, FakeAdapter
):
    """Section 15: confidence, importance, parent report ids, and
    report-level attributes all round-trip through `compile_report` and
    `load_compiled_report_aggregate`, even though none of them appear in
    the VPM matrix itself."""
    from zeromodel.artifacts import (
        AdaptedDimensionDTO,
        AdaptedSubjectDTO,
        ReportFindingRefDTO,
        SourceBindingDTO,
    )
    from zeromodel.artifacts.report_dto import (
        AdaptedValueDTO,
        compute_adapted_report_id,
    )

    contract = make_contract(
        adapter_id="test.lineage_e2e",
        report_kind="test-lineage-e2e",
        compatibility_id="test-lineage-e2e/v1",
    )
    subjects = (AdaptedSubjectDTO(subject_id="s1"),)
    dimensions = (
        AdaptedDimensionDTO(
            dimension_id="d1",
            label="D1",
            score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        ),
    )
    finding_ref = ReportFindingRefDTO(report_id="report-1", finding_id="s1:d1")
    source_binding = SourceBindingDTO(
        subject_id="s1",
        dimension_id="d1",
        finding_ref=finding_ref,
        source_uri="doc://report-1#s1",
        source_start=10,
        source_end=42,
    )
    values = (
        AdaptedValueDTO(
            subject_id="s1",
            dimension_id="d1",
            raw_value=0.66,
            source_binding=source_binding,
            confidence=0.75,
            importance=3.0,
        ),
    )
    parent_report_ids = ("parent-x",)
    attributes = (("run_id", "77"),)
    adapted_report_id = compute_adapted_report_id(
        report_id="report-1",
        report_kind=contract.report_kind,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        subjects=subjects,
        dimensions=dimensions,
        values=values,
        parent_report_ids=parent_report_ids,
        attributes=attributes,
    )
    adapted = AdaptedReportDTO(
        adapted_report_id=adapted_report_id,
        report_id="report-1",
        report_kind=contract.report_kind,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        subjects=subjects,
        dimensions=dimensions,
        values=values,
        parent_report_ids=parent_report_ids,
        attributes=attributes,
    )

    store = InMemoryArtifactStore()
    compiled = compile_report(
        adapter=FakeAdapter(contract, adapted),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    aggregate = load_compiled_report_aggregate(
        ref=compiled.artifact_ref, resolver=store
    )

    reloaded_value = aggregate.adapted_report.values[0]
    assert reloaded_value.confidence == 0.75
    assert reloaded_value.importance == 3.0
    assert reloaded_value.source_binding == source_binding
    assert aggregate.adapted_report.parent_report_ids == parent_report_ids
    assert dict(aggregate.adapted_report.attributes) == dict(attributes)


def test_closure_receipt_records_every_check_passing(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    contract, adapted = ai_artifact_family
    store = InMemoryArtifactStore()
    compiled = compile_report(
        adapter=FakeAdapter(contract, adapted),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    aggregate = load_compiled_report_aggregate(
        ref=compiled.artifact_ref, resolver=store
    )

    receipt = build_compiled_report_closure_receipt(aggregate)

    assert receipt.compiled_report_ref == compiled.artifact_ref
    assert receipt.adapted_report_ref == compiled.adapted_report_ref
    assert receipt.adapter_contract_ref == compiled.adapter_contract_ref
    assert receipt.score_table_ref == compiled.score_table_ref
    assert receipt.layout_recipe_ref == compiled.layout_recipe_ref
    assert receipt.vpm_artifact_ref == compiled.vpm_artifact_ref
    assert receipt.failure_codes == ()
    assert all(passed for _, passed in receipt.checks)
    assert len(receipt.checks) >= 5


def test_closure_receipt_is_not_generated_for_a_failed_aggregate(
    ai_artifact_family, quality_family, source_layout_recipe, FakeAdapter
):
    """`build_compiled_report_closure_receipt` must fail closed - raise,
    not return a partial receipt - for an aggregate assembled from
    otherwise-valid but mutually inconsistent objects."""
    from zeromodel.artifacts.aggregate import ResolvedCompiledReportAggregateDTO
    from zeromodel.artifacts.core_artifact_persistence import load_vpm_artifact

    contract_a, adapted_a = ai_artifact_family
    contract_b, adapted_b = quality_family
    store = InMemoryArtifactStore()
    compiled_a = compile_report(
        adapter=FakeAdapter(contract_a, adapted_a),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    compiled_b = compile_report(
        adapter=FakeAdapter(contract_b, adapted_b),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    aggregate_a = load_compiled_report_aggregate(
        ref=compiled_a.artifact_ref, resolver=store
    )
    foreign_vpm = load_vpm_artifact(compiled_b.vpm_artifact_ref, resolver=store)
    mismatched = ResolvedCompiledReportAggregateDTO(
        compiled_report=aggregate_a.compiled_report,
        adapted_report=aggregate_a.adapted_report,
        adapter_contract=aggregate_a.adapter_contract,
        score_table=aggregate_a.score_table,
        layout_recipe=aggregate_a.layout_recipe,
        vpm_artifact=foreign_vpm,
    )
    with pytest.raises(ReportCompilationError):
        build_compiled_report_closure_receipt(mismatched)


def test_compatibility_schema_id_is_deterministic(ai_artifact_family):
    from zeromodel.artifacts.compatibility_schema import compute_compatibility_schema_id

    _, adapted = ai_artifact_family
    kwargs = dict(dimensions=adapted.dimensions, missing_value_semantics="error")
    assert compute_compatibility_schema_id(**kwargs) == compute_compatibility_schema_id(
        **kwargs
    )
