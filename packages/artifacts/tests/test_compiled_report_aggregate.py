"""Normal-path and adversarial tests for the compiled-report aggregate
(Stage C, sections 8-9, 14, 15): resolving all four artifacts a compiled
report references and proving they form one coherent report, not merely
four individually digest-valid objects.

The adversarial tests construct individually valid but mutually
inconsistent artifacts by hand - bypassing `compile_report()`, which
always happens to produce a consistent aggregate - to prove
`validate_compiled_report_aggregate` (aggregate closure), not a digest
check, is what catches the inconsistency.
"""

from __future__ import annotations

import pytest

from zeromodel.artifacts import (
    ArtifactRef,
    CompiledReportArtifactDTO,
    InMemoryArtifactStore,
    ScoreSemantics,
    build_compiled_report_closure_receipt,
    canonical_json_bytes,
    compile_report,
    load_compiled_report_aggregate,
    validate_compiled_report_aggregate,
)
from zeromodel.artifacts.adapted_report_persistence import store_adapted_report
from zeromodel.artifacts.compatibility_schema import compute_report_semantics_id
from zeromodel.artifacts.compiled_artifact import (
    CompatibilityInfo,
    CoreArtifactRefs,
    ReportSemanticsInfo,
    compiled_report_identity_payload,
    compute_compiled_report_artifact_id,
)
from zeromodel.artifacts.report_dto import AdaptedReportDTO
from zeromodel.artifacts.report_errors import ReportCompilationError
from zeromodel.core.artifact import LayoutRecipe


def _persist_variant(
    compiled: CompiledReportArtifactDTO,
    *,
    store,
    adapted_report_ref: ArtifactRef | None = None,
    score_table_ref: ArtifactRef | None = None,
    layout_recipe_ref: ArtifactRef | None = None,
    vpm_artifact_ref: ArtifactRef | None = None,
    cell_bindings=None,
) -> CompiledReportArtifactDTO:
    """Build and persist a `CompiledReportArtifactDTO` that is a genuinely
    self-consistent, correctly-digested variant of `compiled` with one or
    more references swapped out - the adversarial fixture this module's
    docstring describes: every individual artifact is still valid, but the
    collection is not.
    """
    core_refs = CoreArtifactRefs(
        score_table_ref=score_table_ref or compiled.score_table_ref,
        layout_recipe_ref=layout_recipe_ref or compiled.layout_recipe_ref,
        vpm_artifact_ref=vpm_artifact_ref or compiled.vpm_artifact_ref,
    )
    resolved_adapted_ref = adapted_report_ref or compiled.adapted_report_ref
    resolved_cell_bindings = (
        cell_bindings if cell_bindings is not None else compiled.cell_bindings
    )
    compatibility = CompatibilityInfo(
        compatibility_id=compiled.compatibility_id,
        compatibility_schema_id=compiled.compatibility_schema_id,
        missing_value_semantics=compiled.missing_value_semantics,
    )
    report_semantics = ReportSemanticsInfo(
        report_kind=compiled.report_kind,
        subject_kind=compiled.subject_kind,
        dimension_namespace=compiled.dimension_namespace,
        duplicate_value_semantics=compiled.duplicate_value_semantics,
        report_semantics_id=compiled.report_semantics_id,
    )
    artifact_id = compute_compiled_report_artifact_id(
        adapted_report_ref=resolved_adapted_ref,
        adapter_contract_id=compiled.adapter_contract_id,
        compatibility=compatibility,
        report_semantics=report_semantics,
        core_refs=core_refs,
        subjects=compiled.subjects,
        dimensions=compiled.dimensions,
        cell_bindings=resolved_cell_bindings,
    )
    variant = CompiledReportArtifactDTO(
        artifact_ref=ArtifactRef(
            artifact_kind=compiled.artifact_kind, artifact_id=artifact_id
        ),
        adapted_report_ref=resolved_adapted_ref,
        adapter_contract_id=compiled.adapter_contract_id,
        compatibility_id=compiled.compatibility_id,
        compatibility_schema_id=compiled.compatibility_schema_id,
        missing_value_semantics=compiled.missing_value_semantics,
        report_kind=compiled.report_kind,
        subject_kind=compiled.subject_kind,
        dimension_namespace=compiled.dimension_namespace,
        duplicate_value_semantics=compiled.duplicate_value_semantics,
        report_semantics_id=compiled.report_semantics_id,
        score_table_ref=core_refs.score_table_ref,
        layout_recipe_ref=core_refs.layout_recipe_ref,
        vpm_artifact_ref=core_refs.vpm_artifact_ref,
        subjects=compiled.subjects,
        dimensions=compiled.dimensions,
        cell_bindings=resolved_cell_bindings,
    )
    canonical_bytes = canonical_json_bytes(compiled_report_identity_payload(variant))
    ref = store.put(variant.artifact_kind, canonical_bytes, manifest=None)
    assert ref.artifact_id == artifact_id
    return variant


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


# --- Adversarial: foreign artifact substitution -----------------------------


def test_foreign_vpm_substitution_is_rejected(
    ai_artifact_family, quality_family, source_layout_recipe, FakeAdapter
):
    """Section 14.1: a self-consistent compiled report referencing a valid
    VPM from a different report must be rejected by aggregate closure, not
    by a digest mismatch (every digest here is genuinely correct)."""
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

    variant = _persist_variant(
        compiled_a, store=store, vpm_artifact_ref=compiled_b.vpm_artifact_ref
    )
    with pytest.raises(ReportCompilationError):
        load_compiled_report_aggregate(ref=variant.artifact_ref, resolver=store)


def test_foreign_score_table_substitution_is_rejected(
    ai_artifact_family, quality_family, source_layout_recipe, FakeAdapter
):
    """Section 14.2."""
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

    variant = _persist_variant(
        compiled_a, store=store, score_table_ref=compiled_b.score_table_ref
    )
    with pytest.raises(ReportCompilationError):
        load_compiled_report_aggregate(ref=variant.artifact_ref, resolver=store)


def test_foreign_layout_recipe_substitution_is_rejected(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    """Section 14.3: a different recipe that happens to produce the same
    row/column ordering for this fixture (both are plain source-order
    recipes, just under a different declared name) must still be rejected
    - the aggregate binds the actual recipe object, not merely its
    observed output ordering."""
    contract, adapted = ai_artifact_family
    adapter = FakeAdapter(contract, adapted)
    store = InMemoryArtifactStore()
    compiled = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )

    lookalike_recipe = LayoutRecipe(
        {
            "version": "vpm-layout/0",
            "name": "lookalike-source-order",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    assert lookalike_recipe.digest != source_layout_recipe.digest
    from zeromodel.artifacts.core_artifact_persistence import store_layout_recipe

    lookalike_ref = store_layout_recipe(lookalike_recipe, store=store)

    variant = _persist_variant(compiled, store=store, layout_recipe_ref=lookalike_ref)
    with pytest.raises(ReportCompilationError):
        load_compiled_report_aggregate(ref=variant.artifact_ref, resolver=store)


def test_adapted_report_substitution_with_identical_matrix_is_rejected(
    make_contract, make_value, make_adapted_report, source_layout_recipe, FakeAdapter
):
    """Section 14.4: two adapted reports share identical subjects,
    dimensions, and raw values, but differ in confidence, importance,
    source finding, a parent report id, and a report-level attribute -
    none of which the VPM matrix preserves. Substituting report B under
    compiled report A must be rejected even though the numeric matrix is
    byte-identical."""
    from zeromodel.artifacts import (
        AdaptedDimensionDTO,
        AdaptedSubjectDTO,
        ReportFindingRefDTO,
        SourceBindingDTO,
    )
    from zeromodel.artifacts.report_dto import AdaptedValueDTO

    contract = make_contract(
        adapter_id="test.identical_matrix",
        report_kind="test-identical-matrix",
        compatibility_id="test-identical-matrix/v1",
    )
    subjects = (AdaptedSubjectDTO(subject_id="s1"),)
    dimensions = (
        AdaptedDimensionDTO(
            dimension_id="d1",
            label="D1",
            score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        ),
    )

    def _value(*, confidence, importance, finding_id, report_id):
        finding_ref = ReportFindingRefDTO(report_id=report_id, finding_id=finding_id)
        source_binding = SourceBindingDTO(
            subject_id="s1", dimension_id="d1", finding_ref=finding_ref
        )
        return AdaptedValueDTO(
            subject_id="s1",
            dimension_id="d1",
            raw_value=0.5,
            source_binding=source_binding,
            confidence=confidence,
            importance=importance,
        )

    values_a = (
        _value(
            confidence=0.9, importance=1.0, finding_id="finding-a", report_id="report-a"
        ),
    )
    values_b = (
        _value(
            confidence=0.1, importance=5.0, finding_id="finding-b", report_id="report-a"
        ),
    )
    adapted_a = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values_a
    )
    adapted_b = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values_b
    )
    assert adapted_a.adapted_report_id != adapted_b.adapted_report_id

    store = InMemoryArtifactStore()
    compiled_a = compile_report(
        adapter=FakeAdapter(contract, adapted_a),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    adapted_b_ref = store_adapted_report(adapted_b, store=store)

    variant = _persist_variant(
        compiled_a, store=store, adapted_report_ref=adapted_b_ref
    )
    with pytest.raises(ReportCompilationError):
        load_compiled_report_aggregate(ref=variant.artifact_ref, resolver=store)


def test_wrong_value_index_mapping_is_rejected(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    """Section 14.5: cell coordinates, subject/dimension ids, and source
    bindings are all otherwise plausible, but one cell's `value_index`
    points at a different adapted value than the one it claims to
    describe."""
    contract, adapted = ai_artifact_family
    adapter = FakeAdapter(contract, adapted)
    store = InMemoryArtifactStore()
    compiled = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )

    # A one-directional value_index overwrite would create a duplicate
    # index, which CompiledReportArtifactDTO's own closure check (a
    # bijection over {0, ..., n-1}) already rejects on its own - that
    # would prove the wrong thing. A full swap between two cells keeps
    # the bijection intact so the DTO constructs cleanly; only the
    # aggregate-level cross-check against the *resolved adapted values*
    # can catch that each cell's value_index now names the wrong pair.
    bindings = list(compiled.cell_bindings)
    first, second = bindings[0], bindings[1]
    bindings[0] = type(first)(
        view_row=first.view_row,
        view_column=first.view_column,
        source_row_index=first.source_row_index,
        source_metric_index=first.source_metric_index,
        subject_id=first.subject_id,
        dimension_id=first.dimension_id,
        value_index=second.value_index,
        source_binding=first.source_binding,
    )
    bindings[1] = type(second)(
        view_row=second.view_row,
        view_column=second.view_column,
        source_row_index=second.source_row_index,
        source_metric_index=second.source_metric_index,
        subject_id=second.subject_id,
        dimension_id=second.dimension_id,
        value_index=first.value_index,
        source_binding=second.source_binding,
    )
    assert bindings[0].subject_id != bindings[1].subject_id or (
        bindings[0].dimension_id != bindings[1].dimension_id
    )

    variant = _persist_variant(compiled, store=store, cell_bindings=tuple(bindings))
    with pytest.raises(ReportCompilationError):
        load_compiled_report_aggregate(ref=variant.artifact_ref, resolver=store)


# --- Adversarial: incompatible report/subject semantics ---------------------


def test_different_subject_kind_yields_different_report_semantics_id():
    """Section 14.6: identical dimensions, different subject_kind."""
    sentence_id = compute_report_semantics_id(
        report_kind="claim-evidence",
        subject_kind="sentence",
        dimension_namespace="writer",
        duplicate_value_semantics="reject",
    )
    claim_id = compute_report_semantics_id(
        report_kind="claim-evidence",
        subject_kind="claim",
        dimension_namespace="writer",
        duplicate_value_semantics="reject",
    )
    assert sentence_id != claim_id


def test_different_report_kind_yields_different_report_semantics_id():
    a = compute_report_semantics_id(
        report_kind="report-kind-a",
        subject_kind="sentence",
        dimension_namespace="writer",
        duplicate_value_semantics="reject",
    )
    b = compute_report_semantics_id(
        report_kind="report-kind-b",
        subject_kind="sentence",
        dimension_namespace="writer",
        duplicate_value_semantics="reject",
    )
    assert a != b


def test_different_dimension_namespace_yields_different_report_semantics_id():
    a = compute_report_semantics_id(
        report_kind="claim-evidence",
        subject_kind="sentence",
        dimension_namespace="writer",
        duplicate_value_semantics="reject",
    )
    b = compute_report_semantics_id(
        report_kind="claim-evidence",
        subject_kind="sentence",
        dimension_namespace="critic",
        duplicate_value_semantics="reject",
    )
    assert a != b


def test_report_semantics_id_is_deterministic():
    kwargs = dict(
        report_kind="claim-evidence",
        subject_kind="sentence",
        dimension_namespace="writer",
        duplicate_value_semantics="reject",
    )
    assert compute_report_semantics_id(**kwargs) == compute_report_semantics_id(
        **kwargs
    )


def test_compiled_reports_with_same_dimensions_but_different_subject_kind_are_distinguishable(
    make_contract, make_value, make_adapted_report, source_layout_recipe, FakeAdapter
):
    """End-to-end: two reports share one `compatibility_id` and an
    identical dimension schema, but declare different `subject_kind` -
    their compiled artifacts (and report_semantics_id) must differ."""
    from zeromodel.artifacts import AdaptedDimensionDTO, AdaptedSubjectDTO

    def _build(subject_kind: str):
        contract = make_contract(
            adapter_id=f"test.{subject_kind}",
            report_kind="test-report",
            compatibility_id="shared-label/v1",
            subject_kind=subject_kind,
        )
        subjects = (AdaptedSubjectDTO(subject_id="s1"),)
        dimensions = (
            AdaptedDimensionDTO(
                dimension_id="d1",
                label="D1",
                score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
            ),
        )
        values = (make_value(subject_id="s1", dimension_id="d1", raw_value=0.5),)
        adapted = make_adapted_report(
            contract=contract, subjects=subjects, dimensions=dimensions, values=values
        )
        return compile_report(
            adapter=FakeAdapter(contract, adapted),
            report=object(),
            layout_recipe=source_layout_recipe,
            store=InMemoryArtifactStore(),
        )

    compiled_sentence = _build("sentence")
    compiled_claim = _build("claim")

    assert compiled_sentence.compatibility_id == compiled_claim.compatibility_id
    assert (
        compiled_sentence.compatibility_schema_id
        == compiled_claim.compatibility_schema_id
    )
    assert compiled_sentence.report_semantics_id != compiled_claim.report_semantics_id
    assert (
        compiled_sentence.artifact_ref.artifact_id
        != compiled_claim.artifact_ref.artifact_id
    )
