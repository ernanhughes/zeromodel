"""Adversarial tests for the compiled-report aggregate (Stage C section 14,
plus the external review of 0e56558's four findings): construct
individually valid but mutually inconsistent artifacts by hand - bypassing
`compile_report()`, which always happens to produce a consistent aggregate
- to prove `validate_compiled_report_aggregate` (aggregate closure), not a
digest check, is what catches the inconsistency.

Split out of `test_compiled_report_aggregate.py` (which keeps the
normal-path tests) to stay under the repository's 800-line module hard
limit.
"""

from __future__ import annotations

import numpy as np
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
from zeromodel.artifacts.report_errors import ReportCompilationError
from zeromodel.core.artifact import LayoutRecipe


def _persist_variant(
    compiled: CompiledReportArtifactDTO,
    *,
    store,
    adapted_report_ref: ArtifactRef | None = None,
    adapter_contract_ref: ArtifactRef | None = None,
    score_table_ref: ArtifactRef | None = None,
    layout_recipe_ref: ArtifactRef | None = None,
    vpm_artifact_ref: ArtifactRef | None = None,
    subject_kind: str | None = None,
    cell_bindings=None,
) -> CompiledReportArtifactDTO:
    """Build and persist a `CompiledReportArtifactDTO` that is a genuinely
    self-consistent, correctly-digested variant of `compiled` with one or
    more references (or the `subject_kind` field) swapped out - the
    adversarial fixture this module's docstring describes: every
    individual artifact is still valid, but the collection is not.

    `subject_kind`, when given, is a *lying* value that disagrees with
    whatever `adapter_contract_ref` actually resolves to;
    `report_semantics_id` is recomputed so the record stays internally
    self-consistent (i.e. so `CompiledReportArtifactDTO.__post_init__`'s
    own recompute-and-compare check does not itself catch the lie) - only
    resolving and reconciling the actual contract can.
    """
    core_refs = CoreArtifactRefs(
        score_table_ref=score_table_ref or compiled.score_table_ref,
        layout_recipe_ref=layout_recipe_ref or compiled.layout_recipe_ref,
        vpm_artifact_ref=vpm_artifact_ref or compiled.vpm_artifact_ref,
    )
    resolved_adapted_ref = adapted_report_ref or compiled.adapted_report_ref
    resolved_adapter_contract_ref = (
        adapter_contract_ref or compiled.adapter_contract_ref
    )
    resolved_cell_bindings = (
        cell_bindings if cell_bindings is not None else compiled.cell_bindings
    )
    resolved_subject_kind = subject_kind or compiled.subject_kind
    compatibility = CompatibilityInfo(
        compatibility_id=compiled.compatibility_id,
        compatibility_schema_id=compiled.compatibility_schema_id,
        missing_value_semantics=compiled.missing_value_semantics,
    )
    report_semantics_id = compute_report_semantics_id(
        report_kind=compiled.report_kind,
        subject_kind=resolved_subject_kind,
        dimension_namespace=compiled.dimension_namespace,
        duplicate_value_semantics=compiled.duplicate_value_semantics,
    )
    report_semantics = ReportSemanticsInfo(
        report_kind=compiled.report_kind,
        subject_kind=resolved_subject_kind,
        dimension_namespace=compiled.dimension_namespace,
        duplicate_value_semantics=compiled.duplicate_value_semantics,
        report_semantics_id=report_semantics_id,
    )
    artifact_id = compute_compiled_report_artifact_id(
        adapted_report_ref=resolved_adapted_ref,
        adapter_contract_ref=resolved_adapter_contract_ref,
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
        adapter_contract_ref=resolved_adapter_contract_ref,
        compatibility_id=compiled.compatibility_id,
        compatibility_schema_id=compiled.compatibility_schema_id,
        missing_value_semantics=compiled.missing_value_semantics,
        report_kind=compiled.report_kind,
        subject_kind=resolved_subject_kind,
        dimension_namespace=compiled.dimension_namespace,
        duplicate_value_semantics=compiled.duplicate_value_semantics,
        report_semantics_id=report_semantics_id,
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


# --- Adversarial: fabricated visual representation (external review, 0e56558 #1, BLOCKER) --


def test_vpm_with_fabricated_normalized_pixels_is_rejected(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    """A VPM can embed the *correct* `ScoreTable` and `LayoutRecipe` (same
    content, same digests), keep the *correct* row/column ordering, and
    have cell bindings that still map to the correct raw values - while
    its `normalized_values` (the actual visible pixels) are fabricated.
    `VPMArtifact.validate()` alone only checks that the matrix is finite
    and in `[0, 1]` with a matching self-computed digest; it never
    recomputes the matrix `build_vpm()` would have produced. This is
    distinct from `test_foreign_vpm_substitution_is_rejected` below (a
    VPM belonging to an unrelated report) - here every other field is
    exactly right, only the pixels are lies.
    """
    from zeromodel.artifacts.core_artifact_persistence import store_vpm_artifact
    from zeromodel.core.artifact import VPMArtifact

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
    real_vpm = aggregate.vpm_artifact

    fabricated_matrix = np.zeros_like(real_vpm.normalized_values)
    assert not np.array_equal(fabricated_matrix, real_vpm.normalized_values)
    fabricated_vpm = VPMArtifact(
        source=real_vpm.source,
        recipe=real_vpm.recipe,
        normalized_values=fabricated_matrix,
        row_order=real_vpm.row_order,
        column_order=real_vpm.column_order,
        provenance=real_vpm.provenance,
    )
    # Self-consistent: a different matrix legitimately produces a
    # different (but internally valid) Core artifact_id.
    assert fabricated_vpm.artifact_id != real_vpm.artifact_id
    fabricated_ref = store_vpm_artifact(fabricated_vpm, store=store)

    variant = _persist_variant(compiled, store=store, vpm_artifact_ref=fabricated_ref)
    with pytest.raises(ReportCompilationError, match="normalized_values"):
        load_compiled_report_aggregate(ref=variant.artifact_ref, resolver=store)


def test_vpm_with_fabricated_row_order_is_rejected(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    """Same shape of attack as above, but on the ordering rather than the
    pixel intensities: a VPM claiming a different-but-still-valid row
    permutation than the one `build_vpm()` would have computed for this
    exact score_table/layout_recipe."""
    from zeromodel.artifacts.core_artifact_persistence import store_vpm_artifact
    from zeromodel.core.artifact import VPMArtifact

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
    real_vpm = aggregate.vpm_artifact
    assert real_vpm.shape[0] == 2  # ai_artifact_family has exactly two subjects

    reversed_row_order = tuple(reversed(real_vpm.row_order))
    assert reversed_row_order != real_vpm.row_order
    fabricated_vpm = VPMArtifact(
        source=real_vpm.source,
        recipe=real_vpm.recipe,
        normalized_values=real_vpm.normalized_values[::-1, :],
        row_order=reversed_row_order,
        column_order=real_vpm.column_order,
        provenance=real_vpm.provenance,
    )
    fabricated_ref = store_vpm_artifact(fabricated_vpm, store=store)

    variant = _persist_variant(compiled, store=store, vpm_artifact_ref=fabricated_ref)
    with pytest.raises(ReportCompilationError):
        load_compiled_report_aggregate(ref=variant.artifact_ref, resolver=store)


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


# --- Adversarial: unreconciled adapter contract (external review, 0e56558 #2) ------


def test_subject_kind_disagreeing_with_resolved_adapter_contract_is_rejected(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    """`report_semantics_id` proves the compiled report's copied
    `subject_kind` (and friends) agree *with each other*; it does not
    prove they agree with the adapter contract `adapter_contract_ref`
    actually names. This builds a compiled report declaring
    `subject_kind="claim"` while its `adapter_contract_ref` points at a
    real, correctly-persisted contract that actually declares
    `subject_kind="sentence"` (`ai_artifact_family`'s adapter default).
    `report_semantics_id` is recomputed to stay internally self-consistent
    - only resolving and reconciling the real contract can catch this.
    """
    contract, adapted = ai_artifact_family
    adapter = FakeAdapter(contract, adapted)
    store = InMemoryArtifactStore()
    compiled = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    assert contract.subject_kind == "sentence"

    variant = _persist_variant(compiled, store=store, subject_kind="claim")
    assert variant.subject_kind == "claim"
    assert variant.report_semantics_id != compiled.report_semantics_id

    with pytest.raises(ReportCompilationError, match="subject_kind"):
        load_compiled_report_aggregate(ref=variant.artifact_ref, resolver=store)


# --- Adversarial: unproven caller-assembled aggregate (external review, 0e56558 #3) -


def test_validate_rejects_a_score_table_not_proven_to_come_from_the_declared_ref(
    ai_artifact_family, quality_family, source_layout_recipe, FakeAdapter
):
    """`validate_compiled_report_aggregate` must not simply trust that a
    caller-supplied `score_table` was actually resolved from
    `compiled_report.score_table_ref` - it must prove it. This builds two
    fully independent, internally coherent compiled reports (A and B) and
    swaps B's *locally coherent* `score_table` into A's aggregate while
    leaving `compiled_report` (with its own, unrelated `score_table_ref`)
    untouched. Every pairwise cross-check among the supplied objects would
    still pass if only compared to each other; only recomputing the
    store-level digest against the declared ref catches this.
    """
    from zeromodel.artifacts.aggregate import ResolvedCompiledReportAggregateDTO
    from zeromodel.artifacts.core_artifact_persistence import load_score_table

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
    foreign_score_table = load_score_table(compiled_b.score_table_ref, resolver=store)

    unproven = ResolvedCompiledReportAggregateDTO(
        compiled_report=aggregate_a.compiled_report,  # still declares A's own score_table_ref
        adapted_report=aggregate_a.adapted_report,
        adapter_contract=aggregate_a.adapter_contract,
        score_table=foreign_score_table,  # never resolved from that ref
        layout_recipe=aggregate_a.layout_recipe,
        vpm_artifact=aggregate_a.vpm_artifact,
    )
    with pytest.raises(ReportCompilationError, match="score_table"):
        validate_compiled_report_aggregate(unproven)


def test_receipt_builder_rejects_an_unproven_caller_assembled_aggregate(
    ai_artifact_family, quality_family, source_layout_recipe, FakeAdapter
):
    """The same gap, exercised through the public receipt builder: it must
    not be possible to obtain a `CompiledReportClosureReceiptDTO` - which
    claims the compiled report's declared refs were checked - for an
    aggregate whose objects were never proven to come from those refs."""
    from zeromodel.artifacts.aggregate import ResolvedCompiledReportAggregateDTO
    from zeromodel.artifacts.core_artifact_persistence import load_layout_recipe

    contract_a, adapted_a = ai_artifact_family
    contract_b, adapted_b = quality_family
    store = InMemoryArtifactStore()
    compiled_a = compile_report(
        adapter=FakeAdapter(contract_a, adapted_a),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    # A genuinely different (but equally valid) recipe for B, so its
    # layout_recipe_ref is not simply identical to A's by content coincidence.
    descending_layout_recipe = LayoutRecipe(
        {
            "version": "vpm-layout/0",
            "row_order": {
                "kind": "lexicographic",
                "tie_break": "row_id",
                "keys": [{"metric_id": "quality", "direction": "desc"}],
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    compiled_b = compile_report(
        adapter=FakeAdapter(contract_b, adapted_b),
        report=object(),
        layout_recipe=descending_layout_recipe,
        store=store,
    )
    aggregate_a = load_compiled_report_aggregate(
        ref=compiled_a.artifact_ref, resolver=store
    )
    foreign_layout_recipe = load_layout_recipe(
        compiled_b.layout_recipe_ref, resolver=store
    )
    assert foreign_layout_recipe.digest != aggregate_a.layout_recipe.digest

    unproven = ResolvedCompiledReportAggregateDTO(
        compiled_report=aggregate_a.compiled_report,
        adapted_report=aggregate_a.adapted_report,
        adapter_contract=aggregate_a.adapter_contract,
        score_table=aggregate_a.score_table,
        layout_recipe=foreign_layout_recipe,
        vpm_artifact=aggregate_a.vpm_artifact,
    )
    with pytest.raises(ReportCompilationError):
        build_compiled_report_closure_receipt(unproven)
