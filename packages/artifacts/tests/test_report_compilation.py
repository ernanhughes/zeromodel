from __future__ import annotations

import pytest

from zeromodel.artifacts import (
    AdaptedDimensionDTO,
    AdaptedSubjectDTO,
    ArtifactManifestConflictError,
    ArtifactRef,
    InMemoryArtifactStore,
    ScoreSemantics,
    compile_report,
    load_compiled_report_artifact,
    load_compiled_report_vpm,
    sha256_digest,
)
from zeromodel.artifacts.report_errors import ReportCompilationError
from zeromodel.core.artifact import LayoutRecipe


def test_identical_report_and_layout_yield_identical_compiled_identity(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    contract, adapted = ai_artifact_family
    adapter = FakeAdapter(contract, adapted)
    compiled_a = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=InMemoryArtifactStore(),
    )
    compiled_b = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=InMemoryArtifactStore(),
    )
    assert compiled_a.artifact_ref.artifact_id == compiled_b.artifact_ref.artifact_id


def test_changed_layout_changes_compiled_identity(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    contract, adapted = ai_artifact_family
    adapter = FakeAdapter(contract, adapted)
    reversed_layout = LayoutRecipe(
        {
            "version": "vpm-layout/0",
            "row_order": {
                "kind": "lexicographic",
                "tie_break": "row_id",
                "keys": [{"metric_id": "generic_phrasing", "direction": "desc"}],
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    compiled_a = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=InMemoryArtifactStore(),
    )
    compiled_b = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=reversed_layout,
        store=InMemoryArtifactStore(),
    )
    assert compiled_a.artifact_ref.artifact_id != compiled_b.artifact_ref.artifact_id


def test_every_value_maps_to_exactly_one_cell_binding_with_correct_source(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    contract, adapted = ai_artifact_family
    adapter = FakeAdapter(contract, adapted)
    compiled = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=InMemoryArtifactStore(),
    )

    assert len(compiled.cell_bindings) == len(adapted.subjects) * len(
        adapted.dimensions
    )
    by_key = {
        (cell.subject_id, cell.dimension_id): cell for cell in compiled.cell_bindings
    }
    for value in adapted.values:
        cell = by_key[(value.subject_id, value.dimension_id)]
        assert cell.source_binding == value.source_binding


def test_cell_bindings_use_view_coordinates_not_source_order(
    quality_family, FakeAdapter
):
    """Regression for the external review's Blocker #1, strengthened per
    Stage C section 13: the original version of this test used a fixture
    whose source order already happened to match the sorted order, so
    `view_row == source_row_index` held even with the bug the test claimed
    to catch. This version uses a genuine permutation on both axes.

    `quality_family` declares sentence-001 quality=0.4 (source row 0) and
    sentence-002 quality=0.95 (source row 1). A descending sort on quality
    must place sentence-002 at view_row 0 - a real row permutation, not a
    coincidental identity. The column order is also explicitly reversed
    (source declares quality, clarity; the recipe requests clarity,
    quality) - a real column permutation.
    """
    contract, adapted = quality_family
    adapter = FakeAdapter(contract, adapted)
    permuted_layout = LayoutRecipe(
        {
            "version": "vpm-layout/0",
            "row_order": {
                "kind": "lexicographic",
                "tie_break": "row_id",
                "keys": [{"metric_id": "quality", "direction": "desc"}],
            },
            "column_order": {"kind": "explicit", "metric_ids": ("clarity", "quality")},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    compiled = compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=permuted_layout,
        store=InMemoryArtifactStore(),
    )

    by_view_coord = {
        (cell.view_row, cell.view_column): cell for cell in compiled.cell_bindings
    }

    # Row permutation: sentence-002 (0.95) outranks sentence-001 (0.4), so
    # it lands at view_row 0 even though it is source row 1.
    top_row_cell = by_view_coord[(0, 0)]
    assert top_row_cell.subject_id == "sentence-002"
    assert top_row_cell.source_row_index == 1
    assert top_row_cell.view_row != top_row_cell.source_row_index

    second_row_cell = by_view_coord[(1, 0)]
    assert second_row_cell.subject_id == "sentence-001"
    assert second_row_cell.source_row_index == 0
    assert second_row_cell.view_row != second_row_cell.source_row_index

    # Column permutation: view_column 0 is "clarity" (declared second,
    # source_metric_index 1), view_column 1 is "quality" (declared first,
    # source_metric_index 0).
    assert top_row_cell.dimension_id == "clarity"
    assert top_row_cell.source_metric_index == 1
    assert top_row_cell.view_column != top_row_cell.source_metric_index

    quality_cell = by_view_coord[(0, 1)]
    assert quality_cell.dimension_id == "quality"
    assert quality_cell.source_metric_index == 0

    # At least one cell has both a genuinely permuted row and a genuinely
    # permuted column coordinate, proving the binding resolves through
    # VPMArtifact.cell() rather than assuming source order on either axis.
    assert any(
        cell.view_row != cell.source_row_index
        and cell.view_column != cell.source_metric_index
        for cell in compiled.cell_bindings
    )

    # The closure validator independently re-derives subject_id/dimension_id
    # from source indices for every cell - not merely assumed from position.
    for cell in compiled.cell_bindings:
        assert cell.subject_id == adapted.subjects[cell.source_row_index].subject_id
        assert (
            cell.dimension_id
            == adapted.dimensions[cell.source_metric_index].dimension_id
        )


def test_compile_report_persists_resolvable_core_artifacts(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    """Regression for the external review's Blocker #2: a compiled report
    must be able to resolve its actual VPM after being reloaded from the
    store, not merely name a digest of a since-discarded Python object.
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

    reloaded = load_compiled_report_artifact(ref=compiled.artifact_ref, resolver=store)
    vpm_artifact = load_compiled_report_vpm(ref=reloaded.artifact_ref, resolver=store)

    assert vpm_artifact.shape == (len(adapted.subjects), len(adapted.dimensions))
    for cell_binding in compiled.cell_bindings:
        resolved_cell = vpm_artifact.cell(
            cell_binding.view_row, cell_binding.view_column
        )
        assert resolved_cell.row_id == cell_binding.subject_id
        assert resolved_cell.metric_id == cell_binding.dimension_id
        assert resolved_cell.source_row_index == cell_binding.source_row_index
        assert resolved_cell.source_metric_index == cell_binding.source_metric_index


def test_compiled_artifact_round_trips_through_the_store(
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

    loaded = load_compiled_report_artifact(ref=compiled.artifact_ref, resolver=store)
    assert loaded == compiled


def test_loading_does_not_use_the_manifest_as_authority(
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

    # compile_report always stores manifest=None (empty); attempting to
    # attach a *different* manifest to the same identity must be rejected
    # by the store itself (see packages/artifacts store-hardening tests) -
    # proving the two fixes compose correctly.
    canonical_bytes = store.resolve_canonical_bytes(compiled.artifact_ref)
    with pytest.raises(ArtifactManifestConflictError):
        store.put(
            compiled.artifact_ref.artifact_kind,
            canonical_bytes,
            manifest={"lying": True},
        )

    # And loading never even reads the manifest to begin with.
    class ManifestForbiddenResolver:
        def has(self, ref):
            return store.has(ref)

        def resolve_canonical_bytes(self, ref):
            return store.resolve_canonical_bytes(ref)

        def resolve_manifest(self, ref):
            raise AssertionError(
                "load_compiled_report_artifact must never read the manifest"
            )

    reloaded = load_compiled_report_artifact(
        ref=compiled.artifact_ref, resolver=ManifestForbiddenResolver()
    )
    assert reloaded == compiled


def test_wrong_artifact_kind_fails_closed_on_load(
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

    wrong_kind_ref = ArtifactRef(
        artifact_kind="not-the-right-kind",
        artifact_id=compiled.artifact_ref.artifact_id,
    )
    with pytest.raises(ReportCompilationError, match="artifact_kind"):
        load_compiled_report_artifact(ref=wrong_kind_ref, resolver=store)


def test_wrong_digest_fails_closed_on_load(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    contract, adapted = ai_artifact_family
    adapter = FakeAdapter(contract, adapted)
    store = InMemoryArtifactStore()
    compile_report(
        adapter=adapter,
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )

    fake_ref = ArtifactRef(
        artifact_kind="zeromodel.artifacts.compiled-report/v1",
        artifact_id=sha256_digest(b"never-stored"),
    )
    with pytest.raises(Exception):  # ArtifactNotFoundError from the store itself
        load_compiled_report_artifact(ref=fake_ref, resolver=store)


def test_same_compatibility_id_but_different_dimension_schema_yields_different_schema_id(
    source_layout_recipe, FakeAdapter, make_contract, make_value, make_adapted_report
):
    """Regression for review finding #5: `compatibility_id` is an opaque,
    caller-chosen string - two adapters can declare the same one while
    actually producing different dimension schemas. `compatibility_schema_id`
    must catch that even when the human-readable label is identical.
    """
    contract_a = make_contract(
        adapter_id="test.family_a",
        report_kind="test-family-a",
        compatibility_id="shared-label/v1",
    )
    subjects_a = (AdaptedSubjectDTO(subject_id="s1"),)
    dimensions_a = (
        AdaptedDimensionDTO(
            dimension_id="d1",
            label="D1",
            score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        ),
    )
    values_a = (make_value(subject_id="s1", dimension_id="d1", raw_value=0.5),)
    adapted_a = make_adapted_report(
        contract=contract_a,
        subjects=subjects_a,
        dimensions=dimensions_a,
        values=values_a,
    )

    contract_b = make_contract(
        adapter_id="test.family_b",
        report_kind="test-family-b",
        compatibility_id="shared-label/v1",
    )
    subjects_b = (AdaptedSubjectDTO(subject_id="s1"),)
    dimensions_b = (
        AdaptedDimensionDTO(
            dimension_id="d1",
            label="D1",
            score_semantics=ScoreSemantics.HIGHER_IS_WORSE,
        ),
    )
    values_b = (make_value(subject_id="s1", dimension_id="d1", raw_value=0.5),)
    adapted_b = make_adapted_report(
        contract=contract_b,
        subjects=subjects_b,
        dimensions=dimensions_b,
        values=values_b,
    )

    compiled_a = compile_report(
        adapter=FakeAdapter(contract_a, adapted_a),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=InMemoryArtifactStore(),
    )
    compiled_b = compile_report(
        adapter=FakeAdapter(contract_b, adapted_b),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=InMemoryArtifactStore(),
    )

    assert compiled_a.compatibility_id == compiled_b.compatibility_id
    assert compiled_a.compatibility_schema_id != compiled_b.compatibility_schema_id


def test_missing_value_with_error_semantics_fails_compilation(
    source_layout_recipe, FakeAdapter, make_contract, make_value, make_adapted_report
):
    contract = make_contract(
        adapter_id="test.sparse",
        report_kind="test-sparse",
        compatibility_id="test-sparse/v1",
    )
    subjects = (AdaptedSubjectDTO(subject_id="s1"), AdaptedSubjectDTO(subject_id="s2"))
    dimensions = (
        AdaptedDimensionDTO(
            dimension_id="d1",
            label="D1",
            score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        ),
    )
    # s2/d1 deliberately missing.
    values = (make_value(subject_id="s1", dimension_id="d1", raw_value=0.5),)
    adapted = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values
    )

    adapter = FakeAdapter(contract, adapted)
    with pytest.raises(ReportCompilationError, match="missing value"):
        compile_report(
            adapter=adapter,
            report=object(),
            layout_recipe=source_layout_recipe,
            store=InMemoryArtifactStore(),
        )
