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
