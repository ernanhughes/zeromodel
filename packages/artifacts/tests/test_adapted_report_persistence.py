"""Normal-path tests for persisting/reloading `AdaptedReportDTO` directly
(Stage C, section 5): before this module, the adapted report only ever
existed as a discarded local Python object inside `compile_report()`.
"""

from __future__ import annotations

import pytest

from zeromodel.artifacts import (
    ADAPTED_REPORT_ARTIFACT_KIND,
    ArtifactRef,
    InMemoryArtifactStore,
    load_adapted_report,
    sha256_digest,
    store_adapted_report,
)
from zeromodel.artifacts.report_errors import ReportCompilationError


def test_store_adapted_report_ref_equals_adapted_report_id(ai_artifact_family):
    _, adapted = ai_artifact_family
    store = InMemoryArtifactStore()
    ref = store_adapted_report(adapted, store=store)
    assert ref.artifact_kind == ADAPTED_REPORT_ARTIFACT_KIND
    assert ref.artifact_id == adapted.adapted_report_id


def test_identical_adapted_report_persists_to_identical_ref(ai_artifact_family):
    _, adapted = ai_artifact_family
    store = InMemoryArtifactStore()
    ref_a = store_adapted_report(adapted, store=store)
    ref_b = store_adapted_report(adapted, store=store)
    assert ref_a == ref_b


def test_adapted_report_round_trip_preserves_every_field(ai_artifact_family):
    contract, adapted = ai_artifact_family
    store = InMemoryArtifactStore()
    ref = store_adapted_report(adapted, store=store)

    reloaded = load_adapted_report(ref, resolver=store)

    assert reloaded == adapted
    assert reloaded.adapted_report_id == adapted.adapted_report_id
    assert reloaded.subjects == adapted.subjects
    assert reloaded.dimensions == adapted.dimensions
    assert reloaded.values == adapted.values
    assert reloaded.parent_report_ids == adapted.parent_report_ids
    assert reloaded.attributes == adapted.attributes


def test_confidence_and_importance_survive_round_trip(
    make_contract, make_adapted_report
):
    from zeromodel.artifacts import (
        AdaptedDimensionDTO,
        AdaptedSubjectDTO,
        AdaptedValueDTO,
        ReportFindingRefDTO,
        ScoreSemantics,
        SourceBindingDTO,
    )

    contract = make_contract(
        adapter_id="test.confidence",
        report_kind="test-confidence",
        compatibility_id="test-confidence/v1",
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
        subject_id="s1", dimension_id="d1", finding_ref=finding_ref
    )
    values = (
        AdaptedValueDTO(
            subject_id="s1",
            dimension_id="d1",
            raw_value=0.42,
            source_binding=source_binding,
            confidence=0.87,
            importance=2.5,
        ),
    )
    adapted = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values
    )
    store = InMemoryArtifactStore()
    ref = store_adapted_report(adapted, store=store)
    reloaded = load_adapted_report(ref, resolver=store)

    assert reloaded.values[0].confidence == 0.87
    assert reloaded.values[0].importance == 2.5
    assert reloaded.values[0].source_binding == source_binding


def test_parent_report_ids_and_attributes_survive_round_trip(make_contract, make_value):
    from zeromodel.artifacts.report_dto import (
        AdaptedReportDTO,
        compute_adapted_report_id,
    )
    from zeromodel.artifacts import (
        AdaptedDimensionDTO,
        AdaptedSubjectDTO,
        ScoreSemantics,
    )

    contract = make_contract(
        adapter_id="test.lineage",
        report_kind="test-lineage",
        compatibility_id="test-lineage/v1",
    )
    subjects = (AdaptedSubjectDTO(subject_id="s1"),)
    dimensions = (
        AdaptedDimensionDTO(
            dimension_id="d1",
            label="D1",
            score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        ),
    )
    values = (make_value(subject_id="s1", dimension_id="d1", raw_value=0.3),)
    parent_report_ids = ("parent-report-a", "parent-report-b")
    attributes = (("source_system", "writer"), ("run_id", "42"))
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
    ref = store_adapted_report(adapted, store=store)
    reloaded = load_adapted_report(ref, resolver=store)

    assert reloaded.parent_report_ids == parent_report_ids
    # decode_attributes reconstructs from a canonical dict, so pair order
    # is not preserved - compare as sets of pairs instead.
    assert dict(reloaded.attributes) == dict(attributes)


def test_load_adapted_report_rejects_wrong_artifact_kind(ai_artifact_family):
    _, adapted = ai_artifact_family
    store = InMemoryArtifactStore()
    ref = store_adapted_report(adapted, store=store)
    wrong_kind_ref = ArtifactRef(
        artifact_kind="not-the-right-kind", artifact_id=ref.artifact_id
    )
    with pytest.raises(ReportCompilationError, match="artifact_kind"):
        load_adapted_report(wrong_kind_ref, resolver=store)


def test_load_adapted_report_rejects_wrong_digest(ai_artifact_family):
    _, adapted = ai_artifact_family
    store = InMemoryArtifactStore()
    store_adapted_report(adapted, store=store)
    fake_ref = ArtifactRef(
        artifact_kind=ADAPTED_REPORT_ARTIFACT_KIND,
        artifact_id=sha256_digest(b"never-stored"),
    )
    with pytest.raises(Exception):  # ArtifactNotFoundError from the store itself
        load_adapted_report(fake_ref, resolver=store)


def test_loading_does_not_use_the_manifest_as_authority(ai_artifact_family):
    from zeromodel.artifacts import ArtifactManifestConflictError

    _, adapted = ai_artifact_family
    store = InMemoryArtifactStore()
    ref = store_adapted_report(adapted, store=store)

    canonical_bytes = store.resolve_canonical_bytes(ref)
    with pytest.raises(ArtifactManifestConflictError):
        store.put(ref.artifact_kind, canonical_bytes, manifest={"lying": True})

    class ManifestForbiddenResolver:
        def has(self, r):
            return store.has(r)

        def resolve_canonical_bytes(self, r):
            return store.resolve_canonical_bytes(r)

        def resolve_manifest(self, r):
            raise AssertionError("load_adapted_report must never read the manifest")

    reloaded = load_adapted_report(ref, resolver=ManifestForbiddenResolver())
    assert reloaded == adapted
