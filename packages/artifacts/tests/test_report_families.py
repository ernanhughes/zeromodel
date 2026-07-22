"""Proves that ZeroModel can compile multiple, semantically distinct report
families over the same subject identities without confusing their score
semantics or merging them into one artifact - see the package README's
"Reports remain semantically distinct" claim.
"""

from __future__ import annotations

from zeromodel.artifacts import (
    AdaptedDimensionDTO,
    AdaptedSubjectDTO,
    InMemoryArtifactStore,
    ScoreSemantics,
    compile_report,
)


def test_ai_artifact_and_quality_families_stay_separate(
    ai_artifact_family, quality_family, source_layout_recipe, FakeAdapter
):
    ai_contract, ai_adapted = ai_artifact_family
    quality_contract, quality_adapted = quality_family

    assert ai_adapted.subjects == quality_adapted.subjects  # same subject identities
    assert ai_contract.contract_id != quality_contract.contract_id
    assert ai_adapted.adapted_report_id != quality_adapted.adapted_report_id
    assert ai_adapted.compatibility_id != quality_adapted.compatibility_id

    store = InMemoryArtifactStore()
    ai_compiled = compile_report(
        adapter=FakeAdapter(ai_contract, ai_adapted),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    quality_compiled = compile_report(
        adapter=FakeAdapter(quality_contract, quality_adapted),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    assert (
        ai_compiled.artifact_ref.artifact_id
        != quality_compiled.artifact_ref.artifact_id
    )
    assert (
        ai_compiled.subjects == quality_compiled.subjects
    )  # same subjects, different artifacts


def test_higher_is_worse_and_higher_is_better_remain_distinguishable_after_compilation(
    ai_artifact_family, quality_family, source_layout_recipe, FakeAdapter
):
    ai_contract, ai_adapted = ai_artifact_family
    quality_contract, quality_adapted = quality_family
    store = InMemoryArtifactStore()

    ai_compiled = compile_report(
        adapter=FakeAdapter(ai_contract, ai_adapted),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )
    quality_compiled = compile_report(
        adapter=FakeAdapter(quality_contract, quality_adapted),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )

    assert all(
        dimension.score_semantics == ScoreSemantics.HIGHER_IS_WORSE
        for dimension in ai_compiled.dimensions
    )
    assert all(
        dimension.score_semantics == ScoreSemantics.HIGHER_IS_BETTER
        for dimension in quality_compiled.dimensions
    )


def test_claim_evidence_report_mixes_polarities_in_one_report(
    source_layout_recipe, FakeAdapter, make_contract, make_value, make_adapted_report
):
    """One report may declare dimensions with different polarity, as long
    as the contract explicitly declares each dimension's own semantics -
    this is not automatic inference."""
    contract = make_contract(
        adapter_id="claim_evidence.evaluator",
        report_kind="claim-evidence",
        compatibility_id="claim-evidence/v1",
    )
    subjects = (
        AdaptedSubjectDTO(subject_id="claim-001"),
        AdaptedSubjectDTO(subject_id="claim-002"),
    )
    dimensions = (
        AdaptedDimensionDTO(
            dimension_id="evidence_coverage",
            label="Evidence coverage",
            score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        ),
        AdaptedDimensionDTO(
            dimension_id="hallucination_energy",
            label="Hallucination energy",
            score_semantics=ScoreSemantics.HIGHER_IS_WORSE,
        ),
        AdaptedDimensionDTO(
            dimension_id="citation_entailment",
            label="Citation entailment",
            score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        ),
        AdaptedDimensionDTO(
            dimension_id="semantic_drift",
            label="Semantic drift",
            score_semantics=ScoreSemantics.HIGHER_IS_WORSE,
        ),
    )
    values = tuple(
        make_value(
            subject_id=subject.subject_id,
            dimension_id=dimension.dimension_id,
            raw_value=0.5,
        )
        for subject in subjects
        for dimension in dimensions
    )
    adapted = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values
    )

    compiled = compile_report(
        adapter=FakeAdapter(contract, adapted),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=InMemoryArtifactStore(),
    )

    semantics_by_dimension = {
        d.dimension_id: d.score_semantics for d in compiled.dimensions
    }
    assert (
        semantics_by_dimension["evidence_coverage"] == ScoreSemantics.HIGHER_IS_BETTER
    )
    assert (
        semantics_by_dimension["hallucination_energy"] == ScoreSemantics.HIGHER_IS_WORSE
    )
    assert (
        semantics_by_dimension["citation_entailment"] == ScoreSemantics.HIGHER_IS_BETTER
    )
    assert semantics_by_dimension["semantic_drift"] == ScoreSemantics.HIGHER_IS_WORSE


def test_descriptive_dimension_is_preserved_without_transformation(
    source_layout_recipe, FakeAdapter, make_contract, make_value, make_adapted_report
):
    """A descriptive dimension is stored and compiled like any other
    numeric dimension - no automatic conversion into an attention/priority
    signal happens anywhere in this stage (no projection layer exists yet)."""
    contract = make_contract(
        adapter_id="test.descriptive",
        report_kind="test-descriptive",
        compatibility_id="test-descriptive/v1",
    )
    subjects = (AdaptedSubjectDTO(subject_id="s1"),)
    dimensions = (
        AdaptedDimensionDTO(
            dimension_id="word_count",
            label="Word count",
            score_semantics=ScoreSemantics.DESCRIPTIVE,
        ),
    )
    values = (make_value(subject_id="s1", dimension_id="word_count", raw_value=12.0),)
    adapted = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values
    )

    compiled = compile_report(
        adapter=FakeAdapter(contract, adapted),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=InMemoryArtifactStore(),
    )
    assert compiled.dimensions[0].score_semantics == ScoreSemantics.DESCRIPTIVE
    assert compiled.cell_bindings[0].dimension_id == "word_count"
