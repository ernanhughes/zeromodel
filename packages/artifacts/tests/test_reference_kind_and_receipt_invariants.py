"""Regression tests for the external review of 0e56558 (round 2), findings
2 and 3: a reference is the pair (artifact_kind, artifact_id) - proving
only a digest proves content, not the complete declared reference; and a
`CompiledReportClosureReceiptDTO` must structurally promise "every check
passed", not merely be internally digest-consistent.
"""

from __future__ import annotations

import pytest

from zeromodel.artifacts import (
    ArtifactRef,
    CompiledReportArtifactDTO,
    InMemoryArtifactStore,
    compile_report,
    load_compiled_report_aggregate,
    sha256_digest,
)
from zeromodel.artifacts.aggregate import (
    CompiledReportClosureReceiptDTO,
    closure_receipt_payload,
)
from zeromodel.artifacts.canonicalization import canonical_json_bytes
from zeromodel.artifacts.compiled_artifact import (
    CompatibilityInfo,
    CoreArtifactRefs,
    ReportSemanticsInfo,
    compute_compiled_report_artifact_id,
)
from zeromodel.artifacts.report_errors import ReportCompilationError

# Mirrors zeromodel.artifacts.aggregate._CLOSURE_CHECK_NAMES (private to that
# module) so this test can construct a shuffled-but-complete checks tuple
# without reaching into an underscore-prefixed internal.
_EXPECTED_CHECK_NAMES = (
    "compiled_report_valid",
    "adapted_report_valid",
    "adapter_contract_valid",
    "score_table_valid",
    "layout_recipe_valid",
    "vpm_artifact_valid",
    "resolved_objects_match_declared_refs",
    "adapted_report_matches_compiled",
    "adapter_contract_matches_compiled_and_adapted",
    "score_table_matches_adapted_report",
    "vpm_source_matches_score_table",
    "vpm_recipe_matches_layout",
    "vpm_matches_deterministic_reconstruction",
    "cell_bindings_match_vpm",
    "cell_bindings_match_adapted_values",
    "compatibility_contract_valid",
)


def _compile(ai_artifact_family, source_layout_recipe, FakeAdapter, store):
    contract, adapted = ai_artifact_family
    return compile_report(
        adapter=FakeAdapter(contract, adapted),
        report=object(),
        layout_recipe=source_layout_recipe,
        store=store,
    )


def test_score_table_ref_with_wrong_kind_but_correct_digest_is_rejected_at_construction(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    """Finding 2: a ref whose `artifact_id` genuinely matches the stored
    content but whose `artifact_kind` is wrong must be rejected - a
    matching digest alone does not prove the complete declared reference.
    `CompiledReportArtifactDTO.__post_init__` must catch this itself,
    before the record can ever reach aggregate validation."""
    store = InMemoryArtifactStore()
    compiled = _compile(ai_artifact_family, source_layout_recipe, FakeAdapter, store)

    wrong_kind_score_table_ref = ArtifactRef(
        artifact_kind="wrong-kind", artifact_id=compiled.score_table_ref.artifact_id
    )
    core_refs = CoreArtifactRefs(
        score_table_ref=wrong_kind_score_table_ref,
        layout_recipe_ref=compiled.layout_recipe_ref,
        vpm_artifact_ref=compiled.vpm_artifact_ref,
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
        adapted_report_ref=compiled.adapted_report_ref,
        adapter_contract_ref=compiled.adapter_contract_ref,
        compatibility=compatibility,
        report_semantics=report_semantics,
        core_refs=core_refs,
        subjects=compiled.subjects,
        dimensions=compiled.dimensions,
        cell_bindings=compiled.cell_bindings,
    )
    with pytest.raises(ReportCompilationError, match="score_table_ref"):
        CompiledReportArtifactDTO(
            artifact_ref=ArtifactRef(
                artifact_kind=compiled.artifact_kind, artifact_id=artifact_id
            ),
            adapted_report_ref=compiled.adapted_report_ref,
            adapter_contract_ref=compiled.adapter_contract_ref,
            compatibility_id=compiled.compatibility_id,
            compatibility_schema_id=compiled.compatibility_schema_id,
            missing_value_semantics=compiled.missing_value_semantics,
            report_kind=compiled.report_kind,
            subject_kind=compiled.subject_kind,
            dimension_namespace=compiled.dimension_namespace,
            duplicate_value_semantics=compiled.duplicate_value_semantics,
            report_semantics_id=compiled.report_semantics_id,
            score_table_ref=wrong_kind_score_table_ref,
            layout_recipe_ref=compiled.layout_recipe_ref,
            vpm_artifact_ref=compiled.vpm_artifact_ref,
            subjects=compiled.subjects,
            dimensions=compiled.dimensions,
            cell_bindings=compiled.cell_bindings,
        )


def test_closure_receipt_rejects_incomplete_checks_despite_matching_digest(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    """Finding 3: a receipt claiming only one check ran, with a non-empty
    failure code, must be rejected even though its `receipt_id` correctly
    hashes that (malformed) content - a correct digest proves the bytes
    were not altered relative to themselves, not that the record honors
    the DTO's own promise that every check passed."""
    store = InMemoryArtifactStore()
    compiled = _compile(ai_artifact_family, source_layout_recipe, FakeAdapter, store)
    aggregate = load_compiled_report_aggregate(
        ref=compiled.artifact_ref, resolver=store
    )
    assert aggregate is not None  # sanity: compiled report is genuinely valid

    incomplete_checks = (("compiled_report_valid", True),)
    lying_failure_codes = ("vpm_not_checked",)
    payload = {
        "spec_version": "zeromodel-artifacts-compiled-report-aggregate/v1",
        "compiled_report_ref": {
            "artifact_kind": compiled.artifact_ref.artifact_kind,
            "artifact_id": compiled.artifact_ref.artifact_id,
        },
        "adapted_report_ref": {
            "artifact_kind": compiled.adapted_report_ref.artifact_kind,
            "artifact_id": compiled.adapted_report_ref.artifact_id,
        },
        "adapter_contract_ref": {
            "artifact_kind": compiled.adapter_contract_ref.artifact_kind,
            "artifact_id": compiled.adapter_contract_ref.artifact_id,
        },
        "score_table_ref": {
            "artifact_kind": compiled.score_table_ref.artifact_kind,
            "artifact_id": compiled.score_table_ref.artifact_id,
        },
        "layout_recipe_ref": {
            "artifact_kind": compiled.layout_recipe_ref.artifact_kind,
            "artifact_id": compiled.layout_recipe_ref.artifact_id,
        },
        "vpm_artifact_ref": {
            "artifact_kind": compiled.vpm_artifact_ref.artifact_kind,
            "artifact_id": compiled.vpm_artifact_ref.artifact_id,
        },
        "compatibility_schema_id": compiled.compatibility_schema_id,
        "checks": [list(pair) for pair in incomplete_checks],
        "failure_codes": list(lying_failure_codes),
    }
    lying_receipt_id = sha256_digest(canonical_json_bytes(payload))

    with pytest.raises(ReportCompilationError, match="checks"):
        CompiledReportClosureReceiptDTO(
            receipt_id=lying_receipt_id,
            compiled_report_ref=compiled.artifact_ref,
            adapted_report_ref=compiled.adapted_report_ref,
            adapter_contract_ref=compiled.adapter_contract_ref,
            score_table_ref=compiled.score_table_ref,
            layout_recipe_ref=compiled.layout_recipe_ref,
            vpm_artifact_ref=compiled.vpm_artifact_ref,
            compatibility_schema_id=compiled.compatibility_schema_id,
            checks=incomplete_checks,
            failure_codes=lying_failure_codes,
        )


class _DraftReceipt:
    """A plain attribute bag - not `CompiledReportClosureReceiptDTO` itself
    - used only to compute the canonical digest a shuffled-but-complete
    `checks` tuple would hash to, via the same `closure_receipt_payload`
    function the real DTO uses. Constructing the real DTO with that
    (correct, self-consistent) digest is the actual test."""


def test_closure_receipt_requires_expected_check_names_in_order(
    ai_artifact_family, source_layout_recipe, FakeAdapter
):
    """A receipt whose `checks` tuple is complete and all-True but out of
    order must also be rejected - not merely "the right count of True
    values"."""
    store = InMemoryArtifactStore()
    compiled = _compile(ai_artifact_family, source_layout_recipe, FakeAdapter, store)

    shuffled_checks = tuple((name, True) for name in reversed(_EXPECTED_CHECK_NAMES))
    assert shuffled_checks != tuple((name, True) for name in _EXPECTED_CHECK_NAMES)

    draft = _DraftReceipt()
    draft.compiled_report_ref = compiled.artifact_ref
    draft.adapted_report_ref = compiled.adapted_report_ref
    draft.adapter_contract_ref = compiled.adapter_contract_ref
    draft.score_table_ref = compiled.score_table_ref
    draft.layout_recipe_ref = compiled.layout_recipe_ref
    draft.vpm_artifact_ref = compiled.vpm_artifact_ref
    draft.compatibility_schema_id = compiled.compatibility_schema_id
    draft.checks = shuffled_checks
    draft.failure_codes = ()
    draft.spec_version = "zeromodel-artifacts-compiled-report-aggregate/v1"
    shuffled_receipt_id = sha256_digest(
        canonical_json_bytes(closure_receipt_payload(draft))
    )

    with pytest.raises(ReportCompilationError, match="checks"):
        CompiledReportClosureReceiptDTO(
            receipt_id=shuffled_receipt_id,
            compiled_report_ref=compiled.artifact_ref,
            adapted_report_ref=compiled.adapted_report_ref,
            adapter_contract_ref=compiled.adapter_contract_ref,
            score_table_ref=compiled.score_table_ref,
            layout_recipe_ref=compiled.layout_recipe_ref,
            vpm_artifact_ref=compiled.vpm_artifact_ref,
            compatibility_schema_id=compiled.compatibility_schema_id,
            checks=shuffled_checks,
            failure_codes=(),
        )
