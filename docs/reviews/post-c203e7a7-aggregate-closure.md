# Post-c203e7a7 remediation — Stage C: closed report aggregate, complete report recovery, and auditable trust receipts

**Baseline SHA:** `c203e7a7dddd2f8d3bad1a41e46c18a283c6b163` on `main` ("fix: make compiled report artifacts reloadable and auditable").
**Final working-tree state:** uncommitted — every change below is staged in the working tree, pending an explicit commit decision.
**Objective:** complete the report-artifact architecture so a compiled report can be loaded as one coherent, fully recoverable and auditable aggregate, and give Trust decisions a real content-derived signature-envelope identity instead of the raw signature value.

## 1. Why this stage was required

Stage B (adapter-first report compilation) and its e101b06 review response fixed VPM view-coordinate bindings, Core artifact persistence, compiled-artifact internal closure, deep manifest freezing, metric compatibility schema identity, and Trust decision audit fields. But `AdaptedReportDTO` — the object carrying raw values, per-value confidence, per-value importance, source bindings, and parent-report lineage — still only ever existed as a discarded local Python variable inside `compile_report()`. `CompiledReportArtifactDTO` named it by a bare digest string (`adapted_report_id: str`), not a resolvable reference. And even with all four objects (adapted report, `ScoreTable`, `LayoutRecipe`, `VPMArtifact`) individually persisted and digest-valid, nothing proved they described the *same* report — a self-validating compiled report could reference a `ScoreTable` from report A, a `LayoutRecipe` from report B, and a `VPMArtifact` from report C, every one individually valid.

This stage distinguishes **individual artifact integrity** (a correct digest) from **aggregate semantic closure** (the collection actually describes one report), and closes the gap between them.

## 2. Files added

- `packages/artifacts/src/zeromodel/artifacts/adapted_report_persistence.py` — `ADAPTED_REPORT_ARTIFACT_KIND`, `store_adapted_report`, `load_adapted_report`.
- `packages/artifacts/src/zeromodel/artifacts/report_decode.py` — shared decode helpers (`decode_subject`, `decode_dimension`, `decode_finding_ref`, `decode_source_binding`, `decode_value`, `decode_attributes`) factored out of `report_loading.py` so `adapted_report_persistence.py` reuses the same reconstruction logic rather than a second, driftable copy.
- `packages/artifacts/src/zeromodel/artifacts/aggregate.py` — `ResolvedCompiledReportAggregateDTO`, `load_compiled_report_aggregate`, `validate_compiled_report_aggregate`, `CompiledReportClosureReceiptDTO`, `build_compiled_report_closure_receipt`.
- `packages/artifacts/tests/test_adapted_report_persistence.py` — 8 tests.
- `packages/artifacts/tests/test_compiled_report_aggregate.py` — 19 tests (normal-path + adversarial).
- `packages/trust/tests/test_signature_envelope_revocation.py` — 2 tests, split out of `test_verify_artifact_for_scope.py` to keep that module under the repository's 800-line hard limit.
- This document.

## 3. Files modified

- `packages/artifacts/src/zeromodel/artifacts/compiled_artifact.py` — `CompiledReportArtifactDTO.adapted_report_id: str` → `adapted_report_ref: ArtifactRef` (with a computed `adapted_report_id` property for callers that only need the digest string); five new self-validated report-semantics fields (`report_kind`, `subject_kind`, `dimension_namespace`, `duplicate_value_semantics`, `report_semantics_id`); new `ReportSemanticsInfo` NamedTuple; identity-payload builders (`_identity_payload_fields`, `compiled_report_identity_payload`, `compute_compiled_report_artifact_id`) updated accordingly.
- `packages/artifacts/src/zeromodel/artifacts/compatibility_schema.py` — added `compute_report_semantics_id`.
- `packages/artifacts/src/zeromodel/artifacts/report_compiler.py` — `compile_report()` now persists the adapted report (`store_adapted_report`) before building the score table, and computes/binds `report_semantics_id` from the adapter contract's `subject_kind`/`dimension_namespace`/`duplicate_value_semantics` and the adapted report's own `report_kind`.
- `packages/artifacts/src/zeromodel/artifacts/report_loading.py` — decodes `adapted_report_ref` (an `ArtifactRef`) and the five report-semantics fields; reuses `report_decode.py`'s shared helpers instead of private duplicates.
- `packages/artifacts/src/zeromodel/artifacts/__init__.py` — public API extended (§6).
- `packages/artifacts/tests/test_compiled_artifact_validation.py`, `packages/artifacts/tests/test_report_compilation.py` — updated for the new fields; the latter's `test_cell_bindings_use_view_coordinates_not_source_order` rewritten per §7 below.
- `packages/trust/src/zeromodel/trust/dto.py` — added `signature_envelope_identity_payload`, `compute_signature_envelope_id`; tightened `TrustDecisionDTO.signature_envelope_id` validation to require `sha256:` format when present.
- `packages/trust/src/zeromodel/trust/verify.py` — `_resolve_signature_envelope_id` and `_check_revocations`'s `"signature_envelope"` target now use `compute_signature_envelope_id(...)` instead of the raw `signature_hex`.
- `packages/trust/src/zeromodel/trust/__init__.py` — exports `compute_signature_envelope_id`, `signature_envelope_identity_payload`.
- `packages/trust/tests/test_dto_and_crypto.py`, `packages/trust/tests/test_verify_artifact_for_scope.py` — updated assertions for the content-derived identity; added determinism/uniqueness tests.
- `docs/architecture/adr-artifacts-trust-navigation.md`, `docs/architecture/package-system-next.md`, `docs/reviews/post-e101b06-review-response.md`, `docs/claims-audit.md`, `docs/architecture/package-public-api-1.0.13.csv`, `packages/artifacts/README.md`, `packages/trust/README.md` — documentation (§13).
- `docs/architecture/package-dependency-findings-1.0.13.md`, `docs/architecture/package-import-graph-1.0.13.json`, `docs/architecture/package-inventory-1.0.13.md` — auto-regenerated by the architecture/quality tooling during this session (baseline-commit string only, from `3eaff43c...` to this stage's actual baseline `c203e7a7...`); not hand-edited.

## 4. Public API changes

New exports from `zeromodel.artifacts`:

```
ADAPTED_REPORT_ARTIFACT_KIND, ResolvedCompiledReportAggregateDTO,
CompiledReportClosureReceiptDTO, store_adapted_report, load_adapted_report,
load_compiled_report_aggregate, validate_compiled_report_aggregate,
build_compiled_report_closure_receipt, compute_report_semantics_id
```

`CoreArtifactRefs` was already public (Stage B/e101b06); `ReportSemanticsInfo` and `CompatibilityInfo` remain deliberately unexported implementation detail, reachable via the `compiled_artifact` submodule for tests only — the same restricted-public-API pattern used throughout this workspace.

New exports from `zeromodel.trust`:

```
compute_signature_envelope_id, signature_envelope_identity_payload
```

`docs/architecture/package-public-api-1.0.13.csv` was hand-updated (not regenerated via `scripts/validate_release_candidate.py` — see §12) to add these rows in the file's existing `(distribution, exported_symbol)` sort order; `tests/test_public_api_manifest.py`'s fast subset (15 tests, all passing) verifies every row against the real `__all__` and rejects any manifest/`__all__` mismatch, so a hand-edit that drifted from reality would have failed loudly.

## 5. Artifact-kind and spec-version decisions

`ADAPTED_REPORT_ARTIFACT_KIND = "zeromodel.artifacts.adapted-report/v1"` is a new kind — no existing persisted `v1` payload's identity semantics changed. `CompiledReportArtifactDTO` keeps its existing kind and spec version (`zeromodel.artifacts.compiled-report/v1`) even though its canonical identity payload gained fields (`adapted_report_ref` in place of `adapted_report_id`, plus the five report-semantics fields): this is judged safe without a version bump because **no compiled-report artifact from this kind has ever been durably persisted outside a test process** — `InMemoryArtifactStore` is explicitly non-durable, and this repository has shipped no production consumer of the prior shape. The identity function changing what its digest covers is therefore not "silently reinterpreting an existing persisted `v1` payload" (the case §4.5 of the brief prohibits without a version bump); it is completing a v1 contract that was never actually durable. This decision is recorded here explicitly per that same section's requirement to explain, not silently make, this call.

`zeromodel-artifacts-compiled-report-aggregate/v1` is the new `spec_version` for `CompiledReportClosureReceiptDTO` (`aggregate.py`) — a wholly new artifact kind, not a migration.

## 6. Adapted-report canonical format

The canonical payload stored for an adapted report is exactly `adapted_report_signing_payload(adapted_report)` — the same payload `AdaptedReportDTO.adapted_report_id` is already computed over, with the self-referential id field excluded. Because `ArtifactStore.put()` computes `ref.artifact_id = sha256_digest(canonical_bytes)` over those same canonical bytes, `ref.artifact_id == adapted_report.adapted_report_id` holds unconditionally — the store's content identity and the DTO's own self-validated identity are literally the same digest, not two independently-computed values that happen to agree (this was the brief's §5.1 requirement). `load_adapted_report` follows the established decode-and-verify pattern: resolve canonical bytes → recompute digest → require equality with the requested `ArtifactRef.artifact_id` → decode → reconstruct via `AdaptedReportDTO`'s own `__post_init__` self-validation. The manifest is never read (proven by `test_loading_does_not_use_the_manifest_as_authority`, using the same "manifest-forbidden resolver" pattern established in Stage B).

## 7. Compiled-report identity changes

`CompiledReportArtifactDTO.adapted_report_ref: ArtifactRef` replaces the former `adapted_report_id: str`; a computed `adapted_report_id` property (`self.adapted_report_ref.artifact_id`) preserves the old read-only access pattern for any caller that only wants the digest. The identity payload now binds the adapted report's *full* `ArtifactRef` (kind + id via `_artifact_ref_payload`), not just a digest string — matching how the three Core refs were already bound.

Five new fields close the "report/subject semantics" gap the brief's §9.8 identified: `report_kind`, `subject_kind`, `dimension_namespace`, `duplicate_value_semantics`, and a self-validated `report_semantics_id` digest over those four. Design rationale in §8 below.

**Regression, strengthened per the brief's §13:** `test_cell_bindings_use_view_coordinates_not_source_order` (in `test_report_compilation.py`) previously used `ai_artifact_family`, whose source order already happened to coincide with the sorted (descending) order — `view_row == source_row_index` held for every cell even with the exact bug the test claimed to catch, because the fixture's row 0 already had the highest score. It has been rewritten to use `quality_family` (sentence-001=0.4, sentence-002=0.95 — a genuine inversion under a descending sort) *and* an explicit reversed column order (source declares `quality, clarity`; the recipe requests `clarity, quality`). The test now asserts `view_row != source_row_index` and `view_column != source_metric_index` explicitly for concrete cells, not merely infers correctness from subject names.

## 8. Aggregate object model

```python
@dataclass(frozen=True, slots=True)
class ResolvedCompiledReportAggregateDTO:
    compiled_report: CompiledReportArtifactDTO
    adapted_report: AdaptedReportDTO
    score_table: ScoreTable
    layout_recipe: LayoutRecipe
    vpm_artifact: VPMArtifact
```

`load_compiled_report_aggregate(*, ref, resolver)` resolves `compiled_report` via `load_compiled_report_artifact`, then each of the other four via their own already-established loader (`load_adapted_report`, `load_score_table`, `load_layout_recipe`, `load_vpm_artifact`) — each independently digest-verified and self-validated — then calls `validate_compiled_report_aggregate` before returning. It never returns a partially-valid aggregate: any check failure raises before the function returns, and no failed reference is silently skipped or rebuilt from another object.

## 9. Aggregate-validation algorithm

`validate_compiled_report_aggregate` runs five checks, each a separate function so no single function mixes more than one concern:

1. **`_check_adapted_report_matches_compiled`** — `compiled.adapted_report_ref.artifact_id == adapted.adapted_report_id`; `compiled.adapter_contract_id`, `compiled.compatibility_id`, `compiled.report_kind` all equal the adapted report's own; `compiled.subjects == adapted.subjects` and `compiled.dimensions == adapted.dimensions` (full tuple equality — identity, order, and every field, since these are literal embedded copies on both sides).
2. **`_check_score_table_matches_adapted_report`** — first, and critically, `score_table.metadata["adapted_report_id"] == adapted.adapted_report_id` (see §10 below — this is the check that actually catches identical-matrix substitution). Then `score_table.row_ids`/`metric_ids` equal the adapted report's declared subject/dimension order exactly, and every `score_table.values[i, j]` equals the corresponding `AdaptedValueDTO.raw_value` with exact `float64` equality (no tolerance — the brief is explicit that tolerance-based equality is wrong here, since both were compiled from identical canonical numeric input).
3. **`_check_layout_recipe_matches_vpm`** — the independently loaded `LayoutRecipe`'s digest *and* canonical payload (`to_dict()`) both equal `vpm_artifact.recipe`'s.
4. **`_check_score_table_matches_vpm_source`** — the independently loaded `ScoreTable`'s digest, row/metric ids, and full array values (`np.array_equal`) all equal `vpm_artifact.source`'s.
5. **`_check_cell_bindings_match_vpm_and_values`** — for every `CellBindingDTO`, resolves `vpm_artifact.cell(view_row, view_column)` (Core's sole coordinate-resolution authority) and requires `source_row_index`/`source_metric_index`/`subject_id`(`row_id`)/`dimension_id`(`metric_id`) to agree; then resolves `adapted.values[cell.value_index]` and requires its `subject_id`/`dimension_id`/`source_binding` to match the cell's, and its `raw_value` to equal the VPM cell's `raw_value` — giving `value_index` a real, checked referent instead of an unchecked integer (the brief's §9.7 requirement).

`VPMArtifact`'s own internal identity (source/recipe/permutation/normalized-value bounds/artifact-id/provenance) is not re-validated here — it is already proven by `VPMArtifact.__post_init__` when `load_vpm_artifact` reconstructs it. This module only adds the cross-object checks Core's own authority cannot know about (per the brief's §9.5/§11 instruction not to duplicate Core validation).

Core-identity layering is preserved per §11 of the brief: `ScoreTable.digest`/`VPMArtifact.artifact_id` (Core's own identity scheme) and `ArtifactRef.artifact_id` (the store's content identity) are never required to be numerically equal anywhere in this module — only that an `ArtifactRef` resolves to canonical storage bytes that reconstruct a Core object that independently validates its own Core identity.

## 10. The identical-numeric-matrix gap, found and closed during this stage

Working through adversarial case 14.4 (two adapted reports sharing identical subjects/dimensions/raw values but differing confidence, importance, source finding, and parent lineage) surfaced a real design gap in the first draft of `validate_compiled_report_aggregate`: neither `_check_adapted_report_matches_compiled` (which only proves the compiled report's own ref resolves to *whichever* adapted report was loaded, not that it's the *right* one relative to a substituted `ScoreTable`) nor a naive raw-value comparison (identical by construction in this adversarial case) would have caught the substitution. The fix: `compile_report`'s `_build_score_table` already stamps `metadata={"adapted_report_id": adapted.adapted_report_id}` onto the `ScoreTable` it builds; `_check_score_table_matches_adapted_report` now requires that stamped id to equal the *resolved* adapted report's own id — directly satisfying the brief's §9.1/§9.2 requirement ("Require ScoreTable metadata to bind the adapted report identity") and turning an adversarial test that would otherwise have silently passed into a real regression guard (`test_adapted_report_substitution_with_identical_matrix_is_rejected`).

## 11. Compatibility identity design

**Design B** (separate identities) was chosen per the brief's own two offered options, over Design A (extend the existing `compatibility_schema_id`). `compute_compatibility_schema_id` (dimension ids/order, score semantics, value/target ranges, missing-value semantics) is **unchanged** — its `v1` semantics are not reinterpreted, so no version bump is needed there. A new, additive `compute_report_semantics_id(*, report_kind, subject_kind, dimension_namespace, duplicate_value_semantics)` closes the layer the dimension-only digest cannot: two reports over sentences and claims with an identical dimension schema now produce different `report_semantics_id`s even though `compatibility_id` and `compatibility_schema_id` could coincide. Full compatibility between two compiled reports now requires all three to agree: the human label (`compatibility_id`), the dimension schema digest (`compatibility_schema_id`), and the report/subject schema digest (`report_semantics_id`). `report_kind`/`subject_kind`/`dimension_namespace`/`duplicate_value_semantics` are stored as literal fields on `CompiledReportArtifactDTO` (not merely referenced) so `report_semantics_id` can be self-validated in `__post_init__` the same way `compatibility_schema_id` already is — recomputed from the record's own fields, not trusted from an external contract that isn't part of the aggregate.

## 12. Signature-envelope identity design

**"Compatible design"** was chosen (the brief's own second offered option) over adding a self-validating `signature_envelope_id` field directly to `SignatureEnvelopeDTO`: the preferred design would require every existing construction site (test fixtures across two packages, the example scripts, `verify.py`'s callers) to be touched to supply a correctly-computed id, for a DTO that is never itself persisted through an `ArtifactStore` — it's constructed dynamically at verification time from a raw `sign_digest()` output. Instead, `compute_signature_envelope_id(*, authorization_id, signer_id, signature_hex, key_algorithm, spec_version)` computes the identity centrally; `verify.py`'s `_resolve_signature_envelope_id` calls it once and the result feeds both `TrustDecisionDTO.signature_envelope_id` (the audit-receipt field) and `_check_revocations`'s `"signature_envelope"` target — the same function, so the two can never silently disagree on what "this envelope's identity" means. `TrustDecisionDTO.signature_envelope_id`'s validation was tightened from "non-empty string" to `_require_sha256` (matching every other evidence field on that DTO) when present.

## 13. Revocation migration behavior

Before this stage, `RevocationRecordDTO(target_kind="signature_envelope", target_id=...)` was checked against the raw `signature_hex` — an already-existing but completely untested code path (`TrustFailureCode.REVOKED_ENVELOPE` had zero test coverage before this stage; grep of the pre-stage test suite confirms it). After this stage, the same `target_kind="signature_envelope"` now targets `compute_signature_envelope_id(...)`'s output instead. This is a genuine behavior change for that one target kind, not a new target kind added alongside the old one (per the brief's explicit instruction not to silently support two meanings for the same target kind) — a revocation record authored against the old raw-hex value will no longer match. `test_revoking_raw_signature_hex_does_not_revoke_the_envelope` proves this migration is real (not merely additive) by constructing exactly that stale record and asserting the artifact is still authorized.

## 14. Tests added

- `packages/artifacts/tests/test_adapted_report_persistence.py` (8): ref-equals-id, idempotent identical persist, full round-trip, confidence/importance round-trip, parent-ids/attributes round-trip, wrong-kind fails closed, wrong-digest fails closed, manifest-is-not-authority (two-sided).
- `packages/artifacts/tests/test_compiled_report_aggregate.py` (19):
  - Normal path (8): aggregate loader resolves all five objects; succeeds after original Python objects discarded; closure succeeds for `compile_report` output; closure is deterministic; VPM cells map to correctly-bound adapted values; confidence/importance/parent-ids/attributes survive compilation-and-reload end-to-end; compatibility-schema-id determinism; closure-receipt records every check passing.
  - Adversarial (10, brief §14.1–§14.6): foreign VPM substitution rejected; foreign `ScoreTable` substitution rejected; foreign `LayoutRecipe` substitution rejected (including a recipe engineered to produce the *same* row/column ordering for this fixture, proving the aggregate binds the actual recipe object, not merely observed output); identical-numeric-matrix adapted-report substitution rejected (§10 above); wrong `value_index` mapping rejected (via a full two-cell swap so the DTO's own bijection check cannot be what catches it); closure receipt not generated for a failed aggregate (fail-closed, no partial receipt); different subject_kind/report_kind/dimension_namespace each yield a different `report_semantics_id`; report-semantics-id determinism; end-to-end proof that two compiled reports sharing `compatibility_id` and `compatibility_schema_id` but differing `subject_kind` get different compiled-artifact identities.
- `packages/trust/tests/test_dto_and_crypto.py` (+3): `TrustDecisionDTO.signature_envelope_id` requires sha256 format when present (rejects both the old raw-hex example and bare hex without a prefix); envelope-id determinism; envelope-id changes with any of authorization_id/signer_id/signature_hex/key_algorithm (brief §14.7).
- `packages/trust/tests/test_signature_envelope_revocation.py` (2, split from `test_verify_artifact_for_scope.py` for the line-count hard limit): revoked envelope (by computed id) is rejected; revoking the *raw* signature hex does not revoke the envelope (proves the migration, §13).
- Existing `test_report_compilation.py::test_cell_bindings_use_view_coordinates_not_source_order` rewritten (§7) rather than added alongside — it is the same regression test the brief asked to be strengthened.

## 15. Validation run this session

```
python -m pytest -q packages/artifacts/tests                 # 88 passed
python -m pytest -q packages/trust/tests                     # 51 passed
python -m ruff format --check packages/artifacts packages/trust   # passed (after one auto-format pass)
python -m ruff check packages/artifacts packages/trust            # passed (after one auto-fix pass: unused/duplicate import)
python -m mypy packages/artifacts/src packages/trust/src           # Success: no issues found in 22 source files
python scripts/check_package_boundaries.py                         # passed: 141 production modules
python scripts/check_architecture.py                                # passed: 141 production modules inspected
python scripts/check_quality.py                                     # Quality checks passed (all governed packages)
python scripts/run_fast_tests.py                                    # 975 passed, 1 skipped, 0 failed, 69.58s / 120s budget
python -m pytest -q tests/test_public_api_manifest.py -k "not slow" # 15 passed
```

Two hard-limit violations were introduced along the way and both fixed within this session, per the repository's stated policy of not silently exceeding quality gates:

- `verify_artifact_for_scope` briefly grew to 101 lines (limit 100) after adding the envelope-id resolution call; fixed by inlining one duplicate (cheap, pure) call to `_resolve_signature_envelope_id` instead of holding it in a named local twice.
- `test_verify_artifact_for_scope.py` grew to 868 lines (limit 800) after adding two revocation regression tests; fixed by moving those two tests into the new `test_signature_envelope_revocation.py`.

## 16. Explicitly not run without further authorization

- `scripts/validate_release_candidate.py` (the "release validation" gate) — the brief's own §22 separates this from the standard production gates and requires it be run "only if permitted by the repository's current policy," distinct from the always-run fast/quality gates above. It was not run this session; the public API manifest CSV was instead hand-updated to match the new `__all__` exports (§4), and its correctness is independently verified by the (non-slow) `tests/test_public_api_manifest.py` subset, which passed. The one test in that file requiring the full release-validation pipeline (`test_manifest_generation_is_byte_identical_across_runs`) is itself marked `@pytest.mark.slow` and was not run for the same reason.
- Any `@pytest.mark.slow`/`external`/`research` test — none added, none run.
- Building/publishing to TestPyPI or PyPI.
- Any git commit, push, tag, or branch operation — the working tree remains uncommitted pending explicit instruction, per the brief's own instructions.

## 17. Known limitations (claims boundary)

**Supported claim:** ZeroModel can persist and reload a complete adapted report and its compiled `ScoreTable`, `LayoutRecipe`, and `VPMArtifact` as one content-addressed aggregate, and can verify that every subject, dimension, value, coordinate, source binding, and compatibility contract agrees across all representations. Trust decisions identify the exact signature envelope used through a content-derived envelope identity, and signature-envelope revocation targets that same identity.

**Not supported / explicitly out of scope**, per the brief's own non-goals and this implementation's actual reach:

- `zeromodel.search` — not started; this stage's aggregate-closure guarantee is exactly the prerequisite the ADR names before Search can begin, not Search itself.
- The `ReportAdapterContractDTO` (the adapter contract) is still not persisted as a resolvable artifact — `report_semantics_id` binds its `subject_kind`/`dimension_namespace`/`duplicate_value_semantics` values directly onto `CompiledReportArtifactDTO` rather than via a fifth aggregate member; the brief's required aggregate is exactly the four objects listed (adapted report, `ScoreTable`, `LayoutRecipe`, `VPMArtifact`), which this satisfies, but a future stage could still add contract persistence as a fifth reference if adapter-versioning audit trails become a requirement.
- Sparse reports (`missing_value_semantics="absent"`) remain accepted as a declared contract value but not actually compilable — unchanged from Stage B, still requires a future sparse `ScoreTable`/VPM representation.
- No distributed transactions, no durable/remote store, no automatic artifact repair, no PNG rendering, no attention/priority projection layer — all unchanged non-goals from the original brief.
- Trust-signed compiled reports remain a call-site composition (Artifacts still does not depend on Trust); not implemented or claimed here.

## 18. Next safe architectural stage

Per the brief's own instruction, this report does not generate the next implementation prompt. The brief itself and the ADR both note that Search is the likely later consumer, contingent on Stage 1A, Stage B, and this stage (Stage C) all being complete — which they now are.
