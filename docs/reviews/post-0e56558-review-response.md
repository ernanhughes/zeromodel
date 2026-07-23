# Response to the external review of commit 0e56558

**Status:** Complete.
**Reviewed commit:** `0e56558fad7ba3d2a74ab39d35c7d4ae14357acc` ("feat: close compiled report aggregates and trust receipts").
**Scope:** All four findings from the external review (1 blocker, 3 high), fixed in full.

## Verdict recap

The review kept commit `0e56558` ("Keep the commit and fix forward"), confirming the adapted report is genuinely persistent, the aggregate loader is real, the coordinate regression test performs actual row/column permutations, and the trust-envelope identity is real. It found one blocker (a valid aggregate could embed a fabricated visual map) and three high-priority gaps (the adapter contract was unrecoverable, the public validator/receipt builder trusted unproven caller-supplied objects, and the signature-envelope identity silently dropped a non-default `spec_version`). This document records how each was closed.

## Finding 1 (Blocker) — a valid aggregate could contain a fabricated visual map

Every existing aggregate check proved the VPM *embedded* the correct `ScoreTable`/`LayoutRecipe` and that cell bindings mapped to the correct *raw* values. None of them recomputed the expected `normalized_values`/`row_order`/`column_order` from that `ScoreTable`/`LayoutRecipe` — `VPMArtifact.validate()` only checks that the matrix is finite, in `[0, 1]`, and self-hashes to its own declared `artifact_id`. A VPM with the correct source/recipe/raw-value mappings but arbitrary pixel intensities (or an arbitrary-but-valid permutation) could therefore be re-digested into a new, fully self-consistent artifact and referenced by an otherwise-legitimate compiled report.

**Fix** ([aggregate.py](../../packages/artifacts/src/zeromodel/artifacts/aggregate.py)): a new `_check_vpm_matches_deterministic_reconstruction` rebuilds the VPM `build_vpm()` would deterministically produce from the already-verified `score_table`/`layout_recipe`, using the same provenance `compile_report()` stamps (itself derived from already-cross-validated aggregate fields, never trusted from the loaded VPM), and requires it to match the loaded artifact field-by-field (`row_order`, `column_order`, `normalized_values` via `np.array_equal`, `provenance`) and by full Core `artifact_id` as defense in depth. This runs as part of `validate_compiled_report_aggregate`, so both `load_compiled_report_aggregate` and `build_compiled_report_closure_receipt` are protected.

**Regression tests:** [test_vpm_with_fabricated_normalized_pixels_is_rejected](../../packages/artifacts/tests/test_compiled_report_aggregate_adversarial.py) constructs a VPM sharing the real `source`/`recipe`/`row_order`/`column_order`/`provenance` but with an all-zero `normalized_values` matrix (self-consistent — a different matrix legitimately produces a different Core `artifact_id`), stores it, references it from an otherwise-genuine compiled report, and asserts aggregate loading raises specifically on `normalized_values`. [test_vpm_with_fabricated_row_order_is_rejected](../../packages/artifacts/tests/test_compiled_report_aggregate_adversarial.py) does the analogous check for a reversed-but-still-valid row permutation. Both are distinct from the pre-existing "foreign VPM from an unrelated report" test — here every other field is exactly right; only the visual is a lie.

## Finding 2 (High) — the adapter contract remained unrecoverable

`CompiledReportArtifactDTO` and `AdaptedReportDTO` both repeated the same `adapter_contract_id` string, and aggregate validation checked that they agreed with each other — but the contract itself was never persisted, so nothing could resolve that id and prove the copied `subject_kind`/`dimension_namespace`/`missing_value_semantics`/`duplicate_value_semantics` actually came from a real contract declaring those values.

**Fix:** new module [report_adapter_contract_persistence.py](../../packages/artifacts/src/zeromodel/artifacts/report_adapter_contract_persistence.py) adds `REPORT_ADAPTER_CONTRACT_ARTIFACT_KIND`, `store_report_adapter_contract`, `load_report_adapter_contract`, using the same decode-and-verify pattern as every other artifact kind in this package. `CompiledReportArtifactDTO.adapter_contract_id: str` became `adapter_contract_ref: ArtifactRef` (with a computed `adapter_contract_id` property preserving the old read-only access pattern). `compile_report()` now persists the contract before compiling. `ResolvedCompiledReportAggregateDTO` gained a fifth member, `adapter_contract: ReportAdapterContractDTO`, resolved by `load_compiled_report_aggregate`. A new check, `_check_adapter_contract_matches_compiled_and_adapted`, reconciles every field the contract declares (`report_kind`, `subject_kind`, `dimension_namespace`, `compatibility_id`, `missing_value_semantics`, `duplicate_value_semantics`) against both the compiled and adapted reports.

**Regression test:** [test_subject_kind_disagreeing_with_resolved_adapter_contract_is_rejected](../../packages/artifacts/tests/test_compiled_report_aggregate_adversarial.py) builds a compiled report declaring `subject_kind="claim"` while its `adapter_contract_ref` points at a real, correctly-persisted contract that actually declares `subject_kind="sentence"` — `report_semantics_id` is recomputed so the record stays internally self-consistent (proving `CompiledReportArtifactDTO.__post_init__`'s own recompute-and-compare check does *not* catch the lie), and asserts only resolving and reconciling the real contract does.

## Finding 3 (High) — the public validator/receipt builder trusted unproven objects

Every aggregate check compared the *resolved objects* to each other, never to the `ArtifactRef`s `compiled_report` actually declares. A caller that bypassed `load_compiled_report_aggregate` and constructed `ResolvedCompiledReportAggregateDTO` directly from locally-coherent-but-store-unrelated objects (e.g. a `score_table` resolved from an entirely different ref, left pointing at `compiled_report`'s own unrelated `score_table_ref`) could pass every cross-object check, and `build_compiled_report_closure_receipt` would then issue a receipt claiming the compiled report's *declared* refs were verified.

**Fix:** a new check, `_check_resolved_objects_match_declared_refs`, now runs first in `validate_compiled_report_aggregate`. `AdaptedReportDTO`/`ReportAdapterContractDTO` already self-validate their own id against their own content in `__post_init__`, so comparing their id fields to the declared refs is sufficient. `ScoreTable`/`LayoutRecipe`/`VPMArtifact` have no such self-validating store-level id (Core's own `.digest`/`.artifact_id` is a distinct identity layer), so their store-level canonical digest is recomputed directly — the same computation `store_score_table`/`store_layout_recipe`/`store_vpm_artifact` used to mint the ref in the first place — and compared to `compiled_report`'s declared ref.

**Regression tests:** [test_validate_rejects_a_score_table_not_proven_to_come_from_the_declared_ref](../../packages/artifacts/tests/test_compiled_report_aggregate_adversarial.py) swaps a foreign-but-locally-coherent `score_table` into an otherwise-genuine aggregate and asserts `validate_compiled_report_aggregate` rejects it directly. [test_receipt_builder_rejects_an_unproven_caller_assembled_aggregate](../../packages/artifacts/tests/test_compiled_report_aggregate_adversarial.py) does the same through the public receipt builder, using a genuinely different (not merely content-coincidentally-identical) foreign `layout_recipe`.

## Finding 4 (High) — signature-envelope identity ignored the envelope's actual `spec_version`

`compute_signature_envelope_id`'s payload already included `spec_version`, but `_resolve_signature_envelope_id()` never passed `signature_envelope.spec_version` through, silently defaulting to the function's current version regardless of what the envelope actually declared. For a hypothetical non-default envelope version, both `TrustDecisionDTO.signature_envelope_id` and revocation checks would compute the wrong identity, and a correctly generated revocation for the real versioned envelope would silently fail to match.

**Fix:** [verify.py](../../packages/trust/src/zeromodel/trust/verify.py)'s `_resolve_signature_envelope_id` now passes `spec_version=signature_envelope.spec_version` explicitly. Additionally — since the review noted `SignatureEnvelopeDTO` accepted any caller-provided `spec_version` with no validation — [dto.py](../../packages/trust/src/zeromodel/trust/dto.py)'s `SignatureEnvelopeDTO.__post_init__` now rejects any `spec_version` other than the current supported one, per the review's suggested remediation ("reject unsupported envelope versions... unless multi-version verification is deliberately implemented" — it is not). Both fixes are kept: the DTO-level rejection closes the gap for real callers; the explicit pass-through keeps `compute_signature_envelope_id` itself correct as a general-purpose function independent of that DTO-level restriction.

**Regression tests:** [test_signature_envelope_rejects_unsupported_spec_version](../../packages/trust/tests/test_dto_and_crypto.py) asserts construction fails for a non-default `spec_version`. `test_signature_envelope_id_changes_with_any_component` (extended) now includes a `spec_version` variant in its uniqueness set.

## Validation performed

```
python -m pytest -q packages/artifacts/tests                      # 93 passed
python -m pytest -q packages/trust/tests                          # 52 passed
python -m pytest -q tests/test_public_api_manifest.py -k "not slow"  # 15 passed
python -m ruff format --check packages/artifacts packages/trust   # passed
python -m ruff check packages/artifacts packages/trust            # passed
python -m mypy packages/artifacts/src packages/trust/src          # Success: no issues found in 23 source files
python scripts/check_package_boundaries.py                        # passed: 142 production modules
python scripts/check_architecture.py                              # passed: 142 production modules inspected
python scripts/check_quality.py                                   # Quality checks passed
python scripts/run_fast_tests.py                                  # 981 passed, 1 skipped, 0 failed, 96.11s / 120s budget
python examples/writer_report_adapters_demo.py                    # runs; two distinct compiled artifact ids for identical subjects
```

One hard-limit violation was introduced along the way and fixed within this session: `test_compiled_report_aggregate.py` grew to 1010 lines (limit 800) after adding the new adversarial tests; fixed by splitting the adversarial tests (Stage C section 14 plus all four review findings above) into a new [test_compiled_report_aggregate_adversarial.py](../../packages/artifacts/tests/test_compiled_report_aggregate_adversarial.py), leaving the original file with only normal-path and receipt tests.

## Files added

- `packages/artifacts/src/zeromodel/artifacts/report_adapter_contract_persistence.py`
- `packages/artifacts/tests/test_compiled_report_aggregate_adversarial.py`
- This document.

## Files modified

- `packages/artifacts/src/zeromodel/artifacts/aggregate.py` — five-object aggregate (`adapter_contract` added), `_check_resolved_objects_match_declared_refs`, `_check_adapter_contract_matches_compiled_and_adapted`, `_check_vpm_matches_deterministic_reconstruction`; receipt DTO/functions gained `adapter_contract_ref`.
- `packages/artifacts/src/zeromodel/artifacts/compiled_artifact.py` — `adapter_contract_id: str` → `adapter_contract_ref: ArtifactRef` (+ computed property); identity-payload builders updated.
- `packages/artifacts/src/zeromodel/artifacts/report_compiler.py`, `report_loading.py`, `__init__.py` — wire in contract persistence and the new ref field.
- `packages/trust/src/zeromodel/trust/dto.py` — `SignatureEnvelopeDTO` rejects unsupported `spec_version`.
- `packages/trust/src/zeromodel/trust/verify.py` — `_resolve_signature_envelope_id` passes `spec_version` through explicitly.
- `packages/artifacts/tests/test_compiled_artifact_validation.py`, `test_compiled_report_aggregate.py` — updated for the new `adapter_contract_ref` field; normal-path tests split from adversarial.
- `packages/trust/tests/test_dto_and_crypto.py` — new/extended tests for finding 4.
- `docs/architecture/package-public-api-1.0.13.csv` — three new rows (`REPORT_ADAPTER_CONTRACT_ARTIFACT_KIND`, `load_report_adapter_contract`, `store_report_adapter_contract`).

## Known limitations (unchanged from Stage C)

The claims boundary recorded in [post-c203e7a7-aggregate-closure.md](post-c203e7a7-aggregate-closure.md) still applies. This response closes gaps in *how thoroughly* that stage's own claim is proven; it does not expand the claim itself. `zeromodel.search` remains not started.
