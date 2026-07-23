# Response to the external review of commit e101b06

**Status:** Complete. Superseded on aggregate-closure/report-semantics questions only by [post-c203e7a7-aggregate-closure.md](post-c203e7a7-aggregate-closure.md) (Stage C) — this document's own findings and fixes remain the historical record and are not rewritten.
**Reviewed commit:** `e101b0631baa4a363e7d9d04c108025b0dd250bd` ("feat: add report artifacts and harden trust/navigation").
**Scope:** All six findings from the second external review (2 blockers, 2 high, 2 medium), fixed in full per the user's authorized scope.

## Verdict recap

The review kept commit `e101b06` (no revert), calling the Stage 1A hardening (Navigation/Trust fixes) "real, meaningful closures." It found two blockers and four lesser defects specifically in the new report-artifact compilation/persistence layer (Stage B), and closed with: "the architecture is right, the original security holes are closed, and the remaining defects are concentrated in the new report-artifact persistence layer rather than spread across the system." This document records how each finding was closed.

## Finding 1 (Blocker) — cell bindings used source coordinates, not VPM view coordinates

`_build_matrix_and_bindings()` stamped `row_index`/`column_index` from source-declared order onto `CellBindingDTO`, before `build_vpm()` computed the actual view permutation. A reordering `LayoutRecipe` (e.g. descending sort by score) would silently mismatch: the binding for view row 0 could name the wrong subject.

**Fix** ([report_compiler.py](../../packages/artifacts/src/zeromodel/artifacts/report_compiler.py)): `_build_score_table()` now builds only the matrix and a `(subject_id, dimension_id) -> (index, value)` lookup, with no coordinate assumptions. `build_vpm()` runs next, computing the real `row_order`/`column_order` permutation. `_build_cell_bindings()` then iterates every `(view_row, view_column)` pair and resolves each through `VPMArtifact.cell(view_row, view_column)` — the single authority for view-to-source coordinate translation — filling in both `view_row`/`view_column` and `source_row_index`/`source_metric_index` on [`CellBindingDTO`](../../packages/artifacts/src/zeromodel/artifacts/compiled_artifact.py).

**Regression test:** [test_cell_bindings_use_view_coordinates_not_source_order](../../packages/artifacts/tests/test_report_compilation.py) reproduces the review's exact scenario (descending layout on a two-subject family) and asserts the binding at view_row=0 names the subject that actually sorts first, not the one that happened to load first.

## Finding 2 (Blocker) — the stored artifact could not resolve the actual VPM

`compile_report()` persisted only three digest strings (`score_table_identity`, `layout_recipe_identity`, `vpm_artifact_identity`); the `ScoreTable`/`LayoutRecipe`/`VPMArtifact` objects themselves were never written to the injected `ArtifactStore`. After process restart, "load compiled report → resolve its VPM → render it" was impossible.

**Fix:** New module [core_artifact_persistence.py](../../packages/artifacts/src/zeromodel/artifacts/core_artifact_persistence.py) adds `store_score_table`/`store_layout_recipe`/`store_vpm_artifact` and matching `load_*` functions, using the same decode-and-verify pattern as `report_loading.load_compiled_report_artifact` (resolve canonical bytes → recompute digest → verify against the requested `ArtifactRef.artifact_id` → reconstruct via the type's own `from_dict()`). `CompiledReportArtifactDTO`'s three `*_identity: str` fields became real `score_table_ref`/`layout_recipe_ref`/`vpm_artifact_ref: ArtifactRef` fields (grouped as `CoreArtifactRefs` for identity-payload construction). `compile_report()` now actually calls these store functions. A new convenience function `load_compiled_report_vpm(compiled, resolver=...)` closes the loop end-to-end.

Note: Core's own identity scheme (`ScoreTable.digest`, `VPMArtifact.artifact_id`) is independent of `ArtifactRef.artifact_id` (`sha256_digest(canonical_json_bytes(...))`) — the two digests for the same object are not numerically equal, and that's expected; they're two independent identity layers.

**Regression test:** [test_compile_report_persists_resolvable_core_artifacts](../../packages/artifacts/tests/test_report_compilation.py) round-trips a compiled report through the store, resolves its `VPMArtifact` via `load_compiled_report_vpm`, and verifies every cell binding's coordinates resolve to the same subject/dimension/source-index the compiler declared.

## Finding 3 (High) — no structural closure validator on load

`CompiledReportArtifactDTO.__post_init__` checked artifact kind, non-emptiness, and content digest, but not: unique subject/dimension ids, unique+complete cell-coordinate coverage, index bounds, or that each cell's `subject_id`/`dimension_id` actually matches the subject/dimension at its declared coordinate. A digest proves bytes weren't altered; it doesn't prove they form a valid compiled report.

**Fix:** `_validate_closure()` in [compiled_artifact.py](../../packages/artifacts/src/zeromodel/artifacts/compiled_artifact.py) now checks: unique `subject_id`/`dimension_id` values; exact cell count (`rows × columns`); every `(view_row, view_column)` coordinate covered exactly once (a full bijection onto the grid, not just a matching count); `value_index` values form `{0, ..., n-1}` exactly; every index is in-bounds; and — critically — that `cell.subject_id`/`cell.dimension_id` match the subject/dimension actually declared at `cell.source_row_index`/`cell.source_metric_index`. `CellBindingDTO.__post_init__` additionally cross-checks that `source_binding.subject_id`/`dimension_id` agree with the cell's own.

**Regression tests:** [test_compiled_artifact_validation.py](../../packages/artifacts/tests/test_compiled_artifact_validation.py) — 9 tests constructing `CompiledReportArtifactDTO` directly (bypassing `compile_report`, which always produces valid data) with a genuinely correct digest over deliberately malformed content, proving closure validation — not the digest check — is what catches: wrong cell count, duplicate view coordinates, duplicate value indices, subject/dimension mismatches at a coordinate, out-of-range indices, incomplete coordinate coverage, and duplicate subject ids.

## Finding 4 (High) — manifest immutability was only shallow

`InMemoryArtifactStore.put()` used `MappingProxyType(dict(manifest or {}))`, which only protects the top-level mapping — a nested list or dict inside a manifest value remained the *same object* the caller passed in, so mutating it after `put()` silently changed the stored record.

**Fix:** [store.py](../../packages/artifacts/src/zeromodel/artifacts/store.py) adds `_deep_freeze()`, recursively converting mappings to `MappingProxyType` and sequences to `tuple`. `put()` now freezes before both the conflict-comparison and the stored record, so comparison and storage use the same fully-immutable representation regardless of whether the caller passed lists or tuples.

**Regression test:** [test_store_manifest_is_deeply_immutable_not_just_top_level](../../packages/artifacts/tests/test_artifact_ref_and_store.py) reproduces the review's exact example (`manifest = {"lineage": {"parents": [...]}}`, mutate the nested list after `put()`) and asserts the stored record is unaffected.

## Finding 5 (Medium) — `compatibility_id` didn't establish actual compatibility

`compatibility_id` was an opaque, caller-provided string; two adapters could declare the same one while producing different dimension sets, orderings, score semantics, or value ranges — comparison/search code trusting the string alone could treat structurally incompatible artifacts as compatible.

**Fix:** New module [compatibility_schema.py](../../packages/artifacts/src/zeromodel/artifacts/compatibility_schema.py) adds `compute_compatibility_schema_id()`, a content digest over dimension ids (in declared order), each dimension's score semantics and value/target ranges, and `missing_value_semantics`. `CompiledReportArtifactDTO` gained `compatibility_schema_id`/`missing_value_semantics` fields, self-validated in `__post_init__` by recomputing the digest from the record's own `dimensions` and comparing — the same self-validating pattern used everywhere else in this workspace. Deliberately out of scope: the layout recipe's normalization contract, which is a per-render concern, not a per-report-schema one.

**Regression test:** [test_same_compatibility_id_but_different_dimension_schema_yields_different_schema_id](../../packages/artifacts/tests/test_report_compilation.py) compiles two reports sharing one `compatibility_id` but differing only in one dimension's `score_semantics`, and asserts their `compatibility_schema_id`s differ.

## Finding 6 (Medium) — trust decisions didn't identify their own evidence

`TrustPolicyDTO.policy_id` was already content-derived, but `TrustDecisionDTO` carried only booleans, failure codes, and evaluation time — no way to reconstruct, from the decision alone, which policy/authorization/signer/scope/signature actually produced it.

**Fix:** [dto.py](../../packages/trust/src/zeromodel/trust/dto.py) adds `compute_deployment_scope_id()` (content digest of a `DeploymentScopeDTO`) and extends `TrustDecisionDTO` with `trust_policy_id`, `authorization_id`, `artifact_digest`, `signer_id`, `deployment_scope_id` (all required, sha256-validated where applicable), and `signature_envelope_id` (optional — `None` when no signature was presented). [verify.py](../../packages/trust/src/zeromodel/trust/verify.py)'s `verify_artifact_for_scope()` now populates every field from the objects it already resolved.

**Regression tests:** [test_decision_carries_a_complete_audit_receipt](../../packages/trust/tests/test_verify_artifact_for_scope.py) and [test_decision_without_signature_has_no_signature_envelope_id](../../packages/trust/tests/test_verify_artifact_for_scope.py) assert every evidence field on a real decision matches its source object; [test_dto_and_crypto.py](../../packages/trust/tests/test_dto_and_crypto.py) adds direct `TrustDecisionDTO` construction tests for the new required-field validation.

## Validation performed

- `packages/artifacts/tests`, `packages/trust/tests`, `packages/navigation/tests`: 152 passed (3 pre-existing, environment-only subprocess-isolation failures unrelated to this work — a spawned `sys.executable` subprocess cannot see the editable-installed workspace packages in this environment; reproduced identically before any change in this response).
- `tests/test_public_api_manifest.py` (fast subset): 15 passed. `docs/architecture/package-public-api-1.0.13.csv` regenerated (13 new symbol rows, purely additive) to reflect the new public exports (`CoreArtifactRefs`, `SCORE_TABLE_ARTIFACT_KIND`/`LAYOUT_RECIPE_ARTIFACT_KIND`/`VPM_ARTIFACT_ARTIFACT_KIND`, `store_score_table`/`store_layout_recipe`/`store_vpm_artifact`, `load_score_table`/`load_layout_recipe`/`load_vpm_artifact`/`load_compiled_report_vpm`, `compute_compatibility_schema_id`, `compute_deployment_scope_id`).
- `scripts/check_quality.py`: passed (ruff format, ruff lint, mypy, package boundaries, architecture rules, code-quality limits — two hard-limit violations introduced along the way, an 11-parameter function and a 101-line function, were both fixed by grouping related parameters into a `CompatibilityInfo` NamedTuple and removing one redundant blank line, respectively).
- `python -m mypy packages/artifacts/src packages/trust/src packages/navigation/src`: no issues, 26 source files.

## Files touched

New: `packages/artifacts/src/zeromodel/artifacts/core_artifact_persistence.py`, `packages/artifacts/src/zeromodel/artifacts/compatibility_schema.py`, `packages/artifacts/tests/test_compiled_artifact_validation.py`.

Modified: `packages/artifacts/src/zeromodel/artifacts/{compiled_artifact,report_compiler,report_loading,store,__init__}.py`, `packages/trust/src/zeromodel/trust/{dto,verify,__init__}.py`, plus test files `packages/artifacts/tests/{test_report_compilation,test_artifact_ref_and_store}.py`, `packages/trust/tests/{test_dto_and_crypto,test_verify_artifact_for_scope}.py`, and `docs/architecture/package-public-api-1.0.13.csv`.
