# Post-split remediation — Stage B: adapter-first report compilation on `zeromodel.artifacts`

**Baseline SHA:** `3eaff43cadb54e2d2fb908aab8b7c0eaef5623b7` on `main` (the merged commit carrying the Artifacts/Trust/Navigation kernels), with Stage 1A hardening (see [post-split-stage-1a-hardening.md](post-split-stage-1a-hardening.md)) already applied in the working tree.
**Final working-tree state:** uncommitted — every change below is staged in the working tree, pending an explicit commit decision.
**Objective:** extend the existing `zeromodel.artifacts` storage kernel with an adapter-first capability so external applications can compile typed domain reports (a Writer AI-artifact report, a claim-evidence report, ...) into deterministic, source-bound VPM artifacts, without ZeroModel ever gaining domain-specific knowledge of what those reports mean.

## Files added

- `packages/artifacts/src/zeromodel/artifacts/score_semantics.py` — `ScoreSemantics` enum (`higher_is_better` / `higher_is_worse` / `target_range` / `descriptive`).
- `packages/artifacts/src/zeromodel/artifacts/report_errors.py` — `ReportAdaptationError`, `ReportCompilationError` (both `VPMValidationError` subclasses, following the existing `HierarchyCompilationError`/`HierarchyClosureError` convention from Navigation).
- `packages/artifacts/src/zeromodel/artifacts/report_dto.py` — `ReportAdapterContractDTO`, `AdaptedSubjectDTO`, `AdaptedDimensionDTO`, `ReportFindingRefDTO`, `SourceBindingDTO`, `AdaptedValueDTO`, `AdaptedReportDTO`, plus `_pairs_to_dict` (duplicate-key-rejecting tuple→dict, matching the convention established in Navigation's `dto.py`).
- `packages/artifacts/src/zeromodel/artifacts/adapter.py` — the `ReportAdapter[ReportT]` protocol (contravariant on `ReportT`, since it only appears as an `adapt()` input).
- `packages/artifacts/src/zeromodel/artifacts/compiled_artifact.py` — `CellBindingDTO`, `CompiledReportArtifactDTO`, `CoreArtifactIdentities` (a small `NamedTuple` bundling the three Core-owned digests, introduced to keep the identity-payload builder functions at or under the repository's 10-parameter hard limit).
- `packages/artifacts/src/zeromodel/artifacts/report_compiler.py` — `compile_report()`.
- `packages/artifacts/src/zeromodel/artifacts/report_loading.py` — `load_compiled_report_artifact()`.
- `examples/writer_report_adapters_demo.py` — an illustrative, synthetic external-adapter example (an `AIArtifactReportAdapter` and a `SentenceQualityReportAdapter`, both explicitly documented as belonging in the external application's own codebase in real use, not in `zeromodel.artifacts`).
- Seven new test files under `packages/artifacts/tests/`: `conftest.py` (fixtures), `test_report_adapter_contract.py`, `test_adapted_report_identity.py`, `test_report_compilation.py`, `test_report_families.py` — 30 new tests.
- `docs/architecture/adr-artifacts-trust-navigation.md`, `docs/architecture/package-system-next.md`, `docs/reviews/post-split-stage-1a-hardening.md` (Stage 1A deliverables, described in that document).

## Files modified

- `packages/artifacts/src/zeromodel/artifacts/__init__.py` — public API extended (see below).
- `docs/architecture/package-system-1.0.13.md` — marked historical, content preserved (Stage 1A).
- `packages/artifacts/src/zeromodel/artifacts/store.py`, and the Trust/Navigation source/test files listed in [post-split-stage-1a-hardening.md](post-split-stage-1a-hardening.md) — Stage 1A hardening, not part of this stage's own scope but present in the same working tree.

## Public exports (this stage's additions to `zeromodel.artifacts.__all__`)

```
ReportAdapter, ReportAdapterContractDTO, AdaptedReportDTO, AdaptedSubjectDTO,
AdaptedDimensionDTO, AdaptedValueDTO, ReportFindingRefDTO, SourceBindingDTO,
CellBindingDTO, CompiledReportArtifactDTO, ScoreSemantics,
compile_report, load_compiled_report_artifact,
ReportAdaptationError, ReportCompilationError
```

`CoreArtifactIdentities` and the internal `_pairs_to_dict`/payload-builder helpers are deliberately *not* exported — they're implementation detail, reachable via `zeromodel.artifacts.compiled_artifact`/`report_dto` submodules for tests only, matching the restricted-public-API pattern already established for Navigation.

## Dependency edges

Unchanged: `zeromodel-artifacts → zeromodel` (core) only. `compile_report` reuses `zeromodel.core.artifact`'s `ScoreTable`, `LayoutRecipe`, and `build_vpm` directly rather than reimplementing VPM construction — no new dependency was introduced (the package already depended on `numpy`, which is all `report_compiler.py` additionally needs).

## Adapter contract design

`ReportAdapterContractDTO.contract_id` is a self-validating content digest (same pattern as `ArtifactRef.artifact_id`, `ArtifactAuthorizationDTO.authorization_id`, `NavigationTileDTO.tile_id`) over `adapter_id`, `adapter_version`, `report_kind`, `subject_kind`, `dimension_namespace`, `compatibility_id`, `missing_value_semantics`, `duplicate_value_semantics`. `missing_value_semantics` is restricted to `{"error", "absent"}`; `duplicate_value_semantics` is currently restricted to `{"reject"}` only — the brief's required validation rule ("do not silently choose the last duplicate value") makes any other value unsound, so no other value is accepted yet.

## Identity hierarchy

```
adapter contract_id (adapter identity + policy)
    ↓
adapted_report_id (AdaptedReportDTO: subjects, dimensions, values, bindings)
    ↓ + layout_recipe + Core's ScoreTable/VPMArtifact digests
compiled report artifact_id (CompiledReportArtifactDTO.artifact_ref.artifact_id)
```

Each level's identity payload excludes only its own id field and is built by a shared `_fields`-style function reused by both the DTO's own `__post_init__` self-check and a `compute_*` authoring helper — the same pattern used throughout Trust and Navigation. Changing a raw value, a score semantic, a source binding, confidence, importance, a parent relationship, or the layout recipe actually used all change the compiled artifact's identity; changing only a *projection* (not implemented this stage) must not.

## Score-semantics implementation

`ScoreSemantics` is a plain declared enum on `AdaptedDimensionDTO` — never inferred from a dimension's name or family. `TARGET_RANGE` requires `target_min`/`target_max` to both be set with `target_min <= target_max`; other semantics do not require bounds. `compile_report` and `CompiledReportArtifactDTO` both preserve the declared semantics verbatim alongside the raw, un-transformed value — no attention/priority/normalization-by-polarity transformation exists anywhere in this stage (see "Known limitations" below).

## Source-binding implementation

Every `AdaptedValueDTO` carries a `SourceBindingDTO` (subject_id, dimension_id, a `ReportFindingRefDTO`, optional URI/character-offset span, optional attributes); `AdaptedValueDTO.__post_init__` requires the binding's own `subject_id`/`dimension_id` to match the value's, so a binding can never silently point at the wrong cell. `compile_report` builds one `CellBindingDTO` per matrix cell carrying that exact `source_binding` forward — `test_every_value_maps_to_exactly_one_cell_binding_with_correct_source` proves the round trip cell-by-cell.

## Compilation flow

`compile_report(*, adapter, report, layout_recipe, store)`:
1. `contract = adapter.contract()`; `adapted = adapter.adapt(report)`.
2. Validate `adapted.adapter_contract_id`/`report_kind`/`compatibility_id` against `contract` (an adapter cannot silently adapt into a different contract's identity space).
3. Build the dense numeric matrix (subjects × dimensions, in each side's declared order) and one `CellBindingDTO` per cell; a missing `(subject, dimension)` pair raises `ReportCompilationError` under `missing_value_semantics="error"`, or raises with an explicit "not yet supported" message under `"absent"` (no sparse VPM representation exists yet — see limitations).
4. Build `zeromodel.core.artifact.ScoreTable` from that matrix, then `build_vpm(score_table, layout_recipe, provenance=...)` for the `VPMArtifact` — reusing Core's existing normalization/layout machinery unchanged.
5. Compute the compiled artifact's identity from `adapted_report_id`, `adapter_contract_id`, `compatibility_id`, the three Core digests, `subjects`, `dimensions`, `cell_bindings`.
6. Store the identity payload's canonical bytes through the injected `ArtifactStore` with `manifest=None` — the compiled artifact carries no manifest at all; every field needed to reconstruct it lives in canonical bytes.
7. Return the assembled, self-validating `CompiledReportArtifactDTO`.

Does not render an image and does not compute any attention/priority projection — both are explicitly out of scope for this stage (the brief's own Phase 6 marks projections optional and later).

## Canonical storage format

`load_compiled_report_artifact` follows exactly the decode-and-verify pattern Stage 1A established for `zeromodel.navigation.storage.load_tile`: resolve canonical bytes, recompute the digest, require it to equal the requested `ArtifactRef.artifact_id`, decode the canonical JSON payload, and reconstruct every nested DTO (subjects, dimensions with `ScoreSemantics` re-parsed, cell bindings with nested source bindings and finding refs) directly from that payload — the manifest is never read. `test_loading_does_not_use_the_manifest_as_authority` proves this two ways: attempting to `put()` a different manifest under the same identity is rejected by the Stage-1A-hardened store itself (`ArtifactManifestConflictError`), and a resolver whose `resolve_manifest()` raises `AssertionError` if called still loads successfully.

## Tests added

34 new tests across `packages/artifacts/tests/` (bringing the package's full suite from 14 to 48, all passing):

- `test_report_adapter_contract.py` (6): identity determinism, version-change changes identity, empty adapter_id rejected, fabricated contract_id rejected, invalid `missing_value_semantics`/`duplicate_value_semantics` rejected.
- `test_adapted_report_identity.py` (16): identity determinism, attribute-tuple-order independence, changed raw value/score-semantics/source-binding/confidence/importance all change identity, duplicate subject/dimension ids rejected, duplicate value pair rejected, unknown subject/dimension reference rejected, non-finite raw value rejected, inverted/missing target-range bounds rejected, source-binding subject mismatch rejected.
- `test_report_compilation.py` (8): identical input → identical compiled identity, changed layout → changed identity, every value maps to exactly one correctly-bound cell, round-trip through the store, manifest-is-not-authoritative (two-sided), wrong artifact_kind/wrong digest fail closed on load, missing value with `error` semantics fails compilation.
- `test_report_families.py` (4): AI-artifact and quality report families stay separate over identical subjects, higher-is-worse/higher-is-better remain distinguishable post-compilation, a claim-evidence report mixes polarities correctly within one report, a descriptive dimension is preserved without any automatic transformation.
- `test_artifacts_api_isolation.py` — extended with Stage-B-specific deliberateness assertions (the restricted names above are present; no domain-specific adapter class ever appears in `__all__`).

A real, session-discovered defect was found and fixed while writing this suite: running the full workspace fast suite (not just `packages/artifacts/tests` in isolation) surfaced a bare `conftest` module-name collision — several test files across `packages/navigation/tests` and `packages/artifacts/tests` used `from conftest import <helper>`, which works when a package's tests are collected alone but silently shadows across packages once both are collected in the same pytest session (as `scripts/run_fast_tests.py` always does), because Python caches the first-loaded module under the bare name `conftest` in `sys.modules`. Fixed by converting every shared plain-function helper (`make_source_artifacts`, `make_value`, `make_contract`, `make_adapted_report`, `FakeAdapter`) into a proper pytest fixture factory requested by parameter injection instead of a module import — the idiomatic, collision-free mechanism `conftest.py` is designed for. Confirmed fixed by a full `scripts/run_fast_tests.py` run.

## Validation run this session

```
python -m pytest -q packages/artifacts/tests                      # 48 passed
python -m mypy packages/artifacts/src                               # Success: no issues found
python scripts/check_quality.py                                     # Quality checks passed (all 9 packages)
python scripts/validate_release_candidate.py                        # Release candidate validation passed; 276 public symbols across 9 distributions
python scripts/run_fast_tests.py                                    # 926 passed, 1 skipped, 0 failed, 85.88s / 120s budget
python examples/writer_report_adapters_demo.py                      # runs; prints two distinct compiled artifact ids for identical subjects
```

## Explicitly not run without further authorization

- Any `@pytest.mark.slow`/`external`/`research` test — none added, none run.
- Building/publishing to TestPyPI or PyPI.
- Any git commit, push, tag, or branch operation — the working tree remains uncommitted pending explicit instruction.

## Known limitations (claims boundary)

Supported claim: ZeroModel can adapt a typed multidimensional external report into a deterministic, source-bound VPM artifact while preserving raw values, score semantics, subject identities, dimension identities, and exact report-finding provenance; multiple report families over the same subjects remain semantically separate while sharing one compilation mechanism.

Not yet supported (explicitly, per the original brief's own claims-boundary section and this implementation's actual scope):

- Writer production integration (the example adapter is illustrative/synthetic only).
- Non-numeric/descriptive raw values are accepted structurally (`ScoreSemantics.DESCRIPTIVE` exists) but `raw_value` itself must still be a finite number — there is no separate non-numeric value storage path.
- Sparse reports: `missing_value_semantics="absent"` is accepted as a declared contract value but `compile_report` still requires full (subject × dimension) density and raises `ReportCompilationError` rather than compiling a sparse VPM, since `zeromodel.core.artifact.ScoreTable` has no missing-cell representation.
- Any projection layer (full grid, icon, stripe, priority-ordered view, attention/normalized-polarity view) — none implemented this stage; `CompiledReportArtifactDTO` is the canonical artifact only.
- Signed/Trust-verified compiled report artifacts — Trust integration is a future composition at the call site (Artifacts still does not depend on Trust), not implemented or claimed here.
- Cross-contract comparison, cross-version migration, or automatic report repair.

## Recommended next step

Per the original brief: prove the same mechanism with a second, real-shaped application integration (Writer sentence reports: an AI-artifact report and a quality/clarity report over the same sentence identities) beyond the illustrative example already added, once real Writer report shapes are available. Before that, per this repository's own stated sequencing (see [package-system-next.md](../architecture/package-system-next.md)), `zeromodel.search` remains planned but not started.
