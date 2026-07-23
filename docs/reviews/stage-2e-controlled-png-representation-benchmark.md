# Stage 2E: controlled PNG representation benchmark

## Scope

Branch: `stage-2e-controlled-png-intervention-benchmark`, created from
`stage-2d-provider-evaluation-aggregate` at `c277e28e597b715b538a1b1b08c0d0e5d4f3c62a`
(the corrective commit `7113403` is included; not yet merged into `main` at
the time this branch was created, so the branch was cut from the Stage 2D
branch per the task instructions rather than from `main`).

This stage builds the first controlled experiment on top of the Stage 2D
provider-evaluation aggregate: "can changing only the PNG representation
improve the reliability of an unchanged visual provider against an unchanged
compiled ZeroModel policy?" It is an experiment harness plus (fake-only)
wiring evidence, not a new production package, not a new persistence
aggregate, and not a real-provider result.

## Files added

- `examples/arcade_png_interventions.py` (525 lines) - `ArcadePngOperationSpec`
  / `ArcadePngInterventionRecipe`, deterministic pure pixel operations for
  all 7 variants, recipe identity, `apply_recipe`.
- `examples/arcade_png_representation_runner.py` (476 lines) - execution
  engine: provider configuration, scripted-reply builder, provenance chain
  construction, per-case execution, `run_variant`, `find_resumable_run`.
- `examples/arcade_png_representation_comparison.py` (349 lines) -
  `RepresentationComparisonRow`, `validate_comparable_runs`,
  `classify_variant`, JSON/CSV/Markdown comparison writers.
- `examples/arcade_png_representation_benchmark.py` (466 lines) - CLI,
  orchestration, per-variant output writing, experiment manifest. Entry
  point.
- `docs/research/controlled-png-representation-benchmark.md` - hypothesis,
  controlled/intervention variables, variants, fixture modes, provider
  isolation, provenance, metrics, selection criteria, reproduction gate,
  output structure, limitations, non-goals.
- `tests/test_arcade_png_interventions.py` (19 tests) - recipe identity,
  determinism, no-op rejection, pixel-level visual distinction.
- `tests/test_arcade_png_representation_benchmark.py` (26 tests) -
  comparability, classification, provider isolation, provenance, resume,
  memory + SQLite Store integration.
- `docs/reviews/stage-2e-controlled-png-representation-benchmark.md` (this
  file).

All four new `examples/` modules stay well under the 800-line
`new_module_max_lines` quality-baseline ceiling; no function exceeds the
100-line ceiling; no `quality-baseline.toml` change was needed.

## Files modified (link-only, per the task instructions)

- `README.md` - new "Local vision-model provider evaluation" section linking
  the arcade example doc, the provider-evaluation architecture doc, and the
  new research doc.
- `docs/architecture/provider-evaluation-rmdto.md` - one paragraph linking
  the new research doc under "What remains external research".
- `docs/examples/local_model_zero_arcade_test.md` - one paragraph under
  "Reporting new evidence" linking the new research doc/harness.
- `docs/claims-audit.md` - updated the existing "Controlled visual-
  representation changes..." row (added by Stage 2D) to name the new
  harness, its variant set, comparability validator, and classification
  logic; the status stays **Implemented / unmeasured**; added an explicit
  "claims to avoid" list.

## Controlled-variable contract

Held fixed across every compared run: provider kind, model name/digest,
runtime identity, prompt text/digest, temperature, seed, context length,
inference options, response parser, fixture state set, compiled policy and
policy artifact id, evaluation protocol. Enforced mechanically by
`arcade_png_representation_comparison.validate_comparable_runs` (checks
`provider_configuration_id`, `model_digest`, `prompt_digest`,
`protocol_version`, `policy_artifact_id`, `fixture_identity`, `case_mode`)
and, upstream of that, by every case in every variant calling
`predict(image_bytes, "unlabelled")` regardless of which variant/base render
produced the image - see `FIXED_PREDICT_RENDER_MODE` in
`arcade_png_representation_runner.py`.

## Recipe design

`ArcadePngInterventionRecipe` is a frozen/slotted dataclass with a
content-derived `recipe_id` (`canonical_sha256` over version, variant id,
base render mode, ordered operation specs, metadata) - identical to the
identity pattern already used by `ProviderConfigurationDTO` and every other
video action-set DTO. Seven variants:

- `labelled-v1` / `unlabelled-v1` - zero declared operations (the existing
  reference renders).
- `cooldown-shape-v1` - replaces the coloured cooldown indicator with a
  shape (circle=ready, cross=blocked); the shape function *reads* the
  ready/blocked state directly from the base render's existing pixels
  (`_detect_cooldown_from_pixels`), so recipe parameters never depend on the
  fixture state they will be applied to.
- `cooldown-dual-v1` - shape and colour together.
- `cooldown-redundant-v1` - the dual encoding duplicated at a second
  location via a dedicated `cooldown_marker_duplicate` operation (pixel copy,
  not re-detection - avoids re-detecting state from an already-transformed
  region).
- `lane-enhanced-v1` - stronger lane separators plus alternating
  triangle/diamond markers repeated above and below the lane band; verified
  to leave the cooldown indicator's pixels byte-identical to `unlabelled-v1`.
- `combined-v1` - concatenates one cooldown-family and one lane-family
  variant's operations, named via required `--combined-cooldown`/
  `--combined-lane` CLI arguments - never hard-coded.

## Provenance design

Reuses the exact `ObservationDTO`/`MatrixBlob`/`ObservationOperationChainDTO`
contracts and the same `render_frame` operation-0 convention
`local_model_zero_arcade_test._build_observation` already uses; generalizes
it from one operation to an ordered N-operation chain, one operation per
declared recipe step, each carrying its declared parameters plus a
`full_resolution_png_sha256` (following the existing convention), with a
16x28-grayscale-thumbnail pixel digest as `output_digest`/next
`input_digest`. Observation `metadata` additionally records `variant_id`,
`recipe_id`, `source_full_resolution_image_sha256`, and
`final_full_resolution_image_sha256`. No second provenance graph.

## Comparison and selection logic

`validate_comparable_runs` rejects any pairwise comparison where a fixed
identity dimension differs, with `representation_mode`/`recipe_id` expected
to differ. `classify_variant` labels each candidate `advance` /
`no_material_change` / `regression` / `incompatible` against a baseline
(`unlabelled-v1` when present) using family-specific declared target metrics
(`cooldown` factor correctness for the cooldown family; `tank_column`/
`target_column` factor correctness for the lane family; exact count /
rejected count / latency median for everything else). `combined-v1` is
rejected by the CLI unless a cooldown-family variant is also present in
`--variants` (or `--resume` can find one already run).

## Tests

45 new tests, all passing:
`tests/test_arcade_png_interventions.py` (19) covers recipe identity
determinism (order and parameter changes alter identity), every variant
having a declared recipe, unknown-variant/missing-combined-args rejection,
no-op-transform rejection (a genuine self-duplicate no-op), and pixel-level
distinctions (labelled vs unlabelled, ready vs blocked shape, dual
colour+shape, redundant duplicate markers, lane-enhanced leaving the
cooldown region untouched, determinism).
`tests/test_arcade_png_representation_benchmark.py` (26) covers
comparability (representation-only changes stay comparable; each of the 7
fixed-identity fields independently rejects comparison), classification
(no_material_change / advance / regression via action-changing increase /
regression via rejected increase / incompatible / family-specific factor
metric detecting an improvement the generic metric set misses), provider
isolation (a spy provider proves only image bytes and the fixed render mode
reach `predict()`), provenance (ordered/contiguous chain, `input_digests`
chaining, final digest matching the persisted pixel digest, single-operation
reference variant, `MatrixBlob` linkage, recipe/source-fixture metadata
linkage), resume (idempotent reuse, no-match case, multiple-candidate
rejection), and one `@pytest.mark.integration` SQLite Store round-trip test.

## Exact validation commands and results

All commands run inside a disposable, worktree-local virtualenv
(`.venv-zeromodel-dev`, created fresh for this stage, never committed -
deleted after use, following the same practice Stage 2D used).

```text
python -m pytest tests/test_arcade_png_interventions.py tests/test_arcade_png_representation_benchmark.py tests/test_video_provider_evaluation_rmdto.py tests/test_video_provider_evaluation_sql_store.py tests/test_local_model_zero_arcade_provider_isolation.py tests/test_provider_evaluation_report_adapter.py --run-integration -q
# 130 passed

python -m examples.arcade_png_representation_benchmark --backend fake --store memory --fixture smoke --output-dir local-results/arcade-png-benchmark-validation-memory --compile-reports
# labelled-v1, unlabelled-v1, cooldown-shape-v1, cooldown-dual-v1, cooldown-redundant-v1, lane-enhanced-v1
# all 8/8 exact (perfect scripted provider, as expected for --backend fake - see the reproduction gate
# in docs/research/controlled-png-representation-benchmark.md); comparison.json/csv/md written;
# report.json compiled per variant.

python -m examples.arcade_png_representation_benchmark --backend fake --store sqlite --sqlite-path local-results/arcade-png-benchmark-validation.db --fixture smoke --variants labelled-v1,unlabelled-v1,cooldown-dual-v1 --compile-reports --output-dir local-results/arcade-png-benchmark-validation-sqlite
# 3 variants, same result shape, persisted through zeromodel-sqlalchemy.

python -m mypy packages/core/src packages/analysis/src packages/observation/src packages/vision/src/zeromodel/vision/visual.py packages/vision/src/zeromodel/vision/visual_policy.py packages/vision/src/zeromodel/vision/__init__.py packages/video/src packages/sqlalchemy/src packages/artifacts/src packages/trust/src packages/navigation/src
# Success: no issues found in 152 source files

python scripts/check_package_boundaries.py    # Package boundary check passed: 152 production modules
python scripts/check_architecture.py          # Architecture check: passed (152 production modules inspected)
```

`ruff format .` reformatted 147 files. Diffing before/after `git status`
against the intentional Stage 2E file list, 140 unrelated pre-existing
files were reformatting drift and were reverted with `git checkout --`;
only the 6 new/touched Stage 2E files kept their (already-clean) formatting.

`ruff check .` and `scripts/check_quality.py` (which runs its own scoped
`ruff check`) and `scripts/run_fast_tests.py` (via
`tests/test_ruff_scope_claims.py`) all currently **fail on this branch for a
pre-existing, environment-level reason unrelated to this change**: `ruff` is
unpinned in `requirements-dev.txt` (and in every `.github/workflows/*.yml`
package CI job), and the `ruff` version resolved into this fresh virtualenv
(0.16.0) reports substantially more lint findings across virtually the
entire pre-existing codebase than whatever version the repository's own
`tests/test_ruff_scope_claims.py::test_unscoped_ruff_still_has_known_pre_existing_findings`
was last calibrated against. Confirmed pre-existing and unrelated to this
change: `git diff --stat stage-2d-provider-evaluation-aggregate --
packages/ scripts/ integration_tests/ tests/test_ruff_scope_claims.py`
shows zero changes - this branch never touches any file these checks
inspect. Scoped to only the files this stage adds/touches, both checks are
clean:

```text
ruff check examples/arcade_png_interventions.py examples/arcade_png_representation_benchmark.py examples/arcade_png_representation_comparison.py examples/arcade_png_representation_runner.py tests/test_arcade_png_interventions.py tests/test_arcade_png_representation_benchmark.py
# All checks passed!
```

With only the two ruff-version-sensitive tests deselected,
`scripts/run_fast_tests.py`'s underlying suite (`pytest tests
integration_tests packages/*/tests -m "not slow and not external and not
research"`) is fully green including every new test this stage adds:

```text
1024 passed, 1 skipped, 201 deselected in 50.36s
```

(The 1 skip is pre-existing and unrelated to this change.)

## Deviations from the brief

- Reused several of `local_model_zero_arcade_test.py`'s underscore-prefixed
  helpers directly (`_build_runtime`, `_build_benchmark_identity`,
  `_build_episode_plan`) rather than duplicating them, since the brief's
  overriding instruction was "do not duplicate or reimplement Stage 2D...
  reuse it" and this repository's own test suite already reaches into that
  module's internals the same way
  (`tests/test_local_model_zero_arcade_provider_isolation.py`).
- Did not extract `OllamaProvider` into a separate shared module - direct
  `import examples.local_model_zero_arcade_test as arcade` and reuse was
  sufficient and simpler; extraction was explicitly conditional ("if
  needed") in the brief.
- Each representation variant gets its own `BenchmarkIdentityDTO` and
  `EpisodePlanDTO` (not one shared identity/plan across variants), because
  the identity/episode-plan aggregate requires a unique
  `(seed_digest, split, ordinal)` triple per plan and `ObservationDTO`
  requires a purely decimal `frame_id` sequence suffix - two variants of the
  same fixture state cannot share one episode id. This diversity is
  bookkeeping only; it is explicitly excluded from the provider-evaluation
  comparability fingerprint (`FIXED_IDENTITY_FIELDS`), so it does not weaken
  the controlled-variable contract.
- The "preferred" (non-blocking) selection criteria named in the brief for
  the cooldown family (exact improves, rejection doesn't increase, latency
  doesn't worsen) are reported informationally via the generic target-metric
  set layered on top of the required cooldown-factor signal, rather than as
  a second gating tier - classification stays a single label per candidate.
- `ruff check .` / `scripts/check_quality.py` / `scripts/run_fast_tests.py`
  were run and their actual failing output is reported above rather than
  silently worked around; see "Exact validation commands and results".

## Remaining limitations

See `docs/research/controlled-png-representation-benchmark.md#limitations`
for the full list. In short: no real-provider run of this benchmark has been
executed; the fake backend always scores perfectly by construction and is
wiring evidence only; cooldown-state pixel detection uses a fixed colour
tolerance; canonical (112-state) coverage and repeatability are unexercised.

## Real provider experiment status

No real (Ollama) provider experiment was run as part of this stage, as
required - only `--backend fake` runs. The recommended first real smoke
command (not run here) is:

```powershell
python -m examples.arcade_png_representation_benchmark `
  --backend ollama --model qwen3.5:latest --fixture smoke `
  --variants labelled-v1,unlabelled-v1 --confidence-threshold 0.0 `
  --output-dir local-results/arcade-png-representation-ollama-smoke-v1
```

per the reproduction gate in the research doc, before running any
intervention variant with a real provider.

## Claims-audit changes

Updated the existing Stage 2D row (see "Files modified" above); no claim
status changed (`Implemented / unmeasured` before and after) - this stage
adds machinery and fake-backend wiring evidence, not a measured result.

## Confirmation

No files outside this stage's stated scope were changed; the 140
`ruff format` unrelated-file reformats were reverted before committing.
