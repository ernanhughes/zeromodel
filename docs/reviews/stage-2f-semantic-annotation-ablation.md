# Stage 2F: factorial semantic annotation ablation

## Scope

Branch: `stage-2f-semantic-annotation-ablation`, created from `main` after
PR #75 (Stage 2D, which carried Stage 2E's merged, stacked commits) landed.

This stage extends the Stage 2E controlled PNG representation benchmark
(`docs/research/controlled-png-representation-benchmark.md`) with a small
factorial ablation of the previously bundled `labelled-v1` representation. It
adds four new deterministic recipe variants and reuses every other part of
the Stage 2D/2E machinery unchanged: the provider-evaluation aggregate, the
fixture/renderer/parser/policy from `examples/local_model_zero_arcade_test.py`,
the provenance chain construction, the comparability validator, and the
`advance`/`no_material_change`/`regression`/`incompatible` classifier.

## What Stage 2F is testing

> Why does the labelled representation succeed?

`labelled-v1` bundles three things at once relative to `unlabelled-v1`:
a taller reserved bottom margin, printed lane numerals, and explicit
`READY (cooldown 0)` / `BLOCKED (cooldown 1)` text. A real local-Ollama
run of the Stage 2E benchmark (`docs/results/controlled-png-representation-v1/`)
measured `labelled-v1` at 8/8 exact and `unlabelled-v1` at 3/8 exact, 7/8
action-correct on the eight-state smoke fixture - but that result cannot say
*which* of the three bundled components caused the improvement, because they
were never varied independently.

Stage 2F isolates them:

| Variant | Footer geometry | Lane numerals | Cooldown text |
| --- | ---: | ---: | ---: |
| `unlabelled-v1` | No | No | No |
| `footer-only-v1` | Yes | No | No |
| `lane-numerals-v1` | Yes | Yes | No |
| `cooldown-text-v1` | No | No | Yes |
| `semantic-labelled-v1` | Yes | Yes | Yes |

`semantic-labelled-v1` is the explicitly factored replacement for the
previously bundled `labelled-v1` - it should reproduce roughly the same
visual information `labelled-v1` conveyed, but built from named, independently
testable operations rather than baked into `render()`'s own `mode="labelled"`
branch.

## Why Stage 2E was insufficient for this question

Stage 2E's non-reference variants (`cooldown-shape-v1`, `cooldown-dual-v1`,
`cooldown-redundant-v1`, `lane-enhanced-v1`) tested *implicit*, non-textual
semantics: shape/colour coding for cooldown, and stronger visual separators
for lane position. None of them reproduced `labelled-v1`'s result - the real
Ollama run recorded all four as `no_material_change` or `regression`
(`cooldown-redundant-v1` and `lane-enhanced-v1` both increased
`action_changing_count` from 1 to 2; `lane-enhanced-v1` also produced one
rejected `TANK_COLUMN: 7` response). That result narrowed the hypothesis to
"explicit textual/structural semantics, not implicit visual redundancy," but
still could not separate the footer, numerals, and cooldown text factors from
each other, because `labelled-v1` bundles all three and none of the tested
implicit variants isolated any of them individually. Stage 2F is the
factorial follow-up that result explicitly called for.

## A lesson carried over from `lane-enhanced-v1`

The real-provider `lane-enhanced-v1` result is the direct reason
`lane_numerals_overlay` (this stage) draws digits at the seven lane
*centres* (`range(LANES)`) and never at the eight boundary lines between
them (`range(LANES + 1)`, which `lane_separator_enhance` used). The prior
variant's boundary emphasis correlated with a rejected out-of-range
`TANK_COLUMN: 7` response and more action-changing errors -
`tests/test_arcade_png_interventions.py::TestSemanticAnnotationVariants::test_lane_numerals_draws_exactly_seven_glyph_clusters_not_eight`
guards this directly at the pixel level, and
`docs/results/controlled-png-representation-v1/README.md` records the
original observation.

## What each new variant means

All four are pure post-processing operations layered on the existing
`unlabelled` base render (see `examples/arcade_png_interventions.py`),
exactly like Stage 2E's cooldown/lane operations - no change to `render()`
itself.

- **`footer_reserved_area`** - fills a single 32px band at the bottom of the
  frame (`FOOTER_TOP = IMG_HEIGHT - 32 = 480`) with a distinct colour and one
  top border line. No numerals, no cooldown text, no per-lane structure -
  isolates whether reserved geometry *alone*, with no semantic content in it,
  changes provider behaviour. `FOOTER_TOP` is chosen to stay below the tank
  sprite's lowest point (`bottom - 45 + 33 = 476` for every lane and state,
  computed from `render()`'s own geometry) with a 4px margin, so the footer
  band never occludes the sprite it sits under -
  `test_footer_does_not_occlude_the_tank_sprite` guards this.
- **`lane_numerals_overlay`** - draws the numeral `0`..`6` centred under each
  of the seven lane regions, inside the footer band the prior operation
  introduced. Requires the footer (numerals need somewhere to live); draws no
  cooldown text.
- **`cooldown_text_overlay`** - draws `READY` or `BLOCKED` next to the
  existing colour-coded cooldown indicator, reading the ready/blocked state
  from the indicator's own pixels (`_detect_cooldown_from_pixels`, reused
  from Stage 2E) rather than any out-of-band signal. Needs no footer - kept
  minimal per the brief.
- **`semantic-labelled-v1`** - all three operations, in order (footer, then
  numerals, then cooldown text).

Recipe identity stays content-addressed exactly as in Stage 2E
(`ArcadePngInterventionRecipe.recipe_id`, a `canonical_sha256` over the
declared operations); every new variant changes recipe identity from every
other variant, and `apply_recipe` still rejects any operation that produces
byte-identical output (no silent no-ops).

## Comparison and classification

The comparability validator (`validate_comparable_runs`) is unchanged and
applies identically to the new variants: any run sharing provider
configuration, model digest, prompt digest, protocol version, policy
artifact, fixture identity, and case mode remains comparable regardless of
representation.

For this stage the declared *primary* target metric is `exact_count` alone
(`SEMANTIC_TARGET_METRICS = ("exact_count",)` in
`arcade_png_representation_comparison.py`) - narrower than the generic family
default (`exact_count`, `rejected_count`, `latency_median_us`), matching the
brief's framing that `exact_count` is *the* key metric here while
`action_changing_count`, `action_correct`, and `rejected` stay diagnostic.
`action_changing_count` and `rejected_count` still gate `regression`
unconditionally (that check in `classify_variant` does not depend on the
target-metric set), so a semantic variant that crosses a policy boundary or
increases rejections is still caught - it just cannot, on its own, turn
`rejected_count` improving into an `advance` the way a generic-family variant
could.

## What this stage can and cannot prove

**Can prove (machinery, not measurement):**
- The four new variants are deterministic, content-addressed, and produce
  the intended pixel-level isolation (verified by
  `tests/test_arcade_png_interventions.py::TestSemanticAnnotationVariants`
  and confirmed with the fake backend end-to-end).
- Comparisons among the new variants (and against `unlabelled-v1`) remain
  correctly gated by the same fixed-identity contract Stage 2E established.
- The benchmark can execute, persist, and classify these variants through
  the Stage 2D aggregate exactly as it does for Stage 2E's variants.

**Cannot prove without a real-provider run:**
- Which factor (or combination) actually improves a real model's
  reliability. `--backend fake` always scores every variant perfectly
  (`ScriptedProvider` answers by ground-truth image digest, not perception),
  so no fake-backend result is evidence about a real provider's behaviour -
  see the reproduction gate in
  `docs/research/controlled-png-representation-benchmark.md`.
- That `semantic-labelled-v1` reproduces `labelled-v1`'s real 8/8 result -
  they are visually similar but not byte-identical (different rendering
  path), so this must be checked empirically, not assumed.
- Anything about generalization beyond the eight-state smoke fixture, this
  one model/runtime, or this one prompt/parser.

No real Ollama experiment was run as part of this stage - only
`--backend fake` wiring runs, as required.

## Files changed

- `examples/arcade_png_interventions.py` - three new pure pixel operations
  (`footer_reserved_area`, `lane_numerals_overlay`, `cooldown_text_overlay`),
  four new variant constants and recipe builders, a new `"semantic"` recipe
  family, new exported geometry constants used by tests
  (`FOOTER_TOP`, `IMG_WIDTH`, `IMG_HEIGHT`, `LANES`, `LEFT_MARGIN`,
  `RIGHT_MARGIN`, `COOLDOWN_TEXT_POSITION`).
- `examples/arcade_png_representation_comparison.py` - `SEMANTIC_TARGET_METRICS`.
- `examples/arcade_png_representation_benchmark.py` - `_target_metrics_for_variant`
  now maps the new variants to `SEMANTIC_TARGET_METRICS`; no other
  orchestration change was needed (the CLI, provider building, output
  writing, and comparison steps already iterate generically over
  `args.variants`/`recipes`).
- `tests/test_arcade_png_interventions.py` - `TestSemanticAnnotationVariants`
  (13 new tests): declared-recipe/determinism checks, per-variant operation
  membership, footer/tank-sprite non-occlusion, the seven-centres-not-eight-
  boundaries pixel guard, cooldown-text ready/blocked distinction, and
  purity (no input mutation).
- `tests/test_arcade_png_representation_benchmark.py` -
  `TestSemanticComparability` and `TestSemanticBenchmarkOrchestration`
  (7 new tests): comparability across the new + existing variants, the
  `exact_count`-only target-metric behaviour versus the generic set,
  incompatibility on a mismatched policy artifact, and end-to-end fake-backend
  runs (all five required variants, summary/case consistency, recipe-identity
  round trip).
- `docs/reviews/stage-2f-semantic-annotation-ablation.md` (this file).
- `docs/research/controlled-png-representation-benchmark.md` - variant table
  and run instructions extended with the four new variants.
- `docs/claims-audit.md` - see "Claims audit" below.

## Non-goals confirmed unchanged

No general perception framework, no factorized visual-state compiler, no new
protocol, no search/navigation/trust redesign, no core package
restructuring, no new persistence domain, no new production package, no
attempt at open-world image understanding. `render()` itself, the Stage 2D
aggregate, the report compiler, and package boundaries are all untouched.

## Claims audit

No claim status changed. The existing Stage 2D/2E claims-audit row for
controlled visual-representation comparison already covers "the machinery
can be extended with additional representation variants" implicitly; Stage
2F's variant names are added to that row's evidence description, and the
row's status stays **Implemented / unmeasured** (or whatever status the
real-provider evidence row for the tested implicit interventions already
carries) - this stage adds machinery and fake-backend wiring evidence only,
not a new measurement.
