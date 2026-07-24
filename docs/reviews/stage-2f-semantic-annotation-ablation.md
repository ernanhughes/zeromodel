# Stage 2F: hierarchical semantic annotation ablation

## Status

Implementation and fake-provider wiring only. No new real-provider measurement is
claimed by this PR.

Branch: `stage-2f-semantic-annotation-ablation`

## Research question

The Stage 2E real-provider smoke evidence recorded:

- `labelled-v1`: 8/8 exact, 8/8 action-correct;
- `unlabelled-v1`: 3/8 exact, 7/8 action-correct;
- no implicit Stage 2E intervention improved the unlabelled baseline;
- `cooldown-redundant-v1` and `lane-enhanced-v1` regressed.

The historical labelled renderer bundles three observable changes:

1. labelled playfield/footer geometry;
2. lane numerals `0` through `6` at the seven lane centres;
3. explicit `READY (cooldown 0)` / `BLOCKED (cooldown 1)` text.

Stage 2F asks which of those semantic components is associated with the labelled
result under the same fixed provider, prompt, parser, fixture, and compiled
policy.

## Why this is hierarchical, not a complete factorial

Three binary factors would require eight combinations for a complete 2×2×2
factorial design. Stage 2F intentionally evaluates a smaller, ordered ablation:

| Variant | Historical labelled geometry | Lane numerals | Cooldown text |
|---|---:|---:|---:|
| `unlabelled-v1` | No | No | No |
| `footer-only-v1` | Yes | No | No |
| `lane-numerals-v1` | Yes | Yes | No |
| `cooldown-text-v1` | No | No | Yes |
| `semantic-labelled-v1` | Yes | Yes | Yes |

This supports bounded component attribution along the selected paths. It does not
estimate every interaction term and must not be described as a complete
factorial experiment.

## Corrected renderer decomposition

The first implementation painted a new footer band over the unlabelled image.
That was visually useful but was not an exact decomposition of the historical
labelled renderer: labelled mode also moves the grid bottom and tank upward by
24 pixels and uses longer cooldown text.

The corrective pass defines the variants from the historical renders themselves:

### `footer-only-v1`

Starts from the historical `labelled` render, then deterministically erases:

- cooldown text;
- all seven lane numerals.

The historical labelled geometry remains unchanged.

### `lane-numerals-v1`

Starts from the historical `labelled` render and erases only cooldown text.
The historical geometry and lane numerals remain unchanged.

### `cooldown-text-v1`

Starts from the historical `unlabelled` render and overlays the exact historical
text:

```text
READY (cooldown 0)
BLOCKED (cooldown 1)
```

No lane numerals are added and the unlabelled playfield geometry is retained.

### `semantic-labelled-v1`

Starts from the historical `labelled` render and applies no pixel operation.
Its recipe identity remains distinct, but its final PNG is byte-identical to
`labelled-v1`.

`tests/test_arcade_png_interventions.py::TestHierarchicalSemanticAblation::test_semantic_labelled_is_byte_identical_to_labelled_for_all_112_states`
checks that equivalence over the complete canonical state surface.

## Seven regions, not eight boundaries

The Stage 2E `lane-enhanced-v1` recipe emphasized all eight lane boundaries and
produced an out-of-range `TANK_COLUMN: 7` response. Stage 2F preserves the
historical lane numerals at the seven lane centres. The pixel-level tests verify
seven numeral clusters rather than eight boundary markers.

## Controlled variables

The benchmark continues to hold fixed:

- provider kind and configuration identity;
- model name and model digest;
- Ollama runtime and inference settings;
- prompt text and prompt digest;
- response parser and protocol version;
- fixture identity and case mode;
- compiled policy artifact;
- confidence threshold and seed.

Every provider call still uses the fixed `unlabelled` prompt boundary. Only the
identified PNG representation recipe varies.

## Recipe identity and provenance

Every variant remains an `ArcadePngInterventionRecipe` whose identity is a
canonical hash over:

- variant id;
- base render mode;
- ordered operations;
- operation versions and parameters;
- recipe metadata.

Erasure and overlay operations are recorded in the existing observation
operation chain. They read cooldown state from visible pixels and receive no
fixture truth.

`semantic-labelled-v1` intentionally has no post-render operation. Its distinct
recipe identity comes from its variant id and metadata; its pixel identity is
expected to equal `labelled-v1`.

## Primary and diagnostic metrics

Primary Stage 2F advancement metric:

- `exact_count`.

Unconditional regression gates remain:

- increased `action_changing_count`;
- increased `rejected_count`;
- targeted-factor degradation where applicable.

Diagnostic outputs include:

- action-correct count;
- action-equivalent count;
- action-changing count;
- rejection reasons;
- tank-column, target-column, target-presence, and cooldown factor accuracy;
- latency summaries.

## Required real-provider smoke run

The first real run must include both the historical positive control and the
byte-equivalent decomposed control:

```powershell
$BaseUrl = "http://localhost:11434"
$Model = "qwen3.5:latest"
$Seed = 0
$TimeoutSeconds = 180
$ConfidenceThreshold = 0.0

$Stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$Root = ".\local-results\semantic-annotation-ablation-smoke-$Stamp"
$Database = Join-Path $Root "benchmark.sqlite"

python .\examples\arcade_png_representation_benchmark.py `
  --backend ollama `
  --model $Model `
  --base-url $BaseUrl `
  --fixture smoke `
  --variants labelled-v1,unlabelled-v1,footer-only-v1,lane-numerals-v1,cooldown-text-v1,semantic-labelled-v1 `
  --store sqlite `
  --sqlite-path $Database `
  --output-dir $Root `
  --compile-reports `
  --seed $Seed `
  --timeout $TimeoutSeconds `
  --confidence-threshold $ConfidenceThreshold `
  --write-pngs
```

Use a fresh output directory and fresh SQLite database. Do not resume the Stage
2E evidence database.

Expected provider calls: 6 variants × 8 states = 48 calls.

## Interpretation gates

Before interpreting component effects, verify:

1. `labelled-v1` reproduces the historical positive-control shape;
2. `unlabelled-v1` remains a credible baseline;
3. `semantic-labelled-v1` matches `labelled-v1` both in PNG digests and provider
   outcomes;
4. all runs share the same comparability fingerprint;
5. no variant increases action-changing errors or rejections.

A disagreement between `labelled-v1` and `semantic-labelled-v1` despite identical
PNG bytes is evidence of runtime/provider nondeterminism, not a representation
effect.

## What this stage can establish

After a reviewed real run, Stage 2F may establish bounded evidence about whether:

- labelled geometry alone changes perception;
- explicit lane numerals improve spatial addressing;
- exact cooldown text repairs cooldown recovery;
- the full historical labelled representation remains reproducible.

It cannot establish:

- open-world visual understanding;
- cross-model generality;
- natural-image performance;
- calibrated confidence;
- complete factorial interactions;
- production safety.

## Non-goals

No new production package, provider protocol, persistence aggregate, policy
compiler, prompt optimization, model training, generic image transformation
framework, or automatic representation search is introduced.

## Claims audit

No empirical claim status advances in this PR. The implementation remains
machinery plus fake-provider validation until a real Ollama evidence package is
executed, reviewed, and committed.
