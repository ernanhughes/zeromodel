# Controlled PNG representation benchmark

## Status

**Kind:** bounded experiment harness and evidence machinery.  
**Real-provider evidence:** Stage 2E smoke evidence is committed at
`docs/results/controlled-png-representation-v1/`.  
**Current extension:** Stage 2F hierarchical semantic annotation ablation.  
**System of record:** Stage 2D provider-evaluation aggregate.

The harness asks:

> Can changing only the identified PNG representation change the reliability of
> an unchanged visual provider against an unchanged compiled ZeroModel policy?

It persists exact-state, action-equivalent, action-changing, rejected,
factor-level, confidence, and latency evidence while preserving provider,
prompt, policy, fixture, and recipe identities.

## Controlled variables

Every comparable run holds fixed:

- provider kind and provider configuration id;
- model name and model digest;
- runtime identity and inference options;
- prompt text and prompt digest;
- response parser and protocol version;
- fixture state set and case mode;
- compiled policy artifact;
- confidence threshold and seed.

Every provider call uses:

```text
predict(final_image_bytes, "unlabelled")
```

including calls for `labelled-v1`. The model is shown the pixels but is never
told the representation variant.

## Intervention variable

Only the content-addressed PNG recipe varies. A recipe declares:

- base render mode;
- ordered operation names;
- operation versions;
- canonical parameters;
- recipe metadata;
- deterministic recipe id.

The provider never receives fixture truth, expected policy row, expected action,
or selection criteria.

## Flow

```text
bounded ArcadeState
        ↓
canonical labelled or unlabelled render
        ↓
identified recipe and ordered pixel operations
        ↓
materialized ObservationDTO + MatrixBlob + operation provenance
        ↓
fixed provider and fixed prompt
        ↓
bounded predicted state
        ↓
fixed compiled VPM policy lookup
        ↓
ProviderEvaluationCaseDTO / ProviderEvaluationRunDTO
        ↓
comparison.json / comparison.csv / comparison.md
```

## Representation variants

| Variant | Base render | Representation definition |
|---|---|---|
| `labelled-v1` | labelled | Historical high-information reference render. |
| `unlabelled-v1` | unlabelled | Historical low-information reference render. |
| `cooldown-shape-v1` | unlabelled | Circle for ready, cross for blocked; no text. |
| `cooldown-dual-v1` | unlabelled | Cooldown shape plus colour; no text. |
| `cooldown-redundant-v1` | unlabelled | Dual marker duplicated at a second location. |
| `lane-enhanced-v1` | unlabelled | Stronger separators and boundary markers; no numerals. |
| `footer-only-v1` | labelled | Historical labelled geometry with cooldown text and all seven lane numerals deterministically erased. |
| `lane-numerals-v1` | labelled | Historical labelled geometry and lane numerals with cooldown text erased. |
| `cooldown-text-v1` | unlabelled | Historical unlabelled geometry plus exact historical `READY (cooldown 0)` / `BLOCKED (cooldown 1)` text. |
| `semantic-labelled-v1` | labelled | Distinct recipe identity whose final PNG is byte-identical to `labelled-v1`. |
| `combined-v1` | unlabelled | Explicitly named Stage 2E cooldown and lane-family recipes combined in order. |

## Stage 2E evidence

The committed Stage 2E real-Ollama smoke result records:

| Variant | Exact | Action-correct | Action-changing | Rejected |
|---|---:|---:|---:|---:|
| `labelled-v1` | 8/8 | 8/8 | 0 | 0 |
| `unlabelled-v1` | 3/8 | 7/8 | 1 | 0 |
| `cooldown-shape-v1` | 4/8 | 7/8 | 1 | 0 |
| `cooldown-dual-v1` | 3/8 | 7/8 | 1 | 0 |
| `cooldown-redundant-v1` | 3/8 | 6/8 | 2 | 0 |
| `lane-enhanced-v1` | 2/8 | 5/8 | 2 | 1 |

The implicit interventions did not pass their promotion gates. The lane-enhanced
variant produced an invalid `TANK_COLUMN: 7`, which is why Stage 2F retains lane
semantics at seven region centres rather than emphasizing eight boundaries.

See `docs/results/controlled-png-representation-v1/README.md` for the bounded
measurement, environment, identities, and limitations.

## Stage 2F: hierarchical semantic annotation ablation

The historical labelled renderer bundles:

1. labelled playfield/footer geometry;
2. lane numerals at seven lane centres;
3. explicit cooldown text.

Stage 2F separates those components along five selected paths. It is a
**hierarchical ablation**, not a complete 2×2×2 factorial design. Three binary
factors would require eight combinations; this stage does not estimate every
interaction term.

The corrected implementation uses the historical renderers themselves rather
than painting a new approximate footer:

- `footer-only-v1` starts from labelled pixels and erases both annotation groups;
- `lane-numerals-v1` starts from labelled pixels and erases only cooldown text;
- `cooldown-text-v1` starts from unlabelled pixels and adds the exact historical
  cooldown label;
- `semantic-labelled-v1` starts from labelled pixels and performs no post-render
  operation.

The complete 112-state test suite requires `semantic-labelled-v1` to be
byte-identical to `labelled-v1`.

## Provenance

Every final image is materialized through the existing observation aggregate.
The operation chain begins with `render_frame`, followed by one entry for each
recipe operation. Each step records:

- operation name and version;
- canonical parameters and parameter digest;
- input and output thumbnail digests;
- full-resolution PNG digest;
- operation digest;
- final emitted digest.

Recipe and observation metadata retain source and final full-resolution image
digests. Pixel-identical reference/decomposed variants may share content-addressed
image bytes while retaining distinct representation and recipe identities.

## Comparability

`validate_comparable_runs` rejects comparisons whose fixed identity differs in
any of these fields:

- provider configuration id;
- model digest;
- prompt digest;
- protocol version;
- policy artifact id;
- fixture identity;
- case mode.

Representation mode and recipe id are expected to differ.

## Classification

Candidates are classified against `unlabelled-v1` when present:

- `advance`: compatible, no regression gate fired, and a declared target metric
  improved;
- `no_material_change`: compatible but no declared target metric improved;
- `regression`: action-changing errors or rejections increased, or a targeted
  factor worsened;
- `incompatible`: a fixed identity differs.

Stage 2F uses `exact_count` as its primary advancement metric. Action-changing
errors and rejections remain unconditional regression gates.

## Fixture modes

- `--fixture smoke`: eight representative states;
- `--fixture canonical`: all 112 states: seven tank columns × eight target states
  × two cooldown states.

A real-provider candidate should not advance to canonical evaluation until its
smoke result is reviewed and does not regress policy-impact metrics.

## Run Stage 2F locally

Use a fresh output directory and SQLite database. Include both the historical
positive control and its byte-identical decomposed control:

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

Expected real-provider calls:

```text
6 variants × 8 smoke states = 48 calls
```

Do not resume the Stage 2E experiment database.

### Fake wiring run

```powershell
python .\examples\arcade_png_representation_benchmark.py `
  --backend fake `
  --fixture smoke `
  --variants labelled-v1,unlabelled-v1,footer-only-v1,lane-numerals-v1,cooldown-text-v1,semantic-labelled-v1 `
  --store memory `
  --output-dir local-results\stage-2f-fake-smoke
```

The scripted provider returns ground-truth replies by image digest. Perfect fake
scores validate wiring only; they are not perception evidence.

## Interpretation gate

Before attributing any effect:

1. confirm the model digest and runtime identity;
2. confirm one provider configuration, prompt digest, policy artifact, fixture,
   and case mode across all runs;
3. confirm `labelled-v1` and `semantic-labelled-v1` PNG digests match per case;
4. confirm their provider outcomes match or classify any disagreement as runtime
   nondeterminism;
5. inspect action-changing errors and rejection reasons before exact-count gains.

## Output structure

```text
local-results/<experiment>/
  experiment.json
  comparison.json
  comparison.csv
  comparison.md
  <variant>/
    run.json
    recipe.json
    summary.json
    cases.jsonl
    report.json        # with --compile-reports
    images/
```

## Limitations

This harness and its evidence do not establish:

- open-world visual understanding;
- natural-image performance;
- cross-model generality;
- calibrated confidence;
- complete factorial interactions;
- production safety;
- automatic representation optimization.

## Related documentation

- `docs/architecture/provider-evaluation-rmdto.md`
- `docs/results/controlled-png-representation-v1/README.md`
- `docs/reviews/stage-2f-semantic-annotation-ablation.md`
- `docs/examples/local_model_zero_arcade_test.md`
- `docs/claims-audit.md`
