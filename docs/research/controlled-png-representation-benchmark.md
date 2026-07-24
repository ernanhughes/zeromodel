# Controlled PNG representation benchmark

## Status

**Kind:** Experiment harness and evidence-package machinery, not a result.
**Real provider evidence committed:** No.
**System of record:** Stage 2D provider-evaluation aggregate
(`docs/architecture/provider-evaluation-rmdto.md`).

This document describes `examples/arcade_png_representation_benchmark.py`,
`examples/arcade_png_interventions.py`, `examples/arcade_png_representation_runner.py`,
and `examples/arcade_png_representation_comparison.py`. It does not report a
measured result. Only fake/scripted runs have been executed as of this
writing (see [Reproduction gate](#reproduction-gate)).

## Research question

> Can changing only the PNG representation improve the reliability of an
> unchanged visual provider against an unchanged compiled ZeroModel policy?

This is the first controlled experiment built on top of the Stage 2D
provider-evaluation aggregate
(`zeromodel.video.domains.video_action_set.provider_evaluation_dto`), which
was built to make exactly this kind of comparison possible: persisting
provider evaluations as evidence that separates exact-state correctness from
policy-action correctness, bound to an identified provider configuration,
observation, and compiled policy.

## Controlled variables (held fixed across every compared run)

- provider kind, model name, model digest, runtime identity
- prompt text and prompt digest (every case in every variant sends
  `predict(image_bytes, "unlabelled")` - see
  [Provider isolation](#provider-isolation) below for why the render-mode
  argument is fixed even for the `labelled-v1` reference variant)
- temperature, seed, context length, inference options
- response parser (`local_model_zero_arcade_test.parse_state_text` /
  `Prediction.parse`, reused verbatim)
- fixture state set (`smoke_states()` / `all_states()`, reused verbatim)
- compiled policy and policy artifact id (`policy_reader()`, reused verbatim)
- evaluation protocol (`ProviderEvaluationCaseDTO.build`, Stage 2D, reused
  verbatim)

## The one intervention variable

Only the **PNG representation recipe** - a declared, content-addressed,
ordered sequence of pure pixel operations applied to the canonical base
render - varies between compared runs. See
`examples/arcade_png_interventions.py:ArcadePngInterventionRecipe`.

### Flow

```text
known bounded state (ArcadeState, reused)
        v
canonical render (arcade.render(), reused)
        v
declared PNG intervention recipe (ArcadePngInterventionRecipe)
        v
identified PNG observation (ObservationDTO + operation-chain provenance)
        v
fixed perception provider (ScriptedProvider / OllamaProvider, reused)
        v
predicted bounded state (Prediction, reused)
        v
fixed VPM policy lookup (VPMPolicyLookup, reused)
        v
ProviderEvaluationCaseDTO -> ProviderEvaluationRunDTO (Stage 2D, reused)
```

## Representation variants

All recipes are deterministic and content-addressed
(`ArcadePngInterventionRecipe.recipe_id`).

| Variant | Base render | Declared operations |
|---|---|---|
| `labelled-v1` | labelled | none (reference: the existing high-information labelled render) |
| `unlabelled-v1` | unlabelled | none (reference: the existing low-information baseline) |
| `cooldown-shape-v1` | unlabelled | replace the coloured cooldown indicator with a shape (circle=ready, cross=blocked), distinguishable in grayscale, no text |
| `cooldown-dual-v1` | unlabelled | cooldown via shape **and** colour, no text |
| `cooldown-redundant-v1` | unlabelled | the dual encoding, duplicated at a second declared location (an explicit `cooldown_marker_duplicate` operation) |
| `lane-enhanced-v1` | unlabelled | cooldown unchanged from `unlabelled-v1`; stronger lane separators plus alternating triangle/diamond markers repeated above and below the lane band; no numeric lane labels |
| `footer-only-v1` | unlabelled | a single structured reserved footer band (distinct fill + one top border line); no numerals, no cooldown text |
| `lane-numerals-v1` | unlabelled | the footer band plus the numeral `0`..`6` centred under each of the seven lane *regions* (never at the eight boundary lines - see Stage 2F below); no cooldown text |
| `cooldown-text-v1` | unlabelled | explicit `READY`/`BLOCKED` text next to the cooldown indicator; no footer, no numerals |
| `semantic-labelled-v1` | unlabelled | footer band + lane numerals + cooldown text together - the explicitly factored replacement for `labelled-v1` (Stage 2F) |
| `combined-v1` | unlabelled | the operations of one cooldown-family variant plus one lane-family variant, named explicitly via `--combined-cooldown`/`--combined-lane` - never hard-coded |

Every cooldown operation is a **pure** function of `(image, declared parameters)`:
it reads the ready/blocked state directly from the existing colour-coded
indicator pixels already baked into the base render (see
`_detect_cooldown_from_pixels`) rather than being told the state out of band,
so a recipe's declared parameters never depend on which fixture state it will
later be applied to.

## Fixture modes

- `--fixture smoke`: the same 8 representative states as
  `examples/local_model_zero_arcade_test.py:smoke_states()` (reused, not
  reinvented).
- `--fixture canonical`: the same complete 112-state surface as
  `examples/local_model_zero_arcade_test.py:all_states()` (7 tank columns x 8
  target states x 2 cooldown states, reused, not reinvented).

## Provider isolation

`Provider.predict(image: bytes, render_mode: str) -> ProviderReply` is reused
unmodified from `local_model_zero_arcade_test.py`. Every case in every
variant calls `predict(final_image_bytes, "unlabelled")` -
`FIXED_PREDICT_RENDER_MODE` in `arcade_png_representation_runner.py`. This
keeps the prompt text and prompt digest identical across every compared run,
including `labelled-v1`: the model is never told which representation
variant produced the image it is looking at, only shown the pixels. The
provider never receives fixture state, expected row/action, winner criteria,
or policy contents - `tests/test_arcade_png_representation_benchmark.py::TestProviderIsolation`
proves this at runtime with a spy provider, the same pattern as
`tests/test_local_model_zero_arcade_provider_isolation.py`.

`ScriptedProvider` (fake backend) selects replies by image content digest
only, built from the ground truth at fixture-construction time - it is
wiring-only evidence, not perception evidence, exactly as documented for the
existing arcade example. `OllamaProvider` is reused unmodified.

## Provenance

Every transformed image is a normal materialized `ObservationDTO`, reusing
the existing observation/provenance/`MatrixBlob` contracts - no second
provenance graph. The operation chain always starts with the same
`render_frame` operation `local_model_zero_arcade_test._build_observation`
already uses (index 0, `input_digests=[None]`), then appends one operation
per declared recipe step, in order. Each operation's declared parameters,
16x28-grayscale-thumbnail input/output pixel digest, parameter digest, and
operation digest are recorded; each step's full-resolution PNG digest is
recorded in that operation's `parameters`, mirroring the existing
`full_resolution_png_sha256` convention. The final operation's output digest
equals the chain's `final_emitted_digest`, which equals the persisted
`MatrixBlob`/`ObservationDTO.observation_pixel_digest`. Observation
`metadata` additionally records `variant_id`, `recipe_id`,
`source_full_resolution_image_sha256`, and
`final_full_resolution_image_sha256`, so a transformed image is traceable to
its source state, source image digest, intervention recipe, ordered
operations, and result image digest without a second identity scheme.

## Comparability

`examples/arcade_png_representation_comparison.py:validate_comparable_runs`
rejects any comparison where `provider_configuration_id`, `model_digest`,
`prompt_digest`, `protocol_version` (parser version), `policy_artifact_id`,
`fixture_identity`, or `case_mode` differ between runs.
`representation_mode` and `recipe_id` are expected, and required, to differ.
Every `comparison.md`/`comparison.json` names the shared configuration
explicitly.

## Metrics

`ProviderEvaluationSummaryDTO` (Stage 2D) remains the source of truth for
every count: attempted, accepted, rejected, exact, action-equivalent,
action-changing, action-correct, per-factor correct/denominator, and latency
min/max/total/median/p95. Presentation rates (exact-state accuracy, action
accuracy, factor accuracy) are derived for `summary.json`, never treated as
identity-bearing. The primary policy-impact metric is
`action_changing_count`; the primary representation-quality metric is
`exact_count`. No selection in this harness picks a winner solely from
action accuracy.

## Selection criteria and classification

See `examples/arcade_png_representation_comparison.py:classify_variant`.
Every candidate variant is classified against a baseline (`unlabelled-v1`
when present, otherwise the alphabetically first variant in the comparison
set) as one of:

- **advance** - compatible, `action_changing_count` did not increase, and at
  least one declared target metric improved.
- **no_material_change** - compatible, no regression, no declared target
  metric met its improvement threshold.
- **regression** - `action_changing_count` increased, `rejected_count`
  increased, or a targeted factor's correctness worsened.
- **incompatible** - a fixed-identity dimension differs; never silently
  compared.

Declared target metrics per family:

- generic (reference/combined): `exact_count` improves, `rejected_count`
  decreases, or latency median improves by at least 10%.
- cooldown family (`cooldown-*-v1`): the `cooldown` factor's correctness
  improving is the required signal (checked in addition to the generic set).
- lane family (`lane-enhanced-v1`): `tank_column` or `target_column` factor
  correctness improving is the required signal.
- semantic family (`footer-only-v1`, `lane-numerals-v1`, `cooldown-text-v1`,
  `semantic-labelled-v1`, Stage 2F): `exact_count` alone -
  `SEMANTIC_TARGET_METRICS` in `arcade_png_representation_comparison.py`,
  narrower than the generic set. See
  `docs/reviews/stage-2f-semantic-annotation-ablation.md`.

`combined-v1` must never be selected before its isolated cooldown and lane
family variants have been evaluated in the same experiment (or already
present via `--resume`) - `arcade_png_representation_benchmark.py` enforces
this at the CLI level.

## Smoke -> canonical procedure

Advance a variant from `--fixture smoke` to `--fixture canonical` only after
it classifies as `advance` (or, at minimum, `no_material_change` with no
regression) against `unlabelled-v1` on the smoke fixture. Never promote a
variant whose smoke-fixture classification is `regression` or
`incompatible`.

## Output structure

```text
local-results/<experiment-name>/
  experiment.json
  comparison.json / comparison.csv / comparison.md
  <variant-id>/
    run.json, recipe.json, cases.jsonl, summary.json, report.json (with --compile-reports), images/
```

`local-results/` is gitignored; this harness never auto-commits provider
results. A curated real-provider result belongs under
`docs/results/controlled-png-representation-v1/` as **future work**, after a
real run has been executed and reviewed - not part of this change.

## Running the Stage 2F factorial ablation locally

`docs/reviews/stage-2f-semantic-annotation-ablation.md` covers the design
rationale in full. To execute the five required variants against a real
local Ollama model:

```powershell
python .\examples\arcade_png_representation_benchmark.py `
  --backend ollama `
  --model qwen3.5:latest `
  --fixture smoke `
  --variants unlabelled-v1,footer-only-v1,lane-numerals-v1,cooldown-text-v1,semantic-labelled-v1 `
  --store sqlite `
  --sqlite-path <path> `
  --output-dir <path> `
  --compile-reports `
  --seed 0 `
  --timeout 20 `
  --confidence-threshold 0.0 `
  --write-pngs
```

Or, to verify the wiring first with the fake backend (no Ollama required):

```powershell
python .\examples\arcade_png_representation_benchmark.py `
  --backend fake `
  --fixture smoke `
  --variants unlabelled-v1,footer-only-v1,lane-numerals-v1,cooldown-text-v1,semantic-labelled-v1 `
  --store memory `
  --output-dir local-results\stage-2f-fake-smoke
```

Direct script invocation (`python .\examples\...`) works because the script
inserts the repository root onto `sys.path` itself; `python -m
examples.arcade_png_representation_benchmark` from the repository root also
works.

## Reproduction gate

`docs/results/local-model-zero-arcade-smoke-v1/README.md` records the prior
labelled smoke shape (8/8 exact, 8/8 action-correct) with a real Ollama
provider. The equivalent unlabelled shape used by the Stage 2D test suite's
fixture B (`tests/test_video_provider_evaluation_rmdto.py`) is 3 exact / 4
action-equivalent / 1 action-changing / 7 action-correct out of 8. Before
drawing any new conclusion from a real-provider run of this benchmark, first
run `labelled-v1` and `unlabelled-v1` with `--backend ollama` and check
whether they reproduce those shapes. As of this writing, no real-provider run
of this benchmark has been executed - only `--backend fake` runs, whose
`ScriptedProvider` always answers by ground-truth-keyed image digest and is
therefore expected to score 100% exact on every variant. That is a wiring
proof, not a reproduction result, and must not be described as one.

## Limitations

- No claim is made, or supportable from this harness alone, that any PNG
  representation change improves real-provider accuracy, corrects real
  errors, or produces calibrated confidence. See `docs/claims-audit.md`.
- The scripted fake provider answers by ground-truth image digest; every
  variant necessarily scores identically (perfect) under `--backend fake`.
  Any classification differences observed with the fake backend in tests are
  from synthetic DTOs built directly for that purpose
  (`tests/test_arcade_png_representation_benchmark.py`), not from running the
  CLI against the fake backend.
- Cooldown-state detection from pixels (`_detect_cooldown_from_pixels`) uses
  a fixed colour-distance tolerance; it is a harness-side deterministic
  image-processing step executed before the provider ever sees the image,
  not a claim about robust colour detection in general images.
- 112-state canonical coverage, repeatability, and cross-model comparison are
  all out of scope for this change (see Non-goals).

## Non-goals

Model fine-tuning; prompt optimization; automatic layout search; genetic/RL
recipe optimization; Hamming/error-correcting codes; a generic visual
codebook; a new persistence aggregate; a new production package; natural
image datasets; cloud providers; cross-configuration model comparison;
automatic promotion of a winning variant into production; a sparse report
compiler; Store-wide concurrency redesign.

## Related documentation

- `docs/architecture/provider-evaluation-rmdto.md` - the Stage 2D aggregate
  this benchmark builds on.
- `docs/examples/local_model_zero_arcade_test.md` - the fixture, renderer,
  provider, and policy machinery this benchmark reuses verbatim.
- `docs/claims-audit.md` - the bounded claim and required limitation for this
  capability.
- `docs/reviews/stage-2f-semantic-annotation-ablation.md` - the factorial
  ablation of `labelled-v1`'s bundled footer/numerals/cooldown-text
  components, built on this harness.
