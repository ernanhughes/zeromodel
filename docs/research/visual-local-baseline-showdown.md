# Visual local baseline showdown

**Status:** executed  
**Execution date:** Saturday, July 18, 2026  
**Evidence package:** `docs/results/visual-local-baseline-showdown-v1/`
**Post-analysis package:** `docs/results/visual-local-baseline-showdown-v1-postanalysis/`

## Question

The bounded Phase B Stage 1 question was whether deterministic integer-pixel
registration plus local normalized-pixel comparison could recover a useful
accepted operating point against the frozen System B v2 dataset and selection
protocol without introducing distinguishable false accepts or accepted
conflicting-action errors.

## Result

The registered local baseline produced **Outcome C**.

- Selected quantile: `0.0`
- Selection digest: `881d63866587aec8d87d4d3419431ccd195f6fe57828e9f70580ebea1e0337f4`
- Calibration digest: `b3efc2a4687ba97a5b899961a8d3b53b91f1e0f2c73afc7bde92a1bb8e805265`
- Trace digest: `c1416850155cb464a70b55fd8856a240ea4d86b6ae3f85afb1948899469dbfec`

Final evaluation on `1344` expected-accept observations and `248`
distinguishable rejection opportunities showed:

- accepted benign count `0 / 1344`
- false accepts `0 / 248`
- false rejects `1344 / 1344`
- accepted conflicting-action errors `0`
- raw top-1 exact-row accuracy `1176 / 1344 = 0.875`
- raw top-1 action accuracy `1323 / 1344 = 0.984375`

The bounded registration step materially improved ranking on the concentrated
translation family. In the `final-shift-two` family, raw top-1 exact-row
correctness rose from `224 / 336` to `336 / 336`, and raw top-1 action
correctness rose from `322 / 336` to `336 / 336`. That improvement did not
transfer into a useful accepted operating point under the zero-observed-FAR
constraint.

The held-out two-pixel shifts were unseen during selection, but they were still
inside the declared `[-3, 3]` registration search bounds. The supported
generalization claim is therefore bounded: unseen instances within the declared
translation envelope, not evidence of robustness beyond it.

## Post-analysis

The Stage 1 post-analysis package adds two measurements over the committed
fixture and frozen Stage 1 evidence:

- an independent distance-threshold by ambiguity-margin calibration grid;
- a final benign rejection decomposition by gate and family.

The decoupled grid did not revise the Stage 1 result.

- Feasible decoupled candidates: `11`
- Every feasible point used the most conservative margin slice: `margin quantile = 0.0`
- Best feasible calibration coverage: `6 / 1344`
- Best feasible final coverage: `0 / 1344`

The selected decoupled candidate was:

```text
distance quantile: 0.5
margin quantile: 0.0
threshold: 0.01861482699582847
ambiguity margin: 0.6782577226161598
```

That confirms the coupled Stage 1 search was not merely unlucky along one
curve. The measured registered-pixel mechanism still lacks a useful transferred
governed operating point after decoupling the two calibration dimensions.

The rejection decomposition also sharpens the diagnosis:

```text
overall: 952 margin-only, 392 both, 0 distance-only
```

By family:

```text
final-brightness-unseen: 336 margin-only
final-shift-two: 336 margin-only
final-palette-c: 280 margin-only, 56 both
final-noncritical-patch: 336 both
```

This means the dominant bottleneck is margin separation, not a pure distance
threshold collapse, with the noncritical patch family additionally degrading
both gates at once.

## Interpretation

This is a stronger local result than frozen System B v2 at the ranking layer,
but it still does not justify promotion as a governed visual-address reader.

The evidence supports these narrower statements:

- bounded integer registration repairs part of the translation-locality failure;
- the repaired ranking still collapses under rejection calibration;
- local deterministic alignment alone is insufficient for a useful accepted
  operating point on the bounded arcade fixture.

## Next action

The next bounded experiment remains `translation_equivariant_template_correlation`.

Before Stage 2 freezes, its protocol should add one more control family:

```text
beyond-bounds translation shifts of magnitude 4 or 5 pixels
```

Because the registered Stage 1 mechanism cannot physically align those shifts,
they form the correct out-of-envelope safety control for bounded registration.
The acceptance claim to test is:

```text
the system rejects what its own registration cannot reach
```


## Wider interpretation

See [`visual-ai-research-status-after-registration.md`](visual-ai-research-status-after-registration.md) for the broader interpretation of the Stage 1 result and its position in the ZeroModel visual-AI research programme.
