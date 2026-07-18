# Visual local baseline showdown

**Status:** executed  
**Execution date:** Saturday, July 18, 2026  
**Evidence package:** `docs/results/visual-local-baseline-showdown-v1/`

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
