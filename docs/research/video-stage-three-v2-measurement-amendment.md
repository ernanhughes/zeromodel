# Stage 3 v2 measurement amendment

Date: July 18, 2026

## Reason for amendment

Stage 3 v1 is invalid as a measurement record. The frozen v1 audit under
`docs/results/video-discriminative-local-evidence-v1/measurement-audit/`
classifies the result as `invalid_multiple_failures`.

The invalidity is bounded and specific:

1. The current committed generator does not reproduce the frozen v1 artifacts.
   The committed benchmark manifest records `nonprototype_row_sample_size = 12`,
   while the current benchmark code declares `NON_PROTOTYPE_ROW_SAMPLE_SIZE = 4`,
   and the rebuilt benchmark/split/selection artifacts differ semantically.
2. The provider prototype universe is the full canonical policy-row set, but the
   prototype split manifest in the current committed generator describes only the
   sampled evaluation rows rather than the complete provider prototype universe.
3. Development-backed stability is only present for the sampled rows in the
   current committed generator, while the provider masks and prototype universe
   span all canonical rows.

This amendment exists to correct measurement integrity only. It is not a
license to reinterpret v1 as Kill B, tune thresholds from the invalid v1
outcome, or widen the scientific question.

## Unchanged scientific question

The question remains:

> Can the fixed Stage 3 V4 frame-local discriminative-evidence provider produce
> a governed architecture-selection result under the predeclared bounded arcade
> protocol?

## Unchanged technical scope

The following remain unchanged for v2:

- Architectures `A`, `B`, `C`, and `D`
- V4 provider formulas and scoring semantics
- Materiality rules
- Candidate-set limit
- V5 rules
- Safety hierarchy
- The prohibition on using v1 results to tune thresholds

No new threshold values may be introduced because v1 produced zero utility.

## Required v2 corrections

Only the following corrections are allowed:

1. The prototype split must contain every prototype used by the provider.
2. The benchmark manifest must bind the complete provider prototype universe.
3. Diagnostic development must cover every row whose mask requires stability
   evidence.
4. Development transforms must remain fixed and label-independent.
5. The evaluated-row sampling algorithm and sample size must be frozen and
   reflected consistently in code and manifest.
6. Generator source identity must be bound into the benchmark manifest.
7. Pre-final verification must rebuild into a temporary directory.
8. Exact-frame sanity gates must pass before architecture selection.
9. Architecture selection must fail early when the shared upstream evidence
   layer is nonfunctional.

## Prohibited responses to v1 invalidity

The following are explicitly prohibited in response to v1 invalidity:

- Claiming Kill B from v1
- Claiming there is no feasible V4 architecture from v1
- Claiming candidate sets failed scientifically from v1
- Claiming Architecture D was scientifically ineligible from v1
- Implementing V5 before a valid v2 measurement exists
- Running final evaluation from v1

## v2 identity requirements

The v2 result set must use:

- Benchmark version: `zeromodel-video-discriminative-evidence-stage3/v2`
- Generator version: `zeromodel-video-discriminative-generator/v2`
- Output directory: `docs/results/video-discriminative-local-evidence-v2/`

The v2 seed must be derived from:

```text
sha256(
    "zeromodel-stage3-v2-final|" +
    <v2-measurement-amendment-commit-sha>
)
```

v2 must not reuse v1 final descriptors, observation IDs, clip IDs, benchmark
digest, split digest, or calibration identities.
