# Stage P18A — Observation Transition Evidence VPM

## Objective

Materialize the exact visual change between a source observation and its result as a deterministic, addressable VPM.

```text
before image
    ↓ SourceVPMDTO
exact P4A field schema
    ↑
after image
    ↓ SourceVPMDTO
fieldwise before/after measurement
    ↓
TransitionEvidenceVPMDTO
    ├── mean absolute change
    ├── mean signed change
    ├── changed-value fraction
    ├── exact field identities
    ├── optional tank/alien/object annotation identities
    └── deterministic grayscale change VPM
```

The source and result remain complete canonical Source VPMs. P18A does not compress either observation into an opaque vector. It creates a third artifact that records where and how strongly the normalized pixels changed under the declared field contract.

## Measurements

For every exact field, P18A records:

- normalized mean intensity before the transition;
- normalized mean intensity after the transition;
- mean absolute pixel change;
- mean signed pixel change;
- the number and fraction of values crossing the explicit change threshold;
- semantic annotation identities already bound to the field.

The grayscale transition VPM renders mean absolute change. When channel-separated fields occupy the same spatial region, the rendered pixel keeps the maximum channel change so evidence is not hidden by averaging channels together.

## Annotation boundary

P18A can bind existing `PerceptionRegionAnnotationDTO` identities such as `tank`, `alien`, `projectile`, or `control` to measured fields. Labels do not alter scores, create detections, or establish causality. They let later stages compare declared object expectations with observed state changes.

## Identity and rejection

The transition artifact identity binds:

- before and after Source VPM identities and pixel digests;
- field-schema and encoder identities;
- change threshold and measurement semantics;
- exact per-field measurements;
- annotation identities;
- rendered PNG digest.

P18A rejects incompatible source encoders, shapes, schemas, duplicate annotations, foreign annotation schemas, invalid thresholds, and identity or PNG tampering.

## What this enables

P18A is the first of the three transition-testing slices:

1. **P18A — observed transition map:** what changed between the image and the result?
2. **P18B — annotated transition conformance:** did the marked tank, alien, projectile, or relation fields change as expected?
3. **P18C — unexplained transition discovery:** which unmarked fields repeatedly change and may reveal a missing component or a faulty system assumption?

This is a testing instrument. It does not yet infer object labels, learn causal structure, or decide that a changed field caused the recorded action.
