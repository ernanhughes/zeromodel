# Stage P4B relevance semantics

P4B measures field/action association with `eta_squared_of_field_mean_by_action`.

For each field and training interaction:

1. extract the exact field bytes using the P4A schema;
2. reduce the field to its normalized mean intensity;
3. partition those scalar values by action label;
4. report the fraction of total variation explained by between-action variation.

The resulting score is bounded in `[0, 1]` and is retained together with support
count, action count, within-action variance, and between-action variance.

The Evidence VPM PNG is a deterministic grayscale rendering of the exact scores.
For schemas with separate channel fields, overlapping channel relevance is rendered
using the maximum score while every channel-specific score remains available in the
DTO.

These values describe predictive association only. They are not causal effects,
necessity, sufficiency, calibrated confidence, or intervention results. Those
claims require P4C controls.
