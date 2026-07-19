# Stage 3 v3 Operational Contract

Date: July 18, 2026

Amendment parent: `ad2093590cde95ad1dc984f0573f452693002717`

## Purpose

This document freezes the operational semantics for the Stage 3 v3 current-frame joint-evidence instrument before implementation.

The contract authorizes representation corrections only. It does not authorize:

- threshold tuning from v2 outcomes;
- region changes;
- benchmark sampling changes;
- new perturbation families;
- architecture selection;
- calibration;
- V5 implementation;
- final evaluation;
- expanded scientific claims.

## Versions and identities

- benchmark version: `zeromodel-video-discriminative-evidence-stage3/v3`
- generator version: `zeromodel-video-discriminative-generator/v3`
- provider version: `zeromodel-video-discriminative-provider/v3`
- evidence mechanics version: `zeromodel-video-joint-evidence-mechanics/v1`
- candidate-set version: `zeromodel-video-joint-candidate-set/v1`
- decision version: `zeromodel-video-joint-evidence-decision/v1`

Architecture identities:

- `A3`: direct corrected regional correlation
- `B3`: joint discriminative candidate fit
- `C3`: pairwise joint support and contradiction
- `D3`: combined joint fit and pairwise safety

Output directory:

- `docs/results/video-discriminative-local-evidence-v3/`

Seed material:

- `zeromodel-stage3-v3-final|ad2093590cde95ad1dc984f0573f452693002717`

Seed digest:

- `sha256:cf1c355325f847d7359243a9d0943d7a770dc9f99f78973b2a94d49bca43c2ef`

## Preserved benchmark scope

The v3 benchmark preserves:

- 112 prototype rows;
- 224 development observations;
- 12 evaluation rows selected by `step_sample_with_midpoint_and_last_dedup/v1`;
- existing family definitions;
- existing region geometry;
- existing registration bounds;
- existing source scope;
- existing split roles.

V3 changes evidence representation only. It does not change the data-generating question.

## Immutable concepts

The implementation must expose explicit immutable concepts for:

- region specification;
- candidate mask;
- pairwise mask;
- calibration schema;
- region fit evidence;
- pairwise region evidence;
- row candidate;
- candidate set;
- decision;
- provider;
- self-retrieval result.

Every persisted concept must include:

- explicit version;
- deterministic serialization;
- deterministic digest;
- validation;
- semantic identity binding.

Validation must reject:

- unsupported versions;
- duplicate rows or prototypes;
- duplicate unordered pair identities;
- inconsistent geometry;
- inconsistent row/action identity;
- asymmetric pairwise masks;
- nonfinite evidence;
- negative masses;
- masks outside `[0, 1]`;
- pairwise identity mismatch;
- changed prototype digest;
- changed development digest;
- changed region digest;
- changed source scope;
- changed amendment identity;
- changed operational-contract identity;
- changed architecture semantics.

## Candidate mask semantics

Each candidate row carries four distinct concepts:

- row-informative mask;
- stable mask;
- candidate-fit mask;
- action-conflict declaration.

### Row-informative mask

A candidate pixel is row-informative when the candidate differs beyond the frozen intensity tolerance from at least one competitor.

Frozen declaration:

```text
max(abs(candidate - competitor)) > intensity_tolerance
```

### Stable mask

The stable mask is development-backed and retains the complete v2 stability rule. Every provider row must have valid development input.

### Candidate-fit mask

For B3 and D3 candidate-fit scoring:

```text
candidate_fit_mask = row_informative_mask × stable_mask
```

This mask is binary or explicitly bounded. It must not be multiplied by the v2 nearest-competitor separation weight.

### Evidence accounting

For every candidate, the implementation must persist separate fields for:

- `declared_informative_mass`
- `stable_informative_mass`
- `available_geometric_mass`
- `available_candidate_fit_mass`
- `pairwise_discriminative_mass`
- `actual_scored_mass`

No field may silently substitute for another.

## Pairwise masks

For every unordered candidate-competitor pair, derive one symmetric pairwise mask.

Frozen definition:

```text
pairwise_stable_mask = minimum(stable_candidate, stable_competitor)
pairwise_difference_mask = indicator(abs(candidate_prototype - competitor_prototype) > intensity_tolerance)
pairwise_mask = pairwise_stable_mask × pairwise_difference_mask
```

Pair identity uses the canonical sorted row pair:

```text
(min(row_a, row_b), max(row_a, row_b))
```

Direction is stored separately from mask identity.

Pairwise invariants:

```text
mask(a, b) == mask(b, a)
margin(a, b) == -margin(b, a)
```

within the frozen numerical epsilon.

Zero-mass pairwise masks are neutral evidence. They may prevent exact superiority. They may not create support or contradiction.

## Registration and aligned evidence

V3 continues using deterministic bounded registration. For each candidate and region retain:

- selected `dx`;
- selected `dy`;
- direct registration distance;
- direct registration score;
- geometric overlap;
- candidate-fit available mass;
- pairwise available mass;
- registration tie-break reason;
- registration-contract digest.

Registration may align evidence. It may not select the policy row.

## Architecture formulas

### A3

For every candidate and region:

1. register the candidate prototype to the observation;
2. use the direct registered candidate-to-observation correlation distance;
3. convert it to normalized similarity;
4. aggregate using region weights and available fractions.

Frozen regional similarity:

```text
normalized_regional_similarity = max(0, 1 - registration_distance / 2)
```

Frozen aggregate:

```text
A3(candidate) =
sum(region_weight × available_region_fraction × normalized_regional_similarity)
/
sum(region_weight × available_region_fraction)
```

If the denominator is zero, A3 has no sufficient evidence.

### B3

For each candidate and region, over the aligned candidate-fit mask:

```text
normalized_absolute_error = abs(observation - candidate_prototype) / 255
weighted_error = sum(mask × normalized_absolute_error) / sum(mask)
regional_joint_fit = 1 - weighted_error
```

Frozen candidate aggregate:

```text
B3(candidate) =
1 -
(
    sum(region_weight × sum(mask × normalized_absolute_error))
    /
    sum(region_weight × sum(mask))
)
```

Clamp only for numerical safety to `[0, 1]`.

### C3

For candidate `c` and competitor `j`, over the same aligned symmetric pairwise mask:

```text
fit_c = 1 - weighted_mean(abs(observation - prototype_c) / 255)
fit_j = 1 - weighted_mean(abs(observation - prototype_j) / 255)
pairwise_margin(c, j) = fit_c - fit_j
```

Persist regional contributions and pairwise mass. The frozen candidate summary uses:

- minimum pairwise margin;
- mean pairwise margin;
- mean positive pairwise margin;
- mean negative pairwise margin;
- minimum conflicting-action margin;
- mean conflicting-action margin;
- neutral pair count;
- positive pair count;
- negative pair count.

Frozen C3 strength:

```text
C3(candidate) = clamp(0.5 + 0.5 × minimum_pairwise_margin, 0, 1)
```

Interpretation:

- `> 0.5`: candidate beats every scored competitor;
- `= 0.5`: at least one competitor is unresolved;
- `< 0.5`: at least one competitor beats the candidate.

Zero-mass pairs contribute neutral margin `0`.

### D3

Frozen components:

```text
B3 component = candidate_joint_fit
C3 component = 0.5 + 0.5 × minimum_pairwise_margin
D3(candidate) = minimum(B3 component, C3 component)
```

D3 adds no fifth evidence mechanism.

## Ranking

Use a higher-is-better semantic score contract.

Ranking order:

1. greater semantic candidate strength;
2. greater actual scored mass;
3. greater available candidate-fit mass;
4. greater minimum pairwise margin where present;
5. greater conflicting-action margin where present;
6. lexical row ID;
7. lexical prototype ID.

Persist:

- `semantic_tie_group_size`
- `semantic_tie_group_rows`
- `winner_selected_by_semantic_strength`
- `trace_order`

Lexical ordering may order traces only. It may not manufacture perceptual uniqueness.

## Strict tie handling and exact safety

For every candidate compute:

```text
candidate_superiority_margin =
candidate_strength - maximum_strength_of_any_other_candidate
```

The winner is semantically unique only when:

```text
winner_strength > runner_up_strength + numerical_tie_epsilon
```

Exact acceptance additionally requires:

```text
winner_superiority_margin >= configured_exact_margin
```

When configured exact margin is zero, strict superiority is still required.

Therefore:

```text
winner_strength == runner_up_strength -> no exact acceptance
```

This invariant must hold for two-way, same-action, conflicting-action, 112-way, lexical-order-reversed, and prototype-order-reversed ties.

## Candidate-set behaviour

Outcomes remain:

- `exact_row_accepted`
- `candidate_set_available`
- `no_sufficient_evidence`

Exact acceptance requires:

- one semantic winner;
- strict superiority;
- all exact gates pass;
- no unresolved conflicting-action competitor remains;
- sufficient current-frame evidence.

Candidate sets:

- include all semantically tied qualifying rows;
- never select only the lexical first row from a semantic tie;
- are never silently truncated.

If more than three rows qualify:

```text
outcome = no_sufficient_evidence
reason = candidate_set_too_large
```

## Self-retrieval gates

### Shared instrument gate

The shared instrument is invalid if any of these fail:

- prototype closure;
- development closure;
- mask identity;
- pairwise symmetry;
- provider/direct-construction equivalence;
- deterministic regeneration;
- evidence accounting;
- tie safety;
- canonical observation presence;
- expected rows missing from ranking.

### Per-architecture gate

Each architecture must be classified independently as:

- `eligible_self_retrieval`
- `ineligible_self_retrieval`
- `invalid_architecture_instrument`

For visually unique canonical rows, eligibility requires:

```text
expected row is unique semantic top-1 for all 112 canonical observations
```

Architecture-local failure does not invalidate passing architectures.

If every architecture fails, the block stops at:

```text
invalid_architecture_measurement
```

## Calibration schema without calibration

Define immutable parameter fields for future selection only:

- `minimum_actual_scored_mass`
- `minimum_available_candidate_fit_fraction`
- `minimum_candidate_joint_fit`
- `minimum_pairwise_margin`
- `minimum_conflicting_action_margin`
- `exact_winner_threshold`
- `exact_winner_margin`
- `candidate_relative_margin`
- `maximum_candidate_set_size`

Threshold values are intentionally not frozen from v2 outcomes in this block.

## Invalid-instrument conditions

Stop the block at the exact failure boundary if any shared gate fails, including:

- pairwise symmetry;
- deterministic identity;
- prototype closure;
- development closure;
- provider equivalence;
- strict tie safety;
- evidence accounting;
- regeneration.

## Stop boundary

This block ends after:

- v3 operational contract is frozen;
- v3 contracts and provider are implemented;
- v3 benchmark identities are frozen;
- exhaustive canonical self-retrieval is run for A3/B3/C3/D3;
- per-architecture self-retrieval status is persisted;
- instrument verification is persisted;
- no architecture selection has run;
- no calibration has run;
- no V5 has been implemented;
- no final evaluation has run.
