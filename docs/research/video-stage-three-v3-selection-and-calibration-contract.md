# Stage 3 v3 Selection And Calibration Contract

Date: July 18, 2026

Parent instrument branch: `research/video-discriminative-joint-evidence-v3`

Parent instrument head: `4790165de78557fce63d64e5f2b7ddfde04f1e98`

Parent PR: `#41`

## Purpose

This document freezes the B3-only empirical architecture-selection and calibration policy for the verified Stage 3 v3 current-frame joint-evidence instrument.

This block measures the already-verified B3 instrument on noncanonical selection and calibration observations. It does not modify or retest the representation hypothesis.

## Eligibility universe

The parent instrument must report:

- `instrument verification = pass`
- `B3 = eligible_self_retrieval`

If either condition is false, selection must refuse to run.

Empirical architecture-selection universe:

```text
["B3"]
```

Excluded architectures:

- `A3`: `ineligible_self_retrieval`
- `C3`: `ineligible_self_retrieval`
- `D3`: `ineligible_self_retrieval`

Frozen D3 gateway status:

```text
not_evaluated_architecture_ineligible
```

## Versions

- selection protocol: `zeromodel-video-joint-architecture-selection/v1`
- operating policy: `zeromodel-video-joint-operating-policy/v1`
- operating decision: `zeromodel-video-joint-operating-decision/v1`
- operating-point selection: `zeromodel-video-joint-operating-point-selection/v1`
- calibrated reader: `zeromodel-video-discriminative-reader/v3-b3`

## Phase-access rules

Selection may access only:

- `prototype`
- `diagnostic_development`
- `architecture_selection_benign`
- `architecture_selection_negative`

Calibration may access only:

- `prototype`
- `diagnostic_development`
- `benign_calibration`
- `rejection_calibration`

Selection and calibration may not access:

- `final_benign`
- `final_distinguishable_negative`
- `information_theoretic_control`

Prototype and development observations may be loaded for provider construction but may not enter benign or negative outcome denominators.

## Corrected candidate-gap semantics

Persist the new operating-policy field:

```text
maximum_candidate_relative_gap
```

Frozen definition:

```text
candidate_relative_gap = winner_strength - candidate_strength
```

Properties:

- winner gap = `0`
- non-winner gap >= `0`

Candidate-set inclusion requires:

```text
candidate_relative_gap <= maximum_candidate_relative_gap + epsilon
```

Monotonicity requirement:

```text
gap_limit_1 <= gap_limit_2
-> qualifying_set_1 is a subset of qualifying_set_2
```

The provisional empirical field name `candidate_relative_margin` is not used by the selection layer.

## B3 active and inactive gates

Candidate-set gates:

- `minimum_actual_scored_mass`
- `minimum_available_candidate_fit_fraction`
- `minimum_candidate_joint_fit`
- `minimum_conflicting_action_margin`
- `maximum_candidate_relative_gap`
- `maximum_candidate_set_size`

Exact gates:

- all candidate-set gates
- `exact_winner_threshold`
- strict semantic superiority
- `exact_winner_margin`

Inactive B3 field:

```text
minimum_pairwise_margin = -1.0
```

It is persisted but inactive for B3 empirical policy.

## Candidate-set behaviour

Construction order:

1. evaluate independent evidence gates for every row;
2. calculate each row’s gap from the semantic winner;
3. retain rows whose gap is within the configured maximum;
4. preserve all qualifying rows;
5. never use lexical ordering to remove a semantically qualifying row;
6. never silently truncate.

Exact acceptance requires:

- winner passes all candidate gates;
- winner passes exact threshold;
- winner is strictly semantically superior;
- superiority margin passes configured exact margin;
- conflicting-action margin passes;
- no unresolved equal-strength row exists.

Candidate-set outcome requires:

- exact acceptance not established;
- at least one row qualifies;
- every returned row independently passes evidence gates;
- every returned row lies within the configured maximum candidate gap;
- total qualifying size is within the configured set-size limit.

Oversized qualifying sets yield:

```text
outcome = no_sufficient_evidence
reason = candidate_set_too_large
```

No qualifying rows yield:

```text
outcome = no_sufficient_evidence
reason = no_qualifying_candidates
```

## Architecture-selection grid

Frozen B3 architecture-selection grid:

```text
minimum_actual_scored_mass: [1.0]
minimum_available_candidate_fit_fraction: [0.50, 0.75]
minimum_candidate_joint_fit: [0.90, 0.95]
minimum_conflicting_action_margin: [0.00, 0.01]
minimum_pairwise_margin: [-1.0]
exact_winner_threshold: [0.95, 0.98]
exact_winner_margin: [0.005, 0.010, 0.025]
maximum_candidate_relative_gap: [0.000, 0.010, 0.025]
maximum_candidate_set_size: [3]
```

Expected grid size:

```text
144
```

## Calibration grid

Frozen B3 calibration grid:

```text
minimum_actual_scored_mass: [1.0]
minimum_available_candidate_fit_fraction: [0.50, 0.75, 0.90]
minimum_candidate_joint_fit: [0.90, 0.95, 0.98]
minimum_conflicting_action_margin: [0.00, 0.01]
minimum_pairwise_margin: [-1.0]
exact_winner_threshold: [0.95, 0.98]
exact_winner_margin: [0.005, 0.020, 0.050]
maximum_candidate_relative_gap: [0.000, 0.010, 0.025]
maximum_candidate_set_size: [2, 3]
```

Prune only combinations where:

```text
exact_winner_threshold < minimum_candidate_joint_fit
```

Expected deterministic post-pruning size:

```text
540
```

## Metrics

Benign metrics:

- benign total
- exact accepted count
- exact correct count
- exact incorrect count
- exact-row coverage
- exact accepted precision
- candidate-set count
- useful candidate-set count
- truth-in-set count
- candidate-set recall
- union useful coverage
- unique-action candidate sets
- mixed-action candidate sets
- mean candidate-set size
- maximum candidate-set size
- false rejects
- per-family outcomes

Union useful coverage:

```text
(correct exact acceptances + truth-containing candidate sets) / benign total
```

Negative metrics:

- distinguishable negative total
- exact false accepts
- negative candidate-set support
- conflicting-action exact accepts
- same-action wrong-row exact accepts
- conflicting-action candidate-set support
- oversized-set outcomes
- no-evidence outcomes
- per-family outcomes

Integrity metrics:

- strict-tie violations
- lexical-outcome dependence
- missing expected candidates
- identity mismatches
- phase-access violations

Any nonzero integrity failure invalidates measurement.

## Feasibility

Selection or calibration operating points are feasible only when:

- exact false accepts = `0`
- negative candidate-set support = `0`
- conflicting-action exact accepts = `0`
- same-action wrong-row exact accepts = `0`
- strict-tie violations = `0`
- lexical-outcome dependence = `0`
- nonzero benign utility

Nonzero benign utility means:

```text
exact correct count + useful candidate-set count > 0
```

## Selection hierarchy

Among feasible B3 selection points:

1. maximize correct exact-row coverage;
2. maximize useful candidate-set recall;
3. maximize union useful coverage;
4. maximize exact accepted precision;
5. minimize mixed-action candidate-set count;
6. minimize mean candidate-set size;
7. minimize maximum candidate-set size;
8. apply conservative operating-policy ordering;
9. apply deterministic serialized-policy ordering.

Conservative ordering prefers:

1. greater minimum actual scored mass;
2. greater minimum available fraction;
3. greater minimum candidate joint fit;
4. greater minimum conflicting-action margin;
5. greater exact winner threshold;
6. greater exact winner margin;
7. smaller maximum candidate-relative gap;
8. smaller maximum candidate-set size;
9. lexical policy serialization.

## Calibration hierarchy

Among feasible B3 calibration points:

1. maximize correct exact-row coverage;
2. maximize useful candidate-set recall;
3. maximize union useful coverage;
4. maximize exact accepted precision;
5. minimize mixed-action candidate sets;
6. minimize mean set size;
7. minimize maximum set size;
8. apply the same conservative ordering;
9. apply deterministic serialized-policy ordering.

## Status values

Selection statuses:

- `selected_architecture`
- `no_safe_architecture`
- `invalid_architecture_measurement`

Calibration statuses:

- `selected_operating_point`
- `no_feasible_operating_point`
- `not_run_no_selected_architecture`
- `invalid_calibration_measurement`

## Verification protocol

`--verify-pre-final-v3` must:

1. verify parent instrument identity;
2. verify B3 self-retrieval eligibility;
3. verify A3/C3/D3 exclusion;
4. verify selection-contract identity;
5. rebuild raw B3 rankings;
6. regenerate the 144-point selection grid;
7. reproduce selected-architecture status;
8. regenerate calibration when applicable;
9. reproduce selected-operating-point status;
10. verify phase-access audits;
11. verify no final observations were evaluated;
12. verify no V5 artifacts exist;
13. verify no selection artifact changed the generator identity.

Optional deep verification may rerun `--verify-v3-instrument` first.

## Stop outcomes

Outcome A:

```text
selection status = selected_architecture
selected architecture = B3
calibration status = selected_operating_point
pre-final verification = pass
```

Outcome B:

```text
selection status = no_safe_architecture
calibration status = not_run_no_selected_architecture
pre-final verification = pass
```

Outcome C:

```text
selection status = selected_architecture
selected architecture = B3
calibration status = no_feasible_operating_point
pre-final verification = pass
```

Outcome D:

Any integrity, identity, access, grid, or reproduction invariant fails.
