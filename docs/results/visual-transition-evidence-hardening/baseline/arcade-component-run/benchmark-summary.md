# Visual Transition Debugging Benchmark -- Summary

## Executive result

ZeroModel (P4A field partitioning + P18A transition evidence + P18B action-conditioned conformance) provides a **measurable but narrow** localization advantage over raw pixel differencing: it adds reliable component-name attribution and catches two specific classes of fault (declared-stable-region violations, and declared-must-change absences) that pixel differencing cannot represent at all. It is blind to faults in regions it has no crisp expectation for (alien hit/miss, cooldown state outside FIRE) and to faults that preserve the correct *label* while flipping direction/target. See 'Where ZeroModel failed' below.

## Exact environment

- **git_commit**: 85d4fd50607cbef607ddbe4a5f73c1468ad76955
- **python_version**: 3.11.4
- **numpy_version**: 2.4.6
- **dev_episode_count**: 2
- **eval_episode_count**: 4
- **dev_transition_count**: 36
- **eval_transition_count**: 72
- **dev_seeds**: [0, 1]
- **eval_seeds**: [1000000, 1000001, 1000002, 1000003]
- **categories**: ['tank_moves_left', 'tank_moves_right', 'tank_remains_stationary_at_boundary', 'fire_hits_advances_target', 'fire_hits_clears_wave', 'fire_miss_activates_cooldown', 'cooldown_clears', 'no_op', 'fire_no_projectile', 'fire_no_cooldown', 'alien_disappears_without_hit', 'tank_moves_wrong_action', 'tank_moves_wrong_direction', 'cooldown_changes_incorrectly', 'unrelated_alien_change', 'background_changes_unexpectedly', 'expected_target_remains_unchanged', 'extra_component_changes_alongside']
- **command**: C:\Projects\zeromodel\examples\visual_transition_benchmark\run.py --dev-episodes 2 --eval-episodes 4 --skip-render --output-dir docs/results/visual-transition-evidence-hardening/baseline/arcade-component-run
- **duration_seconds**: 0.882
- **warning_count**: 0

## Main metrics (evaluation split)

| Metric | Pixel diff | Privileged | ZeroModel |
|---|---:|---:|---:|
| Visible changed-component attribution micro-F1 | 0.938 | 1.000 | 1.000 |
| Component exact-set accuracy | 0.889 | 1.000 | 1.000 |
| Field-level mean recall | 0.938 | 1.000 | 1.000 |
| Missing-change detection rate (faulty only) | n/a (0 by construction) | 1.000 | 0.667 |
| Unexpected-change detection rate (faulty only) | n/a (0 by construction) | 1.000 | 0.500 |
| False alarm rate on correct transitions | n/a | 0.000 | 0.125 |
| Mean false-implicated components | 0.000 | 0.167 | 0.167 |

ZeroModel vs. pixel-diff, per transition: **22.2% better**, 77.8% equal, 0.0% worse (n=72).

## Fault detection results by fault type

| Fault type | n | ZeroModel missing-detect | ZeroModel unexpected-detect | ZeroModel false-implicated (mean) |
|---|---:|---:|---:|---:|
| fire_no_projectile | 4 | 0.000 | 0.000 | 0.000 |
| fire_no_cooldown | 4 | 1.000 | 0.000 | 1.000 |
| alien_disappears_without_hit | 4 | 0.000 | 0.000 | 0.000 |
| tank_moves_wrong_action | 4 | 0.000 | 1.000 | 0.000 |
| tank_moves_wrong_direction | 4 | 0.000 | 0.000 | 0.000 |
| cooldown_changes_incorrectly | 4 | 0.000 | 0.000 | 0.000 |
| unrelated_alien_change | 4 | 0.000 | 0.000 | 0.000 |
| background_changes_unexpectedly | 4 | 0.000 | 1.000 | 0.000 |
| expected_target_remains_unchanged | 4 | 1.000 | 0.000 | 1.000 |
| extra_component_changes_alongside | 4 | 0.000 | 1.000 | 0.000 |

## Scientific interpretation

- **What this demonstrates**: within this controlled arcade environment, field-partitioned transition evidence plus action-conditioned conformance checking localizes known visual transitions to named regions, and catches both an unexpected background mutation and a declared movement that silently failed to occur -- two things raw pixel differencing structurally cannot express.
- **What it suggests**: aggregating pixel evidence into declared, named fields is a cheap, effective way to add component-level attribution on top of pixel differencing, and declaring per-action expectations over those fields is enough to catch anomalies in regions where the expectation is crisp (tank motion, background stability).
- **What it does not establish**: it does not establish general vision, causal discovery, or semantic understanding from pixels. It does not generalize past this environment's fixed layout. It cannot resolve faults that require hidden state (hit/miss, exact cooldown counter, movement direction) that isn't recoverable from frames + action alone -- those require either richer metadata or acceptance of the 'unexplained, needs review' bucket instead of a pass/fail claim.

## Architecture implications

- **Genuinely needed**: P4A field partitioning (`fields.py`), P18A transition evidence (`transition_evidence.py`), P18B action-conditioned conformance (`transition_conformance.py`), and P6 region annotations (`expectations.py`, used only for declaring static bands).
- **Used for a secondary demonstration only**: P18C recurrent unexplained-transition discovery (`transition_discovery.py`) -- see the discovery note in this file; it is not required for the core per-transition metrics.
- **Bypassed entirely**: every certification/governance/promotion/lifecycle stage (P12-P17, P18D-P18G). None of it is needed to answer this benchmark's question.

## Recommendation

**Continue and strengthen the visual-debugging direction**, scoped narrowly to P4A/P18A/P18B(+P18C): the representation earns its keep on component attribution and on the two fault families it can structurally express. Do not extend the certification/governance chain on the strength of this result -- it was not exercised and was not needed.
