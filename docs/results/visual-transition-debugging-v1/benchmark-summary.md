# Visual Transition Debugging Benchmark -- Summary

## Executive result

ZeroModel (P4A field partitioning + P18A transition evidence + P18B action-conditioned conformance) provides a **measurable but narrow** localization advantage over raw pixel differencing: it adds reliable component-name attribution and catches two specific classes of fault (declared-stable-region violations, and declared-must-change absences) that pixel differencing cannot represent at all. It is blind to faults in regions it has no crisp expectation for (alien hit/miss, cooldown state outside FIRE) and to faults that preserve the correct *label* while flipping direction/target. See 'Where ZeroModel failed' below.

## Exact environment

- **git_commit**: 80b608d0cdc4d8d8f07af3735470b4a8e7de7048
- **python_version**: 3.11.4
- **numpy_version**: 2.2.3
- **dev_episode_count**: 40
- **eval_episode_count**: 120
- **dev_transition_count**: 720
- **eval_transition_count**: 2160
- **dev_seeds**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
- **eval_seeds**: [1000000, 1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000007, 1000008, 1000009, 1000010, 1000011, 1000012, 1000013, 1000014, 1000015, 1000016, 1000017, 1000018, 1000019, 1000020, 1000021, 1000022, 1000023, 1000024, 1000025, 1000026, 1000027, 1000028, 1000029, 1000030, 1000031, 1000032, 1000033, 1000034, 1000035, 1000036, 1000037, 1000038, 1000039, 1000040, 1000041, 1000042, 1000043, 1000044, 1000045, 1000046, 1000047, 1000048, 1000049, 1000050, 1000051, 1000052, 1000053, 1000054, 1000055, 1000056, 1000057, 1000058, 1000059, 1000060, 1000061, 1000062, 1000063, 1000064, 1000065, 1000066, 1000067, 1000068, 1000069, 1000070, 1000071, 1000072, 1000073, 1000074, 1000075, 1000076, 1000077, 1000078, 1000079, 1000080, 1000081, 1000082, 1000083, 1000084, 1000085, 1000086, 1000087, 1000088, 1000089, 1000090, 1000091, 1000092, 1000093, 1000094, 1000095, 1000096, 1000097, 1000098, 1000099, 1000100, 1000101, 1000102, 1000103, 1000104, 1000105, 1000106, 1000107, 1000108, 1000109, 1000110, 1000111, 1000112, 1000113, 1000114, 1000115, 1000116, 1000117, 1000118, 1000119]
- **categories**: ['tank_moves_left', 'tank_moves_right', 'tank_remains_stationary_at_boundary', 'fire_hits_advances_target', 'fire_hits_clears_wave', 'fire_miss_activates_cooldown', 'cooldown_clears', 'no_op', 'fire_no_projectile', 'fire_no_cooldown', 'alien_disappears_without_hit', 'tank_moves_wrong_action', 'tank_moves_wrong_direction', 'cooldown_changes_incorrectly', 'unrelated_alien_change', 'background_changes_unexpectedly', 'expected_target_remains_unchanged', 'extra_component_changes_alongside']
- **command**: C:\Projects\zeromodel\examples\visual_transition_benchmark\run.py --dev-episodes 40 --eval-episodes 120 --output-dir C:/Projects/zeromodel/artifacts/visual_transition_benchmark
- **duration_seconds**: 61.672
- **warning_count**: 0

## Main metrics (evaluation split)

| Metric | Pixel diff | Privileged | ZeroModel |
|---|---:|---:|---:|
| Component micro-F1 | 0.937 | 1.000 | 1.000 |
| Component exact-set accuracy | 0.889 | 1.000 | 1.000 |
| Field-level mean recall | 0.937 | 1.000 | 1.000 |
| Missing-change detection rate (faulty only) | n/a (0 by construction) | 1.000 | 0.681 |
| Unexpected-change detection rate (faulty only) | n/a (0 by construction) | 1.000 | 0.500 |
| False alarm rate on correct transitions | n/a | 0.000 | 0.125 |
| Mean false-implicated components | 0.000 | 0.174 | 0.174 |

ZeroModel vs. pixel-diff, per transition: **23.0% better**, 77.0% equal, 0.0% worse (n=2160).

## Fault detection results by fault type

| Fault type | n | ZeroModel missing-detect | ZeroModel unexpected-detect | ZeroModel false-implicated (mean) |
|---|---:|---:|---:|---:|
| fire_no_projectile | 120 | 0.000 | 0.000 | 0.000 |
| fire_no_cooldown | 120 | 1.000 | 0.000 | 1.000 |
| alien_disappears_without_hit | 120 | 0.000 | 0.000 | 0.000 |
| tank_moves_wrong_action | 120 | 0.000 | 1.000 | 0.000 |
| tank_moves_wrong_direction | 120 | 1.000 | 0.000 | 0.133 |
| cooldown_changes_incorrectly | 120 | 0.000 | 0.000 | 0.000 |
| unrelated_alien_change | 120 | 0.000 | 0.000 | 0.000 |
| background_changes_unexpectedly | 120 | 0.000 | 1.000 | 0.000 |
| expected_target_remains_unchanged | 120 | 1.000 | 0.000 | 1.000 |
| extra_component_changes_alongside | 120 | 0.000 | 1.000 | 0.000 |

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
