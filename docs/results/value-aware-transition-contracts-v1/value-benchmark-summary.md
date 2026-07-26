# Value-Aware Transition Contracts -- Summary

## Executive result

Value-aware ZeroModel (System D) **resolves stage 1's key blind spot**: wrong-direction tank faults, which stage 1 could not flag at all (component label looks correct either way), are now caught by an exact direction contract. Across this evaluation split, **1178 of 1800 faulty transitions look completely clean to the component-level system but are demonstrably value-wrong** -- exactly the failure mode a correct component label can hide. Target/alien identity correctness remains an honest, unresolved blind spot: no non-privileged contract here can name the *correct* next alien without the hidden alien queue.

## Exact environment

- **git_commit**: f28d8a1611f667a92f36f93ae4d3909aa91050aa
- **python_version**: 3.11.4
- **numpy_version**: 2.2.3
- **dev_episode_count**: 40
- **eval_episode_count**: 120
- **eval_transition_count**: 2760
- **reused_stage1_transition_count**: 2160
- **new_value_fault_transition_count**: 600
- **value_fault_categories**: ['tank_moves_too_far', 'cooldown_activates_with_wrong_value', 'cooldown_decreases_to_wrong_value', 'wrong_alien_disappears', 'two_aliens_disappear_instead_of_one']
- **command**: C:\Projects\zeromodel\examples\visual_transition_benchmark\value_run.py --dev-episodes 40 --eval-episodes 120 --output-dir C:/Projects/zeromodel/artifacts/value_aware_transition_contracts
- **duration_seconds**: 54.126
- **warning_count**: 0

## Value-level accuracy (decoded value vs. true simulated state)

| Split | n | Movement-direction | State-delta (exact) | Cooldown-value | Target-selection |
|---|---:|---:|---:|---:|---:|
| all | 2760 | 0.870 | 0.826 | 0.826 | 0.783 |
| reused stage-1 categories | 2160 | 0.833 | 0.833 | 0.889 | 0.833 |
| new value-fault categories | 600 | 1.000 | 0.800 | 0.600 | 0.600 |
| ordinary (non-faulty) | 960 | 1.000 | 1.000 | 1.000 | 1.000 |

## Value-level fault localization (ZeroModel's own, non-privileged flags)

- **all**: detection_rate=0.769 (n_relevant=1560), false_alarm_rate_on_correct=0.100 (n_clean=1200)
- **reused stage-1 categories**: detection_rate=0.875 (n_relevant=960), false_alarm_rate_on_correct=0.100 (n_clean=1200)
- **new value-fault categories**: detection_rate=0.600 (n_relevant=600), false_alarm_rate_on_correct=0.000 (n_clean=0)

## Component-level still correctly reported alongside (unchanged stage-1 metrics)

Visible changed-component attribution micro-F1: pixel_diff=0.956, privileged=1.000, zeromodel=1.000 (identical mechanism to stage 1 -- included here only so component-level and value-level results sit side by side, never conflated).

## By-category breakdown

| Category | n | Direction acc. | Delta acc. | Cooldown acc. | Target acc. | Value-fault detect | Relation-flag rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| tank_moves_left | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| tank_moves_right | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| tank_remains_stationary_at_boundary | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| fire_hits_advances_target | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| fire_hits_clears_wave | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| fire_miss_activates_cooldown | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| cooldown_clears | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| no_op | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| fire_no_projectile | 120 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| fire_no_cooldown | 120 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| alien_disappears_without_hit | 120 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| tank_moves_wrong_action | 120 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| tank_moves_wrong_direction | 120 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| cooldown_changes_incorrectly | 120 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| unrelated_alien_change | 120 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| background_changes_unexpectedly | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| expected_target_remains_unchanged | 120 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| extra_component_changes_alongside | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| tank_moves_too_far | 120 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cooldown_activates_with_wrong_value | 120 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| cooldown_decreases_to_wrong_value | 120 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| wrong_alien_disappears | 120 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| two_aliens_disappear_instead_of_one | 120 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Scientific interpretation

- **What this demonstrates**: adding typed, decoded values on top of the existing P4A/P18A representation (no new perception-package code) resolves a specific, previously-documented blind spot (wrong movement direction) and adds a genuinely new capability (exact-magnitude and cooldown-value checks, plus one cross-field relation) using only frames + the action label -- no hidden simulator state.
- **What it suggests**: presence/absence conformance (stage 1) and value correctness (stage 2) are complementary, not substitutable -- a system needs both, reported separately, or a correct label will silently hide a wrong value.
- **What it does not establish**: target/alien *identity* correctness remains unresolved without privileged state -- target-selection accuracy is reported here only as a ground-truth comparison, not as something System D can assert on its own. This is the same class of limitation stage 1 reported for hit/miss, now confirmed to persist under value-awareness too.

## Recommendation

**Continue and strengthen**: value-aware contracts are cheap (reuse of existing P4A/P18A, just at finer field resolution) and close a real, previously-documented gap. Do not attempt target-identity resolution without first deciding whether richer, still-non-privileged metadata (e.g. an episode-level alien-queue commitment declared once per episode) is an acceptable input -- that is a scope decision for a future stage, not a bug in this one.
