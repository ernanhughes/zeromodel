# Evidence Contract Compiler -- Summary

## Exact environment

- **git_commit**: aea9cab1b4ef2abb46548c13a93f8c4a09638900
- **python_version**: 3.11.4
- **numpy_version**: 2.2.3
- **dev_samples_per_category**: 15
- **eval_samples_per_category**: 40
- **case_count**: 12
- **duration_seconds**: 841.014
- **command**: C:\Projects\zeromodel\examples\visual_transition_benchmark\compiler_run.py --dev-samples 15 --eval-samples 40 --output-dir examples/visual_transition_benchmark/artifacts/evidence_contract_compiler

## Per-case outcomes

| Domain | Case | Status | Selected (dev) | Dev acc. | Held-out acc. | Fixed-coarse held-out | Always-pixel held-out | Manual held-out |
|---|---|---|---|---:|---:|---:|---:|---:|
| arcade | tank_presence | compiled | 3x4 presence_threshold | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| arcade | tank_position | compiled | 3x4 argmax_field | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| arcade | tank_direction | compiled | 3x4 signed_delta_over_position | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| arcade | tank_movement_magnitude | compiled | 3x4 exact_delta_over_position | 1.000 | 1.000 | 1.000 | 0.853 | 1.000 |
| arcade | cooldown_value | compiled | 1x1 dominant_field_value | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| arcade | alien_target_identity | insufficient_observability | - | - | - | 0.000 | 0.000 | n/a |
| warehouse | robot_position | compiled | 6x6 argmax_field | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| warehouse | robot_direction | compiled | 6x6 signed_delta_over_position | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| warehouse | robot_movement_magnitude | compiled | 6x6 exact_delta_over_position | 1.000 | 1.000 | 0.810 | 1.000 | 1.000 |
| warehouse | battery_value | compiled | 4x10 nearest_permitted_value | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| warehouse | door_state | compiled | 1x1 dominant_field_value | 0.957 | 0.957 | 0.000 | 0.000 | 0.957 |
| warehouse | crate_identity | compiled | 1x1 local_marker_pattern | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |

Manual = the literal historical hand-built representation from `compiler/MANUAL_REPRESENTATION_INVENTORY.md`, evaluated on the same held-out split as every other strategy (not a cited number from a different run). `n/a` marks `alien_target_identity`, for which no manual representation was ever successfully built.

## Status counts

- **compiled**: 11
- **insufficient_observability**: 1

## Known limitations surfaced per case

- **arcade/alien_target_identity**: insufficient_observability: no representation can recover this property

