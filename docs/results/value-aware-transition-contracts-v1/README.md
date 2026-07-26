# Value-Aware Transition Contracts -- Frozen Result Record (v1)

Full narrative and interpretation: [`docs/research/value-aware-transition-contracts.md`](../../research/value-aware-transition-contracts.md)
Claims boundary: [`docs/claims-audit.md`](../../claims-audit.md)
Prior stage's frozen record: [`docs/results/visual-transition-debugging-v1/`](../visual-transition-debugging-v1/)

## Contents

- `environment.json` -- exact commit, versions, command, transition counts, test tallies
- `value-benchmark-summary.md` -- the program-generated summary (executive result, accuracy/detection tables, by-category breakdown, interpretation)
- `value-benchmark-results.json` -- full machine-readable metrics (accuracy, fault localization, hidden-value-fault counts, component-attribution comparison, all broken out by split and by category)
- `representative-artifacts/` -- 5 curated diagnostic panels (not the full generated corpus; see below)

The full generated evidence -- `value-transition-level-results.jsonl` (one row per evaluation transition) and 610 diagnostic PNGs -- is reproducible from the command in `environment.json` but is **not committed**, per the same policy as the prior stage's frozen record.

## Representative artifacts

| File | Transition | What it shows |
|---|---|---|
| `wrong-direction-detected.png` | `tank_moves_wrong_direction` | Component-level system (System C) reports this transition as entirely clean; System D's direction contract catches it. The primary result of this experiment. |
| `wrong-cooldown-value-detected.png` | `cooldown_activates_with_wrong_value` | Cooldown region visibly changes; System C is satisfied; the decoded intensity matches neither canonical level, and System D's cooldown-value contract catches it. |
| `wrong-magnitude-detected.png` | `tank_moves_too_far` | Correct direction, wrong distance -- "a component changes correctly but the relation is violated." |
| `wrong-target-unresolved.png` | `wrong_alien_disappears` | Honest negative: a legitimate-looking hit with a wrong target substitution. No contract here can catch it -- the correct target identity does not exist anywhere in the two frames or the action label. |
| `fire-no-projectile-unresolved.png` | `fire_no_projectile` (stage 1) | Stage 1's original blind spot, confirmed still unresolved under value-awareness. Reused from the stage-1 frozen artifact set. |

## One-line result

1,178 of 1,800 faulty transitions in this evaluation split were component-label-clean yet value-wrong. Wrong direction, wrong cooldown value, and wrong magnitude are now detected; wrong target identity is not, and is not expected to be without richer non-privileged observations.
