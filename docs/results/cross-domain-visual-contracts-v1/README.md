# Cross-Domain Visual Contract Replication -- Frozen Result Record (v1)

Full narrative and interpretation: [`docs/research/cross-domain-visual-contract-replication.md`](../../research/cross-domain-visual-contract-replication.md)
Claims boundary: [`docs/claims-audit.md`](../../claims-audit.md)
Prior stages' frozen records: [`visual-transition-debugging-v1/`](../visual-transition-debugging-v1/), [`value-aware-transition-contracts-v1/`](../value-aware-transition-contracts-v1/)

## Contents

- `environment.json` -- exact commit, versions, command, transition counts, test tallies (and the mid-execution fix note)
- `cross-domain-summary.md` -- program-generated summary: capability table, hidden-fault headline
- `cross-domain-results.json` -- full machine-readable results: capability table with replication decisions, both domains' full metric reports
- `domain-results/arcade.json`, `domain-results/warehouse.json` -- complete per-domain metric breakdowns
- `representative-artifacts/` -- 7 curated diagnostic panels (not the full generated corpus; see below)

The full generated evidence -- `transition-level-results.jsonl` (one row per evaluation transition across both domains) and the complete diagnostic-panel corpus -- is reproducible from the command in `environment.json` but is **not committed**, per the same policy as the prior stages' frozen records.

## Representative artifacts

| File | Domain / transition | What it shows |
|---|---|---|
| `arcade-wrong-direction-detected.png` | arcade / `tank_moves_wrong_direction` | Stage 2's flagship result, reused as the arcade-side baseline for this comparison. |
| `arcade-fire-no-projectile-unresolved.png` | arcade / `fire_no_projectile` | Stage 1's original blind spot, still unresolved. |
| `arcade-background-fault-detected.png` | arcade / `background_changes_unexpectedly` | Stage 1's strongest presence-level catch, reused as the baseline. |
| `warehouse-wrong-direction-detected.png` | warehouse / `robot_moves_wrong_direction` | The direction-fix result **replicated** in an unrelated domain: component-level analysis reports this transition as clean; the value layer catches it. |
| `warehouse-wall-fault-detected.png` | warehouse / `wall_changes_unexpectedly` | Presence-level catch replicated: an always-stable region flagged correctly. |
| `warehouse-zorder-occlusion-unresolved.png` | warehouse / `push_advances_robot_without_crate` | A genuinely new, domain-specific blind spot: the renderer's draw order hides a crate that failed to follow the robot. Unrecoverable from pixels by any system, including the privileged baseline if it were pixel-based. |
| `warehouse-wrong-crate-identity-unresolved.png` | warehouse / `wrong_crate_moves` | Identity remains unresolved: the wrong crate's marker appears at the pushed-to cell, and no non-privileged contract can tell. |

## One-line result

Component-level attribution and the one implemented cross-field relation replicated cleanly (>=0.95 in both domains, same scoring functions, two unrelated renderers). Direction/value correctness clustered at 82-91% in *both* domains, consistently depressed by the same class of precondition-blocked edge case -- a replicated *limitation*, not domain noise. Visible object identity was measured at 25.0% correctness and confirmed to require a representation change, not a smarter analyzer.
