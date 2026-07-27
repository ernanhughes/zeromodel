# Evidence Contract Compiler -- Frozen Result Record (v1)

Full narrative and interpretation: [`docs/research/evidence-contract-representation-compiler.md`](../../research/evidence-contract-representation-compiler.md)
Claims boundary: [`docs/claims-audit.md`](../../claims-audit.md)
Prior stages' frozen records: [`value-aware-transition-contracts-v1/`](../value-aware-transition-contracts-v1/), [`cross-domain-visual-contracts-v1/`](../cross-domain-visual-contracts-v1/)

## Contents

- `environment.json` -- exact commit, versions, command, case counts, test tallies
- `compiler-summary.md` -- program-generated summary: the full per-case outcome table (status, selected candidate, dev/held-out accuracy, and all three reference-strategy accuracies)
- `compiler-results.json` -- full machine-readable results: every requirement, every candidate considered, every strategy's held-out evaluation, per case

The full generated evidence -- `all_dev_evaluations` per case is already included in `compiler-results.json` (development-split scores for every candidate in the bounded search, not just the selected one) -- is complete in this package; there is no larger uncommitted corpus for this stage (unlike stages 1-3, which additionally render per-transition diagnostic panels this stage does not produce).

## One-line result

11 of 12 declared evidence requirements compiled to a representation across two independently-built domains (arcade, warehouse), including automatic rediscovery of both previously hand-fixed representation bugs (the tank/robot max-aggregation tie; the cooldown/door dilution bug, repaired via a new development-only auto-narrowing decoder). The one non-compiling case, arcade alien target identity, correctly reports `insufficient_observability`: the sprite carries no identity marker in the rendered frame at all, not a representation the bounded search failed to find. Where the compiler and the historical hand-built representation both compile, they achieve identical held-out accuracy in every case; the compiler was never observed to beat or fall short of the manual baseline it validates.

## Repair note

An earlier merge (`29d314e`) shipped this compiler with two evidence gaps, both fixed in `a39c64d` before this frozen run: (1) the alien-identity case classified as `insufficient_representation` rather than `insufficient_observability`, because the decoder used for a property with no declared canonical vocabulary silently passed through a continuous, non-degenerate-looking value instead of being recognized as uninformative; (2) the runner compared only two naive reference strategies (fixed-coarse, always-pixel) with no comparison against the literal historical hand-built representation. Both are fixed and reflected in the numbers above; see `a39c64d`'s commit message and `docs/research/evidence-contract-representation-compiler.md` for the full account.
