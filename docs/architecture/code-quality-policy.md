# Code Quality Policy

ZeroModel uses a repository quality ratchet rather than a one-time cleanup sweep.
Existing debt may remain temporarily. Existing debt may not increase. New code must meet
the current standard. Touched code should improve. Exceptions must be explicit, measured,
owned, and reduced.

## Module boundaries

- One dominant responsibility belongs in each module.
- New modules should normally remain below 500 lines.
- New production modules must remain below 800 lines.
- Local import cycles are not allowed.
- Production code must not import from `tests`.
- Pure computation should stay separate from filesystem I/O.
- Verification code stays downstream from runtime implementation.
- Benchmark orchestration stays separate from reusable runtime mechanics.

## Function boundaries

- Functions should normally remain below 50 lines.
- New functions must remain below 100 lines.
- Hidden global mutable state should be avoided.
- Import-time computation should be avoided.
- Configuration identity should be passed explicitly.
- Public APIs should be deliberately exported.

## Scientific-code rules

- Published schema versions remain immutable.
- Digest behavior remains deterministic.
- Stored status fields are not treated as independent verification.
- Experimental claims remain separate from production capability.
- Research documents are not implementation proof.

## Legacy exception policy

`zeromodel/video_action_set_benchmark.py` is temporarily grandfathered at its
present scale and must shrink through later behavior-preserving extraction PRs.
The exception ceiling must never be raised merely to permit new additions.

Legacy exceptions must be file-specific. Each exception needs a reason, a numeric
line ceiling, and an owner. An exception does not suppress syntax errors, import
cycles, or forbidden architecture edges.
