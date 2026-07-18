# ZeroModel Visual Programme Status Cut

**Date:** Saturday, July 18, 2026  
**Main SHA:** `336f7c4752e505d3be921be372c732bd19dc57b8`  
**Test state:** `python -m pytest -q` -> `209 passed, 1 skipped`  
**Status:** research cut, not programme completion

## Executive summary

The ZeroModel visual programme has produced a full first chapter of bounded
results. The deterministic policy core remains stable. The visual work has
shown that exact canonical visual addressing is possible on the declared arcade
fixture, that global DINOv2 CLS retrieval is not a governed reader for this
task, that normalized pixels preserve useful raw ranking signal but not a useful
governed operating point, and that bounded registration confirms spatial
alignment as a real mechanism while still failing to yield useful accepted
coverage.

The current programme cut closes documentation for that chapter. It does not
declare the visual programme complete.

## What is complete

- The bounded ZeroModel policy core is stable within its declared scope:
  deterministic artifacts, exact policy lookup, provenance, and related tests
  remain intact.
- The exact canonical observation-addressed reader is implemented, tested, and
  measured on the declared arcade fixture. Canonical evidence boundary:
  [visual-sign-reader.md](/C:/Projects/zeromodel/docs/research/visual-sign-reader.md).
- The first held-out visual benchmark chapter is complete as a research cycle:
  specification, implementation, measured negative result, evidence repair,
  adjudication, and follow-up mechanism test.

## What is measured

- `A / exact canonical deterministic reader`
  Research question: can canonical observations recover the same policy row and
  action as the symbolic path?
  Headline result: yes, on the declared exact-codeword fixture.
  Outcome classification: validated within bounded conditions.
  Evidence: [visual-sign-reader.md](/C:/Projects/zeromodel/docs/research/visual-sign-reader.md).
  Claim status: bounded positive.

- `Phase 1 global benchmark systems B, C, D, G`
  Research question: can approximate global readers recover governed policy
  addresses under held-out family variation?
  Headline result: no promoted governed reader emerged; DINOv2 and ridge probe
  were negative.
  Outcome classification: measured negative / unsupported depending system.
  Evidence: [visual-address-research-status.md](/C:/Projects/zeromodel/docs/research/visual-address-research-status.md),
  [visual-address-phase-one-v1](/C:/Projects/zeromodel/docs/results/visual-address-phase-one-v1).
  Claim status: global DINOv2 CLS path retired as promoted direction.

- `System B v2 / normalized pixels`
  Research question: does repaired normalized-pixel retrieval have a useful
  governed operating point?
  Headline result: raw top-1 exact row `1008 / 1344`, raw top-1 action
  `1302 / 1344`, accepted benign `0 / 1344`, false accepts `0 / 248`.
  Outcome classification: `C`.
  Evidence: [visual-address-system-b-v2](/C:/Projects/zeromodel/docs/results/visual-address-system-b-v2).
  Claim status: measured / unsupported.

- `R1 / registered_local_normalized_pixels`
  Research question: can bounded integer registration produce a useful governed
  operating point?
  Headline result: raw top-1 exact row `1176 / 1344`, raw top-1 action
  `1323 / 1344`, held-out two-pixel translation exact row `224 / 336 -> 336 / 336`,
  accepted benign `0 / 1344`, false accepts `0 / 248`.
  Outcome classification: `C`.
  Evidence: [visual-local-baseline-showdown-v1](/C:/Projects/zeromodel/docs/results/visual-local-baseline-showdown-v1),
  [visual-local-baseline-showdown-v1-postanalysis](/C:/Projects/zeromodel/docs/results/visual-local-baseline-showdown-v1-postanalysis).
  Claim status: measured / unsupported as a governed reader, but positive as a
  bounded mechanism result.

## What is confirmed

- VPM images authored by ZeroModel and external observations found in the world
  are different kinds of visual object and should not be treated as the same
  reader problem. Canonical source:
  [visual-representation-identity.md](/C:/Projects/zeromodel/docs/adr/visual-representation-identity.md).
- Exact policy-row identity remains distinct from action accuracy. Phase 1 and
  later reports repeatedly show correct-action retrieval from the wrong row.
- Ranking and governed acceptance are separate capabilities. System B v2 and R1
  both preserve useful raw ranking while failing to produce useful accepted
  coverage.
- Registration confirmed a real nuisance mechanism. On the declared fixture,
  bounded deterministic alignment repaired the held-out two-pixel translation
  family at raw top-1 inside the configured search envelope.

## What is unsupported or retired

- General global DINOv2 CLS retrieval as the promoted visual architecture.
- The ridge linear probe as a useful lower-complexity governed reader.
- Any claim that normalized pixels are already a safe bounded Level 1 reader.
- Any claim that the current registered-pixel reader is production-ready.
- Any claim of general visual intelligence, real-camera validation, arbitrary
  translation invariance, semantic object understanding, real-time viability, or
  a complete Visual State Compiler.

## What exists as preparation only

- [arcade_visual_local_evidence_benchmark.py](/C:/Projects/zeromodel/examples/arcade_visual_local_evidence_benchmark.py)
  and its tests define a fresh v3 local-evidence benchmark fixture.
- [arcade_visual_registered_calibration_v2.py](/C:/Projects/zeromodel/examples/arcade_visual_registered_calibration_v2.py)
  and its tests define an independent registered-pixel calibration run on that
  fresh fixture.
- Tests covering that preparation pass on current `main`, including
  [test_arcade_visual_registered_calibration_v2.py](/C:/Projects/zeromodel/tests/test_arcade_visual_registered_calibration_v2.py)
  and [test_visual_local_evidence_benchmark.py](/C:/Projects/zeromodel/tests/test_visual_local_evidence_benchmark.py).
- No committed final evidence package for `visual-registered-calibration-v2`
  exists on current `main`.
- No committed Stage 2 local-correlation evidence or adjudication exists on
  current `main`.

These items are implemented and tested. They are not yet measured, evidenced,
or adjudicated.

## What remains unresolved

- Whether independent registered calibration on the fresh v3 fixture changes
  the original R1 operating-point conclusion.
- Whether the dominant rejection bottleneck remains ambiguity margin separation.
- Whether local correlation produces nontrivial transferred benign coverage
  while preserving zero observed distinguishable false accepts and zero accepted
  conflicting-action errors.
- How the next local systems behave outside the declared registration bound.
- Whether deterministic geometry extraction outperforms correlation-only local
  evidence.
- Whether fixed-camera bounded validation changes the architecture choice.
- Whether the full ZeroModel governance stack materially beats a lighter audit
  wrapper.

## Claim boundaries

Use bounded wording only.

- Say: on the declared synthetic fixture.
- Say: zero observed distinguishable false accepts at the selected operating
  point.
- Say: inside the configured translation envelope.
- Say: raw top-1 ranking.
- Say: no useful governed operating point found under the declared calibration
  search.
- Say: implemented and tested, but not yet measured.

Do not say:

- general visual intelligence
- general zero false-accept risk
- production readiness
- real-camera validation
- arbitrary translation invariance
- semantic object understanding
- real-time viability
- complete Visual State Compiler

Undefined accepted precision must not be presented as zero precision. Correct
action retrieval must not be presented as exact-state understanding.

## Immediate continuation point

Begin the next research session by choosing one bounded target:

1. close the independent registered-pixel calibration on the fresh v3 fixture
   and commit evidence if it is executed cleanly; or
2. leave Stage 1 frozen and begin the first measured
   `translation_equivariant_template_correlation` provider.

Do not mix those scopes in one pass.

## Canonical sources

- [visual-research-logbook.md](/C:/Projects/zeromodel/docs/research/visual-research-logbook.md)
- [visual-sign-reader.md](/C:/Projects/zeromodel/docs/research/visual-sign-reader.md)
- [visual-address-phase-zero.md](/C:/Projects/zeromodel/docs/research/visual-address-phase-zero.md)
- [visual-address-phase-one.md](/C:/Projects/zeromodel/docs/research/visual-address-phase-one.md)
- [visual-address-research-status.md](/C:/Projects/zeromodel/docs/research/visual-address-research-status.md)
- [visual-address-review-adjudication.md](/C:/Projects/zeromodel/docs/research/visual-address-review-adjudication.md)
- [visual-address-system-b-v2-adjudication.md](/C:/Projects/zeromodel/docs/research/visual-address-system-b-v2-adjudication.md)
- [visual-local-baseline-showdown.md](/C:/Projects/zeromodel/docs/research/visual-local-baseline-showdown.md)
- [visual-ai-research-status-after-registration.md](/C:/Projects/zeromodel/docs/research/visual-ai-research-status-after-registration.md)
- [visual-address-phase-one-v1](/C:/Projects/zeromodel/docs/results/visual-address-phase-one-v1)
- [visual-address-system-b-v2](/C:/Projects/zeromodel/docs/results/visual-address-system-b-v2)
- [visual-local-baseline-showdown-v1](/C:/Projects/zeromodel/docs/results/visual-local-baseline-showdown-v1)
- [visual-local-baseline-showdown-v1-postanalysis](/C:/Projects/zeromodel/docs/results/visual-local-baseline-showdown-v1-postanalysis)
- [claims-audit.md](/C:/Projects/zeromodel/docs/claims-audit.md)
