# ZM-HAND-02 — Anatomical Topology and Residual Evidence

An **unpromoted example** that builds on (and does not modify) ZM-HAND-01 in
`examples/hand_anatomy/`. It tests one sharp question:

> Can a VPM compiled from explicit anatomical topology and unexplained visual
> structure distinguish an extra finger from a legitimate five-finger pose when
> global image similarity cannot?

**Outcome: no.** The preregistered rule rejected the approach. See
[RESULTS.md](RESULTS.md) and [PREREGISTRATION.md](PREREGISTRATION.md).

## What was added

| Piece | File | Purpose |
|---|---|---|
| Counterfactual pairs | `pairs.py` | same hand, valid vs exactly one defect, identical rigid placement; `indistinguishable` computed from the two masks |
| Pose refinement | `refine.py` | deterministic coordinate descent, joint limits enforced by `geometry.CONSTRAINTS`; used as a second baseline (IoU after refinement) so gains are attributed correctly |
| Residual analysis | `residual.py` | residual components after the fit, with finger-likeness (length, elongation, width, attachment) and digit coverage |
| Topology | `topology.py` | Zhang-Suen skeleton and digit-branch count (heuristic; merges when fingers overlap) |
| Residual VPM + contract | `vpm02.py` | genuine `ScoreTable`/`LayoutRecipe`/`VPMArtifact`; raw cells; decision contract `hand-decision/2` |
| Pipeline/report/replay | `inspect02.py` | report schema `hand-anatomy-report/2`, validate, replay, overlays |
| Benchmark | `benchmark02.py` | matched-FPR comparison, exact sign test, frozen rule |

Reused unchanged from ZM-HAND-01: template library, alignment, comparison metrics,
observation loading, visualisation. No package, `package-boundaries.toml` or `VERSION` was touched.

## Commands (PowerShell, repository root)

```powershell
# calibration split only (about 90 s); the test split is NOT scored
python -m examples.hand_anatomy_02.cli benchmark --output artifacts/hand_anatomy_02/benchmark

# score the test split ONCE under the frozen rule (about 5 min). Do not repeat after changing methods.
python -m examples.hand_anatomy_02.cli benchmark --output artifacts/hand_anatomy_02/benchmark --final

# re-evaluate saved per-case records without re-scoring
python -m examples.hand_anatomy_02.cli benchmark --output artifacts/hand_anatomy_02/benchmark --reuse-records

python -m examples.hand_anatomy_02.cli pair-sheet --output artifacts/hand_anatomy_02/pair_sheet.png
python -m examples.hand_anatomy_02.cli inspect --mask path/to/mask.png `
  --contract artifacts/hand_anatomy_02/benchmark/calibrated_contract02.json --output artifacts/hand_anatomy_02/inspection
python -m examples.hand_anatomy_02.cli validate-report --report artifacts/hand_anatomy_02/inspection/report.json
python -m examples.hand_anatomy_02.cli replay --report artifacts/hand_anatomy_02/inspection/report.json --mask path/to/mask.png
```

`calibrated_contract02.json` only sets a residual threshold at 10% false positives on
calibration data. It is **not validated for use** (the method was rejected).

Tests (marked `research`, outside the fast gate): `python -m pytest examples/hand_anatomy_02/tests -q`

## Overlay colours

Same as ZM-HAND-01 (green matched, red unexplained, blue missing, grey tolerance) plus a
yellow box around the residual component with the highest finger-likeness.

## Limitations

* Synthetic silhouettes only; the independent generator is still written by the same author.
* A silhouette-hidden extra digit is undetectable by construction; those pairs are reported, never credited.
* Finger-likeness constants and the branch heuristic are fixed by hand; the branch count merges overlapping fingers.
* No real or AI-generated image has been tested. That remains a mandatory separate validation stage.

## Terminology correction

The code, `results.json` and the frozen preregistration use the label `indistinguishable`
for pairs whose silhouettes differ by less than 2% of the valid area. That word is wrong: a
small extra digit can be obvious to a human and topologically important while changing the
area very little. Read it as **low-silhouette-difference**. Identifiers were left unchanged so
the scored artifacts and the preregistration hash stay exactly as run. Those pairs are a
separate evaluation group that is reported, not discarded, but the 48-pair primary set that
excludes them is a diagnostic subset, not the full distribution of malformed hands. The
no-credit rule applied to them only because the preregistration said so; it is not a claim
that those defects are undetectable in general.
