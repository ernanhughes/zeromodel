# Bertin-inspired pattern detection

Status: **Research / synthetic-fixture validation**

ZeroModel can already build deterministic views from declared layout recipes. This experiment asks a different question:

> Can ZeroModel discover a useful row ordering from an apparently disordered scored matrix, calibrate the apparent structure against a null model, and freeze the result as linked identity-bearing artifacts?

The MVP is deliberately narrow. It implements:

- one NumPy-only spectral-seriation method;
- two numeric objectives: adjacent coherence and anti-Robinson structure;
- a selection-corrected permutation null;
- an explicit discovered row order;
- a pattern-report VPM linked to the analyzed artifact;
- a discovered-view VPM linked to both the source and the report.

It does **not** yet implement biclustering, anomaly interpretation, domain-semantic labels, multiple ordering methods, column seriation, human-inspection evidence, or real-world validation.

## Why the null is selection-corrected

The detector does not compare an optimized real matrix against unoptimized shuffled matrices. Every null sample:

1. independently permutes values within each metric column, preserving column marginals while destroying joint row structure;
2. reruns the complete ordering pipeline;
3. evaluates every declared objective;
4. contributes its maximum standardized objective score to the family null distribution.

This corrects for selecting the most favorable objective after looking at the data. Adding future ordering methods must also add those methods to every null run.

## Determinism and identity

Spectral discovery is a compile-time analysis. The result is frozen as an explicit row-ID permutation:

```text
source VPM
    ↓ analyzed by
pattern-report VPM
    ↓ orders
explicit discovered-view VPM
```

The discovered view does not depend on rerunning the eigensolver. Its `LayoutRecipe` stores the complete row order, and normal VPM source mappings remain available.

The report stores:

- analyzed artifact ID;
- detector/checker version;
- method and objective IDs;
- value source;
- null sample count and seed;
- specification digest;
- discovered row order;
- observed scores;
- null mean and standard deviation;
- per-objective permutation p-values;
- family-level p-value;
- degeneracy flag.

## Public API

```python
from zeromodel import MatrixPatternDetector, PatternAnalysisSpec

spec = PatternAnalysisSpec(
    value_source="normalized",
    null_samples=199,
    seed=11,
)
detector = MatrixPatternDetector(spec)
report = detector.detect(source_artifact)
view = detector.build_view(source_artifact, report)

print(report.family_p_value)
print(report.primary_objective)
print(report.row_order)
```

The lower-level functions `detect_patterns()` and `build_discovered_view()` are also available.

## Current evidence

The committed tests establish the following narrow claims:

- a planted three-block matrix is recovered with strong same-block adjacency;
- its family-level statistic is significant under the declared permutation null;
- selected pure-noise fixtures are not reported as significant;
- constant matrices are marked degenerate and fall back to source order;
- identical analysis inputs produce identical report digests and artifact identities;
- changing only the calibration seed changes the report lineage but not the frozen discovered ordering;
- report and view artifacts preserve lineage and `.vpm` round trips;
- incomplete explicit row permutations are rejected.

These tests do not establish broad matrix-pattern discovery or practical inspection benefit.

## Kill conditions

The direction should be reconsidered if broader synthetic evaluation shows any of the following:

1. false positives materially exceed the declared family alpha on pure noise;
2. planted structure recovery does not beat source order or simple conventional baselines at moderate signal-to-noise ratios;
3. a commodity seriation or clustered-heatmap implementation provides the same analytical and lineage value with substantially less machinery;
4. discovered layouts do not improve an automated or human inspection task.

## Next experiment

The next step is not to add more algorithms. It is to run a generated benchmark across signal strengths and random seeds, reporting:

- family false-positive rate;
- block recovery;
- stability under small perturbations;
- runtime by row count;
- comparison with source order and one conventional clustering baseline.

Only after that gate should the detector add biclustering, anomalies, multiple methods, or a real deployment-policy fixture.
