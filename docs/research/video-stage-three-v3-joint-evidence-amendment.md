# Stage 3 v3 Joint-Evidence Amendment

Date: July 18, 2026

## Question

Can deterministic current-frame joint discriminative evidence recover safe exact-row or bounded candidate-set utility over the frozen finite arcade policy universe, and can temporal intersection later narrow only those currently supported candidates?

## Frozen inputs preserved

- Complete 112-row prototype universe.
- Complete development closure.
- Unchanged 12-row evaluation sampling algorithm.
- Existing region geometry and registration bounds.
- Source-scope discipline and split roles.
- Safety hierarchy and maximum candidate-set size of 3.
- Zero distinguishable false-accept requirement.
- V5 non-injection rule and no-stale-state rule.
- Phase-access auditing, rebuild verification, and human claim adjudication.

## V3 permitted corrections

### 1. Exact tie safety

Exact acceptance requires strict semantic superiority. Lexical ordering may order traces only.

### 2. Independent Architecture A

`A3` must use direct registered regional candidate-to-observation correlation distance. It must not depend on B/C support or contradiction classification.

### 3. Joint candidate fit for `B3`

`B3` must compare the observation with a candidate over the complete available stable discriminative pattern. A suitable normalized fit is:

```text
1 - weighted mean normalized absolute candidate error
```

over candidate-relative stable row-informative evidence.

### 4. Pairwise joint evidence for `C3`

`C3` must compute candidate-versus-competitor evidence over a pairwise discriminative mask. For each candidate and competitor:

```text
pairwise margin = candidate joint fit - competitor joint fit
```

Support and contradiction must derive from pairwise or region-level margins, not pointwise nearest-prototype comparisons.

### 5. Action-conflict evidence

Different-action evidence must use pairwise masks against different-action competitors. A same-action match at one pixel may not zero all action-conflict weight at that pixel.

### 6. Evidence accounting

Persist separately:

- declared informative mass
- available geometric mass
- stable candidate-fit mass
- pairwise discriminative mass
- actual scored mass

### 7. Exhaustive self-retrieval gate

Before architecture selection, run all 112 canonical exact observations.

For visually unique prototypes:

```text
expected row must be unique top-1
```

For aliases:

```text
all aliases remain in the maximum-evidence equivalence set
no lexical exact selection
```

### 8. New identities

Use:

- `zeromodel-video-discriminative-evidence-stage3/v3`
- `zeromodel-video-discriminative-generator/v3`
- `zeromodel-video-discriminative-provider/v3`
- `docs/results/video-discriminative-local-evidence-v3/`

Derive the v3 seed from the v3 amendment commit SHA.

## Architecture identities

- `A3`: direct corrected regional correlation
- `B3`: joint discriminative candidate fit
- `C3`: pairwise joint support and contradiction
- `D3`: combined B3/C3 mechanisms, gateway-controlled

Bare `A/B/C/D` must not be reused in v3 artifacts without an architecture-semantics version.

## Grid policy

Do not derive thresholds from v2 failure outcomes. Freeze normalized score semantics and finite grids before implementation. Exact-match sanity is a prerequisite, not a tuned threshold result.

## Adversarial questions

1. Joint support is treated as a bounded representation correction to match the declared scientific target, not as evidence that a broader hypothesis is already validated.
2. Pairwise evidence must remain deterministic over 112 rows with fixed ordering, fixed masks, and finite per-pair computation.
3. Pairwise masks must be audited so they do not leak row identity trivially through degenerate one-hot coverage.
4. Candidate-relative masking is permitted only when the same construction is available symmetrically for each candidate-competitor comparison.
5. Symmetry is maintained by evaluating both candidate and competitor on the same pairwise mask for a given pair.
6. Same-action competitors remain part of the pairwise comparison universe for row disambiguation.
7. Conflicting-action competitors additionally drive action-conflict evidence through different-action pairwise masks.
8. Zero-mass pairwise masks must yield neutral evidence, never synthetic support.
9. Region and cross-region conjunctions must be represented explicitly through joint region vectors and cross-region tuples.
10. Lexical ordering may not affect acceptance under any v3 decision path.
11. Exhaustive property tests must prove canonical self-retrieval, alias handling, tie safety, and ordering invariance.
12. Even after successful self-retrieval, v3 may not support claims about final performance, impossibility of safe readers, or V5 behavior.

## Prohibited claims remain prohibited

- No safe discriminative visual reader exists.
- Candidate sets are scientifically useless.
- Joint evidence cannot work.
- Temporal narrowing cannot work.
- Local correlation cannot work.
- The policy rows are visually indistinguishable.
- Final benchmark performance conclusions.
