# ZeroModel Visual Research Logbook

## Purpose

This logbook records how the ZeroModel visual research programme changed over
time: what was built, what was measured, what the evidence changed, and what
those changes meant.

## How to use this logbook

Use the repository memory layers in this order:

```text
Chats and reviews
    -> raw reasoning

Git commits and PRs
    -> exact repository chronology

Specifications
    -> questions and protocols

Evidence bundles and reports
    -> measured outcomes

Research synthesis
    -> interpretation and programme consequences

Logbook
    -> chronological memory of how understanding changed

Publications
    -> edited external account
```

Commits record what changed. Reports record what happened. This logbook records
what those events meant.

## Current research position

### Stable ZeroModel core

The ZeroModel policy core is stable inside its declared bounded scope. The
visual programme has not invalidated the deterministic artifact, policy lookup,
or provenance results recorded elsewhere in the repository.

### Visual research question

The open question is whether external visual observations can be compiled into a
trustworthy policy address, with acceptance and rejection semantics strong
enough for governed use.

### Made images and found images

ZeroModel now treats two visual objects as different problems.

- Made image: ZeroModel authored the representation and knows its coordinate
  semantics.
- Found image: the world produced the observation and ZeroModel must infer
  bounded evidence from it.

Those must not be treated as the same reader problem.

### Systems measured

The repository contains committed measured visual evidence for:

- the exact canonical observation-addressed reader in
  [visual-sign-reader.md](/C:/Projects/zeromodel/docs/research/visual-sign-reader.md);
- the Phase 1 held-out benchmark and DINOv2/global baselines in
  [visual-address-research-status.md](/C:/Projects/zeromodel/docs/research/visual-address-research-status.md)
  and [visual-address-phase-one-v1](/C:/Projects/zeromodel/docs/results/visual-address-phase-one-v1);
- the repaired normalized-pixel System B result in
  [visual-address-system-b-v2-adjudication.md](/C:/Projects/zeromodel/docs/research/visual-address-system-b-v2-adjudication.md)
  and [visual-address-system-b-v2](/C:/Projects/zeromodel/docs/results/visual-address-system-b-v2);
- the bounded registration showdown and post-analysis in
  [visual-local-baseline-showdown.md](/C:/Projects/zeromodel/docs/research/visual-local-baseline-showdown.md),
  [visual-local-baseline-showdown-v1](/C:/Projects/zeromodel/docs/results/visual-local-baseline-showdown-v1),
  and [visual-local-baseline-showdown-v1-postanalysis](/C:/Projects/zeromodel/docs/results/visual-local-baseline-showdown-v1-postanalysis).

### Confirmed mechanisms

The repository supports these bounded mechanism claims:

- exact canonical visual codeword addressing works on the declared arcade
  fixture;
- raw visual ranking can preserve strong action information without solving
  exact policy-row identity;
- ranking and governed acceptance are separate capabilities;
- bounded integer registration repairs a real translation-locality mechanism
  inside the declared `[-3, 3]` envelope;
- registration improved ranking without creating a useful governed operating
  point.

### Unsupported or retired approaches

Current repository evidence does not support promotion of:

- global DINOv2 CLS medoid retrieval;
- global DINOv2 CLS all-prototype k-NN as a governed reader;
- the ridge linear probe;
- normalized pixels or registered normalized pixels as useful governed readers
  at the committed operating points.

### Preparation-only work

Current `main` also contains implemented and tested, but not yet measured or
adjudicated:

- the fresh v3 local-evidence fixture in
  [arcade_visual_local_evidence_benchmark.py](/C:/Projects/zeromodel/examples/arcade_visual_local_evidence_benchmark.py);
- the independent registered-pixel calibration path in
  [arcade_visual_registered_calibration_v2.py](/C:/Projects/zeromodel/examples/arcade_visual_registered_calibration_v2.py);
- supporting tests in
  [test_visual_local_evidence_benchmark.py](/C:/Projects/zeromodel/tests/test_visual_local_evidence_benchmark.py)
  and
  [test_arcade_visual_registered_calibration_v2.py](/C:/Projects/zeromodel/tests/test_arcade_visual_registered_calibration_v2.py).

No committed final evidence package or adjudication for those preparation paths
exists on current `main`.

### Current claim boundaries

The current visual reader is not production-ready. The strongest bounded claims
are on the declared synthetic fixture, with explicit separation between raw
top-1 ranking and accepted governed decisions.

### Immediate open questions

Open questions include:

- whether independent registered-pixel calibration on the fresh v3 fixture
  changes the original operating-point conclusion;
- whether margin separation, rather than distance alone, remains the binding
  rejection mechanism;
- whether local correlation yields a useful accepted region;
- how systems behave outside the registration bound;
- whether deterministic geometry extraction helps;
- whether fixed-camera validation changes the picture;
- whether governance parity justifies the full visual governance stack.

### Next continuation point

The next declared measured provider remains
`translation_equivariant_template_correlation`. As of Saturday, July 18, 2026,
it is not yet implemented as a completed measured Stage 2 evidence package on
`main`.

---

# Chronological research log

## 2026-07-11 — Deterministic policy infrastructure is rebuilt before vision

### Starting position

Before the visual programme opened, the repository was being reconstructed
around a smaller deterministic artifact contract and clearer public claims.

### Question

What policy infrastructure had to become stable before visual addressing could
be interpreted as a bounded extension rather than a rewrite of the core?

### Work performed

The repository rebuilt the ZeroModel package around an immutable VPM artifact
kernel, reintroduced consumer modules on top of that kernel, added a claims
audit, hardened artifact and spatial contracts, and then established
reproducible sign-demo, policy-lookup, verification, and exhaustive arcade
validation scaffolding.

### Commits

- `a9793e0` `Merge pull request #2 from ernanhughes/codex/zeromodel-first-principles`
- `98775ab` `Merge pull request #3 from ernanhughes/codex/vpm-artifact-kernel`
- `d909e9d` `Merge pull request #4 from ernanhughes/codex/clean-v2-package`
- `69ca00e` `Merge pull request #6 from ernanhughes/codex/claims-audit`
- `22f2699` `Merge pull request #17 from ernanhughes/agent/pre-1-claims-hardening`
- `554737b` `Merge pull request #19 from ernanhughes/agent/reproducible-signs-demo`
- `cb95a1c` `Merge pull request #20 from ernanhughes/agent/v1.0.11-criticality-verification`
- `e113bd4` `Merge pull request #21 from ernanhughes/test/exhaustive-arcade-validation`

### Specifications and reports

- [vpm-artifact-v0.md](/C:/Projects/zeromodel/docs/spec/vpm-artifact-v0.md)
- [claims-audit.md](/C:/Projects/zeromodel/docs/claims-audit.md)
- [sign-reader.md](/C:/Projects/zeromodel/docs/examples/sign-reader.md)
- [criticality-verification.md](/C:/Projects/zeromodel/docs/examples/criticality-verification.md)
- [arcade-validation.md](/C:/Projects/zeromodel/docs/results/arcade-validation.md)
- [viper-policy-compilation.md](/C:/Projects/zeromodel/docs/research/viper-policy-compilation.md)

### Measured evidence

The committed exhaustive arcade validation package established that the
deterministic policy path could be replayed and checked exhaustively before any
visual approximation layer was introduced. The current repository still carries
that validation record in
[arcade-validation.md](/C:/Projects/zeromodel/docs/results/arcade-validation.md)
and
[arcade-validation-pytest.txt](/C:/Projects/zeromodel/docs/results/arcade-validation-pytest.txt).

### Interpretation

This was mostly enabling infrastructure rather than a visual result, but it
changed what later visual evidence could mean. The visual programme would be
judged against a stable artifact identity contract, exact policy lookup, and a
known-good exhaustive fixture, rather than against an evolving symbolic core.

### Decision

Keep the artifact kernel conservative and treat visual work as a consumer layer
that must compile uncertain observations into the existing deterministic policy
contract.

### Claim effect

The repository could support strong bounded claims about deterministic policy
identity, lookup, and replay before it claimed anything about visual
observations.

### Remaining question

Could an external visual observation recover that same deterministic policy
address without collapsing the distinction between ranking, acceptance, and
exact policy-row identity?

### Next step

Open the visual research programme on top of the stabilized deterministic core.

## 2026-07-16 — The visual problem is opened

### Starting position

ZeroModel already had a stable deterministic policy artifact and exact
state-addressed lookup path.

### Question

Can a bounded visual observation address the same compiled policy without an
explicit symbolic runtime state ID?

### Work performed

The repository added the observation-addressed policy path, first through the
exact deterministic visual sign reader and then through a provider-neutral
visual-address seam.

### Commits

- `7e3e352` `feat: add calibrated visual sign reader`
- `92ed26b` `Merge observation-addressed visual policy lookup`
- `9d34197` `docs: define phase zero visual address benchmark`

### Specifications and reports

- [visual-sign-reader.md](/C:/Projects/zeromodel/docs/research/visual-sign-reader.md)
- [visual-address-phase-zero.md](/C:/Projects/zeromodel/docs/research/visual-address-phase-zero.md)

### Measured evidence

The committed exact canonical reader recovered all 112 canonical frames and
matched the symbolic policy across 31,213 runtime observations in the exhaustive
arcade replay recorded in
[visual-sign-reader.md](/C:/Projects/zeromodel/docs/research/visual-sign-reader.md).

### Interpretation

This established a closed-world exact canonical baseline, not a tolerant or
general visual reader.

### Decision

Continue into held-out visual variation with benchmarked approximate readers.

### Claim effect

The repository could support exact canonical observation-addressed lookup on the
declared fixture, but not open-world visual understanding.

### Remaining question

Would non-exact observations still support trustworthy policy-row recovery and
safe rejection?

### Next step

Build the held-out visual benchmark and compare simple and learned baselines.

## 2026-07-17 — Made images and found images

### Starting position

The deterministic visual index still sat too close to VPM artifact language,
which risked treating authored policy imagery and external observations as one
reader problem.

### Question

How should ZeroModel represent dense visual observations without conflating them
with authored policy maps?

### Work performed

The repository accepted an architectural separation between visual-address
providers and policy lookup, and between dense representation tensors and VPM
policy artifacts.

### Commits

- `76e827a` `docs: record visual representation identity decision`
- `58b7a34` `Merge pull request #29 from ernanhughes/feat/governed-visual-address-contract`

### Specifications and reports

- [visual-representation-identity.md](/C:/Projects/zeromodel/docs/adr/visual-representation-identity.md)
- [visual-address-phase-zero.md](/C:/Projects/zeromodel/docs/research/visual-address-phase-zero.md)

### Interpretation

This was a design correction, not a measured result.

The repository now supports the following interpretation:

```text
Made image:
ZeroModel authored the representation and knows its coordinate semantics.

Found image:
The world produced the observation and ZeroModel must infer bounded evidence.
```

The architectural consequence is that the observation reader should compile
uncertain evidence into candidate state claims before deterministic policy
lookup. It should not be treated as another VPM coordinate reader.

### Decision

Keep the visual address and policy separate, store dense representations as
`MatrixBlob`, and treat observation providers as independent governed seams.

### Claim effect

This narrowed the visual thesis and made it harder to over-claim that one image
reader problem covered both authored and found visual objects.

### Remaining question

Which representation families preserve the exact distinctions needed for
governed policy-row identity?

### Next step

Measure approximate readers on the held-out Phase 1 benchmark.

## 2026-07-17 — DINOv2 and invariance

### Starting position

Phase 1 implemented the first family-held-out benchmark for approximate visual
addressing, including normalized pixels, pinned DINOv2 CLS retrieval, and a
ridge probe.

### Question

Can a pinned frozen global representation recover governed policy rows and
reject distinguishable invalid observations on the declared arcade fixture?

### Work performed

The benchmark machinery, baselines, calibration, and result closure were added,
then the full Phase 1 result was interpreted and adjudicated.

### Commits

- `08eb31f` `Merge pull request #30 from ernanhughes/feat/phase-one-visual-benchmark`
- `bed5fe8` `Merge pull request #31 from ernanhughes/agent/visual-evidence-closure`
- `d3ba2ef` `docs: adjudicate visual address external review`

### Specifications and reports

- [visual-address-phase-one.md](/C:/Projects/zeromodel/docs/research/visual-address-phase-one.md)
- [visual-address-research-status.md](/C:/Projects/zeromodel/docs/research/visual-address-research-status.md)
- [visual-address-review-adjudication.md](/C:/Projects/zeromodel/docs/research/visual-address-review-adjudication.md)
- [visual-address-phase-one-v1](/C:/Projects/zeromodel/docs/results/visual-address-phase-one-v1)

### Measured evidence

The Phase 1 research-status record reports:

- System B benign action accuracy `73.44%`, exact benign row accuracy `62.50%`,
  FAR `51.21%`
- System C benign action accuracy `70.01%`, exact benign row accuracy `30.73%`,
  FAR `82.26%`
- System D benign action accuracy `74.78%`, exact benign row accuracy `30.36%`,
  FAR `73.79%`
- System G benign action accuracy `59.45%`, exact benign row accuracy `28.79%`,
  FAR `100.00%`

System D achieved `1,005` correct benign actions but only `408` exact benign
rows, meaning `597` correct actions came from the wrong policy row.

### Interpretation

The central interpretation recorded in the repository is that invariance can
preserve broad scene meaning while losing exact policy-row identity. The more
precise later wording in
[visual-address-review-adjudication.md](/C:/Projects/zeromodel/docs/research/visual-address-review-adjudication.md)
is:

> In systems where small visible distinctions determine exact governed state
> identity, representation invariance can become a governance failure rather
> than a robustness benefit.

### Decision

Do not promote the global DINOv2 CLS retrieval path or the ridge probe as the
visual architecture.

### Claim effect

The repository now supports a bounded negative result for the tested global
learned path. It does not support the stronger claim that every learned or local
representation will fail.

### Remaining question

Was the decisive problem ranking, calibration, locality, or some combination of
them?

### Next step

Repair evidence semantics, separate ranking from acceptance, and adjudicate
System B directly.

## 2026-07-17 — System B and the protocol repair

### Starting position

The original Phase 1 result showed that normalized pixels were materially
stronger than the DINOv2 baselines on exact-row recovery, but the result mixed
ranking and accepted operating-point claims too loosely.

### Question

Does normalized-pixel retrieval contain a useful governed operating point after
repairing calibration semantics and evidence closure?

### Work performed

The repository repaired calibration semantics, separated ranking from governed
acceptance, regenerated the frozen System B v2 evidence, and committed a formal
adjudication.

### Commits

- `197d1e2` `feat: separate visual ranking from calibration`
- `b288fb6` `fix: align visual calibration and runtime semantics`
- `35d5153` `fix(visual): bind adjudication semantics and calibration boundaries`
- `e7d2670` `Merge pull request #32 from ernanhughes/research/visual-b-operating-curves`

### Specifications and reports

- [visual-address-system-b-v2-adjudication.md](/C:/Projects/zeromodel/docs/research/visual-address-system-b-v2-adjudication.md)
- [visual-address-review-adjudication.md](/C:/Projects/zeromodel/docs/research/visual-address-review-adjudication.md)
- [visual-address-system-b-v2](/C:/Projects/zeromodel/docs/results/visual-address-system-b-v2)

### Measured evidence

The committed System B v2 adjudication records:

- selected quantile `1.0`
- calibration benign coverage `1 / 1344`
- final benign coverage `0 / 1344`
- distinguishable false accepts `0 / 248`
- false rejects `1344 / 1344`
- raw top-1 exact-row accuracy `1008 / 1344 = 75.0%`
- raw top-1 action accuracy `1302 / 1344 = 96.875%`

### Interpretation

System B preserved strong pre-rejection ranking signal, but ranking was not the
same thing as acceptance. The result showed that raw top-1 quality and governed
operation had to be reported separately.

### Decision

Treat normalized pixels as a useful comparator and mechanistic clue, not as a
promoted reader.

### Claim effect

The repository supports the bounded statement that normalized pixels contain
useful ranking information on the fixture, but no useful zero-observed-FAR
operating point was found under the repaired protocol.

### Remaining question

Was the remaining failure primarily locality and alignment rather than absence
of visible task information?

### Next step

Run the registered local baseline showdown.

## 2026-07-18 — Registration confirms a mechanism

### Starting position

System B v2 preserved strong raw ranking, but the translation family remained a
clear mechanistic weakness.

### Question

Can bounded deterministic integer registration repair enough translation
locality failure to create a useful governed operating point?

### Work performed

The repository added a bounded registration module, a registered local baseline
provider, frozen showdown evidence, and a post-analysis package.

### Commits

- `20f2d90` `Add registered local visual baseline showdown`
- `58ddd49` `Record registered local baseline showdown evidence`
- `7c88afd` `Merge pull request #34 from ernanhughes/research/visual-local-baseline-showdown`
- `592cee8` `docs(research): summarize visual AI status after registration`

### Specifications and reports

- [visual-local-baseline-showdown.md](/C:/Projects/zeromodel/docs/research/visual-local-baseline-showdown.md)
- [visual-ai-research-status-after-registration.md](/C:/Projects/zeromodel/docs/research/visual-ai-research-status-after-registration.md)
- [visual-local-baseline-showdown-v1](/C:/Projects/zeromodel/docs/results/visual-local-baseline-showdown-v1)
- [visual-local-baseline-showdown-v1-postanalysis](/C:/Projects/zeromodel/docs/results/visual-local-baseline-showdown-v1-postanalysis)

### Measured evidence

The committed Stage 1 registered result records:

- selected quantile `0.0`
- final benign accepted `0 / 1344`
- false accepts `0 / 248`
- false rejects `1344 / 1344`
- raw top-1 exact-row `1176 / 1344 = 87.5%`
- raw top-1 action `1323 / 1344 = 98.4375%`

Compared with frozen System B v2:

```text
exact-row top-1: 1008 / 1344 -> 1176 / 1344
action top-1:    1302 / 1344 -> 1323 / 1344
```

On the held-out two-pixel translation family:

```text
exact-row: 224 / 336 -> 336 / 336
action:    322 / 336 -> 336 / 336
```

The post-analysis records:

- feasible decoupled candidates `11`
- every feasible point at `margin quantile = 0.0`
- best feasible calibration coverage `6 / 1344`
- best feasible final coverage `0 / 1344`
- overall rejection decomposition: `952` margin-only, `392` both, `0`
  distance-only

### Interpretation

Registration generalized to unseen translation instances inside the declared
search envelope and confirmed spatial alignment as a real nuisance mechanism.
It also separated two problems:

- recognition and ranking;
- governed acceptance.

A reader can rank the correct answer highly without possessing enough evidence
to govern the decision.

### Decision

Do not promote the registered reader. Preserve the mechanism result and move to
the next local-evidence provider.

### Claim effect

The repository supports a bounded mechanism claim for registration. It does not
support general translation invariance, production readiness, or a useful
governed operating point.

### Remaining question

What evidence representation can create nontrivial accepted benign coverage
without distinguishable false accepts or accepted conflicting-action errors?

### Next step

Proceed to `translation_equivariant_template_correlation`, with explicit
beyond-bounds translation controls.

## 2026-07-18 — Scope correction between Stage 1 and Stage 2

### Starting position

After the registration result, the work on `main` began to mix three separate
activities:

- independent registered-pixel calibration on a fresh v3 fixture;
- fresh local-evidence benchmark preparation;
- Stage 2 local-correlation preparation.

### Question

What does current `main` actually contain, and which parts are implemented,
tested, executed, evidenced, or adjudicated?

### Work performed

Current `main` at `336f7c4752e505d3be921be372c732bd19dc57b8` includes:

- implemented and tested fresh-fixture preparation in
  [arcade_visual_local_evidence_benchmark.py](/C:/Projects/zeromodel/examples/arcade_visual_local_evidence_benchmark.py)
  and
  [test_visual_local_evidence_benchmark.py](/C:/Projects/zeromodel/tests/test_visual_local_evidence_benchmark.py);
- implemented and tested independent registered-calibration machinery in
  [arcade_visual_registered_calibration_v2.py](/C:/Projects/zeromodel/examples/arcade_visual_registered_calibration_v2.py)
  and
  [test_arcade_visual_registered_calibration_v2.py](/C:/Projects/zeromodel/tests/test_arcade_visual_registered_calibration_v2.py);
- no committed final evidence directory for
  `docs/results/visual-registered-calibration-v2/`;
- no committed Stage 2 local-correlation evidence package;
- no adjudication document closing Stage 2.

### Commits

- `336f7c4` `stage 3`

### Interpretation

This is where Stage 1 closure and Stage 2 construction were conceptually mixed.
The repository now needs the distinction stated plainly:

- the fresh v3 benchmark and independent registered calibration are implemented
  and tested;
- they are not committed measured evidence on `main`;
- Stage 2 local correlation has not yet been measured or adjudicated.

### Decision

Separate “closing the current Stage 1 analysis” from “building the next Stage 2
provider.”

### Claim effect

Current `main` supports:

- `implemented only` for the new scripts themselves;
- `tested` for the associated unit and integration tests;
- not yet `executed` as frozen final evidence on `main`;
- not yet `supported by a committed evidence package`;
- not yet `adjudicated`.

### Remaining question

Should the next session first close the fresh independent registered calibration
or begin the local-correlation provider itself?

### Next step

Start the next research session from the status cut and choose one bounded
target before further implementation.

## 2026-07-18 — First visual research chapter cut

### Starting position

The latest merged visual work is already on `main`, but the repository lacked a
durable logbook and concise programme-cut summary.

### Work performed

This closeout recorded the visual programme state on current `main` without
modifying source code, tests, or frozen evidence.

### Commits

- `336f7c4752e505d3be921be372c732bd19dc57b8` documented as current `main` at
  the start of the cut

### Specifications and reports

- [visual-programme-status-cut-2026-07-18.md](/C:/Projects/zeromodel/docs/research/visual-programme-status-cut-2026-07-18.md)
- [claims-audit.md](/C:/Projects/zeromodel/docs/claims-audit.md)

### Measured evidence

As of Saturday, July 18, 2026 03:42:50 +01:00:

- `python -m pytest -q` passed with `209 passed, 1 skipped`
- measured visual systems with committed evidence remain the canonical reader,
  Phase 1 benchmark systems, System B v2, and registered Stage 1 showdown

### Interpretation

The project is near the end of its first complete visual-research chapter, not
near the end of the full visual programme.

### Decision

Close the current documentation session without reopening scope.

### Claim effect

This cut does not declare the visual programme complete. The programme is
finished only when every declared research question receives a measured and
bounded answer, including valid negative answers.

### Remaining question

Which single bounded experiment should begin the next chapter: independent
registered calibration closure or the first measured Stage 2 local-correlation
provider?

### Next step

Resume from the immediate continuation point in
[visual-programme-status-cut-2026-07-18.md](/C:/Projects/zeromodel/docs/research/visual-programme-status-cut-2026-07-18.md).
