# Return: Bounded Recovery After Memory Loses Authority (frozen spec)

**Status: FROZEN and implementation-ready — no implementation yet.**
Implementation against this document is judged mechanically. Any
deviation requires a spec amendment, not a code comment.

Follows the completed future-memory experiment (`future-memory-action-
selection.md`, frozen) and the memory-authority experiment
(`future-memory-authority.md`, frozen). Those documents are not revised
by this one.

## 0. Architectural rule (frozen)

> **VPMs carry remembered structure. The Observer addresses those VPMs and
> produces transient judgments about what they may presently do.**

Therefore:

```text
FutureMemoryValidityDTO
    = remembered structure (memory-about-memory)

MemoryAuthorityAssessmentDTO
    = transient Observer judgment (audit digest only; never persisted
      as memory merely because it has an identity)

ReturnPolicy
    = remembered policy governing what to do after authority failure
      (compiled VPM artifact; tiny, frozen, ungated)

ReturnDecision
    = transient Observer decision produced by addressing that policy
```

The Return policy is deliberately terminal in the meta-policy hierarchy.
Its mistakes are measured through `return_harm`; no authority layer is
created for the authority layer.

## 1. Research question

> **Can a small, compiled Return policy convert justified abstentions
> into correct decisions without manufacturing false confidence?**

Precisely: when the ordinary action path cannot safely commit because its
remembered policy or expected consequence is OOD, contradicted without
support, unsupported, or rejected, can ZeroModel choose a bounded recovery
operation that improves eventual task success while preserving the
original refusal semantics?

## 2. First failure case: warehouse goal-change

Observed behavior to beat ( अभियंत्रण control, re-baselined under §6):

```text
environment changes materially
        ↓
historical future memory loses local support
        ↓
projection becomes OOD
        ↓
world-action gate abstains
        ↓
coverage ≈ 0.95
harm ≈ 0.05
```

This is not the ridge false-veto failure (bad memory retaining excessive
authority). It is the distinct failure: memory correctly loses authority,
the gate correctly abstains, and **no recovery mechanism exists**.

## 3. Return selects recovery operations, never task actions

Bounded first operations (frozen):

```text
REOBSERVE
FALLBACK
STOP
```

Terminal disposition is a deterministic interpretation of STOP, not a
separate policy (see §3.3): `STOP` with an empty trajectory path reports
ABSTAIN; `STOP` with a used path reports ESCALATE. This keeps the
compiled surface at 16 rows instead of 32 while preserving the crisp
semantics — did recovery fail before or after a recovery attempt? No
driver policy hides outside the artifact.

EXPLORE, INVOKE_MODEL, ASK_HUMAN, SWITCH_SENSOR and any other response
are outside this experiment unless evidence demands them.

### 3.1 REOBSERVE means switch observation profile, not "look again"

The renderers are deterministic: re-rendering yields identical bytes, so
"look again" is vacuous, and re-rendering from symbolic state would
smuggle privileged ground truth into the decision path. Frozen rule:

- Two declared observation profiles exist from the outset:
  `native-L` (current frames) and `coarse-L` (the *captured* before-frame
  downsampled 2×2 box-mean; a pure observation-path transform of captured
  data, privilege-free).
- Each profile has its own field schema, P3 predictor memory, transition
  memory (empirical; ridge optional, not required), validity state, and
  per-profile declared expectations — all fitted/declared train-side and
  frozen. Arcade coarse bands are the adapter's static bands remapped to
  coarse fields; warehouse coarse uses whole-canvas annotations.
- REOBSERVE re-encodes the captured before-frame under the coarse
  profile and reruns the complete Observer loop exactly once (P3 ranking,
  authority consult, candidate gating, normal rejection). At most one
  re-observation per trajectory.

Pre-registered: B helps noise splits through averaging only; it should
not help genuine shift/change (Kill 2 watches this arm).

### 3.2 FALLBACK addresses an identified alternative ranking

Fallback is the **native-profile P3 baseline ranking** (uniform
nearest-neighbour over the same P3 memory), accepted or rejected under
P3's own rejection semantics, with **no future-memory veto**. Its model
identity is fixed before evaluation; Return cannot invent it. It is
future-memory-free, not memory-free: P3 is itself compiled historical
memory. It may recommit an action the future-memory path vetoed; that
is precisely what is under test (see §10), and `return_harm`
adjudicates whether abandoning the veto was justified.

Claim boundary (frozen): a future-transition contradiction here is a
predictive/behavioral expectation and may be bypassed by this
experiment. A hard `NEVER perform action X under condition Y`
constraint would belong outside this fallback mechanism and must remain
enforced. That keeps the eventual safety claim boundary clean.

Rationale: when the gated path abstains, the fallback answers "what
would the memory-free policy have done." The ablation (§13) then
determines whether Return adds value over the fallback alone.

Fallback coherence rule: the fallback ranking must differ in kind from
the failed path (here: memory-free ranking vs gated ranking). A fallback
that reproduces the failed ranking is not a recovery operation.

### 3.3 STOP resolves to ABSTAIN or ESCALATE by path, not by policy

- **ABSTAIN** = STOP with an empty trajectory path: no recovery path
  was available. The refusal stands.
- **ESCALATE** = STOP with a non-empty trajectory path: at least one
  permitted recovery path was attempted and the trajectory still cannot
  commit (`decision = none`, `reason = escalation_required`). No human
  or model is invoked; this preserves the future integration point
  without smuggling another intelligence system into the experiment.

Return is therefore not required to restore coverage to 100%.

## 4. Trigger computation (frozen, reachable)

Return requires terminal failure first: `CoupledActionPrediction.accepted
== False`. Only then is the trigger derived from the rank-0
(baseline-top) candidate's fate. A low-authority rank-0 candidate that
the ordinary policy routes around (rank 0 OOD, rank 1 supported →
rank 1 selected) must NOT invoke Return.

Trigger values (frozen):

```text
baseline.status != accepted              → PREDICTOR_REJECTED
rank-0 OOD-vetoed                        → OOD
rank-0 contradiction-vetoed (MAY_VETO)   → CONTRADICTED
rank-0 insufficient/unsupported-vetoed   → INSUFFICIENT
```

`INSUFFICIENT` means terminal `insufficient_future_evidence` under the
experiment's declared `reject_on_insufficient=True`; it does not mean
low confidence in general, and ordinary low-confidence SUPPORT_ONLY
memory is never a Return condition. STALE and SUPPORT_ONLY answer how
much authority a memory has, not whether selection failed: under the
frozen veto rules their candidates are annotated survivors, so the gate
commits and Return correctly never fires for them. They persist as
authority telemetry and veto modulation.

Deviations from earlier sketches, deliberate: STALE and SUPPORT_ONLY
cannot cause abstention under the frozen veto rules (their candidates
are annotated survivors, so the gate commits) — they persist as
authority telemetry and veto modulation, not as triggers. INSUFFICIENT
is reachable only because this experiment sets
`reject_on_insufficient = True` (see §6); without it the axis would
contain a dead value.

## 5. The compiled Return policy (16 rows, frozen scores)

Situation vocabulary (all axes per-query computable from Observer
evidence):

```text
trigger ∈ {PREDICTOR_REJECTED, OOD, CONTRADICTED, INSUFFICIENT}
fallback_decisive ∈ {yes, no}    # fallback artifact exists AND its P3
                                 # ranking accepts on this observation
reobserve ∈ {yes, no}            # coarse profile compiled AND unspent
```

`fallback_decisive` (not mere existence: existence is static
post-compile, hence a dead axis) is evaluated on the current
observation. Scoring is winner-takes-all via `VPMPolicyLookup` over a
hand-scored `ScoreTable` (precedent: arcade `compile_policy_artifact`);
row IDs are deterministic
(`trigger=…|reobserve=…|fallback=…`); the artifact carries provenance
`kind = compiled_return_policy`:

| trigger | reobserve | fallback | REOBSERVE | FALLBACK | STOP |
|---|---|---|---|---|---|
| OOD | yes | yes/no | **1.0** | 0.0 | 0.0 |
| OOD | no | yes | 0.0 | **1.0** | 0.0 |
| OOD | no | no | 0.0 | 0.0 | **1.0** |
| CONTRADICTED | yes | yes/no | 0.0 | **1.0** | 0.0 |
| CONTRADICTED | no | yes | 0.0 | **1.0** | 0.0 |
| CONTRADICTED | no | no | 0.0 | 0.0 | **1.0** |
| INSUFFICIENT | yes | yes/no | **1.0** | 0.0 | 0.0 |
| INSUFFICIENT | no | yes | 0.0 | **1.0** | 0.0 |
| INSUFFICIENT | no | no | 0.0 | 0.0 | **1.0** |
| PREDICTOR_REJECTED | yes | yes/no | **1.0** | 0.0 | 0.0 |
| PREDICTOR_REJECTED | no | yes | 0.0 | **1.0** | 0.0 |
| PREDICTOR_REJECTED | no | no | 0.0 | 0.0 | **1.0** |

(Each `yes/no` pair expands to two rows; 16 rows total. Rows are
tie-free by construction; `VPMPolicyLookup` tie-break is never
exercised.)

Reading: re-observation is attempted only for support failures
(OOD/INSUFFICIENT) and recognition failures, never to re-litigate a
confident contradiction; fallback serves contradiction and
unsupported cases. D (combined) is this table with full budget — not
a hardcoded sequence: REOBSERVE-first order emerges from the axes,
and trajectory depth ≤ 2 emerges from the budget (max 1
re-observation, max 1 fallback switch).

Terminal disposition (frozen, no hidden policy): STOP with an empty
trajectory path reports ABSTAIN ("no path was available"); STOP with
a non-empty path, or a fallback that was attempted but rejected,
reports ESCALATE ("attempted and failed"). The distinction is a
deterministic reading of path history, not a second decision.

Budget is an execution invariant, carried explicitly as
`reobserve_spent`, `fallback_spent`, `trajectory_depth`: an operation
whose budget is spent cannot execute even if a malformed artifact
scores it 1.0.

## 6. Experiment configuration (frozen)

- Eval gate policy sets `reject_on_insufficient = True` (changed from
  the prior experiment; System A control is re-baselined under the
  identical policy).
- Arms differ ONLY by configured availability, never by table:
  - A (control): current gate, no Return.
  - B: Return with coarse profile available, no fallback artifact.
  - C: Return with fallback artifact, no coarse profile.
  - D: both available.
  - Plus: fallback-directly arm (native P3 baseline alone, no gate,
    no Return) for the §10 ablation.
- Trajectory budget state (`reobserve_spent`, `fallback_spent`,
  `trajectory_depth`) travels explicitly through the trajectory driver
  (harness-held for the experiment; production would carry it).
- Privilege boundary (construction-time, violations invalidate the
  experiment, not just the result): Return, fallback, and
  re-observation consume only captured observation data and frozen
  artifacts. No symbolic environment state. No post-enactment `after`
  state anywhere in the decision path (afters remain scoring- and
  verification-only, as now).

## 7. Return trace (transient)

```python
@dataclass(frozen=True)
class ReturnDecisionDTO:
    decision_id: str
    source_vpm_id: str
    trigger: str
    trigger_authority: str
    trigger_prediction_id: str
    operation: str  # REOBSERVE | FALLBACK | STOP
    disposition: str | None  # ABSTAIN | ESCALATE once terminal, else None
    return_policy_id: str
    recovery_memory_id: str | None
    observation_profile_id: str | None
    path: tuple[str, ...]
    decided_by: str
    budget_reobserve_spent: bool
    budget_fallback_spent: bool
    budget_trajectory_depth: int
    reasons: tuple[str, ...]
```

Transient Observer decision, not memory. `decided_by` names the step
that produced the final decision (`PRIMARY`, `REOBSERVE`, `FALLBACK`);
`path` records the traversed operations for trajectory decomposition;
`disposition` carries the deterministic STOP reading (§3.3).

## 8. Full loop

```text
OBSERVE → Source VPM
    ↓
REMEMBER → Policy / Evidence / Future / Validity VPMs
    ↓
ASSESS AUTHORITY → SUPPORT_ONLY / MAY_VETO / STALE / OOD
    ↓
CHOOSE → candidate action
    ├─ authorized ─→ COMMIT → ENACT → VERIFY → STRENGTHEN
    └─ authority insufficient ─→ RETURN → Return Policy VPM
            ├── REOBSERVE (once) ─→ OBSERVE (coarse profile) → loop
            ├── FALLBACK (once) ─→ native P3 ranking, no veto
            └── STOP ─→ ABSTAIN (empty path) / ESCALATE (used path)
```

Invariant (frozen): Return happens **before task enactment**. After a
real task action is enacted there is no "return": observe consequence,
verify, update validity, next cycle. Pre-commit contradiction goes to
Return; post-enactment failure goes to Evidence.

## 9. Metrics

Decision outcome: baseline accuracy, final accuracy after Return,
initial/final coverage, correction gain/harm/net, abstention rate,
escalation rate. Return-specific: invocation rate; REOBSERVE / FALLBACK
/ STOP counts with ABSTAIN-vs-ESCALATE disposition split; recovery
success and harm rates; success by trigger; **support-restored rate**
(B's primary mechanism metric: coarse projection supported after
native OOD — if B improves accuracy while support-restored ≈ 0, the
claimed mechanism is wrong); **fallback acceptance rate** and
**fallback correct rate** (C/D mechanism telemetry showing where
recovery actually came from). Mechanism telemetry per recovery:
initial trigger, operation, new observation/memory identity, second
authority assessment, final decision, final correctness — decomposing
observation-improved vs fallback-differed vs unresolved.

Definitions (frozen):

```text
return_gain = initial abstains/wrong AND Return produces correct action
return_harm = initial correct/safely-abstained AND Return produces incorrect
return_net  = return_gain - return_harm
```

Coverage reported separately; converting abstentions into guesses
without net gain fails.

## 10. Success, kills, ablations

GO only if `return_net > 0` with bounded harm and telemetry attributing
gain to the declared mechanism. Pre-registered arm expectations:

```text
A — control.
B — predicted weak/negative overall; possible benefit on noise only.
C — predicted to recover some task-change losses toward native P3
    baseline performance.
D — not expected to create a new source of intelligence; any gain
    should be attributable to successful B-style support restoration
    or C-style fallback recovery, and telemetry must decompose it
    accordingly.
```

Kill 1: `return_harm >= return_gain` (guessing, not recovery). Kill 2:
REOBSERVE restores no support / repeats the decision (representation
churn). Kill 3: fallback or re-observation consumes hidden state
(structurally precluded by §6; an audit failure here invalidates the
run). Kill 4: useful Return needs unbounded trajectory or repeated
model invocation.

Ablations: the four arms, plus fallback-directly versus
fallback-after-justified-Return (determines whether Return adds value
or the fallback should have been primary).

Valid negative result: Return does not improve goal-change; OOD
abstention stands as the least-harmful compiled behavior. Do not add
complexity to rescue coverage.

## 11. Explicitly not built

Autonomous exploration, arbitrary planning, LLM fallback, human
approval workflow, memory promotion, regional authority VPM, learned
Return network, neural world model, indefinite retries, automatic
policy rewriting, automatic retraining.

Regional validity stays out until the frozen trigger is observed:
memory valid in region A but stale in region B, with global STALE
demonstrably costing correct region-A decisions. Until then,
(model, action) validity stands. This experiment adds one
telemetry-only instrument toward that tripwire: mismatch rate by
coarse observation cluster (e.g. frame quadrants crossed with action),
reported but never gated. The cluster partition must be predeclared
or fitted from training data and frozen before evaluation — no
evaluation-time clustering followed by post-hoc discovery of a bad
region. No authority changes, no gating, no regional policy from
that telemetry.

## 12. Verdict questions (to answer at completion)

1. Does bounded Return improve goal-change behavior?
2. Which recovery operation accounts for the gain?
3. Does re-observation genuinely add new decision evidence?
4. Does fallback outperform honest abstention?
5. How much additional harm does Return introduce?
6. When should the Observer stop returning and escalate?
7. Is the Return policy stable enough as a compiled VPM?
8. Did any result trigger the regional-authority tripwire?
9. What next? 10. What explicitly not?

## 13. Claim boundary

Success would support: *when remembered policy or expected consequence
loses decision authority, ZeroModel can address a bounded compiled
recovery policy that determines whether to re-observe, use an
identified fallback, or remain abstinent before enactment.* It would
not support general planning, autonomous self-correction, open-world
recovery, guaranteed safe fallback, causal reasoning, or universal
agent behavior.

## 14. Core hypothesis

> **A system that knows when its memory should not command action can
> sometimes recover by addressing another bounded source of evidence,
> without surrendering the right to abstain.**

A relevant paper must now contribute to detecting the need to Return,
choosing a recovery operation, acquiring genuinely new evidence, or
deciding when recovery must terminate. Better predictors alone are
secondary.
