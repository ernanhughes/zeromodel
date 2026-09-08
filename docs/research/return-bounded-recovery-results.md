# Return experiment: results (frozen negative)

Companion to the frozen spec `return-bounded-recovery.md`. No spec text
is revised here; this note records the outcome, answers the §23 verdict
questions, and freezes the negative result. Seeds {0,1,2}, train 24
episodes, eval 10 per split. Results JSON (git-ignored artifacts):
`artifacts/return_benchmark/seeds012/return-results.json`.

## Accounting correction (spec compliance)

The first run counted harm as correct→wrong only, which is
structurally zero because recovery runs solely post-abstention. The
frozen definition counts abstain→wrong-commit as harm
(`_return_harmed`, unit-pinned in `test_return_benchmark.py`):
recovery commits are harm when they overturn a correct initial OR
convert a safe abstention into a wrong action. All numbers below use
the corrected accounting. No policy, table, or threshold was changed.

## Headline: NO-GO — harmful recovery on goal-change

Warehouse goal-change (focal condition), corrected metrics:

| arm | final acc | coverage | return_gain | return_harm | return_net |
|---|---|---|---|---|---|
| A gate-only | 0.091 | 0.765 | 0 | 0 | +0.000 |
| B reobserve | 0.091 | 0.897 | 0.000 | 0.050 | −0.050 |
| C fallback | 0.091 | 0.950 | 0.000 | 0.103 | −0.103 |
| D combined | 0.091 | 1.000 | 0.000 | 0.153 | −0.153 |
| E fallback-direct | 0.141 | 1.000 | 0 | 0 | +0.000 |

Fallback commits: 16 on change, correct 0.000. Reobserve restores
support (rate 1.000) with zero correctness gain. Kill 1 fires
(`return_harm >= return_gain`) on every split where Return engages:
clean C/D −0.014, bg-shift C/D −0.028, noise −0.005…−0.019.

The experiment's own verdict: **Return restored coverage without
restoring knowledge, converting justified uncertainty into incorrect
commitment.** Support restored ≠ knowledge restored ≠ correctness
restored. Decision support ≠ decision competence.

## Claim boundaries (frozen with the result)

1. `candidate_count=1` is an experimental terminalization. With the
   native `candidate_count=3`, vetoed candidates rerank internally and
   Return rarely fires — Return is not load-bearing in the normal
   multi-candidate path and this configuration must not quietly become
   the default.
2. OOD abstention stands as the least-harmful behavior under the
   available memories (valid §20 negative). E (0.141) beating C/D-final
   (0.091) shows fallback-directly outperforming gated Return, so
   Return adds no value here — but E is also mostly wrong; nothing
   in this condition beats honest abstention on harm.
3. Support restoration is not information acquisition: coarse pooling
   makes shifted inputs look familiar (support 1.0) without revealing
   changed task semantics (gain 0). Kill 2 fires for goal-change.
4. A real fallback needs different evidence or an independent knowledge
   source (fresh compiled memory, separate sensor, symbolic
   instrumentation) — not the same evidence through an older policy.
   Nothing of that kind is built to rescue this result.

## Mechanism integrity (why this negative counts)

- Triggers diversify as designed (warehouse change: 16 CONTRADICTED +
  11 PREDICTOR_REJECTED); dispositions split ABSTAIN/ESCALATE
  correctly; budget never overspent (malformed-artifact test pins this).
- Halves (per-episode pooled): D-change harm 0.210 → 0.105 as STALE
  withdraws veto authority — the authority loop visibly engages even
  while Return itself fails to help.
- Regional mismatch telemetry concentrates on FIRE/push clusters under
  change (e.g. bottom-left|FIRE 0.69) with no valid-in-A/stale-in-B
  inversion: the regional-authority tripwire does not fire.
- Arcade contributes nothing (ceiling; one invocation total): reported,
  not hidden.

## §23 verdict answers

1. No improvement on goal-change; harm where it engages. 2. No
operation accounts for gain (gain is 0; fallback commits explain the
harm). 3. Re-observation adds no decision evidence here (support
without correctness). 4. Fallback does not beat honest abstention
(0.000 correct on 16 commits). 5. Harm 0.005–0.153 per split where
Return engages, 0 elsewhere. 6. Escalate when the trajectory attempted
recovery and still cannot commit; the data never justified escalation
over abstention on accuracy. 7. Yes: deterministic recompile verified,
all 16 rows select frozen winners. 8. No regional tripwire. 9. Next:
only a condition where fallback knowledge is actually fresh; periodic
recompilation is the obvious candidate, unevaluated. 10. Not: any of
§21, plus no rescuing this null with complexity.

## Architectural status (unchanged)

Implemented mechanism ✅ · deterministic policy ✅ · bounded
trajectory ✅ · mechanism attribution ✅ · improved decisions ❌.
The mechanism is worth keeping (bounded, traceable, falsifiable
recovery attempts); its operational value is not established. The
Observer loop stands as: Observe → Remember → Assess → Choose →
Return-or-Commit → Verify → Strengthen, with honest abstention the
correct terminal behavior when no memory contains the answer.
