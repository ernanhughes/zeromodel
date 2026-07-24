# Stage P17G — Audit-Gated Governed Execution

## Objective

Make the P17F cross-store integrity audit an explicit precondition and postcondition of governed rollback execution.

```text
lifecycle + governance + attempt journal
    ↓
pre-execution integrity audit
    ↓
audit authorization gate
    ↓
P17E journaled execution / P17D recovery
    ↓
post-execution integrity audit
    ↓
GovernanceExecutionGateDTO
```

## Accepted pre-states

P17G accepts only:

- `clean`: the complete governance audit is `valid`;
- `recover_prepared_attempt`: the only non-informational finding is one `attempt_prepared_incomplete` warning for the exact deterministic attempt being resumed.

An `invalid` audit always blocks execution. An `attention_required` audit concerning another attempt, a legacy receipt, or any unrelated warning also blocks execution.

This narrow exception prevents the integrity gate from deadlocking the crash-recovery path introduced in P17D and journaled in P17E.

## Postcondition

After execution or recovery, P17G reruns the full P17F audit. The operation is certified only when the resulting report is `valid`.

A lifecycle function returning successfully is therefore not sufficient evidence on its own. The lifecycle transition, governance receipt, attempt history, terminal event, target model, and pointer revision must compose into one valid chain.

## Gate artifact

`GovernanceExecutionGateDTO` content-addresses:

- recommendation and disposition identities;
- deterministic attempt identity;
- receipt identity;
- pre- and post-audit report identities;
- authorization mode;
- resulting pointer revision;
- gate semantics and version.

The gate artifact records why execution was permitted and which integrity reports surrounded it. It is not a second lifecycle authority.

## Safety boundary

P17G does not:

- repair invalid governance history;
- waive unrelated warnings;
- retry terminally failed attempts;
- infer that an old unjournaled receipt is safe to reuse;
- replace explicit operator approval;
- change lifecycle state outside the existing P17D/P17E path.

## Completion of P17

P17G closes the operational recommendation loop:

```text
health evidence
    ↓
recommendation
    ↓
operator disposition
    ↓
durable governance ledger
    ↓
crash-recoverable execution
    ↓
append-only attempt journal
    ↓
cross-store integrity audit
    ↓
audit-gated execution certification
```
