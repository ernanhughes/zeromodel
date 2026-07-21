# Video Action-Set Final Evaluation Protocol Review

This document is a review checklist for a future human-approved final evaluation
protocol. It is not an approval record.

Required review outcomes:

- The protocol status must remain `draft` or `review` until an authorized reviewer
  records `protocol_status: approved`, `approved_utc`, and `approved_by`.
- The protocol must bind the benchmark seed digest, sealed final plan digest,
  policy artifact identity, candidate set identity, and selected provider identity.
- Decision rules must be fixed before final access. They cannot depend on final
  evidence, tuning output, post-final threshold choice, or alternate operating
  points.
- The approved protocol must identify complete final evidence requirements and
  what makes the result `passed`, `failed`, or `indeterminate`.
- Any replacement protocol requires a new digest and a new authorization.

Codex agents must not approve this protocol, choose thresholds, select providers,
or transform this template into a live final authorization.
