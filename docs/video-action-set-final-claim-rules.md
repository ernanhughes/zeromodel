# Video Action-Set Final Claim Rules

Final claims can be promoted only after all of these are true:

- A final evaluation protocol is approved before final access.
- A final execution authorization with matching protocol and sealed-plan digests
  is present.
- The access ledger reaches `completed` exactly once and has a valid receipt.
- The final report is reconstructed from durable final artifacts and the ledger,
  not from rerunning final access.
- The final evaluation result is `passed`; `failed` or `indeterminate` outcomes
  cannot be restated as positive claims.

Claims must cite the final receipt digest, final evaluation digest, protocol
digest, sealed-plan digest, and access event-chain digest.

Claim and report builders accept a validated `FinalEvaluationResultDTO`, not a
free-form mapping. Its evaluation, evidence, protocol, and decision fields must
match the receipt exactly. A fabricated pass mapping or result from another
protocol/evidence bundle cannot produce claims or a report.

This repository intentionally contains no approved final claim text.
