# P18A Acceptance Criteria

P18A is complete when:

- identical inputs produce exactly equal transition artifacts;
- before and after Source VPMs must satisfy the same field-schema contract;
- every field has normalized before, after, absolute-change, signed-change, and threshold-crossing measurements;
- annotation identities are optional, schema-owned, unique, and measurement-neutral;
- the rendered transition VPM is deterministic and digest-verified;
- the top-level identity detects payload or PNG tampering;
- the package root exposes the P18A contracts while retaining the complete P17K surface;
- the clean wheel-installed perception suite passes.
