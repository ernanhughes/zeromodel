# 1. Executive verdict

Disposition: supporting infrastructure. At commit 2b12b58096676baa8dfb5d59b512e07b161612d4, the VPM artifact kernel is coherent for finite scored tables and bounded policy lookup. It is useful as an integration contract for identity, mapping, persistence, traces, and portable consumers. It is not, by itself, a compelling replacement for a simple decision table unless those integration properties are required.

# 2. Frozen target and environment

Resolved SHA: 2b12b58096676baa8dfb5d59b512e07b161612d4. OS: Windows-10 AMD64. Python: 3.11.4. pip: 26.1.2. Lua: unavailable in this environment. Existing uncommitted changes were present before the audit; audit outputs were written under audit-output.

# 3. Simplest accurate description

A VPM is an immutable source score table plus an explicit layout recipe, normalized view matrix, row/column permutations, provenance, and a SHA-256 content identity. Policy lookup treats source rows as finite states and selected metrics as candidate actions.

# 4. Architecture and package ownership

Core owns ScoreTable, LayoutRecipe, VPMArtifact, bundle, render, views, VPMPolicyLookup, and Lua export. Analysis adds Q diagnostics and finite property checking. Video supplies the arcade policy fixture used by examples. Package boundaries are coherent for this slice: core has no internal dependency; video depends on core.

# 5. Claim-by-claim findings

C1 passes for bounded finite tables. Identity includes spec_version, source values, row IDs, metric IDs, source metadata, layout recipe, normalized values, row/column order, and provenance. C2 passes through VPMArtifact.cell. C3 passes for tested ViewProfile layouts. C4 and C5 pass for finite row-addressed lookup using raw values by default. C6 passes for minimal .vpm bundle round-trip and tamper rejection. C7 export passes, but Lua runtime parity was not reproduced locally. C8 is partially supported as integration utility, not as novelty.

# 6. Reproduced evidence

Focused core tests: 11 passed, 1 skipped. Focused analysis tests: 8 passed. arcade_shooter_policy cleared the wave in 22 steps with score 4 versus random average 0.4. lua_edge_policy generated audit-output/generated_arcade_policy.lua with policy artifact eb7523f406b45ac30b478fe9528db8f89a548693b0add2fc8d3e51c4badd857e and plan a7f318d764ebdb7d509cae9728aa5ecbd4a0b89ea51a72067885427e94cc01b7. criticality_verification generated policy, verification, failed, and repaired artifacts and preserved the unsafe counterexample.

# 7. Adversarial checks

The probe changed one raw value, row ID, metric ID, row order, column order, metadata, recipe name, provenance, and parent relationship; each changed artifact identity. Canonically identical reconstruction preserved identity. Duplicate IDs, NaN, incomplete explicit row order, shape mismatch, bad artifact ID, unsupported bundle version, changed matrix, and changed manifest ID were rejected. Reordered mappings survived bundle restoration. Unknown row, duplicate action metrics, and action/evidence overlap failed explicitly.

# 8. Baseline comparison

An ordinary matrix or decision table can already store finite values, identifiers, JSON metadata, deterministic argmax, traces, content hashes, and JSON serialization. ZeroModel adds validated construction, layout recipes, normalized views, source/view coordinate mapping, provenance/parent conventions, rendering, bundle identity, evidence/action separation, compiled plan identity, and Lua source export. These are essential for audit/version/portable-consumer workflows; decorative for a one-off lookup table.

# 9. Utility analysis

The strongest application is a closed enumerable policy compiled once, reviewed, shipped, and read without invoking the producer. Users are embedded/runtime engineers and auditors. They replace ad hoc policy tables, model calls, or scripts. Required evidence before broader adoption: Lua runtime parity in target environments, performance/size measurements, human inspection study, and clearer governance/authenticity integration.

# 10. Documentation and terminology issues

README language around Visual AI Computing and intelligence is broad but partly bounded by docs/claims-audit.md. Safe wording is finite identified score/policy artifacts with deterministic lookup and traceability. PNG rendering is not self-describing. Compiled means materialized lookup plan/source, not performance-optimized binary compilation.

# 11. Positive findings

The artifact kernel has deterministic identity, rejects malformed finite-table inputs, preserves mappings, and restores bundles without ID drift. Policy lookup returns full decision traces and defaults to raw source policy semantics. The examples import production implementations rather than forked logic.

# 12. Negative findings

Lua runtime consumption was not reproduced because Lua is not installed. Utility over a fair conventional baseline is not proven by tests; it remains an engineering/integration argument. Rendering smoke tests do not prove visual inspection usefulness. Bundle integrity is hash-based reconstruction, not authenticity.

# 13. Boundaries

No open-world generalization, semantic correctness of scores, authorship, authorization, real-time performance, or unknown-state fallback is established. Identity is canonical content identity, not behavioral or semantic equivalence.

# 14. Recommended disposition

Supporting infrastructure. Keep it as the conservative core artifact contract used by higher-level packages. Do not market the kernel alone as a core product until utility evidence exists against a fair baseline.

# 15. Required next experiments

Install Lua 5.4 and run complete arcade Lua parity. Add malformed zip/member tests. Add property-style mapping tests over permutations and numeric edge cases including negative zero. Compare reviewer time/error rate using VPM traces versus a conventional decision table. Measure construction, lookup, bundle size, and Lua module size on larger finite policies.

# 16. Complete finding index

F01 deterministic identity; F02 mapping persistence; F03 minimal bundle boundary; F04 raw policy lookup and evidence exclusion; F05 Lua runtime parity not reproduced; F06 utility is integration value, not novelty.
