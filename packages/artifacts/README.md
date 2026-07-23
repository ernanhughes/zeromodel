# zeromodel-artifacts

Canonical artifact reference, resolution, and content-addressed storage for
the ZeroModel workspace.

This package defines the stable, cross-package `ArtifactRef` identity and the
`ArtifactResolver` / `ArtifactStore` protocols other packages (such as
`zeromodel-trust` and `zeromodel-navigation`) use to persist and resolve their
own artifacts, without each package inventing its own storage layer.

It reuses `zeromodel.core`'s existing canonicalization and digest primitives
(`canonical_json_bytes`, `sha256_digest`) rather than redefining them.

## Report compilation and the compiled-report aggregate

On top of the storage kernel, this package also compiles typed external
reports into deterministic, source-bound VPM artifacts (`ReportAdapter`,
`AdaptedReportDTO`, `compile_report()`) and resolves the complete result as
one coherent aggregate:

```text
AdaptedReportDTO (persisted, resolvable)
    ↓
CompiledReportArtifactDTO (aggregate root)
    ├── adapted_report_ref  -> AdaptedReportDTO
    ├── score_table_ref     -> ScoreTable
    ├── layout_recipe_ref   -> LayoutRecipe
    └── vpm_artifact_ref    -> VPMArtifact
```

`load_compiled_report_aggregate()` resolves all four referenced artifacts
and runs `validate_compiled_report_aggregate()`, which proves the
collection is semantically closed - not merely that each object's own
digest is valid. A compiled report can reference a `ScoreTable` from one
report, a `LayoutRecipe` from another, and a `VPMArtifact` from a third,
each individually digest-valid; aggregate validation is what proves the
five objects actually describe the same report (cross-checked subjects,
dimensions, raw values, VPM view coordinates, and per-cell source
bindings). `build_compiled_report_closure_receipt()` produces an
auditable, content-addressed receipt only after every check passes -
never a partial result.

Compatibility between two compiled reports requires three independent
identities to agree: `compatibility_id` (a human label), and two content
digests: `compatibility_schema_id` (dimension ids, order, score semantics,
and value/target ranges) and `report_semantics_id` (report kind, subject
kind, dimension namespace, and duplicate-value policy - the layer that
distinguishes, for example, a report over sentences from a structurally
identical report over claims).

**Claims boundary:** ZeroModel can persist and reload a complete adapted
report and its compiled `ScoreTable`, `LayoutRecipe`, and `VPMArtifact` as
one content-addressed aggregate, and verify that every subject, dimension,
value, coordinate, source binding, and compatibility contract agrees
across all four representations. This does not include Trust-signed
compiled reports (Trust integration is a call-site composition, not
implemented here), cross-schema conversion, or automatic report repair.
