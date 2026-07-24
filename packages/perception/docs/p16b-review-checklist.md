# P16B Review Checklist

Reviewers should verify:

- every interaction sharing one `sequence_id` receives one split;
- manifest construction is order-independent;
- exact duplicate `source_pixel_digest` values cannot cross partitions silently;
- conflicting actions for identical pixels remain a separate error;
- manually assembled manifests cannot assign one sequence to multiple partitions;
- `/1` dataset manifests are treated as historical lineage rather than silently reinterpreted;
- downstream callers continue to resolve splits by interaction ID.
