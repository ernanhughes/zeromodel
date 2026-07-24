# Stage P16B — Group-Owned Dataset Splits

P16B repairs the dataset leakage identified by the P0–P16 adversarial review.

Split ownership now belongs to `RecordedInteractionDTO.sequence_id`, not to individual interaction identity. Every frame or interaction in one sequence/episode therefore receives the same deterministic train, validation, or test assignment. `SplitAssignmentDTO` remains interaction-addressable so existing downstream consumers can resolve a partition without learning a new storage contract.

The manifest and split-assignment semantic versions advance to `/2`. Existing P2 split artifacts remain historical lineage and must not be treated as equivalent to P16B manifests.

Dataset validation also emits the error finding `identical_source_across_splits` whenever one exact `source_pixel_digest` appears in more than one partition, even when every duplicate has the same action label. Conflicting labels continue to emit `conflicting_actions_for_identical_source`. A defensive `sequence_crosses_splits` finding and manifest invariant prevent manually assembled manifests from violating group ownership.

P16B does not yet introduce manifest-owned validation/test partition DTOs, temporal-window overlap analysis, near-duplicate image detection, or re-run the downstream calibration and promotion experiments. Those belong to the subsequent provenance and vertical-integration repair slices.
