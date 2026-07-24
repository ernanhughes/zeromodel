# P16B Migration Note

`PerceptionDatasetManifestDTO` and `SplitAssignmentDTO` now use semantic version `/2`.

Consumers must rebuild manifests from authoritative `RecordedInteractionDTO` inputs. Existing `/1` manifests were split by `interaction_id`; they cannot be upgraded by changing the version string because their partition ownership and dataset identity were derived under different semantics.

Rebuilding may move interactions between train, validation, and test. Any calibration, promotion, untouched-test report, reference profile, or health report derived from `/1` splits belongs to the old lineage and must be regenerated before being used as current evidence.
