"""Regression guard for the ObservationDTO / OBSERVATION_RECORD_KEYS contract.

Stage A2.1 background: a Stage A2 session initially reported this as a
"production schema mismatch" after `tests/test_video_action_set_benchmark.py`
(a research file) failed with `VPMValidationError: observation record keys
mismatch`. Deeper investigation (tracing `ObservationDTO.from_record()` and
`to_record()` line by line) showed that claim was wrong: `OBSERVATION_RECORD_KEYS`
is not supposed to equal every dataclass field. It is deliberately the
top-level JSON RECORD schema - the keys `from_record()` reads directly off
its `record` argument - which is narrower than the full dataclass because
several fields are populated a different way:

- `benchmark_seed_digest`, `episode_plan_digest`, `provider_observation_descriptor`,
  `provider_observation_digest`, `observation_operation_chain` (-> `operation_chain`)
  are all read from *inside* the record's own `metadata` dict, not as separate
  top-level keys.
- `matrix_blob_id` is derived from the optional `pixels` key (already accounted
  for by `_record_keys()`'s second allowed key-set) plus `observation_pixel_digest`,
  not stored as its own top-level key.
- `final_access_id` is supplied as an external keyword argument to
  `from_record()`, never part of the record payload at all.

The actual bug was a stale, incomplete fixture in that one research test
(`fake_records` was missing 8 required top-level keys). That fixture was
corrected in the same stage; this test guards the *production* contract
those fields must continue to satisfy, independent of any one fixture.
"""

from __future__ import annotations

from dataclasses import fields

from zeromodel.video.domains.video_action_set.observation_dto import (
    OBSERVATION_RECORD_KEYS,
    ObservationDTO,
)

# Dataclass fields that are intentionally NOT top-level JSON record keys,
# with the reason each is excluded. If a new field is added to ObservationDTO
# and this test starts failing, decide explicitly which bucket it belongs to
# rather than silently appending it to either set.
METADATA_EMBEDDED_FIELDS = frozenset(
    {
        "benchmark_seed_digest",  # read from metadata["seed_digest"]
        "episode_plan_digest",  # read from metadata["episode_plan_digest"]
        "provider_observation_descriptor",  # read from metadata["provider_observation_descriptor"]
        "provider_observation_digest",  # read from metadata["provider_observation_digest"]
        "operation_chain",  # read from metadata["observation_operation_chain"]
    }
)
DERIVED_OR_EXTERNAL_FIELDS = frozenset(
    {
        "matrix_blob_id",  # derived from the optional "pixels" record key + observation_pixel_digest
        "final_access_id",  # supplied as an external keyword argument to from_record(), never in the record payload
    }
)


def test_observation_record_keys_is_a_strict_subset_of_dataclass_fields() -> None:
    dto_fields = {field.name for field in fields(ObservationDTO)}
    assert set(OBSERVATION_RECORD_KEYS).issubset(dto_fields)


def test_every_excluded_dataclass_field_is_explicitly_accounted_for() -> None:
    dto_fields = {field.name for field in fields(ObservationDTO)}
    excluded = dto_fields - set(OBSERVATION_RECORD_KEYS)
    accounted_for = METADATA_EMBEDDED_FIELDS | DERIVED_OR_EXTERNAL_FIELDS
    assert excluded == accounted_for, (
        f"ObservationDTO gained or lost a field without updating this contract test: "
        f"unaccounted={excluded - accounted_for}, stale_exclusions={accounted_for - excluded}"
    )


def test_metadata_embedded_and_derived_field_sets_do_not_overlap() -> None:
    assert METADATA_EMBEDDED_FIELDS.isdisjoint(DERIVED_OR_EXTERNAL_FIELDS)


def test_observation_record_keys_has_no_duplicates() -> None:
    assert len(OBSERVATION_RECORD_KEYS) == len(set(OBSERVATION_RECORD_KEYS))
