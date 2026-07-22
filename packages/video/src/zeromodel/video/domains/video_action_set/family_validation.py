from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    FRAME_INVALID_CLOSURE_VERSION,
    VALID_OBSERVATION_UNIVERSE_VERSION,
)
from zeromodel.video.domains.video_action_set.observation_universe import (
    _canonical_observation_digest_index,
    canonical_observation_universe,
)
from zeromodel.video.domains.video_action_set.pixel_digest import array_digest


def validate_materialized_family_record(record: Mapping[str, Any]) -> str:
    pixels = record.get("pixels")
    if pixels is not None and record.get("observation_pixel_digest") != array_digest(
        np.ascontiguousarray(pixels, dtype=np.uint8)
    ):
        return "stale_observation_digest"
    metadata = record.get("metadata", {})
    trace = metadata.get("family_intervention_trace")
    if trace:
        output_digest = trace.get("output_observation_digest")
        if output_digest is not None and output_digest != record.get(
            "observation_pixel_digest"
        ):
            return "family_output_digest_mismatch"
        if trace.get("changed_pixel_count") == 0:
            return "family_no_op"
    if (
        record.get("expected_disposition") == "distinguishable_invalid_input"
        and pixels is not None
    ):
        digest = str(
            record.get("observation_pixel_digest")
            or array_digest(np.ascontiguousarray(pixels, dtype=np.uint8))
        )
        if _canonical_observation_digest_index().get(digest):
            return "invalid_family_valid_state_collision"
    if record.get("event_type") == "gap_unknown" and pixels is not None:
        return "gap_event_has_pixels"
    return "ok"


def frame_invalid_closure_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    canonical = canonical_observation_universe()
    canonical_index = _canonical_observation_digest_index()
    by_family: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("expected_disposition") != "distinguishable_invalid_input":
            continue
        family = str(record.get("family"))
        row = by_family.setdefault(
            family,
            {
                "family_id": family,
                "frame_count": 0,
                "canonical_collision_count": 0,
                "valid_decode_count": 0,
                "no_op_count": 0,
                "malformed_count": 0,
                "collision_examples": [],
            },
        )
        row["frame_count"] += 1
        pixels = record.get("pixels")
        if pixels is None:
            row["malformed_count"] += 1
            continue
        digest = str(
            record.get("observation_pixel_digest")
            or array_digest(np.ascontiguousarray(pixels, dtype=np.uint8))
        )
        collisions = canonical_index.get(digest, [])
        if collisions:
            row["canonical_collision_count"] += 1
            row["valid_decode_count"] += 1
            if len(row["collision_examples"]) < 3:
                row["collision_examples"].append(
                    {
                        "frame_id": record.get("frame_id"),
                        "observation_pixel_digest": digest,
                        "canonical_rows": collisions,
                    }
                )
        trace = record.get("metadata", {}).get("family_intervention_trace", {})
        if trace.get("changed_pixel_count") == 0:
            row["no_op_count"] += 1
        status = validate_materialized_family_record(record)
        if status != "ok":
            row["malformed_count"] += 1
    totals = {
        "frame_count": sum(row["frame_count"] for row in by_family.values()),
        "canonical_collision_count": sum(
            row["canonical_collision_count"] for row in by_family.values()
        ),
        "valid_decode_count": sum(
            row["valid_decode_count"] for row in by_family.values()
        ),
        "no_op_count": sum(row["no_op_count"] for row in by_family.values()),
        "malformed_count": sum(row["malformed_count"] for row in by_family.values()),
    }
    passed = (
        totals["frame_count"] > 0
        and totals["canonical_collision_count"] == 0
        and totals["valid_decode_count"] == 0
        and totals["no_op_count"] == 0
        and totals["malformed_count"] == 0
    )
    payload = {
        "version": FRAME_INVALID_CLOSURE_VERSION,
        "canonical_universe_version": canonical["version"],
        "canonical_universe_digest": canonical["universe_digest"],
        "valid_observation_universe_version": VALID_OBSERVATION_UNIVERSE_VERSION,
        "families": list(by_family.values()),
        "totals": totals,
        "status": "passed" if passed else "failed",
    }
    payload["closure_digest"] = canonical_sha256(payload)
    return payload


__all__ = ["frame_invalid_closure_summary", "validate_materialized_family_record"]
