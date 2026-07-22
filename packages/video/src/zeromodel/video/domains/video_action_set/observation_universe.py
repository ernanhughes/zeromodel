from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from zeromodel.video.arcade_policy import ACTIONS, ShooterConfig, compile_policy_artifact
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.observation.visual_address import ImageObservation
from zeromodel.video.domains.video_action_set.arcade_observation import (
    render_row_frame,
    renderer_identity,
    shooter_config_payload,
)
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    ARCADE_RENDERER_CONTRACT_VERSION,
    CANONICAL_OBSERVATION_UNIVERSE_VERSION,
    FRAME_SHAPE,
    TRANSFORMATION_FAMILY_VERSION,
    VALID_OBSERVATION_UNIVERSE_VERSION,
    VALID_TRANSFORMATION_PARAMETER_UNIVERSE_VERSION,
)
from zeromodel.video.domains.video_action_set.pixel_digest import array_digest
from zeromodel.video.domains.video_action_set.transformations import _apply_transformation


def canonical_prototypes(
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, tuple[str, str, str, ImageObservation]]:
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    prototypes = {}
    for row_id in policy.source.row_ids:
        frame = render_row_frame(str(row_id), config=config)
        observation = ImageObservation(frame, source_id=f"canonical:{row_id}")
        prototypes[f"prototype:{row_id}"] = (
            str(row_id),
            lookup.choose(str(row_id)),
            observation.raw_digest,
            observation,
        )
    return prototypes


def canonical_observation_universe(
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    digest_to_rows: dict[str, list[dict[str, str]]] = {}
    prototypes = (
        canonical_prototypes()
        if config == ShooterConfig()
        else canonical_prototypes(config)
    )
    for prototype_id, (
        row_id,
        action_id,
        observation_raw_digest,
        observation,
    ) in prototypes.items():
        pixel_digest = array_digest(observation.pixels)
        entry = {
            "prototype_id": prototype_id,
            "row_id": row_id,
            "action_id": action_id,
            "image_observation_raw_digest": observation_raw_digest,
            "observation_pixel_digest": pixel_digest,
        }
        rows.append(entry)
        digest_to_rows.setdefault(pixel_digest, []).append(
            {"row_id": row_id, "action_id": action_id}
        )
    duplicate_groups = [
        {"observation_pixel_digest": digest, "rows": grouped}
        for digest, grouped in sorted(digest_to_rows.items())
        if len(grouped) > 1
    ]
    payload = {
        "version": CANONICAL_OBSERVATION_UNIVERSE_VERSION,
        "digest_semantics": "raw rendered uint8 bytes, excluding ImageObservation namespace",
        "frame_shape": list(FRAME_SHAPE),
        "row_count": len(rows),
        "rows": rows,
        "digest_to_rows": digest_to_rows,
        "duplicate_digest_group_count": len(duplicate_groups),
        "duplicate_digest_groups": duplicate_groups,
    }
    payload["universe_digest"] = canonical_sha256(payload)
    return payload


def _valid_transformation_parameter_key(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dx": int(params["dx"]),
        "dy": int(params["dy"]),
        "scale_percent": int(params["scale_percent"]),
        "offset": int(params["offset"]),
        "occlusion": None
        if params.get("occlusion") is None
        else dict(params["occlusion"]),
    }


def _valid_transformation_parameter_universe() -> dict[str, Any]:
    if hasattr(_valid_transformation_parameter_universe, "_cache"):
        return _valid_transformation_parameter_universe._cache  # type: ignore[attr-defined]
    by_key: dict[str, dict[str, Any]] = {}
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for scale in range(90, 106):
                for offset in range(0, 6):
                    params = {
                        "version": TRANSFORMATION_FAMILY_VERSION,
                        "family": "bounded_translation_photometric",
                        "seed": 0,
                        "dx": dx,
                        "dy": dy,
                        "scale_percent": scale,
                        "offset": offset,
                        "occlusion": None,
                    }
                    params["parameter_digest"] = canonical_sha256(
                        {
                            key: value
                            for key, value in params.items()
                            if key != "parameter_digest"
                        }
                    )
                    by_key[
                        canonical_sha256(_valid_transformation_parameter_key(params))
                    ] = params
                    for top in range(0, 3):
                        for left in range(0, 3):
                            occluded = dict(params)
                            occluded["family"] = "compound_bounded"
                            occluded["occlusion"] = {
                                "top": top,
                                "left": left,
                                "height": 2,
                                "width": 3,
                                "value": 64,
                            }
                            occluded["parameter_digest"] = canonical_sha256(
                                {
                                    key: value
                                    for key, value in occluded.items()
                                    if key != "parameter_digest"
                                }
                            )
                            by_key[
                                canonical_sha256(
                                    _valid_transformation_parameter_key(occluded)
                                )
                            ] = occluded
    parameters = [by_key[key] for key in sorted(by_key)]
    payload = {
        "version": VALID_TRANSFORMATION_PARAMETER_UNIVERSE_VERSION,
        "transformation_contract_version": TRANSFORMATION_FAMILY_VERSION,
        "parameter_count": len(parameters),
        "parameters": parameters,
    }
    payload["parameter_universe_digest"] = canonical_sha256(payload)
    _valid_transformation_parameter_universe._cache = payload  # type: ignore[attr-defined]
    return payload


def _valid_transformed_observation_digest_index(
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, Any]:
    cache_key = canonical_sha256(
        {
            "config": shooter_config_payload(config),
            "transformation_contract_version": TRANSFORMATION_FAMILY_VERSION,
        }
    )
    if not hasattr(_valid_transformed_observation_digest_index, "_cache"):
        _valid_transformed_observation_digest_index._cache = {}  # type: ignore[attr-defined]
    cache: dict[str, dict[str, Any]] = getattr(
        _valid_transformed_observation_digest_index,
        "_cache",
    )
    if cache_key in cache:
        return cache[cache_key]
    canonical = canonical_observation_universe(config)
    parameter_universe = _valid_transformation_parameter_universe()
    digest_to_first: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    duplicate_examples: list[dict[str, Any]] = []
    for canonical_row in canonical["rows"]:
        row_id = str(canonical_row["row_id"])
        source_pixels = render_row_frame(row_id, config=config)
        for params in parameter_universe["parameters"]:
            output, _trace = _apply_transformation(source_pixels, params)
            digest = array_digest(output)
            provenance = {
                "row_id": row_id,
                "parameter_digest": params["parameter_digest"],
                "parameter_key_digest": canonical_sha256(
                    _valid_transformation_parameter_key(params)
                ),
            }
            previous = digest_to_first.get(digest)
            if previous is None:
                digest_to_first[digest] = provenance
            else:
                duplicate_count += 1
                if len(duplicate_examples) < 20:
                    duplicate_examples.append(
                        {
                            "observation_pixel_digest": digest,
                            "first": previous,
                            "duplicate": provenance,
                        }
                    )
    digest_set = sorted(digest_to_first)
    summary: dict[str, Any] = {
        "version": VALID_OBSERVATION_UNIVERSE_VERSION,
        "policy_artifact_id": compile_policy_artifact(config).artifact_id,
        "row_action_universe_digest": canonical_sha256(
            {
                "row_action": [
                    {"row_id": row["row_id"], "action_id": row["action_id"]}
                    for row in canonical["rows"]
                ]
            }
        ),
        "renderer_contract_version": ARCADE_RENDERER_CONTRACT_VERSION,
        "renderer_identity": renderer_identity(config),
        "shooter_config": shooter_config_payload(config),
        "shooter_config_digest": canonical_sha256(shooter_config_payload(config)),
        "transformation_contract_version": TRANSFORMATION_FAMILY_VERSION,
        "parameter_universe_digest": parameter_universe["parameter_universe_digest"],
        "parameter_universe_count": parameter_universe["parameter_count"],
        "canonical_universe_digest": canonical["universe_digest"],
        "exact_canonical_digest_count": len(canonical["digest_to_rows"]),
        "transformed_valid_digest_count": len(digest_set),
        "transformed_valid_output_count": len(canonical["rows"])
        * int(parameter_universe["parameter_count"]),
        "duplicate_transformed_digest_count": duplicate_count,
        "duplicate_transformed_digest_examples": duplicate_examples,
        "transformed_valid_digest_set_digest": canonical_sha256(digest_set),
    }
    payload: dict[str, Any] = {"digest_index": set(digest_set), "summary": summary}
    summary["universe_digest"] = canonical_sha256(summary)
    cache[cache_key] = payload
    return payload


def valid_observation_universe(
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, Any]:
    canonical = canonical_observation_universe(config)
    transformed = _valid_transformed_observation_digest_index(config)
    payload = {
        **transformed["summary"],
        "canonical_universe_version": canonical["version"],
        "valid_categories": ["canonical_exact", "bounded_transformation_family"],
        "exact_canonical_digest_to_rows": canonical["digest_to_rows"],
        "bounded_transformation_contract_version": TRANSFORMATION_FAMILY_VERSION,
    }
    payload["universe_digest"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "universe_digest"}
    )
    return payload


def _canonical_observation_digest_index(
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, list[dict[str, str]]]:
    return {
        str(digest): [dict(item) for item in rows]
        for digest, rows in canonical_observation_universe(config)[
            "digest_to_rows"
        ].items()
    }


def _canonical_collision_rows(
    pixels: np.ndarray,
    *,
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, str]]:
    return _canonical_observation_digest_index(config).get(
        array_digest(np.ascontiguousarray(pixels, dtype=np.uint8)),
        [],
    )


__all__ = [
    "_canonical_collision_rows",
    "_canonical_observation_digest_index",
    "_valid_transformation_parameter_key",
    "_valid_transformation_parameter_universe",
    "_valid_transformed_observation_digest_index",
    "canonical_observation_universe",
    "canonical_prototypes",
    "valid_observation_universe",
]
