from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

import research.benchmarks.video_action_set_benchmark as benchmark
from zeromodel.video.arcade_policy import ShooterConfig, compile_policy_artifact
from zeromodel.video.domains.video_action_set import arcade_observation
from zeromodel.video.domains.video_action_set import observation_universe as universe
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    ARCADE_RENDERER_CONTRACT_VERSION,
    ARCADE_RENDERER_IDENTITY_VERSION,
    CANONICAL_OBSERVATION_UNIVERSE_VERSION,
    FRAME_SHAPE,
    TRANSFORMATION_FAMILY_VERSION,
    VALID_OBSERVATION_UNIVERSE_VERSION,
    VALID_TRANSFORMATION_PARAMETER_UNIVERSE_VERSION,
)
from zeromodel.video.domains.video_action_set.pixel_digest import array_digest
from zeromodel.observation.visual_address import ImageObservation


DEFAULT_CONFIG_PAYLOAD = {"width": 7, "wave": [0, 6, 1, 5], "max_steps": 32}
DEFAULT_CONFIG_DIGEST = (
    "sha256:9189c6aab306713893497c99f190ab9d6c84392033a628db0a96b371af9dbddd"
)
CUSTOM_CONFIG = ShooterConfig(width=5, wave=(4, 2, 0), max_steps=9)
CUSTOM_CONFIG_PAYLOAD = {"width": 5, "wave": [4, 2, 0], "max_steps": 9}
CUSTOM_CONFIG_DIGEST = (
    "sha256:c38ddd39e4e5f24c35d1b4f23589a46f3859938c783f6a1cf0dfa81720e508c0"
)

DEFAULT_RENDERER_IDENTITY = {
    "version": "zeromodel-arcade-shooter-renderer-identity/v1",
    "function": "zeromodel.arcade_policy.render_state_frame",
    "frame_shape": [16, 28],
    "cell_pixels": 4,
    "target_value": 220,
    "tank_value": 255,
    "cooldown_ready_value": 40,
    "cooldown_blocked_value": 160,
    "config": DEFAULT_CONFIG_PAYLOAD,
    "renderer_identity_digest": (
        "sha256:a08e75ed50d080b6944a9caf5e64674f3c3cb91c91aab1d8679d93ee2f121c16"
    ),
}
CUSTOM_RENDERER_DIGEST = (
    "sha256:2ef9610c1918d15a2303fcc9502dbf72275632ebe87ae2d3174e51fd26b52f72"
)

ROW_FRAME_DIGESTS = {
    "tank=0|target=0|cooldown=0": (
        "sha256:49e46341a170608e2bf8db63064e3d2235727a64ceddafa0cf9970c2707fe443"
    ),
    "tank=0|target=none|cooldown=0": (
        "sha256:76d6b55ab164806fa6d621e669de37b60b9cbd8f8fff2bbb610b6b0e034461f0"
    ),
    "tank=0|target=0|cooldown=1": (
        "sha256:fbd8cdf59131334bec00f39063847f869da54f963f81ac6c0b8c9e36c8a23df6"
    ),
    "tank=6|target=6|cooldown=0": (
        "sha256:63280c3c42468851dfe637431e33721faefaa30924a1fcf7bdf8aba4b38859bf"
    ),
    "tank=6|target=none|cooldown=1": (
        "sha256:7b71bf932b093ad9b6ebebb1ef50e8a43cda8ca35e5640f1edffe756f5ab71f5"
    ),
}

SELECTED_PROTOTYPES = {
    "prototype:tank=0|target=none|cooldown=0": (
        "tank=0|target=none|cooldown=0",
        "STAY",
        "sha256:3e3dc2d6661c710405e00ee6448c3966fa27a875ab7430d00f6bad2ab358d668",
        "sha256:76d6b55ab164806fa6d621e669de37b60b9cbd8f8fff2bbb610b6b0e034461f0",
    ),
    "prototype:tank=0|target=none|cooldown=1": (
        "tank=0|target=none|cooldown=1",
        "STAY",
        "sha256:ae08a24f369ba051a162349c04962dc1948a0c8901f726564ebd9ade70d365a7",
        "sha256:6170d693f571210bb02918cb4dba8be19dcc91ad842383e3c531fd7f2210bb2f",
    ),
    "prototype:tank=6|target=6|cooldown=0": (
        "tank=6|target=6|cooldown=0",
        "FIRE",
        "sha256:34f49182fe9701b001821e903af4b278ded06c741f659100c23e33e372f29f8d",
        "sha256:63280c3c42468851dfe637431e33721faefaa30924a1fcf7bdf8aba4b38859bf",
    ),
    "prototype:tank=6|target=6|cooldown=1": (
        "tank=6|target=6|cooldown=1",
        "STAY",
        "sha256:58ae1f648dfb98fbf9a10d9ff80cdd29eeb289b6eeb0e9a0c7c2e945b6bd0009",
        "sha256:ed6867cf59a01b51467bc540ce8e7e16474222386c8ceb0f31ad697741f47e8b",
    ),
}

CANONICAL_UNIVERSE_DIGEST = (
    "sha256:8fd3073bb511894d71d369610885e69d796cf686255e240e1407f5f7c7653385"
)
SELECTED_CANONICAL_ROWS = [
    {
        "prototype_id": "prototype:tank=0|target=none|cooldown=0",
        "row_id": "tank=0|target=none|cooldown=0",
        "action_id": "STAY",
        "image_observation_raw_digest": (
            "sha256:3e3dc2d6661c710405e00ee6448c3966fa27a875ab7430d00f6bad2ab358d668"
        ),
        "observation_pixel_digest": (
            "sha256:76d6b55ab164806fa6d621e669de37b60b9cbd8f8fff2bbb610b6b0e034461f0"
        ),
    },
    {
        "prototype_id": "prototype:tank=0|target=none|cooldown=1",
        "row_id": "tank=0|target=none|cooldown=1",
        "action_id": "STAY",
        "image_observation_raw_digest": (
            "sha256:ae08a24f369ba051a162349c04962dc1948a0c8901f726564ebd9ade70d365a7"
        ),
        "observation_pixel_digest": (
            "sha256:6170d693f571210bb02918cb4dba8be19dcc91ad842383e3c531fd7f2210bb2f"
        ),
    },
    {
        "prototype_id": "prototype:tank=6|target=6|cooldown=0",
        "row_id": "tank=6|target=6|cooldown=0",
        "action_id": "FIRE",
        "image_observation_raw_digest": (
            "sha256:34f49182fe9701b001821e903af4b278ded06c741f659100c23e33e372f29f8d"
        ),
        "observation_pixel_digest": (
            "sha256:63280c3c42468851dfe637431e33721faefaa30924a1fcf7bdf8aba4b38859bf"
        ),
    },
    {
        "prototype_id": "prototype:tank=6|target=6|cooldown=1",
        "row_id": "tank=6|target=6|cooldown=1",
        "action_id": "STAY",
        "image_observation_raw_digest": (
            "sha256:58ae1f648dfb98fbf9a10d9ff80cdd29eeb289b6eeb0e9a0c7c2e945b6bd0009"
        ),
        "observation_pixel_digest": (
            "sha256:ed6867cf59a01b51467bc540ce8e7e16474222386c8ceb0f31ad697741f47e8b"
        ),
    },
]

FIRST_PARAMETER = {
    "version": "zeromodel-video-action-set-transformation-family/v1",
    "family": "compound_bounded",
    "seed": 0,
    "dx": 1,
    "dy": 0,
    "scale_percent": 97,
    "offset": 5,
    "occlusion": {"top": 2, "left": 1, "height": 2, "width": 3, "value": 64},
    "parameter_digest": (
        "sha256:3a909e187b05e0fd0298d83e4fdcc62aa000dd6c0531c479b269218def4b7df8"
    ),
}
NON_OCCLUDED_PARAMETER = {
    "version": "zeromodel-video-action-set-transformation-family/v1",
    "family": "bounded_translation_photometric",
    "seed": 0,
    "dx": 1,
    "dy": -1,
    "scale_percent": 92,
    "offset": 2,
    "occlusion": None,
    "parameter_digest": (
        "sha256:007adfa53503b6d64cca245a64f0a6c3df0e7a10bd0b00360f87c2930032a759"
    ),
}
LAST_PARAMETER = {
    "version": "zeromodel-video-action-set-transformation-family/v1",
    "family": "compound_bounded",
    "seed": 0,
    "dx": -1,
    "dy": 1,
    "scale_percent": 100,
    "offset": 5,
    "occlusion": {"top": 2, "left": 1, "height": 2, "width": 3, "value": 64},
    "parameter_digest": (
        "sha256:8b5710759ab0d4499e2ee17cedf6bddfd8bd516b2003e1a29294dc079eab01b2"
    ),
}


def test_shooter_config_payload_is_frozen() -> None:
    assert arcade_observation.shooter_config_payload() == DEFAULT_CONFIG_PAYLOAD
    assert canonical_sha256(DEFAULT_CONFIG_PAYLOAD) == DEFAULT_CONFIG_DIGEST
    assert arcade_observation.shooter_config_payload(CUSTOM_CONFIG) == (
        CUSTOM_CONFIG_PAYLOAD
    )
    assert canonical_sha256(CUSTOM_CONFIG_PAYLOAD) == CUSTOM_CONFIG_DIGEST


@pytest.mark.parametrize(("row_id", "digest"), list(ROW_FRAME_DIGESTS.items()))
def test_render_row_frame_bytes_are_frozen(row_id: str, digest: str) -> None:
    frame = arcade_observation.render_row_frame(row_id)

    assert frame.shape == FRAME_SHAPE
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous
    assert array_digest(frame) == digest


def test_renderer_identity_is_frozen_and_config_bound() -> None:
    assert ARCADE_RENDERER_IDENTITY_VERSION == (
        "zeromodel-arcade-shooter-renderer-identity/v1"
    )
    assert ARCADE_RENDERER_CONTRACT_VERSION == (
        "zeromodel-arcade-shooter-render-state-frame/v1"
    )
    assert arcade_observation.renderer_identity() == DEFAULT_RENDERER_IDENTITY
    assert (
        arcade_observation.renderer_identity(CUSTOM_CONFIG)["renderer_identity_digest"]
        == CUSTOM_RENDERER_DIGEST
    )
    assert (
        arcade_observation.renderer_identity(CUSTOM_CONFIG)["renderer_identity_digest"]
        != DEFAULT_RENDERER_IDENTITY["renderer_identity_digest"]
    )


def test_canonical_prototypes_are_frozen() -> None:
    prototypes = universe.canonical_prototypes()
    keys = list(prototypes)

    assert len(prototypes) == 112
    assert keys[:3] == [
        "prototype:tank=0|target=none|cooldown=0",
        "prototype:tank=0|target=none|cooldown=1",
        "prototype:tank=0|target=0|cooldown=0",
    ]
    assert keys[-3:] == [
        "prototype:tank=6|target=5|cooldown=1",
        "prototype:tank=6|target=6|cooldown=0",
        "prototype:tank=6|target=6|cooldown=1",
    ]
    for key, expected in SELECTED_PROTOTYPES.items():
        row_id, action_id, raw_digest, observation = prototypes[key]
        assert (row_id, action_id, raw_digest, array_digest(observation.pixels)) == (
            expected
        )
        assert raw_digest != array_digest(observation.pixels)


def test_canonical_observation_universe_is_frozen() -> None:
    payload = universe.canonical_observation_universe()

    assert payload["version"] == CANONICAL_OBSERVATION_UNIVERSE_VERSION
    assert payload["digest_semantics"] == (
        "raw rendered uint8 bytes, excluding ImageObservation namespace"
    )
    assert payload["frame_shape"] == [16, 28]
    assert payload["row_count"] == 112
    assert payload["duplicate_digest_group_count"] == 0
    assert payload["duplicate_digest_groups"] == []
    assert payload["universe_digest"] == CANONICAL_UNIVERSE_DIGEST
    assert [
        payload["rows"][0],
        payload["rows"][1],
        payload["rows"][-2],
        payload["rows"][-1],
    ] == (SELECTED_CANONICAL_ROWS)
    for row in SELECTED_CANONICAL_ROWS:
        assert payload["digest_to_rows"][row["observation_pixel_digest"]] == [
            {"row_id": row["row_id"], "action_id": row["action_id"]}
        ]
        assert row["image_observation_raw_digest"] != row["observation_pixel_digest"]


def test_canonical_universe_preserves_default_call_semantics(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    pixels = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    observation = ImageObservation(pixels, source_id="canonical:test")

    def fake_prototypes(*args: object) -> dict[str, tuple[str, str, str, Any]]:
        calls.append(args)
        return {
            "prototype:test": (
                "test",
                "STAY",
                observation.raw_digest,
                observation,
            )
        }

    monkeypatch.setattr(universe, "canonical_prototypes", fake_prototypes)

    universe.canonical_observation_universe()
    universe.canonical_observation_universe(CUSTOM_CONFIG)

    assert calls == [(), (CUSTOM_CONFIG,)]


def test_parameter_key_includes_only_declared_fields() -> None:
    base = deepcopy(NON_OCCLUDED_PARAMETER)
    base_key = universe._valid_transformation_parameter_key(base)
    base_digest = canonical_sha256(base_key)
    assert base_key == {
        "dx": 1,
        "dy": -1,
        "scale_percent": 92,
        "offset": 2,
        "occlusion": None,
    }
    for field, value in {
        "dx": -1,
        "dy": 0,
        "scale_percent": 93,
        "offset": 3,
        "occlusion": {"top": 0, "left": 0, "height": 2, "width": 3, "value": 64},
    }.items():
        mutated = deepcopy(base)
        mutated[field] = value
        assert canonical_sha256(
            universe._valid_transformation_parameter_key(mutated)
        ) != (base_digest)
    for field, value in {
        "version": "changed-version",
        "family": "compound_bounded",
        "seed": 99,
        "parameter_digest": "sha256:" + "0" * 64,
    }.items():
        mutated = deepcopy(base)
        mutated[field] = value
        assert universe._valid_transformation_parameter_key(mutated) == base_key


def test_valid_transformation_parameter_universe_is_frozen() -> None:
    payload = universe._valid_transformation_parameter_universe()

    assert payload is universe._valid_transformation_parameter_universe()
    assert payload["version"] == VALID_TRANSFORMATION_PARAMETER_UNIVERSE_VERSION
    assert payload["transformation_contract_version"] == TRANSFORMATION_FAMILY_VERSION
    assert payload["parameter_count"] == 8640
    assert payload["parameter_universe_digest"] == (
        "sha256:9fc78f73b196125e2723871f61fe6b5ee9614ac83a5768d91031c4b9b2780ac2"
    )
    assert payload["parameters"][0] == FIRST_PARAMETER
    assert payload["parameters"][21] == NON_OCCLUDED_PARAMETER
    assert payload["parameters"][-1] == LAST_PARAMETER
    key_digests = [
        canonical_sha256(universe._valid_transformation_parameter_key(params))
        for params in payload["parameters"]
    ]
    assert len(key_digests) == len(set(key_digests))


def _fake_parameter(index: int) -> dict[str, Any]:
    return {
        "version": TRANSFORMATION_FAMILY_VERSION,
        "family": "bounded_translation_photometric",
        "seed": 0,
        "dx": 0,
        "dy": 0,
        "scale_percent": 100,
        "offset": index,
        "occlusion": None,
        "parameter_digest": "sha256:" + f"{index:064x}",
    }


def test_transformed_index_mechanics_with_bounded_fixture(monkeypatch) -> None:
    config = ShooterConfig(width=2, wave=(0,), max_steps=3)
    params = [_fake_parameter(index) for index in range(22)]
    canonical = {
        "rows": [
            {"row_id": "row:a", "action_id": "LEFT"},
            {"row_id": "row:b", "action_id": "RIGHT"},
        ],
        "digest_to_rows": {
            "sha256:canonical-a": [{"row_id": "row:a", "action_id": "LEFT"}],
            "sha256:canonical-b": [{"row_id": "row:b", "action_id": "RIGHT"}],
        },
        "universe_digest": "sha256:canonical",
    }
    render_calls: list[str] = []

    def fake_render(row_id: str, *, config: ShooterConfig) -> np.ndarray:
        render_calls.append(row_id)
        return np.array([[1 if row_id == "row:a" else 2]], dtype=np.uint8)

    def fake_apply(_source: np.ndarray, _params: dict[str, Any]):
        return np.array([[7]], dtype=np.uint8), {}

    monkeypatch.setattr(
        universe, "canonical_observation_universe", lambda _config: canonical
    )
    monkeypatch.setattr(
        universe,
        "_valid_transformation_parameter_universe",
        lambda: {
            "parameter_universe_digest": "sha256:params",
            "parameter_count": len(params),
            "parameters": params,
        },
    )
    monkeypatch.setattr(universe, "render_row_frame", fake_render)
    monkeypatch.setattr(universe, "_apply_transformation", fake_apply)
    monkeypatch.setattr(
        universe._valid_transformed_observation_digest_index,
        "_cache",
        {},
        raising=False,
    )

    result = universe._valid_transformed_observation_digest_index(config)
    summary = result["summary"]
    output_digest = array_digest(np.array([[7]], dtype=np.uint8))
    first_provenance = {
        "row_id": "row:a",
        "parameter_digest": params[0]["parameter_digest"],
        "parameter_key_digest": canonical_sha256(
            universe._valid_transformation_parameter_key(params[0])
        ),
    }

    assert result["digest_index"] == {output_digest}
    assert render_calls == ["row:a", "row:b"]
    assert summary["version"] == VALID_OBSERVATION_UNIVERSE_VERSION
    assert summary["policy_artifact_id"] == compile_policy_artifact(config).artifact_id
    assert summary["renderer_contract_version"] == ARCADE_RENDERER_CONTRACT_VERSION
    assert summary["renderer_identity"] == arcade_observation.renderer_identity(config)
    assert summary["shooter_config"] == arcade_observation.shooter_config_payload(
        config
    )
    assert summary["shooter_config_digest"] == canonical_sha256(
        arcade_observation.shooter_config_payload(config)
    )
    assert summary["parameter_universe_digest"] == "sha256:params"
    assert summary["parameter_universe_count"] == 22
    assert summary["canonical_universe_digest"] == "sha256:canonical"
    assert summary["exact_canonical_digest_count"] == 2
    assert summary["transformed_valid_digest_count"] == 1
    assert summary["transformed_valid_output_count"] == 44
    assert summary["duplicate_transformed_digest_count"] == 43
    assert len(summary["duplicate_transformed_digest_examples"]) == 20
    assert summary["duplicate_transformed_digest_examples"][0] == {
        "observation_pixel_digest": output_digest,
        "first": first_provenance,
        "duplicate": {
            "row_id": "row:a",
            "parameter_digest": params[1]["parameter_digest"],
            "parameter_key_digest": canonical_sha256(
                universe._valid_transformation_parameter_key(params[1])
            ),
        },
    }
    assert summary["transformed_valid_digest_set_digest"] == canonical_sha256(
        [output_digest]
    )
    assert summary["universe_digest"] == canonical_sha256(
        {key: value for key, value in summary.items() if key != "universe_digest"}
    )
    assert universe._valid_transformed_observation_digest_index(config) is result


def test_valid_universe_assembly_replaces_inherited_digest(monkeypatch) -> None:
    canonical = {
        "version": CANONICAL_OBSERVATION_UNIVERSE_VERSION,
        "digest_to_rows": {
            "sha256:canonical-a": [{"row_id": "row:a", "action_id": "LEFT"}]
        },
    }
    transformed_summary = {
        "version": VALID_OBSERVATION_UNIVERSE_VERSION,
        "policy_artifact_id": "artifact:test",
        "universe_digest": "sha256:inherited",
    }

    monkeypatch.setattr(
        universe, "canonical_observation_universe", lambda _config: canonical
    )
    monkeypatch.setattr(
        universe,
        "_valid_transformed_observation_digest_index",
        lambda _config: {
            "digest_index": {"sha256:bounded"},
            "summary": transformed_summary,
        },
    )

    payload = universe.valid_observation_universe()

    assert payload["valid_categories"] == [
        "canonical_exact",
        "bounded_transformation_family",
    ]
    assert (
        payload["canonical_universe_version"] == CANONICAL_OBSERVATION_UNIVERSE_VERSION
    )
    assert payload["exact_canonical_digest_to_rows"] == canonical["digest_to_rows"]
    assert payload["bounded_transformation_contract_version"] == (
        TRANSFORMATION_FAMILY_VERSION
    )
    assert payload["universe_digest"] != "sha256:inherited"
    assert payload["universe_digest"] == canonical_sha256(
        {key: value for key, value in payload.items() if key != "universe_digest"}
    )


def test_canonical_collision_lookup_normalizes_and_returns_copies() -> None:
    prototype = universe.canonical_prototypes()[
        "prototype:tank=0|target=none|cooldown=0"
    ][3]
    expected = [{"row_id": "tank=0|target=none|cooldown=0", "action_id": "STAY"}]

    assert universe._canonical_collision_rows(prototype.pixels) == expected
    assert universe._canonical_collision_rows(
        np.array(prototype.pixels, copy=True)
    ) == (expected)
    assert universe._canonical_collision_rows(prototype.pixels.astype(np.int16)) == (
        expected
    )
    assert universe._canonical_collision_rows(prototype.pixels.tolist()) == expected
    assert universe._canonical_collision_rows(np.zeros_like(prototype.pixels)) == []

    returned = universe._canonical_observation_digest_index()[
        array_digest(prototype.pixels)
    ]
    returned[0]["row_id"] = "mutated"
    assert universe._canonical_observation_digest_index()[
        array_digest(prototype.pixels)
    ] == (expected)


def test_legacy_benchmark_facade_exposes_moved_universe_symbols() -> None:
    assert benchmark._transition_config_payload is (
        arcade_observation.shooter_config_payload
    )
    assert benchmark._render_row_frame is arcade_observation.render_row_frame
    assert benchmark._renderer_identity is arcade_observation.renderer_identity
    assert benchmark.canonical_prototypes is universe.canonical_prototypes
    assert benchmark.canonical_observation_universe is (
        universe.canonical_observation_universe
    )
    assert benchmark.valid_observation_universe is universe.valid_observation_universe
    assert benchmark._valid_transformation_parameter_key is (
        universe._valid_transformation_parameter_key
    )
    assert benchmark._valid_transformation_parameter_universe is (
        universe._valid_transformation_parameter_universe
    )
    assert benchmark._valid_transformed_observation_digest_index is (
        universe._valid_transformed_observation_digest_index
    )
    assert benchmark._canonical_observation_digest_index is (
        universe._canonical_observation_digest_index
    )
    assert benchmark._canonical_collision_rows is universe._canonical_collision_rows
    assert benchmark.CANONICAL_OBSERVATION_UNIVERSE_VERSION == (
        CANONICAL_OBSERVATION_UNIVERSE_VERSION
    )
    assert benchmark.VALID_OBSERVATION_UNIVERSE_VERSION == (
        VALID_OBSERVATION_UNIVERSE_VERSION
    )
    assert benchmark._transition_config_payload() == DEFAULT_CONFIG_PAYLOAD
    assert (
        array_digest(benchmark._render_row_frame("tank=0|target=0|cooldown=0"))
        == ROW_FRAME_DIGESTS["tank=0|target=0|cooldown=0"]
    )
