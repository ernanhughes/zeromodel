from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set.arcade_observation import (
    render_row_frame,
    shooter_config_payload,
)
from zeromodel.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.domains.video_action_set.contracts import (
    ARCADE_RENDERER_CONTRACT_VERSION,
    GAP_EVENT_VERSION,
    OBSERVATION_OPERATION_CHAIN_VERSION,
)
from zeromodel.domains.video_action_set.observation_legacy_adapters import (
    operation_chain,
    operation_record,
)
from zeromodel.domains.video_action_set.observation_provenance import (
    gap_event_operation_chain,
    valid_frame_operation_chain,
)
from zeromodel.domains.video_action_set.observation_provenance_dto import (
    ObservationOperationChainDTO,
    ObservationOperationDTO,
)
from zeromodel.domains.video_action_set.observation_replay import (
    replay_observation_operation_chain,
    validate_observation_operation_chain,
)
from zeromodel.domains.video_action_set.pixel_digest import array_digest
from zeromodel.domains.video_action_set.transformations import (
    _transformation_parameters,
)


ROW_ID = "tank=0|target=0|cooldown=0"
OTHER_ROW_ID = "tank=1|target=1|cooldown=0"
THIRD_ROW_ID = "tank=2|target=2|cooldown=0"
EXACT_PARAMS = _transformation_parameters("exact", 12345)
TRANSFORMED_PARAMS = _transformation_parameters(
    "bounded_translation_photometric",
    12345,
)
SOURCE_DIGEST = (
    "sha256:49e46341a170608e2bf8db63064e3d2235727a64ceddafa0cf9970c2707fe443"
)
TRANSFORMED_DIGEST = (
    "sha256:c173cfaa13d5f8d5da8ba4d72e129416d733f383219bbb44897608c321a7a86f"
)
FAKE_DIGEST = "sha256:" + "a" * 64
OTHER_FAKE_DIGEST = "sha256:" + "b" * 64

EXPECTED_EXACT_CHAIN = {
    "version": "zeromodel-video-observation-operation-chain/v1",
    "final_emitted_digest": SOURCE_DIGEST,
    "operation_chain_digest": (
        "sha256:a90cd05c48b2a6232b53ba8ac0b0b6e90596ad7aa127273a629b60a5dd3affce"
    ),
    "operations": [
        {
            "index": 0,
            "operation": "render_canonical_row",
            "operation_version": "zeromodel-arcade-shooter-render-state-frame/v1",
            "input_digests": [],
            "parameters": {
                "row_id": ROW_ID,
                "renderer": "zeromodel.arcade_policy.render_state_frame",
                "config": {"width": 7, "wave": [0, 6, 1, 5], "max_steps": 32},
            },
            "parameter_digest": (
                "sha256:7a0435e38e07209f4cf9fcf4707057ac9afb33a727090d5d72f985176326a9b4"
            ),
            "output_digest": SOURCE_DIGEST,
            "operation_digest": (
                "sha256:01390f67c13d6959645fb2fd4506e085f768c21c71766a288daa2e4c638a1239"
            ),
        },
        {
            "index": 1,
            "operation": "apply_bounded_transformation",
            "operation_version": "zeromodel-video-action-set-transformation-family/v1",
            "input_digests": [SOURCE_DIGEST],
            "parameters": {"transformation_parameters": EXACT_PARAMS},
            "parameter_digest": (
                "sha256:ef5f3d6d9500db5c270da6edb3a88362fc66c42d10bb50b78b18aded9cc49a8a"
            ),
            "output_digest": SOURCE_DIGEST,
            "operation_digest": (
                "sha256:a72a385b4979dc924a3e3f0c162e56340d1edc4a01eeb9cee7f26ccc2a396c7c"
            ),
        },
        {
            "index": 2,
            "operation": "emit_observation",
            "operation_version": "zeromodel-video-observation-operation-chain/v1",
            "input_digests": [SOURCE_DIGEST],
            "parameters": {"event_type": "frame"},
            "parameter_digest": (
                "sha256:45da9ed29fdf87e86e95196cd98a172ff3684a40cdf31ca52775f6acd5d18d75"
            ),
            "output_digest": SOURCE_DIGEST,
            "operation_digest": (
                "sha256:99e9f654442d6a8d5ba183b40b649c2f52640202c827d45cbf9f8c84c1fcf415"
            ),
        },
    ],
}

EXPECTED_TRANSFORMED_CHAIN = {
    "version": "zeromodel-video-observation-operation-chain/v1",
    "final_emitted_digest": TRANSFORMED_DIGEST,
    "operation_chain_digest": (
        "sha256:4662c45b549809a86b23fff8460d8acd78fd294259308d21e0c77a2e33da46c0"
    ),
    "operations": [
        EXPECTED_EXACT_CHAIN["operations"][0],
        {
            "index": 1,
            "operation": "apply_bounded_transformation",
            "operation_version": "zeromodel-video-action-set-transformation-family/v1",
            "input_digests": [SOURCE_DIGEST],
            "parameters": {"transformation_parameters": TRANSFORMED_PARAMS},
            "parameter_digest": (
                "sha256:0af58c5b5cba4915da08e9c0682d91fa9278bded80aaed33bf504e0d16e01a32"
            ),
            "output_digest": TRANSFORMED_DIGEST,
            "operation_digest": (
                "sha256:fd58511e161772c6a5b6fbeb239ab0bc41ab19a47d44adeaa7de70cab93640cc"
            ),
        },
        {
            "index": 2,
            "operation": "emit_observation",
            "operation_version": "zeromodel-video-observation-operation-chain/v1",
            "input_digests": [TRANSFORMED_DIGEST],
            "parameters": {"event_type": "frame"},
            "parameter_digest": (
                "sha256:45da9ed29fdf87e86e95196cd98a172ff3684a40cdf31ca52775f6acd5d18d75"
            ),
            "output_digest": TRANSFORMED_DIGEST,
            "operation_digest": (
                "sha256:e8ce74e554289ee256562e56d13339fa024f368f9dd9023d870ab4f081010c9c"
            ),
        },
    ],
}

GAP_EVENT = {
    "version": "zeromodel-video-action-set-gap-event/v1",
    "position": 2,
    "duration_frames": 1,
    "reason": "declared_gap_or_unknown_action",
    "event_id": "sha256:" + "1" * 64,
}

EXPECTED_GAP_CHAIN = {
    "version": "zeromodel-video-observation-operation-chain/v1",
    "final_emitted_digest": None,
    "operation_chain_digest": (
        "sha256:89e78eb86fc3a2d309d68cc6be92c22644986381eb207b80b6c1fb77608e0e6f"
    ),
    "operations": [
        {
            "index": 0,
            "operation": "emit_typed_gap_event",
            "operation_version": "zeromodel-video-action-set-gap-event/v1",
            "input_digests": [],
            "parameters": GAP_EVENT,
            "parameter_digest": (
                "sha256:0b286cba80ad09f1fb1740a05ef971076deccf8759f63a21c8eb2295584af3eb"
            ),
            "output_digest": None,
            "operation_digest": (
                "sha256:275e275a07b57387510d4dc6b821d6988819dc64baf36b6eccb7fba6aaa1e3d9"
            ),
        },
        {
            "index": 1,
            "operation": "emit_observation",
            "operation_version": "zeromodel-video-observation-operation-chain/v1",
            "input_digests": [None],
            "parameters": {"event_type": "gap_unknown"},
            "parameter_digest": (
                "sha256:bd3805cfc593ddc0c885e000683fff8a80d6a8df665f2859aa33afff5537c305"
            ),
            "output_digest": None,
            "operation_digest": (
                "sha256:6657e267f5280e620fecc26e946f35c5f2ad6db862db04c37c0b644087720758"
            ),
        },
    ],
}


def _unused_conflicting_splice_executor(**_kwargs: Any) -> tuple[np.ndarray, dict]:
    raise AssertionError("conflicting-splice executor must not be used")


def _unused_critical_corruption_executor(
    _source: np.ndarray,
    _coordinate_manifest: dict,
) -> tuple[np.ndarray, dict]:
    raise AssertionError("critical-corruption executor must not be used")


def _replay(chain: dict[str, Any]) -> dict[str, Any]:
    return replay_observation_operation_chain(
        chain,
        conflicting_splice_executor=_unused_conflicting_splice_executor,
        critical_corruption_executor=_unused_critical_corruption_executor,
    )


def _validate(record: dict[str, Any]) -> str:
    return validate_observation_operation_chain(
        record,
        conflicting_splice_executor=_unused_conflicting_splice_executor,
        critical_corruption_executor=_unused_critical_corruption_executor,
    )


def _render_op(index: int, row_id: str) -> dict[str, Any]:
    pixels = render_row_frame(row_id)
    return operation_record(
        index=index,
        operation="render_canonical_row",
        operation_version=ARCADE_RENDERER_CONTRACT_VERSION,
        input_digests=[],
        parameters={
            "row_id": row_id,
            "renderer": "zeromodel.arcade_policy.render_state_frame",
            "config": shooter_config_payload(),
        },
        output_digest=array_digest(pixels),
    )


def _emit_op(index: int, digest: str | None, *, event_type: str = "frame") -> dict:
    return operation_record(
        index=index,
        operation="emit_observation",
        operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
        input_digests=[digest],
        parameters={"event_type": event_type},
        output_digest=digest,
    )


def _record(
    chain: dict[str, Any],
    pixels: object,
    *,
    event_type: str = "frame",
    observation_pixel_digest: str | None | object = ...,
) -> dict[str, Any]:
    if observation_pixel_digest is ...:
        observation_pixel_digest = chain["final_emitted_digest"]
    return {
        "metadata": {"observation_operation_chain": chain},
        "observation_pixel_digest": observation_pixel_digest,
        "event_type": event_type,
        "pixels": pixels,
    }


def _array_with_digest_change(source: np.ndarray) -> np.ndarray:
    output = np.array(source, copy=True)
    output[0, 0] = 255 if int(output[0, 0]) != 255 else 0
    return output


def _stale_chain(
    *,
    operation: str = "stale_repeat_replace",
    second_input_digest: str | None = None,
    output_digest: str | None = None,
) -> dict[str, Any]:
    render_op = _render_op(0, OTHER_ROW_ID)
    replacement_digest = str(render_op["output_digest"])
    second_input_digest = (
        replacement_digest if second_input_digest is None else second_input_digest
    )
    output_digest = replacement_digest if output_digest is None else output_digest
    ops = [
        render_op,
        operation_record(
            index=1,
            operation=operation,
            operation_version="zeromodel-video-action-set-family-temporal/v1",
            input_digests=[FAKE_DIGEST, second_input_digest],
            parameters={
                "source_frame_index": 0,
                "destination_frame_index": 1,
                "maximum_stale_horizon": 1,
            },
            output_digest=output_digest,
        ),
        _emit_op(2, output_digest),
    ]
    return operation_chain(ops, output_digest)


def _provider_control_chain() -> dict[str, Any]:
    render_op = _render_op(0, ROW_ID)
    digest = str(render_op["output_digest"])
    ops = [
        render_op,
        operation_record(
            index=1,
            operation="emit_provider_identical_control_observation",
            operation_version="zeromodel-video-action-set-family-information-control/v3",
            input_digests=[digest],
            parameters={
                "current_row_id": ROW_ID,
                "provider_observation_boundary_version": (
                    "zeromodel-video-action-set-provider-observation-boundary/v1"
                ),
            },
            output_digest=digest,
        ),
        _emit_op(2, digest),
    ]
    return operation_chain(ops, digest)


def _splice_chain(
    *,
    second_input_digest: str | None = None,
    output_digest: str | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    primary_op = _render_op(0, ROW_ID)
    secondary_op = _render_op(1, OTHER_ROW_ID)
    output = np.maximum(render_row_frame(ROW_ID), render_row_frame(OTHER_ROW_ID))
    actual_output_digest = array_digest(output)
    output_digest = actual_output_digest if output_digest is None else output_digest
    second_input_digest = (
        str(secondary_op["output_digest"])
        if second_input_digest is None
        else second_input_digest
    )
    ops = [
        primary_op,
        secondary_op,
        operation_record(
            index=2,
            operation="compose_simultaneous_target_evidence",
            operation_version="zeromodel-video-action-set-family-conflicting-action-splice/v3",
            input_digests=[primary_op["output_digest"], second_input_digest],
            parameters={
                "primary_row_id": ROW_ID,
                "secondary_row_id": OTHER_ROW_ID,
                "primary_action_id": "left",
                "secondary_action_id": "right",
                "mask_manifest": {"version": "test"},
            },
            output_digest=output_digest,
        ),
        _emit_op(3, output_digest),
    ]
    return operation_chain(ops, output_digest), output


def _critical_chain(
    *,
    input_digest: str | None = None,
    output_digest: str | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    render_op = _render_op(0, ROW_ID)
    output = _array_with_digest_change(render_row_frame(ROW_ID))
    actual_output_digest = array_digest(output)
    input_digest = (
        str(render_op["output_digest"]) if input_digest is None else input_digest
    )
    output_digest = actual_output_digest if output_digest is None else output_digest
    ops = [
        render_op,
        operation_record(
            index=1,
            operation="apply_critical_coordinate_corruption",
            operation_version="zeromodel-video-action-set-family-critical-evidence-corruption/v1",
            input_digests=[input_digest],
            parameters={"critical_coordinates": {"coordinates": [[0, 0]]}},
            output_digest=output_digest,
        ),
        _emit_op(2, output_digest),
    ]
    return operation_chain(ops, output_digest), output


def _copy_with_operation(
    chain: dict[str, Any],
    index: int,
    updates: dict[str, Any],
) -> dict[str, Any]:
    copied = deepcopy(chain)
    copied["operations"][index].update(updates)
    copied["operations"][index]["operation_digest"] = canonical_sha256(
        {
            key: value
            for key, value in copied["operations"][index].items()
            if key != "operation_digest"
        },
    )
    copied["operation_chain_digest"] = canonical_sha256(
        {
            key: value
            for key, value in copied.items()
            if key != "operation_chain_digest"
        },
    )
    return copied


def test_valid_frame_operation_chains_are_frozen() -> None:
    assert valid_frame_operation_chain(ROW_ID, EXACT_PARAMS) == EXPECTED_EXACT_CHAIN
    assert (
        valid_frame_operation_chain(ROW_ID, TRANSFORMED_PARAMS)
        == EXPECTED_TRANSFORMED_CHAIN
    )


def test_gap_event_operation_chain_is_frozen_and_uses_contract_constant() -> None:
    assert GAP_EVENT_VERSION == "zeromodel-video-action-set-gap-event/v1"
    chain = gap_event_operation_chain(GAP_EVENT)

    assert chain == EXPECTED_GAP_CHAIN
    assert chain["operations"][0]["operation_version"] == GAP_EVENT_VERSION


def test_benchmark_retains_direct_aliases_and_dto_backed_adapters() -> None:
    digest = SOURCE_DIGEST
    op = _emit_op(0, digest)
    chain = operation_chain([op], digest)

    assert benchmark._valid_frame_operation_chain is valid_frame_operation_chain
    assert benchmark._gap_event_operation_chain is gap_event_operation_chain
    assert ObservationOperationDTO.from_dict(op).to_dict() == op
    assert ObservationOperationChainDTO.from_dict(chain).to_dict() == chain


def test_valid_frame_replay_reconstructs_pixels_and_state() -> None:
    replay = _replay(EXPECTED_TRANSFORMED_CHAIN)

    assert replay["final_emitted_digest"] == TRANSFORMED_DIGEST
    assert replay["typed_gap"] is False
    assert replay["operation_count"] == 3
    assert replay["pixels"].shape == (16, 28)
    assert replay["pixels"].dtype == np.uint8
    assert replay["pixels"].flags.c_contiguous
    assert array_digest(replay["pixels"]) == TRANSFORMED_DIGEST


def test_typed_gap_replay_emits_no_pixels() -> None:
    replay = _replay(EXPECTED_GAP_CHAIN)

    assert replay == {
        "pixels": None,
        "final_emitted_digest": None,
        "typed_gap": True,
        "operation_count": 2,
    }


@pytest.mark.parametrize(
    ("operation", "missing_message", "mismatch_message"),
    [
        (
            "stale_repeat_replace",
            "stale repeat input digest missing",
            "stale repeat output digest mismatch",
        ),
        (
            "impossible_transition_replace",
            "impossible transition input digest missing",
            "impossible transition output digest mismatch",
        ),
    ],
)
def test_replacement_operations_require_only_second_input_digest(
    operation: str,
    missing_message: str,
    mismatch_message: str,
) -> None:
    chain = _stale_chain(operation=operation)
    replay = _replay(chain)

    assert replay["final_emitted_digest"] == chain["final_emitted_digest"]
    assert array_digest(replay["pixels"]) == chain["final_emitted_digest"]

    with pytest.raises(VPMValidationError, match=missing_message):
        _replay(_stale_chain(operation=operation, second_input_digest=FAKE_DIGEST))
    with pytest.raises(VPMValidationError, match=mismatch_message):
        _replay(_stale_chain(operation=operation, output_digest=OTHER_FAKE_DIGEST))


def test_provider_identical_control_passes_through_pixels() -> None:
    chain = _provider_control_chain()
    replay = _replay(chain)

    assert replay["typed_gap"] is False
    assert replay["final_emitted_digest"] == SOURCE_DIGEST
    assert array_digest(replay["pixels"]) == SOURCE_DIGEST


def test_conflicting_splice_replay_uses_injected_executor() -> None:
    chain, output = _splice_chain()
    calls: list[dict[str, Any]] = []

    def executor(**kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        calls.append(kwargs)
        np.testing.assert_array_equal(
            kwargs["primary_pixels"], render_row_frame(ROW_ID)
        )
        np.testing.assert_array_equal(
            kwargs["secondary_pixels"],
            render_row_frame(OTHER_ROW_ID),
        )
        return output, {"trace": "test"}

    replay = replay_observation_operation_chain(
        chain,
        conflicting_splice_executor=executor,
        critical_corruption_executor=_unused_critical_corruption_executor,
    )

    assert len(calls) == 1
    assert array_digest(replay["pixels"]) == chain["final_emitted_digest"]

    with pytest.raises(VPMValidationError, match="splice input digest missing"):
        replay_observation_operation_chain(
            _splice_chain(second_input_digest=FAKE_DIGEST)[0],
            conflicting_splice_executor=executor,
            critical_corruption_executor=_unused_critical_corruption_executor,
        )
    with pytest.raises(VPMValidationError, match="splice output digest mismatch"):
        replay_observation_operation_chain(
            _splice_chain(output_digest=FAKE_DIGEST)[0],
            conflicting_splice_executor=executor,
            critical_corruption_executor=_unused_critical_corruption_executor,
        )


def test_critical_corruption_replay_uses_injected_executor() -> None:
    chain, output = _critical_chain()
    calls: list[dict[str, Any]] = []

    def executor(
        source: np.ndarray, coordinate_manifest: dict
    ) -> tuple[np.ndarray, dict]:
        calls.append({"source": source, "coordinate_manifest": coordinate_manifest})
        np.testing.assert_array_equal(source, render_row_frame(ROW_ID))
        assert coordinate_manifest == {"coordinates": [[0, 0]]}
        return output, {"trace": "test"}

    replay = replay_observation_operation_chain(
        chain,
        conflicting_splice_executor=_unused_conflicting_splice_executor,
        critical_corruption_executor=executor,
    )

    assert len(calls) == 1
    assert array_digest(replay["pixels"]) == chain["final_emitted_digest"]

    with pytest.raises(
        VPMValidationError,
        match="critical corruption input digest missing",
    ):
        replay_observation_operation_chain(
            _critical_chain(input_digest=FAKE_DIGEST)[0],
            conflicting_splice_executor=_unused_conflicting_splice_executor,
            critical_corruption_executor=executor,
        )
    with pytest.raises(
        VPMValidationError,
        match="critical corruption output digest mismatch",
    ):
        replay_observation_operation_chain(
            _critical_chain(output_digest=FAKE_DIGEST)[0],
            conflicting_splice_executor=_unused_conflicting_splice_executor,
            critical_corruption_executor=executor,
        )


@pytest.mark.parametrize(
    ("chain", "message"),
    [
        (
            operation_chain(
                [
                    operation_record(
                        index=0,
                        operation="unsupported",
                        operation_version="test/v1",
                        input_digests=[],
                        parameters={},
                        output_digest=SOURCE_DIGEST,
                    ),
                ],
                SOURCE_DIGEST,
            ),
            "unsupported operation chain operation",
        ),
        (
            operation_chain(
                [
                    operation_record(
                        index=0,
                        operation="emit_typed_gap_event",
                        operation_version=GAP_EVENT_VERSION,
                        input_digests=[],
                        parameters=GAP_EVENT,
                        output_digest=SOURCE_DIGEST,
                    ),
                ],
                SOURCE_DIGEST,
            ),
            "typed gap operation must not emit pixels",
        ),
        (
            operation_chain(
                [
                    operation_record(
                        index=0,
                        operation="emit_observation",
                        operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
                        input_digests=[],
                        parameters={"event_type": "gap_unknown"},
                        output_digest=None,
                    ),
                ],
                None,
            ),
            "no-pixel emit input mismatch",
        ),
    ],
)
def test_replay_rejects_unsupported_and_emit_mismatches(
    chain: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(VPMValidationError, match=message):
        _replay(chain)


def test_replay_rejects_emit_digest_mismatch() -> None:
    chain = operation_chain(
        [
            _render_op(0, ROW_ID),
            operation_record(
                index=1,
                operation="emit_observation",
                operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
                input_digests=[FAKE_DIGEST],
                parameters={"event_type": "frame"},
                output_digest=SOURCE_DIGEST,
            ),
        ],
        SOURCE_DIGEST,
    )

    with pytest.raises(
        VPMValidationError, match="emit operation input/output mismatch"
    ):
        _replay(chain)


def test_replay_rejects_render_and_transformation_digest_mismatches() -> None:
    with pytest.raises(
        VPMValidationError,
        match="render operation output digest mismatch",
    ):
        _replay(
            _copy_with_operation(
                EXPECTED_EXACT_CHAIN, 0, {"output_digest": FAKE_DIGEST}
            )
        )

    bad_input = deepcopy(EXPECTED_EXACT_CHAIN)
    bad_input["operations"][1] = operation_record(
        index=1,
        operation="apply_bounded_transformation",
        operation_version=bad_input["operations"][1]["operation_version"],
        input_digests=[FAKE_DIGEST],
        parameters=bad_input["operations"][1]["parameters"],
        output_digest=SOURCE_DIGEST,
    )
    bad_input["operation_chain_digest"] = canonical_sha256(
        {
            key: value
            for key, value in bad_input.items()
            if key != "operation_chain_digest"
        },
    )
    with pytest.raises(VPMValidationError, match="transformation input digest missing"):
        _replay(bad_input)

    bad_output = deepcopy(EXPECTED_EXACT_CHAIN)
    bad_output["operations"][1] = operation_record(
        index=1,
        operation="apply_bounded_transformation",
        operation_version=bad_output["operations"][1]["operation_version"],
        input_digests=[SOURCE_DIGEST],
        parameters=bad_output["operations"][1]["parameters"],
        output_digest=FAKE_DIGEST,
    )
    bad_output["operations"][2] = _emit_op(2, FAKE_DIGEST)
    bad_output["final_emitted_digest"] = FAKE_DIGEST
    bad_output["operation_chain_digest"] = canonical_sha256(
        {
            key: value
            for key, value in bad_output.items()
            if key != "operation_chain_digest"
        },
    )
    with pytest.raises(
        VPMValidationError,
        match="transformation output digest mismatch",
    ):
        _replay(bad_output)


def test_validate_observation_operation_chain_record_contract() -> None:
    pixels = render_row_frame(ROW_ID)

    assert _validate(_record(EXPECTED_EXACT_CHAIN, pixels)) == "ok"
    assert (
        _validate(_record(EXPECTED_GAP_CHAIN, None, event_type="gap_unknown")) == "ok"
    )
    assert _validate({"metadata": {}, "event_type": "frame"}) == (
        "final_observation_provenance_mismatch"
    )
    assert _validate(
        _record(EXPECTED_EXACT_CHAIN, pixels, observation_pixel_digest=FAKE_DIGEST)
    ) == ("final_observation_provenance_mismatch")
    assert _validate(_record(EXPECTED_GAP_CHAIN, None, event_type="frame")) == (
        "final_observation_provenance_mismatch"
    )
    bad_pixels = np.array(pixels, copy=True)
    bad_pixels[0, 0] = 255 if int(bad_pixels[0, 0]) != 255 else 0
    assert _validate(_record(EXPECTED_EXACT_CHAIN, bad_pixels)) == (
        "final_observation_provenance_mismatch"
    )


def test_validate_observation_pixels_are_normalized_to_uint8_contiguous() -> None:
    pixels = render_row_frame(ROW_ID)
    wide = np.zeros((16, 56), dtype=np.uint8)
    wide[:, ::2] = pixels
    non_contiguous = wide[:, ::2]
    assert not non_contiguous.flags.c_contiguous

    for value in (pixels.tolist(), pixels.astype(np.int16), non_contiguous):
        assert _validate(_record(EXPECTED_EXACT_CHAIN, value)) == "ok"


def test_legacy_wrappers_delegate_to_core_and_real_family_executors() -> None:
    replay = benchmark.replay_observation_operation_chain(EXPECTED_EXACT_CHAIN)

    assert replay["final_emitted_digest"] == SOURCE_DIGEST
    assert array_digest(replay["pixels"]) == SOURCE_DIGEST
    assert (
        benchmark.validate_observation_operation_chain(
            _record(EXPECTED_EXACT_CHAIN, replay["pixels"]),
        )
        == "ok"
    )
    assert (
        benchmark.validate_observation_operation_chain(
            _record(EXPECTED_GAP_CHAIN, None, event_type="gap_unknown"),
        )
        == "ok"
    )

    critical_chain = benchmark._critical_corruption_operation_chain(
        ROW_ID,
        EXACT_PARAMS,
        benchmark._critical_coordinate_manifest(),
    )
    critical_replay = benchmark.replay_observation_operation_chain(critical_chain)
    assert (
        benchmark.validate_observation_operation_chain(
            _record(critical_chain, critical_replay["pixels"]),
        )
        == "ok"
    )
