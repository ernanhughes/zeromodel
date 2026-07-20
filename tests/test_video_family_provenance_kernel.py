from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set.arcade_observation import render_row_frame
from zeromodel.domains.video_action_set import family_provenance as provenance
from zeromodel.domains.video_action_set import family_validation as validation
from zeromodel.domains.video_action_set import frame_family_kernels as kernels
from zeromodel.domains.video_action_set.observation_replay import (
    replay_observation_operation_chain,
)
from zeromodel.domains.video_action_set.contracts import (
    CONFLICTING_ACTION_SPLICE_VERSION,
    CRITICAL_COORDINATE_SET_VERSION,
    CRITICAL_EVIDENCE_CORRUPTION_VERSION,
    CRITICAL_REGION_ID,
    FINAL_VISIBLE_TARGET_ACTION_EVIDENCE_VERSION,
    FRAME_INVALID_CLOSURE_VERSION,
    IMPOSSIBLE_TRANSITION_VERSION,
    INFORMATION_CONTROL_VERSION,
    SPLICE_MASK_VERSION,
    STALE_REPEATED_FRAME_VERSION,
    TARGET_REGION_ID,
)
from zeromodel.domains.video_action_set.pixel_digest import array_digest
from zeromodel.domains.video_action_set.transformations import (
    _transformation_parameters,
)


PRIMARY_ROW = "tank=0|target=0|cooldown=0"
SECONDARY_ROW = "tank=0|target=1|cooldown=0"
THIRD_ROW = "tank=0|target=2|cooldown=0"
EXACT_PARAMS = _transformation_parameters("exact", 12345)
PRIMARY_DIGEST = (
    "sha256:49e46341a170608e2bf8db63064e3d2235727a64ceddafa0cf9970c2707fe443"
)
SECONDARY_DIGEST = (
    "sha256:b72f28f8210df1a77c6bbb7f6486e9eabfe92b460f42a2d1c4f93e6d742a1bf6"
)
THIRD_DIGEST = "sha256:0c8dbd9bc06cceadda299372427a3ae57dfed9954a4955ba61270d53092266f5"
SPLICE_DIGEST = (
    "sha256:ca73312136f3875a09eba69d6372425ade15dcb2bbb31abeee8da75aeeef33cc"
)
CRITICAL_DIGEST = (
    "sha256:0751a64fcfb6d18949aa5f75a202118ceaa465de25dc69e05e2b8ad5146a8ab6"
)
ROW_ACTIONS = {
    PRIMARY_ROW: "FIRE",
    SECONDARY_ROW: "RIGHT",
    THIRD_ROW: "RIGHT",
}


def _spliced() -> tuple[np.ndarray, dict[str, Any]]:
    return kernels.apply_conflicting_splice(
        primary_pixels=render_row_frame(PRIMARY_ROW),
        secondary_pixels=render_row_frame(SECONDARY_ROW),
        primary_row_id=PRIMARY_ROW,
        secondary_row_id=SECONDARY_ROW,
        primary_action_id="FIRE",
        secondary_action_id="RIGHT",
        mask_manifest=kernels.splice_mask_manifest(),
    )


def _critical() -> tuple[np.ndarray, dict[str, Any]]:
    return kernels.apply_critical_corruption(
        render_row_frame(PRIMARY_ROW),
        kernels.critical_coordinate_manifest(),
    )


def _splice_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "primary_pixels": render_row_frame(PRIMARY_ROW),
        "secondary_pixels": render_row_frame(SECONDARY_ROW),
        "primary_row_id": PRIMARY_ROW,
        "secondary_row_id": SECONDARY_ROW,
        "primary_action_id": "FIRE",
        "secondary_action_id": "RIGHT",
        "mask_manifest": kernels.splice_mask_manifest(),
    }
    kwargs.update(overrides)
    return kwargs


def _frame_record(
    *,
    family: str,
    pixels: np.ndarray,
    trace: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "expected_disposition": "distinguishable_invalid_input",
        "event_type": "frame",
        "observation_pixel_digest": array_digest(pixels),
        "pixels": pixels,
        "metadata": {"family_intervention_trace": trace},
    }


def _source_record(row_id: str, digest: str) -> dict[str, Any]:
    return {
        "expected_row": row_id,
        "observation_pixel_digest": digest,
        "metadata": {"transformation_parameters": EXACT_PARAMS},
    }


def _conflicting_chain() -> dict[str, Any]:
    return provenance.conflicting_splice_operation_chain(
        primary_row_id=PRIMARY_ROW,
        secondary_row_id=SECONDARY_ROW,
        primary_action_id="FIRE",
        secondary_action_id="RIGHT",
        primary_transformation_parameters=EXACT_PARAMS,
        mask_manifest=kernels.splice_mask_manifest(),
    )


def _critical_chain() -> dict[str, Any]:
    return provenance.critical_corruption_operation_chain(
        PRIMARY_ROW,
        EXACT_PARAMS,
        kernels.critical_coordinate_manifest(),
    )


def test_frame_geometry_manifests_and_visible_evidence_are_frozen() -> None:
    mask = kernels.splice_mask_manifest()
    critical = kernels.critical_coordinate_manifest()
    primary = render_row_frame(PRIMARY_ROW)

    assert kernels.target_signal_coordinates()[0] == (2, 0)
    assert kernels.target_signal_coordinates()[-1] == (4, 27)
    assert len(kernels.target_signal_coordinates()) == 84
    assert kernels.target_slot_signal_coordinates(0) == (
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 2),
    )
    assert kernels.critical_coordinates() == (
        (7, 25),
        (7, 26),
        (8, 25),
        (8, 26),
    )
    assert critical == {
        "version": CRITICAL_COORDINATE_SET_VERSION,
        "criticality_source": "tiny_arcade_shooter_rendering",
        "criticality_region_id": CRITICAL_REGION_ID,
        "coordinates": [[7, 25], [7, 26], [8, 25], [8, 26]],
        "coordinate_set_digest": (
            "sha256:bdb6fcb07eafaeb00978a0fecabb470886f2d5ced3dee9383274aaefaf6fb081"
        ),
    }
    assert mask["version"] == SPLICE_MASK_VERSION
    assert mask["target_region_id"] == TARGET_REGION_ID
    assert mask["coordinate_count"] == 84
    assert mask["mask_digest"] == (
        "sha256:1850329620179f93ec0ddec778d559c16b7b5f9464b1ee8344e6c91b8b27e781"
    )
    assert kernels.final_visible_target_action_evidence(primary, ROW_ACTIONS) == {
        "version": FINAL_VISIBLE_TARGET_ACTION_EVIDENCE_VERSION,
        "final_visible_target_slots": [0],
        "final_tank_slot": 0,
        "final_cooldown": 0,
        "visible_target_action_map": {"0": "FIRE"},
        "visible_action_set": ["FIRE"],
        "conflicting_action_evidence_present": False,
        "visible_action_evidence_digest": (
            "sha256:2bd0d7336fa88ec919362ccdeb4bfc7474cc57d9e626d70796ba0972bf70835c"
        ),
    }


def test_splice_and_critical_intervention_outputs_are_frozen() -> None:
    spliced, splice_trace = _spliced()
    corrupted, critical_trace = _critical()

    assert array_digest(spliced) == SPLICE_DIGEST
    assert splice_trace["splice_trace_digest"] == (
        "sha256:0560f6990b65c67fc5f151fb1b2eb7c4fe34714dec11fb58a6b690ba2e8b9697"
    )
    assert splice_trace["visible_action_set"] == ["FIRE", "RIGHT"]
    assert splice_trace["canonical_collision_count"] == 0
    assert splice_trace["changed_pixel_count"] == 7
    assert splice_trace["action_relevant_region_contribution_counts"] == {
        "primary_target_pixel_count": 7,
        "secondary_target_pixel_count": 7,
        "secondary_additive_target_pixel_count": 7,
        "target_overlap_pixel_count": 0,
    }
    assert array_digest(corrupted) == CRITICAL_DIGEST
    assert critical_trace["critical_corruption_digest"] == (
        "sha256:942ad2b962b0b05271abbbd3fb47d80cb6cdcc911fbe0b6281d4d19c6ca6c8bd"
    )
    assert critical_trace["changed_pixel_count"] == 4
    assert [item["replacement"] for item in critical_trace["changes"]] == [
        255,
        255,
        255,
        255,
    ]


def test_conflicting_splice_validation_messages_and_precedence(monkeypatch) -> None:
    def expect(message: str, **overrides: Any) -> None:
        with pytest.raises(VPMValidationError, match=message):
            kernels.apply_conflicting_splice(**_splice_kwargs(**overrides))

    expect(
        "conflicting splice requires different governed actions",
        secondary_action_id="FIRE",
        secondary_pixels=np.zeros((15, 28), dtype=np.uint8),
    )
    expect(
        "splice sources must have the same shape",
        secondary_pixels=np.zeros((15, 28), dtype=np.uint8),
    )
    expect(
        "conflicting splice requires the canonical frame shape",
        primary_pixels=np.zeros((15, 28), dtype=np.uint8),
        secondary_pixels=np.zeros((15, 28), dtype=np.uint8),
    )
    expect(
        "unsupported conflicting splice mask version",
        mask_manifest=kernels.splice_mask_manifest() | {"version": "test/v0"},
    )
    expect(
        "conflicting splice requires the simultaneous target-evidence mask",
        mask_manifest=kernels.splice_mask_manifest() | {"mask_kind": "test"},
    )
    expect(
        "conflicting splice target coordinates do not match the frozen renderer geometry",
        mask_manifest=kernels.splice_mask_manifest() | {"coordinates": [[2, 0]]},
    )
    expect(
        "splice requires visible target evidence from both sources",
        primary_pixels=np.zeros((16, 28), dtype=np.uint8),
    )
    expect(
        "splice requires distinct secondary target evidence",
        secondary_pixels=render_row_frame(PRIMARY_ROW),
    )
    spliced, _trace = _spliced()
    expect(
        "splice output must not equal either source observation",
        secondary_pixels=spliced,
    )
    expect(
        "conflicting splice final visible target evidence does not imply conflicting actions",
        primary_pixels=render_row_frame(SECONDARY_ROW),
        secondary_pixels=render_row_frame(THIRD_ROW),
        primary_row_id=SECONDARY_ROW,
        secondary_row_id=THIRD_ROW,
        primary_action_id="RIGHT",
        secondary_action_id="FIRE",
    )
    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            kernels,
            "_canonical_collision_rows",
            lambda _pixels: [{"row_id": PRIMARY_ROW, "action_id": "FIRE"}],
        )
        expect("conflicting splice output collides with a canonical valid observation")
    with monkeypatch.context() as local_patch:
        primary_mask = np.zeros((16, 28), dtype=bool)
        primary_mask[2, 1] = True
        secondary_mask = np.zeros((16, 28), dtype=bool)
        secondary_mask[2, 2] = True
        masks = iter([primary_mask, secondary_mask])
        local_patch.setattr(kernels, "target_signal_mask", lambda _pixels: next(masks))
        expect(
            "splice requires nonzero effective contribution from both sources",
            primary_pixels=np.zeros((16, 28), dtype=np.uint8),
            secondary_pixels=np.zeros((16, 28), dtype=np.uint8),
        )


def test_critical_corruption_preserves_forged_manifest_identity() -> None:
    forged_digest = "sha256:" + "a" * 64
    forged = kernels.critical_coordinate_manifest() | {
        "version": "forged-critical-manifest/v0",
        "coordinate_set_digest": forged_digest,
    }

    corrupted, trace = kernels.apply_critical_corruption(
        render_row_frame(PRIMARY_ROW),
        forged,
    )

    assert array_digest(corrupted) == CRITICAL_DIGEST
    assert trace["criticality_artifact_identity"] == CRITICAL_COORDINATE_SET_VERSION
    assert trace["critical_coordinate_set_identity"] == forged_digest
    assert trace["critical_corruption_digest"] == (
        "sha256:09c05d930ea1eaf81554735e025cf722acd7f30fe013ac0011ab5ef0f050d4f6"
    )


def test_family_operation_chains_are_frozen() -> None:
    stale_plan = {
        "source_frame_index": 0,
        "destination_frame_index": 1,
        "maximum_stale_horizon": 1,
    }
    impossible_plan = {
        "source_frame_index": 0,
        "destination_frame_index": 1,
        "source_row_id": PRIMARY_ROW,
        "source_action_id": "FIRE",
        "destination_row_id": THIRD_ROW,
        "destination_action_id": "RIGHT",
    }
    chains = {
        "conflicting": provenance.conflicting_splice_operation_chain(
            primary_row_id=PRIMARY_ROW,
            secondary_row_id=SECONDARY_ROW,
            primary_action_id="FIRE",
            secondary_action_id="RIGHT",
            primary_transformation_parameters=EXACT_PARAMS,
            mask_manifest=kernels.splice_mask_manifest(),
        ),
        "critical": provenance.critical_corruption_operation_chain(
            PRIMARY_ROW,
            EXACT_PARAMS,
            kernels.critical_coordinate_manifest(),
        ),
        "stale": provenance.stale_repeat_operation_chain(
            _source_record(PRIMARY_ROW, PRIMARY_DIGEST),
            _source_record(SECONDARY_ROW, SECONDARY_DIGEST),
            stale_plan,
        ),
        "impossible": provenance.impossible_transition_operation_chain(
            _source_record(PRIMARY_ROW, PRIMARY_DIGEST),
            impossible_plan,
        ),
        "control": provenance.information_control_operation_chain(PRIMARY_ROW),
    }

    assert chains["conflicting"]["operation_chain_digest"] == (
        "sha256:ad903845d7542e69e0bee0fcca0fff929e6f7df4a2e6d7802dac095d5a3db999"
    )
    assert chains["critical"]["operation_chain_digest"] == (
        "sha256:1a2432d0540542bdd2cd204aa0f36d33f17852aaea32fb283716d2f83da2a5ff"
    )
    assert chains["stale"]["operation_chain_digest"] == (
        "sha256:1b176a60f36540c9d5c7e056c6f581141fd0d0eabd535d674dafcd75c7a045dd"
    )
    assert chains["impossible"]["operation_chain_digest"] == (
        "sha256:46a1f450f665910e5fbef778e34b94a865b8adc9105083b6f2d2a5f4870ee683"
    )
    assert chains["control"]["operation_chain_digest"] == (
        "sha256:fedc9cd54944fba25c6e0c4d93d30c024e4f49aa23da32e06dfef99bb54e8edb"
    )
    assert chains["conflicting"]["final_emitted_digest"] == SPLICE_DIGEST
    assert chains["critical"]["final_emitted_digest"] == CRITICAL_DIGEST
    assert chains["stale"]["final_emitted_digest"] == SECONDARY_DIGEST
    assert chains["impossible"]["final_emitted_digest"] == THIRD_DIGEST
    assert [op["operation_version"] for op in chains["conflicting"]["operations"]][
        3
    ] == CONFLICTING_ACTION_SPLICE_VERSION
    assert [op["operation_version"] for op in chains["critical"]["operations"]][
        2
    ] == CRITICAL_EVIDENCE_CORRUPTION_VERSION
    assert [op["operation_version"] for op in chains["stale"]["operations"]][
        4
    ] == STALE_REPEATED_FRAME_VERSION
    assert [op["operation_version"] for op in chains["impossible"]["operations"]][
        3
    ] == IMPOSSIBLE_TRANSITION_VERSION
    assert [op["operation_version"] for op in chains["control"]["operations"]][
        1
    ] == INFORMATION_CONTROL_VERSION


def test_replay_core_uses_real_extracted_family_executors() -> None:
    splice_replay = replay_observation_operation_chain(
        _conflicting_chain(),
        conflicting_splice_executor=kernels.apply_conflicting_splice,
        critical_corruption_executor=kernels.apply_critical_corruption,
    )
    critical_replay = replay_observation_operation_chain(
        _critical_chain(),
        conflicting_splice_executor=kernels.apply_conflicting_splice,
        critical_corruption_executor=kernels.apply_critical_corruption,
    )

    assert splice_replay["final_emitted_digest"] == SPLICE_DIGEST
    assert array_digest(splice_replay["pixels"]) == SPLICE_DIGEST
    assert critical_replay["final_emitted_digest"] == CRITICAL_DIGEST
    assert array_digest(critical_replay["pixels"]) == CRITICAL_DIGEST


def test_family_validation_statuses_and_closure_digest_are_frozen() -> None:
    spliced, splice_trace = _spliced()
    corrupted, critical_trace = _critical()
    splice_record = _frame_record(
        family="conflicting_action_splice",
        pixels=spliced,
        trace=splice_trace,
    )
    critical_record = _frame_record(
        family="critical_evidence_corruption",
        pixels=corrupted,
        trace=critical_trace,
    )

    assert validation.validate_materialized_family_record(splice_record) == "ok"
    assert (
        validation.validate_materialized_family_record(
            splice_record | {"observation_pixel_digest": CRITICAL_DIGEST}
        )
        == "stale_observation_digest"
    )
    assert (
        validation.validate_materialized_family_record(
            {
                "metadata": {
                    "family_intervention_trace": {
                        "output_observation_digest": CRITICAL_DIGEST,
                        "changed_pixel_count": 1,
                    }
                },
                "observation_pixel_digest": SPLICE_DIGEST,
            }
        )
        == "family_output_digest_mismatch"
    )
    assert (
        validation.validate_materialized_family_record(
            {
                "metadata": {"family_intervention_trace": {"changed_pixel_count": 0}},
                "observation_pixel_digest": SPLICE_DIGEST,
            }
        )
        == "family_no_op"
    )
    assert (
        validation.validate_materialized_family_record(
            {
                "expected_disposition": "distinguishable_invalid_input",
                "observation_pixel_digest": PRIMARY_DIGEST,
                "pixels": render_row_frame(PRIMARY_ROW),
            }
        )
        == "invalid_family_valid_state_collision"
    )
    assert (
        validation.validate_materialized_family_record(
            {
                "event_type": "gap_unknown",
                "observation_pixel_digest": PRIMARY_DIGEST,
                "pixels": render_row_frame(PRIMARY_ROW),
            }
        )
        == "gap_event_has_pixels"
    )

    closure = validation.frame_invalid_closure_summary([splice_record, critical_record])
    assert closure["version"] == FRAME_INVALID_CLOSURE_VERSION
    assert closure["status"] == "passed"
    assert closure["totals"] == {
        "frame_count": 2,
        "canonical_collision_count": 0,
        "valid_decode_count": 0,
        "no_op_count": 0,
        "malformed_count": 0,
    }
    assert closure["closure_digest"] == (
        "sha256:9c3feb21a81ee14b5206faf4a999dc650162ffc3c337b4995eba74fe52075766"
    )


@pytest.mark.parametrize("metadata", [None, [], "malformed"])
def test_family_validation_preserves_non_mapping_metadata_failure(
    metadata: object,
) -> None:
    with pytest.raises(AttributeError):
        validation.validate_materialized_family_record({"metadata": metadata})


def test_family_validation_status_precedence_is_frozen() -> None:
    pixels = render_row_frame(PRIMARY_ROW)
    collision_record = {
        "expected_disposition": "distinguishable_invalid_input",
        "event_type": "gap_unknown",
        "observation_pixel_digest": PRIMARY_DIGEST,
        "pixels": pixels,
    }

    assert (
        validation.validate_materialized_family_record(
            collision_record
            | {
                "observation_pixel_digest": SPLICE_DIGEST,
                "metadata": {
                    "family_intervention_trace": {
                        "output_observation_digest": CRITICAL_DIGEST,
                        "changed_pixel_count": 0,
                    }
                },
            }
        )
        == "stale_observation_digest"
    )
    assert (
        validation.validate_materialized_family_record(
            {
                "metadata": {
                    "family_intervention_trace": {
                        "output_observation_digest": CRITICAL_DIGEST,
                        "changed_pixel_count": 0,
                    }
                },
                "observation_pixel_digest": SPLICE_DIGEST,
            }
        )
        == "family_output_digest_mismatch"
    )
    assert (
        validation.validate_materialized_family_record(
            collision_record
            | {
                "metadata": {
                    "family_intervention_trace": {
                        "output_observation_digest": PRIMARY_DIGEST,
                        "changed_pixel_count": 0,
                    }
                }
            }
        )
        == "family_no_op"
    )
    assert (
        validation.validate_materialized_family_record(collision_record)
        == "invalid_family_valid_state_collision"
    )


def test_legacy_benchmark_exposes_direct_family_kernel_aliases() -> None:
    assert benchmark._target_signal_coordinates is kernels.target_signal_coordinates
    assert benchmark._target_signal_mask is kernels.target_signal_mask
    assert benchmark._splice_evidence_counts is kernels.splice_evidence_counts
    assert (
        benchmark._target_slot_signal_coordinates
        is kernels.target_slot_signal_coordinates
    )
    assert benchmark._detect_visible_target_slots is kernels.detect_visible_target_slots
    assert benchmark._detect_tank_slot is kernels.detect_tank_slot
    assert benchmark._detect_cooldown_state is kernels.detect_cooldown_state
    assert (
        benchmark._final_visible_target_action_evidence
        is kernels.final_visible_target_action_evidence
    )
    assert benchmark._critical_coordinates is kernels.critical_coordinates
    assert (
        benchmark._critical_coordinate_manifest is kernels.critical_coordinate_manifest
    )
    assert benchmark._splice_mask_manifest is kernels.splice_mask_manifest
    assert benchmark._apply_conflicting_splice is kernels.apply_conflicting_splice
    assert benchmark._apply_critical_corruption is kernels.apply_critical_corruption
    assert (
        benchmark._conflicting_splice_operation_chain
        is provenance.conflicting_splice_operation_chain
    )
    assert (
        benchmark._critical_corruption_operation_chain
        is provenance.critical_corruption_operation_chain
    )
    assert benchmark._stale_repeat_operation_chain is (
        provenance.stale_repeat_operation_chain
    )
    assert (
        benchmark._impossible_transition_operation_chain
        is provenance.impossible_transition_operation_chain
    )
    assert (
        benchmark._information_control_operation_chain
        is provenance.information_control_operation_chain
    )
    assert benchmark.validate_materialized_family_record is (
        validation.validate_materialized_family_record
    )
    assert benchmark._frame_invalid_closure_summary is (
        validation.frame_invalid_closure_summary
    )
