from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest

import research.benchmarks.video_action_set_benchmark as benchmark
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set import materialization_kernels as kernels


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RECORD_KEYS = [
    "benchmark_version",
    "generator_version",
    "split",
    "episode_id",
    "clip_id",
    "frame_id",
    "sequence_number",
    "event_type",
    "family",
    "expected_disposition",
    "episode_family",
    "episode_disposition",
    "frame_disposition",
    "denominator_class",
    "expected_row",
    "expected_action",
    "actual_executed_action",
    "action_known",
    "gap_declaration",
    "observation_pixel_digest",
    "metadata",
]


def _first_frame_plan():
    identity = benchmark.load_identity(REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    plan = benchmark._episode_plans_for_split(
        identity, "selection", row_ids, row_actions
    )[0]
    row = str(plan["source_row_id"])
    tank, target, cooldown = benchmark.parse_state_row_id(row)
    base = benchmark.render_state_frame(
        tank, target, cooldown, width=benchmark.ShooterConfig().width
    )
    return row, base, plan["frame_plans"][0]


def test_apply_frame_plan_digest_and_trace_are_frozen() -> None:
    _row, base, frame_plan = _first_frame_plan()

    pixels, trace = kernels.apply_frame_plan(base, frame_plan)

    assert benchmark._array_digest(pixels) == (
        "sha256:424d87c22e58789875eb09d1c430d6e1cb7a45e1de1cd6f32b19400cad0bb8da"
    )
    assert trace == {
        "source_observation_digest": "sha256:76d6b55ab164806fa6d621e669de37b60b9cbd8f8fff2bbb610b6b0e034461f0",
        "transformed_observation_digest": "sha256:424d87c22e58789875eb09d1c430d6e1cb7a45e1de1cd6f32b19400cad0bb8da",
        "transformation_parameter_digest": "sha256:fd10a0e5b42d0abfed681a2b8a97dbd11a20c64e26c01291ebc96110f25a5a1f",
        "changed_pixel_count": 439,
    }


def test_frame_descriptor_preserves_top_level_record_contract() -> None:
    row, base, frame_plan = _first_frame_plan()
    pixels, _trace = kernels.apply_frame_plan(base, frame_plan)

    kwargs = dict(
        split="selection",
        episode_id="probe",
        frame_index=0,
        row_id=row,
        expected_action="STAY",
        actual_action="STAY",
        family=str(frame_plan["transformation_family"]),
        pixels=pixels,
        expected_disposition="valid",
        episode_family="valid",
        episode_disposition="valid",
        frame_disposition="valid_frame_payload",
        denominator_class="positive_valid_policy_action",
        metadata={"event_type": "frame", "probe": True},
    )
    record = kernels.frame_descriptor(**kwargs)

    assert list(record.keys()) == EXPECTED_RECORD_KEYS
    assert record["observation_pixel_digest"] == (
        "sha256:424d87c22e58789875eb09d1c430d6e1cb7a45e1de1cd6f32b19400cad0bb8da"
    )
    assert record["metadata"]["provider_observation_digest"] == (
        "sha256:80ae23c1cc7de1d60dc9e735544ec972fd5badb193e5f1ae7b0b1e870d9ca3cd"
    )


def test_frame_descriptor_signature_and_binding_contract_are_preserved() -> None:
    row, base, frame_plan = _first_frame_plan()
    pixels, _trace = kernels.apply_frame_plan(base, frame_plan)
    kwargs = dict(
        split="selection",
        episode_id="probe",
        frame_index=0,
        row_id=row,
        expected_action="STAY",
        actual_action="STAY",
        family=str(frame_plan["transformation_family"]),
        pixels=pixels,
        expected_disposition="valid",
        episode_family="valid",
        episode_disposition="valid",
        frame_disposition="valid_frame_payload",
        denominator_class="positive_valid_policy_action",
        metadata={"event_type": "frame", "probe": True},
    )

    assert inspect.signature(benchmark._frame_descriptor) == inspect.signature(
        kernels.frame_descriptor
    )
    signature = inspect.signature(kernels.frame_descriptor)
    assert list(signature.parameters) == [
        "split",
        "episode_id",
        "frame_index",
        "row_id",
        "expected_action",
        "actual_action",
        "family",
        "pixels",
        "expected_disposition",
        "episode_family",
        "episode_disposition",
        "frame_disposition",
        "denominator_class",
        "metadata",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    with pytest.raises(TypeError, match="required keyword-only argument"):
        kernels.frame_descriptor(
            **{key: value for key, value in kwargs.items() if key != "split"}
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        kernels.frame_descriptor(**(kwargs | {"unexpected": True}))


def test_frame_plan_digest_mismatch_is_rejected_before_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _row, base, frame_plan = _first_frame_plan()
    tampered = deepcopy(frame_plan)
    tampered["transformation_parameter_digest"] = "sha256:" + "0" * 64

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("transformation executor must not be called")

    monkeypatch.setattr(kernels, "_apply_transformation", fail_if_called)

    with pytest.raises(VPMValidationError, match="parameter digest mismatch"):
        kernels.apply_frame_plan(base, tampered)


def test_materialization_kernel_aliases_are_direct() -> None:
    assert benchmark._apply_family is kernels.apply_family
    assert benchmark._apply_frame_plan is kernels.apply_frame_plan
    assert benchmark._frame_descriptor is kernels.frame_descriptor
