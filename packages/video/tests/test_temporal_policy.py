from __future__ import annotations

import numpy as np
import pytest

from zeromodel.core import LayoutRecipe, ScoreTable, build_vpm
from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.core.policy_transitions import PolicyTransitionSpec
from zeromodel.observation import (
    ImageObservation,
    VisualAddressContract,
    VisualAddressDecision,
)
from zeromodel.video import InMemoryVideoFrameSource, VideoPolicyReader


def _policy():
    table = ScoreTable(
        values=[[1.0, 0.0], [0.0, 1.0]],
        row_ids=["left", "right"],
        metric_ids=["A", "B"],
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "video-policy",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(table, recipe)


class FakeProvider:
    def __init__(self, policy_id: str, rows: tuple[str | None, ...]) -> None:
        self.rows = rows
        self.index = 0
        self._contract = VisualAddressContract(
            provider_kind="fake",
            provider_version="fake/v1",
            score_semantics="distance",
            observation_spec_digest="obs",
            representation_spec_digest="repr",
            address_artifact_id="addr",
            calibration_artifact_id="cal",
            policy_artifact_id=policy_id,
        )

    def contract(self) -> VisualAddressContract:
        return self._contract

    def read(self, observation: ImageObservation) -> VisualAddressDecision:
        row = self.rows[self.index]
        self.index += 1
        accepted = row is not None
        return VisualAddressDecision(
            accepted=accepted,
            reason="accepted" if accepted else "provider_rejected",
            observation_digest=observation.raw_digest,
            representation_digest=observation.raw_digest,
            provider_kind=self._contract.provider_kind,
            provider_version=self._contract.provider_version,
            score_semantics=self._contract.score_semantics,
            address_artifact_id=self._contract.address_artifact_id,
            calibration_artifact_id=self._contract.calibration_artifact_id,
            policy_artifact_id=self._contract.policy_artifact_id,
            nearest_row_id=row,
            nearest_score=0.0 if accepted else None,
            second_row_id=None,
            second_score=None,
            ambiguity_measure=None,
            matched_row_id=row,
            exact_match=accepted,
            accepted_by=("fake",) if accepted else (),
        )


def _reader(rows: tuple[str | None, ...]) -> VideoPolicyReader:
    artifact = _policy()
    return VideoPolicyReader(
        FakeProvider(artifact.artifact_id, rows),
        VPMPolicyLookup(artifact, action_metric_ids=("A", "B")),
        PolicyTransitionSpec(
            {"left": ("left", "right"), "right": ("right",)},
            maximum_frame_gap=1,
        ),
    )


def test_temporal_policy_uses_provider_protocol_without_vision() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [
            np.zeros((1, 1), dtype=np.uint8),
            np.ones((1, 1), dtype=np.uint8),
            np.full((1, 1), 2, dtype=np.uint8),
        ],
        clip_id="clip-a",
        nominal_fps=1.0,
    )

    trace = _reader(("left", "right", "right")).read(source)

    assert [decision.accepted_action_id for decision in trace.decisions] == [
        "A",
        "B",
        "B",
    ]
    assert [decision.temporal.transition.status for decision in trace.decisions] == [
        "initial",
        "possible",
        "possible",
    ]
    assert trace.accepted_count == 3
    assert trace.trace_id == trace.trace_id


def test_temporal_policy_rejects_provider_and_unknown_policy_rows() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [np.zeros((1, 1), dtype=np.uint8)],
        clip_id="clip-a",
        nominal_fps=1.0,
    )
    rejected = _reader((None,)).read(source).decisions[0]
    assert rejected.accepted is False
    assert rejected.rejection_reasons == ("provider_rejected",)
    assert rejected.policy is None

    with pytest.raises(VPMValidationError, match="Unknown policy row_id"):
        _reader(("missing",)).read(source)


def test_temporal_policy_rejects_policy_manifest_mismatch() -> None:
    artifact = _policy()
    with pytest.raises(VPMValidationError, match="different policy"):
        VideoPolicyReader(
            FakeProvider("other-policy", ("left",)),
            VPMPolicyLookup(artifact, action_metric_ids=("A", "B")),
            PolicyTransitionSpec({"left": ("left",), "right": ("right",)}),
        )
