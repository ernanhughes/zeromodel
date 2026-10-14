import dataclasses

import numpy as np
import pytest

from examples.hand_anatomy._common import HandAnatomyError
from examples.hand_anatomy.geometry import (
    FINGERS,
    STANDARD_TOPOLOGY,
    Camera,
    HandTopology,
    build_skeleton,
    landmark_segment_ratios,
    pose_violations,
    validate_camera,
    validate_pose,
)
from examples.hand_anatomy.pose_library import SHAPES, make_pose


def test_valid_finger_topology():
    STANDARD_TOPOLOGY.validate()
    assert tuple(STANDARD_TOPOLOGY.chains) == FINGERS and len(FINGERS) == 5


def test_invalid_topology_rejected():
    chains = dict(STANDARD_TOPOLOGY.chains)
    chains["sixth"] = ("mcp", "pip", "dip", "tip")
    with pytest.raises(HandAnatomyError):
        HandTopology(chains=chains).validate()
    with pytest.raises(HandAnatomyError):
        HandTopology(attachment={f: "wrist" for f in FINGERS}).validate()


def test_joint_constraints_enforced():
    pose = make_pose("open", "right", SHAPES["standard"])
    bad = dataclasses.replace(pose, flex={**pose.flex, "index": (120.0, 0.0, 0.0)})
    assert any("index.mcp" in p for p in pose_violations(bad))
    with pytest.raises(HandAnatomyError):
        validate_pose(bad)
    with pytest.raises(HandAnatomyError):
        validate_camera(Camera(yaw=120))
    nan = dataclasses.replace(pose, thumb_opposition=float("nan"))
    assert pose_violations(nan)


def test_left_right_are_mirrors():
    r = build_skeleton(make_pose("relaxed", "right", SHAPES["standard"])).landmarks
    left = build_skeleton(make_pose("relaxed", "left", SHAPES["standard"])).landmarks
    assert np.allclose(left, r * np.array([-1, 1, 1]))
    assert r[1, 0] > 0  # right-hand thumb base is on +x in the palm-facing view


def test_pose_identity_deterministic_and_sensitive():
    a = make_pose("fist", "right", SHAPES["standard"])
    assert a.geometry_id == make_pose("fist", "right", SHAPES["standard"]).geometry_id
    assert a.geometry_id != make_pose("fist", "left", SHAPES["standard"]).geometry_id
    assert a.geometry_id != make_pose("open", "right", SHAPES["standard"]).geometry_id


def test_landmark_ratios_and_degenerate_rejected():
    lm = build_skeleton(make_pose("open", "right", SHAPES["standard"])).landmarks
    assert landmark_segment_ratios(lm)["middle_seg0"] == pytest.approx(0.44, abs=0.05)
    with pytest.raises(HandAnatomyError):
        landmark_segment_ratios(np.zeros((21, 3)))
