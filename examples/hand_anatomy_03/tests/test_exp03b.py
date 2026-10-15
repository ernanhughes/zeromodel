import json
from pathlib import Path

import numpy as np
import pytest

from examples.hand_anatomy_03 import exp03b, mask_tool

pytestmark = pytest.mark.research
DATA = Path("examples/hand_anatomy_03/data")


def test_hull_solidity_and_auc_helpers():
    m = np.zeros((60, 60), bool)
    m[10:50, 10:20] = True
    m[40:50, 10:50] = True  # L shape
    h = exp03b.hull_mask(m)
    assert h.sum() > m.sum() and (h | m).sum() == h.sum()
    assert 0.5 < exp03b.solidity(m) < 1.0
    assert exp03b.solidity(h) > 0.97
    assert exp03b.auc([3, 4], [1, 2]) == 1.0 and exp03b.auc([1], [1]) == 0.5
    pos, neg = [0.9, 0.8, 0.7, 0.4], [0.6, 0.5, 0.3, 0.2]
    assert exp03b.boot_ci(pos, neg) == exp03b.boot_ci(pos, neg)  # fixed seed


def test_mask_tool_add_and_subtract(tmp_path):
    spec = {"polygons": [{"op": "add", "points": [[10, 10], [90, 10], [90, 90], [10, 90]]},
                         {"op": "sub", "points": [[40, 40], [60, 40], [60, 60], [40, 60]]}]}
    mask = mask_tool.build_mask((100, 100), spec)
    assert mask[20, 20] and not mask[50, 50] and not mask[5, 5]
    with pytest.raises(ValueError):
        mask_tool.build_mask((100, 100), {"polygons": [{"op": "add", "points": [[0, 0], [1, 1]]}]})
    with pytest.raises(ValueError):
        mask_tool.build_mask((100, 100), {"polygons": [{"op": "xor", "points": [[0, 0], [9, 0], [9, 9]]}]})


def test_screening_and_sample_files_are_consistent_with_the_protocol():
    sc = json.loads((DATA / "screening_03b.json").read_text())
    decided = len(sc["eligible"]) + len(sc["excluded"]) + len(sc["pool2"]["eligible"]) + len(sc["pool2"]["excluded"]) \
        + len(sc["pool3"]["eligible"]) + len(sc["pool3"]["excluded"])
    assert decided == 168 and len(sc["eligible"]) + len(sc["pool2"]["eligible"]) + len(sc["pool3"]["eligible"]) == 50
    sample = json.loads((DATA / "sample_03b.json").read_text())["images"]
    labels = json.loads((DATA / "sample_03b_labels.json").read_text())
    assert len(sample) == 28 and sum(labels.values()) == 14
    assert len({s["video"] for s in sample}) == 28  # one crop per source video
    assert all("label" not in s for s in sample)  # labels are kept out of the sample file


def test_model_review_record_and_prereg_threshold():
    rev = json.loads((DATA / "model_review_03b.json").read_text())
    assert len(rev) == 28 and {v["verdict"] for v in rev.values()} <= {"accept", "needs_fix", "reject"}
    accepted = sum(1 for v in rev.values() if v["verdict"] == "accept")
    assert accepted < exp03b.MIN_ACCEPTED  # the preregistered suitability threshold is not met
    assert sum(1 for v in rev.values() if v["verdict"] != "reject") < exp03b.MIN_ACCEPTED  # not reachable by a fix round
    text = Path("docs/research/ZM-HAND-03b-preregistration.md").read_text(encoding="utf-8")
    assert "at least 0.05" in text and "at least 0.03" in text and "fewer than 20" in text
    assert exp03b.G1_BLOB_MARGIN == 0.05 and exp03b.G1_HULL_MARGIN == 0.03 and exp03b.G2_MIN_AUC == 0.70


def test_human_review_template_covers_every_mask():
    t = json.loads((DATA / "human_review_template_03b.json").read_text())
    assert len(t["masks"]) == 28 and all(v["verdict"] == "" for v in t["masks"].values())
