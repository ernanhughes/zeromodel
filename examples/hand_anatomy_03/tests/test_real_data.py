import json
from pathlib import Path

import numpy as np
import pytest

from examples.hand_anatomy_03 import vbench_data as vd

pytestmark = pytest.mark.research


def _rows(n_videos=400, per_video=3):
    rows = []
    for v in range(n_videos):
        for k in range(per_video):
            rows.append(["%032x_%d_hand_0.jpg" % (v, k), (v + k) % 2, 1.0])
    return rows


def test_video_id_and_partition_are_deterministic_and_video_level():
    rows = _rows()
    vids = [vd.video_id(r[0]) for r in rows]
    a, b = vd.partition_videos(vids), vd.partition_videos(list(reversed(vids)))
    assert a == b and len(a) == 400
    assert {v: a[v] for v in a}.keys() == set(vids)
    counts = {p: sum(1 for x in a.values() if x == p) for p in ("development", "unused", "heldout")}
    assert counts["development"] == 20 and counts["heldout"] == 200


def test_selection_is_balanced_one_crop_per_video_and_pool_disjoint():
    rows = _rows(2000)
    part = vd.partition_videos([vd.video_id(r[0]) for r in rows])
    dev = vd.select_crops(rows, part, "development", 20)
    held = vd.select_crops(rows, part, "heldout", 30)
    assert len(dev) == 40 and sum(d["label"] for d in dev) == 20
    assert len({d["video"] for d in dev}) == 40  # one crop per source video
    assert not {d["video"] for d in dev} & {h["video"] for h in held}
    assert dev == vd.select_crops(rows, part, "development", 20)


def test_split_integrity_flags_overlapping_videos():
    train = [["aaa_1_hand_0.jpg", 0, 1], ["bbb_1_hand_0.jpg", 1, 1]]
    test = [["aaa_2_hand_0.jpg", 0, 1], ["ccc_1_hand_0.jpg", 1, 1]]
    info = vd.split_integrity(train, test)
    assert info["video_disjoint"] is False and info["test_videos_also_in_train"] == 1
    assert info["test_images_whose_video_is_in_train"] == 1
    assert vd.split_integrity(train[1:], test[1:])["video_disjoint"] is True


def test_unsafe_names_are_rejected_before_any_network_use(tmp_path):
    with pytest.raises(ValueError):
        vd.fetch_images(["../evil.jpg"], {}, tmp_path)


def test_committed_dev_selection_matches_its_declared_rule():
    sel = json.loads(Path("examples/hand_anatomy_03/data/vbench_dev_selection.json").read_text())
    imgs = sel["images"]
    assert len(imgs) == 40 and sum(i["label"] for i in imgs) == 20
    assert len({i["video"] for i in imgs}) == 40 and sel["salt"] == vd.SPLIT_SALT


def test_forearm_free_variant_removes_only_the_forearm():
    from examples.hand_anatomy_03.real_pipeline import forearm_free_library
    from examples.hand_anatomy.pose_library import build_library

    base, free = build_library(), forearm_free_library()
    assert len(base) == len(free) and base.library_id != free.library_id
    assert all(not (t.labels == 2).any() for t in free.templates)
    t0 = base.templates[0]
    twin = next(t for t in free.templates if t.name == t0.name + "/no-forearm")
    assert np.array_equal(twin.mask, t0.mask & (t0.labels != 2))
    assert twin.template_id != t0.template_id
