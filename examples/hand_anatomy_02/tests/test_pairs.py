import numpy as np
import pytest

from examples.hand_anatomy_02.pairs import (
    INDISTINGUISHABLE_DELTA,
    PAIR_PLAN,
    PRIMARY_SUBTYPES,
    build_dataset,
    make_pair,
    place_fixed,
)


@pytest.fixture(scope="module")
def ds():
    return build_dataset()


def test_pair_construction_is_deterministic():
    a, b = make_pair("cal02", "extra_finger", "outer", 0), make_pair("cal02", "extra_finger", "outer", 0)
    assert a.pair_id == b.pair_id and np.array_equal(a.defect, b.defect) and np.array_equal(a.valid, b.valid)
    assert make_pair("cal02", "extra_finger", "outer", 1).pair_id != a.pair_id


def test_twins_share_pose_and_placement_and_differ_only_by_the_defect():
    p = make_pair("test02", "extra_finger", "outer", 0)
    assert (p.valid & ~p.defect).sum() < 0.03 * p.valid.sum()  # nothing removed
    assert (p.defect & ~p.valid).sum() > 0.04 * p.valid.sum()  # one extra finger added
    m = make_pair("test02", "missing_finger", "any", 0)
    assert (m.defect & ~m.valid).sum() < 0.03 * m.valid.sum() < (m.valid & ~m.defect).sum()


def test_indistinguishable_is_computed_from_masks_only():
    for p in (make_pair("cal02", "extra_finger", s, i) for s in ("hidden", "adjacent") for i in range(4)):
        delta = float((p.valid ^ p.defect).sum()) / float(p.valid.sum())
        assert p.indistinguishable == (delta < INDISTINGUISHABLE_DELTA)
    assert all(make_pair("cal02", "extra_finger", "hidden", i).indistinguishable for i in range(4))


def test_place_fixed_is_rigid_and_identical_for_both_members():
    hand = np.zeros((420, 420), bool)
    hand[100:200, 100:150] = True
    a = place_fixed(hand, 12.0, 1.0, 5, 7)
    assert np.array_equal(a, place_fixed(hand, 12.0, 1.0, 5, 7))
    assert abs(int(a.sum()) - int(hand.sum())) < 0.05 * hand.sum()


def test_splits_disjoint_and_primary_set_large_enough(ds):
    cal, test = ds["cal02"]["cases"], ds["test02"]["cases"]
    assert not {c.case_id for c in cal} & {c.case_id for c in test}
    assert not {c.mask_id for c in cal} & {c.mask_id for c in test}
    primary = [p for p in ds["test02"]["pairs"] if p.category == "extra_finger"
               and p.subtype in PRIMARY_SUBTYPES and not p.indistinguishable]
    assert len(primary) >= 40
    assert len(ds["test02"]["pairs"]) == sum(v[1] for v in PAIR_PLAN.values())
    roles = {c.role for c in test}
    assert roles == {"valid_twin", "defect", "valid_pool"}
    assert all(c.label == ("malformed" if c.role == "defect" else "valid") for c in test)


def test_new_seed_space_differs_from_hand_anatomy_01():
    from examples.hand_anatomy.synthetic import build_cases

    old = {c.mask_id for v in build_cases().values() for c in v}
    new = {c.mask_id for v in build_dataset().values() for c in v["cases"]}
    assert not old & new
