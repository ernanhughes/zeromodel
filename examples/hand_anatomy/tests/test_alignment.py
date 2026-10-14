import numpy as np
import pytest

from examples.hand_anatomy._common import HandAnatomyError, UnsupportedInput
from examples.hand_anatomy.alignment import AlignmentRecipe, _best, align
from examples.hand_anatomy.observations import observation_from_arrays

from .conftest import observation_from_template


def _tpl(lib, name):
    return next(t for t in lib.templates if t.name == name)


@pytest.fixture(scope="module")
def target(small_lib):
    return _tpl(small_lib, "right/open/standard/yaw0")


def test_translation_is_recovered_and_search_is_deterministic(small_lib, target):
    a = align(observation_from_template(target, shift=(10, 50)), small_lib)
    b = align(observation_from_template(target, shift=(200, 120)), small_lib)
    assert a[0].template_id == target.template_id and a[0].iou_post > 0.95
    # work-frame alignment is translation-invariant: identical result for shifted input
    assert (a[0].rotation_deg, a[0].residual_scale, a[0].iou_post) == (
        b[0].rotation_deg, b[0].residual_scale, b[0].iou_post)
    again = align(observation_from_template(target, shift=(10, 50)), small_lib)
    assert [x.params() for x in again] == [x.params() for x in a]


def test_scale_is_recovered(small_lib, target):
    for s in (0.6, 1.0, 1.7):
        res = align(observation_from_template(target, scale=s, canvas=520), small_lib)
        assert res[0].template_id == target.template_id and res[0].iou_post > 0.93


def test_in_plane_rotation_recovered_within_tolerance(small_lib, target):
    res = align(observation_from_template(target, rot=20.0), small_lib)
    assert res[0].template_id == target.template_id
    assert abs(abs(res[0].rotation_deg) - 20.0) <= 3.0 and res[0].iou_post > 0.92


def test_search_is_bounded_rotation_outside_range_fits_poorly(small_lib, target):
    narrow = AlignmentRecipe(rotation_range_deg=15.0)
    res = align(observation_from_template(target, rot=70.0), small_lib, narrow)
    assert abs(res[0].rotation_deg) <= 15.0 + 3.0 + 1e-9
    assert res[0].iou_post < 0.7


def test_pre_and_post_alignment_scores_kept_separately(small_lib, target):
    res = align(observation_from_template(target, rot=20.0), small_lib)[0]
    assert res.iou_pre < res.iou_post


def test_tie_break_prefers_smaller_transform():
    stack = np.zeros((2, 8, 8), dtype=np.float32)
    stack[:, 2:6, 2:6] = 1.0  # identical masks for both combos
    obs = np.zeros((8, 8), dtype=bool)
    obs[2:6, 2:6] = True
    from examples.hand_anatomy.alignment import _shift_grid, _windows

    shifts = _shift_grid(1)
    (rot, res, iou, sx, sy), _ = _best(
        stack, stack.reshape(2, -1).sum(1), 16.0, _windows(obs, 1),
        [(5.0, 1.065), (0.0, 1.0)], shifts)
    assert (rot, res, sx, sy) == (0.0, 1.0, 0, 0) and iou == 1.0


def test_zero_area_and_invalid_inputs():
    with pytest.raises(UnsupportedInput):
        observation_from_arrays(np.zeros((100, 100), bool))
    with pytest.raises(HandAnatomyError):
        AlignmentRecipe(rotation_range_deg=0.0).validate()
    with pytest.raises(HandAnatomyError):
        AlignmentRecipe(shortlist_size=0).validate()


def test_recipe_identity_changes_with_parameters():
    assert AlignmentRecipe().recipe_id != AlignmentRecipe(rotation_range_deg=30.0).recipe_id
    assert AlignmentRecipe.from_dict(AlignmentRecipe().to_dict()) == AlignmentRecipe()
