import numpy as np
from PIL import Image, ImageDraw

from examples.hand_anatomy.pose_library import SHAPES, make_pose
from examples.hand_anatomy.template_renderer import render_template
from examples.hand_anatomy_02.residual import ResidualRecipe, analyze_residual
from examples.hand_anatomy_02.topology import digit_branches, thin


def _star(fingers):
    img = Image.new("L", (192, 192), 0)
    d = ImageDraw.Draw(img)
    d.ellipse([66, 66, 126, 126], fill=255)
    for ang in fingers:
        a = np.radians(ang)
        d.line([96, 96, 96 + 80 * np.sin(a), 96 - 80 * np.cos(a)], fill=255, width=9)
    return np.asarray(img) > 127


def test_thinning_preserves_connectivity_and_thins():
    m = _star([0, 72, 144, 216, 288])
    skel = thin(m)
    assert 0 < skel.sum() < 0.1 * m.sum()
    assert not (skel & ~m).any()


def test_branch_count_follows_digit_count_when_digits_are_separated():
    assert digit_branches(_star([-60, -30, 0, 30, 60]))["count"] == 5
    assert digit_branches(_star([-60, -30, 0, 30, 60, 150]))["count"] == 6


def test_merged_fingers_undercount_a_declared_limitation():
    assert digit_branches(_star([0, 3, 6]))["count"] < 3


def _setup():
    tpl = render_template(make_pose("open", "right", SHAPES["standard"]))
    mask = tpl.mask
    labels = {c: tpl.labels == c for c in range(1, 8) if (tpl.labels == c).any()}
    return tpl, mask, labels


def test_attached_elongated_residual_is_finger_like_and_sliver_is_not():
    _, mask, labels = _setup()
    ys, xs = np.nonzero(mask)
    obs = mask.copy()
    obs[ys.min() + 20: ys.min() + 75, xs.min() - 9: xs.min() + 2] = True  # strip attached to the left edge
    res = analyze_residual(obs, mask, labels)
    top = res["components"][0]
    assert top["attached"] and res["max_finger_likeness"] > 0.5
    far = mask.copy()
    far[5:40, 150:156] = True  # unattached bar far from the hand
    res2 = analyze_residual(far, mask, labels)
    assert res2["max_finger_likeness"] == 0.0
    assert analyze_residual(mask, mask, labels)["max_finger_likeness"] == 0.0


def test_digit_coverage_counts_missing_digits():
    _, mask, labels = _setup()
    full = analyze_residual(mask, mask, labels)
    assert full["digit_coverage"] == 1.0 and full["digits_present"] == 5
    obs = mask.copy()
    obs[labels[3]] = False  # remove the thumb component
    assert analyze_residual(obs, mask, labels)["digits_explained"] == 4


def test_recipe_identity_changes_with_constants():
    assert ResidualRecipe().recipe_id != ResidualRecipe(length_ref=30.0).recipe_id
