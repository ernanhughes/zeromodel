import numpy as np
from PIL import Image
import pytest

from examples.hand_anatomy._common import UnsupportedInput
from examples.hand_anatomy.observations import load_observation, observation_from_arrays


def _blob(h=300, w=300, box=(80, 60, 220, 200)):
    m = np.zeros((h, w), dtype=bool)
    m[box[0]:box[2], box[1]:box[3]] = True
    return m


def test_valid_mask_is_identified_and_original_preserved():
    m = _blob()
    o = observation_from_arrays(m)
    assert o.observation_id == observation_from_arrays(m.copy()).observation_id
    assert np.array_equal(o.mask, m) and o.work_mask.shape == (192, 192)
    assert "handedness" in o.unknowns
    moved = observation_from_arrays(np.roll(m, 5, axis=1))
    assert moved.observation_id != o.observation_id
    assert moved.work_mask_id == o.work_mask_id  # normalization is translation-invariant


@pytest.mark.parametrize(
    "mask",
    [np.zeros((200, 200), bool), _blob(box=(10, 10, 15, 15)), np.ones((200, 200), bool)],
)
def test_empty_tiny_or_full_masks_rejected(mask):
    with pytest.raises(UnsupportedInput):
        observation_from_arrays(mask)


def test_oversize_and_bad_shapes_rejected():
    with pytest.raises(UnsupportedInput):
        observation_from_arrays(np.zeros((3000, 10), bool))
    with pytest.raises(UnsupportedInput):
        observation_from_arrays(np.zeros((5, 5, 3), bool))


def test_file_loading_validations(tmp_path):
    m = _blob()
    Image.fromarray((m * 255).astype(np.uint8)).save(tmp_path / "mask.png")
    Image.fromarray(np.zeros((300, 300, 3), np.uint8) + 20).save(tmp_path / "img.png")
    o = load_observation(tmp_path / "img.png", tmp_path / "mask.png")
    assert o.image_digest.startswith("sha256:") and o.method["mask_source"] == "supplied-mask"
    Image.fromarray(np.zeros((100, 100, 3), np.uint8)).save(tmp_path / "small.png")
    with pytest.raises(UnsupportedInput):
        load_observation(tmp_path / "small.png", tmp_path / "mask.png")  # dimension mismatch
    with pytest.raises(UnsupportedInput):
        load_observation(tmp_path / "img.png")  # no mask, no alpha, no segmentation
    with pytest.raises(UnsupportedInput):
        load_observation(tmp_path / "missing.png", tmp_path / "mask.png")
    (tmp_path / "junk.png").write_bytes(b"not an image")
    with pytest.raises(UnsupportedInput):
        load_observation(tmp_path / "junk.png", tmp_path / "mask.png")
    with pytest.raises(UnsupportedInput):
        load_observation()


def test_alpha_and_background_segmentation(tmp_path):
    m = _blob()
    rgba = np.zeros((300, 300, 4), np.uint8)
    rgba[..., :3] = 200
    rgba[..., 3] = m * 255
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    assert load_observation(tmp_path / "a.png").method["mask_source"] == "alpha-channel"
    rgb = np.full((300, 300, 3), 240, np.uint8)
    rgb[m] = (120, 60, 50)
    Image.fromarray(rgb).save(tmp_path / "b.png")
    seg = load_observation(tmp_path / "b.png", segment_background=True)
    assert seg.method["mask_source"] == "uniform-background-threshold" and seg.area == m.sum()
    noisy = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    Image.fromarray(noisy).save(tmp_path / "c.png")
    with pytest.raises(UnsupportedInput):
        load_observation(tmp_path / "c.png", segment_background=True)  # non-uniform background


def test_landmarks_validated():
    m = _blob()
    with pytest.raises(UnsupportedInput):
        observation_from_arrays(m, landmarks=np.full((21, 2), np.nan))
    with pytest.raises(UnsupportedInput):
        observation_from_arrays(m, landmarks=np.zeros((5, 2)))
