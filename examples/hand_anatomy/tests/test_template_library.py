import pytest

from examples.hand_anatomy._common import HandAnatomyError
from examples.hand_anatomy.geometry import Camera
from examples.hand_anatomy.pose_library import (
    SHAPES,
    TemplateLibrary,
    build_library,
    load_library,
    make_pose,
    save_library,
    validate_library,
)
from examples.hand_anatomy.template_renderer import RenderRecipe, render_template


def test_same_input_identical_mask_bytes_and_ids():
    pose = make_pose("open", "right", SHAPES["standard"])
    a, b = render_template(pose, Camera(yaw=40)), render_template(pose, Camera(yaw=40))
    assert a.labels.tobytes() == b.labels.tobytes() and a.template_id == b.template_id


def test_camera_changes_render_identity_not_geometry_identity():
    pose = make_pose("open", "right", SHAPES["standard"])
    a, b = render_template(pose, Camera(yaw=0)), render_template(pose, Camera(yaw=40))
    assert a.geometry_id == b.geometry_id and a.render_id != b.render_id
    assert a.mask_id != b.mask_id
    c = render_template(pose, Camera(yaw=0), RenderRecipe(pixels_per_unit=36.0))
    assert c.render_id != a.render_id and c.geometry_id == a.geometry_id


def test_invalid_render_parameters_rejected():
    pose = make_pose("open", "right", SHAPES["standard"])
    with pytest.raises(HandAnatomyError):
        render_template(pose, recipe=RenderRecipe(canvas=8))
    with pytest.raises(HandAnatomyError):
        render_template(pose, recipe=RenderRecipe(pixels_per_unit=float("nan")))
    with pytest.raises(HandAnatomyError):
        render_template(pose, recipe=RenderRecipe(pixels_per_unit=300.0))  # does not fit canvas


def test_library_deterministic_bounded_and_consistent(small_lib):
    again = build_library(gestures=["open", "spread", "fist"], yaws=(0.0, 40.0), shapes=["standard"])
    assert again.library_id == small_lib.library_id and len(small_lib) == 12
    assert validate_library(small_lib) == []
    assert len(build_library()) == 96
    with pytest.raises(HandAnatomyError):
        TemplateLibrary([], RenderRecipe())


def test_save_load_roundtrip_and_tamper_detection(small_lib, tmp_path):
    save_library(small_lib, tmp_path)
    assert load_library(tmp_path).library_id == small_lib.library_id
    meta = (tmp_path / "library.json").read_text()
    (tmp_path / "library.json").write_text(meta.replace('"mask_id": "sha256:', '"mask_id": "sha256:0', 1))
    with pytest.raises(HandAnatomyError):
        load_library(tmp_path)


def test_left_right_masks_are_mirror_images():
    r = render_template(make_pose("relaxed", "right", SHAPES["standard"]))
    left = render_template(make_pose("relaxed", "left", SHAPES["standard"]))
    assert r.template_id != left.template_id
    assert (r.mask ^ left.mask[:, ::-1]).sum() < 0.06 * r.mask.sum()
