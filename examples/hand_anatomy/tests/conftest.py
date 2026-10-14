from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from examples.hand_anatomy import observations as ob
from examples.hand_anatomy import pose_library as pl


def pytest_collection_modifyitems(items):
    for item in items:
        if "hand_anatomy" in item.path.parts:
            item.add_marker(pytest.mark.research)


@pytest.fixture(scope="session")
def small_lib():
    return pl.build_library(
        gestures=["open", "spread", "fist"], yaws=(0.0, 40.0), shapes=["standard"]
    )


def observation_from_template(tpl, rot=0.0, scale=1.0, shift=(30, 20), canvas=420, extra=None):
    """Place a library template into a bigger image (optionally rotated/scaled/edited)."""
    mask = tpl.mask.copy()
    if extra is not None:
        mask = extra(mask)
    img = Image.fromarray((mask * 255).astype(np.uint8), "L").rotate(rot, resample=Image.BILINEAR)
    size = int(round(img.width * scale))
    arr = np.asarray(img.resize((size, size), Image.BILINEAR)) > 127
    out = np.zeros((canvas, canvas), dtype=bool)
    out[shift[1]:shift[1] + size, shift[0]:shift[0] + size] = arr
    return ob.observation_from_arrays(out)
