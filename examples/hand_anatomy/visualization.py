"""Image outputs.  Colour convention for overlays (also documented in README):

* green  - observed structure explained by the aligned template (matched)
* red    - observed structure NOT explained by the aligned template
* blue   - structure the template expects but the observation lacks
* grey   - observed pixels inside the declared tolerance band of the template
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .pose_library import TemplateLibrary

GREEN = (60, 190, 90)
RED = (225, 50, 50)
BLUE = (60, 110, 230)
GREY = (150, 150, 150)
BLACK = (0, 0, 0)


def mask_image(mask: np.ndarray, scale: int = 1) -> Image.Image:
    img = Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), "L").convert("RGB")
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST) if scale != 1 else img


def overlay_image(
    observed: np.ndarray,
    template: np.ndarray,
    unexplained: np.ndarray,
    missing: np.ndarray,
    scale: int = 2,
) -> Image.Image:
    obs = np.asarray(observed, dtype=bool)
    out = np.zeros(obs.shape + (3,), dtype=np.uint8)
    out[obs] = GREY
    out[obs & np.asarray(template, dtype=bool)] = GREEN
    out[np.asarray(missing, dtype=bool)] = BLUE
    out[np.asarray(unexplained, dtype=bool)] = RED
    img = Image.fromarray(out, "RGB")
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST) if scale != 1 else img


def _labelled(tile: Image.Image, text: str, size: Tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", (size[0], size[1] + 14), (255, 255, 255))
    canvas.paste(tile.resize(size, Image.NEAREST), (0, 14))
    ImageDraw.Draw(canvas).text((2, 1), text, fill=(0, 0, 0))
    return canvas


def tile_sheet(tiles: Sequence[Tuple[str, Image.Image]], columns: int, tile: int = 120) -> Image.Image:
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (tile + 4), rows * (tile + 18)), (255, 255, 255))
    for i, (label, img) in enumerate(tiles):
        t = _labelled(img.convert("RGB"), label, (tile, tile))
        sheet.paste(t, ((i % columns) * (tile + 4), (i // columns) * (tile + 18)))
    return sheet


_PALETTE = np.array(
    [[255, 255, 255], [235, 200, 170], [200, 170, 140], [220, 90, 90], [90, 160, 220],
     [90, 200, 120], [230, 190, 60], [170, 100, 200]],
    dtype=np.uint8,
)


def contact_sheet(library: TemplateLibrary, path: str | Path, hand: str | None = None) -> Path:
    """Library overview: one tile per template, grouped by hand/gesture/shape/yaw."""
    selected = [t for t in library.templates if hand is None or t.pose.handedness == hand]
    selected.sort(key=lambda t: t.name)
    tiles = [
        (t.name.replace("standard", "std"), Image.fromarray(_PALETTE[t.labels], "RGB"))
        for t in selected
    ]
    sheet = tile_sheet(tiles, columns=6, tile=110)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    return target


def vpm_render(
    normalized: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    raw: np.ndarray,
    path: str | Path,
    cell: int = 44,
) -> Path:
    """Annotated VPM: grey level = normalized view value, text = RAW compatibility."""
    rows, cols = normalized.shape
    left = 150
    top = 74
    img = Image.new("RGB", (left + cols * cell, top + rows * cell), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for r in range(rows):
        d.text((2, top + r * cell + cell // 2 - 5), row_labels[r][:26], fill=(0, 0, 0))
        for c in range(cols):
            v = int(round(float(np.clip(normalized[r, c], 0, 1)) * 255))
            x0, y0 = left + c * cell, top + r * cell
            d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=(v, v, v))
            d.text(
                (x0 + 4, y0 + cell // 2 - 5),
                "%.2f" % raw[r, c],
                fill=(255, 255, 255) if v < 128 else (0, 0, 0),
            )
    for c in range(cols):
        tile = Image.new("RGB", (top, 12), (255, 255, 255))
        ImageDraw.Draw(tile).text((0, 0), col_labels[c][:14], fill=(0, 0, 0))
        tile = tile.rotate(90, expand=True)
        img.paste(tile, (left + c * cell + cell // 2 - 6, 0))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    img.save(target)
    return target
