"""Polygon mask helper for creating reference hand masks by hand (ZM-HAND-03b).

    python -m examples.hand_anatomy_03.mask_tool grid   IMAGE OUT.png [--scale 3]
    python -m examples.hand_anatomy_03.mask_tool render IMAGE POLYGONS.json OUT_PREFIX [--scale 3]

``grid`` writes an enlarged copy of the image with coordinate gridlines every 20 original
pixels (labelled in ORIGINAL pixel coordinates, x to the right, y downward).

POLYGONS.json:  {"polygons": [{"op": "add", "points": [[x, y], ...]},
                              {"op": "sub", "points": [[x, y], ...]}]}
Operations are applied in order on an empty mask: ``add`` fills, ``sub`` clears (use it for
gaps between fingers).  ``render`` writes OUT_PREFIX.mask.png (original size, white = hand)
and OUT_PREFIX.overlay.png (enlarged, red tint, yellow outline, gridlines).

Drawing rule: mask what is VISIBLE.  Never complete, repair or normalise the anatomy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw

GRID = 20


def _grid(img: Image.Image, scale: int) -> Image.Image:
    big = img.convert("RGB").resize((img.width * scale, img.height * scale), Image.LANCZOS)
    d = ImageDraw.Draw(big)
    for x in range(0, img.width, GRID):
        d.line([(x * scale, 0), (x * scale, big.height)], fill=(0, 255, 255) if x % 100 == 0 else (120, 220, 220), width=1)
        d.text((x * scale + 2, 2), str(x), fill=(255, 255, 0))
    for y in range(0, img.height, GRID):
        d.line([(0, y * scale), (big.width, y * scale)], fill=(0, 255, 255) if y % 100 == 0 else (120, 220, 220), width=1)
        d.text((2, y * scale + 2), str(y), fill=(255, 255, 0))
    return big


def build_mask(size: tuple, spec: Dict[str, Any]) -> np.ndarray:
    add = Image.new("L", size, 0)
    mask = np.zeros((size[1], size[0]), dtype=bool)
    for poly in spec["polygons"]:
        pts = [(float(x), float(y)) for x, y in poly["points"]]
        if len(pts) < 3:
            raise ValueError("polygon needs at least 3 points")
        layer = Image.new("L", size, 0)
        ImageDraw.Draw(layer).polygon(pts, fill=255)
        layer_arr = np.asarray(layer) > 127
        if poly["op"] == "add":
            mask |= layer_arr
        elif poly["op"] == "sub":
            mask &= ~layer_arr
        else:
            raise ValueError("op must be add or sub")
    del add
    return mask


def render(image: Path, spec_path: Path, prefix: Path, scale: int) -> Dict[str, Any]:
    img = Image.open(image).convert("RGB")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    mask = build_mask(img.size, spec)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8), "L").save(str(prefix) + ".mask.png")
    base = np.asarray(img.resize((img.width * scale, img.height * scale), Image.LANCZOS)).astype(np.float32)
    big = np.asarray(Image.fromarray((mask * 255).astype(np.uint8), "L").resize(base.shape[1::-1], Image.NEAREST)) > 127
    out = base.copy()
    out[big] = 0.6 * base[big] + 0.4 * np.array([255.0, 40.0, 40.0], dtype=np.float32)
    pad = np.pad(big, 1)
    edge = big & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    out[edge] = (255, 255, 0)
    ov = Image.fromarray(out.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(ov)
    for x in range(0, img.width, GRID):
        d.text((x * scale + 2, 2), str(x), fill=(0, 255, 255))
    for y in range(0, img.height, GRID):
        d.text((2, y * scale + 2), str(y), fill=(0, 255, 255))
    ov.save(str(prefix) + ".overlay.png")
    return {"area_fraction": float(mask.mean()), "size": list(img.size)}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("image")
    g.add_argument("out")
    g.add_argument("--scale", type=int, default=3)
    r = sub.add_parser("render")
    r.add_argument("image")
    r.add_argument("polygons")
    r.add_argument("prefix")
    r.add_argument("--scale", type=int, default=3)
    args = ap.parse_args(argv)
    if args.cmd == "grid":
        img = Image.open(args.image)
        _grid(img, args.scale).save(args.out)
        print(args.out)
    else:
        print(json.dumps(render(Path(args.image), Path(args.polygons), Path(args.prefix), args.scale)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
