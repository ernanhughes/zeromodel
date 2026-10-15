"""Non-hand control: the same pipeline on synthetic blobs (ellipse, rounded rectangle, L, blobs).

python -m examples.hand_anatomy_03.blob_control
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from examples.hand_anatomy_03.real_pipeline import assess_one


def blob(kind, rng, side=160):
    img = Image.new("L", (side, side), 0)
    d = ImageDraw.Draw(img)
    cx, cy = side / 2 + rng.uniform(-8, 8), side / 2 + rng.uniform(-8, 8)
    if kind == "ellipse":
        a, b = rng.uniform(30, 55), rng.uniform(22, 45)
        d.ellipse([cx - a, cy - b, cx + a, cy + b], fill=255)
    elif kind == "rounded_rect":
        a, b = rng.uniform(28, 55), rng.uniform(22, 48)
        d.rounded_rectangle(
            [cx - a, cy - b, cx + a, cy + b], radius=int(rng.uniform(6, 18)), fill=255
        )
    elif kind == "L_shape":
        a = rng.uniform(40, 55)
        t = rng.uniform(18, 26)
        d.rectangle([cx - a, cy - a, cx - a + t, cy + a], fill=255)
        d.rectangle([cx - a, cy + a - t, cx + a, cy + a], fill=255)
    elif kind == "blobs":
        for _ in range(4):
            r = rng.uniform(14, 26)
            x, y = cx + rng.uniform(-24, 24), cy + rng.uniform(-24, 24)
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return np.asarray(img) > 127


if __name__ == "__main__":
    out = Path("artifacts/zm_hand_03/control")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    jobs = []
    for i, kind in enumerate(["ellipse", "rounded_rect", "L_shape", "blobs"] * 3):
        m = blob(kind, rng)
        name = "ctl%02d_%s.jpg" % (i, kind)
        Image.fromarray((m * 255).astype(np.uint8)).save(
            out / (name[:-4] + ".mask.png")
        )
        jobs.append(
            {
                "name": name,
                "label": -1,
                "video": None,
                "image": "",
                "mask": str(out / (name[:-4] + ".mask.png")),
                "seg": {
                    "status": "ok",
                    "reasons": [],
                    "input_size": [160, 160],
                    "mask_area_fraction": float(m.mean()),
                    "mean_probability_inside_mask": 1.0,
                },
                "overlay_dir": str(out / "compare"),
            }
        )
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(assess_one, jobs))
    json.dump(res, open(out / "control_cases.json", "w"), indent=1)
    for r in res:
        print(
            r["name"],
            r.get("eligible"),
            r.get("best_library_iou"),
            r.get("best_refined_iou"),
            r.get("scores", {}).get("C_residual"),
        )
    li = [r["best_library_iou"] for r in res if r.get("eligible")]
    ri = [r["best_refined_iou"] for r in res if r.get("eligible")]
    print(
        "control library IoU median %.2f range %.2f-%.2f | refined median %.2f range %.2f-%.2f"
        % (np.median(li), min(li), max(li), np.median(ri), min(ri), max(ri))
    )
