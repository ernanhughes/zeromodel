"""Controlled test: does the segmenter preserve a KNOWN extra finger?

Counterfactual pairs from ZM-HAND-02 (cal02 split, distinguishable extra-finger pairs) are
rendered as skin-toned photographs-like images (shaded fill, blurred edge, textured
background) in two conditions, then segmented with the isolated u2netp provider.

Suitability rule, declared BEFORE running (easy condition, defect images only):
    median IoU(predicted, true) >= 0.90  AND
    extra-region recall >= 0.80 on at least 80% of defect images,
where extra-region = true defect mask minus the valid twin mask.  If not met, the
segmenter is marked UNSUITABLE for anatomy-defect work.  The hard condition (skin-like
background) is reported but not part of the rule.

    python -m examples.hand_anatomy_03.segmenter_preservation_test --output artifacts/zm_hand_03/preservation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageFilter

from examples.hand_anatomy_02.pairs import PRIMARY_SUBTYPES, build_dataset

RULE = {"median_iou_min": 0.90, "extra_region_recall_min": 0.80, "share_of_defect_images_min": 0.80}
SIDE = 192
TONES = ((196, 142, 112), (150, 100, 78), (226, 180, 150))


def _crop(mask: np.ndarray, box: tuple) -> np.ndarray:
    y0, y1, x0, x1 = box
    img = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), "L").resize((SIDE, SIDE), Image.BILINEAR)
    return np.asarray(img) > 127


def render(mask: np.ndarray, tone: tuple, hard: bool, rng: np.random.Generator) -> Image.Image:
    low = rng.uniform(0, 1, (6, 6, 3))
    bg = np.asarray(Image.fromarray((low * 255).astype(np.uint8)).resize((SIDE, SIDE), Image.BICUBIC), dtype=np.float32)
    if hard:
        bg = 0.35 * bg + 0.65 * np.array(tone, dtype=np.float32) * rng.uniform(0.85, 1.1)
    else:
        bg = 0.45 * bg + 0.55 * np.array((70, 110, 90), dtype=np.float32)
    shade = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)), dtype=np.float32) / 255
    fg = np.array(tone, dtype=np.float32)[None, None] * (0.75 + 0.35 * shade[..., None])
    fg = fg + rng.normal(0, 4, fg.shape)
    alpha = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)),
                       dtype=np.float32)[..., None] / 255
    out = np.clip(bg * (1 - alpha) + fg * alpha + rng.normal(0, 3, bg.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--python", default="artifacts/zm_hand_03/venv_seg/Scripts/python.exe")
    ap.add_argument("--model", default="artifacts/zm_hand_03/seg/u2netp.onnx")
    args = ap.parse_args(argv)
    out = Path(args.output)
    pairs = [p for p in build_dataset()["cal02"]["pairs"] if p.category == "extra_finger"
             and p.subtype in PRIMARY_SUBTYPES and not p.indistinguishable][: args.pairs]
    meta: Dict[str, Any] = {}
    for hard in (False, True):
        cond = "hard" if hard else "easy"
        img_dir = out / ("images_" + cond)
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(pairs):
            rng = np.random.default_rng(1000 + i)
            ys, xs = np.nonzero(p.defect | p.valid)
            box = (max(0, ys.min() - 24), min(520, ys.max() + 24), max(0, xs.min() - 24), min(520, xs.max() + 24))
            side = max(box[1] - box[0], box[3] - box[2])
            box = (box[0], box[0] + side, box[2], box[2] + side)
            tone = TONES[i % len(TONES)]
            for role, m in (("valid", p.valid), ("defect", p.defect)):
                mm = np.zeros((600, 600), bool)
                mm[:520, :520] = m
                name = "pair%02d_%s.jpg" % (i, role)
                render(_crop(mm, box), tone, hard, rng).save(img_dir / name, quality=92)
                meta[cond + "/" + name] = {"truth": _crop(mm, box)}
                if role == "defect":
                    vv = np.zeros((600, 600), bool)
                    vv[:520, :520] = p.valid
                    meta[cond + "/" + name]["extra"] = _crop(mm, box) & ~_crop(vv, box)
        subprocess.run([str(Path(args.python).resolve()), str(Path(__file__).with_name("segment_u2netp.py")), "--images",
                        str(img_dir), "--output", str(out / ("seg_" + cond)), "--model", str(Path(args.model).resolve())],
                       check=True,
                       stdout=subprocess.DEVNULL)
    results: Dict[str, Any] = {"rule": RULE, "conditions": {}}
    for cond in ("easy", "hard"):
        ious: List[float] = []
        recalls: List[float] = []
        merged_valid = []
        for key, m in meta.items():
            if not key.startswith(cond + "/"):
                continue
            stem = key.split("/")[1][:-4]
            pred = np.asarray(Image.open(out / ("seg_" + cond) / (stem + ".mask.png"))) > 127
            truth = m["truth"]
            iou = float((pred & truth).sum()) / max(1.0, float((pred | truth).sum()))
            if "extra" in m:
                ious.append(iou)
                extra = m["extra"]
                recalls.append(float((pred & extra).sum()) / max(1.0, float(extra.sum())))
            else:
                merged_valid.append(iou)
        results["conditions"][cond] = {
            "defect_images": len(ious), "median_iou_defect": round(statistics.median(ious), 4),
            "median_iou_valid": round(statistics.median(merged_valid), 4),
            "median_extra_region_recall": round(statistics.median(recalls), 4),
            "share_extra_recall_ge_0.8": round(sum(1 for r in recalls if r >= 0.8) / len(recalls), 4),
            "extra_region_recall_values": [round(r, 3) for r in recalls],
        }
    e = results["conditions"]["easy"]
    ok = (e["median_iou_defect"] >= RULE["median_iou_min"]
          and e["share_extra_recall_ge_0.8"] >= RULE["share_of_defect_images_min"])
    results["verdict"] = "SUITABLE_FOR_THIS_CONDITION" if ok else "UNSUITABLE_FOR_ANATOMY_DEFECT_WORK"
    (out / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "rule"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
