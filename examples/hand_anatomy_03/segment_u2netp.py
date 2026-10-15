"""Standalone salient-object hand segmentation with U^2-Net-small (u2netp, ONNX).

Runs in an ISOLATED virtual environment (onnxruntime + numpy + pillow only); it imports
nothing from ZeroModel.  Model: ``u2netp.onnx`` (U^2-Net, Apache-2.0; ONNX export
distributed with rembg, MIT).  The checkpoint sha256 is verified and recorded.

Preprocessing (identical to the rembg reference): RGB -> resize 320x320 (Lanczos) ->
divide by max pixel value -> ImageNet mean/std -> NCHW float32.  Postprocessing: first
output map, min-max normalised, resized back to the input size, threshold at 0.5.

Per image it writes ``<stem>.mask.png``, ``<stem>.prob.png``, ``<stem>.overlay.png`` and
one ``segmentation.json`` with explicit status and quality evidence.  The statuses are
evidence, not corrections: a mask is never edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL_SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"
SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THRESHOLD = 0.5
MIN_AREA_FRACTION = 0.02
MAX_AREA_FRACTION = 0.95
LOW_CONFIDENCE = 0.80
MIN_SIDE_PX = 32


def load_session(model: Path) -> ort.InferenceSession:
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise SystemExit("checkpoint sha256 mismatch: %s" % digest)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])


def predict(session: ort.InferenceSession, img: Image.Image) -> np.ndarray:
    rgb = img.convert("RGB")
    arr = np.asarray(rgb.resize((SIZE, SIZE), Image.LANCZOS), dtype=np.float32)
    arr = arr / max(float(arr.max()), 1e-6)
    arr = ((arr - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    out = session.run(None, {session.get_inputs()[0].name: arr})[0][0, 0]
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
    prob = Image.fromarray((out * 255).astype(np.uint8), "L").resize(rgb.size, Image.LANCZOS)
    return np.asarray(prob, dtype=np.float32) / 255.0


def _components(mask: np.ndarray) -> Tuple[int, float]:
    """8-connected component count and largest-component share (small masks only)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: List[int] = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack, n = [(y, x)], 0
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            n += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        sizes.append(n)
    total = sum(sizes)
    return len(sizes), (max(sizes) / total if total else 0.0)


def assess(prob: np.ndarray, mask: np.ndarray, size: Tuple[int, int]) -> Dict[str, Any]:
    area = float(mask.mean())
    inside = prob[mask]
    ncomp, largest = _components(mask) if mask.sum() < 400000 else (-1, 0.0)
    ring = np.zeros_like(mask)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    status, reasons = "ok", []
    if min(size) < MIN_SIDE_PX:
        status, reasons = "failed", ["image_smaller_than_%dpx" % MIN_SIDE_PX]
    elif area < MIN_AREA_FRACTION:
        status, reasons = "failed", ["mask_area_below_%.2f" % MIN_AREA_FRACTION]
    elif area > MAX_AREA_FRACTION:
        status, reasons = "failed", ["mask_covers_nearly_everything"]
    else:
        if float(inside.mean()) < LOW_CONFIDENCE:
            status = "low_confidence"
            reasons.append("mean_probability_inside_mask_below_%.2f" % LOW_CONFIDENCE)
        if ncomp > 1 and largest < 0.9:
            status = "low_confidence" if status == "ok" else status
            reasons.append("fragmented_mask")
    return {
        "status": status, "reasons": reasons, "mask_area_fraction": round(area, 4),
        "mean_probability_inside_mask": round(float(inside.mean()), 4) if inside.size else 0.0,
        "uncertain_pixel_fraction_0.3_0.7": round(float(((prob > 0.3) & (prob < 0.7)).mean()), 4),
        "components": ncomp, "largest_component_share": round(largest, 4),
        "border_contact_fraction": round(float((mask & ring).sum()) / max(1, int(ring.sum())), 4),
        "input_size": list(size),
    }


def overlay(img: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(img.convert("RGB")).astype(np.float32)
    red = np.array([255.0, 40.0, 40.0], dtype=np.float32)
    out = base.copy()
    out[mask] = 0.55 * base[mask] + 0.45 * red
    edge = mask & ~np.pad(mask, 1)[:-2, 1:-1] | mask & ~np.pad(mask, 1)[2:, 1:-1] | \
        mask & ~np.pad(mask, 1)[1:-1, :-2] | mask & ~np.pad(mask, 1)[1:-1, 2:]
    out[edge] = (255, 255, 0)
    return Image.fromarray(out.astype(np.uint8), "RGB")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args(argv)
    session = load_session(Path(args.model))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Any] = {}
    t_all = time.perf_counter()
    for path in sorted([*Path(args.images).glob("*.jpg"), *Path(args.images).glob("*.png")]):
        t0 = time.perf_counter()
        img = Image.open(path)
        img.load()
        prob = predict(session, img)
        mask = prob > THRESHOLD
        rec = assess(prob, mask, img.size)
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        records[path.name] = rec
        Image.fromarray((mask * 255).astype(np.uint8), "L").save(out / (path.stem + ".mask.png"))
        Image.fromarray((prob * 255).astype(np.uint8), "L").save(out / (path.stem + ".prob.png"))
        overlay(img, mask).save(out / (path.stem + ".overlay.png"))
    meta = {
        "provider": "u2netp (U^2-Net small) ONNX", "checkpoint_sha256": MODEL_SHA256,
        "onnxruntime": ort.__version__, "numpy": np.__version__,
        "settings": {"input_size": SIZE, "mean": MEAN.tolist(), "std": STD.tolist(), "threshold": THRESHOLD,
                     "min_area_fraction": MIN_AREA_FRACTION, "max_area_fraction": MAX_AREA_FRACTION,
                     "low_confidence": LOW_CONFIDENCE, "min_side_px": MIN_SIDE_PX},
        "total_seconds": round(time.perf_counter() - t_all, 2),
    }
    (out / "segmentation.json").write_text(json.dumps({"meta": meta, "images": records}, indent=1, sort_keys=True) + "\n",
                                           encoding="utf-8")
    counts: Dict[str, int] = {}
    for r in records.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(json.dumps({"images": len(records), "status": counts, "seconds": meta["total_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
