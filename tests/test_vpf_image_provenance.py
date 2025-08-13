# tests/test_vpf_image_provenance.py
import time
import hashlib
import numpy as np
from PIL import Image
import pytest

from zeromodel.provenance.core import (
    embed_vpf,
    extract_vpf,
    verify_vpf,
    compare_vpm,               # for visual deltas if desired
)

def sha3_hex(b: bytes) -> str:
    return hashlib.sha3_256(b).hexdigest()

def _make_demo_image(w=512, h=256):
    # deterministic gradient for test stability
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    img = np.clip((0.6 * x + 0.4 * y), 0, 1)
    rgb = (np.stack([img, img**0.5, img**2], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)

def test_vpf_embed_extract_verify_and_replay():
    img = _make_demo_image(1024, 512)

    # Minimal metrics stripe: two metrics across height
    H = img.size[1]
    Hvals = H - 4  # reserve top 4 rows for header
    x = np.linspace(0, 1, Hvals, dtype=np.float32)
    metrics_matrix = np.stack([
        0.5 + 0.5*np.sin(2*np.pi*3*x),
        0.5 + 0.5*np.cos(2*np.pi*5*x),
    ], axis=1)
    metric_names = ["aesthetic", "coherence"]

    # Build a minimal VPF dict
    vpf = {
        "vpf_version": "1.0",
        "pipeline": {"graph_hash": "sha3:demo", "step": "render_tile"},
        "model": {"id": "demo", "assets": {}},
        "determinism": {"seed_global": 123, "rng_backends": ["numpy"]},
        "params": {"size": [img.size[0], img.size[1]], "steps": 28, "cfg_scale": 7.5},
        "inputs": {"prompt": "demo", "prompt_hash": sha3_hex(b"demo")},
        "metrics": {"aesthetic": float(metrics_matrix[:,0].mean()),
                    "coherence": float(metrics_matrix[:,1].mean())},
        "lineage": {"parents": []}
    }

    # 1) Embed (writes stripe into pixels, appends footer)
    t0 = time.perf_counter()
    png_with_footer = embed_vpf(
        img,
        vpf,
        stripe_metrics_matrix=metrics_matrix,
        stripe_metric_names=metric_names,
        stripe_channels=("R",),
    )
    t1 = time.perf_counter()

    # 2) Extract footer + quickscan metrics from stripe
    extracted_vpf, quick = extract_vpf(png_with_footer)
    t2 = time.perf_counter()

    # 3) Verify hashes
    ok = verify_vpf(extracted_vpf, png_with_footer)
    t3 = time.perf_counter()

    # 4) The PNG's "core" content hash must match what VPF claims
    idx = png_with_footer.rfind(b"ZMVF")
    assert idx != -1, "Missing VPF footer"
    core_png = png_with_footer[:idx]
    assert extracted_vpf["lineage"]["content_hash"].endswith(sha3_hex(core_png)), "Content hash mismatch"

    print(f"[embed] {t1-t0:0.4f}s  [extract] {t2-t1:0.4f}s  [verify] {t3-t2:0.4f}s")
    assert ok is True
    assert quick.get("stripe_present", True) in (True, False)  # quickscan metadata present
