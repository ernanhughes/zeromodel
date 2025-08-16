# tests/test_vpf_image_provenance.py
import hashlib
import time

import numpy as np
import pytest
from PIL import Image

from zeromodel.images.vpf import embed_vpf, extract_vpf, verify_vpf


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
    import time
    from io import BytesIO

    import numpy as np
    from PIL import Image

    # imports from the canonical images package
    from zeromodel.images import (embed_vpf, extract_vpf_from_png_bytes,
                                  replay_from_vpf, verify_vpf)
    from zeromodel.images.utils import \
        sha3_bytes as sha3_hex  # hex string (no 'sha3:' tag)

    # --- demo image ----------------------------------------------------------
    def _make_demo_image(w, h):
        return Image.new("RGB", (w, h), (9, 9, 9))

    img = _make_demo_image(1024, 512)

    # Optional: keep your stripe inputs around (not required for the new API)
    H = img.size[1]
    Hvals = H - 4
    x = np.linspace(0, 1, Hvals, dtype=np.float32)
    metrics_matrix = np.stack([
        0.5 + 0.5*np.sin(2*np.pi*3*x),
        0.5 + 0.5*np.cos(2*np.pi*5*x),
    ], axis=1)
    metric_names = ["aesthetic", "coherence"]

    # --- minimal VPF dict ----------------------------------------------------
    vpf = {
        "vpf_version": "1.0",
        "pipeline": {"graph_hash": "sha3:demo", "step": "render_tile"},
        "model": {"id": "demo", "assets": {}},
        "determinism": {"seed_global": 123, "rng_backends": ["numpy"]},
        "params": {"size": [img.size[0], img.size[1]], "steps": 28, "cfg_scale": 7.5},
        "inputs": {"prompt": "demo", "prompt_hash": sha3_hex(b"demo")},
        "metrics": {
            "aesthetic": float(metrics_matrix[:, 0].mean()),
            "coherence": float(metrics_matrix[:, 1].mean()),
        },
        "lineage": {"parents": []},
    }

    # 1) Embed — request a stripe/footer for backward-compat. New API stores VPF in iTXt.
    t0 = time.perf_counter()
    png_bytes = embed_vpf(
        img,
        vpf,
        mode="stripe",                    # ensures a ZMVF footer is appended
        # The new API can draw a cosmetic stripe internally; you don't need to pass matrix.
        # If you want to force your custom matrix/names, add these **only if implemented**:
        # stripe_metrics_matrix=metrics_matrix,
        # stripe_metric_names=metric_names,
        # stripe_channels=("R",),
    )
    t1 = time.perf_counter()

    # 2) Extract (from PNG bytes). Returns a plain dict + minimal metadata.
    extracted_vpf, quick = extract_vpf_from_png_bytes(png_bytes)
    t2 = time.perf_counter()

    # 3) Verify: by default, the new API compares SHA3 over the **full PNG bytes**.
    ok = verify_vpf(extracted_vpf, png_bytes)
    t3 = time.perf_counter()
    print(f"Embed: {t1-t0:.3f}s, Extract: {t2-t1:.3f}s, Verify: {t3-t2:.3f}s")

    # 4) If you still want to sanity-check the footer, it’s present in stripe mode.
    # Footer should be present in stripe mode
    idx = png_bytes.rfind(b"ZMVF")
    assert idx != -1, "Missing VPF footer when mode='stripe'"

    # Compute the content hash on the *core* PNG (no footer, no iTXt 'vpf' chunk)
    core_png_no_footer = _strip_footer(png_bytes)
    core_png = _strip_vpf_itxt(core_png_no_footer)

    assert extracted_vpf["lineage"]["content_hash"].endswith(sha3_hex(core_png)), "Content hash mismatch"


def _strip_vpf_itxt(png: bytes) -> bytes:
    """
    Return PNG bytes with any iTXt/tEXt/zTXt chunk whose key is 'vpf' removed.
    Does not recompress image data; preserves chunk order otherwise.
    """
    import struct
    sig = b"\x89PNG\r\n\x1a\n"
    assert png.startswith(sig), "Not a PNG"
    out = bytearray(sig)
    i = len(sig)
    while i < len(png):
        if i + 12 > len(png):
            break
        length = struct.unpack(">I", png[i:i+4])[0]
        ctype  = png[i+4:i+8]
        data_start = i + 8
        data_end   = data_start + length
        crc_end    = data_end + 4
        chunk = png[i:crc_end]

        if ctype in (b"iTXt", b"tEXt", b"zTXt"):
            # parse key for iTXt / tEXt / zTXt
            data = png[data_start:data_end]
            key = data.split(b"\x00", 1)[0]  # key is up to first NUL
            if key == b"vpf":
                # skip this chunk
                i = crc_end
                continue

        out.extend(chunk)
        i = crc_end
        if ctype == b"IEND":
            break
    return bytes(out)

def _strip_footer(png: bytes) -> bytes:
    foot = b"ZMVF"
    idx = png.rfind(foot)
    return png if idx == -1 else png[:idx]
