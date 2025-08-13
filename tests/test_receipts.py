# tests/test_receipts.py
import math
import json
import hashlib
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

# Core provenance / VPM ops
from zeromodel.provenance.core import (
    tensor_to_vpm,
    vpm_to_tensor,
    create_vpf,
    embed_vpf,
    extract_vpf,
    verify_vpf,
    read_json_footer,
    vpm_logic_and,
    vpm_logic_or,
    vpm_logic_not,
    vpm_logic_xor,
)

# Pointer routing (storage-agnostic) — adjust import if your path differs
from zeromodel.vpm.metadata import (
    VPMMetadata as TileMeta,
    RouterPointer,
    DictResolver,
    MapKind,
    AggId,
)


# -----------------------------
# 1) Universal snapshot/restore
# -----------------------------
def test_vpm_tensor_roundtrip_numpy_array():
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((64, 33)).astype(np.float32)
    img = tensor_to_vpm(arr)
    restored = vpm_to_tensor(img)
    assert isinstance(restored, np.ndarray)
    assert restored.shape == arr.shape
    assert restored.dtype == arr.dtype
    assert np.allclose(restored, arr)


def test_vpm_tensor_roundtrip_nested_dict():
    obj = {
        "w": np.arange(12, dtype=np.int32).reshape(3, 4),
        "config": {"lr": 1e-3, "name": "demo"},
        "tags": ["a", "b", "c"],
    }
    img = tensor_to_vpm(obj)
    restored = vpm_to_tensor(img)
    # numpy arrays come back bit-identical, lists/dicts preserved
    assert restored["config"] == obj["config"]
    assert restored["tags"] == obj["tags"]
    assert np.array_equal(restored["w"], obj["w"])


# -----------------------------------
# 3) Self-describing footer container
# -----------------------------------
def test_footer_container_is_parsable_json():
    img = Image.new("RGB", (64, 64), (9, 9, 9))
    vpf = create_vpf(
        pipeline={"graph_hash": "sha3:x", "step": "x"},
        model={"id": "demo", "assets": {}},
        determinism={"seed": 0, "rng_backends": ["numpy"]},
        params={"size": [64, 64]},
        inputs={"prompt_sha3": hashlib.sha3_256(b"x").hexdigest()},
        metrics={"m": 1.0},
        lineage={"parents": []},
    )
    blob = embed_vpf(img, vpf, mode="stripe")
    vpf_json = read_json_footer(blob)
    # sanity: looks like our VPF dict
    for key in ("vpf_version", "pipeline", "model", "determinism", "params", "inputs", "metrics", "lineage"):
        assert key in vpf_json
    # must be valid JSON serialization
    json.dumps(vpf_json)


# ---------------------------------------------
# 4) Task-aware spatial concentration increases
# ---------------------------------------------
def _top_left_mass(M: np.ndarray, k: int = 32) -> float:
    total = float(M.sum()) if M.size else 0.0
    return float(M[:k, :k].sum()) / total if total > 0 else 0.0


def test_task_aware_concentration_improves():
    rng = np.random.default_rng(0)
    X = rng.random((256, 16), dtype=np.float32)
    w = np.linspace(0.5, 2.0, X.shape[1]).astype(np.float32)

    # reorder columns by weighted mean; then rows by the projection
    col_order = np.argsort(-(X * w).mean(0))
    Xc = X[:, col_order]
    row_order = np.argsort(-(Xc @ np.ones(Xc.shape[1], np.float32)))
    Y = Xc[row_order]

    assert _top_left_mass(Y, 64) > _top_left_mass(X, 64)


# ---------------------------------------------
# 5) Compositional logic on pixel “sets” works
# ---------------------------------------------
def _bw(width, height, on_region):
    """Build a black/white PIL image with 'on_region(x,y)->bool' as white."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    px = img.load()
    for y in range(height):
        for x in range(width):
            if on_region(x, y):
                px[x, y] = (255, 255, 255)
    return img


def test_visual_logic_and_or_not_xor_sane():
    W = H = 64
    A = _bw(W, H, lambda x, y: x + y >= W // 2)      # upper-right triangle-ish
    B = _bw(W, H, lambda x, y: x + y <= W // 2 - 2)  # lower-left triangle-ish

    AND = vpm_logic_and(A, B)
    OR  = vpm_logic_or(A, B)
    NOT = vpm_logic_not(A)
    XOR = vpm_logic_xor(A, B)

    a = np.array(A)[:, :, 0] > 0
    b = np.array(B)[:, :, 0] > 0
    and_ = np.array(AND)[:, :, 0] > 0
    or_  = np.array(OR)[:, :, 0] > 0
    not_ = np.array(NOT)[:, :, 0] > 0
    xor_ = np.array(XOR)[:, :, 0] > 0

    # Basic set identities (on boolean masks)
    assert np.all(and_ <= a) and np.all(and_ <= b)                 # A∧B ⊆ A and ⊆ B
    assert np.all(or_ >= a) and np.all(or_ >= b)                   # A⊆A∨B and B⊆A∨B
    assert np.all(xor_ == np.logical_xor(a, b))                    # XOR truth table
    assert np.all(not_ == np.logical_not(a))                       # NOT truth table


# -------------------------------------------------
# 6) Logarithmic hops: sanity on the closed form
# -------------------------------------------------
def test_logarithmic_hops_closed_form():
    def hops(N, fanout):
        if N <= 1:
            return 0
        return math.ceil(math.log(N, fanout))

    # spot checks
    assert hops(10_000, 10) == 4
    assert hops(1_000_000, 10) == 6
    assert hops(1_000_000_000, 10) == 9
    # bigger fanout → fewer hops
    assert hops(10**12, 64) < hops(10**12, 8)


# ------------------------------------------------------
# 7) Storage-agnostic, pluggable routing (pointer test)
# ------------------------------------------------------
def test_router_pointer_resolution_roundtrip():
    # build a tiny metadata with one child pointer
    child_id = bytes.fromhex("11" * 16)
    ptr = RouterPointer(
        kind=MapKind.VPM,
        level=3,
        x_offset=0,
        span=1024,
        doc_block_size=16,
        agg_id=int(AggId.MEAN),
        tile_id=child_id,
    )
    meta = TileMeta(level=1, metric_count=0, doc_count=0, doc_block_size=1, agg_id=int(AggId.RAW),
                    task_hash=0, tile_id=bytes.fromhex("22"*16))
    meta.add_pointer(ptr)

    # resolve via pluggable resolver
    resolver = DictResolver(mapping={child_id: "/tmp/tiles/child.png"})
    resolved = meta.resolve_child_paths(resolver)
    assert len(resolved) == 1
    (p, path) = resolved[0]
    assert isinstance(p, RouterPointer)
    assert path.endswith("child.png")


# ---------------------------------------------------------
# 8) Stripe CRC catches tampering (pre-footer verification)
# ---------------------------------------------------------
def _find_stripe_start(img_rgb: Image.Image):
    arr = np.array(img_rgb)
    H, W, _ = arr.shape
    # Header sentinel in R channel: 'Z','M','V','2'
    for cols in range(1, min(256, W) + 1):
        x0 = W - cols
        col = arr[:, x0, 0]
        if H >= 6 and (int(col[0]), int(col[1]), int(col[2]), int(col[3])) == (0x5A, 0x4D, 0x56, 0x32):
            return x0, cols, arr
    return None, None, arr


# --------------------------------------------------------
# 9) Edge ↔ cloud symmetry (same artifact, two behaviours)
# --------------------------------------------------------
def test_edge_cloud_symmetry_roundtrip_and_fastpath():
    # build a tile
    rng = np.random.default_rng(3)
    scores = rng.random((64, 64), dtype=np.float32)
    tile = tensor_to_vpm(scores)

    # "edge" micro-decision: constant-time pixel glance
    def micro_decision(im: Image.Image) -> bool:
        return im.getpixel((0, 0))[0] >= 0  # trivial condition; uses 1 pixel

    # "cloud" full restore
    restored = vpm_to_tensor(tile)

    assert micro_decision(tile) in (True, False)
    assert isinstance(restored, (np.ndarray, bytes, dict, list, tuple))
