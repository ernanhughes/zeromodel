# tests/test_vpm_metadata_suite.py
import logging
import math
import os
import time

import numpy as np
import pytest

from zeromodel import VPMImageReader, VPMImageWriter
from zeromodel.vpm.metadata import (_META_FIXED_SIZE, _ROUTER_PTR_SIZE, AggId,
                                    DictResolver, FilenameResolver, MapKind,
                                    RouterPointer, VPMMetadata)

logger = logging.getLogger(__name__)

# --------------------
# helpers
# --------------------

def _mk_scores(M, D, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((M, D))
    scales = (rng.random(M) * 4.0 + 0.5).reshape(M, 1)
    offsets = rng.random(M).reshape(M, 1) * 2.0
    return base * scales + offsets

def _q4(w):
    # match nibble packing (round to nearest of 16 buckets)
    w = max(0.0, min(1.0, float(w)))
    return round(w * 15) / 15.0

def _weights_for_metrics(metrics, seed=123):
    rng = np.random.default_rng(seed)
    return {m: float(rng.random()) for m in metrics}

# --------------------
# struct sanity
# --------------------

def test_struct_sizes_regression_guard():
    # Header is fixed; if this drifts, serialization will break
    assert _META_FIXED_SIZE == 84
    # Router pointer must be 36 bytes
    assert _ROUTER_PTR_SIZE == 36

# --------------------
# 1) weights-only roundtrip (TILE)
# --------------------

def test_meta_weights_roundtrip_tile():
    M, D = 12, 256
    metrics = [f"m{i}" for i in range(M)]
    task_weights = {
        "m0": 0.00, "m1": 0.07, "m2": 0.14, "m3": 0.33, "m4": 0.50,
        "m5": 0.66, "m6": 0.75, "m7": 0.90, "m8": 1.00,
    }
    tile_id = VPMMetadata.make_tile_id(b"tile:weights_only")
    meta = VPMMetadata.for_tile(
        level=1, metric_count=M, doc_count=D,
        doc_block_size=1, agg_id=int(AggId.RAW),
        metric_weights=task_weights, metric_names=metrics,
        task_hash=0x12345678, tile_id=tile_id
    )
    blob = meta.to_bytes()
    back = VPMMetadata.from_bytes(blob)

    assert back.kind == MapKind.VPM
    assert back.level == 1
    assert back.agg_id == int(AggId.RAW)
    assert back.metric_count == M
    assert back.doc_count == D
    assert back.task_hash == 0x12345678
    assert back.tile_id == tile_id

    decoded = back.get_weights(metrics)
    for m in metrics:
        expect = _q4(task_weights.get(m, 0.0))
        assert decoded[m] == pytest.approx(expect, abs=1/15)

# --------------------
# 2) router frame roundtrip + step linkage
# --------------------

def test_meta_router_frame_roundtrip():
    metrics = ["lane_a", "lane_b", "lane_c"]
    lane_weights = {"lane_a": 0.2, "lane_b": 0.55, "lane_c": 0.9}
    tid = VPMMetadata.make_tile_id(b"router:child")
    pid = VPMMetadata.make_tile_id(b"router:parent")

    meta = VPMMetadata.for_router_frame(
        step_id=123, parent_step_id=122,
        lane_weights=lane_weights, metric_names=metrics,
        tile_id=tid, parent_id=pid, level=7, timestamp_ns=42
    )
    blob = meta.to_bytes()
    back = VPMMetadata.from_bytes(blob)

    assert back.kind == MapKind.ROUTER_FRAME
    assert back.level == 7
    assert back.step_id == 123
    assert back.parent_step_id == 122
    assert back.timestamp_ns == 42
    assert back.tile_id == tid and back.parent_id == pid

    w = back.get_weights(metrics)
    for k, v in lane_weights.items():
        assert w[k] == pytest.approx(_q4(v), abs=1/15)

# --------------------
# 3) search view (orthogonal) encoding (no PNG)
# --------------------

def test_meta_search_view_weights_only():
    metrics = [f"dim{i}" for i in range(5)]
    weights = {"dim0": 0.1, "dim1": 0.9}
    meta = VPMMetadata(
        version=1, kind=MapKind.SEARCH_VIEW, level=0,
        agg_id=int(AggId.MEAN),
        metric_count=len(metrics), doc_count=0, doc_block_size=1,
        task_hash=0xA5A5A5A5, tile_id=VPMMetadata.make_tile_id(b"search:view"),
        weights_nibbles=b""
    )
    meta.set_weights(weights, metrics)
    back = VPMMetadata.from_bytes(meta.to_bytes())
    assert back.kind == MapKind.SEARCH_VIEW
    assert back.agg_id == int(AggId.MEAN)
    got = back.get_weights(metrics)
    for m in metrics:
        assert got[m] == pytest.approx(_q4(weights.get(m, 0.0)), abs=1/15)

# --------------------
# 4) router pointer pack/unpack
# --------------------

def test_router_pointer_pack_unpack():
    tid = VPMMetadata.make_tile_id(b"child")
    ptr = RouterPointer(
        kind=MapKind.VPM,
        level=3,
        x_offset=12_345,
        span=1_000,
        doc_block_size=8,
        agg_id=int(AggId.MAX),
        tile_id=tid
    )
    b = ptr.to_bytes()
    assert len(b) == 36
    back = RouterPointer.from_bytes(b)
    assert back.kind == MapKind.VPM
    assert back.level == 3
    assert back.x_offset == 12_345
    assert back.span == 1_000
    assert back.doc_block_size == 8
    assert back.agg_id == int(AggId.MAX)
    assert back.tile_id == tid

# --------------------
# 5) pointers list + resolvers
# --------------------

def test_pointers_and_resolvers():
    ptrs = []
    mapping = {}
    for i in range(5):
        tid = VPMMetadata.make_tile_id(f"child-{i}".encode())
        ptrs.append(RouterPointer(
            kind=MapKind.VPM, level=1, x_offset=i*1000, span=1000,
            doc_block_size=1, agg_id=int(AggId.MAX), tile_id=tid
        ))
        mapping[tid] = f"/data/tiles/{tid.hex()}.png"

    meta = VPMMetadata(
        version=1, kind=MapKind.VPM, level=0, agg_id=int(AggId.RAW),
        metric_count=10, doc_count=100, doc_block_size=1,
        tile_id=VPMMetadata.make_tile_id(b"root"), pointers=ptrs
    )

    # DictResolver
    paths = meta.resolve_child_paths(DictResolver(mapping=mapping))
    assert len(paths) == 5
    for (ptr, path) in paths:
        assert path == mapping[ptr.tile_id]

    # FilenameResolver (patterned)
    fr = FilenameResolver(pattern="vpm_{hexid}_L{level}_B{block}.png", default_level=1, default_block=2)
    paths2 = meta.resolve_child_paths(fr)
    for (ptr, path) in paths2:
        assert path.endswith("_L1_B2.png")
        assert ptr.tile_id.hex() in path

# --------------------
# 6) integrate into PNG: embed metadata; spill to extra rows; read bytes back
# --------------------

def test_png_embed_and_read_metadata_spillover(tmp_path):
    D = 32            # narrow width limits first header row capacity
    M = 200           # enough metrics to force spill across rows
    metrics = [f"m{i}" for i in range(M)]
    scores = _mk_scores(M, D, seed=7)

    # dense weights
    dense = _weights_for_metrics(metrics, seed=123)

    # a bunch of pointers (make payload bigger)
    ptrs = []
    for i in range(24):
        tid = VPMMetadata.make_tile_id(f"child-{i}".encode())
        ptrs.append(RouterPointer(
            kind=MapKind.VPM, level=2,
            x_offset=i * 1000, span=1000,
            doc_block_size=1, agg_id=int(AggId.MAX),
            tile_id=tid
        ))

    tile_id = VPMMetadata.make_tile_id(b"png:meta:spill")
    meta = VPMMetadata.for_tile(
        level=3, metric_count=M, doc_count=D,
        doc_block_size=1, agg_id=int(AggId.RAW),
        metric_weights=dense, metric_names=metrics,
        task_hash=0xDEADBEEF, tile_id=tile_id
    )
    for p in ptrs:
        meta.add_pointer(p)

    payload = meta.to_bytes()

    out = tmp_path / "vpm_meta_spill.png"
    VPMImageWriter(
        score_matrix=scores,
        store_minmax=False,
        compression=6,
        level=3,
        doc_block_size=1,
        agg_id=int(AggId.RAW),
        metadata_bytes=payload,
    ).write(str(out))

    r = VPMImageReader(str(out))
    raw = r.read_metadata_bytes()
    assert len(raw) == len(payload)

    back = VPMMetadata.from_bytes(raw)
    assert back.level == 3
    assert back.doc_count == D
    assert back.metric_count == M
    assert len(back.pointers) == len(ptrs)
    # spot-check a couple of weights
    got = back.get_weights(metrics)
    for m in ["m0", "m1", "m50", "m123", "m199"]:
        assert got[m] == pytest.approx(_q4(dense[m]), abs=1/15)

# --------------------
# 7) behavior when PNG has no metadata chunk
# --------------------

def test_png_without_metadata_returns_empty(tmp_path):
    M, D = 6, 64
    metrics = [f"m{i}" for i in range(M)]
    scores = _mk_scores(M, D, seed=5)

    out = tmp_path / "vpm_no_meta.png"
    VPMImageWriter(
        score_matrix=scores,
        store_minmax=False,
        compression=6,
        level=1,
        doc_block_size=1,
        agg_id=int(AggId.RAW),
        metadata_bytes=b"",  # none
    ).write(str(out))

    r = VPMImageReader(str(out))
    raw = r.read_metadata_bytes()
    assert raw == b""

    # You can still create a minimal default metadata object:
    default_meta = VPMMetadata()  # empty/default values serialize fine
    roundtrip = VPMMetadata.from_bytes(default_meta.to_bytes())
    assert roundtrip.kind == MapKind.VPM
    assert roundtrip.metric_count == 0
    assert roundtrip.doc_count == 0

# --------------------
# 8) quantization boundaries (nibbles)
# --------------------

@pytest.mark.parametrize("val", [0.0, 1.0, 0.5, 1/15, 14/15])
def test_weight_quantization_boundaries(val):
    metrics = ["a", "b", "c"]
    meta = VPMMetadata(
        kind=MapKind.VPM, level=0, agg_id=int(AggId.RAW),
        metric_count=len(metrics), doc_count=10, doc_block_size=1,
        tile_id=VPMMetadata.make_tile_id(b"qtest")
    )
    meta.set_weights({"a": val, "b": val, "c": val}, metrics)
    back = VPMMetadata.from_bytes(meta.to_bytes())
    got = back.get_weights(metrics)
    for m in metrics:
        assert got[m] == pytest.approx(_q4(val), abs=1e-9)


