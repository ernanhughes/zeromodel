# tests/test_vpm_metadata_v2.py
import numpy as np
import pytest
from zeromodel.vpm.metadata import (
    VPMMetadata, MapKind, AggId, RouterPointer
)

# If your writer/reader expose PNG embedding, import them; otherwise skip those parts.
from zeromodel import VPMImageWriter, VPMImageReader  # adjust import if needed

def _mk_scores(M, D, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((M, D))
    scales = (rng.random(M) * 4.0 + 0.5).reshape(M, 1)
    offsets = rng.random(M).reshape(M, 1) * 2.0
    return base * scales + offsets

def test_vpmmeta_roundtrip_weights_only(tmp_path):
    M, D = 12, 256
    metrics = [f"m{i}" for i in range(M)]
    scores = _mk_scores(M, D, seed=1)

    # Mixed weights
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

    # Serialize and read back
    blob = meta.to_bytes()
    meta2 = VPMMetadata.from_bytes(blob)

    assert meta2.version == 1
    assert meta2.kind == MapKind.VPM
    assert meta2.level == 1
    assert meta2.agg_id == int(AggId.RAW)
    assert meta2.metric_count == M
    assert meta2.doc_count == D
    assert meta2.doc_block_size == 1
    assert meta2.task_hash == 0x12345678
    assert meta2.tile_id == tile_id
    assert len(meta2.pointers) == 0

    # Validate weights after 4-bit quantization
    decoded = meta2.get_weights(metrics)
    for m in metrics:
        orig = task_weights.get(m, 0.0)
        # 4-bit quantization tolerance: value = round(orig*15)/15
        expect = round(max(0.0, min(1.0, orig)) * 15) / 15.0
        assert abs(decoded[m] - expect) <= (1/15 + 1e-9)

    # Optional: embed into PNG if your writer supports metadata_bytes
    out = tmp_path / "vpm_meta_weights_only.png"
    VPMImageWriter(
        score_matrix=scores,
        store_minmax=True,
        compression=6,
        level=1,
        doc_block_size=1,
        agg_id=int(AggId.RAW),
        metadata_bytes=blob,
    ).write(str(out))
    r = VPMImageReader(str(out))
    raw = r.read_metadata_bytes()
    roundtrip = VPMMetadata.from_bytes(raw)
    assert roundtrip.tile_id == tile_id

def test_vpmmeta_roundtrip_with_pointers_spillover(tmp_path):
    D = 32      # narrow width so your PNG header has limited first-row capacity
    M = 200     # enough metrics so metadata spills to extra rows
    metrics = [f"m{i}" for i in range(M)]
    scores = _mk_scores(M, D, seed=7)

    # Dense weights
    rng = np.random.default_rng(123)
    dense_weights = {m: float(rng.random()) for m in metrics}

    # Create enough pointers to make payload sizeable
    ptrs = []
    for i in range(24):
        tid = VPMMetadata.make_tile_id(f"child-{i}".encode())
        ptrs.append(RouterPointer(
            kind=MapKind.VPM,
            level=2,
            x_offset=i * 1000,
            span=1000,
            doc_block_size=1,
            agg_id=int(AggId.MAX),
            tile_id=tid
        ))

    tile_id = VPMMetadata.make_tile_id(b"tile:pointers")
    meta = VPMMetadata.for_tile(
        level=3, metric_count=M, doc_count=D,
        doc_block_size=1, agg_id=int(AggId.RAW),
        metric_weights=dense_weights, metric_names=metrics,
        task_hash=0xDEADBEEF, tile_id=tile_id
    )
    for p in ptrs:
        meta.add_pointer(p)

    blob = meta.to_bytes()
    back = VPMMetadata.from_bytes(blob)

    assert back.level == 3
    assert back.metric_count == M
    assert back.doc_count == D
    assert back.task_hash == 0xDEADBEEF
    assert len(back.pointers) == len(ptrs)
    for a, b in zip(ptrs, back.pointers):
        assert a.kind == b.kind
        assert a.level == b.level
        assert a.x_offset == b.x_offset
        assert a.span == b.span
        assert a.doc_block_size == b.doc_block_size
        assert a.agg_id == b.agg_id
        assert a.tile_id == b.tile_id

    # Optional: PNG embedding round-trip
    out = tmp_path / "vpm_meta_pointers_spill.png"
    VPMImageWriter(
        score_matrix=scores,
        store_minmax=False,
        compression=6,
        level=3,
        doc_block_size=1,
        agg_id=int(AggId.RAW),
        metadata_bytes=blob,
    ).write(str(out))
    r = VPMImageReader(str(out))
    raw = r.read_metadata_bytes()
    back2 = VPMMetadata.from_bytes(raw)
    assert len(back2.pointers) == 24

def test_vpmmeta_absent_bytes(tmp_path):
    M, D = 6, 64
    metrics = [f"m{i}" for i in range(M)]
    scores = _mk_scores(M, D, seed=5)

    out = tmp_path / "vpm_no_meta_v2.png"
    VPMImageWriter(
        score_matrix=scores,
        store_minmax=False,
        compression=6,
        level=1,
        doc_block_size=1,
        agg_id=int(AggId.RAW),
        metadata_bytes=b"",  # explicitly none
    ).write(str(out))

    r = VPMImageReader(str(out))
    raw = r.read_metadata_bytes()
    assert raw == b""

    # Decode behavior with empty bytes:
    # If you adopted default=0.5 in get_weights, uncomment that path below.
    meta = VPMMetadata.from_bytes(VPMMetadata().to_bytes())  # minimal default
    # choose your convention:
    # defaults = meta.get_weights(metrics, default=0.5)
    defaults = meta.get_weights(metrics, default=0.0)
    for m in metrics:
        assert m in defaults
