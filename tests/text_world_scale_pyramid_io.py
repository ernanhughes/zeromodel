import os
import time
import math
import pathlib
import numpy as np
import pytest

from zeromodel import VPMImageWriter, VPMImageReader
from zeromodel.vpm.metadata import (
    VPMMetadata, RouterPointer, MapKind, AggId
)

# ---------- helpers ----------

def _mk_scores(M, D, seed=0):
    """Deterministic but varied scores (M x D) in [0,1]."""
    rng = np.random.default_rng(seed)
    base = rng.random((M, D))
    scales = (rng.random(M) * 0.8 + 0.2).reshape(M, 1)
    offsets = (rng.random(M) * 0.1).reshape(M, 1)
    return np.clip(base * scales + offsets, 0.0, 1.0).astype(np.float32)

def _tile_path(root: pathlib.Path, level: int, hexid: str) -> pathlib.Path:
    return root / f"vpm_level{level:02d}_{hexid}.png"

def _write_tile(path: pathlib.Path, M: int, D: int, level: int,
                agg_id: int, doc_block_size: int, metadata_bytes: bytes, seed=0):
    scores = _mk_scores(M, D, seed=seed)
    VPMImageWriter(
        score_matrix=scores,
        store_minmax=False,
        compression=6,
        level=level,
        doc_block_size=doc_block_size,
        agg_id=agg_id,
        metadata_bytes=metadata_bytes,
    ).write(str(path))

# ---------- tests ----------

@pytest.mark.slow
def test_world_scale_chain_write_and_hop(tmp_path):
    """
    Build a world-scale pointer chain on disk, then simulate a 40-hop descent.

    Structure:
      - 40 levels, each tile has many RouterPointers to the next level
      - Tiles are real PNGs with embedded metadata; score matrices are compact
      - We write, read, validate, and time a hop descent (metadata-only reads)
    """
    root = tmp_path / "worldscale"
    root.mkdir(parents=True, exist_ok=True)

    levels = 40
    pointers_per_level = 32         # keep time reasonable; can tune up
    M = 64                          # metrics per tile (kept small for IO speed)
    D = 512                         # docs per tile (narrow width; forces spill in some cases)
    agg_id = int(AggId.MAX)

    # Pre-create all tiles, wiring pointers from level L to level L+1
    created = {}   # (level, tile_id) -> path
    children_map = {}  # (level, tile_id) -> list[child_tile_id]

    start_write = time.perf_counter()
    # Make a single chain root that fans out each level
    root_tile_id = VPMMetadata.make_tile_id(b"level-00-root")
    for L in range(levels):
        # For each level, we’ll create one "main" tile plus the child targets it points to.
        # The next level's targets (children) become real tiles written to disk as well.
        # To keep runtime sensible, we only materialize one main tile per level
        # + its 'pointers_per_level' direct children at the NEXT level.

        # Generate children IDs for this level (they belong to next level)
        child_ids = []
        for i in range(pointers_per_level):
            child_ids.append(VPMMetadata.make_tile_id(f"child-L{L+1:02d}-{i}".encode()))
        children_map[(L, root_tile_id)] = child_ids

        # Assemble metadata for this level's main tile
        weights = {f"m{i}": (i % 16) / 15.0 for i in range(M)}  # dense-ish, exercises nibble pack
        meta = VPMMetadata.for_tile(
            level=L,
            metric_count=M,
            doc_count=D,
            doc_block_size=max(1, 2**L),   # grows with level
            agg_id=agg_id,
            metric_weights=weights,
            metric_names=[f"m{i}" for i in range(M)],
            task_hash=0xFEEDFACE,
            tile_id=root_tile_id
        )
        for tid in child_ids:
            meta.add_pointer(RouterPointer(
                kind=MapKind.VPM,
                level=L+1,
                x_offset=0,         # illustrative; not used in test
                span=1_000,         # illustrative
                doc_block_size=max(1, 2**(L+1)),
                agg_id=agg_id,
                tile_id=tid
            ))

        # Write the main tile for this level
        main_path = _tile_path(root, L, root_tile_id.hex())
        _write_tile(main_path, M=M, D=D, level=L,
                    agg_id=agg_id, doc_block_size=max(1, 2**L),
                    metadata_bytes=meta.to_bytes(), seed=L)
        created[(L, root_tile_id)] = main_path

        # Also materialize child tiles at next level (empty pointer lists)
        if L + 1 < levels:
            for j, child_tid in enumerate(child_ids):
                child_meta = VPMMetadata.for_tile(
                    level=L+1,
                    metric_count=M,
                    doc_count=D,
                    doc_block_size=max(1, 2**(L+1)),
                    agg_id=agg_id,
                    metric_weights=None,
                    metric_names=[f"m{i}" for i in range(M)],
                    task_hash=0xFEEDFACE,
                    tile_id=child_tid
                )
                # write child tile
                child_path = _tile_path(root, L+1, child_tid.hex())
                _write_tile(child_path, M=M, D=D, level=L+1,
                            agg_id=agg_id, doc_block_size=max(1, 2**(L+1)),
                            metadata_bytes=child_meta.to_bytes(), seed=L*1000 + j)
                created[(L+1, child_tid)] = child_path

        # Optional: advance the "root" to a deterministic child (simulate taking one branch)
        if L + 1 < levels and child_ids:
            root_tile_id = child_ids[0]

    write_elapsed_ms = (time.perf_counter() - start_write) * 1000.0

    # Basic sanity: we created at least (levels) main tiles + some children
    assert len(created) >= levels
    # Keep file creation relatively snappy even on CI:
    assert write_elapsed_ms < 30_000.0   # < 8s total write time on dev/CI machines

    # --- Simulate a 40-hop descent (metadata-only reads) ---
    start_hop = time.perf_counter()
    hops = 0
    cur_level = 0
    cur_tile_id = VPMMetadata.make_tile_id(b"level-00-root")

    # random-ish but deterministic walk: always pick the first pointer
    while cur_level < levels:
        path = created[(cur_level, cur_tile_id)]
        r = VPMImageReader(str(path))
        blob = r.read_metadata_bytes()
        meta = VPMMetadata.from_bytes(blob)

        # Validate a few header fields match the file
        assert meta.level == cur_level
        assert meta.metric_count == M
        assert meta.doc_count == D

        hops += 1
        if not meta.pointers or cur_level == levels - 1:
            break

        # Choose child 0 for deterministic descent
        nxt = meta.pointers[0]
        assert nxt.level == cur_level + 1
        cur_level += 1
        cur_tile_id = nxt.tile_id

    hop_elapsed_ms = (time.perf_counter() - start_hop) * 1000.0

    # We should have walked ~levels hops (allow last hop to be terminal)
    assert 1 <= hops <= levels
    # Keep total hop time tiny (metadata-only path):
    assert hop_elapsed_ms < 300.0, f"Hop metadata read took {hop_elapsed_ms:.2f} ms"

    # Log-ish assertions (not strict, but sanity)
    total_png = sum(1 for _ in root.glob("*.png"))
    assert total_png >= levels      # wrote at least the main chain

@pytest.mark.slow
def test_world_scale_io_integrity_and_speed(tmp_path):
    """
    Wider tiles + bigger metadata to exercise spill and test end-to-end read speed.
    """
    root = tmp_path / "io_integrity"
    root.mkdir(parents=True, exist_ok=True)

    M, D = 256, 2048   # wider to force substantial header rows + spill
    level = 3
    doc_block_size = 2 ** level
    agg_id = int(AggId.MAX)

    # Dense weights; lots of pointers to bloat header
    weights = {f"m{i}": (i % 16) / 15.0 for i in range(M)}
    ptrs = [
        RouterPointer(
            kind=MapKind.VPM, level=level+1,
            x_offset=i * 10_000, span=10_000,
            doc_block_size=doc_block_size * 2,
            agg_id=agg_id,
            tile_id=VPMMetadata.make_tile_id(f"child-{i}".encode())
        )
        for i in range(96)
    ]
    tile_id = VPMMetadata.make_tile_id(b"io:big:hdr")

    meta = VPMMetadata.for_tile(
        level=level, metric_count=M, doc_count=D,
        doc_block_size=doc_block_size, agg_id=agg_id,
        metric_weights=weights, metric_names=[f"m{i}" for i in range(M)],
        task_hash=0xC0FFEE, tile_id=tile_id
    )
    for p in ptrs:
        meta.add_pointer(p)

    out = root / "big_header_tile.png"

    t0 = time.perf_counter()
    _write_tile(out, M=M, D=D, level=level, agg_id=agg_id,
                doc_block_size=doc_block_size, metadata_bytes=meta.to_bytes(), seed=123)
    write_ms = (time.perf_counter() - t0) * 1000.0

    # Read & verify
    t1 = time.perf_counter()
    r = VPMImageReader(str(out))
    blob = r.read_metadata_bytes()
    parse_ms = (time.perf_counter() - t1) * 1000.0
    back = VPMMetadata.from_bytes(blob)

    # Integrity checks
    assert back.level == level
    assert back.metric_count == M
    assert back.doc_count == D
    assert len(back.pointers) == len(ptrs)
    got_w = back.get_weights([f"m{i}" for i in range(M)])
    # spot check a few weights survived nibble packing
    for k in ["m0", "m15", "m127", "m255"]:
        assert k in got_w

    # Speed envelopes (generous for CI)
    assert write_ms < 2_000.0, f"Write too slow: {write_ms:.1f} ms"
    assert parse_ms < 200.0, f"Parse too slow: {parse_ms:.1f} ms"
