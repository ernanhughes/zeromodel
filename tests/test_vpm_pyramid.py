# tests/test_vpm_pyramid.py

import math
import numpy as np
import pytest

from zeromodel.vpm.image import VPMImageWriter, VPMImageReader, _round_u16, _u16_clip
from zeromodel.vpm.pyramid import VPMPyramidBuilder
from zeromodel.vpm.metadata import VPMMetadata, RouterPointer, MapKind, AggId


def _mk_scores(M, D, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((M, D))
    scales = (rng.random(M) * 4.0 + 0.5).reshape(M, 1)  # [0.5..4.5]
    offsets = (rng.random(M) * 2.0).reshape(M, 1)       # [0..2]
    return base * scales + offsets


def _expected_percentiles_u16(R_u16_row: np.ndarray) -> np.ndarray:
    P = R_u16_row.shape[0]
    if P == 1:
        return np.array([32767], dtype=np.uint16)
    ranks = np.argsort(np.argsort(R_u16_row, axis=0), axis=0)
    return _round_u16((ranks / (P - 1)) * 65535.0).astype(np.uint16)


def _build_child(tmp_path, M=7, D=103, level=5, doc_block=1, store_minmax=False, seed=1):
    scores = _mk_scores(M, D, seed=seed)
    out = tmp_path / "child.png"
    # IMPORTANT: keep args in sync with the current writer signature
    VPMImageWriter(
        score_matrix=scores,
        store_minmax=store_minmax,
        compression=3,
        level=level,
        doc_block_size=doc_block,
        agg_id=int(AggId.RAW),
        # no metric_ids / doc_ids / metadata_bytes in this writer
    ).write(str(out))
    return str(out)


def test_pyramid_max_parent_roundtrip(tmp_path):
    M, D, K = 9, 113, 8
    child_path = _build_child(tmp_path, M=M, D=D, level=7, doc_block=2, seed=7)
    child = VPMImageReader(child_path)

    # Make a few pointers to exercise pointer payload, even if writer ignores it
    ptrs = []
    for i in range(4):
        tid = VPMMetadata.make_tile_id(f"child-{i}".encode())
        ptrs.append(RouterPointer(
            kind=MapKind.VPM,
            level=child.level - 1,
            x_offset=i * 1000,
            span=1000,
            doc_block_size=child.doc_block_size,
            agg_id=int(AggId.MAX),
            tile_id=tid
        ))

    parent_tile_id = VPMMetadata.make_tile_id(b"parent:MAX")
    meta = VPMMetadata.for_tile(
        level=child.level - 1,
        metric_count=M,
        doc_count=(D + K - 1)//K,
        doc_block_size=child.doc_block_size * K,
        agg_id=int(AggId.MAX),
        metric_weights={},
        metric_names=[f"m{i}" for i in range(M)],
        task_hash=0xA5A5A5A5,
        tile_id=parent_tile_id,
        parent_id=getattr(child, "tile_id", b"\x00"*16) or b"\x00"*16,
    )
    for p in ptrs:
        meta.add_pointer(p)
    meta_bytes = meta.to_bytes()

    out_parent = tmp_path / "parent_max.png"
    builder = VPMPyramidBuilder(K=K, agg_id=int(AggId.MAX), compression=3)
    M_parent, P_parent = builder.build_parent(
        child,
        str(out_parent),
        metadata=meta,
        metric_names=[f"m{i}" for i in range(M)],
    )
    assert M_parent == M
    assert P_parent == math.ceil(D / K)

    parent = VPMImageReader(str(out_parent))
    assert parent.M == M
    assert parent.D == P_parent
    assert parent.level == child.level - 1
    assert parent.doc_block_size == child.doc_block_size * K
    assert parent.agg_id == int(AggId.MAX)

    # Metadata parity only if writer supports embedding via aux hook.
    if hasattr(VPMImageWriter, "write_with_channels") and hasattr(parent, "read_metadata_bytes"):
        got_meta = parent.read_metadata_bytes()
        assert got_meta == meta_bytes
    elif hasattr(parent, "read_metadata_bytes"):
        # Fallback path: no aux method, so metadata isn't embedded.
        assert parent.read_metadata_bytes() in (b"", None)

    # Numeric channel checks (MAX)
    R_child = child.image[child.h_meta:, :, 0].astype(np.uint16)
    R_parent = parent.image[parent.h_meta:, :, 0].astype(np.uint16)
    G_parent = parent.image[parent.h_meta:, :, 1].astype(np.uint16)
    B_parent = parent.image[parent.h_meta:, :, 2].astype(np.uint16)

    P = P_parent
    exp_R = np.zeros((M, P), dtype=np.uint16)
    exp_B = np.zeros((M, P), dtype=np.uint16)
    for p in range(P):
        lo = p * K
        hi = min(D, lo + K)
        blk = R_child[:, lo:hi]
        vmax = blk.max(axis=1)
        exp_R[:, p] = vmax
        if hi - lo > 1:
            argm = blk.argmax(axis=1)
            exp_B[:, p] = _round_u16((argm / (hi - lo - 1)) * 65535.0)
        else:
            exp_B[:, p] = 0

    np.testing.assert_array_equal(R_parent, exp_R)
    np.testing.assert_array_equal(B_parent, exp_B)

    for m in range(M):
        np.testing.assert_array_equal(G_parent[m], _expected_percentiles_u16(R_parent[m]))


def test_pyramid_mean_parent_roundtrip(tmp_path):
    M, D, K = 6, 64, 4
    child_path = _build_child(tmp_path, M=M, D=D, level=4, doc_block=1, seed=11)
    child = VPMImageReader(child_path)

    out_parent = tmp_path / "parent_mean.png"
    builder = VPMPyramidBuilder(K=K, agg_id=int(AggId.MEAN), compression=6)
    M_parent, P_parent = builder.build_parent(child, str(out_parent))
    assert M_parent == M
    assert P_parent == math.ceil(D / K)

    parent = VPMImageReader(str(out_parent))
    assert parent.M == M
    assert parent.D == P_parent
    assert parent.agg_id == int(AggId.MEAN)

    R_child = child.image[child.h_meta:, :, 0].astype(np.uint16)
    R_parent = parent.image[parent.h_meta:, :, 0].astype(np.uint16)
    G_parent = parent.image[parent.h_meta:, :, 1].astype(np.uint16)
    B_parent = parent.image[parent.h_meta:, :, 2].astype(np.uint16)

    P = P_parent
    exp_R = np.zeros((M, P), dtype=np.uint16)
    for p in range(P):
        lo = p * K
        hi = min(D, lo + K)
        blk = R_child[:, lo:hi].astype(np.float64)
        vmean = np.round(blk.mean(axis=1))
        exp_R[:, p] = _u16_clip(vmean)

    np.testing.assert_array_equal(R_parent, exp_R)
    assert np.all(B_parent == 0)
    for m in range(M):
        np.testing.assert_array_equal(G_parent[m], _expected_percentiles_u16(R_parent[m]))


@pytest.mark.parametrize("levels", [1, 3])
def test_pyramid_build_chain(tmp_path, levels):
    M, D, K = 5, 50, 5
    child_path = _build_child(tmp_path, M=M, D=D, level=9, doc_block=2, seed=5)
    child = VPMImageReader(child_path)

    paths = [str(tmp_path / f"p_{i}.png") for i in range(levels)]
    builder = VPMPyramidBuilder(K=K, agg_id=int(AggId.MAX), compression=3)
    shapes = builder.build_chain(child, paths)

    prev_D = D
    for (m, pD) in shapes:
        assert m == M
        assert pD == math.ceil(prev_D / K)
        prev_D = pD

    last = VPMImageReader(paths[-1])

    # Physical PNG width may be padded; check the logical count from metadata.
    from zeromodel.vpm.metadata import VPMMetadata
    mb = last.read_metadata_bytes()
    md_last = VPMMetadata.from_bytes(mb) if mb else None
    assert md_last is not None
    assert md_last.doc_count == prev_D


def test_pyramid_fallback_direct_png(tmp_path):
    M, D, K = 4, 33, 8
    child_path = _build_child(tmp_path, M=M, D=D, level=3, doc_block=1, seed=3)
    child = VPMImageReader(child_path)

    out_parent = tmp_path / "parent_fallback.png"
    builder = VPMPyramidBuilder(K=K, agg_id=int(AggId.MAX), compression=3)

    # Force fallback: no aux helper usage
    M_parent, P_parent = builder.build_parent(child, str(out_parent), use_writer_aux=False)
    assert M_parent == M
    assert P_parent == math.ceil(D / K)

    parent = VPMImageReader(str(out_parent))
    if hasattr(parent, "read_metadata_bytes"):
        assert parent.read_metadata_bytes() in (b"", None)
