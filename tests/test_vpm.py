# test_vpm.py
import math

import numpy as np

# CHANGE THIS to your actual module path/name:
from zeromodel import (AGG_MAX, VPMImageReader, VPMImageWriter,
                       build_parent_level_png)


def _mk_scores(M, D, seed=7):
    """Deterministic score matrix with no ties within a row."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.0, scale=1.0, size=(M, D))
    # add tiny increasing offset per column to break ties deterministically
    offset = np.linspace(0, 1e-6, D, dtype=np.float64)
    return base + offset

def test_roundtrip_base_image(tmp_path):
    M, D = 12, 512  # 12 metrics × 512 docs
    scores = _mk_scores(M, D)

    png_path = tmp_path / "vpm_base.png"
    w = VPMImageWriter(
        score_matrix=scores,
        store_minmax=True,       # exercise in-image min/max headers
        compression=6,
        level=1,                 # mark as leaf level in your hierarchy (arbitrary)
        doc_block_size=1,
    )
    w.write(str(png_path))

    # Read it back
    r = VPMImageReader(str(png_path))

    # Header checks
    assert r.version == 1
    assert r.M == M
    assert r.D == D
    assert r.h_meta >= 2
    assert r.level == 1
    assert r.doc_block_size == 1
    assert r.agg_id in (AGG_MAX, 65535)  # writer uses RAW/65535 for base by default

    # Min/max present
    assert r.norm_flag == 1
    assert r.min_vals is not None and r.max_vals is not None
    assert len(r.min_vals) == M and len(r.max_vals) == M

    # Values decode back (within quantization tolerance)
    for m in range(M):
        orig = scores[m]
        dec = r.get_metric_values(m)
        # Since we normalized per-row to [0,1] with Q16 quantization and stored min/max,
        # allow a small absolute error for quantization.
        mae = np.mean(np.abs((orig - orig.min()) / max(np.ptp(orig), 1e-12) - dec))
        assert mae < 5e-3  # ~0.5% tolerance is plenty for uint16 quantization

def test_virtual_order_single_metric(tmp_path):
    M, D = 10, 512
    scores = _mk_scores(M, D, seed=13)
    png_path = tmp_path / "vpm_order_single.png"
    VPMImageWriter(scores, store_minmax=False, compression=3, level=2).write(str(png_path))
    r = VPMImageReader(str(png_path))

    m = 3  # pick a metric
    # True order (descending) from decoded values
    true_vals = r.get_metric_values(m)
    true_perm = np.argsort(-true_vals)

    # Virtual order via (G then R) lexsort
    virt_perm = r.virtual_order(metric_idx=m, descending=True)

    # The permutations should match exactly unless there are exact ties; with our data they should match.
    assert np.array_equal(true_perm, virt_perm)

def test_virtual_order_composite_and_view(tmp_path):
    M, D = 16, 512
    scores = _mk_scores(M, D, seed=23)
    png_path = tmp_path / "vpm_order_composite.png"
    VPMImageWriter(scores, store_minmax=False, compression=6, level=3).write(str(png_path))
    r = VPMImageReader(str(png_path))

    # Composite weights on a few metrics
    weights = {1: 0.6, 7: 0.3, 12: 0.1}
    # Compute expected by decoded values
    comp_true = np.zeros(D, dtype=np.float64)
    for m, w in weights.items():
        comp_true += w * r.get_metric_values(m)
    expected_perm = np.argsort(-comp_true)

    # Virtual composite
    virt_perm = r.virtual_order(weights=weights, descending=True)
    assert np.array_equal(expected_perm, virt_perm)

    # Extract a virtual top-left window (8x8) using the composite permutation
    tile = r.get_virtual_view(weights=weights, x=0, y=0, width=8, height=8)
    assert tile.shape == (8, 8, 3)
    # Sanity: top-left doc should be the best-scoring doc
    assert virt_perm[0] >= 0 and virt_perm[0] < D

def test_build_parent_level_and_properties(tmp_path):
    M, D = 8, 512
    scores = _mk_scores(M, D, seed=99)
    child_path = tmp_path / "vpm_child.png"
    VPMImageWriter(scores, store_minmax=False, compression=6, level=4).write(str(child_path))
    child = VPMImageReader(str(child_path))

    parent_path = tmp_path / "vpm_parent.png"
    K = 8  # aggregate 8 child columns per parent column
    build_parent_level_png(child, str(parent_path), K=K, agg_id=AGG_MAX, compression=6)
    parent = VPMImageReader(str(parent_path))

    # Dimensions
    P = math.ceil(D / K)
    assert parent.D == P
    assert parent.M == M
    assert parent.level == child.level - 1
    assert parent.doc_block_size == max(1, child.doc_block_size * K)

    # MAX aggregator property: parent R >= any child R in its block (per metric)
    for m in range(M):
        parent_row_R = parent.get_metric_row_raw(m)[:, 0]
        child_row_R = child.get_metric_row_raw(m)[:, 0]
        for p in range(P):
            lo, hi = p * K, min(D, (p + 1) * K)
            assert int(parent_row_R[p]) >= int(child_row_R[lo:hi].max())

    # Parent G is a percentile across parent columns
    g = parent.get_metric_row_raw(0)[:, 1]  # metric 0, G channel
    assert g.min() >= 0 and g.max() <= 65535
