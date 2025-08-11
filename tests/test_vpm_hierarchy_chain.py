# tests/test_vpm_hierarchy_chain.py
import numpy as np
import pytest

from zeromodel import VPMImageWriter, VPMImageReader
from zeromodel.vpm.image import build_parent_level_png, AGG_MAX, AGG_MEAN  # adjust path if needed

def _mk_scores(M, D, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.random((M, D))
    scales = (rng.random(M) * 3.0 + 0.5).reshape(M, 1)
    offsets = (rng.random(M) * 1.5).reshape(M, 1)
    return base * scales + offsets

@pytest.mark.parametrize("agg_id", [AGG_MAX, AGG_MEAN])
@pytest.mark.parametrize("K", [2, 4, 8])  # docs per block
def test_hierarchy_chain_until_single_column(tmp_path, agg_id, K):
    M, D = 12, 512
    scores = _mk_scores(M, D, seed=123)

    # --- Build leaf (child) ---
    leaf_path = tmp_path / "leaf.png"
    VPMImageWriter(
        score_matrix=scores,
        store_minmax=False,
        compression=3,
        level=10,            # arbitrary leaf level for visibility
        doc_block_size=1,
        agg_id=65535,        # RAW at leaf
    ).write(str(leaf_path))

    # --- Chain up until width becomes 1 ---
    chain = []
    child = VPMImageReader(str(leaf_path))
    chain.append(("leaf", child))

    lvl = child.level
    while child.D > 1:
        parent_path = tmp_path / f"parent_{len(chain)}.png"
        build_parent_level_png(child, str(parent_path), K=K, agg_id=agg_id, compression=3)
        parent = VPMImageReader(str(parent_path))
        chain.append((f"p{len(chain)-1}", parent))
        child = parent

    # Basic sanity: last level has width 1, level decreased, block size multiplied
    names, readers = zip(*chain)
    leaf = readers[0]
    top  = readers[-1]
    assert top.D == 1
    assert top.level == leaf.level - (len(chain)-1)
    assert top.doc_block_size == leaf.doc_block_size * (K ** (len(chain)-1))
    assert top.agg_id == agg_id

    # --- Verify aggregation correctness level-by-level ---
    for i in range(len(chain)-1):
        _, child = chain[i]
        _, parent = chain[i+1]
        # child data
        R_child = child.image[child.h_meta:, :, 0].astype(np.uint16)
        # parent data
        R_parent = parent.image[parent.h_meta:, :, 0].astype(np.uint16)
        B_parent = parent.image[parent.h_meta:, :, 2].astype(np.uint16)  # aux/argmax

        P = parent.D
        for p in range(P):
            lo = p * K
            hi = min(child.D, lo + K)
            block = R_child[:, lo:hi]  # (M, block_size)

            if agg_id == AGG_MEAN:
                expect = np.round(block.mean(axis=1)).astype(np.uint16)
                assert np.allclose(R_parent[:, p], expect, atol=1)
            else:  # AGG_MAX
                vmax = block.max(axis=1)
                assert np.array_equal(R_parent[:, p], vmax)

                # Use B to navigate back to child column and check equality
                if hi - lo > 1:
                    rel_pos = (B_parent[:, p].astype(np.float64) / 65535.0) * (hi - lo - 1)
                    sel = lo + np.rint(rel_pos).astype(int)
                else:
                    sel = np.full(R_parent.shape[0], lo, dtype=int)
                picked = R_child[np.arange(M), sel]
                assert np.array_equal(picked, vmax)

def test_follow_argmax_across_multiple_levels(tmp_path):
    """Pick a random parent column at a mid level and walk down to the leaf,
    verifying argmax correspondence at each hop."""
    M, D, K = 10, 256, 4
    scores = _mk_scores(M, D, seed=7)

    leaf_path = tmp_path / "leaf.png"
    VPMImageWriter(scores, store_minmax=False, compression=3, level=20,
                   doc_block_size=1, agg_id=65535).write(str(leaf_path))
    chain = []
    child = VPMImageReader(str(leaf_path))
    chain.append(child)
    # build a few levels
    for _ in range(5):
        parent_path = tmp_path / f"p{_}.png"
        build_parent_level_png(chain[-1], str(parent_path), K=K, agg_id=AGG_MAX, compression=3)
        chain.append(VPMImageReader(str(parent_path)))

    # pick a mid parent (not the top) and a random column
    parent = chain[-2]
    child  = chain[-3]
    p = np.random.default_rng(0).integers(0, parent.D)

    R_child = child.image[child.h_meta:, :, 0].astype(np.uint16)
    R_parent = parent.image[parent.h_meta:, :, 0].astype(np.uint16)
    B_parent = parent.image[parent.h_meta:, :, 2].astype(np.uint16)

    lo = p * K
    hi = min(child.D, lo + K)
    block = R_child[:, lo:hi]
    vmax = block.max(axis=1)
    assert np.array_equal(R_parent[:, p], vmax)

    # follow argmax to child doc indices
    if hi - lo > 1:
        rel_pos = (B_parent[:, p].astype(np.float64) / 65535.0) * (hi - lo - 1)
        sel = lo + np.rint(rel_pos).astype(int)
    else:
        sel = np.full(M, lo, dtype=int)
    picked = R_child[np.arange(M), sel]
    assert np.array_equal(picked, vmax)
