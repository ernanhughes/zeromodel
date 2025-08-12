# tests/test_vpm_logic_diagonal_demo.py
"""
Visual demo + assertions for pixel-wise VPM logic with simple diagonal masks.

We create:
- A: upper-triangular (including diagonal) = 1, else 0
- B: lower-triangular (excluding diagonal) = 1, else 0  (i.e., opposite half)

Then compute:
- AND, OR, NOT(A), NOR(A,B), XOR(A,B)
and save all as grayscale images for quick inspection.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import your pixel-wise ops
from zeromodel.vpm.logic import (
    vpm_and, vpm_or, vpm_not, vpm_xor, vpm_nor
)

OUTDIR = "images/logic_demo"
os.makedirs(OUTDIR, exist_ok=True)


def _save(img: np.ndarray, title: str, fname: str):
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis("off")
    plt.colorbar()
    path = os.path.join(OUTDIR, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"wrote {path}")


def _mk_diagonal_masks(n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    A: upper-triangular (including diag) -> 1.0, else 0.0
    B: lower-triangular (strictly below diag) -> 1.0, else 0.0
    Shapes: (n, n) float32 with values in {0.0, 1.0}
    """
    A = np.triu(np.ones((n, n), dtype=np.float32), k=0)       # diag + above
    B = np.tril(np.ones((n, n), dtype=np.float32), k=-1)      # strictly below
    return A, B


def test_vpm_logic_diagonal_demo():
    # 1) build A and B
    A, B = _mk_diagonal_masks(256)

    # 2) save sources
    _save(A, "Image A (upper half set)", "diag_A.png")
    _save(B, "Image B (lower half set)", "diag_B.png")

    # 3) logic ops
    C_and = vpm_and(A, B)
    C_or  = vpm_or(A, B)
    C_notA = vpm_not(A)
    C_nor = vpm_nor(A, B)        # NOT(OR(A,B))
    C_xor = vpm_xor(A, B)

    # 4) save results
    _save(C_and, "A AND B (should be all zeros except maybe diag overlap)", "diag_AND.png")
    _save(C_or,  "A OR  B (should be almost all ones)",                      "diag_OR.png")
    _save(C_notA,"NOT A (lower half incl. below diag)",                      "diag_NOT_A.png")
    _save(C_nor, "NOR(A,B) = NOT(OR) (should be near all zeros)",            "diag_NOR.png")
    _save(C_xor, "A XOR B (should be ones everywhere except diagonal)",      "diag_XOR.png")

    # 5) assertions — sanity checks on counts/patterns
    n = A.shape[0]

    # A is upper-tri incl diag: ones = n(n+1)/2
    ones_A = int(A.sum())
    assert ones_A == n*(n+1)//2

    # B is strictly lower: ones = n(n-1)/2
    ones_B = int(B.sum())
    assert ones_B == n*(n-1)//2

    # AND of disjoint halves should be zeros everywhere (upper vs strictly lower)
    assert np.count_nonzero(C_and) == 0

    # OR should be ones everywhere except where both were zero — here, that’s nowhere
    # Because A covers diag+upper and B covers strictly lower; union covers full matrix.
    assert np.allclose(C_or, 1.0)

    # NOT(A) should be 1.0 in strictly-lower region, 0.0 in diag+upper
    assert np.allclose(C_notA + A, 1.0)

    # NOR = NOT(OR). Since OR is all ones, NOR should be all zeros.
    assert np.count_nonzero(C_nor) == 0

    # XOR should be ones everywhere except on the diagonal (where A=1 and B=0, still 1)
    # Wait: on the diagonal, A=1, B=0 -> XOR==1, so XOR should also be all ones.
    # But B excludes the diagonal, so there is no overlap anywhere; XOR==1 everywhere.
    assert np.allclose(C_xor, 1.0)

    # shape & dtype checks (normalized float assumed)
    for img in (A, B, C_and, C_or, C_notA, C_nor, C_xor):
        assert img.shape == (n, n)
        assert img.dtype in (np.float32, np.float64)
        assert img.min() >= 0.0 and img.max() <= 1.0


def test_vpm_logic_diagonal_grid():
    A, B = _mk_diagonal_masks(256)

    results = {
        "A (upper)": A,
        "B (lower)": B,
        "A AND B": vpm_and(A, B),
        "A OR B": vpm_or(A, B),
        "NOT A": vpm_not(A),
        "NOR(A,B)": vpm_nor(A, B),
        "A XOR B": vpm_xor(A, B)
    }

    # --- build figure ---
    fig, axes = plt.subplots(1, len(results), figsize=(3 * len(results), 3))
    for ax, (title, img) in zip(axes, results.items()):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTDIR, "logic_demo_grid.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Wrote {out_path}")

    # --- quick shape/dtype sanity ---
    for img in results.values():
        assert img.shape == A.shape
        assert img.dtype in (np.float32, np.float64)
