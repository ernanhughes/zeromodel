import numpy as np
import pytest

from zeromodel.core import ZeroModel
from zeromodel.vpm.explain import OcclusionVPMInterpreter
from zeromodel.vpm.encoder import VPMEncoder


def build_synthetic_matrix(H=20, K=9):
    """
    Build a synthetic score matrix for VPM testing.

    - Shape: H documents x (3*K) metrics.
    - Values: small random noise + strong signal in the top-left block.
    - The signal is injected into the first 5 rows and the first K pixels-worth
      of metrics (i.e., 3*K metrics for RGB → K visible pixels).
    - This simulates a case where the most important region is top-left.
    """
    M = 3 * K  # number of metric columns
    mat = np.random.rand(H, M) * 0.1
    # Inject strong signal in top-left window
    mat[:5, :M] += 0.8
    mat = np.clip(mat, 0, 1)
    names = [f"m{i}" for i in range(M)]
    return mat, names


def test_interpreter_highlights_top_left_region():
    """
    Test 1: Baseline occlusion test for top-left prior.
    ---------------------------------------------------
    We generate a matrix where the most important features are in the top-left.
    Using prior="top_left", the interpreter's importance map should show a
    higher average importance in the top-left region than in the bottom-right.
    """
    score, names = build_synthetic_matrix(H=24, K=6)
    zm = ZeroModel(names)
    zm.prepare(score, "SELECT * FROM virtual_index ORDER BY m0 DESC")

    interp = OcclusionVPMInterpreter(
        patch_h=2, patch_w=2, stride=1,
        baseline="zero",
        prior="top_left"
    )
    imp, meta = interp.explain(zm)

    # Sanity: importance map same height/width as VPM
    # Use encoder to get VPM shape without calling deprecated encode
    vpm = VPMEncoder('float32').encode(zm.sorted_matrix)
    assert imp.shape == vpm.shape[:2]

    # Expect higher mean importance near the top-left than bottom-right
    H, W = imp.shape
    tl = imp[:H//3, :W//3].mean()
    br = imp[-H//3:, -W//3:].mean()
    assert tl > br, "Top-left should be more important in this synthetic setup"


def test_interpreter_invariance_under_constant_shift():
    """
    Test 2: Invariance to uniform brightness shifts.
    -------------------------------------------------
    If we add a constant to all scores (i.e., uniformly brighten the VPM),
    the *relative* importance structure should remain the same.

    We check this by:
      - Generating importance maps before and after adding a constant offset.
      - Computing their correlation — should be high (>0.7).
    """
    score, names = build_synthetic_matrix(H=20, K=5)
    zm1 = ZeroModel(names)
    zm1.prepare(score, "SELECT * FROM virtual_index ORDER BY m0 DESC")
    interp = OcclusionVPMInterpreter(
        patch_h=2, patch_w=2, stride=2,
        baseline="mean"
    )
    imp1, _ = interp.explain(zm1)

    # Add a constant +0.1 to all values; structure unchanged
    score2 = np.clip(score + 0.1, 0, 1)
    zm2 = ZeroModel(names)
    zm2.prepare(score2, "SELECT * FROM virtual_index ORDER BY m0 DESC")
    imp2, _ = interp.explain(zm2)

    corr = np.corrcoef(imp1.flatten(), imp2.flatten())[0, 1]
    assert corr > 0.7


def test_interpreter_detects_moved_hotspot():
    """
    Test 3: Sensitivity to hotspot location.
    ----------------------------------------
    We build two matrices:
      - 'left': hotspot (high scores) in the left third of the VPM.
      - 'right': hotspot in the right third of the VPM.

    We bypass ZeroModel sorting so the spatial layout is preserved exactly.
    The interpreter is run with prior='uniform' to remove positional bias.

    Expectations:
      - For 'left', the mean importance in the left third > right third.
      - For 'right', the mean importance in the right third > left third.
    """
    K = 6
    M = 3 * K  # 18 metrics (divisible by 3 for RGB encoding)
    H = 20

    rng = np.random.default_rng(0)
    base = rng.random((H, M)) * 0.1

    # Left-hotspot version
    left = base.copy()
    left[:5, :M//3] += 0.8     # LEFT third hotspot (first 6 metrics)
    left = np.clip(left, 0, 1)

    # Right-hotspot version
    right = base.copy()
    right[:5, -M//3:] += 0.8   # RIGHT third hotspot (last 6 metrics)
    right = np.clip(right, 0, 1)

    names = [f"m{i}" for i in range(M)]

    # --- Bypass ZeroModel sorting so spatial position is preserved ---
    zmL = ZeroModel(names)
    zmL.sorted_matrix = left
    zmL.doc_order = np.arange(H)
    zmL.metric_order = np.arange(M)

    zmR = ZeroModel(names)
    zmR.sorted_matrix = right
    zmR.doc_order = np.arange(H)
    zmR.metric_order = np.arange(M)
    # ---------------------------------------------------------------

    # FIXED: Use proper patch width that accounts for RGB encoding
    # Since 3 metrics = 1 pixel in VPM, we need patch_w=2 to cover 6 metrics
    interp = OcclusionVPMInterpreter(
        patch_h=2, 
        patch_w=2,  # Now correctly covers 6 metrics (2 pixels × 3 channels)
        stride=1, 
        baseline="mean",  # FIXED: Changed from "zero" to "mean" for better baseline
        prior="uniform"
    )
    
    impL, _ = interp.explain(zmL)
    impR, _ = interp.explain(zmR)

    Hh, Ww = impL.shape
    # FIXED: Calculate thirds based on metric positions, not pixel positions
    # Since 3 metrics = 1 pixel, the left/right thirds in metrics = left/right 2 pixels
    left_third_pixels = Ww // 3
    right_third_pixels = Ww // 3
    
    L_mean = impL[:, :left_third_pixels].mean()
    R_mean = impL[:, -right_third_pixels:].mean()
    assert L_mean > R_mean, f"Left hotspot not detected: L={L_mean:.6f}, R={R_mean:.6f}"

    L_mean2 = impR[:, :left_third_pixels].mean()
    R_mean2 = impR[:, -right_third_pixels:].mean()
    # FIXED: Added more informative error message
    assert R_mean2 > L_mean2, f"Right hotspot not detected properly: L={L_mean2:.6f}, R={R_mean2:.6f}"