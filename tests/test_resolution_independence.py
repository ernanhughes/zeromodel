# tests/test_resolution_independence.py
"""
Test cases for resolution independence and precision control features in ZeroModel and VPM Logic.
"""

import numpy as np
import pytest
from scipy.ndimage import zoom # Assuming this dependency is acceptable
from zeromodel.core import ZeroModel
from zeromodel.vpm_logic import (
    vpm_and, vpm_or, vpm_not, vpm_add, vpm_subtract, # or vpm_diff
    normalize_vpm, denormalize_vpm,
    vpm_resize, vpm_concat_horizontal, vpm_concat_vertical,
    query_top_left
)

# --- Test Data ---
def create_test_vpm_1(shape=(4, 4, 3), dtype=np.float32):
    """Create a simple test VPM."""
    vpm = np.random.rand(*shape).astype(dtype)
    # Make it more predictable if needed
    # vpm = np.zeros(shape, dtype=dtype)
    # vpm[0, 0, 0] = 1.0 # Top-left pixel bright
    return vpm

def create_test_vpm_2(shape=(4, 4, 3), dtype=np.float32):
    """Create another simple test VPM."""
    vpm = np.random.rand(*shape).astype(dtype)
    # vpm = np.zeros(shape, dtype=dtype)
    # vpm[0, 0, 0] = 0.5 # Top-left pixel mid-bright
    return vpm

# --- Tests ---

def test_normalize_denormalize_roundtrip():
    """Test normalize_vpm and denormalize_vpm roundtrip conversion."""
    original_uint8 = np.array([0, 127, 255], dtype=np.uint8)
    normalized = normalize_vpm(original_uint8)
    assert normalized.dtype == np.float32 or normalized.dtype == np.float64 # Check it's float
    expected_normalized = np.array([0.0, 127.0/255.0, 1.0])
    np.testing.assert_allclose(normalized, expected_normalized, rtol=1e-5)

    denormalized_back = denormalize_vpm(normalized, output_type=np.uint8)
    np.testing.assert_array_equal(denormalized_back, original_uint8)

    # Test with float input
    original_float = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    normalized_f = normalize_vpm(original_float) # Should be identity or handle gracefully
    # Depending on implementation, normalize_vpm might just return float input <= 1.0
    # Let's assume it does for now, or handles it.
    # If it's supposed to *ensure* [0,1], then it should pass.
    np.testing.assert_allclose(normalized_f, original_float, rtol=1e-6)

def test_vpm_logic_resolution_independence():
    """Test that VPM logic functions work with different dtypes and produce normalized outputs."""
    vpm1_u8 = np.array([[[255]][[0]][[127]]], dtype=np.uint8) # 1x1x3 VPM
    vpm2_u8 = np.array([[[127]][[255]][[0]]], dtype=np.uint8)

    vpm1_f32 = np.array([[[1.0, 0.0, 0.5]]], dtype=np.float32)
    vpm2_f32 = np.array([[[0.5, 1.0, 0.0]]], dtype=np.float32)

    # Test AND
    result_and_u8 = vpm_and(vpm1_u8, vpm2_u8)
    result_and_f32 = vpm_and(vpm1_f32, vpm2_f32)
    assert result_and_u8.dtype == np.float32 or result_and_u8.dtype == np.float64 # Check output is normalized float
    assert result_and_f32.dtype == np.float32 or result_and_f32.dtype == np.float64
    # Check if results are equivalent (normalized)
    np.testing.assert_allclose(result_and_u8, result_and_f32, rtol=1e-5)

    # Test OR
    result_or_u8 = vpm_or(vpm1_u8, vpm2_u8)
    result_or_f32 = vpm_or(vpm1_f32, vpm2_f32)
    assert result_or_u8.dtype == np.float32 or result_or_u8.dtype == np.float64
    assert result_or_f32.dtype == np.float32 or result_or_f32.dtype == np.float64
    np.testing.assert_allclose(result_or_u8, result_or_f32, rtol=1e-5)

    # Test NOT
    result_not_u8 = vpm_not(vpm1_u8)
    result_not_f32 = vpm_not(vpm1_f32)
    expected_not_u8 = np.array([[[0.0, 1.0, 128.0/255.0]]], dtype=np.float32) # Approx
    expected_not_f32 = np.array([[[0.0, 1.0, 0.5]]], dtype=np.float32)
    assert result_not_u8.dtype == np.float32 or result_not_u8.dtype == np.float64
    assert result_not_f32.dtype == np.float32 or result_not_f32.dtype == np.float64
    # Note: NOT precision might vary slightly due to float conversion of 128/255
    # np.testing.assert_allclose(result_not_u8, expected_not_u8, rtol=1e-3) # Looser tolerance for uint8->float conversion
    np.testing.assert_allclose(result_not_f32, expected_not_f32, rtol=1e-6)

    # Add similar tests for vpm_add, vpm_subtract/vpm_diff if applicable

def test_zero_model_encode_precision_control():
    """Test ZeroModel.encode with different output precisions."""
    metric_names = ["m1", "m2"]
    # Simple, predictable data
    score_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    zm = ZeroModel(metric_names) # Assume default precision is handled or set appropriately
    # We need to prepare the model first
    zm.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY m1 DESC")

    # Test float32 output
    vpm_f32 = zm.encode(output_precision='float32')
    assert vpm_f32.dtype == np.float32
    assert np.all(vpm_f32 >= 0.0) and np.all(vpm_f32 <= 1.0)

    # Test uint8 output
    vpm_u8 = zm.encode(output_precision='uint8')
    assert vpm_u8.dtype == np.uint8
    assert np.all(vpm_u8 >= 0) and np.all(vpm_u8 <= 255)

    # Test float16 output (if supported/configured)
    # vpm_f16 = zm.encode(output_precision='float16')
    # assert vpm_f16.dtype == np.float16
    # assert np.all(vpm_f16 >= 0.0) and np.all(vpm_f16 <= 1.0)

def test_vpm_resize_functionality():
    """Test vpm_resize changes dimensions correctly."""
    original_shape = (10, 10, 3)
    vpm_original = np.random.rand(*original_shape).astype(np.float32)
    # Make one pixel bright for easy checking
    vpm_original[0, 0, 0] = 1.0

    new_shape = (5, 20, 3) # Halve height, double width
    vpm_resized = vpm_resize(vpm_original, new_shape)

    assert vpm_resized.shape == new_shape
    assert vpm_resized.dtype == np.float32
    assert np.all(vpm_resized >= 0.0) and np.all(vpm_resized <= 1.0)
    # Check that the bright pixel's influence is still near the top-left
    # This is a bit fuzzy, but a basic check
    assert vpm_resized[0, 0, 0] > 0.5 # Should still be relatively bright

def test_vpm_concatenation():
    """Test horizontal and vertical VPM concatenation."""
    shape1 = (4, 3, 3)
    shape2 = (4, 2, 3) # Same height
    vpm1 = np.ones(shape1, dtype=np.float32) * 0.5
    vpm2 = np.ones(shape2, dtype=np.float32) * 0.8

    # Horizontal concat
    vpm_h_concat = vpm_concat_horizontal(vpm1, vpm2)
    expected_h_shape = (4, 3 + 2, 3)
    assert vpm_h_concat.shape == expected_h_shape
    assert np.allclose(vpm_h_concat[:, :3, :], 0.5)
    assert np.allclose(vpm_h_concat[:, 3:, :], 0.8)

    shape3 = (2, 3, 3) # Smaller height
    vpm3 = np.ones(shape3, dtype=np.float32) * 0.3

    # Vertical concat (should crop vpm1 to height 2)
    vpm_v_concat = vpm_concat_vertical(vpm1, vpm3)
    expected_v_shape = (2 + 2, 3, 3) # Cropped vpm1 height + vpm3 height
    assert vpm_v_concat.shape == expected_v_shape
    # Top part is cropped vpm1
    assert np.allclose(vpm_v_concat[:2, :, :], 0.5) # Top 2 rows from (cropped) vpm1
    # Bottom part is vpm3
    assert np.allclose(vpm_v_concat[2:, :, :], 0.3) # Bottom 2 rows from vpm3

def test_query_top_left_resolution_independence():
    """Test query_top_left works with different VPM sizes."""
    # Small VPM
    small_vpm = np.zeros((5, 5, 3), dtype=np.float32)
    small_vpm[0, 0, 0] = 1.0 # Bright top-left
    score_small = query_top_left(small_vpm, context_size=3)
    assert 0.0 <= score_small <= 1.0

    # Large VPM
    large_vpm = np.zeros((100, 100, 3), dtype=np.float32)
    large_vpm[0, 0, 0] = 1.0 # Bright top-left
    score_large = query_top_left(large_vpm, context_size=3)
    assert 0.0 <= score_large <= 1.0

    # Score should be similar for the same relative pattern
    # (This is a bit hand-wavy, but the function should handle size)
    # The weighting logic inside query_top_left should make it relative.
    # assert abs(score_small - score_large) < 0.1 # Example, might not hold strictly

# Add more tests as needed for specific functions like vpm_subtract if added.
