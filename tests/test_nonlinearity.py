# tests/test_nonlinearity.py
"""
Test cases for the non-linearity hint feature in ZeroModel.
"""

import numpy as np
import pytest
from zeromodel import ZeroModel # Adjust import path if needed

def test_prepare_with_xor_hint():
    """Test that the 'xor' hint adds the correct features."""
    metric_names = ["x", "y", "other_metric"]
    # Create simple data where x*y and |x-y| are meaningful
    score_matrix = np.array([
        [0.8, 0.9, 0.1],  # High product (0.72), Low diff (0.1)
        [0.2, 0.1, 0.9],  # Low product (0.02), Low diff (0.1)
        [0.7, 0.2, 0.5],  # Low product (0.14), High diff (0.5)
        [0.3, 0.8, 0.3],  # Low product (0.24), High diff (0.5)
    ])

    zeromodel = ZeroModel(metric_names)
    
    # Test without hint first
    zeromodel_no_hint = ZeroModel(metric_names)
    zeromodel_no_hint.prepare(score_matrix, "SELECT * FROM virtual_index")
    assert zeromodel_no_hint.sorted_matrix.shape[1] == 3 # Original 3 metrics
    original_cols = zeromodel_no_hint.sorted_matrix.shape[1]

    # Test with 'xor' hint
    zeromodel.prepare(score_matrix, "SELECT * FROM virtual_index", nonlinearity_hint='xor')
    
    # Check that new features were added
    new_cols = zeromodel.sorted_matrix.shape[1]
    assert new_cols == original_cols + 2 # product and abs_diff should be added
    assert "hint_product_x_y" in zeromodel.get_metadata().get("metric_names", [])
    assert "hint_abs_diff_x_y" in zeromodel.get_metadata().get("metric_names", [])
    
    # Check if sorting by the new product feature works
    # Product values: 0.72, 0.02, 0.14, 0.24
    # Descending order should be indices corresponding to 0.72, 0.24, 0.14, 0.02
    # Which are original rows [0, 3, 2, 1]
    # doc_order reflects the new order of rows after sorting
    expected_doc_order_for_product = [0, 3, 2, 1]
    zeromodel_product_sort = ZeroModel(metric_names)
    zeromodel_product_sort.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY hint_product_x_y DESC", nonlinearity_hint='xor')
    assert np.array_equal(zeromodel_product_sort.doc_order, expected_doc_order_for_product)


def test_prepare_with_radial_hint():
    """Test that the 'radial' hint adds the correct features."""
    metric_names = ["coord_x", "coord_y", "value"]
    # Data points
    score_matrix = np.array([
        [0.5, 0.5, 0.1],  # Center (distance ~0)
        [0.9, 0.1, 0.9],  # Corner (distance ~ sqrt(0.4^2 + 0.4^2) = sqrt(0.32) ~ 0.566)
        [0.1, 0.9, 0.2],  # Corner (distance ~ 0.566)
        [0.7, 0.7, 0.5],  # Mid-ish (distance ~ sqrt(0.2^2 + 0.2^2) = sqrt(0.08) ~ 0.283)
    ])

    zeromodel = ZeroModel(metric_names)
    zeromodel.prepare(score_matrix, "SELECT * FROM virtual_index", nonlinearity_hint='radial')

    # Check that new features were added
    original_cols = 3
    new_cols = zeromodel.sorted_matrix.shape[1]
    assert new_cols == original_cols + 2 # distance and angle should be added
    assert "hint_radial_distance" in zeromodel.get_metadata().get("metric_names", [])
    assert "hint_radial_angle" in zeromodel.get_metadata().get("metric_names", [])
    
    # Check if sorting by the new distance feature works
    # Distances (approx): 0.0, 0.566, 0.566, 0.283
    # Descending order should be indices corresponding to 0.566, 0.566, 0.283, 0.0
    # Which are original rows [1, 2, 3, 0] (or [2, 1, 3, 0] as 1 and 2 are similar)
    # Ascending order (closest to center first) should be [0, 3, 1/2, 1/2]
    zeromodel_dist_sort = ZeroModel(metric_names)
    zeromodel_dist_sort.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY hint_radial_distance ASC", nonlinearity_hint='radial')
    # Row 0 (center) should be first
    assert zeromodel_dist_sort.doc_order[0] == 0


@pytest.mark.skip(reason="This test is now in test_nonlinearity_auto.py") 
def test_prepare_with_auto_hint():
    """Test that the 'auto' hint adds a set of features."""
    metric_names = ["a", "b", "c"]
    score_matrix = np.random.rand(5, 3) # Random data is fine for shape checks

    # --- Create a FRESH ZeroModel instance for this test ---
    zeromodel = ZeroModel(metric_names)
    zeromodel.prepare(score_matrix, "SELECT * FROM virtual_index", nonlinearity_hint='auto')
    # --- End of using fresh instance ---

    # Check that multiple new features were added
    original_cols = 3
    new_cols = zeromodel.sorted_matrix.shape[1]
    # auto should add for 3 inputs: product_ab, product_ac, product_bc, square_a, square_b
    # Total added = 5
    expected_new_cols = 5
    assert new_cols == original_cols + expected_new_cols, f"Expected {original_cols} + {expected_new_cols} = {original_cols + expected_new_cols} columns, got {new_cols}"
    
    metric_names_after = zeromodel.get_metadata().get("metric_names", [])
    assert any("auto_product" in name for name in metric_names_after)
    assert any("auto_square" in name for name in metric_names_after)
    assert "auto_product_a_b" in metric_names_after
    assert "auto_square_a" in metric_names_after


def test_prepare_no_hint_unchanged():
    """Test that not providing a hint leaves the data unchanged."""
    metric_names = ["m1", "m2"]
    score_matrix = np.array([[0.1, 0.9], [0.8, 0.2]])

    zeromodel = ZeroModel(metric_names)
    zeromodel.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY m1 DESC")
    
    # Should only have original metrics
    assert zeromodel.sorted_matrix.shape[1] == 2
    assert len(zeromodel.get_metadata().get("metric_names", [])) == 2


def test_prepare_unknown_hint_warning(caplog):
    """Test that an unknown hint logs a warning and proceeds."""
    import logging
    metric_names = ["m1", "m2"]
    score_matrix = np.array([[0.1, 0.9], [0.8, 0.2]])

    zeromodel = ZeroModel(metric_names)
    
    with caplog.at_level(logging.WARNING):
        zeromodel.prepare(score_matrix, "SELECT * FROM virtual_index", nonlinearity_hint='unknown_feature_set')
    
    # Check that a warning was logged
    assert "Unknown nonlinearity_hint" in caplog.text
    
    # Should proceed with original metrics only
    assert zeromodel.sorted_matrix.shape[1] == 2


def test_prepare_hint_with_insufficient_metrics():
    """Test that hints work gracefully with insufficient input metrics."""
    metric_names = ["single_metric"] # Only 1 metric
    score_matrix = np.array([[0.1], [0.9], [0.5]])

    zeromodel = ZeroModel(metric_names)
    # 'xor' and 'radial' need 2 metrics, 'auto' might try but add fewer
    zeromodel.prepare(score_matrix, "SELECT * FROM virtual_index", nonlinearity_hint='xor')
    
    # Should gracefully handle it, likely adding no new features or just squares
    # The core functionality (sorting by the single metric) should still work
    assert zeromodel.sorted_matrix is not None
    assert zeromodel.sorted_matrix.shape[0] == 3 # Number of documents preserved
    # Sorting by the only metric
    sorted_model = ZeroModel(metric_names)
    sorted_model.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY single_metric DESC", nonlinearity_hint='xor')
    # Descending order of [0.1, 0.9, 0.5] is [1, 2, 0]
    assert np.array_equal(sorted_model.doc_order, [1, 2, 0])
