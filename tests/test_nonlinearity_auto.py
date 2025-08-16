# In tests/test_nonlinearity.py
import numpy as np
import pytest

from zeromodel import ZeroModel  # Adjust import path if needed


def test_prepare_with_auto_hint():
    """Test that the 'auto' hint adds a set of features."""
    metric_names = ["a", "b", "c"]
    score_matrix = np.random.rand(5, 3) # Random data is fine for shape checks

    # --- Create a FRESH ZeroModel instance for this test ---
    zeromodel = ZeroModel(metric_names)
    # --- Add Debug Check ---
    # Verify the initial state of the fresh instance
    initial_metric_count = len(zeromodel.effective_metric_names)
    print(f"\nDEBUG: Fresh ZeroModel created. Initial metric_names count: {initial_metric_count}")
    assert initial_metric_count == 3, f"Fresh ZeroModel should have 3 initial metrics, got {initial_metric_count}"
    # --- End Debug Check ---
    
    zeromodel.prepare(score_matrix, "SELECT * FROM virtual_index", nonlinearity_hint='auto')
    # --- End of using fresh instance ---

    # Check that multiple new features were added
    original_cols = 3
    new_cols = zeromodel.sorted_matrix.shape[1]
    # auto should add for 3 inputs: product_ab, product_ac, product_bc, square_a, square_b
    # Total added = 5
    expected_new_cols = 5
    assert new_cols == original_cols + expected_new_cols, f"Expected {original_cols} + {expected_new_cols} = {original_cols + expected_new_cols} columns, got {new_cols}. Sorted matrix shape: {zeromodel.sorted_matrix.shape}, Metric names count: {len(zeromodel.get_metadata().get('metric_names', []))}"
    
    metric_names_after = zeromodel.get_metadata().get("metric_names", [])
    print(f"DEBUG: Metric names after prepare: {metric_names_after}") # Add this debug print
    assert any("auto_product" in name for name in metric_names_after)
    assert any("auto_square" in name for name in metric_names_after)
    assert "auto_product_a_b" in metric_names_after
    assert "auto_square_a" in metric_names_after