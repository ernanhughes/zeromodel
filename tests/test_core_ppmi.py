# test_zeromodel_ppm.py
import os
import tempfile
import numpy as np
import pytest

from zeromodel import ZeroModel

def test_zero_model_ppm_view_compilation():
    """
    Test that ZeroModel can create a canonical PPM-IMG and compile
    different virtual views (sort orders) from it, producing distinct
    critical tiles. This test is robust to specific data values.
    """
    # --- 1. Generate Sample Data ---
    np.random.seed(42) # For reproducibility
    num_docs = 20
    num_metrics = 5
    metric_names = [f"metric_{i}" for i in range(num_metrics)]

    # Create random data
    score_matrix = np.random.rand(num_docs, num_metrics).astype(np.float32)
    
    # Modify specific elements to create a clear, detectable difference for testing
    # We will ensure doc A has a unique high value in metric 0
    # and doc B has a unique high value in metric 1.
    # This makes it highly likely they sort to the top of their respective views,
    # even after normalization, unless another doc has an even higher value.
    doc_a_for_metric_0 = 5
    doc_b_for_metric_1 = 7
    unique_high_val = 0.99 # A value very unlikely to be exceeded by random [0,1) data
    
    score_matrix[doc_a_for_metric_0, 0] = unique_high_val # Metric 0
    score_matrix[doc_b_for_metric_1, 1] = unique_high_val # Metric 1

    # Simple SQL query for initial analysis (doesn't affect canonical sort)
    sql_query = "SELECT * FROM virtual_index" # Placeholder

    # --- 2. Prepare ZeroModel (Creates PPM-IMG) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        ppm_path = os.path.join(os.getcwd(), "test_canonical.ppm.png")

        model = ZeroModel(metric_names=metric_names)
        model.prepare(
            score_matrix=score_matrix,
            sql_query=sql_query,
            ppm_output_path=ppm_path
        )

        # --- 3. Validate PPM-IMG was created ---
        assert os.path.exists(ppm_path), "PPM-IMG file was not created."
        assert model.ppm_image_path == ppm_path, "Model's ppm_image_path not set correctly."
        assert model.canonical_matrix is not None, "Canonical matrix should be stored."
        expected_width = max(score_matrix.shape[0], 12) # PPM-IMG minimum width
        expected_height = score_matrix.shape[1]
        # Note: canonical_matrix might be padded, so check rows, cols separately if needed
        assert model.canonical_matrix.shape[0] == num_docs, "Canonical matrix doc count mismatch."
        assert model.canonical_matrix.shape[1] == num_metrics, "Canonical matrix metric count mismatch."

        # --- 4. Compile Views and Extract Critical Tiles ---
        metric_0_idx = 0
        metric_1_idx = 1

        # Compile view sorted by Metric 0
        perm_metric_0 = model.compile_view(metric_idx=metric_0_idx)
        tile_metric_0 = model.extract_critical_tile(metric_idx=metric_0_idx, size=4)

        # Compile view sorted by Metric 1
        perm_metric_1 = model.compile_view(metric_idx=metric_1_idx)
        tile_metric_1 = model.extract_critical_tile(metric_idx=metric_1_idx, size=4)

        # --- 5. Validate Different Views Produce Different Results ---
        
        # Core Test 1: Permutations for different metrics should generally be different
        # Check the top few documents to allow for potential ties in lower ranks
        top_n_to_check = min(5, len(perm_metric_0), len(perm_metric_1))
        assert not np.array_equal(perm_metric_0[:top_n_to_check], perm_metric_1[:top_n_to_check]), \
            "Permutations for different metrics should differ in the top ranks."

        # Core Test 2: Extracted tiles should be different arrays
        assert tile_metric_0.shape == (4, 4, 3), "Extracted tile has incorrect shape."
        assert tile_metric_1.shape == (4, 4, 3), "Extracted tile has incorrect shape."
        assert tile_metric_0.dtype == np.uint16, "Extracted tile should be uint16."
        assert tile_metric_1.dtype == np.uint16, "Extracted tile should be uint16."
        
        # Asserting the tiles are not identical is a strong validation of the view system
        assert not np.array_equal(tile_metric_0, tile_metric_1), \
            "Critical tiles from different views should be different."

        # Optional/Robust Check: Verify our 'unique' docs are likely at the top of their views
        # This is probabilistic but very likely given the unique high value.
        # We check if they are in the top few positions.
        assert doc_a_for_metric_0 in perm_metric_0[:3], \
            f"Doc {doc_a_for_metric_0} (high metric 0) should be in top 3 for metric 0 view."
        assert doc_b_for_metric_1 in perm_metric_1[:3], \
            f"Doc {doc_b_for_metric_1} (high metric 1) should be in top 3 for metric 1 view."

        print(f"View 0 (Metric {metric_0_idx}) top docs: {perm_metric_0[:5]}")
        print(f"View 1 (Metric {metric_1_idx}) top docs: {perm_metric_1[:5]}")
        # Print top-left R values for manual inspection if needed
        print(f"View 0 top-left R value: {tile_metric_0[0, 0, 0]}")
        print(f"View 1 top-left R value: {tile_metric_1[0, 0, 0]}")

        # --- 6. Validate PPM-IMG Data Encoding (Basic Check) ---
        assert np.any(tile_metric_0[:, :, 0] > 0), "Metric 0 view tile R channel is completely blank."
        assert np.any(tile_metric_1[:, :, 0] > 0), "Metric 1 view tile R channel is completely blank."

        # --- 7. Test Decision API ---
        decision_doc_0, decision_rel_0 = model.get_decision_by_metric(metric_idx=metric_0_idx, context_size=4)
        decision_doc_1, decision_rel_1 = model.get_decision_by_metric(metric_idx=metric_1_idx, context_size=4)

        # The decision doc should be one of the top docs from its respective permutation
        assert decision_doc_0 in perm_metric_0[:3], \
            f"Decision doc {decision_doc_0} for metric {metric_0_idx} not in top 3 of its view {perm_metric_0[:3]}"
        assert decision_doc_1 in perm_metric_1[:3], \
            f"Decision doc {decision_doc_1} for metric {metric_1_idx} not in top 3 of its view {perm_metric_1[:3]}"

        # Relevance should be calculable (0-1 range check is basic)
        assert 0.0 <= decision_rel_0 <= 1.0, f"Relevance for metric 0 is out of range: {decision_rel_0}"
        assert 0.0 <= decision_rel_1 <= 1.0, f"Relevance for metric 1 is out of range: {decision_rel_1}"

        print("All PPM-IMG view compilation tests passed!")


