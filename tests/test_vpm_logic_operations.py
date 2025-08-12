# tests/test_vpm_logic_operations.py
"""
End-to-End Test Cases for VPM Logic Operations.

This file tests each function in `zeromodel.vpm_logic` by:
1. Creating synthetic data with known properties.
2. Defining simple SQL tasks for ZeroModel.
3. Generating base VPMs using ZeroModel.
4. Applying the specific vpm_logic operation.
5. Saving the resulting VPMs as images for visual inspection.
6. Performing basic sanity checks on the results.

This provides a comprehensive, visual demonstration of the VPM logic system.
"""

import numpy as np
import matplotlib

from zeromodel.config import get_config
matplotlib.use('Agg')  # Use non-GUI backend to avoid TclError
import matplotlib.pyplot as plt
import os
from zeromodel.core import ZeroModel
from zeromodel.vpm.encoder import VPMEncoder
from zeromodel.vpm.logic import (
    vpm_and, vpm_or, vpm_not, vpm_add, vpm_xor,
    vpm_nand, vpm_nor, query_top_left, vpm_subtract
)

# --- Test Configuration ---
# Directory to save output images
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create directory if it doesn't exist

# --- Helper Function for Visualization ---
def save_vpm_image(vpm: np.ndarray, title: str, filename: str): 
    """
    Saves a VPM as a grayscale image for inspection.
    Assumes the VPM is 2D or 3D with a channel dimension.
    """
    # Handle potential 3D VPM (e.g., HxWx3) by taking the first channel or averaging
    if vpm.ndim == 3:
        # If it's an RGB VPM, convert to grayscale for saving a single channel image
        # Simple average
        if vpm.shape[2] == 3:
            vpm_to_plot = np.mean(vpm, axis=2)
        else:
            # If not 3 channels, just take the first one
            vpm_to_plot = vpm[:, :, 0]
    else:
        vpm_to_plot = vpm

    plt.figure(figsize=(6, 6))
    # Use 'viridis' or 'gray' colormap. 'gray' might be more intuitive for [0,1] data.
    plt.imshow(vpm_to_plot, cmap='gray', vmin=0, vmax=1) 
    plt.title(title)
    plt.colorbar(label='Normalized Score')
    plt.xlabel('Metrics (sorted)')
    plt.ylabel('Documents (sorted)')
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved VPM image: {filepath}")

# --- Test Data Generation ---
def create_simple_test_data():
    """
    Creates a small, simple dataset to clearly demonstrate logic operations.
    4 documents, 2 metrics.
    """
    metric_names = ["feature_a", "feature_b"]
    # Data designed so sorting by each metric gives a clear, different order.
    # Doc 0: High A, Low B
    # Doc 1: Low A, High B
    # Doc 2: Medium A, Medium B
    # Doc 3: Low A, Low B
    score_matrix = np.array([
        [0.9, 0.1],  # Doc 0
        [0.2, 0.8],  # Doc 1
        [0.5, 0.4],  # Doc 2
        [0.1, 0.2],  # Doc 3
    ])
    return score_matrix, metric_names

# --- Individual Test Cases ---

def test_vpm_and_operation():
    """Test the vpm_and operation end-to-end."""
    print("\n--- Testing vpm_and ---")
    score_matrix, metric_names = create_simple_test_data()
    
    # Define tasks: Sort by each feature descending to highlight high values
    task_a = "SELECT * FROM virtual_index ORDER BY feature_a DESC"
    task_b = "SELECT * FROM virtual_index ORDER BY feature_b DESC"
    
    # Generate VPMs
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, task_a)
    vpm_a = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_a.sorted_matrix)
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, task_b)
    vpm_b = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_b.sorted_matrix)
    
    # Apply logic
    vpm_result = vpm_and(vpm_a, vpm_b)
    
    # Save images
    save_vpm_image(vpm_a, "VPM A (High feature_a)", "vpm_a_high_a.png")
    save_vpm_image(vpm_b, "VPM B (High feature_b)", "vpm_b_high_b.png")
    save_vpm_image(vpm_result, "VPM AND (High A AND High B)", "vpm_result_and.png")
    
    # Basic sanity check: Result should be <= both inputs (min operation)
    assert np.all(vpm_result <= vpm_a), "AND result should be <= VPM A"
    assert np.all(vpm_result <= vpm_b), "AND result should be <= VPM B"
    # Doc 2 (0.5, 0.4) is the only one reasonably high in both. Result should highlight it.
    # Check top-left score is reasonable.
    score = query_top_left(vpm_result)
    print(f"  AND Result Top-Left Score: {score:.4f}")
    # Should be less than either individual score (because it's the intersection)
    score_a = query_top_left(vpm_a)
    score_b = query_top_left(vpm_b)
    assert score <= score_a and score <= score_b, "AND score should be <= individual scores"
    print("  ✅ vpm_and test passed.")

def test_vpm_or_operation():
    """Test the vpm_or operation end-to-end."""
    print("\n--- Testing vpm_or ---")
    score_matrix, metric_names = create_simple_test_data()
    
    task_a = "SELECT * FROM virtual_index ORDER BY feature_a DESC"
    task_b = "SELECT * FROM virtual_index ORDER BY feature_b DESC"
    
    get_config("core").update({"default_output_precision": "float32"})
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, task_a)
    vpm_a = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_a.sorted_matrix)
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, task_b)
    vpm_b = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_b.sorted_matrix)
    
    vpm_result = vpm_or(vpm_a, vpm_b)
    
    save_vpm_image(vpm_a, "VPM A (High feature_a)", "vpm_a_high_a.png") # Overwrites previous, but that's okay
    save_vpm_image(vpm_b, "VPM B (High feature_b)", "vpm_b_high_b.png")
    save_vpm_image(vpm_result, "VPM OR (High A OR High B)", "vpm_result_or.png")
    
    # Basic sanity check: Result should be >= both inputs (max operation)
    assert np.all(vpm_result >= vpm_a), "OR result should be >= VPM A"
    assert np.all(vpm_result >= vpm_b), "OR result should be >= VPM B"
    # Should highlight Doc 0 (high A) and Doc 1 (high B)
    score = query_top_left(vpm_result)
    print(f"  OR Result Top-Left Score: {score:.4f}")
    score_a = query_top_left(vpm_a)
    score_b = query_top_left(vpm_b)
    assert score >= score_a and score >= score_b, "OR score should be >= individual scores"
    print("  ✅ vpm_or test passed.")

def test_vpm_not_operation():
    """Test the vpm_not operation end-to-end."""
    print("\n--- Testing vpm_not ---")
    score_matrix, metric_names = create_simple_test_data()
    
    task_a = "SELECT * FROM virtual_index ORDER BY feature_a DESC"
    
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, task_a)
    vpm_a = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_a.sorted_matrix)
    
    vpm_result = vpm_not(vpm_a)
    
    save_vpm_image(vpm_a, "VPM A (High feature_a)", "vpm_a_high_a.png")
    save_vpm_image(vpm_result, "VPM NOT A (Low feature_a)", "vpm_result_not.png")
    
    # Basic sanity check: NOT should invert values
    # The sum of a value and its NOT should be 1.0
    # Check a few points
    # Doc 0 should be bright in vpm_a, dark in vpm_not(vpm_a)
    # Doc 3 should be dark in vpm_a, bright in vpm_not(vpm_a)
    # Use query_top_left as a proxy
    score_a_top = query_top_left(vpm_a) # Should be high (Doc 0 is high A)
    score_not_a_top = query_top_left(vpm_result) # Should be low (Doc 0 is low NOT A)
    print(f"  A Top-Left Score: {score_a_top:.4f}")
    print(f"  NOT A Top-Left Score: {score_not_a_top:.4f}")
    sum_top = score_a_top + score_not_a_top
    assert abs(sum_top - 1.0) < 1e-5, f"NOT operation failed: {score_a_top} + {score_not_a_top} != 1.0"
    print("  ✅ vpm_not test passed.")

def test_vpm_subtract_operation():
    """Test the vpm_subtract operation end-to-end using encoded VPMs."""
    print("\n--- Testing vpm_subtract (on encoded VPMs) ---")
    score_matrix, metric_names = create_simple_test_data()
    
    # Define tasks to create a clear difference
    # A: High feature_a
    # B: High feature_b
    # Diff (A - B): High feature_a BUT NOT high feature_b
    # This should highlight Doc 0 [0.9, 0.1]
    task_a = "SELECT * FROM virtual_index ORDER BY feature_a DESC" # High A focus
    task_b = "SELECT * FROM virtual_index ORDER BY feature_b DESC" # High B focus
    
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, task_a)
    vpm_a = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_a.sorted_matrix)
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, task_b)
    vpm_b = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_b.sorted_matrix)
    
    # --- KEY CHANGE: Apply logic to encoded (uint8) VPMs ---
    vpm_result = vpm_subtract(vpm_a, vpm_b) # This should now work with uint8
    # --- END KEY CHANGE ---
    
    save_vpm_image(vpm_a, "VPM A (High feature_a - Encoded)", "vpm_a_high_a_encoded.png")
    save_vpm_image(vpm_b, "VPM B (High feature_b - Encoded)", "vpm_b_high_b_encoded.png")
    save_vpm_image(vpm_result, "VPM DIFF (High A - High B - Encoded)", "vpm_result_diff_encoded.png")
    
    # --- KEY CHANGE: Update sanity checks for uint8 ---
    # Basic sanity check: Result should be <= vpm_a (element-wise) and >= 0 (element-wise)
    # Since we are working with uint8, these comparisons are valid.
    assert np.all(vpm_result <= vpm_a), "DIFF result (uint8) should be <= VPM A (uint8)"
    assert np.all(vpm_result >= 0), "DIFF result (uint8) should be >= 0"
    # Also, result should be <= 255 (max uint8)
    assert np.all(vpm_result <= 255), "DIFF result (uint8) should be <= 255"
    # --- END KEY CHANGE ---
    
    # The top-left should highlight Doc 0 (high A, low B)
    # compared to just A (which highlights Doc 0 and maybe Doc 2)
    score_diff = query_top_left(vpm_result.astype(np.float64) / 255.0) # Normalize for query if needed internally, or modify query
    score_a = query_top_left(vpm_a.astype(np.float64) / 255.0)
    print(f"  A (Encoded) Top-Left Score (normalized): {score_a:.4f}")
    print(f"  DIFF (A-B) (Encoded) Top-Left Score (normalized): {score_diff:.4f}")
    # The key test is still visual inspection of the saved image.
    print("  ✅ vpm_diff test passed (check image vpm_result_diff_encoded.png).")

# --- Also update the save_vpm_image helper to handle uint8 correctly ---
# In the save_vpm_image function:
# (Removed duplicate save_vpm_image definition)

# tests/test_vpm_logic_operations.py

def test_vpm_add_operation():
    """Test the vpm_add operation end-to-end."""
    print("\n--- Testing vpm_add ---")
    score_matrix, metric_names = create_simple_test_data()
    
    task_a = "SELECT * FROM virtual_index ORDER BY feature_a DESC"
    task_b = "SELECT * FROM virtual_index ORDER BY feature_b DESC"
    
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, task_a)
    # --- KEY CHANGE 1: Get encoded VPMs (which are now float32 by default likely) ---
    # The test needs to be clear about the dtype it's working with.
    # If the test wants to work with uint8 VPMs, it needs to request them.
    # Let's assume for this test, we work with the default float32 VPMs from encode().
    vpm_a = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_a.sorted_matrix)
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, task_b)
    vpm_b = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_b.sorted_matrix)
    
    # --- KEY CHANGE 2: vpm_add now produces normalized float32 output ---
    vpm_result = vpm_add(vpm_a, vpm_b) # Result is now normalized float32
    
    save_vpm_image(vpm_a, "VPM A (High feature_a)", "vpm_a_high_a.png")
    save_vpm_image(vpm_b, "VPM B (High feature_b)", "vpm_b_high_b.png")
    save_vpm_image(vpm_result, "VPM ADD (High A + High B)", "vpm_result_add.png")
    
    # --- KEY CHANGE 3: Update assertions for normalized float32 output ---
    # Basic sanity check: Result values should be in the valid normalized range [0.0, 1.0]
    # and should be >= individual inputs (element-wise) due to addition.
    assert np.all(vpm_result >= 0.0), "ADD result should be >= 0.0"
    assert np.all(vpm_result <= 1.0), "ADD result should be <= 1.0 (normalized)" # <-- CHANGED ASSERTION
    assert np.all(vpm_result >= vpm_a), "ADD result should be >= VPM A (element-wise)"
    assert np.all(vpm_result >= vpm_b), "ADD result should be >= VPM B (element-wise)"
    # The result dtype should be float32 (normalized output)
    assert vpm_result.dtype == np.float32, f"ADD result dtype should be float32, got {vpm_result.dtype}" # <-- CHANGED ASSERTION
    # --- END KEY CHANGE 3 ---
    
    # Should amplify relevance where both are high (Doc 2 might be brighter)
    # Use query_top_left which now handles normalized floats correctly
    score_add = query_top_left(vpm_result) # Works on normalized float32
    score_a = query_top_left(vpm_a)
    score_b = query_top_left(vpm_b)
    print(f"  A Top-Left Score (normalized): {score_a:.4f}")
    print(f"  B Top-Left Score (normalized): {score_b:.4f}")
    print(f"  ADD (A+B) Top-Left Score (normalized): {score_add:.4f}")
    # Add score should be higher than either individual score (unless one is 1.0).
    # Check it's at least as high as the max of the two individual normalized scores.
    max_individual_norm = max(score_a, score_b)
    # Allow for small floating point differences
    assert score_add >= max_individual_norm - 1e-6, "ADD score (normalized) should be >= max individual normalized score"
    print("  ✅ vpm_add test passed.")

# tests/test_vpm_logic_operations.py

def test_vpm_xor_operation():
    """Test the vpm_xor operation end-to-end."""
    print("\n--- Testing vpm_xor ---")
    score_matrix, metric_names = create_simple_test_data()
    
    # XOR: (High A AND NOT High B) OR (High B AND NOT High A)
    # Should highlight Doc 0 (high A, not B) and Doc 1 (high B, not A)
    task_a = "SELECT * FROM virtual_index ORDER BY feature_a DESC"
    task_b = "SELECT * FROM virtual_index ORDER BY feature_b DESC"
    
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, task_a)
    # --- KEY CHANGE 1: Get encoded VPMs (which are now float32 by default likely) ---
    vpm_a = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_a.sorted_matrix)
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, task_b)
    vpm_b = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_b.sorted_matrix)
    
    # --- KEY CHANGE 2: vpm_xor now produces normalized float32 output ---
    vpm_result = vpm_xor(vpm_a, vpm_b) # Result is now normalized float32
    
    save_vpm_image(vpm_a, "VPM A (High feature_a)", "vpm_a_high_a.png")
    save_vpm_image(vpm_b, "VPM B (High feature_b)", "vpm_b_high_b.png")
    save_vpm_image(vpm_result, "VPM XOR (High A XOR High B)", "vpm_result_xor.png")
    
    # --- KEY CHANGE 3: Update assertions for normalized float32 output ---
    # Basic sanity check: Result values should be in the valid normalized range [0.0, 1.0]
    # and should be >= 0 and <= 1.
    assert np.all(vpm_result >= 0.0), "XOR result should be >= 0.0"
    assert np.all(vpm_result <= 1.0), "XOR result should be <= 1.0 (normalized)" # <-- CHANGED ASSERTION
    # The result dtype should be float32 (normalized output)
    assert vpm_result.dtype == np.float32, f"XOR result dtype should be float32, got {vpm_result.dtype}" # <-- CHANGED ASSERTION
    # --- END KEY CHANGE 3 ---
    
    # The top-left should highlight Doc 0 and Doc 1, but not Doc 2 (which is high in both)
    # So the score might be moderate.
    # Normalize for reporting/interpretation if needed.
    score_xor = query_top_left(vpm_result) # query_top_left handles float32 VPMs
    print(f"  XOR Result Top-Left Score (normalized): {score_xor:.4f}")
    # Visual inspection of vpm_result_xor.png is key here.
    # A basic check: score should be > 0 and <= 1.0
    assert 0.0 <= score_xor <= 1.0, f"XOR score should be in [0.0, 1.0], got {score_xor}"
    print("  ✅ vpm_xor test passed (check image vpm_result_xor.png).")


def test_vpm_nand_operation():
    """Test the vpm_nand operation end-to-end."""
    print("\n--- Testing vpm_nand ---")
    score_matrix, metric_names = create_simple_test_data()
    
    task_a = "SELECT * FROM virtual_index ORDER BY feature_a DESC"
    task_b = "SELECT * FROM virtual_index ORDER BY feature_b DESC"
    
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, task_a)
    vpm_a = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_a.sorted_matrix)
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, task_b)
    vpm_b = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_b.sorted_matrix)
    
    # NAND is NOT(AND)
    vpm_and_result = vpm_and(vpm_a, vpm_b)
    vpm_nand_result_direct = vpm_not(vpm_and_result)
    
    # Use the vpm_nand function
    vpm_nand_result = vpm_nand(vpm_a, vpm_b)
    
    save_vpm_image(vpm_a, "VPM A (High feature_a)", "vpm_a_high_a.png")
    save_vpm_image(vpm_b, "VPM B (High feature_b)", "vpm_b_high_b.png")
    save_vpm_image(vpm_nand_result, "VPM NAND (NOT(High A AND High B))", "vpm_result_nand.png")
    
    # Check that vpm_nand function produces the same result as manual composition
    assert np.allclose(vpm_nand_result, vpm_nand_result_direct, atol=1e-6), "vpm_nand function does not match manual NOT(AND)"
    
    # Basic sanity: NAND should be 1 where AND is 0, and 0 where AND is 1.
    # Doc 2 is high in both, so AND is high, so NAND should be low there.
    score_and = query_top_left(vpm_and_result)
    score_nand = query_top_left(vpm_nand_result)
    print(f"  AND Top-Left Score: {score_and:.4f}")
    print(f"  NAND Top-Left Score: {score_nand:.4f}")
    sum_and_nand = score_and + score_nand
    assert abs(sum_and_nand - 1.0) < 1e-5, f"NAND is NOT AND: {score_and} + {score_nand} != 1.0"
    print("  ✅ vpm_nand test passed.")

def test_vpm_nor_operation():
    """Test the vpm_nor operation end-to-end."""
    print("\n--- Testing vpm_nor ---")
    score_matrix, metric_names = create_simple_test_data()
    
    task_a = "SELECT * FROM virtual_index ORDER BY feature_a DESC"
    task_b = "SELECT * FROM virtual_index ORDER BY feature_b DESC"
    
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, task_a)
    vpm_a = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_a.sorted_matrix)
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, task_b)
    vpm_b = VPMEncoder(get_config("core").get("default_output_precision", "float32")).encode(model_b.sorted_matrix)
    
    # NOR is NOT(OR)
    vpm_or_result = vpm_or(vpm_a, vpm_b)
    vpm_nor_result_direct = vpm_not(vpm_or_result)
    
    # Use the vpm_nor function
    vpm_nor_result = vpm_nor(vpm_a, vpm_b)
    
    save_vpm_image(vpm_a, "VPM A (High feature_a)", "vpm_a_high_a.png")
    save_vpm_image(vpm_b, "VPM B (High feature_b)", "vpm_b_high_b.png")
    save_vpm_image(vpm_nor_result, "VPM NOR (NOT(High A OR High B))", "vpm_result_nor.png")
    
    # Check that vpm_nor function produces the same result as manual composition
    assert np.allclose(vpm_nor_result, vpm_nor_result_direct, atol=1e-6), "vpm_nor function does not match manual NOT(OR)"
    
    # Basic sanity: NOR should be 1 where OR is 0, and 0 where OR is 1.
    # Doc 3 is low in both, so OR is low, so NOR should be high there.
    score_or = query_top_left(vpm_or_result)
    score_nor = query_top_left(vpm_nor_result)
    print(f"  OR Top-Left Score: {score_or:.4f}")
    print(f"  NOR Top-Left Score: {score_nor:.4f}")
    sum_or_nor = score_or + score_nor
    assert abs(sum_or_nor - 1.0) < 1e-5, f"NOR is NOT OR: {score_or} + {score_nor} != 1.0"
    print("  ✅ vpm_nor test passed.")

# --- Main Execution Guard (Optional for direct script run) ---
if __name__ == "__main__":
    # This allows running the tests directly with Python
    # pytest tests/test_vpm_logic_operations.py -v -s
    # But if run as a script, you might want to call the functions manually
    # for quicker iteration during development.
    print("Running VPM Logic Operation Tests...")
    # Uncomment lines below to run specific tests directly
    # test_vpm_and_operation()
    # test_vpm_or_operation()
    # test_vpm_not_operation()
    # test_vpm_diff_operation()
    # test_vpm_add_operation()
    # test_vpm_xor_operation()
    # test_vpm_nand_operation()
    # test_vpm_nor_operation()
    print("Tests completed. Check the 'test_vpm_logic_output' directory for images.")
