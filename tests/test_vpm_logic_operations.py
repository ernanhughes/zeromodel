# tests/test_vpm_logic_operations.py

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from zeromodel.vpm.logic import (
    vpm_and, vpm_or, vpm_not, vpm_xor, 
    vpm_nand, vpm_nor, vpm_add, vpm_subtract,
    vpm_query_top_left
)
from zeromodel.core import ZeroModel

# Test configuration
OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def _to_normalized_array(obj):
    """Convert PIL Image or numpy array to normalized float32 array in [0,1] range."""
    if isinstance(obj, Image.Image):
        # Convert PIL Image to numpy array
        arr = np.array(obj.convert("RGB"))
        # Normalize to [0,1] range
        if arr.dtype != np.float32:
            if np.issubdtype(arr.dtype, np.integer):
                max_val = np.iinfo(arr.dtype).max
                arr = arr.astype(np.float32) / max_val
            else:
                arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
        return arr
    elif isinstance(obj, np.ndarray):
        return np.clip(obj.astype(np.float32), 0.0, 1.0)
    else:
        raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(obj)}")

def save_vpm_image(vpm, title: str, filename: str):
    """Save VPM as image with proper handling of both array and PIL Image types."""
    # Convert to normalized array for consistent processing
    arr = _to_normalized_array(vpm)
    
    # Handle 3D arrays (RGB) by converting to grayscale if needed
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            # Convert RGB to grayscale
            arr = 0.2989 * arr[:,:,0] + 0.5870 * arr[:,:,1] + 0.1140 * arr[:,:,2]
        else:
            # Take first channel
            arr = arr[:,:,0]
    
    # Create and save image
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar(label='Normalized Score')
    plt.xlabel('Metrics (sorted)')
    plt.ylabel('Documents (sorted)')
    
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved VPM image: {filepath}")

def create_test_data():
    """Create test data that works well with logic operations."""
    # Create a simple 4x4 matrix with clear patterns
    score_matrix = np.array([
        [0.9, 0.1, 0.8, 0.2],  # High A, Low B, High C, Low D
        [0.2, 0.8, 0.3, 0.7],  # Low A, High B, Low C, High D
        [0.6, 0.6, 0.6, 0.6],  # Medium values
        [0.1, 0.2, 0.3, 0.4],  # Low values
    ], dtype=np.float32)
    
    metric_names = ["metric_a", "metric_b", "metric_c", "metric_d"]
    return score_matrix, metric_names

def test_vpm_and_operation():
    """Test the vpm_and operation with proper precision handling."""
    print("\n--- Testing vpm_and ---")
    score_matrix, metric_names = create_test_data()
    
    # Create VPMs for high A and high B
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_a DESC")
    vpm_a = model_a.sorted_matrix
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_b DESC")
    vpm_b = model_b.sorted_matrix
    
    # Apply AND operation
    vpm_result = vpm_and(vpm_a, vpm_b)
    
    # Save images
    save_vpm_image(vpm_a, "VPM A (High metric_a)", "vpm_a_and.png")
    save_vpm_image(vpm_b, "VPM B (High metric_b)", "vpm_b_and.png")
    save_vpm_image(vpm_result, "VPM AND (High A AND High B)", "vpm_result_and.png")
    
    # Basic sanity check with relaxed precision
    # AND should be <= both inputs (minimum operation)
    assert np.all(vpm_result <= vpm_a + 1e-6), "AND result should be <= VPM A"
    assert np.all(vpm_result <= vpm_b + 1e-6), "AND result should be <= VPM B"
    
    # Check that result is in valid range
    assert np.all(vpm_result >= 0.0 - 1e-6), "AND result should be >= 0.0"
    assert np.all(vpm_result <= 1.0 + 1e-6), "AND result should be <= 1.0"
    
    print("  ✅ vpm_and test passed.")

def test_vpm_or_operation():
    """Test the vpm_or operation with proper precision handling."""
    print("\n--- Testing vpm_or ---")
    score_matrix, metric_names = create_test_data()
    
    # Create VPMs for high A and high B
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_a DESC")
    vpm_a = model_a.sorted_matrix
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_b DESC")
    vpm_b = model_b.sorted_matrix
    
    # Apply OR operation
    vpm_result = vpm_or(vpm_a, vpm_b)
    
    # Save images
    save_vpm_image(vpm_a, "VPM A (High metric_a)", "vpm_a_or.png")
    save_vpm_image(vpm_b, "VPM B (High metric_b)", "vpm_b_or.png")
    save_vpm_image(vpm_result, "VPM OR (High A OR High B)", "vpm_result_or.png")
    
    # Basic sanity check with relaxed precision
    # OR should be >= both inputs (maximum operation)
    assert np.all(vpm_result >= vpm_a - 1e-6), "OR result should be >= VPM A"
    assert np.all(vpm_result >= vpm_b - 1e-6), "OR result should be >= VPM B"
    
    # Check that result is in valid range
    assert np.all(vpm_result >= 0.0 - 1e-6), "OR result should be >= 0.0"
    assert np.all(vpm_result <= 1.0 + 1e-6), "OR result should be <= 1.0"
    
    print("  ✅ vpm_or test passed.")

def test_vpm_not_operation():
    """Test the vpm_not operation with proper precision handling."""
    print("\n--- Testing vpm_not ---")
    score_matrix, metric_names = create_test_data()
    
    # Create VPM for high A
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_a DESC")
    vpm_a = model_a.sorted_matrix
    
    # Apply NOT operation
    vpm_result = vpm_not(vpm_a)
    
    # Save images
    save_vpm_image(vpm_a, "VPM A (High metric_a)", "vpm_a_not.png")
    save_vpm_image(vpm_result, "VPM NOT A (Low metric_a)", "vpm_result_not.png")
    
    # Basic sanity check with relaxed precision
    # NOT should invert values: a + not_a ≈ 1.0
    sum_check = vpm_a + vpm_result
    assert np.allclose(sum_check, 1.0, atol=1e-5), "NOT operation should invert values"
    
    # Check that result is in valid range
    assert np.all(vpm_result >= 0.0 - 1e-6), "NOT result should be >= 0.0"
    assert np.all(vpm_result <= 1.0 + 1e-6), "NOT result should be <= 1.0"
    
    print("  ✅ vpm_not test passed.")

def test_vpm_xor_operation():
    """Test the vpm_xor operation with proper array handling."""
    print("\n--- Testing vpm_xor ---")
    score_matrix, metric_names = create_test_data()
    
    # Create VPMs for high A and high B
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_a DESC")
    vpm_a = model_a.sorted_matrix
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_b DESC")
    vpm_b = model_b.sorted_matrix
    
    # Apply XOR operation
    vpm_result = vpm_xor(vpm_a, vpm_b)
    
    # Save images
    save_vpm_image(vpm_a, "VPM A (High metric_a)", "vpm_a_xor.png")
    save_vpm_image(vpm_b, "VPM B (High metric_b)", "vpm_b_xor.png")
    save_vpm_image(vpm_result, "VPM XOR (High A XOR High B)", "vpm_result_xor.png")
    
    # Basic sanity check
    # XOR should highlight differences between inputs
    assert np.all(vpm_result >= 0.0 - 1e-6), "XOR result should be >= 0.0"
    assert np.all(vpm_result <= 1.0 + 1e-6), "XOR result should be <= 1.0"
    
    # XOR of identical inputs should be zero
    vpm_identical = vpm_xor(vpm_a, vpm_a)
    assert np.allclose(vpm_identical, 0.0, atol=1e-6), "XOR of identical inputs should be zero"
    
    print("  ✅ vpm_xor test passed.")

def test_vpm_add_operation():
    """Test the vpm_add operation with proper array handling."""
    print("\n--- Testing vpm_add ---")
    score_matrix, metric_names = create_test_data()
    
    # Create VPMs for high A and high B
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_a DESC")
    vpm_a = model_a.sorted_matrix
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_b DESC")
    vpm_b = model_b.sorted_matrix
    
    # Apply ADD operation
    vpm_result = vpm_add(vpm_a, vpm_b)
    
    # Save images
    save_vpm_image(vpm_a, "VPM A (High metric_a)", "vpm_a_add.png")
    save_vpm_image(vpm_b, "VPM B (High metric_b)", "vpm_b_add.png")
    save_vpm_image(vpm_result, "VPM ADD (High A + High B)", "vpm_result_add.png")
    
    # Basic sanity check
    # ADD should combine inputs (clipped to [0,1])
    assert np.all(vpm_result >= 0.0 - 1e-6), "ADD result should be >= 0.0"
    assert np.all(vpm_result <= 1.0 + 1e-6), "ADD result should be <= 1.0"
    
    # Result should be >= individual inputs (due to addition)
    assert np.all(vpm_result >= vpm_a - 1e-6), "ADD result should be >= VPM A"
    assert np.all(vpm_result >= vpm_b - 1e-6), "ADD result should be >= VPM B"
    
    print("  ✅ vpm_add test passed.")

def test_vpm_subtract_operation():
    """Test the vpm_subtract operation with proper array handling."""
    print("\n--- Testing vpm_subtract ---")
    score_matrix, metric_names = create_test_data()
    
    # Create VPMs for high A and high B
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_a DESC")
    vpm_a = model_a.sorted_matrix
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_b DESC")
    vpm_b = model_b.sorted_matrix
    
    # Apply SUBTRACT operation
    vpm_result = vpm_subtract(vpm_a, vpm_b)
    
    # Save images
    save_vpm_image(vpm_a, "VPM A (High metric_a)", "vpm_a_subtract.png")
    save_vpm_image(vpm_b, "VPM B (High metric_b)", "vpm_b_subtract.png")
    save_vpm_image(vpm_result, "VPM SUBTRACT (High A - High B)", "vpm_result_subtract.png")
    
    # Basic sanity check
    # SUBTRACT should highlight where A > B (clipped to [0,1])
    assert np.all(vpm_result >= 0.0 - 1e-6), "SUBTRACT result should be >= 0.0"
    assert np.all(vpm_result <= 1.0 + 1e-6), "SUBTRACT result should be <= 1.0"
    
    print("  ✅ vpm_subtract test passed.")

def test_vpm_nand_operation():
    """Test the vpm_nand operation with proper precision handling."""
    print("\n--- Testing vpm_nand ---")
    score_matrix, metric_names = create_test_data()
    
    # Create VPMs for high A and high B
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_a DESC")
    vpm_a = model_a.sorted_matrix
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_b DESC")
    vpm_b = model_b.sorted_matrix
    
    # Apply NAND operation
    vpm_result = vpm_nand(vpm_a, vpm_b)
    
    # Save images
    save_vpm_image(vpm_a, "VPM A (High metric_a)", "vpm_a_nand.png")
    save_vpm_image(vpm_b, "VPM B (High metric_b)", "vpm_b_nand.png")
    save_vpm_image(vpm_result, "VPM NAND (NOT(High A AND High B))", "vpm_result_nand.png")
    
    # NAND is NOT(AND)
    vpm_and_result = vpm_and(vpm_a, vpm_b)
    vpm_nand_manual = vpm_not(vpm_and_result)
    
    # Check that vpm_nand function produces the same result as manual composition
    assert np.allclose(vpm_result, vpm_nand_manual, atol=1e-6), "vpm_nand function does not match manual NOT(AND)"
    
    # Basic sanity check
    assert np.all(vpm_result >= 0.0 - 1e-6), "NAND result should be >= 0.0"
    assert np.all(vpm_result <= 1.0 + 1e-6), "NAND result should be <= 1.0"
    
    print("  ✅ vpm_nand test passed.")

def test_vpm_nor_operation():
    """Test the vpm_nor operation with proper precision handling."""
    print("\n--- Testing vpm_nor ---")
    score_matrix, metric_names = create_test_data()
    
    # Create VPMs for high A and high B
    model_a = ZeroModel(metric_names)
    model_a.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_a DESC")
    vpm_a = model_a.sorted_matrix
    
    model_b = ZeroModel(metric_names)
    model_b.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_b DESC")
    vpm_b = model_b.sorted_matrix
    
    # Apply NOR operation
    vpm_result = vpm_nor(vpm_a, vpm_b)
    
    # Save images
    save_vpm_image(vpm_a, "VPM A (High metric_a)", "vpm_a_nor.png")
    save_vpm_image(vpm_b, "VPM B (High metric_b)", "vpm_b_nor.png")
    save_vpm_image(vpm_result, "VPM NOR (NOT(High A OR High B))", "vpm_result_nor.png")
    
    # NOR is NOT(OR)
    vpm_or_result = vpm_or(vpm_a, vpm_b)
    vpm_nor_manual = vpm_not(vpm_or_result)
    
    # Check that vpm_nor function produces the same result as manual composition
    assert np.allclose(vpm_result, vpm_nor_manual, atol=1e-6), "vpm_nor function does not match manual NOT(OR)"
    
    # Basic sanity check
    assert np.all(vpm_result >= 0.0 - 1e-6), "NOR result should be >= 0.0"
    assert np.all(vpm_result <= 1.0 + 1e-6), "NOR result should be <= 1.0"
    
    print("  ✅ vpm_nor test passed.")

def test_vpm_query_top_left():
    """Test the vpm_query_top_left function."""
    print("\n--- Testing vpm_query_top_left ---")
    score_matrix, metric_names = create_test_data()
    
    # Create a VPM
    model = ZeroModel(metric_names)
    model.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY metric_a DESC")
    vpm = model.sorted_matrix
    
    # Query top-left region
    score = vpm_query_top_left(vpm, context_size=4)
    
    # Basic sanity check
    assert 0.0 <= score <= 1.0, f"Top-left score should be in [0,1], got {score}"
    
    # For our test data, document 0 (high A) should be prominent
    # So top-left score should be relatively high
    assert score > 0.4, f"Top-left score should be high for high-A document, got {score}"
    
    print(f"  Top-left query score: {score:.4f}")
    print("  ✅ vpm_query_top_left test passed.")

if __name__ == "__main__":
    # Run all tests
    print("Running ZeroModel VPM Logic Operations Tests...")
    
    test_vpm_and_operation()
    test_vpm_or_operation()
    test_vpm_not_operation()
    test_vpm_xor_operation()
    test_vpm_add_operation()
    test_vpm_subtract_operation()
    test_vpm_nand_operation()
    test_vpm_nor_operation()
    test_vpm_query_top_left()
    
    print("\n🎉 All ZeroModel VPM Logic Operations tests passed!")
    print(f"📊 Check the '{OUTPUT_DIR}' directory for generated VPM images.")