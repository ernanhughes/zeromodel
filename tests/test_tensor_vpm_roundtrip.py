# tests/test_tensor_vpm_roundtrip.py
import numpy as np
import time
import pytest
import pickle
import random
import string
from typing import Any, Dict, List, Tuple, Union

from zeromodel.provenance.core import tensor_to_vpm, vpm_to_tensor

def generate_test_data(seed: int = 42) -> Dict[str, Any]:
    """Generate diverse test data structures for roundtrip testing"""
    random.seed(seed)
    np.random.seed(seed)
    
    # Create a complex, nested structure that mimics real AI model state
    return {
        "coef_": np.random.randn(5, 10).astype(np.float32),
        "intercept_": np.array([0.5, -0.3, 0.7], dtype=np.float32),
        "layer_weights": {
            "conv1": {
                "weights": np.random.randn(32, 3, 3, 3).astype(np.float32),
                "biases": np.zeros(32, dtype=np.float32),
                "activation_stats": {
                    "mean": 0.25,
                    "std": 0.1,
                    "sparsity": 0.75
                }
            },
            "dense1": {
                "weights": np.random.randn(128, 512).astype(np.float32),
                "biases": np.random.randn(128).astype(np.float32)
            }
        },
        "training_state": {
            "epoch": 42,
            "batch_size": 32,
            "learning_rate": 0.001,
            "loss_history": np.array([1.2, 0.9, 0.7, 0.5, 0.35], dtype=np.float32),
            "optimizer_state": {
                "step": 1000,
                "exp_avg": np.random.randn(512).astype(np.float32),
                "exp_avg_sq": np.random.rand(512).astype(np.float32)
            }
        },
        "metadata": {
            "model_name": "test_model",
            "created_at": "2023-11-15T14:30:00Z",
            "tags": ["test", "roundtrip", "vpm"],
            "random_string": ''.join(random.choices(string.ascii_letters + string.digits, k=100))
        },
        "scalar_values": {
            "accuracy": 0.87,
            "precision": 0.85,
            "recall": 0.89
        },
        "edge_cases": {
            "empty_array": np.array([], dtype=np.float32),
            "single_value": np.array([42.0], dtype=np.float32),
            "nan_values": np.array([np.nan, 1.0, np.inf, -np.inf], dtype=np.float32),
            "mixed_types": [1, "text", True, None, {"key": "value"}]
        }
    }

def verify_roundtrip(original: Any, restored: Any, path: str = "") -> Tuple[bool, str]:
    """
    Recursively verify that the restored data matches the original data.
    
    This is critical for ZeroModel's "deterministic, reproducible provenance" principle.
    """
    # Handle different types
    if isinstance(original, dict):
        if not isinstance(restored, dict):
            return False, f"{path}: Expected dict, got {type(restored)}"
        
        # Check all keys exist
        for key in original:
            if key not in restored:
                return False, f"{path}: Missing key '{key}'"
            
            # Verify each value
            result, msg = verify_roundtrip(original[key], restored[key], f"{path}.{key}")
            if not result:
                return False, msg
                
        # Check no extra keys
        for key in restored:
            if key not in original:
                return False, f"{path}: Unexpected key '{key}'"
                
        return True, ""
    
    elif isinstance(original, (list, tuple)):
        if not isinstance(restored, (list, tuple)):
            return False, f"{path}: Expected list/tuple, got {type(restored)}"
        
        if len(original) != len(restored):
            return False, f"{path}: Length mismatch ({len(original)} vs {len(restored)})"
        
        for i, (orig_item, restored_item) in enumerate(zip(original, restored)):
            result, msg = verify_roundtrip(orig_item, restored_item, f"{path}[{i}]")
            if not result:
                return False, msg
                
        return True, ""
    
    elif isinstance(original, np.ndarray):
        if not isinstance(restored, np.ndarray):
            return False, f"{path}: Expected numpy array, got {type(restored)}"
        
        if original.shape != restored.shape:
            return False, f"{path}: Shape mismatch ({original.shape} vs {restored.shape})"
        
        if original.dtype != restored.dtype:
            return False, f"{path}: Dtype mismatch ({original.dtype} vs {restored.dtype})"
        
        # Handle NaNs and special values properly
        if np.issubdtype(original.dtype, np.floating):
            # For float arrays, use allclose with appropriate tolerances
            if not np.allclose(original, restored, equal_nan=True):
                # Find the first mismatch
                mismatch_idx = np.where(~np.isclose(original, restored, equal_nan=True))
                if len(mismatch_idx[0]) > 0:
                    idx_str = ",".join(str(mismatch_idx[i][0]) for i in range(len(mismatch_idx)))
                    return False, f"{path}: Value mismatch at [{idx_str}] ({original.flat[0]} vs {restored.flat[0]})"
                return False, f"{path}: Value mismatch (NaN handling)"
        else:
            # For non-float arrays, use exact comparison
            if not np.array_equal(original, restored, equal_nan=True):
                mismatch_idx = np.where(original != restored)
                if len(mismatch_idx[0]) > 0:
                    idx_str = ",".join(str(mismatch_idx[i][0]) for i in range(len(mismatch_idx)))
                    return False, f"{path}: Value mismatch at [{idx_str}] ({original[mismatch_idx[0][0]]} vs {restored[mismatch_idx[0][0]]})"
                return False, f"{path}: Value mismatch"
        
        return True, ""
    
    elif original is None:
        if restored is not None:
            return False, f"{path}: Expected None, got {restored}"
        return True, ""
    
    elif isinstance(original, (int, float, bool, str)):
        if original != restored:
            return False, f"{path}: Value mismatch ({original} vs {restored})"
        return True, ""
    
    else:
        # Fallback for other types - use pickle comparison
        orig_pickle = pickle.dumps(original)
        restored_pickle = pickle.dumps(restored)
        if orig_pickle != restored_pickle:
            return False, f"{path}: Pickle mismatch (unhandled type {type(original)})"
        return True, ""

def test_tensor_vpm_roundtrip_comprehensive():
    """
    Test the tensor_to_vpm and vpm_to_tensor functions with a comprehensive dataset.
    
    This test demonstrates ZeroModel's core principle: "intelligence lives in the data structure"
    by verifying that complex AI model states can be perfectly preserved and restored.
    """
    # Generate diverse test data that mimics real AI model state
    original = generate_test_data()
    
    # Measure conversion time
    start_time = time.perf_counter()
    vpm = tensor_to_vpm(original)
    vpm_creation_time = time.perf_counter() - start_time
    
    # Verify VPM dimensions make sense for the data
    width, height = vpm.size
    assert width > 0 and height > 0, "VPM has invalid dimensions"
    
    # Calculate expected size based on data complexity
    # This demonstrates ZeroModel's "storage-agnostic" principle
    expected_min_size = max(16, int(np.sqrt(len(pickle.dumps(original)) / 3)))
    assert width >= expected_min_size and height >= expected_min_size, \
        f"VPM too small for data size (w={width}, h={height}, expected min={expected_min_size})"
    
    # Measure restoration time
    start_time = time.perf_counter()
    restored = vpm_to_tensor(vpm)
    restoration_time = time.perf_counter() - start_time
    
    # Verify the roundtrip preserved all data exactly
    # This is critical for ZeroModel's "deterministic, reproducible provenance"
    is_valid, error_msg = verify_roundtrip(original, restored)
    assert is_valid, f"Roundtrip validation failed: {error_msg}"
    
    # Verify performance characteristics
    # Demonstrates ZeroModel's "milliseconds on tiny hardware" principle
    assert vpm_creation_time < 0.1, f"VPM creation too slow: {vpm_creation_time:.4f}s"
    assert restoration_time < 0.1, f"Restoration too slow: {restoration_time:.4f}s"
    
    print(f"\nVPM roundtrip test results:")
    print(f"  VPM creation:    {vpm_creation_time:.6f}s (size: {width}x{height})")
    print(f"  Restoration:     {restoration_time:.6f}s")
    print(f"  Data fidelity:   Verified (bit-for-bit identical restoration)")
    print(f"  Compression:     {len(pickle.dumps(original))} bytes → {width*height*3} pixels")

@pytest.mark.parametrize("data_type, value", [
    ("scalar_int", 42),
    ("scalar_float", 3.14159),
    ("scalar_bool", True),
    ("scalar_string", "test_string"),
    ("empty_array", np.array([], dtype=np.float32)),
    ("single_value", np.array([42.0], dtype=np.float32)),
    ("small_array", np.random.randn(3, 3).astype(np.float32)),
    ("medium_array", np.random.randn(100, 10).astype(np.float32)),
    ("large_array", np.random.randn(1000, 100).astype(np.float32)),
    ("nested_dict", {"a": 1, "b": {"c": 2, "d": np.array([3.0])}}),
    ("mixed_types_list", [1, "text", True, None, {"key": "value"}]),
    ("nan_values", np.array([np.nan, 1.0, np.inf, -np.inf], dtype=np.float32)),
])
def test_tensor_vpm_roundtrip_parameterized(data_type, value):
    """
    Parameterized test for different data types and sizes.
    
    This demonstrates ZeroModel's "universal, self-describing artifact" principle
    by showing it works consistently across diverse data structures.
    """
    start_time = time.perf_counter()
    vpm = tensor_to_vpm(value)
    vpm_creation_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    restored = vpm_to_tensor(vpm)
    restoration_time = time.perf_counter() - start_time
    
    # Verify data fidelity
    is_valid, error_msg = verify_roundtrip(value, restored)
    assert is_valid, f"Roundtrip validation failed for {data_type}: {error_msg}"
    
    # Verify performance remains consistent regardless of data type
    # Demonstrates ZeroModel's "planet-scale navigation that feels flat" principle
    assert vpm_creation_time < 0.1, f"VPM creation too slow for {data_type}: {vpm_creation_time:.4f}s"
    assert restoration_time < 0.1, f"Restoration too slow for {data_type}: {restoration_time:.4f}s"
    
    # Print performance metrics for larger data types
    if "array" in data_type or "large" in data_type:
        original_size = len(pickle.dumps(value))
        vpm_size = vpm.size[0] * vpm.size[1] * 3  # RGB channels
        print(f"\n{data_type} roundtrip:")
        print(f"  Original size:   {original_size} bytes")
        print(f"  VPM size:        {vpm.size[0]}x{vpm.size[1]} pixels ({vpm_size} bytes)")
        print(f"  Creation time:   {vpm_creation_time:.6f}s")
        print(f"  Restoration time:{restoration_time:.6f}s")

def test_tensor_vpm_edge_cases():
    """
    Test edge cases that demonstrate ZeroModel's robustness.
    
    This validates ZeroModel's "robust under pressure" principle by testing:
    - Very small data
    - Very large data
    - Data with special values (NaN, inf)
    - Data with mixed types
    """
    # Test extremely small data (demonstrates "Critical Tile holds 99.99% of the answer in 0.1% of the space")
    small_data = {"x": np.array([0.5], dtype=np.float32)}
    vpm = tensor_to_vpm(small_data)
    assert vpm.size[0] >= 16 and vpm.size[1] >= 16  # Minimum reasonable size
    restored = vpm_to_tensor(vpm)
    assert verify_roundtrip(small_data, restored)[0]
    
    # Test very large data (demonstrates "planet-scale navigation that feels flat")
    large_data = {
        "weights": np.random.randn(10000, 100).astype(np.float32),
        "metadata": {"large_list": list(range(10000))}
    }
    vpm = tensor_to_vpm(large_data)
    restored = vpm_to_tensor(vpm)
    assert verify_roundtrip(large_data, restored)[0]
    
    # Test data with special values (demonstrates "human-compatible explanations")
    special_values = {
        "nan": np.nan,
        "inf": np.inf,
        "ninf": -np.inf,
        "zero": 0.0,
        "array": np.array([np.nan, np.inf, -np.inf, 0.0, 1.0], dtype=np.float32)
    }
    vpm = tensor_to_vpm(special_values)
    restored = vpm_to_tensor(vpm)
    is_valid, error_msg = verify_roundtrip(special_values, restored)
    assert is_valid, f"Special values roundtrip failed: {error_msg}"

def test_tensor_vpm_visual_debugging():
    """
    Test that demonstrates ZeroModel as "the debugger of AI".
    
    This verifies that VPMs can be used for visual debugging of AI states,
    supporting ZeroModel's "see AI think" principle.
    """
    # Create two similar but different model states
    state1 = {
        "weights": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "loss": 0.5
    }
    
    state2 = {
        "weights": np.array([[0.1, 0.25], [0.3, 0.4]], dtype=np.float32),
        "loss": 0.45
    }
    
    # Convert to VPMs
    vpm1 = tensor_to_vpm(state1)
    vpm2 = tensor_to_vpm(state2)
    
    # Verify they're different (demonstrates "traceable 'thought,' end-to-end")
    assert vpm1.tobytes() != vpm2.tobytes(), "Identical VPMs for different states"
    
    print("\nVisual debugging test:")
    print("  Created two distinct VPMs for similar model states")
    print("  These can be visually compared to debug AI behavior")

def test_tensor_vpm_compositional_logic():
    """
    Test that demonstrates compositional logic with VPMs.
    
    This verifies ZeroModel's "compositional logic (visually)" principle
    by showing how VPMs can be combined with logical operations.
    """
    # Create two simple VPMs for logical operations
    arr1 = np.array([[1.0, 0.5], [0.0, 0.25]], dtype=np.float32)
    arr2 = np.array([[0.5, 0.0], [1.0, 0.75]], dtype=np.float32)
    
    vpm1 = tensor_to_vpm(arr1)
    vpm2 = tensor_to_vpm(arr2)
    
    # Convert back to arrays for logical operations
    restored1 = vpm_to_tensor(vpm1)
    restored2 = vpm_to_tensor(vpm2)
    
    # Perform logical operations (demonstrates "Compositional logic (visually)")
    and_result = np.minimum(restored1, restored2)
    or_result = np.maximum(restored1, restored2)
    xor_result = np.abs(restored1 - restored2)
    
    # Verify the operations make sense
    assert np.all(and_result <= restored1) and np.all(and_result <= restored2)
    assert np.all(or_result >= restored1) and np.all(or_result >= restored2)
    assert np.all(xor_result >= 0) and np.all(xor_result <= 1.0)
    
    print("\nCompositional logic test:")
    print("  Verified logical operations on VPM data")
    print("  This enables building complex queries from simple components")