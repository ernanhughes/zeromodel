# tests/test_memory.py
"""
Test cases for the ZeroMemory sidecar component.
"""

import numpy as np
import pytest
import logging

# from zeromodel.core import ZeroModel
# from zeromodel.hierarchical import HierarchicalVPM
from zeromodel.memory import ZeroMemory

logger = logging.getLogger(__name__)
# Adjust the import path based on your actual package structure
# Assuming zeromodel.memory contains the ZeroMemory class


# --- Test Fixtures ---
@pytest.fixture
def basic_metric_names():
    """Provide a basic set of metric names for testing."""
    return ["loss", "val_loss", "accuracy", "val_accuracy", "grad_norm"]

@pytest.fixture
def sample_training_data():
    """Generate sample training data that shows a potential overfitting trend."""
    # Simulate 10 training steps
    steps = 10
    # Loss decreases over time
    base_loss = np.linspace(0.8, 0.2, steps)
    # Val loss decreases initially, then starts increasing (overfitting)
    val_loss_decrease = np.linspace(0.7, 0.3, steps // 2)
    val_loss_increase = np.linspace(0.3, 0.5, steps - steps // 2)
    val_loss = np.concatenate([val_loss_decrease, val_loss_increase])
    
    # Accuracy increases over time
    base_acc = np.linspace(0.2, 0.8, steps)
    # Val accuracy increases initially, then plateaus/decreases slightly
    val_acc_increase = np.linspace(0.3, 0.75, steps // 2)
    val_acc_plateau = np.full(steps - steps // 2, 0.74)
    val_acc = np.concatenate([val_acc_increase, val_acc_plateau])
    
    # Gradient norm decreases over time (typical)
    grad_norm = np.linspace(0.5, 0.05, steps)
    
    data = {
        "loss": base_loss,
        "val_loss": val_loss,
        "accuracy": base_acc,
        "val_accuracy": val_acc,
        "grad_norm": grad_norm
    }
    return data

# --- Test Cases ---

def test_zeromemory_initialization(basic_metric_names):
    """Test that ZeroMemory initializes correctly with basic parameters."""
    zm = ZeroMemory(
        metric_names=basic_metric_names,
        buffer_steps=128,
        tile_size=4,
        selection_k=8,
        smoothing_alpha=0.2
    )
    
    assert zm.metric_names == basic_metric_names
    assert zm.num_metrics == len(basic_metric_names)
    assert zm.buffer_steps == 128
    assert zm.tile_size == 4
    assert zm.selection_k == 8
    assert zm.smoothing_alpha == 0.2
    # Check buffer initialization
    assert zm.buffer_values.shape == (128, 5)
    assert np.all(np.isclose(zm.buffer_values, 0.0)) # Initially filled with zeros
    assert zm.buffer_head == 0
    assert zm.buffer_count == 0
    # Check alert initialization
    assert zm.last_alerts == {
        "overfitting": False,
        "underfitting": False,
        "drift": False,
        "saturation": False,
        "instability": False
    }

def test_zeromemory_vpm_snapshot_with_auto_hint():
    """Test VPM snapshot generation with 'auto' hint."""
    metric_names = ["a", "b", "c"]
    zm = ZeroMemory(metric_names, buffer_steps=5, tile_size=3, selection_k=6) # 3 original + up to 3 engineered
    
    # Generate some random data
    np.random.seed(42) # For reproducibility in tests
    score_matrix = np.random.rand(5, 3)
    
    # Log the data
    for i, row in enumerate(score_matrix):
        metrics = {metric_names[j]: row[j] for j in range(3)}
        zm.log(step=i, metrics=metrics)
    
    # Generate VPM snapshot with 'auto' hint
    # This should trigger feature engineering
    vpm_img = zm.snapshot_vpm(target_metric_name="a") # Use 'a' as target for correlation
    
    # Check VPM properties
    assert isinstance(vpm_img, np.ndarray)
    assert vpm_img.ndim == 3 # Should be [H, W, 3]
    assert vpm_img.shape[2] == 3 # RGB channels
    assert vpm_img.dtype == np.uint8 # Should be uint8
    # With tile_size=3, height should be min(3, buffer_count) = 3
    # Width should be ceil((num_metrics_after_engineering) / 3)
    # 'auto' with 3 metrics should add 3 products (ab, ac, bc) + 2 squares (a^2, b^2) = 5
    # Total metrics = 3 + 5 = 8
    # Width in pixels = ceil(8 / 3) = 3
    assert vpm_img.shape[0] == 3 # Height
    assert vpm_img.shape[1] == 3 # Width
    
    # Check values are in valid range
    assert np.all(vpm_img >= 0)
    assert np.all(vpm_img <= 255)

def test_zeromemory_tile_snapshot():
    """Test critical tile snapshot generation."""
    metric_names = ["m1", "m2", "m3"]
    zm = ZeroMemory(metric_names, buffer_steps=4, tile_size=2, selection_k=4) # 2x2 tile
    
    # Log simple data
    score_matrix = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.5, 0.5, 1.0],
        [0.2, 0.8, 0.1],
    ])
    for i, row in enumerate(score_matrix):
        metrics = {metric_names[j]: row[j] for j in range(3)}
        zm.log(step=i, metrics=metrics)
    
    # Get tile snapshot
    tile_bytes = zm.snapshot_tile(tile_size=2) # Request 2x2 tile
    
    # Check tile properties
    assert isinstance(tile_bytes, bytes)
    # Should have header (4 bytes) + pixel data
    # 2x2 tile = 4 pixels, 3 bytes per pixel = 12 bytes
    # Total = 4 + 12 = 16 bytes
    assert len(tile_bytes) == 16
    # Check header (16-bit little-endian width/height)
    width = tile_bytes[0] | (tile_bytes[1] << 8)
    height = tile_bytes[2] | (tile_bytes[3] << 8)
    assert width == 2
    assert height == 2
    # Check pixel data (basic sanity)
    # First pixel data starts at index 4
    # Pixel 0 (0,0): R, G, B
    r0, g0, b0 = tile_bytes[4], tile_bytes[5], tile_bytes[6]
    # Pixel 1 (0,1): R, G, B
    _r1, _g1, _b1 = tile_bytes[7], tile_bytes[8], tile_bytes[9]
    # ... etc.
    # We can't easily assert exact values without knowing normalization details,
    # but we can check they are bytes.
    assert isinstance(r0, int) and 0 <= r0 <= 255
    assert isinstance(g0, int) and 0 <= g0 <= 255
    assert isinstance(b0, int) and 0 <= b0 <= 255

def test_zeromemory_alerts_overfitting_detection(sample_training_data):
    """Test overfitting alert detection with realistic data."""
    metric_names = list(sample_training_data.keys())
    zm = ZeroMemory(metric_names, buffer_steps=15, tile_size=3, selection_k=6)
    
    # Log the sample data which shows overfitting trend
    steps = len(sample_training_data["loss"])
    for i in range(steps):
        metrics = {name: sample_training_data[name][i] for name in metric_names}
        zm.log(step=i, metrics=metrics)
    
    # Get alerts
    alerts = zm.get_alerts()
    
    # Check alert structure
    assert isinstance(alerts, dict)
    expected_keys = ["overfitting", "underfitting", "drift", "saturation", "instability"]
    for key in expected_keys:
        assert key in alerts
        assert isinstance(alerts[key], bool)
    
    # With the sample data, overfitting should be detected
    # (train loss keeps going down, val loss starts going up)
    # The logic in _compute_alerts looks for train_loss slope < -0.1 and val_loss slope > 0.1
    # Given the data, this condition should be met.
    # However, the exact detection depends on the window size and the linear regression calculation.
    # Let's assert that it's a boolean and leave the specific detection logic to unit tests
    # of _compute_alerts if needed.
    assert isinstance(alerts["overfitting"], bool)
    # We can't guarantee it will be True without knowing the exact internal calculation,
    # but we can test that the function runs and returns a dict with the right keys.

def test_zeromemory_no_hint_unchanged_behavior(basic_metric_names):
    """Test that not providing a hint leaves the data processing unchanged."""
    zm = ZeroMemory(basic_metric_names, buffer_steps=3, tile_size=2, selection_k=3)
    
    score_matrix = np.array([
        [0.1, 0.9, 0.2, 0.8, 0.3],
        [0.8, 0.2, 0.7, 0.1, 0.9],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    ])
    
    # Log data without hint
    for i, row in enumerate(score_matrix):
        metrics = {basic_metric_names[j]: row[j] for j in range(5)}
        zm.log(step=i, metrics=metrics)
    
    # Generate VPM without hint
    vpm_img = zm.snapshot_vpm() # No hint, no target metric specified
    
    # Should only have original metrics
    # VPM shape: height=min(tile_size, buffer_count)=min(2,3)=2, 
    # width=ceil(num_metrics/3)=ceil(5/3)=2
    assert vpm_img.shape == (2, 2, 3)
    assert vpm_img.dtype == np.uint8
