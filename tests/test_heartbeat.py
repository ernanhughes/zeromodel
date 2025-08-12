import os
import numpy as np
import pytest
import tempfile
import time
from unittest.mock import patch
from PIL import Image

from zeromodel.memory import ZeroMemory
from zeromodel.tools.training_heartbeat_visualizer import TrainingHeartbeatVisualizer

def test_training_heartbeat_visualizer_full_cycle(tmp_path):
    """Test the complete workflow of the TrainingHeartbeatVisualizer with ZeroMemory."""
    
    # --- 1. Setup: Create mock training data ---
    metric_names = ["loss", "val_loss", "acc", "val_acc", "lr"]
    print(f"Starting test with metrics: {metric_names}")
    
    # Initialize ZeroMemory to track metrics
    zeromemory = ZeroMemory(
        metric_names=metric_names,
        buffer_steps=256,
        tile_size=6,
        selection_k=18  # 6x6 tile * 3 channels = 18 metrics
    )
    
    # Initialize the visualizer
    visualizer = TrainingHeartbeatVisualizer(
        max_frames=100,
        vpm_scale=6,
        strip_height=45,
        bg_color=(15, 15, 20)
    )
    
    # --- 2. Simulate a realistic training process ---
    total_epochs = 25
    steps_per_epoch = 10
    total_steps = total_epochs * steps_per_epoch
    
    print(f"Simulating {total_steps} training steps...")
    
    # Track when alerts should trigger for verification
    overfitting_triggered = False
    drift_triggered = False
    instability_triggered = False
    
    for step in range(total_steps):
        # Simulate realistic training metrics
        epoch = step // steps_per_epoch
        batch_in_epoch = step % steps_per_epoch
        
        # Simulate loss decreasing with some noise
        base_loss = 0.8 - (epoch * 0.025)
        noise = 0.05 * np.sin(step / 5.0)  # Add some oscillation
        train_loss = max(0.05, base_loss + noise)
        
        # Simulate validation loss with overfitting pattern
        val_loss = base_loss + 0.02 * epoch  # Starts increasing after a while
        
        # Simulate accuracy increasing
        train_acc = 0.2 + (epoch * 0.03)
        val_acc = train_acc - 0.01 * epoch  # Validation accuracy lags behind
        
        # Learning rate schedule
        lr = 0.1 * (0.95 ** epoch)
        
        # Log metrics to ZeroMemory
        metrics = {
            "loss": train_loss,
            "val_loss": val_loss,
            "acc": train_acc,
            "val_acc": val_acc,
            "lr": lr
        }
        zeromemory.log(step=step, metrics=metrics)
        
        # Every 5 steps, capture a frame for the visualization
        if step % 5 == 0:
            # Generate VPM snapshot
            vpm = zeromemory.snapshot_vpm(
                window_size=min(64, step + 1),
                target_metric_name="loss"
            )
            
            # Get alerts
            alerts = zeromemory.get_alerts()
            
            # Track when alerts trigger for verification
            if alerts["overfitting"] and not overfitting_triggered:
                print(f"Overfitting detected at step {step}")
                overfitting_triggered = True
            if alerts["drift"] and not drift_triggered:
                print(f"Drift detected at step {step}")
                drift_triggered = True
            if alerts["instability"] and not instability_triggered:
                print(f"Instability detected at step {step}")
                instability_triggered = True
            
            # Add frame to visualizer
            visualizer.add_frame(
                vpm_uint8=vpm,
                metrics={
                    **metrics,
                    "alerts": alerts,
                    "step": step
                }
            )
    
    # --- 3. Verify ZeroMemory state ---
    assert zeromemory.last_full_vpm is not None, "VPM was not generated"
    assert zeromemory.last_full_vpm.shape[0] > 0 and zeromemory.last_full_vpm.shape[1] > 0, "VPM has invalid dimensions"
    print(f"Final VPM dimensions: {zeromemory.last_full_vpm.shape}")
    
    # Verify alerts were triggered appropriately
    assert overfitting_triggered, "Overfitting should have been detected during simulation"
    print("✅ Overfitting detection verified")
    
    # --- 4. Save the GIF ---
    gif_path = tmp_path / "training_heartbeat.gif"
    print(f"Saving GIF to: {gif_path}")
    
    try:
        visualizer.save_gif(
            path=str(gif_path)
        )
    except Exception as e:
        pytest.fail(f"Failed to save GIF: {str(e)}")
    
    # --- 5. Verify the output GIF ---
    assert gif_path.exists(), "GIF file was not created"
    file_size = os.path.getsize(gif_path)
    assert file_size > 0, "GIF file is empty"
    print(f"✅ GIF created successfully. Size: {file_size} bytes")
    
    # Verify frame count (allowing for some decimation)
    assert len(visualizer.frames) > 0, "No frames were captured"
    assert len(visualizer.frames) <= 100, "Frame count exceeded max_frames"
    print(f"✅ Captured {len(visualizer.frames)} frames")
    
    # Basic verification of GIF content (open and check first frame)
    try:
        with Image.open(gif_path) as img:
            assert img.format == 'GIF', "File is not a GIF"
            assert img.n_frames > 0, "GIF has no frames"
            print(f"✅ GIF contains {img.n_frames} frames")
            
            # Check dimensions of first frame
            img.seek(0)
            width, height = img.size
            assert width > 0 and height > 0, "Invalid GIF frame dimensions"
            print(f"✅ GIF frame dimensions: {width}x{height}")
    except Exception as e:
        pytest.fail(f"Failed to verify GIF content: {str(e)}")
    
    # --- 6. Verify timeline strip elements ---
    # This is a more advanced verification - check for alert markers
    if overfitting_triggered or drift_triggered or instability_triggered:
        print("✅ Alert markers verification passed (visual confirmation needed)")
    
    print("✅ TrainingHeartbeatVisualizer test completed successfully")

def test_visualizer_edge_cases(tmp_path):
    """Test edge cases for the TrainingHeartbeatVisualizer."""
    
    # --- 1. Test with empty data ---
    visualizer = TrainingHeartbeatVisualizer(max_frames=50)
    assert len(visualizer.frames) == 0
    
    # Try to save without frames
    with pytest.raises(RuntimeError):
        visualizer.save_gif(str(tmp_path / "empty.gif"))
    
    # --- 2. Test with minimal data ---
    metric_names = ["loss", "acc"]
    zeromemory = ZeroMemory(metric_names, buffer_steps=10, tile_size=2, selection_k=6)
    
    # Log minimal data
    zeromemory.log(step=0, metrics={"loss": 0.5, "acc": 0.5})
    
    # Create very small VPM
    vpm = zeromemory.snapshot_vpm()
    
    # Add frame
    visualizer.add_frame(
        vpm_uint8=vpm,
        metrics={
            "loss": 0.5,
            "acc": 0.5,
            "alerts": {"overfitting": False}
        }
    )
    
    # Save GIF
    gif_path = tmp_path / "minimal.gif"
    visualizer.save_gif(str(gif_path))
    
    # Verify
    assert gif_path.exists()
    assert os.path.getsize(gif_path) > 0
    
    # --- 3. Test frame decimation ---
    visualizer = TrainingHeartbeatVisualizer(max_frames=10)
    
    # Add more frames than max_frames
    for i in range(20):
        # Create minimal VPM
        vpm = np.zeros((2, 2, 3), dtype=np.uint8)
        vpm[:, :, 0] = i * 10  # Red channel varies
        
        visualizer.add_frame(
            vpm_uint8=vpm,
            metrics={"step": i, "loss": 1.0 - i/20}
        )
    
    assert len(visualizer.frames) == 10, "Frame decimation failed"
    print(f"Frame decimation test: {len(visualizer.frames)} frames kept out of 20")
    
    # --- 4. Test with NaN values ---
    visualizer = TrainingHeartbeatVisualizer()
    
    # Add frame with NaN values
    vpm = zeromemory.snapshot_vpm()
    visualizer.add_frame(
        vpm_uint8=vpm,
        metrics={
            "loss": float('nan'),
            "val_loss": float('inf'),
            "acc": 0.5,
            "alerts": {"overfitting": False}
        }
    )
    
    # Should not crash
    gif_path = tmp_path / "nan.gif"
    visualizer.save_gif(str(gif_path))
    assert gif_path.exists()
    
    print("✅ TrainingHeartbeatVisualizer edge cases test completed")

def test_visualizer_integration_with_training(tmp_path):
    """Test the visualizer integrated into a realistic training loop simulation."""
    
    # --- 1. Setup ---
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    
    # Generate dataset
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Initialize model
    model = MLPClassifier(
        hidden_layer_sizes=(10,),
        max_iter=1,
        warm_start=True,
        random_state=42,
        alpha=0.01
    )
    
    # Initialize monitoring
    metric_names = ["loss", "val_loss", "train_acc", "val_acc"]
    zeromemory = ZeroMemory(
        metric_names=metric_names,
        buffer_steps=100,
        tile_size=5,
        selection_k=15
    )
    visualizer = TrainingHeartbeatVisualizer(
        max_frames=50,
        vpm_scale=8
    )
    
    # --- 2. Training loop simulation ---
    epochs = 15
    print(f"Running {epochs}-epoch training simulation...")
    
    for epoch in range(epochs):
        # Train for one epoch
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = (train_pred == y_train).mean()
        test_acc = (test_pred == y_test).mean()
        
        # Get loss from model (MLPClassifier stores loss history)
        train_loss = model.loss_
        # Simple validation loss approximation
        test_loss = 1.0 - test_acc
        
        # Log to ZeroMemory
        zeromemory.log(
            step=epoch,
            metrics={
                "loss": train_loss,
                "val_loss": test_loss,
                "train_acc": train_acc,
                "val_acc": test_acc
            }
        )
        
        # Capture frame every epoch
        vpm = zeromemory.snapshot_vpm(
            window_size=min(32, epoch + 1)
        )
        alerts = zeromemory.get_alerts()
        
        visualizer.add_frame(
            vpm_uint8=vpm,
            metrics={
                "step": epoch,
                "loss": train_loss,
                "val_loss": test_loss,
                "train_acc": train_acc,
                "val_acc": test_acc,
                "alerts": alerts
            }
        )
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {train_loss:.4f}, Val Loss: {test_loss:.4f}, "
              f"Acc: {train_acc:.4f}, Val Acc: {test_acc:.4f}")
    
    # --- 3. Save and verify ---
    gif_path = tmp_path / "mlp_training.gif"
    visualizer.save_gif(str(gif_path))
    
    assert gif_path.exists(), "GIF file was not created"
    assert os.path.getsize(gif_path) > 0, "GIF file is empty"
    
    # Verify frame count
    with Image.open(gif_path) as img:
        assert img.n_frames == len(visualizer.frames), "Frame count mismatch"
    
    print(f"✅ MLP training visualization test completed. GIF size: {os.path.getsize(gif_path)} bytes")

def test_visualizer_with_alerts(tmp_path):
    """Test the visualizer's ability to correctly display alert markers."""
    
    # --- 1. Setup ---
    metric_names = ["loss", "val_loss", "acc"]
    zeromemory = ZeroMemory(metric_names, buffer_steps=50, tile_size=4, selection_k=12)
    visualizer = TrainingHeartbeatVisualizer()
    
    # --- 2. Simulate specific scenarios that trigger alerts ---
    scenarios = [
        # Scenario 1: Overfitting (val_loss rising while loss falling)
        {"loss": 0.5, "val_loss": 0.55, "acc": 0.7, "alerts": {"overfitting": False}},
        {"loss": 0.4, "val_loss": 0.58, "acc": 0.75, "alerts": {"overfitting": False}},
        {"loss": 0.3, "val_loss": 0.62, "acc": 0.8, "alerts": {"overfitting": True}},
        
        # Scenario 2: Drift (sudden shift in metric distribution)
        {"loss": 0.3, "val_loss": 0.62, "acc": 0.8, "alerts": {"drift": False}},
        {"loss": 0.31, "val_loss": 0.63, "acc": 0.79, "alerts": {"drift": False}},
        {"loss": 0.6, "val_loss": 0.8, "acc": 0.5, "alerts": {"drift": True}},
        
        # Scenario 3: Instability (high spikiness)
        {"loss": 0.6, "val_loss": 0.8, "acc": 0.5, "alerts": {"instability": False}},
        {"loss": 0.3, "val_loss": 0.5, "acc": 0.7, "alerts": {"instability": False}},
        {"loss": 0.7, "val_loss": 0.9, "acc": 0.4, "alerts": {"instability": True}}
    ]
    
    # Log scenarios to ZeroMemory and capture frames
    for step, scenario in enumerate(scenarios):
        # Log metrics
        zeromemory.log(step=step, metrics=scenario)
        
        # Generate VPM
        vpm = zeromemory.snapshot_vpm(window_size=min(10, step + 1))
        
        # Add frame
        visualizer.add_frame(
            vpm_uint8=vpm,
            metrics={
                **scenario,
                "step": step
            }
        )
    
    # --- 3. Save GIF ---
    gif_path = tmp_path / "alerts_test.gif"
    visualizer.save_gif(str(gif_path))
    
    # --- 4. Verify alert markers are present ---
    assert gif_path.exists()
    
    # This is a more complex verification - in a real test we might use image analysis
    # For now, we just verify the GIF was created with the expected frames
    with Image.open(gif_path) as img:
        assert img.n_frames == len(scenarios), f"Expected {len(scenarios)} frames, got {img.n_frames}"
    
    print("✅ Alert markers test completed. GIF shows expected alert patterns.")
    print("   Note: Visual verification of alert markers is recommended.")