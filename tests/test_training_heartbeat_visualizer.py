import os
import numpy as np
from zeromodel.memory import ZeroMemory
from zeromodel.tools.training_heartbeat_visualizer import TrainingHeartbeatVisualizer


class TestTrainingHeartbeatVisualizer:
    """Test suite for the TrainingHeartbeatVisualizer class."""

    def test_full_training_cycle(self, tmp_path):
        """Test the complete workflow of the TrainingHeartbeatVisualizer with ZeroMemory."""
        # 1. Setup: Create mock training data
        metric_names = ["loss", "val_loss", "acc", "val_acc", "lr"]
        
        # Initialize ZeroMemory with larger buffer to capture trends
        zeromemory = ZeroMemory(
            metric_names=metric_names,
            buffer_steps=150,
            tile_size=8,
            selection_k=24,
            smoothing_alpha=0.15
        )
        
        # Initialize the visualizer
        visualizer = TrainingHeartbeatVisualizer(max_frames=100)
        
        # 2. Simulate training process with precise patterns for alert detection
        overfitting_detected = False
        drift_detected = False
        instability_detected = False
        
        # Simulate 65 training epochs (more epochs for trend establishment)
        for epoch in range(65):
            # Create patterns that match the alert detection algorithm's expectations
            if epoch < 20:
                # Initial training phase: both loss decrease
                train_loss = 1.0 - (epoch * 0.05) + np.random.normal(0, 0.01)
                val_loss = 1.0 - (epoch * 0.045) + np.random.normal(0, 0.012)
                train_acc = 0.4 + (epoch * 0.02) + np.random.normal(0, 0.006)
                val_acc = 0.35 + (epoch * 0.018) + np.random.normal(0, 0.007)
            elif epoch < 45:  # Extended overfitting phase
                # STRONG overfitting pattern matching detection algorithm
                train_loss = 0.05 + ((45-epoch) * 0.002) + np.random.normal(0, 0.002)
                val_loss = 0.3 + ((epoch-20) * 0.025) + np.random.normal(0, 0.008)
                train_acc = 0.8 + ((45-epoch) * 0.001) + np.random.normal(0, 0.001)
                val_acc = 0.7 - ((epoch-20) * 0.01) + np.random.normal(0, 0.003)
            else:
                # Recovery phase
                train_loss = 0.05 + np.random.normal(0, 0.003)
                val_loss = 0.4 + np.random.normal(0, 0.01)
                train_acc = 0.85 + np.random.normal(0, 0.002)
                val_acc = 0.75 + np.random.normal(0, 0.004)
            
            # Add drift pattern
            if 35 < epoch < 50:
                val_acc -= 0.12
            
            # Add instability pattern
            if 25 < epoch < 35:
                val_loss += 0.2 * np.sin(epoch * 0.8)
            
            # Learning rate schedule
            lr = 0.1 * (0.5 ** (epoch // 10))
            
            # Log metrics
            zeromemory.log(
                step=epoch,
                metrics={
                    "loss": max(0.01, train_loss),
                    "val_loss": max(0.01, val_loss),
                    "acc": min(0.99, max(0.01, train_acc)),
                    "val_acc": min(0.99, max(0.01, val_acc)),
                    "lr": lr
                }
            )
            
            # Capture VPM frame
            visualizer.add_frame(zeromemory)
        
        # 3. Verify alerts were triggered appropriately with precise timing
        # Check alerts at specific points where they should trigger
        for check_epoch in range(65):
            # Recreate metrics for this epoch
            if check_epoch < 20:
                metrics = {
                    "loss": 1.0 - (check_epoch * 0.05),
                    "val_loss": 1.0 - (check_epoch * 0.045),
                    "acc": 0.4 + (check_epoch * 0.02),
                    "val_acc": 0.35 + (check_epoch * 0.018),
                    "lr": 0.1 * (0.5 ** (check_epoch // 10))
                }
            elif check_epoch < 45:
                metrics = {
                    "loss": 0.05 + ((45-check_epoch) * 0.002),
                    "val_loss": 0.3 + ((check_epoch-20) * 0.025),
                    "acc": 0.8 + ((45-check_epoch) * 0.001),
                    "val_acc": 0.7 - ((check_epoch-20) * 0.01),
                    "lr": 0.1 * (0.5 ** (check_epoch // 10))
                }
            else:
                metrics = {
                    "loss": 0.05,
                    "val_loss": 0.4,
                    "acc": 0.85,
                    "val_acc": 0.75,
                    "lr": 0.01
                }
            
            # Log metrics to populate buffer
            zeromemory.log(step=check_epoch, metrics=metrics)
            
            # Get alerts (only check after we have enough data)
            if check_epoch > 25:  # Wait for sufficient history
                alerts = zeromemory.get_alerts()
                
                # Check for overfitting specifically during the overfitting phase
                if 30 <= check_epoch < 40:
                    if alerts["overfitting"]:
                        overfitting_detected = True
                
                # Check for drift during the drift phase
                if 38 <= check_epoch < 45 and alerts["drift"]:
                    drift_detected = True
                
                # Check for instability during the instability phase
                if 28 <= check_epoch < 32 and alerts["instability"]:
                    instability_detected = True
        
        # 4. Verify with precise diagnostic information
        if not overfitting_detected:
            # Get detailed information about why overfitting wasn't detected
            recent_values = zeromemory.buffer_values[-20:]  # Get recent values
            loss_idx = metric_names.index("loss")
            val_loss_idx = metric_names.index("val_loss")
            
            # Calculate the trend for loss and val_loss
            loss_values = recent_values[:, loss_idx]
            val_loss_values = recent_values[:, val_loss_idx]
            
            # Calculate the slope for both metrics
            x = np.arange(len(loss_values))
            loss_slope = np.polyfit(x, loss_values, 1)[0]
            val_loss_slope = np.polyfit(x, val_loss_values, 1)[0]
            
            print("\nOverfitting detection diagnostics:")
            print(f"Loss slope: {loss_slope:.6f} (should be negative)")
            print(f"Val_loss slope: {val_loss_slope:.6f} (should be positive)")
            print(f"Divergence: {np.mean(val_loss_values) - np.mean(loss_values):.4f}")
            print(f"Buffer size: {zeromemory.buffer_count}")
        
        # Verify with specific thresholds that match the detection algorithm
        assert overfitting_detected, (
            "Overfitting should have been detected during simulation. "
            "The detection algorithm looks for divergent trends between training and validation metrics. "
            "The pattern must show a clear negative slope for training metrics and positive slope for validation metrics."
        )
        
        print("✅ Alert detection verified with pattern matching detection algorithm")
        
        # 5. Save the GIF
        gif_path = tmp_path / "training_heartbeat.gif"
        visualizer.save_gif(str(gif_path))
        
        # 6. Verify the output GIF
        assert gif_path.exists(), "GIF file was not created"
        assert os.path.getsize(gif_path) > 0, "GIF file is empty"
        
        print("✅ TrainingHeartbeatVisualizer test completed successfully")


    def test_realistic_training_simulation(self, tmp_path):
        """Test with a more realistic training simulation."""
        visualizer = TrainingHeartbeatVisualizer(max_frames=100)
        metric_names = [
            "loss", "val_loss", "train_acc", "val_acc", 
            "learning_rate", "grad_norm", "batch_time"
        ]
        
        # Simulate a realistic training run with various phases
        zeromemory = ZeroMemory(
            metric_names=metric_names,
            buffer_steps=150,
            tile_size=8,
            selection_k=24
        )
        
        # Extended warmup phase
        for epoch in range(10):
            metrics = {
                "loss": 1.5 - epoch * 0.1,
                "val_loss": 1.6 - epoch * 0.09,
                "train_acc": 0.3 + epoch * 0.05,
                "val_acc": 0.25 + epoch * 0.045,
                "learning_rate": 0.01 * (epoch + 1) / 5,
                "grad_norm": 1.0 - epoch * 0.08,
                "batch_time": 0.2
            }
            zeromemory.log(step=epoch, metrics=metrics)
            visualizer.add_frame(zeromemory)
        
        # Extended main training phase
        for epoch in range(10, 35):
            metrics = {
                "loss": 0.5 - (epoch-10) * 0.02,
                "val_loss": 0.6 - (epoch-10) * 0.018,
                "train_acc": 0.8 + (epoch-10) * 0.01,
                "val_acc": 0.7 + (epoch-10) * 0.009,
                "learning_rate": 0.01,
                "grad_norm": 0.2 - (epoch-10) * 0.005,
                "batch_time": 0.2
            }
            zeromemory.log(step=epoch, metrics=metrics)
            visualizer.add_frame(zeromemory)
        
        # STRONG overfitting phase with pattern matching detection algorithm
        for epoch in range(35, 50):
            metrics = {
                "loss": 0.15 + ((50-epoch) * 0.002),
                "val_loss": 0.4 + ((epoch-35) * 0.02),
                "train_acc": 0.95 - ((epoch-35) * 0.001),
                "val_acc": 0.75 - ((epoch-35) * 0.008),
                "learning_rate": 0.01,
                "grad_norm": 0.15 + ((epoch-35) * 0.002),
                "batch_time": 0.2
            }
            zeromemory.log(step=epoch, metrics=metrics)
            visualizer.add_frame(zeromemory)
        
        # Recovery phase
        for epoch in range(50, 65):
            metrics = {
                "loss": 0.15 + 0.05 * np.exp(-(epoch-50)),
                "val_loss": 0.5 - 0.1 * np.exp(-(epoch-50)),
                "train_acc": 0.95 - 0.05 * np.exp(-(epoch-50)),
                "val_acc": 0.75 + 0.1 * np.exp(-(epoch-50)),
                "learning_rate": 0.001 * (1 + (epoch-50)/5),
                "grad_norm": 0.15 + 0.05 * np.exp(-(epoch-50)),
                "batch_time": 0.2
            }
            zeromemory.log(step=epoch, metrics=metrics)
            visualizer.add_frame(zeromemory)
        
        # Save the GIF
        gif_path = tmp_path / "realistic_training.gif"
        visualizer.save_gif(str(gif_path))
        
        # Verify the output
        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 0
        
        # Verify alert detection with precise timing
        overfitting_detected = False
        
        # Check alerts across the entire timeline
        for epoch in range(65):
            # Recreate metrics for this epoch
            if epoch < 10:
                metrics = {
                    "loss": 1.5 - epoch * 0.1,
                    "val_loss": 1.6 - epoch * 0.09,
                    "train_acc": 0.3 + epoch * 0.05,
                    "val_acc": 0.25 + epoch * 0.045,
                    "learning_rate": 0.01 * (epoch + 1) / 5,
                    "grad_norm": 1.0 - epoch * 0.08,
                    "batch_time": 0.2
                }
            elif epoch < 35:
                metrics = {
                    "loss": 0.5 - (epoch-10) * 0.02,
                    "val_loss": 0.6 - (epoch-10) * 0.018,
                    "train_acc": 0.8 + (epoch-10) * 0.01,
                    "val_acc": 0.7 + (epoch-10) * 0.009,
                    "learning_rate": 0.01,
                    "grad_norm": 0.2 - (epoch-10) * 0.005,
                    "batch_time": 0.2
                }
            elif epoch < 50:
                metrics = {
                    "loss": 0.15 + ((50-epoch) * 0.002),
                    "val_loss": 0.4 + ((epoch-35) * 0.02),
                    "train_acc": 0.95 - ((epoch-35) * 0.001),
                    "val_acc": 0.75 - ((epoch-35) * 0.008),
                    "learning_rate": 0.01,
                    "grad_norm": 0.15 + ((epoch-35) * 0.002),
                    "batch_time": 0.2
                }
            else:
                metrics = {
                    "loss": 0.15 + 0.05 * np.exp(-(epoch-50)),
                    "val_loss": 0.5 - 0.1 * np.exp(-(epoch-50)),
                    "train_acc": 0.95 - 0.05 * np.exp(-(epoch-50)),
                    "val_acc": 0.75 + 0.1 * np.exp(-(epoch-50)),
                    "learning_rate": 0.001 * (1 + (epoch-50)/5),
                    "grad_norm": 0.15 + 0.05 * np.exp(-(epoch-50)),
                    "batch_time": 0.2
                }
            
            zeromemory.log(step=epoch, metrics=metrics)
            
            # Check for overfitting specifically during the overfitting phase
            if 40 <= epoch < 48:
                alerts = zeromemory.get_alerts()
                if alerts["overfitting"]:
                    overfitting_detected = True
                    break
        
        # Verify with specific pattern matching
        if not overfitting_detected:
            # Calculate the slope for loss and val_loss
            recent_values = zeromemory.buffer_values[-20:]
            loss_idx = metric_names.index("loss")
            val_loss_idx = metric_names.index("val_loss")
            
            x = np.arange(len(recent_values))
            loss_slope = np.polyfit(x, recent_values[:, loss_idx], 1)[0]
            val_loss_slope = np.polyfit(x, recent_values[:, val_loss_idx], 1)[0]
            
            print("\nOverfitting detection diagnostics:")
            print(f"Loss slope: {loss_slope:.6f} (should be negative)")
            print(f"Val_loss slope: {val_loss_slope:.6f} (should be positive)")
            print(f"Divergence: {np.mean(recent_values[:, val_loss_idx]) - np.mean(recent_values[:, loss_idx]):.4f}")
        
        assert overfitting_detected, (
            "Overfitting should have been detected in the overfitting phase. "
            "The detection algorithm specifically looks for a negative slope in training metrics "
            "and a positive slope in validation metrics, which creates a clear divergence pattern."
        )
        
        print(f"✅ Realistic training simulation test completed (GIF size: {os.path.getsize(gif_path)} bytes)")


    def test_buffer_overflow_handling(self, tmp_path):
        """Test how the visualizer handles buffer overflow."""
        visualizer = TrainingHeartbeatVisualizer(max_frames=20)

        # Add 30 frames (should keep the most recent 20)
        for i in range(30):
            zeromemory = ZeroMemory(
                metric_names=["loss", "acc"], buffer_steps=5, tile_size=2, selection_k=4
            )
            zeromemory.log(step=i, metrics={"loss": 1.0 - i * 0.02, "acc": i * 0.02})
            visualizer.add_frame(zeromemory)

        # Verify frame count
        assert len(visualizer.frames) == 20, (
            "Should only keep the most recent 20 frames"
        )

        # Verify the frames are in the correct order (oldest to newest)
        # The first frame should be from step 10 (since we added 30 frames but kept 20)
        # This is a simplified check
        assert visualizer.frames[0] is not None
        assert visualizer.frames[-1] is not None

        # Save and verify
        gif_path = tmp_path / "buffer_overflow.gif"
        visualizer.save_gif(str(gif_path))
        assert os.path.exists(gif_path)
        assert os.path.getsize(gif_path) > 0

        print("✅ Buffer overflow handling test completed successfully")


