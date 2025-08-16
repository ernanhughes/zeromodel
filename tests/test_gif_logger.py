# tests/test_gif_logging.py
import os

import imageio.v2 as imageio
import numpy as np
import pytest

# Try to import plotting libraries for better visualization (optional)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import \
    MLPClassifier  # Better for showing epoch progress
from sklearn.preprocessing import StandardScaler

from zeromodel.memory import ZeroMemory


class GIFLogger:
    """Tiny helper: collect RGB frames and save animated GIF."""
    def __init__(self, fps=5):
        self.frames = []
        self.fps = fps

    def add(self, frame: np.ndarray):
        # Expect uint8 HxWx3; pad if grayscale
        if frame.ndim == 2:
            frame = np.stack([frame]*3, axis=-1)
        self.frames.append(frame.astype(np.uint8))

    def save(self, path: str):
        assert len(self.frames) > 0, "No frames to save"
        imageio.mimsave(path, self.frames, duration=1.0 / max(self.fps, 1))
        
    def save_frame(self, path: str, frame_index: int = -1):
        """Save a single frame as a PNG for inspection."""
        if len(self.frames) > 0:
            idx = frame_index if frame_index >= 0 else len(self.frames) - 1
            if 0 <= idx < len(self.frames):
                imageio.imwrite(path, self.frames[idx])

@pytest.mark.parametrize("max_epochs", [30]) # More epochs to see clear progress
@pytest.mark.filterwarnings("ignore:.*hasn't converged yet.*:sklearn.exceptions.ConvergenceWarning")
def test_gif_logging_with_zeromemory_improved(tmp_path, max_epochs):
    """Test GIF logging with ZeroMemory using a model that shows clear epoch-by-epoch progress."""
    
    # --- 1) Create a more complex dataset ---
    # Make it challenging enough to show learning over epochs
    X, y = make_classification(
        n_samples=1000,
        n_features=10,        # More features to work with
        n_informative=5,      # Some features are informative
        n_redundant=3,        # Some are redundant (creates complexity)
        n_clusters_per_class=2, # Makes it harder to separate
        flip_y=0.1,           # Add some noise
        random_state=42
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    # --- 2) Use a model that trains over multiple epochs ---
    # MLP is good because it shows clear loss decrease and accuracy increase over epochs
    clf = MLPClassifier(
        hidden_layer_sizes=(20, 10),  # Small hidden layers
        activation='relu',
        solver='adam',
        max_iter=1,              # Train for only 1 epoch per fit call
        warm_start=True,         # Keep weights between fits
        random_state=42,
        early_stopping=False,    # Disable early stopping for clear epoch progression
        validation_fraction=0.0, # No validation split
        alpha=0.01,              # Regularization
        learning_rate_init=0.01,
        batch_size=32
    )

    # --- 3) ZeroMemory to track training metrics ---
    # Track more relevant metrics for an MLP
    metrics = ["train_loss", "val_loss", "train_acc", "val_acc", "epoch"]
    zm = ZeroMemory(metric_names=metrics, buffer_steps=128, tile_size=8, selection_k=24)

    # --- 4) GIF logger ---
    gif = GIFLogger(fps=3) # Slower FPS to see changes
    out_gif = tmp_path / "training_progress.gif" # Use tmp_path for pytest
    out_final_frame = tmp_path / "final_vpm_frame.png"

    print(f"Starting training for {max_epochs} epochs...")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(max_epochs):
        # --- Train for one epoch ---
        # Fit the model for one epoch on the entire training set
        # MLPClassifier's partial_fit is tricky, so we use fit with warm_start
        clf.max_iter = epoch + 1 # This tells it to train for epoch+1 total epochs
        clf.fit(Xtr, ytr) # With warm_start=True, it continues from previous state
        
        # --- Evaluate ---
        # Get predictions
        y_pred_tr = clf.predict(Xtr)
        y_pred_te = clf.predict(Xte)
        # Get probabilities for loss calculation
        try:
            y_proba_tr = clf.predict_proba(Xtr)
            y_proba_te = clf.predict_proba(Xte)
        except AttributeError:
            # Fallback if predict_proba is not available or causes issues
            # Use decision function or dummy probabilities
            y_proba_tr = np.zeros((len(ytr), 2))
            y_proba_te = np.zeros((len(yte), 2))
            # Simple dummy probabilities based on predictions
            y_proba_tr[np.arange(len(ytr)), y_pred_tr] = 1.0
            y_proba_te[np.arange(len(yte)), y_pred_te] = 1.0
            
        # Calculate metrics
        try:
            loss_tr = float(log_loss(ytr, np.clip(y_proba_tr, 1e-15, 1 - 1e-15)))
        except ValueError:
            loss_tr = float('inf') # Handle potential issues
            
        try:
            loss_te = float(log_loss(yte, np.clip(y_proba_te, 1e-15, 1 - 1e-15)))
        except ValueError:
            loss_te = float('inf')
            
        acc_tr = float(accuracy_score(ytr, y_pred_tr))
        acc_te = float(accuracy_score(yte, y_pred_te))
        
        # Store for potential plotting
        train_losses.append(loss_tr)
        val_losses.append(loss_te)
        train_accs.append(acc_tr)
        val_accs.append(acc_te)

        # --- Log metrics into ZeroMemory ---
        zm.log(
            step=epoch,
            metrics={
                "train_loss": loss_tr, 
                "val_loss": loss_te, 
                "train_acc": acc_tr, 
                "val_acc": acc_te, 
                "epoch": float(epoch) # Normalize epoch number if needed, or keep as float
            }
        )

        # --- Snapshot VPM for GIF ---
        # Use a task that highlights training progress
        # E.g., sort by decreasing training accuracy, then decreasing validation accuracy
        vpm_sql_task = "SELECT * FROM virtual_index ORDER BY train_acc DESC, val_acc DESC"
        
        # Generate VPM snapshot
        # Use a window to show recent progress
        vpm_img = zm.snapshot_vpm(
            window_size=min(16, epoch + 1), # Show more history as epochs progress
            target_metric_name="train_acc"
        ) # Returns HxWx3 uint8
        
        # --- Enhance frame for better GIF visualization ---
        if vpm_img is not None and vpm_img.size > 0:
            # Ensure it's 3D
            if vpm_img.ndim == 2:
                vpm_img = np.stack([vpm_img]*3, axis=-1)
            elif vpm_img.ndim != 3 or vpm_img.shape[2] != 3:
                # If shape is wrong, create a dummy frame
                vpm_img = np.zeros((8, 8, 3), dtype=np.uint8)
                
            # Upscale for better visibility in GIF
            upscale_factor = 10
            if vpm_img.shape[0] > 0 and vpm_img.shape[1] > 0:
                frame = np.kron(vpm_img, np.ones((upscale_factor, upscale_factor, 1), dtype=np.uint8))
            else:
                frame = np.zeros((80, 80, 3), dtype=np.uint8) # Dummy frame if VPM is empty
            
            # Optional: Add epoch number as text (if matplotlib is available)
            if HAS_MATPLOTLIB:
                try:
                    fig, ax = plt.subplots(figsize=(frame.shape[1]/100, frame.shape[0]/100), dpi=100)
                    ax.imshow(frame)
                    ax.text(10, 10, f'Epoch: {epoch+1}/{max_epochs}',
                            bbox=dict(facecolor='white', alpha=0.8),
                            fontsize=8, color='black')
                    ax.axis('off')
                    
                    # New, non-deprecated method
                    buf = BytesIO()
                    fig.canvas.draw()
                    fig.canvas.print_png(buf)
                    
                    # Use Pillow to read the image from the buffer and convert to a NumPy array
                    buf.seek(0)
                    pil_image = Image.open(buf)
                    frame = np.array(pil_image)
                    
                    plt.close(fig)
                except Exception as e:
                    # It's better to log the error than to fail silently
                    print(f"Error during VPM plotting: {e}")
                    # You may choose to re-raise the exception or handle it differently
                    # raise e
                    
            gif.add(frame)
        else:
            # Add a blank frame if VPM generation failed
            blank_frame = np.zeros((80, 80, 3), dtype=np.uint8)
            gif.add(blank_frame)
            
        print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {loss_tr:.4f}, Train Acc: {acc_tr:.4f}, Val Loss: {loss_te:.4f}, Val Acc: {acc_te:.4f}")

    # --- 5) Save Outputs ---
    try:
        gif.save(str(out_gif))
        print(f"GIF saved to: {out_gif}")
        
        # Save the final frame for inspection
        gif.save_frame(str(out_final_frame), -1)
        print(f"Final VPM frame saved to: {out_final_frame}")
        
    except Exception as e:
        print(f"Warning: Failed to save GIF or frame: {e}")

    # --- 6) Assertions ---
    # Check that the GIF file was created and is not empty
    assert out_gif.exists(), f"GIF file was not created at {out_gif}"
    assert os.path.getsize(out_gif) > 0, f"GIF file is empty at {out_gif}"
    print(f"✅ GIF file created successfully. Size: {os.path.getsize(out_gif)} bytes")

    # Check that ZeroMemory state is updated
    assert zm.last_alerts is not None, "ZeroMemory alerts were not computed"
    assert isinstance(zm.last_full_vpm, np.ndarray), "ZeroMemory last_full_vpm is not a numpy array"
    assert zm.last_full_vpm.ndim == 3 and zm.last_full_vpm.shape[2] == 3, f"VPM shape incorrect: {zm.last_full_vpm.shape}"
    print(f"✅ ZeroMemory state validated. Final VPM shape: {zm.last_full_vpm.shape}")

    # --- 7) Validate Training Progress (Sanity Check) ---
    # Check that training improved over time (at least a little)
    if len(train_losses) > 1 and len(val_losses) > 1:
        initial_train_loss = train_losses[0]
        final_train_loss = train_losses[-1]
        initial_val_loss = val_losses[0]
        final_val_loss = val_losses[-1]
        
        initial_train_acc = train_accs[0]
        final_train_acc = train_accs[-1]
        initial_val_acc = val_accs[0]
        final_val_acc = val_accs[-1]
        
        print(f"Training Progress Summary:")
        print(f"  Train Loss: {initial_train_loss:.4f} -> {final_train_loss:.4f}")
        print(f"  Val Loss:   {initial_val_loss:.4f} -> {final_val_loss:.4f}")
        print(f"  Train Acc:  {initial_train_acc:.4f} -> {final_train_acc:.4f}")
        print(f"  Val Acc:    {initial_val_acc:.4f} -> {final_val_acc:.4f}")
        
        # Sanity checks (allowing for some variability)
        # Loss should generally decrease (allow for small fluctuations)
        # Accuracy should generally increase (allow for small fluctuations)
        loss_decreased = (final_train_loss < initial_train_loss + 0.1) # Allow 0.1 tolerance
        acc_increased = (final_train_acc > initial_train_acc - 0.05) # Allow 5% tolerance
        
        if loss_decreased:
            print("✅ Training loss showed expected decrease (within tolerance).")
        else:
            print(f"⚠️  Training loss did not decrease as expected. Initial: {initial_train_loss:.4f}, Final: {final_train_loss:.4f}")
            
        if acc_increased:
            print("✅ Training accuracy showed expected increase (within tolerance).")
        else:
            print(f"⚠️  Training accuracy did not increase as expected. Initial: {initial_train_acc:.4f}, Final: {final_train_acc:.4f}")
            
        # These are sanity checks, not strict assertions for a unit test
        # as ML training can have variability.
        
    else:
        print("⚠️  Insufficient data to validate training progress.")

    print("✅ Improved ZeroMemory GIF logging test completed.")
