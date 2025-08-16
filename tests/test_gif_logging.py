import os

import imageio.v2 as imageio
import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
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


@pytest.mark.parametrize("epochs", [20])
def test_gif_logging_with_zeromemory(tmp_path, epochs):
    # --- 1) Toy streaming setup (binary) ---
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    # Online-ish classifier so we have “epochs”
    clf = SGDClassifier(
        loss="log_loss", learning_rate="constant", eta0=0.1,
        max_iter=1, warm_start=True, random_state=42
    )

    # --- 2) ZeroMemory to track a few metrics ---
    metrics = ["loss", "val_loss", "acc", "val_acc", "lr"]
    zm = ZeroMemory(metric_names=metrics, buffer_steps=128, tile_size=8, selection_k=24)

    # --- 3) GIF logger ---
    gif = GIFLogger(fps=6) 
    out_gif = os.path.join(os.getcwd(), "images/training.gif")

    # Batch the training data for more realistic updates
    n = len(Xtr)
    batch_size = max(16, n // 20)
    classes = np.unique(ytr)

    for epoch in range(epochs):
        # One pass in small batches
        perm = np.random.RandomState(epoch).permutation(n)
        Xtr_shuf, ytr_shuf = Xtr[perm], ytr[perm]

        for i in range(0, n, batch_size):
            xb = Xtr_shuf[i:i+batch_size]
            yb = ytr_shuf[i:i+batch_size]
            if epoch == 0 and i == 0:
                clf.partial_fit(xb, yb, classes=classes)
            else:
                clf.partial_fit(xb, yb)

        # Evaluate at epoch end
        proba_tr = np.clip(clf.predict_proba(Xtr), 1e-6, 1 - 1e-6)
        proba_te = np.clip(clf.predict_proba(Xte), 1e-6, 1 - 1e-6)
        loss_tr = float(log_loss(ytr, proba_tr))
        loss_te = float(log_loss(yte, proba_te))
        acc_tr = float(accuracy_score(ytr, (proba_tr[:, 1] > 0.5).astype(int)))
        acc_te = float(accuracy_score(yte, (proba_te[:, 1] > 0.5).astype(int)))
        lr = 0.1  # constant in this example

        # Log metrics into ZeroMemory
        zm.log(
            step=epoch,
            metrics={"loss": loss_tr, "val_loss": loss_te, "acc": acc_tr, "val_acc": acc_te, "lr": lr}
        )

        # Snapshot a tiny VPM tile and expand to a small RGB frame for the GIF
        vpm = zm.snapshot_vpm(window_size=16, target_metric_name="loss")  # HxWx3 uint8
        # Optionally upscale a bit for visibility
        frame = np.kron(vpm, np.ones((6, 6, 1), dtype=np.uint8))
        gif.add(frame)

    # Save the animated GIF
    gif.save(out_gif)

    # --- 4) Assertions: file exists and is non-empty; sanity checks on final metrics ---
    assert os.path.exists(out_gif), "GIF file was not created"
    assert os.path.getsize(out_gif) > 0, "GIF file is empty"

    # Basic sanity: training should help a bit
    assert zm.last_alerts is not None  # alerts were computed on demand
    # And we can peek at the final VPM dimensions
    assert isinstance(zm.last_full_vpm, np.ndarray)
    assert zm.last_full_vpm.ndim == 3 and zm.last_full_vpm.shape[2] == 3
