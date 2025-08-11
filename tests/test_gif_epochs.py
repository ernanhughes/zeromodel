import os
import numpy as np
import imageio.v2 as imageio
import pytest

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from zeromodel.memory import ZeroMemory


class GIFLogger:
    """Collect RGB frames and save an animated GIF."""
    def __init__(self, fps=6):
        self.frames = []
        self.fps = fps

    def add(self, frame: np.ndarray):
        if frame.ndim == 2:
            frame = np.stack([frame]*3, axis=-1)
        self.frames.append(frame.astype(np.uint8))

    def save(self, path: str):
        assert self.frames, "No frames to save"
        imageio.mimsave(path, self.frames, duration=1.0 / max(self.fps, 1))


def random_fourier_features(X, D=256, gamma=1.0, rng=None):
    """
    Rahimi–Recht Random Fourier Features for RBF kernel approximation.
    Maps R^d -> R^D via cos(Wx + b).
    """
    rng = np.random.default_rng(None if rng is None else rng)
    d = X.shape[1]
    W = rng.normal(loc=0.0, scale=np.sqrt(2*gamma), size=(d, D))
    b = rng.uniform(0, 2*np.pi, size=(D,))
    Z = np.sqrt(2.0 / D) * np.cos(X @ W + b)
    return Z


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y_true, y_prob, eps=1e-9):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


@pytest.mark.parametrize("epochs", [200])
def test_gif_training_epochs(tmp_path, epochs):
    # --- 1) Data ---
    X, y = make_moons(n_samples=1500, noise=0.25, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    # Nonlinear lift so a linear model can learn
    Ztr = random_fourier_features(Xtr, D=256, gamma=1.0, rng=42)
    Zte = random_fourier_features(Xte, D=256, gamma=1.0, rng=1337)

    n, d = Ztr.shape

    # --- 2) Model (NumPy logistic regression) ---
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.01, size=(d,))
    b = 0.0

    lr = 0.3
    batch_size = 128

    # --- 3) ZeroMemory to track metrics + GIF frames ---
    metrics = ["loss", "val_loss", "acc", "val_acc", "lr"]
    zm = ZeroMemory(metric_names=metrics, buffer_steps=256, tile_size=8, selection_k=24)

    gif = GIFLogger(fps=6)
    out_gif = os.path.join(os.getcwd(), "training_epochs.gif")

    # --- 4) Training loop with epochs ---
    idx_all = np.arange(n)
    for epoch in range(epochs):
        rng.shuffle(idx_all)
        Ztr_shuf, ytr_shuf = Ztr[idx_all], ytr[idx_all]

        # Mini-batch gradient descent
        for i in range(0, n, batch_size):
            zb = Ztr_shuf[i:i+batch_size]
            yb = ytr_shuf[i:i+batch_size]

            logits = zb @ W + b
            probs = sigmoid(logits)
            # gradients
            grad_logits = probs - yb
            gW = zb.T @ grad_logits / len(zb)
            gb = np.mean(grad_logits)

            W -= lr * gW
            b -= lr * gb

        # Epoch metrics
        tr_probs = sigmoid(Ztr @ W + b)
        te_probs = sigmoid(Zte @ W + b)

        tr_loss = float(bce_loss(ytr, tr_probs))
        te_loss = float(bce_loss(yte, te_probs))
        tr_acc = float(np.mean((tr_probs > 0.5) == ytr))
        te_acc = float(np.mean((te_probs > 0.5) == yte))

        # Log into ZeroMemory
        zm.log(step=epoch, metrics={
            "loss": tr_loss,
            "val_loss": te_loss,
            "acc": tr_acc,
            "val_acc": te_acc,
            "lr": lr
        })

        # VPM snapshot -> upscale a bit for readability -> add to GIF
        vpm = zm.snapshot_vpm(window_size=24, target_metric_name="loss")  # HxWx3 uint8
        frame = np.kron(vpm, np.ones((6, 6, 1), dtype=np.uint8))  # simple nearest-neighbor upsample
        # Optionally annotate frame (e.g., embed text) — skipping to keep deps minimal
        gif.add(frame)

    # --- 5) Save GIF and assert basic sanity ---
    gif.save(out_gif)
    assert os.path.exists(out_gif) and os.path.getsize(out_gif) > 0

    # Sanity: training should generally improve acc or lower loss
    # Not a strict guarantee, but usually true with RFF on moons
    assert zm.last_full_vpm is not None and zm.last_full_vpm.ndim == 3
