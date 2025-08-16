import os

import imageio.v2 as imageio
import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from zeromodel.memory import ZeroMemory


class GIFLogger:
    def __init__(self, fps=6): self.frames, self.fps = [], fps
    def add(self, frame: np.ndarray):
        if frame.ndim == 2: frame = np.stack([frame]*3, axis=-1)
        self.frames.append(frame.astype(np.uint8))
    def save(self, path: str):
        assert self.frames, "No frames to save"
        imageio.mimsave(path, self.frames, duration=1.0/max(self.fps, 1), loop=0)

def upscale_nn(img_u8: np.ndarray, factor: int = 10) -> np.ndarray:
    """nearest-neighbor upscale for crisp pixels"""
    return np.kron(img_u8, np.ones((factor, factor, 1), dtype=np.uint8))

def contrast_stretch(img_u8: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """per-channel [p2,p98] stretch + gamma for punchier highlights."""
    out = img_u8.astype(np.float32) / 255.0
    for c in range(out.shape[2]):
        ch = out[..., c]
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 > p2:
            ch = (ch - p2) / max(1e-6, (p98 - p2))
            ch = np.clip(ch, 0, 1)
        # mild gamma < 1 brightens highlights (top-left pops)
        ch = np.power(ch, gamma)
        out[..., c] = ch
    return (np.clip(out, 0, 1) * 255.0).astype(np.uint8)

def draw_grid(img_u8: np.ndarray, step: int = 10, thickness: int = 1) -> np.ndarray:
    """light grid to emphasize the matrix structure after upscaling."""
    h, w, _ = img_u8.shape
    g = img_u8.copy()
    # soft gray lines
    line_val = 200
    for y in range(0, h, step):
        g[max(0, y-thickness):min(h, y+thickness), :, :] = line_val
    for x in range(0, w, step):
        g[:, max(0, x-thickness):min(w, x+thickness), :] = line_val
    return g

@pytest.mark.parametrize("epochs", [160])
def test_gif_training_epochs_obvious(tmp_path, epochs):
    # --- 1) Data ---
    X, y = make_moons(n_samples=2000, noise=0.25, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    # Random Fourier Features -> clear separability after training
    def rff(X, D=384, gamma=1.0, seed=0):
        rng = np.random.default_rng(seed)
        d = X.shape[1]
        W = rng.normal(0.0, np.sqrt(2*gamma), size=(d, D))
        b = rng.uniform(0, 2*np.pi, size=(D,))
        return np.sqrt(2.0/D) * np.cos(X @ W + b)

    Ztr = rff(Xtr, D=384, gamma=1.0, seed=0)
    Zte = rff(Xte, D=384, gamma=1.0, seed=1)

    n, d = Ztr.shape
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.02, size=(d,))
    b = 0.0

    def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
    def bce(y_true, y_prob, eps=1e-9):
        y_prob = np.clip(y_prob, eps, 1-eps)
        return -np.mean(y_true*np.log(y_prob) + (1-y_true)*np.log(1-y_prob))

    # --- 2) Make the VPM larger & more informative ---
    #   - bigger buffer -> wider image
    #   - larger tile_size/selection_k -> more rows/columns to visualize
    metrics = ["loss", "val_loss", "acc", "val_acc", "lr"]
    zm = ZeroMemory(metric_names=metrics, buffer_steps=1024, tile_size=24, selection_k=64)

    gif = GIFLogger(fps=6)
    out_gif = os.path.join(os.getcwd(), "images/training_epochs_obvious.gif")
    out_last_png = os.path.join(os.getcwd(), "images/training_epochs_obvious_last.png")

    # Learning-rate schedule to create visible phases (adds structure)
    lr0, lr1, lr2 = 0.3, 0.1, 0.03
    batch_size = 128
    idx_all = np.arange(n)

    # Snapshot every K epochs so the GIF steps are distinct
    SNAP_EVERY = 2
    WINDOW = 96  # larger virtual window -> larger tile
    UPSCALE = 12 # bigger pixels on disk

    for epoch in range(epochs):
        # three-phase LR schedule
        lr = lr0 if epoch < epochs*0.4 else (lr1 if epoch < epochs*0.75 else lr2)

        rng.shuffle(idx_all)
        Ztr_shuf, ytr_shuf = Ztr[idx_all], ytr[idx_all]
        for i in range(0, n, batch_size):
            zb = Ztr_shuf[i:i+batch_size]
            yb = ytr_shuf[i:i+batch_size]
            logits = zb @ W + b
            probs = sigmoid(logits)
            grad_logits = probs - yb
            gW = zb.T @ grad_logits / len(zb)
            gb = np.mean(grad_logits)
            W -= lr * gW
            b -= lr * gb

        tr_probs = sigmoid(Ztr @ W + b)
        te_probs = sigmoid(Zte @ W + b)
        tr_loss = float(bce(ytr, tr_probs))
        te_loss = float(bce(yte, te_probs))
        tr_acc  = float(np.mean((tr_probs > 0.5) == ytr))
        te_acc  = float(np.mean((te_probs > 0.5) == yte))

        zm.log(step=epoch, metrics={
            "loss": tr_loss, "val_loss": te_loss,
            "acc": tr_acc, "val_acc": te_acc, "lr": lr
        })

        # Only snapshot sometimes; make each frame large & high-contrast
        if epoch % SNAP_EVERY == 0 or epoch == (epochs - 1):
            vpm = zm.snapshot_vpm(window_size=WINDOW, target_metric_name="loss")  # HxWx3 uint8
            vpm = contrast_stretch(vpm, gamma=0.8)
            frame = upscale_nn(vpm, UPSCALE)
            frame = draw_grid(frame, step=16, thickness=1)
            gif.add(frame)

    # Final still PNG (last frame) for the blog
    last = gif.frames[-1]
    imageio.imwrite(out_last_png, last)
    gif.save(out_gif)

    assert os.path.exists(out_gif) and os.path.getsize(out_gif) > 0
    assert os.path.exists(out_last_png) and os.path.getsize(out_last_png) > 0
    assert zm.last_full_vpm is not None and zm.last_full_vpm.ndim == 3
