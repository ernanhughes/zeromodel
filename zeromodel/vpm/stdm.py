# zeromodel/stdm.py
from typing import Callable, List, Optional, Tuple

import numpy as np

# ---------- Core: ordering & scoring ----------

def top_left_mass(Y: np.ndarray, Kr: int, Kc: int, alpha: float = 0.95) -> float:
    Kr = min(Kr, Y.shape[0]); Kc = min(Kc, Y.shape[1])
    weights = alpha ** (np.add.outer(np.arange(Kr), np.arange(Kc)))
    return float((Y[:Kr, :Kc] * weights).sum())

def order_columns(X: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(-u)  # descending interest
    return idx, X[:, idx]

def order_rows(Xc: np.ndarray, w_aligned: np.ndarray, Kc: int) -> Tuple[np.ndarray, np.ndarray]:
    Kc = min(Kc, Xc.shape[1])
    r = Xc[:, :Kc] @ w_aligned[:Kc]
    ridx = np.argsort(-r)
    return ridx, Xc[ridx, :]

def phi_transform(X: np.ndarray, u: np.ndarray, w: np.ndarray, Kc: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cidx, Xc = order_columns(X, u)
    ridx, Y   = order_rows(Xc, w[cidx], Kc)  # align w with chosen columns
    return Y, ridx, cidx

def gamma_operator(series: List[np.ndarray], u_fn: Callable[[int, np.ndarray], np.ndarray],
                   w: np.ndarray, Kc: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    Ys, col_orders, row_orders = [], [], []
    for t, Xt in enumerate(series):
        u_t = u_fn(t, Xt)
        Yt, ridx, cidx = phi_transform(Xt, u_t, w, Kc)
        Ys.append(Yt); col_orders.append(cidx); row_orders.append(ridx)
    return Ys, col_orders, row_orders

# ---------- Learning: weight vector to maximize TL ----------

def learn_w(series: List[np.ndarray], Kc: int, Kr: int,
            u_mode: str = "mirror_w", alpha: float = 0.97,
            l2: float = 2e-3, iters: int = 120, step: float = 8e-3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = series[0].shape[1]
    w0 = np.ones(M, dtype=np.float64) / np.sqrt(M)

    # Prefer SciPy optimizer if available; fallback to projected ascent
    try:
        from scipy.optimize import minimize
        SCIPY = True
    except Exception:
        SCIPY = False

    def _project(w):
        w = np.maximum(0.0, w)
        n = np.linalg.norm(w) + 1e-12
        return w / n

    def _u_for(w, Xt):
        return w if u_mode == "mirror_w" else Xt.mean(axis=0)

    if SCIPY:
        def objective(w_raw):
            w = _project(w_raw)
            val = 0.0
            for Xt in series:
                u_t = _u_for(w, Xt)
                Yt, _, _ = phi_transform(Xt, u_t, w, Kc)
                val += top_left_mass(Yt, Kr, Kc, alpha)
            val -= l2 * float(w @ w)
            return -val
        bounds = [(0.0, None)] * M
        res = minimize(objective, w0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
        return _project(res.x)

    # ---- Fallback: projected finite-difference ascent ----
    w = w0.copy()
    for _ in range(iters):
        grad = np.zeros_like(w)
        for Xt in series:
            u_t = _u_for(w, Xt)
            Yt, _, cidx = phi_transform(Xt, u_t, w, Kc)
            base = top_left_mass(Yt, Kr, Kc, alpha)
            eps = 1e-3
            for j in range(M):
                w_try = w.copy()
                w_try[j] += eps
                Yp, _, _ = phi_transform(Xt, u_t, w_try, Kc)
                grad[j] += (top_left_mass(Yp, Kr, Kc, alpha) - base) / eps
        grad -= 2 * l2 * w
        w = np.maximum(0.0, w + step * grad)
        w /= (np.linalg.norm(w) + 1e-12)
    return w
# ---------- Metric graph & canonical layout ----------

def metric_graph(col_orders: List[np.ndarray], tau: float = 8.0) -> np.ndarray:
    M = col_orders[0].size
    T = len(col_orders)
    positions = [np.empty(M, int) for _ in range(T)]
    for t, cidx in enumerate(col_orders):
        positions[t][cidx] = np.arange(M)

    W = np.zeros((M, M), dtype=np.float64)
    for m in range(M):
        for n in range(M):
            s = 0.0
            for t in range(T):
                pm, pn = positions[t][m], positions[t][n]
                s += np.exp(-abs(pm - pn) / tau) * np.exp(-(min(pm, pn)) / tau)
            W[m, n] = s / T
    return W

def canonical_layout(W: np.ndarray) -> np.ndarray:
    d = W.sum(axis=1)
    L = np.diag(d) - W
    vals, vecs = np.linalg.eigh(L)
    f = vecs[:, 1] if vecs.shape[1] > 1 else vecs[:, 0]
    return np.argsort(f)

# ---------- Diagnostics: curvature & critical region ----------

def curvature_over_time(Ys: List[np.ndarray]) -> np.ndarray:
    """‖second finite difference‖_F per time; length T."""
    T = len(Ys)
    curv = np.zeros(T, dtype=np.float64)
    if T < 3: 
        return curv
    for t in range(1, T-1):
        curv[t] = np.linalg.norm(Ys[t+1] - 2*Ys[t] + Ys[t-1])
    return curv

def critical_mask(Y: np.ndarray, theta: float = 0.8) -> np.ndarray:
    m = Y.max()
    if m <= 0: 
        return np.zeros_like(Y, dtype=bool)
    return (Y >= theta * m)
