# zeromodel/stdm.py
from typing import Callable, List, Tuple
import numpy as np

# ---------- Core: ordering & scoring ----------

def top_left_mass(Y: np.ndarray, Kr: int, Kc: int, alpha: float = 0.95) -> float:
    """
    Weighted sum of the top-left Kr×Kc block with exponential decay from the corner.
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got {Y.ndim}D")

    Kr = max(0, min(Kr, Y.shape[0]))
    Kc = max(0, min(Kc, Y.shape[1]))
    if Kr == 0 or Kc == 0:
        return 0.0

    block = Y[:Kr, :Kc]
    weights = alpha ** (np.add.outer(np.arange(Kr), np.arange(Kc)))
    return float((block * weights).sum())


def order_columns(X: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort columns (metrics) by descending 'interest' vector u.
    Returns (column_indices, X_sorted_by_columns).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")
    if u.shape[0] != X.shape[1]:
        raise ValueError(f"u length ({u.shape[0]}) must match X columns ({X.shape[1]})")

    idx = np.argsort(-u)  # descending
    return idx, X[:, idx]


def order_rows(Xc: np.ndarray, w_aligned: np.ndarray, Kc: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort rows (documents) by weighted relevance using the first Kc columns of Xc
    with the correspondingly-aligned weights.
    """
    if Xc.ndim != 2:
        raise ValueError(f"Xc must be 2D, got {Xc.ndim}D")
    if w_aligned.shape[0] != Xc.shape[1]:
        raise ValueError(f"w_aligned length ({w_aligned.shape[0]}) must match Xc columns ({Xc.shape[1]})")

    Kc = max(0, min(Kc, Xc.shape[1]))
    if Kc == 0:
        ridx = np.arange(Xc.shape[0])
        return ridx, Xc

    r = Xc[:, :Kc] @ w_aligned[:Kc]
    ridx = np.argsort(-r)  # descending relevance
    return ridx, Xc[ridx, :]


def phi_transform(
    X: np.ndarray, u: np.ndarray, w: np.ndarray, Kc: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dual ordering:
      1) Columns by u (descending)
      2) Rows by relevance under aligned w on top-Kc columns
    Returns (Y, row_indices, col_indices).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")
    if u.shape[0] != X.shape[1]:
        raise ValueError(f"u length ({u.shape[0]}) must match X columns ({X.shape[1]})")
    if w.shape[0] != X.shape[1]:
        raise ValueError(f"w length ({w.shape[0]}) must match X columns ({X.shape[1]})")

    cidx, Xc = order_columns(X, u)
    w_aligned = w[cidx]
    ridx, Y = order_rows(Xc, w_aligned, Kc)
    return Y, ridx, cidx


def gamma_operator(
    series: List[np.ndarray],
    u_fn: Callable[[int, np.ndarray], np.ndarray],
    w: np.ndarray,
    Kc: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Apply phi_transform to each time slice in 'series' with time-varying u = u_fn(t, Xt),
    and fixed w and Kc.
    Returns lists of (Y_t, col_order_t, row_order_t).
    """
    Ys, col_orders, row_orders = [], [], []
    for t, Xt in enumerate(series):
        u_t = u_fn(t, Xt)
        Yt, ridx, cidx = phi_transform(Xt, u_t, w, Kc)
        Ys.append(Yt)
        col_orders.append(cidx)
        row_orders.append(ridx)
    return Ys, col_orders, row_orders


# ---------- Learning: weight vector to maximize TL ----------

def learn_w(
    series: List[np.ndarray],
    Kc: int,
    Kr: int,
    u_mode: str = "mirror_w",
    alpha: float = 0.97,
    l2: float = 2e-3,
    iters: int = 120,
    step: float = 8e-3,
    seed: int = 0,
) -> np.ndarray:
    """
    Learn a nonnegative, L2-normalized weight vector w (len = M) that
    maximizes the sum over time of top_left_mass(phi_transform(X_t; u_t, w, Kc), Kr, Kc, alpha)
    minus l2*||w||^2 regularization.

    Uses SciPy L-BFGS-B if available; otherwise falls back to projected
    finite-difference ascent.
    """
    rng = np.random.default_rng(seed)
    M = series[0].shape[1]
    w0 = np.ones(M, dtype=np.float64) / np.sqrt(M)

    def _project(w: np.ndarray) -> np.ndarray:
        w = np.maximum(0.0, w)
        n = np.linalg.norm(w) + 1e-12
        return w / n

    def _u_for(w: np.ndarray, Xt: np.ndarray) -> np.ndarray:
        if u_mode == "mirror_w":
            return w
        elif u_mode == "mean":
            return Xt.mean(axis=0)
        else:
            # default to mirror behavior for unknown modes
            return w

    def _objective(w_raw: np.ndarray) -> float:
        # Project to feasible set (nonnegative, unit norm)
        w = _project(w_raw)
        total = 0.0
        for Xt in series:
            u_t = _u_for(w, Xt)
            Yt, _, _ = phi_transform(Xt, u_t, w, Kc)
            total += top_left_mass(Yt, Kr, Kc, alpha)
        total -= l2 * float(w @ w)
        # We MINIMIZE in optimizers, so return negative of what we maximize
        return -total

    # ---- Try SciPy optimizer for speed/robustness ----
    try:
        from scipy.optimize import minimize

        # L-BFGS-B with nonnegativity bounds; we'll still project inside objective
        bounds = [(0.0, None)] * M

        res = minimize(
            _objective,
            w0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(iters), "ftol": 1e-10},
        )
        w_opt = _project(res.x if res.success else w0)
        return w_opt
    except Exception:
        # SciPy not present or failed → manual fallback
        pass

    # ---- Manual fallback: projected finite-difference ascent ----
    w = w0.copy()
    eps = 1e-4

    for _ in range(iters):
        # central-difference gradient over the whole series (batched objective)
        grad = np.zeros_like(w)
        # We compute a symmetric FD per dimension (still O(M) objectives).
        for j in range(M):
            w_plus = w.copy();  w_plus[j] += eps
            w_minus = w.copy(); w_minus[j] -= eps
            f_plus = _objective(w_plus)
            f_minus = _objective(w_minus)
            # d/dw_j of (-total) ≈ (f_plus - f_minus)/(2*eps)
            # ascent on total ⇒ descent on objective ⇒ w = w - step*grad_f
            grad[j] = (f_plus - f_minus) / (2.0 * eps)

        # Gradient step to DECREASE objective (i.e., INCREASE total)
        w = w - step * grad
        w = _project(w)

    return w


# ---------- Metric graph & canonical layout ----------

def metric_graph(col_orders: List[np.ndarray], tau: float = 8.0) -> np.ndarray:
    """
    Build an affinity between metrics from the per-time column orders.
    """
    M = col_orders[0].size
    T = len(col_orders)
    positions = [np.empty(M, dtype=int) for _ in range(T)]
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
    """
    Spectral layout of metrics: Fiedler vector of Laplacian(W).
    """
    d = W.sum(axis=1)
    L = np.diag(d) - W
    vals, vecs = np.linalg.eigh(L)
    f = vecs[:, 1] if vecs.shape[1] > 1 else vecs[:, 0]
    return np.argsort(f)


# ---------- Diagnostics: curvature & critical region ----------

def curvature_over_time(Ys: List[np.ndarray]) -> np.ndarray:
    """
    Second finite-difference Frobenius norm across time steps (length T).
    """
    T = len(Ys)
    curv = np.zeros(T, dtype=np.float64)
    if T < 3:
        return curv
    for t in range(1, T - 1):
        curv[t] = np.linalg.norm(Ys[t + 1] - 2.0 * Ys[t] + Ys[t - 1])
    return curv


def critical_mask(Y: np.ndarray, theta: float = 0.8) -> np.ndarray:
    """
    Threshold a matrix at a fraction of its max; returns boolean mask.
    """
    if Y.size == 0:
        return np.zeros_like(Y, dtype=bool)
    m = Y.max()
    if m <= 0:
        return np.zeros_like(Y, dtype=bool)
    return Y >= (theta * m)


# ---------- Optional: normalized TL metric (not used by existing tests) ----------

def top_left_mass_norm(Y: np.ndarray, Kr: int, Kc: int, alpha: float = 0.95) -> float:
    """
    Column-normalized variant of TL mass to reduce energy bias toward wide/high-energy columns.
    Keep signature separate to avoid breaking callers that expect raw TL mass.
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got {Y.ndim}D")

    Kr = max(0, min(Kr, Y.shape[0]))
    Kc = max(0, min(Kc, Y.shape[1]))
    if Kr == 0 or Kc == 0:
        return 0.0

    col_norm = np.maximum(np.linalg.norm(Y, ord=1, axis=0), 1e-12)
    Yn = Y / col_norm
    block = Yn[:Kr, :Kc]
    weights = alpha ** (np.add.outer(np.arange(Kr), np.arange(Kc)))
    return float((block * weights).sum())
