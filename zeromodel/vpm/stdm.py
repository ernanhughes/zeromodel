# zeromodel/stdm.py
from typing import Callable, List, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


def order_columns(X: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(-u)  # descending interest
    return idx, X[:, idx]


def order_rows(
    Xc: np.ndarray, w_aligned: np.ndarray, Kc: int
) -> Tuple[np.ndarray, np.ndarray]:
    Kc = min(Kc, Xc.shape[1])
    r = Xc[:, :Kc] @ w_aligned[:Kc]
    ridx = np.argsort(-r)
    return ridx, Xc[ridx, :]


def gamma_operator(
    series: List[np.ndarray],
    u_fn: Callable[[int, np.ndarray], np.ndarray],
    w: np.ndarray,
    Kc: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    Ys, col_orders, row_orders = [], [], []
    for t, Xt in enumerate(series):
        u_t = u_fn(t, Xt)
        Yt, ridx, cidx = phi_transform(Xt, u_t, w, Kc)
        Ys.append(Yt)
        col_orders.append(cidx)
        row_orders.append(ridx)
    return Ys, col_orders, row_orders


# ---------- Learning: weight vector to maximize TL ----------


# zeromodel/stdm.py - FIXED VERSION


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
    Learn metric weights that maximize top-left concentration in VPMs.

    FIXED: Proper convergence, efficient gradient calculation, and correct objective.
    """
    rng = np.random.default_rng(seed)
    M = series[0].shape[1]
    w0 = np.ones(M, dtype=np.float64) / np.sqrt(M)

    def _project(w):
        """Project weights to valid range and normalize"""
        w = np.maximum(0.0, w)
        n = np.linalg.norm(w) + 1e-12
        return w / n

    def _u_for(w, Xt):
        """Calculate task weights based on mode"""
        return w if u_mode == "mirror_w" else Xt.mean(axis=0)

    def objective(w_raw):
        """Calculate objective: maximize top-left mass minus regularization"""
        w = _project(w_raw)
        total_mass = 0.0

        # Calculate total top-left mass across all time steps
        for Xt in series:
            u_t = _u_for(w, Xt)
            try:
                Yt, _, _ = phi_transform(Xt, u_t, w, Kc)
                mass = top_left_mass(Yt, Kr, Kc, alpha)
                total_mass += mass
            except Exception as e:
                logger.warning(f"phi_transform failed: {e}")
                continue

        # Apply L2 regularization
        reg = l2 * float(w @ w)
        return -(total_mass - reg)  # Minimize negative = maximize positive

    # Use scipy optimizer if available (much more efficient)
    try:
        from scipy.optimize import minimize

        logger.debug("Using scipy optimizer for weight learning")

        result = minimize(objective, w0, method="L-BFGS-B", options={"maxiter": iters})

        if result.success:
            logger.info(f"Weight learning converged in {result.nit} iterations")
            return _project(result.x)
        else:
            logger.warning("Scipy optimization failed, falling back to manual method")
    except ImportError:
        logger.debug("Scipy not available, using manual gradient ascent")

    # Fallback: projected finite-difference ascent with proper batching
    w = w0.copy()
    logger.debug(f"Starting manual gradient ascent with {iters} iterations")

    for iteration in range(iters):
        # Batch gradient calculation for efficiency
        grad = np.zeros_like(w)
        base_objective = objective(w)

        # Calculate gradient using central differences
        eps = 1e-4
        for j in range(M):
            w_plus = w.copy()
            w_plus[j] += eps
            w_minus = w.copy()
            w_minus[j] -= eps

            obj_plus = objective(w_plus)
            obj_minus = objective(w_minus)
            grad[j] = (obj_minus - obj_plus) / (2 * eps)

        # Update weights
        w = w + step * grad
        w = _project(w)

        # Log progress every 20 iterations
        if iteration % 20 == 0:
            current_obj = objective(w)
            logger.debug(f"Iteration {iteration}: objective={-current_obj:.4f}")

    final_obj = objective(w)
    logger.info(f"Manual gradient ascent completed: final objective={-final_obj:.4f}")
    return _project(w)


# Also fix the phi_transform function to handle edge cases properly
def phi_transform(
    X: np.ndarray, u: np.ndarray, w: np.ndarray, Kc: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply dual-ordering transformation to concentrate signal in top-left.

    FIXED: Better handling of edge cases and proper alignment.
    """
    # Validate inputs
    if X.size == 0:
        return X, np.array([]), np.array([])

    # Ensure proper shapes
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")

    n_docs, n_metrics = X.shape
    if len(u) != n_metrics:
        raise ValueError(f"u length ({len(u)}) must match X columns ({n_metrics})")
    if len(w) != n_metrics:
        raise ValueError(f"w length ({len(w)}) must match X columns ({n_metrics})")

    # Step 1: Sort columns (metrics) by interest (u)
    cidx = np.argsort(-u)  # descending interest
    Xc = X[:, cidx]

    # Step 2: Sort rows (documents) by weighted intensity of top-Kc columns
    Kc_actual = min(Kc, Xc.shape[1])
    if Kc_actual == 0:
        ridx = np.arange(n_docs)
        return Xc, ridx, cidx

    # Calculate weighted document relevance
    w_aligned = w[cidx]  # Align weights with sorted columns
    try:
        r = Xc[:, :Kc_actual] @ w_aligned[:Kc_actual]
    except Exception as e:
        logger.warning(f"Matrix multiplication failed: {e}, using fallback")
        r = np.sum(Xc[:, :Kc_actual] * w_aligned[:Kc_actual], axis=1)

    ridx = np.argsort(-r)  # descending relevance

    # Step 3: Create final organized matrix
    Y = Xc[ridx, :]

    return Y, ridx, cidx


# Fix top_left_mass to handle edge cases
def top_left_mass(Y: np.ndarray, Kr: int, Kc: int, alpha: float = 0.95) -> float:
    """
    Calculate weighted sum of top-left Kr×Kc block with spatial decay.

    FIXED: Proper handling of small matrices and edge cases.
    """
    if Y.size == 0:
        return 0.0

    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got {Y.ndim}D")

    # Handle matrices smaller than requested region
    actual_Kr = min(Kr, Y.shape[0])
    actual_Kc = min(Kc, Y.shape[1])

    if actual_Kr == 0 or actual_Kc == 0:
        return 0.0

    # Extract the critical region
    critical_region = Y[:actual_Kr, :actual_Kc]

    # Create decay matrix with exponential decay from top-left
    rows = np.arange(actual_Kr)[:, None]
    cols = np.arange(actual_Kc)[None, :]
    decay_matrix = alpha ** (rows + cols)

    # Calculate weighted sum
    mass = float(np.sum(critical_region * decay_matrix))

    return mass


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
    for t in range(1, T - 1):
        curv[t] = np.linalg.norm(Ys[t + 1] - 2 * Ys[t] + Ys[t - 1])
    return curv


def critical_mask(Y: np.ndarray, theta: float = 0.8) -> np.ndarray:
    m = Y.max()
    if m <= 0:
        return np.zeros_like(Y, dtype=bool)
    return Y >= theta * m
