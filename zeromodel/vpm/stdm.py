"""
Spatio-Temporal Dual Modeling (STDM) Module

Provides core functionality for:
- Dual ordering of temporal data matrices
- Learning optimal weight vectors
- Computing topological metrics
- Analyzing temporal curvature

Key Concepts:
- phi_transform: Dual ordering (columns by interest vector u, rows by relevance under w)
- gamma_operator: Applies phi_transform across time series
- learn_w: Optimizes weight vector w to maximize top-left mass
- top_left_mass: Measures concentration of significant values in top-left corner
- curvature_over_time: Quantifies rate of change in ordered matrices
"""

import logging
from typing import Callable, List, Tuple

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# ---------- Core: ordering & scoring ----------

def top_left_mass(Y: np.ndarray, Kr: int, Kc: int, alpha: float = 0.95) -> float:
    """
    Calculate weighted sum of top-left Kr×Kc block with exponential decay.
    
    Measures concentration of significant values in the top-left corner of the
    reordered matrix, with weights decaying exponentially from the (0,0) position.
    
    Args:
        Y: 2D matrix (rows x columns)
        Kr: Number of top rows to consider
        Kc: Number of left columns to consider
        alpha: Decay factor (0.9-0.99 typical)
        
    Returns:
        Weighted sum of top-left block
    """
    if Y.ndim != 2:
        logger.error(f"top_left_mass: Y must be 2D, got {Y.ndim}D")
        raise ValueError(f"Y must be 2D, got {Y.ndim}D")

    Kr = max(0, min(Kr, Y.shape[0]))
    Kc = max(0, min(Kc, Y.shape[1]))
    if Kr == 0 or Kc == 0:
        logger.warning("top_left_mass: Kr or Kc is zero, returning 0")
        return 0.0

    block = Y[:Kr, :Kc]
    # Create exponential decay weights matrix
    row_decay = alpha ** np.arange(Kr)
    col_decay = alpha ** np.arange(Kc)
    weights = np.outer(row_decay, col_decay)
    
    result = float((block * weights).sum())
    logger.debug(f"top_left_mass: Kr={Kr}, Kc={Kc}, alpha={alpha:.3f}, "
                 f"block_min={block.min():.3f}, block_max={block.max():.3f}, "
                 f"mass={result:.4f}")
    return result


def order_columns(X: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort matrix columns by descending interest vector u.
    
    Args:
        X: Input matrix (N x M)
        u: Interest vector (M,)
        
    Returns:
        Tuple: (column_indices, X_sorted)
    """
    if X.ndim != 2:
        logger.error(f"order_columns: X must be 2D, got {X.ndim}D")
        raise ValueError(f"X must be 2D, got {X.ndim}D")
    if u.shape[0] != X.shape[1]:
        logger.error(f"order_columns: u length {u.shape[0]} != X columns {X.shape[1]}")
        raise ValueError(f"u length ({u.shape[0]}) must match X columns ({X.shape[1]})")

    idx = np.argsort(-u)  # descending order
    X_sorted = X[:, idx]
    logger.debug(f"order_columns: sorted {X.shape[1]} columns, "
                 f"u_min={u.min():.3f}, u_max={u.max():.3f}")
    return idx, X_sorted


def order_rows(Xc: np.ndarray, w_aligned: np.ndarray, Kc: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort rows by relevance using aligned weights on first Kc columns.
    
    Args:
        Xc: Column-ordered matrix (N x M)
        w_aligned: Weight vector aligned with Xc columns (M,)
        Kc: Number of columns to consider for ranking
        
    Returns:
        Tuple: (row_indices, X_sorted)
    """
    if Xc.ndim != 2:
        logger.error(f"order_rows: Xc must be 2D, got {Xc.ndim}D")
        raise ValueError(f"Xc must be 2D, got {Xc.ndim}D")
    if w_aligned.shape[0] != Xc.shape[1]:
        logger.error(f"order_rows: w_aligned length {w_aligned.shape[0]} != Xc columns {Xc.shape[1]}")
        raise ValueError(f"w_aligned length ({w_aligned.shape[0]}) must match Xc columns ({Xc.shape[1]})")

    Kc = max(0, min(Kc, Xc.shape[1]))
    if Kc == 0:
        logger.warning("order_rows: Kc=0, returning original order")
        ridx = np.arange(Xc.shape[0])
        return ridx, Xc

    # Compute relevance scores using only first Kc columns
    r = Xc[:, :Kc] @ w_aligned[:Kc]
    ridx = np.argsort(-r)  # descending relevance
    X_sorted = Xc[ridx, :]
    
    logger.debug(f"order_rows: sorted {Xc.shape[0]} rows using Kc={Kc}, "
                 f"r_min={r.min():.3f}, r_max={r.max():.3f}")
    return ridx, X_sorted


def phi_transform(
    X: np.ndarray, u: np.ndarray, w: np.ndarray, Kc: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply dual ordering to matrix:
      1. Sort columns by u (descending)
      2. Sort rows by relevance under w on top-Kc columns
      
    Args:
        X: Input matrix (N x M)
        u: Column interest vector (M,)
        w: Row weight vector (M,)
        Kc: Number of columns for row ordering
        
    Returns:
        Tuple: (Y, row_indices, col_indices)
    """
    # logger.debug(f"phi_transform: input shape={X.shape}, Kc={Kc}")
    
    # Validate inputs
    if X.ndim != 2:
        logger.error(f"phi_transform: X must be 2D, got {X.ndim}D")
        raise ValueError(f"X must be 2D, got {X.ndim}D")
    if u.shape[0] != X.shape[1]:
        logger.error(f"phi_transform: u length {u.shape[0]} != X columns {X.shape[1]}")
        raise ValueError(f"u length ({u.shape[0]}) must match X columns ({X.shape[1]})")
    if w.shape[0] != X.shape[1]:
        logger.error(f"phi_transform: w length {w.shape[0]} != X columns {X.shape[1]}")
        raise ValueError(f"w length ({w.shape[0]}) must match X columns ({X.shape[1]})")

    # Column ordering
    cidx, Xc = order_columns(X, u)
    w_aligned = w[cidx]
    
    # Row ordering
    ridx, Y = order_rows(Xc, w_aligned, Kc)
    
    # logger.debug(f"phi_transform: completed, output shape={Y.shape}")
    return Y, ridx, cidx


def gamma_operator(
    series: List[np.ndarray],
    u_fn: Callable[[int, np.ndarray], np.ndarray],
    w: np.ndarray,
    Kc: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Apply phi_transform to each time slice in a series.
    
    Args:
        series: List of matrices across time
        u_fn: Function returning u vector given (time, matrix)
        w: Fixed weight vector for row ordering
        Kc: Number of columns for row ordering
        
    Returns:
        Tuple: (Y_list, col_orders, row_orders)
    """
    logger.info(f"gamma_operator: processing {len(series)} time steps, Kc={Kc}")
    
    if not series:
        logger.warning("gamma_operator: empty series input")
        return [], [], []
    
    Ys, col_orders, row_orders = [], [], []
    for t, Xt in enumerate(series):
        logger.debug(f"Processing time step {t}/{len(series)}")
        
        # Get time-varying u vector
        u_t = u_fn(t, Xt)
        logger.debug(f"u_t: min={u_t.min():.3f}, max={u_t.max():.3f}")
        
        # Apply dual ordering
        Yt, ridx, cidx = phi_transform(Xt, u_t, w, Kc)
        
        Ys.append(Yt)
        col_orders.append(cidx)
        row_orders.append(ridx)
    
    logger.info("gamma_operator completed successfully")
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
    Learn nonnegative, L2-normalized weight vector w that maximizes
    the sum over time of top_left_mass(phi_transform(X_t; u_t, w, Kc), Kr, Kc, alpha)
    
    Uses SciPy L-BFGS-B if available; otherwise falls back to finite-difference ascent.
    
    Args:
        series: List of matrices across time
        Kc: Number of columns for row ordering
        Kr: Number of rows for TL mass
        u_mode: 'mirror_w' (u_t = w) or 'mean' (u_t = column mean)
        alpha: TL mass decay factor
        l2: L2 regularization strength
        iters: Optimization iterations
        step: Learning rate for manual optimization
        seed: Random seed
        
    Returns:
        Optimized weight vector w
    """
    logger.info(f"learn_w: starting optimization for {len(series)} time steps, "
                f"Kc={Kc}, Kr={Kr}, u_mode={u_mode}, alpha={alpha:.3f}, "
                f"l2={l2:.1e}, iters={iters}, step={step:.1e}")
    
    rng = np.random.default_rng(seed)
    M = series[0].shape[1]
    w0 = np.ones(M, dtype=np.float64) / np.sqrt(M)
    logger.debug(f"Initial w0: min={w0.min():.3f}, max={w0.max():.3f}, "
                 f"norm={np.linalg.norm(w0):.3f}")

    def _project(w: np.ndarray) -> np.ndarray:
        """Project to nonnegative, L2-normalized weights"""
        w = np.maximum(0.0, w)
        n = np.linalg.norm(w) + 1e-12
        return w / n

    def _u_for(w: np.ndarray, Xt: np.ndarray) -> np.ndarray:
        """Determine u vector based on mode"""
        if u_mode == "mirror_w":
            return w
        elif u_mode == "mean":
            return Xt.mean(axis=0)
        else:
            logger.warning(f"Unknown u_mode '{u_mode}', defaulting to mirror_w")
            return w

    def _objective(w_raw: np.ndarray) -> float:
        """Objective function to minimize (negative TL mass + regularization)"""
        w = _project(w_raw)
        total = 0.0
        for i, Xt in enumerate(series):
            u_t = _u_for(w, Xt)
            Yt, _, _ = phi_transform(Xt, u_t, w, Kc)
            tl = top_left_mass(Yt, Kr, Kc, alpha)
            total += tl
            logger.debug(f"Time {i}: TL mass = {tl:.4f}")
        
        reg = l2 * float(w @ w)
        logger.debug(f"Objective: total={total:.4f}, reg={reg:.6f}, net={total - reg:.4f}")
        return -(total - reg)  # Minimize negative (mass - reg)

    # ---- Try SciPy optimizer ----
    try:
        from scipy.optimize import minimize
        logger.info("Using SciPy L-BFGS-B optimizer")

        bounds = [(0.0, None)] * M  # Nonnegativity constraints

        res = minimize(
            _objective,
            w0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(iters), "ftol": 1e-10},
        )
        
        if res.success:
            logger.info(f"Optimization successful: {res.message}")
            logger.debug(f"Final objective: {res.fun:.4f}")
        else:
            logger.warning(f"Optimization failed: {res.message}")
        
        w_opt = _project(res.x)
        logger.info(f"Optimized w: min={w_opt.min():.3f}, max={w_opt.max():.3f}, "
                    f"norm={np.linalg.norm(w_opt):.3f}")
        return w_opt
    except ImportError:
        logger.warning("SciPy not available, falling back to manual optimization")
    except Exception as e:
        logger.error(f"SciPy optimization failed: {str(e)}")

    # ---- Manual fallback: finite-difference ascent ----
    logger.info("Using manual finite-difference optimization")
    w = w0.copy()
    eps = 1e-4

    for iter in range(iters):
        grad = np.zeros_like(w)
        
        # Compute finite-difference gradient
        for j in range(M):
            w_plus = w.copy(); w_plus[j] += eps
            w_minus = w.copy(); w_minus[j] -= eps
            f_plus = _objective(w_plus)
            f_minus = _objective(w_minus)
            grad[j] = (f_plus - f_minus) / (2.0 * eps)
        
        # Gradient descent step
        w = w - step * grad
        w = _project(w)
        
        if iter % 10 == 0:
            obj_val = _objective(w)
            logger.debug(f"Iter {iter:03d}: objective={obj_val:.4f}, "
                         f"grad_norm={np.linalg.norm(grad):.4f}")

    logger.info(f"Manual optimization completed after {iters} iterations")
    return w


# ---------- Metric graph & canonical layout ----------

def metric_graph(col_orders: List[np.ndarray], tau: float = 8.0) -> np.ndarray:
    """
    Build metric affinity graph from column orders.
    
    Args:
        col_orders: List of column orderings per time step
        tau: Decay factor for position differences
        
    Returns:
        Affinity matrix W (M x M)
    """
    if not col_orders:
        logger.error("metric_graph: empty col_orders")
        raise ValueError("col_orders cannot be empty")
    
    M = col_orders[0].size
    T = len(col_orders)
    logger.info(f"Building metric graph: M={M} metrics, T={T} time steps")
    
    # Precompute positions for each metric at each time
    positions = [np.empty(M, dtype=int) for _ in range(T)]
    for t, cidx in enumerate(col_orders):
        positions[t][cidx] = np.arange(M)
    
    # Build affinity matrix
    W = np.zeros((M, M), dtype=np.float64)
    for m in range(M):
        for n in range(M):
            s = 0.0
            for t in range(T):
                pm, pn = positions[t][m], positions[t][n]
                # Double exponential decay: position difference and min position
                decay = np.exp(-abs(pm - pn)/tau * np.exp(-min(pm, pn)/tau))
                s += decay
            W[m, n] = s / T
    
    logger.debug(f"Metric graph: min_affinity={W.min():.3f}, max_affinity={W.max():.3f}")
    return W


def canonical_layout(W: np.ndarray) -> np.ndarray:
    """
    Compute spectral layout of metrics using Fiedler vector.
    
    Args:
        W: Affinity matrix (M x M)
        
    Returns:
        Sorted metric indices
    """
    if W.size == 0:
        logger.error("canonical_layout: empty affinity matrix")
        return np.array([], dtype=int)
    
    d = W.sum(axis=1)
    L = np.diag(d) - W  # Unnormalized Laplacian
    
    try:
        vals, vecs = np.linalg.eigh(L)
        # Sort eigenvalues and get second smallest (Fiedler vector)
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]] if vecs.shape[1] > 1 else vecs[:, 0]
        sorted_idx = np.argsort(fiedler)
        logger.debug("Computed canonical layout via eigendecomposition")
        return sorted_idx
    except np.linalg.LinAlgError as e:
        logger.error(f"Eigendecomposition failed: {str(e)}")
        # Fallback to arbitrary order
        return np.arange(W.shape[0])


# ---------- Diagnostics: curvature & critical region ----------

def curvature_over_time(Ys: List[np.ndarray]) -> np.ndarray:
    """
    Compute second finite-difference (acceleration) across time steps.
    
    Args:
        Ys: List of matrices ordered by gamma_operator
        
    Returns:
        Curvature vector (length T) with values at interior points
    """
    T = len(Ys)
    curv = np.zeros(T, dtype=np.float64)
    
    if T < 3:
        logger.warning(f"curvature_over_time: Only {T} time steps, "
                       "needs at least 3 for curvature")
        return curv
    
    logger.info(f"Computing curvature over {T} time steps")
    
    for t in range(1, T - 1):
        # Second finite difference: Y''(t) ≈ Y[t+1] - 2Y[t] + Y[t-1]
        diff = Ys[t + 1] - 2.0 * Ys[t] + Ys[t - 1]
        curv[t] = np.linalg.norm(diff, 'fro')  # Frobenius norm
        logger.debug(f"Time {t}: curvature = {curv[t]:.4f}")
    
    logger.debug(f"Curvature range: min={curv.min():.4f}, max={curv.max():.4f}")
    return curv


def critical_mask(Y: np.ndarray, theta: float = 0.8) -> np.ndarray:
    """
    Threshold matrix to identify critical regions.
    
    Args:
        Y: Input matrix
        theta: Fraction of max value for threshold
        
    Returns:
        Boolean mask where values >= theta * max
    """
    if Y.size == 0:
        logger.warning("critical_mask: empty matrix input")
        return np.zeros_like(Y, dtype=bool)
    
    ymax = Y.max()
    if ymax <= 0:
        logger.warning("critical_mask: max <= 0, returning all False")
        return np.zeros_like(Y, dtype=bool)
    
    threshold = theta * ymax
    mask = Y >= threshold
    logger.debug(f"critical_mask: threshold={threshold:.3f}, "
                 f"critical_ratio={mask.mean():.3f}")
    return mask


# ---------- Helper Test Functions ----------

def test_stdm_basic_operations():
    """Test core STDM operations with small synthetic data"""
    logger.info("===== Running basic STDM operations test =====")
    
    # Create test data
    X = np.array([
        [1.0, 0.5, 0.3],
        [0.2, 0.8, 0.1],
        [0.9, 0.4, 0.6]
    ])
    u = np.array([0.8, 0.5, 0.3])
    w = np.array([0.7, 0.2, 0.1])
    
    # Test order_columns
    col_idx, Xc = order_columns(X, u)
    logger.debug(f"Column order: {col_idx}")
    logger.debug(f"Column-ordered matrix:\n{Xc}")
    
    # Test order_rows
    row_idx, Xr = order_rows(Xc, w, Kc=2)
    logger.debug(f"Row order: {row_idx}")
    logger.debug(f"Dual-ordered matrix:\n{Xr}")
    
    # Test phi_transform
    Y, ridx, cidx = phi_transform(X, u, w, Kc=2)
    logger.debug(f"phi_transform result:\n{Y}")
    
    # Test top_left_mass
    tl = top_left_mass(Y, Kr=2, Kc=2, alpha=0.9)
    logger.debug(f"Top-left mass: {tl:.4f}")
    
    logger.info("Basic operations test completed")

def test_learn_w_small():
    """Test weight learning with small synthetic series"""
    logger.info("===== Running learn_w test =====")
    
    # Create small time series
    series = [
        np.array([[1.0, 0.5], [0.8, 0.9]]),
        np.array([[0.9, 0.6], [0.7, 0.8]]),
        np.array([[0.8, 0.7], [0.6, 0.7]])
    ]
    
    # Learn weights
    w_opt = learn_w(
        series,
        Kc=1,
        Kr=2,
        u_mode="mirror_w",
        alpha=0.95,
        l2=0.01,
        iters=50,
        step=0.1,
        seed=42
    )
    
    logger.info(f"Learned weights: {w_opt}")
    logger.info("learn_w test completed")

def test_curvature_calculation():
    """Test curvature calculation"""
    logger.info("===== Running curvature test =====")
    
    # Create small time series of matrices
    Ys = [
        np.array([[1.0, 0.5], [0.8, 0.9]]),
        np.array([[0.9, 0.6], [0.7, 0.8]]),
        np.array([[0.8, 0.7], [0.6, 0.7]])
    ]
    
    curv = curvature_over_time(Ys)
    logger.info(f"Curvature values: {curv}")
    
    logger.info("Curvature test completed")

def test_metric_graph():
    """Test metric graph construction"""
    logger.info("===== Running metric graph test =====")
    
    # Create column orders
    col_orders = [
        np.array([2, 0, 1]),  # Time 0
        np.array([1, 2, 0]),  # Time 1
        np.array([0, 1, 2])   # Time 2
    ]
    
    W = metric_graph(col_orders, tau=5.0)
    logger.debug(f"Affinity matrix:\n{W}")
    
    layout = canonical_layout(W)
    logger.info(f"Canonical layout: {layout}")
    
    logger.info("Metric graph test completed")

if __name__ == "__main__":
    # Set up logging when run directly
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("stdm_debug.log")
        ]
    )
    
    # Run test suite
    test_stdm_basic_operations()
    test_learn_w_small()
    test_curvature_calculation()
    test_metric_graph()
    
    logger.info("All STDM tests completed")