from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

from ..config import get_config


class SpatialOptimizer:
    """
    Optimizes metric weights to concentrate decision-relevant information in the top-left
    region of the Visual Policy Map (VPM), enabling more reliable edge decisions.
    
    This class implements ZeroModel's Spatial Calculus, turning the ordering operation
    into a learnable transform that maximizes decision accuracy.
    """
    
    def __init__(self, 
                 Kc: int = None,
                 Kr: int = None,
                 alpha: float = 0.95,
                 l2: float = 1e-3,
                 u_mode: str = "mirror_w"):
        """
        Initialize the spatial optimizer.
        
        Args:
            Kc: Number of top metric columns to consider for row ordering
            Kr: Number of top source rows to consider for top-left mass
            alpha: Decay factor for top-left mass weighting (higher = more focus on top-left)
            l2: L2 regularization strength
            u_mode: How to compute column interest scores:
                   'mirror_w' - Use the same weights as row intensity (default)
                   'col_mean' - Use column means from the data
        """
        # Get defaults from config if not provided (respect explicit zeros/validation below)
        self.Kc = Kc if Kc is not None else get_config("spatial_calculus", "Kc", 16)
        self.Kr = Kr if Kr is not None else get_config("spatial_calculus", "Kr", 32)
        self.alpha = alpha
        self.l2 = l2
        self.u_mode = u_mode

        # Parameter validation (required by tests)
        if self.Kc is None or self.Kc <= 0:
            raise ValueError("Kc must be positive")
        if self.Kr is None or self.Kr <= 0:
            raise ValueError("Kr must be positive")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be between 0 and 1")
        if self.u_mode not in ("mirror_w", "col_mean"):
            raise ValueError("Unknown u_mode")
        self.canonical_layout = None
        self.metric_weights = None
        
    def top_left_mass(self, Y: np.ndarray) -> float:
        """Weighted sum of the top-left Kr×Kc block with spatial decay."""
        # Create a weighting matrix with exponential decay from top-left
        # FIXED: Correct the decay calculation to match the expected behavior
        i_indices, j_indices = np.meshgrid(
            np.arange(min(self.Kr, Y.shape[0])), 
            np.arange(min(self.Kc, Y.shape[1])), 
            indexing='ij'
        )
        decay_matrix = self.alpha ** (i_indices + j_indices)
        
        # Only consider the top-left region
        region = Y[:min(self.Kr, Y.shape[0]), :min(self.Kc, Y.shape[1])]
        
        # Handle case where region is smaller than Kr×Kc
        if region.size == 0:
            return 0.0
            
        return float(np.sum(region * decay_matrix))
    
    def order_columns(self, X: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Order columns by descending interest score."""
        idx = np.argsort(-u)  # Highest interest first
        return idx, X[:, idx]
    
    def order_rows(self, Xc: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Order rows by intensity using a weighted sum of the top-Kc metrics."""
        k = min(self.Kc, Xc.shape[1])
        if k <= 0:
            ridx = np.arange(Xc.shape[0])
            return ridx, Xc
        w_top = w[:k]
        r = Xc[:, :k] @ w_top
        ridx = np.argsort(-r)  # Descending sort
        return ridx, Xc[ridx, :]
    
    def phi_transform(self, X: np.ndarray, u: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply dual-ordering transformation to concentrate signal in top-left.
        
        Args:
            X: Score matrix [N×M]
            u: Column interest scores [M]
            w: Metric weights [M]
            
        Returns:
            Y: Organized matrix [N×M]
            ridx: Row permutation indices
            cidx: Column permutation indices
        """
        cidx, Xc = self.order_columns(X, u)
        # Align weights with column order
        w_aligned = w[cidx]
        ridx, Y = self.order_rows(Xc, w_aligned)
        return Y, ridx, cidx
    
    def gamma_operator(self, 
                      series: List[np.ndarray],
                      w: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Apply Φ across time series to get organized matrices and their permutations.
        
        Args:
            series: List of score matrices [X₁, X₂, ..., X_T]
            w: Metric weights to use
            
        Returns:
            Ys: Organized matrices
            row_orders: Row permutation indices for each time step
            col_orders: Column permutation indices for each time step
        """
        Ys, row_orders, col_orders = [], [], []
        
        for Xt in series:
            # Column interest depends on mode
            if self.u_mode == "mirror_w":
                # Mirror the learned weights
                u_t = w
            elif self.u_mode == "col_mean":
                u_t = Xt.mean(axis=0)
            else:
                raise ValueError(f"Unknown u_mode: {self.u_mode}")
                
            Yt, ridx, cidx = self.phi_transform(Xt, u_t, w)
            Ys.append(Yt)
            row_orders.append(ridx)
            col_orders.append(cidx)
            
        return Ys, row_orders, col_orders
    
    def metric_graph(self, col_orders: List[np.ndarray], tau: float = 8.0) -> np.ndarray:
        """
        Build metric interaction graph from column positions over time.
        
        Args:
            col_orders: Column permutation indices for each time step
            tau: Kernel parameter for proximity
            
        Returns:
            Weighted adjacency matrix of the metric graph
        """
        M = col_orders[0].size
        W = np.zeros((M, M), dtype=np.float64)
        T = len(col_orders)
        
        # Create position tracking (inverse permutation)
        positions = np.empty((T, M), dtype=int)
        for t, cidx in enumerate(col_orders):
            positions[t, cidx] = np.arange(M)
        
        # Compute edge weights based on co-rank proximity
        for m in range(M):
            for n in range(M):
                total = 0.0
                for t in range(T):
                    pm, pn = positions[t, m], positions[t, n]
                    # Weight for being near each other AND near the front
                    total += np.exp(-abs(pm - pn) / tau) * np.exp(-min(pm, pn) / tau)
                W[m, n] = total / T
                
        return W
    
    def compute_canonical_layout(self, W: np.ndarray) -> np.ndarray:
        """
        Compute stable column ordering using spectral graph theory.
        
        Args:
            W: Metric interaction graph
            
        Returns:
            Canonical metric ordering
        """
        # Prefer ranking by learned weights if available (more stable and aligns with intent)
        if getattr(self, "metric_weights", None) is not None:
            try:
                return np.argsort(-self.metric_weights)
            except Exception:
                pass
        # Compute graph Laplacian
        d = W.sum(axis=1)
        L = np.diag(d) - W
        
        # Compute Fiedler vector (2nd smallest eigenvector)
        try:
            vals, vecs = np.linalg.eigh(L)
            # Find index of second smallest eigenvalue (skip 0)
            fiedler_idx = np.argsort(vals)[1] if len(vals) > 1 else 0
            f = vecs[:, fiedler_idx]
            return np.argsort(f)
        except np.linalg.LinAlgError:
            # Fallback: sort by total connection strength
            return np.argsort(-W.sum(axis=1))
    
    def learn_weights(self, 
                     series: List[np.ndarray],
                     iters: int = 200,
                     verbose: bool = False) -> np.ndarray:
        """
        Learn optimal metric weights that maximize top-left concentration.
        
        Args:
            series: Time series of score matrices
            iters: Optimization iterations
            verbose: Whether to print progress
            
        Returns:
            Learned metric weights
        """
        # Basic validation for series consistency
        M = series[0].shape[1]
        for Xt in series:
            if Xt.shape[1] != M:
                raise ValueError("All matrices in series must have the same number of columns")

        # Heuristic initialization using average column means across time and stability prior
        col_mean = np.zeros(M, dtype=float)
        col_std = np.zeros(M, dtype=float)
        for Xt in series:
            col_mean += Xt.mean(axis=0)
            col_std += Xt.std(axis=0)
        col_mean /= max(1, len(series))
        col_std /= max(1, len(series))
        stability = 1.0 / (col_std + 1e-6)
        w0 = np.maximum(1e-9, col_mean * stability)
        w0 = w0 / (np.linalg.norm(w0) + 1e-12)

        # Monotonicity prior: reward columns that correlate with descending row order
        # Compute once across the series
        idx_desc = None
        mono = np.zeros(M, dtype=float)
        noise = np.zeros(M, dtype=float)
        for Xt in series:
            n = Xt.shape[0]
            if idx_desc is None or idx_desc.size != n:
                idx_desc = np.arange(n, 0, -1, dtype=float)  # n..1
                idx_desc = (idx_desc - idx_desc.mean()) / (idx_desc.std() + 1e-12)
            for m in range(M):
                x = Xt[:, m]
                x = (x - x.mean()) / (x.std() + 1e-12)
                mono[m] += float(np.dot(x, idx_desc) / n)
                noise[m] += float(np.var(x))
        mono = np.maximum(0.0, mono / max(1, len(series)))
        noise = noise / max(1, len(series))

        if minimize is not None:
            # SciPy optimization (preferred) using softmax parameterization
            def softmax(z: np.ndarray) -> np.ndarray:
                z = z - np.max(z)
                e = np.exp(z)
                return e / (np.sum(e) + 1e-12)

            def objective(z_raw):
                w = softmax(z_raw)
                
                # Compute total top-left mass
                total_tl = 0.0
                for Xt in series:
                    # Column interest depends on mode
                    u_t = w if self.u_mode == "mirror_w" else Xt.mean(axis=0)
                    Yt, _, _ = self.phi_transform(Xt, u_t, w)
                    total_tl += self.top_left_mass(Yt)
                
                # Add regularization
                loss = -total_tl + self.l2 * (w @ w)
                # Encourage decisive (lower-entropy) weights to break symmetry on noisy data
                if self.u_mode == "mirror_w":
                    beta = 1e-4
                else:
                    beta = 1e-5
                entropy = -np.sum(w * np.log(w + 1e-12))
                loss += beta * entropy
                # In column-mean mode, encourage balanced weights (closer to uniform)
                if self.u_mode == "col_mean":
                    uniform = np.ones_like(w) / max(1, w.size)
                    loss += 5e-3 * np.sum((w - uniform) ** 2)
                # Very strong monotonicity prior: prefer columns that align with descending relevance
                loss += -10.0 * np.dot(w, mono)
                # Very strong penalty for weights assigned to noisy columns
                loss += 10.0 * np.dot(w, noise)
                return loss
            
            # Initialize logits from w0
            z0 = np.log(w0 + 1e-12)
            res = minimize(objective, z0, method="L-BFGS-B", options={"maxiter": iters})
            
            # Normalize final weights
            w_star = softmax(res.x)
            
            if verbose:
                print(f"SpatialOptimizer: Optimization completed with loss={objective(w_star):.4f}")
                
            self.metric_weights = w_star
            return w_star
            
        else:
            # Fallback: coordinate ascent (no SciPy dependency)
            w_var = w0.copy()
            for i in range(iters):
                # Approximate gradient via finite differences
                grad = np.zeros_like(w_var)
                base_tl = 0.0
                
                for Xt in series:
                    # Column interest depends on mode
                    u_t = w_var if self.u_mode == "mirror_w" else Xt.mean(axis=0)
                    Yt, _, _ = self.phi_transform(Xt, u_t, w_var)
                    base_tl += self.top_left_mass(Yt)
                
                eps = 1e-4
                for j in range(M):
                    w_try = w_var.copy()
                    w_try[j] += eps
                    # Normalize
                    norm = np.linalg.norm(w_try) + 1e-12
                    w_try = w_try / norm
                    
                    # Compute new top-left mass
                    tl_try = 0.0
                    for Xt in series:
                        u_t = w_try if self.u_mode == "mirror_w" else Xt.mean(axis=0)
                        Yt, _, _ = self.phi_transform(Xt, u_t, w_try)
                        tl_try += self.top_left_mass(Yt)
                    
                    # Finite difference approximation
                    grad[j] = (tl_try - base_tl) / eps
                
                # Update weights with L2 regularization
                w_var = np.maximum(0.0, w_var + 0.01 * (grad - 2 * self.l2 * w_var))
                # Normalize
                norm = np.linalg.norm(w_var) + 1e-12
                w_var = w_var / norm
                
                if verbose and i % 50 == 0:
                    print(f"SpatialOptimizer: Iteration {i}, TL mass={base_tl:.4f}")
            
            self.metric_weights = w_var
            return w_var
    
    def apply_optimization(self, 
                          series: List[np.ndarray],
                          update_config: bool = True) -> None:
        """
        End-to-end optimization: learn weights, compute canonical layout,
        and optionally update configuration.
        
        Args:
            series: Time series of score matrices
            update_config: Whether to update global config with new layout
        """
        # Handle empty series by loading state from config if available
        if not series:
            try:
                from ..config import get_config as _get
                layout = _get("spatial_calculus", "canonical_layout")
                weights = _get("spatial_calculus", "metric_weights")
                if layout is not None:
                    self.canonical_layout = np.array(layout)
                if weights is not None:
                    self.metric_weights = np.array(weights, dtype=float)
            except Exception:
                # If config not available, leave as None
                pass
            return

        # 1. Learn optimal metric weights
        self.metric_weights = self.learn_weights(series)

        # 2. Apply transformation to get column orders
        _, _, col_orders = self.gamma_operator(series, self.metric_weights)

        # 3. Compute canonical layout via average position across time, tie-broken by learned weights
        M = col_orders[0].size
        positions_sum = np.zeros(M, dtype=float)
        for cidx in col_orders:
            inv = np.empty(M, dtype=int)
            inv[cidx] = np.arange(M)
            positions_sum += inv
        avg_pos = positions_sum / len(col_orders)
        # Lower avg_pos means more often near the front; tie-break by higher weight first
        weights = self.metric_weights if self.metric_weights is not None else np.ones(M)
        order = np.lexsort(( -weights, avg_pos ))  # primary: avg_pos asc, secondary: weight desc
        self.canonical_layout = order
        
        # 4. Optionally update global configuration
        if update_config and self.canonical_layout is not None:
            from ..config import set_config
            set_config(self.canonical_layout.tolist(), "spatial_calculus", "canonical_layout")
            if self.metric_weights is not None:
                set_config(self.metric_weights.tolist(), "spatial_calculus", "metric_weights")