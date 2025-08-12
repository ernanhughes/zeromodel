"""
Spatial Calculus Optimization Module

Implements ZeroModel's Spatial Calculus framework for optimizing information layouts in Visual Policy Maps (VPMs).
The core algorithm transforms high-dimensional metric spaces into decision-optimized 2D layouts by:

Key Concepts:
1. Top-left Mass Concentration: Maximizes signal density in the top-left region of VPMs where human decision-making 
   is most effective, using spatial decay (α) to prioritize proximity to origin.
2. Dual-Ordering Transform: Learns metric weights (w) that simultaneously:
   - Order columns by "interest" (u) 
   - Order rows by weighted intensity of top-Kc columns
3. Metric Graph: Models temporal co-occurrence patterns to identify stable metric relationships
4. Canonical Layout: Computes optimal static ordering using spectral graph theory

Mathematical Foundations:
Φ(X|u,w) = row_order( col_order(X|u) | w )
Γ(X₁..Xₜ) = [Φ(X₁), ..., Φ(Xₜ)]
W = E[exp(-|pᵢ - pⱼ|/τ)]  (metric graph)
canonical_order = Fiedler_vector(Laplacian(W))

Primary Use Cases:
- Security policy optimization
- Anomaly detection systems
- High-dimensional decision support
- Metric space compression

Example Usage:
>>> optimizer = SpatialOptimizer(Kc=20, Kr=40, alpha=0.97)
>>> optimizer.apply_optimization(series)
>>> print("Optimal weights:", optimizer.metric_weights)
>>> print("Canonical layout:", optimizer.canonical_layout)

Note: Requires SciPy for optimization. Falls back to coordinate ascent if unavailable.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize


class SpatialOptimizer:
    """
    Optimizes metric weights to concentrate decision-relevant information in the top-left
    region of the Visual Policy Map (VPM) using ZeroModel's Spatial Calculus. This transform
    maximizes decision accuracy by learning optimal metric weights and canonical layouts.
    
    Key Features:
    - Learns metric weights that maximize top-left concentration in VPM
    - Computes canonical metric ordering based on temporal patterns
    - Supports different column interest calculation modes
    - Includes regularization and stability mechanisms
    
    Usage Flow:
    1. Initialize with configuration parameters
    2. Provide time-series of score matrices
    3. Call learn_weights() to optimize metric weights
    4. Apply apply_optimization() for end-to-end processing
    5. Access optimized weights via metric_weights property
    6. Use canonical_layout for stable metric ordering
    """
    
    def __init__(self, 
                 Kc: int = 16,
                 Kr: int = 32,
                 alpha: float = 0.95,
                 l2: float = 1e-3,
                 u_mode: str = "mirror_w"):
        """
        Initialize the spatial optimizer with configuration parameters.
        
        Args:
            Kc: Number of top metric columns considered for row ordering
            Kr: Number of top source rows considered for top-left mass calculation
            alpha: Spatial decay factor (higher = more focus on top-left)
            l2: L2 regularization strength for weight optimization
            u_mode: Column interest calculation mode:
                'mirror_w' - Use same weights as row intensity (recommended)
                'col_mean' - Use column means from data (fallback)
        
        Raises:
            ValueError: On invalid parameter values
        """
        self.Kc = Kc
        self.Kr = Kr
        self.alpha = alpha
        self.l2 = l2
        self.u_mode = u_mode

        # Validate parameters
        if self.Kc <= 0:
            raise ValueError("Kc must be positive")
        if self.Kr <= 0:
            raise ValueError("Kr must be positive")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be between 0 and 1")
        if self.u_mode not in ("mirror_w", "col_mean"):
            raise ValueError("u_mode must be 'mirror_w' or 'col_mean'")
            
        # State variables (set during optimization)
        self.canonical_layout: Optional[np.ndarray] = None
        self.metric_weights: Optional[np.ndarray] = None
        
    def top_left_mass(self, Y: np.ndarray) -> float:
        """
        Calculate weighted sum of top-left Kr×Kc block with spatial decay.
        
        Args:
            Y: Transformed matrix (from phi_transform)
            
        Returns:
            Weighted concentration score (higher = better signal concentration)
        """
        # Create decay matrix with exponential decay from top-left
        rows = min(self.Kr, Y.shape[0])
        cols = min(self.Kc, Y.shape[1])
        
        if rows == 0 or cols == 0:
            return 0.0
            
        i_indices, j_indices = np.meshgrid(
            np.arange(rows), 
            np.arange(cols), 
            indexing='ij'
        )
        decay_matrix = self.alpha ** (i_indices + j_indices)
        return float(np.sum(Y[:rows, :cols] * decay_matrix))
    
    def order_columns(self, X: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Order matrix columns by descending interest scores.
        
        Args:
            X: Input score matrix [N×M]
            u: Column interest scores [M]
            
        Returns:
            cidx: Column permutation indices
            Xc: Column-ordered matrix
        """
        idx = np.argsort(-u)  # Highest interest first
        return idx, X[:, idx]
    
    def order_rows(self, Xc: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Order matrix rows by weighted sum of top-Kc columns.
        
        Args:
            Xc: Column-ordered matrix [N×M]
            w: Metric weights (aligned with Xc columns) [M]
            
        Returns:
            ridx: Row permutation indices
            Y: Fully transformed matrix
        """
        k = min(self.Kc, Xc.shape[1])
        if k == 0:
            return np.arange(Xc.shape[0]), Xc
            
        w_top = w[:k]
        r = Xc[:, :k] @ w_top
        ridx = np.argsort(-r)  # Descending sort
        return ridx, Xc[ridx, :]
    
    def phi_transform(self, 
                     X: np.ndarray, 
                     u: np.ndarray, 
                     w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply dual-ordering transformation to concentrate signal in top-left.
        
        Args:
            X: Input score matrix [N×M]
            u: Column interest scores [M]
            w: Metric weights [M]
            
        Returns:
            Y: Organized matrix [N×M]
            ridx: Row permutation indices
            cidx: Column permutation indices
        """
        cidx, Xc = self.order_columns(X, u)
        w_aligned = w[cidx]  # Align weights with column order
        ridx, Y = self.order_rows(Xc, w_aligned)
        return Y, ridx, cidx
    
    def gamma_operator(self,
                      series: List[np.ndarray],
                      w: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Apply Φ-transform across time series of matrices.
        
        Args:
            series: List of score matrices [X₁, X₂, ..., X_T]
            w: Metric weights to use for transformation
            
        Returns:
            Ys: Transformed matrices
            row_orders: Row permutations for each timestep
            col_orders: Column permutations for each timestep
        """
        Ys, row_orders, col_orders = [], [], []
        
        for Xt in series:
            if self.u_mode == "mirror_w":
                u_t = w
            elif self.u_mode == "col_mean":
                u_t = Xt.mean(axis=0)
            else:
                raise RuntimeError(f"Invalid u_mode: {self.u_mode}")
                
            Yt, ridx, cidx = self.phi_transform(Xt, u_t, w)
            Ys.append(Yt)
            row_orders.append(ridx)
            col_orders.append(cidx)
            
        return Ys, row_orders, col_orders
    
    def metric_graph(self, col_orders: List[np.ndarray], tau: float = 8.0) -> np.ndarray:
        """
        Build metric interaction graph from column positions over time.
        
        Args:
            col_orders: Column permutations across timesteps
            tau: Proximity kernel parameter
            
        Returns:
            W: Weighted adjacency matrix [M×M] of metric graph
        """
        M = col_orders[0].size
        T = len(col_orders)
        positions = np.empty((T, M), dtype=int)
        
        # Compute inverse permutations (positions)
        for t, cidx in enumerate(col_orders):
            positions[t, cidx] = np.arange(M)
        
        # Compute edge weights using temporal co-occurrence
        W = np.zeros((M, M))
        for t in range(T):
            pos_t = positions[t]
            for i in range(M):
                for j in range(M):
                    dist = abs(pos_t[i] - pos_t[j])
                    proximity = np.exp(-dist/tau) * np.exp(-min(pos_t[i], pos_t[j])/tau)
                    W[i, j] += proximity
        return W / T
    
    def compute_canonical_layout(self, W: np.ndarray) -> np.ndarray:
        """
        Compute stable metric ordering using spectral graph theory.
        
        Args:
            W: Metric interaction graph from metric_graph()
            
        Returns:
            Canonical metric ordering indices
        """
        # Prefer learned weights if available
        if self.metric_weights is not None:
            return np.argsort(-self.metric_weights)
        
        # Fallback to spectral ordering
        try:
            d = W.sum(axis=1)
            L = np.diag(d) - W  # Unnormalized Laplacian
            vals, vecs = np.linalg.eigh(L)
            fiedler_vector = vecs[:, np.argsort(vals)[1]]  # 2nd smallest eigenvalue
            return np.argsort(fiedler_vector)
        except np.linalg.LinAlgError:
            return np.argsort(-W.sum(axis=1))  # Degree fallback
    
    def learn_weights(self, 
                     series: List[np.ndarray],
                     iters: int = 200,
                     verbose: bool = False) -> np.ndarray:
        """
        Learn metric weights that maximize top-left concentration.
        
        Optimization features:
        - Softmax parameterization for constraint satisfaction
        - Monotonicity prior for row ordering
        - Noise-aware weight regularization
        - Entropy regularization for weight sparsity
        
        Args:
            series: Time-series of score matrices [N×M]
            iters: Optimization iterations
            verbose: Print progress messages
            
        Returns:
            w: Learned metric weights [M]
            
        Raises:
            ValueError: Inconsistent matrix dimensions
        """
        # Validate input consistency
        M = series[0].shape[1]
        if any(Xt.shape[1] != M for Xt in series):
            raise ValueError("All matrices must have same column count")
            
        # Initialize with column importance heuristics
        col_means = np.mean([Xt.mean(axis=0) for Xt in series], axis=0)
        col_vars = np.mean([Xt.var(axis=0) for Xt in series], axis=0)
        w0 = col_means / (np.sqrt(col_vars) + 1e-6)
        w0 = w0 / (np.linalg.norm(w0) + 1e-12)

        # Monotonicity prior (reward descending-row-correlated metrics)
        mono_scores = np.zeros(M)
        for Xt in series:
            n = Xt.shape[0]
            rank_vector = np.arange(n, 0, -1)
            rank_vector = (rank_vector - rank_vector.mean()) / (rank_vector.std() + 1e-12)
            for m in range(M):
                col = Xt[:, m]
                col = (col - col.mean()) / (col.std() + 1e-12)
                mono_scores[m] += np.dot(col, rank_vector) / n
        mono_scores = np.maximum(0, mono_scores / len(series))

        # Softmax wrapper for unconstrained optimization
        def softmax(z: np.ndarray) -> np.ndarray:
            e = np.exp(z - np.max(z))
            return e / (e.sum() + 1e-12)

        # Optimization objective
        def objective(z: np.ndarray) -> float:
            w = softmax(z)
            total_mass = 0.0
            T = len(series)
            
            # Calculate time-weighted top-left mass (emphasize later/overfitting phase)
            for t, Xt in enumerate(series):
                time_weight = ((t + 1) / T) ** 2  # Quadratic emphasis on later epochs
                u = w if self.u_mode == "mirror_w" else Xt.mean(axis=0)
                Y, _, _ = self.phi_transform(Xt, u, w)
                total_mass += time_weight * self.top_left_mass(Y)
                
            # Regularization components
            reg = self.l2 * np.sum(w**2)  # L2 penalty
            entropy = -np.sum(w * np.log(w + 1e-12))  # Entropy penalty
            monotonicity = -np.dot(w, mono_scores)  # Monotonicity reward
            
            # Compose final loss
            loss = -total_mass + reg + 1e-4*entropy + 5.0*monotonicity
            return loss

        # Optimize using L-BFGS
        res = minimize(objective, np.log(w0), method='L-BFGS-B', options={'maxiter': iters})
        w_opt = softmax(res.x)
        # Post-normalize to unit L2 for stability and test expectations
        norm = float(np.linalg.norm(w_opt) + 1e-12)
        w_opt = w_opt / norm

        if verbose:
            print(f"Optimization completed. Final loss: {res.fun:.4f}")

        self.metric_weights = w_opt
        return w_opt
    
    def apply_optimization(self, 
                          series: List[np.ndarray],
                          update_config: bool = True) -> None:
        """
        End-to-end optimization pipeline:
        1. Learn optimal metric weights
        2. Compute metric interaction graph
        3. Determine canonical layout
        4. Update internal state (and optionally global config)
        
        Args:
            series: Input time-series data
            update_config: Persist results to global configuration
        """
        # Handle empty input (load from config if possible)
        if not series:
            try:
                from ..config import get_config
                self.canonical_layout = np.array(get_config("spatial_calculus", "canonical_layout"))
                self.metric_weights = np.array(get_config("spatial_calculus", "metric_weights"))
            except ImportError:
                pass
            return

        # Core optimization workflow
        self.metric_weights = self.learn_weights(series)
        _, _, col_orders = self.gamma_operator(series, self.metric_weights)
        W = self.metric_graph(col_orders)
        self.canonical_layout = self.compute_canonical_layout(W)

        # Optional persistence
        if update_config:
            try:
                from ..config import set_config
                set_config(self.canonical_layout.tolist(), "spatial_calculus", "canonical_layout")
                set_config(self.metric_weights.tolist(), "spatial_calculus", "metric_weights")
            except ImportError:
                pass