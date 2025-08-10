import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, List, Callable, Optional

class DecisionManifold:
    """Represents the Spatial-Temporal Decision Manifold for ZeroModel."""
    
    def __init__(self, time_series: List[np.ndarray]):
        """
        Initialize with a time series of score matrices.
        
        Args:
            time_series: List of matrices [M_t1, M_t2, ...] where
                        M_t ∈ ℝ^(S×V) (S=sources, V=metrics)
        """
        self.time_series = time_series
        self.T = len(time_series)
        self.S = time_series[0].shape[0]
        self.V = time_series[0].shape[1]
        self.organized_series = []
        self.metric_orders = []
        self.source_orders = []
        self.metric_graph = None
        
    def organize(self, 
                metric_priority_fn: Callable[[int], np.ndarray] = None,
                intensity_weight: np.ndarray = None) -> None:
        """
        Apply the organizing operator Φ across all time slices.
        
        Args:
            metric_priority_fn: Function that returns metric priority order for time t
            intensity_weight: Weight vector for intensity calculation
        """
        if intensity_weight is None:
            # Default: equal weight for all metrics
            intensity_weight = np.ones(self.V) / self.V
        
        for t, M_t in enumerate(self.time_series):
            # Apply column permutation (metric ordering)
            if metric_priority_fn:
                metric_order = metric_priority_fn(t)
            else:
                # Default: sort by variance (more variable = more important)
                variances = np.var(M_t, axis=0)
                metric_order = np.argsort(-variances)
                
            P_col = np.eye(self.V)[:, metric_order]
            M_col = M_t @ P_col
            
            # Apply row permutation (source ordering)
            row_scores = M_col @ intensity_weight
            source_order = np.argsort(-row_scores)
            P_row = np.eye(self.S)[source_order]
            
            M_star = P_row @ M_col
            
            self.organized_series.append(M_star)
            self.metric_orders.append(metric_order)
            self.source_orders.append(source_order)
    
    def compute_metric_graph(self, tau: float = 2.0) -> np.ndarray:
        """
        Compute the metric interaction graph.
        
        Args:
            tau: Kernel parameter for front proximity
            
        Returns:
            Weighted adjacency matrix of the metric graph
        """
        # Convert metric orders to positions (1-based for kernel)
        positions = np.array(self.metric_orders) + 1
        T, V = positions.shape
        
        # Compute kernel: k(a) = exp(-(a-1)/tau)
        kernel = np.exp(-(positions - 1) / tau)
        
        # Compute edge weights: W_mn = (1/T) * Σ_t k(pos_t(m)) * k(pos_t(n))
        W = np.zeros((V, V))
        for m in range(V):
            for n in range(V):
                W[m, n] = np.mean(kernel[:, m] * kernel[:, n])
        
        self.metric_graph = W
        return W
    
    def find_critical_manifold(self, theta: float = 0.8) -> Dict[Tuple[int, int, int], float]:
        """
        Identify the critical manifold where relevance exceeds threshold.
        
        Args:
            theta: Relevance threshold (0-1)
            
        Returns:
            Dictionary of points in the critical manifold with their values
        """
        critical_points = {}
        
        for t, M_star in enumerate(self.organized_series):
            max_val = np.max(M_star)
            threshold = theta * max_val
            
            # Find all points above threshold
            indices = np.where(M_star >= threshold)
            
            for i, j in zip(indices[0], indices[1]):
                critical_points[(i, j, t)] = M_star[i, j]
        
        return critical_points
    
    def compute_curvature(self) -> np.ndarray:
        """
        Compute decision curvature across time.
        
        Returns:
            Array of curvature values for each time step (except boundaries)
        """
        if not self.organized_series:
            raise ValueError("Must call organize() first")
        
        # Stack all organized matrices along time dimension
        manifold = np.stack(self.organized_series, axis=-1)
        T = manifold.shape[-1]
        
        # Compute first and second derivatives
        d1 = np.diff(manifold, axis=-1)
        d2 = np.diff(d1, axis=-1)
        
        # Compute Frobenius norm of second derivative (curvature)
        curvature = np.zeros(T)
        curvature[1:-1] = np.linalg.norm(d2, axis=(0, 1))
        
        return curvature
    
    def find_inflection_points(self, threshold: float = 0.1) -> List[int]:
        """
        Find time steps with significant decision curvature (inflection points).
        
        Args:
            threshold: Curvature threshold to consider as inflection
            
        Returns:
            List of time indices with significant curvature
        """
        curvature = self.compute_curvature()
        inflection_points = np.where(curvature > threshold)[0]
        return inflection_points.tolist()
    
    def get_decision_flow(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the decision flow field at time t.
        
        Args:
            t: Time index
            
        Returns:
            Tuple of (x_gradient, y_gradient) representing the flow field
        """
        if t < 0 or t >= len(self.organized_series):
            raise ValueError(f"Time index {t} out of bounds [0, {len(self.organized_series)-1}]")
        
        M_star = self.organized_series[t]
        
        # Compute gradients (negative because higher values = more relevant)
        dy, dx = np.gradient(-M_star)
        
        return dx, dy
    
    def find_decision_rivers(self, t: int, num_rivers: int = 3) -> List[List[Tuple[int, int]]]:
        """
        Identify decision rivers (paths of steepest ascent) in the decision landscape.
        
        Args:
            t: Time index
            num_rivers: Number of rivers to identify
            
        Returns:
            List of paths (each path is a list of (i,j) coordinates)
        """
        # Bounds checking
        if t < 0 or t >= len(self.organized_series):
            raise ValueError(f"Time index {t} out of bounds [0, {len(self.organized_series)-1}]")
        
        M_star = self.organized_series[t]
        dx, dy = np.gradient(M_star)

        # Create a cost matrix (lower cost = more relevant)
        max_val = np.max(M_star)
        if max_val < 1e-10:  # Avoid division by zero
            max_val = 1e-10
        cost = np.sqrt(dx**2 + dy**2) + (1 - M_star / max_val)

        # Find local maxima (good seeds)
        maxima = []
        for i in range(1, M_star.shape[0]-1):
            for j in range(1, M_star.shape[1]-1):
                if (M_star[i,j] > M_star[i-1,j] and
                    M_star[i,j] > M_star[i+1,j] and
                    M_star[i,j] > M_star[i,j-1] and
                    M_star[i,j] > M_star[i,j+1]):
                    maxima.append((i, j, M_star[i,j]))

        # Sort maxima by value (highest relevance first)
        maxima.sort(key=lambda x: -x[2])
        
        # Keep only the top num_rivers maxima
        maxima = maxima[:num_rivers]
        
        # If no maxima found, return empty list
        if not maxima:
            return []
        
        # Build 4-neighbour sparse graph from the cost grid (required by dijkstra)
        H, W = cost.shape
        N = H * W
        rows = []
        cols = []
        data = []
        idx = np.arange(N).reshape(H, W)
        for i in range(H):
            for j in range(W):
                u = idx[i, j]
                cu = cost[i, j]
                if i + 1 < H:
                    v = idx[i + 1, j]
                    w = 0.5 * (cu + cost[i + 1, j])
                    rows += [u, v]
                    cols += [v, u]
                    data += [w, w]
                if j + 1 < W:
                    v = idx[i, j + 1]
                    w = 0.5 * (cu + cost[i, j + 1])
                    rows += [u, v]
                    cols += [v, u]
                    data += [w, w]
        
        graph = csr_matrix((data, (rows, cols)), shape=(N, N))

        # For each seed, find the lowest-cost path to the global maximum
        rivers = []
        max_node = np.argmax(M_star)
        for si, sj, _ in maxima:
            start = si * W + sj
            _, predecessors = dijkstra(
                csgraph=graph, 
                directed=False,
                indices=start, 
                return_predecessors=True
            )
            
            # Reconstruct path start -> max
            path = []
            cur = max_node
            if predecessors[cur] == -9999:
                # no path found (shouldn't happen on connected grid); skip
                continue
                
            while cur != -9999 and cur != start:
                ci, cj = divmod(cur, W)
                path.append((ci, cj))
                cur = predecessors[cur]
                
            si2, sj2 = divmod(start, W)
            path.append((si2, sj2))
            rivers.append(path[::-1])
            
        return rivers