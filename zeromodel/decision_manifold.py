import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, List, Callable

class DecisionManifold:
    """
    Represents a Spatial-Temporal Decision Manifold for analyzing dynamic decision landscapes.
    
    This class processes time-series data of multi-metric scores to:
    - Organize decision spaces using metric/source permutations
    - Compute metric interaction graphs
    - Identify critical decision regions
    - Analyze decision curvature and inflection points
    - Trace decision pathways (rivers)
    
    Attributes:
        time_series (List[np.ndarray]): Original time series of score matrices
        T (int): Number of time steps
        S (int): Number of data sources
        V (int): Number of evaluation metrics
        organized_series (List[np.ndarray]): Reorganized matrices after applying Φ-operator
        metric_orders (List[np.ndarray]): Metric permutation indices per time step
        source_orders (List[np.ndarray]): Source permutation indices per time step
        metric_graph (np.ndarray): Metric interaction graph adjacency matrix
    """
    
    def __init__(self, time_series: List[np.ndarray]):
        """
        Initialize decision manifold with time-series score data.

        Args:
            time_series: List of matrices [M_t1, M_t2, ...] where
                         M_t ∈ ℝ^(S×V) (S = sources, V = metrics)
        
        Raises:
            ValueError: If matrices have inconsistent dimensions
        """
        # Validate input consistency
        if not all(m.shape == time_series[0].shape for m in time_series):
            raise ValueError("All matrices in time_series must have same dimensions")
        
        self.time_series = time_series
        self.T = len(time_series)
        self.S = time_series[0].shape[0]  # Sources dimension
        self.V = time_series[0].shape[1]  # Metrics dimension
        self.organized_series = []      # Φ-transformed matrices
        self.metric_orders = []         # Metric permutations per timestep
        self.source_orders = []         # Source permutations per timestep
        self.metric_graph = None       # Metric interaction graph

    def organize(self, 
                metric_priority_fn: Callable[[int], np.ndarray] = None,
                intensity_weight: np.ndarray = None) -> None:
        """
        Apply organizing operator Φ to all time slices (column and row permutations).
        
        Transformation pipeline:
        1. Metric ordering (columns): Prioritize metrics via permutation
        2. Source ordering (rows): Sort sources by relevance intensity
        
        Args:
            metric_priority_fn: Function f(t) → metric permutation indices for time t.
                                If None, uses variance-based prioritization.
            intensity_weight: Weight vector for relevance calculation (size V).
                              If None, uses uniform weights.
                              
        Example:
            >>> dm = DecisionManifold([np.random.rand(5,3)])
            >>> dm.organize()
            >>> print(dm.organized_series[0].shape)
            (5, 3)
        """
        # Default to uniform metric weights if not provided
        if intensity_weight is None:
            intensity_weight = np.ones(self.V) / self.V
        
        for t, M_t in enumerate(self.time_series):
            # -- Metric ordering --
            if metric_priority_fn:
                metric_order = metric_priority_fn(t)  # Custom priority
            else:
                variances = np.var(M_t, axis=0)       # Default: variance sort
                metric_order = np.argsort(-variances)  # Descending order
                
            # Apply column permutation
            P_col = np.eye(self.V)[:, metric_order]
            M_col = M_t @ P_col
            
            # -- Source ordering --
            row_scores = M_col @ intensity_weight  # Relevance scores
            source_order = np.argsort(-row_scores)  # Descending sort
            P_row = np.eye(self.S)[source_order]
            
            # Apply row permutation and store
            self.organized_series.append(P_row @ M_col)
            self.metric_orders.append(metric_order)
            self.source_orders.append(source_order)

    def compute_metric_graph(self, tau: float = 2.0) -> np.ndarray:
        """
        Compute metric interaction graph using kernelized position similarity.
        
        Edge weight formula:
        W_mn = (1/T) ∑ₜ [exp(-posₜ(m)/τ) * exp(-posₜ(n)/τ)]
        
        Higher weights indicate metrics that consistently appear together in prominent positions.
        
        Args:
            tau: Exponential decay parameter (small τ = sharper position decay)
            
        Returns:
            V×V adjacency matrix of metric graph
            
        Example:
            >>> W = dm.compute_metric_graph(tau=1.5)
            >>> print(W.shape)
            (3, 3)
        """
        # Convert orders to 1-based positions
        positions = np.array(self.metric_orders) + 1
        T, V = positions.shape
        
        # Compute position kernels
        kernel = np.exp(-(positions - 1) / tau)
        
        # Compute pairwise metric affinity
        W = np.zeros((V, V))
        for m in range(V):
            for n in range(V):
                W[m, n] = np.mean(kernel[:, m] * kernel[:, n])
        
        self.metric_graph = W
        return W

    def find_critical_manifold(self, theta: float = 0.8) -> Dict[Tuple[int, int, int], float]:
        """
        Identify critical regions where relevance ≥ θ × global maximum.
        
        Args:
            theta: Relative threshold (0.0-1.0)
            
        Returns:
            {(i, j, t): value} mapping for critical coordinates
            
        Example:
            >>> crit = dm.find_critical_manifold(theta=0.9)
            >>> print(list(crit.keys())[:2])
            [(2, 1, 0), (3, 2, 0)]
        """
        critical_points = {}
        for t, M_star in enumerate(self.organized_series):
            threshold = theta * np.max(M_star)
            for i, j in zip(*np.where(M_star >= threshold)):
                critical_points[(i, j, t)] = M_star[i, j]
        return critical_points

    def compute_curvature(self) -> np.ndarray:
        """
        Compute temporal curvature via second derivative Frobenius norms.
        
        Returns:
            curvature: Array of length T (boundaries = 0)
            
        Raises:
            ValueError: If organize() hasn't been called
            
        Example:
            >>> curv = dm.compute_curvature()
            >>> print(curv.shape)
            (10,)
        """
        if not self.organized_series:
            raise ValueError("Call organize() before computing curvature")
        
        manifold = np.stack(self.organized_series, axis=-1)
        d1 = np.diff(manifold, axis=-1)
        d2 = np.diff(d1, axis=-1)
        
        curvature = np.zeros(manifold.shape[-1])
        curvature[1:-1] = np.linalg.norm(d2, axis=(0, 1))
        return curvature

    def find_inflection_points(self, threshold: float = 0.1) -> List[int]:
        """
        Detect time steps with significant decision landscape changes.
        
        Args:
            threshold: Minimum curvature magnitude
            
        Returns:
            List of significant time indices
            
        Example:
            >>> inflections = dm.find_inflection_points(threshold=0.15)
            >>> print(inflections)
            [5, 7]
        """
        curvature = self.compute_curvature()
        return np.where(curvature > threshold)[0].tolist()

    def get_decision_flow(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute decision flow field as negative gradient of relevance surface.
        
        Args:
            t: Time index
            
        Returns:
            dx: Gradient component along metric dimension
            dy: Gradient component along source dimension
            
        Raises:
            IndexError: For invalid time index
            
        Example:
            >>> dx, dy = dm.get_decision_flow(0)
            >>> print(dx.shape, dy.shape)
            (5, 3) (5, 3)
        """
        if not (0 <= t < len(self.organized_series)):
            raise IndexError(f"t must be in [0, {len(self.organized_series)-1}]")
        
        dy, dx = np.gradient(-self.organized_series[t])
        return dx, dy

    def find_decision_rivers(self, t: int, num_rivers: int = 3) -> List[List[Tuple[int, int]]]:
        """
        Trace steepest-ascent paths from local maxima to global maximum.
        
        Args:
            t: Time index
            num_rivers: Maximum number of rivers to return
            
        Returns:
            List of paths where each path is [(i0,j0), (i1,j1), ...]
            
        Example:
            >>> rivers = dm.find_decision_rivers(t=0, num_rivers=2)
            >>> print(len(rivers[0]))
            7
        """
        # Validate input
        if not (0 <= t < len(self.organized_series)):
            raise IndexError(f"t must be in [0, {len(self.organized_series)-1}]")
        
        M_star = self.organized_series[t]
        H, W = M_star.shape
        
        # Create cost surface
        dx, dy = np.gradient(M_star)
        max_val = np.max(M_star) or 1e-10
        cost = np.sqrt(dx**2 + dy**2) + (1 - M_star/max_val)
        
        # Find local maxima
        maxima = []
        for i in range(1, H-1):
            for j in range(1, W-1):
                if (M_star[i, j] > M_star[i-1:i+2, j].mean() and 
                    M_star[i, j] > M_star[i, j-1:j+2].mean()):
                    maxima.append((i, j, M_star[i, j]))
        maxima.sort(key=lambda x: -x[2])
        maxima = maxima[:num_rivers]
        
        # Build grid graph
        N = H * W
        idx = np.arange(N).reshape(H, W)
        rows, cols, data = [], [], []
        for i in range(H):
            for j in range(W):
                u = idx[i, j]
                if i+1 < H:  # Down neighbor
                    v = idx[i+1, j]
                    w = 0.5*(cost[i,j] + cost[i+1,j])
                    rows += [u, v]; cols += [v, u]; data += [w, w]
                if j+1 < W:  # Right neighbor
                    v = idx[i, j+1]
                    w = 0.5*(cost[i,j] + cost[i,j+1])
                    rows += [u, v]; cols += [v, u]; data += [w, w]
        graph = csr_matrix((data, (rows, cols)), shape=(N, N))
        
        # Trace paths
        rivers = []
        global_max = np.argmax(M_star)
        for i, j, _ in maxima:
            start = i*W + j
            _, preds = dijkstra(graph, False, start, return_predecessors=True)
            path = []
            current = global_max
            while current != start and current != -9999:
                path.append(divmod(current, W))
                current = preds[current]
            if current == start:
                path.append((i, j))
                rivers.append(path[::-1])
        return rivers