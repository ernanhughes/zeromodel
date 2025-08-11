import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
from zeromodel.decision_manifold import DecisionManifold

def generate_test_data(T: int = 20, S: int = 50, V: int = 10, 
                      concept_drift: bool = True) -> List[np.ndarray]:
    """
    Generate synthetic test data with evolving decision structure.
    
    Args:
        T: Number of time steps
        S: Number of sources (rows)
        V: Number of metrics (columns)
        concept_drift: Whether to include concept drift
        
    Returns:
        List of score matrices [M_t1, M_t2, ...]
    """
    time_series = []
    
    for t in range(T):
        # Base pattern: two clusters of relevant items
        M = np.zeros((S, V))
        
        # Cluster 1: metrics 0-2 important
        cluster1_size = 10
        M[:cluster1_size, :3] = np.random.beta(3, 1, size=(cluster1_size, 3))
        
        # Cluster 2: metrics 3-5 important
        cluster2_size = 8
        M[cluster1_size:cluster1_size+cluster2_size, 3:6] = np.random.beta(3, 1, size=(cluster2_size, 3))
        
        # Noise
        M += np.random.normal(0, 0.05, size=(S, V))
        M = np.clip(M, 0, 1)
        
        # Concept drift: after t=10, Cluster 2 becomes more important
        if concept_drift and t > T//2:
            M[cluster1_size:cluster1_size+cluster2_size, 3:6] *= 1.5
            M = np.clip(M, 0, 1)
        
        time_series.append(M)
    
    return time_series

def visualize_decision_manifold():
    """Comprehensive visualization of the Spatial-Temporal Decision Manifold."""
    # Generate test data
    time_series = generate_test_data(T=20, concept_drift=True)
    
    # Create and organize the decision manifold
    dm = DecisionManifold(time_series)
    dm.organize()
    dm.compute_metric_graph()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)
    
    # 1. Original vs Organized Matrix (time t=10)
    t = 10
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(time_series[t], cmap='viridis', aspect='auto')
    ax1.set_title(f'Original Matrix (t={t})')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Sources')
    fig.colorbar(im1, ax=ax1, label='Score')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(dm.organized_series[t], cmap='viridis', aspect='auto')
    ax2.set_title(f'Organized Matrix (t={t})')
    ax2.set_xlabel('Metrics (Ordered)')
    ax2.set_ylabel('Sources (Ordered)')
    fig.colorbar(im2, ax=ax2, label='Score')
    
    # Highlight the critical manifold
    critical = dm.find_critical_manifold(theta=0.7)
    for (i, j, _), _ in critical.items():
        if i < 15 and j < 15:  # Only show top-left region
            ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                      fill=False, edgecolor='red', linewidth=1))
    
    # 2. Metric Interaction Graph
    ax3 = fig.add_subplot(gs[0, 2])
    W = dm.metric_graph
    
    # Create a circular layout for the graph
    n = W.shape[0]
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Plot nodes
    ax3.scatter(x, y, s=300, c='skyblue', zorder=5)
    
    # Plot edges (only strong connections)
    for i in range(n):
        for j in range(i+1, n):
            if W[i, j] > 0.3:  # Only show strong connections
                ax3.plot([x[i], x[j]], [y[i], y[j]], 
                        'gray', alpha=W[i, j], linewidth=2*W[i, j])
    
    # Label nodes
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax3.text(xi*1.15, yi*1.15, f'm{i}', 
                ha='center', va='center', fontweight='bold')
    
    ax3.set_title('Metric Interaction Graph')
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    # 3. Decision Curvature
    ax4 = fig.add_subplot(gs[1, :])
    curvature = dm.compute_curvature()
    inflection_points = dm.find_inflection_points(threshold=np.mean(curvature) + 0.5*np.std(curvature))
    
    ax4.plot(range(len(curvature)), curvature, 'b-', linewidth=2)
    ax4.scatter(inflection_points, curvature[inflection_points], 
               c='red', s=100, zorder=5, label='Inflection Points')
    
    ax4.set_title('Decision Curvature Over Time')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Curvature')
    ax4.grid(True)
    ax4.legend()
    
    # 4. Decision Rivers
    ax5 = fig.add_subplot(gs[2, 0])
    rivers = dm.find_decision_rivers(t)
    
    im5 = ax5.imshow(dm.organized_series[t], cmap='viridis', aspect='auto')
    for i, river in enumerate(rivers):
        path = np.array(river)
        ax5.plot(path[:, 1], path[:, 0], 'r-', linewidth=2, alpha=0.7)
        ax5.scatter(path[0, 1], path[0, 0], c='red', s=100, zorder=5)  # Start
        ax5.scatter(path[-1, 1], path[-1, 0], c='green', s=100, zorder=5)  # End
    
    ax5.set_title(f'Decision Rivers (t={t})')
    ax5.set_xlabel('Metrics')
    ax5.set_ylabel('Sources')
    fig.colorbar(im5, ax=ax5, label='Score')
    
    # 5. Critical Manifold Evolution
    ax6 = fig.add_subplot(gs[2, 1:])
    critical_sizes = []
    
    for t in range(len(time_series)):
        critical = dm.find_critical_manifold(theta=0.7)
        # Count points in top-left 10x10 region
        size = sum(1 for (i, j, t_val) in critical.keys() if i < 10 and j < 10 and t_val == t)
        critical_sizes.append(size)
    
    ax6.plot(range(len(critical_sizes)), critical_sizes, 'g-', linewidth=2)
    ax6.set_title('Critical Manifold Size Over Time')
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Size of Critical Region')
    ax6.grid(True)
    
    # Highlight concept drift point
    if len(time_series) > 10:
        ax6.axvline(x=10, color='r', linestyle='--', alpha=0.7)
        ax6.text(10.5, max(critical_sizes)*0.9, 'Concept Drift', 
                rotation=90, color='r')
    
    plt.tight_layout()
    plt.savefig('decision_manifold_visualization.png', dpi=300, bbox_inches='tight')

def test_decision_manifold():
    """Comprehensive test of the DecisionManifold class."""
    print("="*50)
    print("Testing Spatial-Temporal Decision Manifold")
    print("="*50)
    
    # Generate test data
    time_series = generate_test_data(T=15, S=100, V=12)
    print(f"Generated test data: {len(time_series)} time steps, each {time_series[0].shape}")
    
    # Create and organize the decision manifold
    dm = DecisionManifold(time_series)
    dm.organize()
    print("✓ Organized decision manifold across time")
    
    # Test metric graph
    W = dm.compute_metric_graph()
    assert W.shape == (12, 12), "Metric graph has wrong shape"
    assert np.all(W >= 0) and np.all(W <= 1), "Metric weights out of range [0,1]"
    print(f"✓ Computed metric interaction graph (shape: {W.shape})")
    
    # Test critical manifold
    critical = dm.find_critical_manifold(theta=0.7)
    assert len(critical) > 0, "Critical manifold is empty"
    print(f"✓ Found critical manifold with {len(critical)} points")
    
    # Test curvature
    curvature = dm.compute_curvature()
    assert len(curvature) == len(time_series), "Curvature length mismatch"
    print(f"✓ Computed decision curvature (max: {np.max(curvature):.4f})")
    
    # Test inflection points
    inflection = dm.find_inflection_points(threshold=np.mean(curvature) + 0.5*np.std(curvature))
    print(f"✓ Found {len(inflection)} inflection points at times: {inflection}")
    
    # Test decision rivers
    rivers = dm.find_decision_rivers(t=5)
    assert len(rivers) > 0, "No decision rivers found"
    print(f"✓ Found {len(rivers)} decision rivers")
    
    print("\nAll tests passed successfully!")
    print("="*50)
    
    # Visualize the manifold
    print("\nGenerating visualization...")
    visualize_decision_manifold()
    print("Visualization saved as 'decision_manifold_visualization.png'")

