"""
zeromodel Demonstration Script

This script demonstrates the complete zeromodel workflow:
1. Generate synthetic score data
2. Process with zeromodel
3. Encode as visual policy map
4. Transform for specific task
5. Extract critical tile for edge device
6. Make decision

Run this script to see zeromodel in action.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from .core import ZeroModel
from .transform import transform_vpm, get_critical_tile
from .edge import EdgeProtocol

def generate_synthetic_data(num_docs: int = 100, num_metrics: int = 50) -> Tuple[np.ndarray, List[str]]:
    """Generate synthetic score data for demonstration"""
    # Create realistic score distributions
    scores = np.zeros((num_docs, num_metrics))
    
    # Uncertainty: higher for early documents
    scores[:, 0] = np.linspace(0.9, 0.1, num_docs)
    
    # Size: random but correlated with uncertainty
    scores[:, 1] = 0.5 + 0.5 * np.random.rand(num_docs) - 0.3 * scores[:, 0]
    
    # Quality: higher for later documents
    scores[:, 2] = np.linspace(0.2, 0.9, num_docs)
    
    # Novelty: random
    scores[:, 3] = np.random.rand(num_docs)
    
    # Coherence: correlated with quality
    scores[:, 4] = scores[:, 2] * 0.7 + 0.3 * np.random.rand(num_docs)
    
    # Fill remaining metrics with random values
    for i in range(5, num_metrics):
        scores[:, i] = np.random.rand(num_docs)
    
    # Ensure values are in [0,1] range
    scores = np.clip(scores, 0, 1)
    
    # Create metric names
    metric_names = [
        "uncertainty", "size", "quality", "novelty", "coherence",
        "relevance", "diversity", "complexity", "readability", "accuracy"
    ]
    # Add numbered metrics for the rest
    for i in range(10, num_metrics):
        metric_names.append(f"metric_{i}")
    
    return scores[:num_docs, :num_metrics], metric_names[:num_metrics]

def visualize_vpm(vpm: np.ndarray, title: str, output_path: str = None):
    """Visualize a visual policy map"""
    plt.figure(figsize=(10, 8))
    plt.imshow(vpm)
    plt.title(title)
    plt.xlabel("Metrics (sorted by importance)")
    plt.ylabel("Documents (sorted by relevance)")
    plt.colorbar(label="Score (0-255)")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    plt.close()

def demo_zeromodel():
    """Run the complete zeromodel demonstration"""
    print("="*50)
    print("Zero-Model Intelligence (zeromodel) Demonstration")
    print("="*50)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic policy evaluation data...")
    score_matrix, metric_names = generate_synthetic_data(num_docs=100, num_metrics=20)
    print(f"   Generated {score_matrix.shape[0]} documents × {score_matrix.shape[1]} metrics")
    
    # 2. Process with zeromodel
    print("\n2. Processing data with zeromodel...")
    zeromodel = zeromodel(metric_names)
    
    # Example task 1: Find uncertain large documents
    print("   Processing for task: 'Find uncertain large documents'")
    zeromodel.set_task("Find uncertain large documents")
    zeromodel.process(score_matrix)
    vpm1 = zeromodel.encode()
    
    # Example task 2: Find high-quality novel documents
    print("   Processing for task: 'Find high-quality novel documents'")
    zeromodel.set_task("Find high-quality novel documents")
    zeromodel.process(score_matrix)
    vpm2 = zeromodel.encode()
    
    # 3. Visualize results
    print("\n3. Visualizing results...")
    visualize_vpm(vpm1, "Uncertain Large Documents Task", "demo/vpm_uncertain_large.png")
    visualize_vpm(vpm2, "High-Quality Novel Documents Task", "demo/vpm_high_quality.png")
    
    # 4. Edge device simulation
    print("\n4. Simulating edge device decision making...")
    tile = zeromodel.get_critical_tile()
    print(f"   Critical tile size: {len(tile)} bytes")
    
    # Edge device would run minimal code like:
    # is_relevant = tile[4] < 128  # Check top-left pixel
    
    decision = EdgeProtocol.make_decision(tile)
    is_relevant = decision[2]
    print(f"   Edge device decision: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
    
    # 5. Get top decision
    doc_idx, relevance = zeromodel.get_decision()
    print(f"\n5. Top decision: Document #{doc_idx} with relevance {relevance:.2f}")
    
    print("\nzeromodel demonstration complete!")
    print("Check the 'demo' directory for visualizations.")

if __name__ == "__main__":
    demo_zeromodel()