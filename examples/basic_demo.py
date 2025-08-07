# examples/basic_demo.py
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
from zeromodel import EdgeProtocol, HierarchicalVPM, ZeroModel, HierarchicalEdgeProtocol


def generate_synthetic_data(
    num_docs: int = 100, num_metrics: int = 50
) -> Tuple[np.ndarray, List[str]]:
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
        "uncertainty",
        "size",
        "quality",
        "novelty",
        "coherence",
        "relevance",
        "diversity",
        "complexity",
        "readability",
        "accuracy",
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
    print("=" * 50)
    print("Zero-Model Intelligence (zeromodel) Demonstration")
    print("=" * 50)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic policy evaluation data...")
    score_matrix, metric_names = generate_synthetic_data(num_docs=100, num_metrics=20)
    print(
        f"   Generated {score_matrix.shape[0]} documents × {score_matrix.shape[1]} metrics"
    )

    # 2. Process with zeromodel
    print("\n2. Processing data with zeromodel...")
    zeromodel = ZeroModel(metric_names)

    # Example task 1: Find uncertain large documents
    print("   Processing for task: 'Find uncertain large documents'")
    zeromodel.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC")
    vpm1 = zeromodel.encode()

    # Example task 2: Find high-quality novel documents
    print("   Processing for task: 'Find high-quality novel documents'")
    zeromodel.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY quality DESC, novelty DESC")

    vpm2 = zeromodel.encode()

    # 3. Visualize results
    print("\n3. Visualizing results...")
    visualize_vpm(
        vpm1, "Uncertain Large Documents Task", "demo/vpm_uncertain_large.png"
    )
    visualize_vpm(
        vpm2, "High-Quality Novel Documents Task", "demo/vpm_high_quality.png"
    )

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


# Add this function to the demo
def demo_hierarchical_vpm():
    """Demonstrate hierarchical Visual Policy Maps"""
    print("\n" + "=" * 50)
    print("Hierarchical Visual Policy Maps Demonstration")
    print("=" * 50)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic policy evaluation data...")
    score_matrix, metric_names = generate_synthetic_data(num_docs=100, num_metrics=20)
    print(
        f"   Generated {score_matrix.shape[0]} documents × {score_matrix.shape[1]} metrics"
    )

    # 2. Create hierarchical VPM
    print("\n2. Creating hierarchical Visual Policy Map...")
    hvpm = HierarchicalVPM(metric_names=metric_names)
    hvpm.process(
        score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC"
    )

    # 3. Show level information
    print("\n3. Hierarchical levels information:")
    for i, level in enumerate(hvpm.levels):
        meta = level["metadata"]
        print(f"   Level {i} (Type: {level['type']}):")
        print(f"      - {meta['documents']} documents")
        print(f"      - {meta['metrics']} metrics")
        print(f"      - VPM shape: {level['vpm'].shape}")

    # 4. Demonstrate hierarchical decision making
    print("\n4. Hierarchical decision process:")
    current_level = 0
    doc_idx = 0
    metric_idx = 0

    for step in range(3):
        # Get decision at current level
        level, doc_idx, relevance = hvpm.get_decision(current_level)
        print(
            f"   Step {step + 1}: Level {level} decision - Document #{doc_idx} (relevance: {relevance:.2f})"
        )

        # Determine next level to zoom into
        current_level = hvpm.zoom_in(current_level, doc_idx, metric_idx)
        if current_level == level:
            print("      Reached most detailed level")
            break

    # 5. Edge device simulation
    print("\n5. Edge device hierarchical interaction:")
    # Level 0 tile (most abstract)
    tile0 = hvpm.get_tile(0)
    print(f"   Level 0 tile size: {len(tile0)} bytes")

    # Edge device processes tile and decides to zoom in
    decision = HierarchicalEdgeProtocol.make_decision(tile0)
    level = decision[2]
    is_relevant = decision[3]
    print(
        f"   Edge device decision (Level {level}): {'RELEVANT' if is_relevant else 'NOT RELEVANT'}"
    )

    # Edge device requests zoom
    zoom_request = HierarchicalEdgeProtocol.request_zoom(tile0, "in")
    new_level = zoom_request[3]
    print(f"   Edge device requests zoom to Level {new_level}")

    # Get tile for new level
    tile1 = hvpm.get_tile(new_level)
    print(f"   Level {new_level} tile size: {len(tile1)} bytes")

    print("\nHierarchical VPM demonstration complete!")

    # Add hierarchical demo


if __name__ == "__main__":
    demo_zeromodel()
    demo_hierarchical_vpm()
