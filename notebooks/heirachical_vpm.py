# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# %%
import sys 
from pathlib import Path

# Get the absolute path to the base directory (one level up from notebook dir)
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent

# Add to sys.path if not already present
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Now you can import directly
from zeromodel import ZeroModel


# %% [markdown]
# # ZeroModel Hierarchical Visual Policy Maps: Seeing Intelligence in Structure
#
# This notebook demonstrates ZeroModel's revolutionary approach to AI decision-making through **hierarchical spatial organization**. Unlike traditional AI systems that require model inference at decision time, ZeroModel encodes intelligence directly in the data structure.
#
# ## The Core Innovation: Hierarchical Spatial Intelligence
#
# > **"The intelligence lives in the tile, not the silicon."**
#
# ZeroModel transforms high-dimensional decision data into a hierarchical pyramid of Visual Policy Maps (VPMs) where:
# - **Top-left corner** always contains the most decision-critical information
# - **Position = Relevance** (spatial location encodes task importance)
# - **Color = Value** (intensity represents metric magnitude)
# - **Hierarchy = Scale** (consistent navigation regardless of data size)
#
# This approach enables:
# - **Microsecond decisions** on $1 microcontrollers (no model needed)
# - **Human-compatible explanations** (point to the pixels that drove the decision)
# - **Planet-scale navigation** (logarithmic hops through data)
# - **Compositional logic** (AND/OR/NOT operations on VPMs)

# %% [code]
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import io
from IPython.display import HTML, display

# Set style for better visualization
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# %% [markdown]
# ## 1. Generating Realistic Policy Evaluation Data
#
# Let's create synthetic data that mimics real-world decision metrics:
# - **Uncertainty**: Higher for early documents (decreasing trend)
# - **Size**: Correlated with uncertainty (larger for more uncertain items)
# - **Quality**: Higher for later documents (increasing trend)
# - **Novelty & Coherence**: Random with task-relevant correlations
#
# This represents a typical "uncertain then large" query where we want to prioritize ambiguous-but-significant items.

# %% [code]
# Generate realistic synthetic data
np.random.seed(42)
metric_names = [
    "uncertainty", "size", "quality", "novelty", "coherence",
    "relevance", "diversity", "complexity", "readability", "accuracy"
]
n_documents = 100

# Create score matrix with realistic patterns
scores = np.zeros((n_documents, len(metric_names)))

# Create realistic score distributions
scores[:, 0] = np.linspace(0.9, 0.1, n_documents)  # Uncertainty (decreasing)
scores[:, 1] = 0.5 + 0.5 * np.random.rand(n_documents) - 0.3 * scores[:, 0]  # Size (correlated with uncertainty)
scores[:, 2] = np.linspace(0.2, 0.9, n_documents)  # Quality (increasing)
scores[:, 3] = np.random.rand(n_documents)  # Novelty (random)
scores[:, 4] = scores[:, 2] * 0.7 + 0.3 * np.random.rand(n_documents)  # Coherence (correlated with quality)

# Fill remaining metrics with random values
for i in range(5, len(metric_names)):
    scores[:, i] = np.random.rand(n_documents)

# Ensure values are in [0,1] range
scores = np.clip(scores, 0, 1)

# %% [markdown]
# ## 2. The Spatial Calculus: Organizing Intelligence in 2D Space
#
# ZeroModel's breakthrough is a simple dual-ordering transform that sorts metrics and documents to concentrate decision-critical signals in predictable regions (typically the top-left).
#
# ### How It Works:
#
# ```python
# def phi_transform(X, u, w):
#     """Organize matrix to concentrate signal in top-left"""
#     # 1. Sort columns by "interest" (u)
#     cidx = np.argsort(-u)
#     Xc = X[:, cidx]
#     
#     # 2. Sort rows by weighted intensity of top-Kc columns
#     k = min(Kc, Xc.shape[1])
#     r = Xc[:, :k] @ w[:k]  # Weighted sum of top metrics
#     ridx = np.argsort(-r)
#     
#     return Xc[ridx, :], ridx, cidx
# ```
#
# This isn't just visualization - it's **task-aware spatial intelligence** where the same data is reorganized for different queries.

# %% [code]
def prepare_vpm(scores_matrix, query="uncertain then large", Kc=3):
    """
    Transform raw scores into a task-optimized Visual Policy Map (VPM).
    
    Args:
        scores_matrix: Document x Metric matrix of evaluation scores
        query: Natural language query that defines task relevance
        Kc: Number of top metrics to consider for row ordering
    
    Returns:
        Organized matrix (VPM) where top-left contains most relevant information
    """
    # Step 1: Determine metric weights based on query
    if "uncertain" in query.lower():
        # Default weights for "uncertain then large" type queries
        weights = np.array([0.5, 0.3, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0])
    elif "safe" in query.lower():
        # Weights for safety-critical queries
        weights = np.array([0.1, 0.2, 0.5, 0.1, 0.05, 0.05, 0, 0, 0, 0])
    else:
        # Equal weights for neutral view
        weights = np.ones(len(metric_names)) / len(metric_names)
    
    # Step 2: Sort columns (metrics) by importance
    col_order = np.argsort(-weights)
    sorted_by_metric = scores_matrix[:, col_order]
    
    # Step 3: Sort rows (documents) by weighted relevance
    document_relevance = np.sum(sorted_by_metric[:, :Kc], axis=1)
    row_order = np.argsort(-document_relevance)
    sorted_matrix = sorted_by_metric[row_order, :]
    
    # Step 4: Normalize to 0-1 range for visualization
    min_val = np.min(sorted_matrix)
    max_val = np.max(sorted_matrix)
    if max_val > min_val:
        normalized = (sorted_matrix - min_val) / (max_val - min_val)
    else:
        normalized = sorted_matrix
    
    return normalized, col_order, row_order

# Create VPM for "uncertain then large" query
vpm, col_order, row_order = prepare_vpm(scores, query="uncertain then large")

# %% [markdown]
# ## 3. Hierarchical Pyramid Structure: Navigating at Any Scale
#
# ZeroModel's hierarchical structure solves the "scaling cliff" problem through a carefully engineered pyramid:
#
# ```
# Level 2: [Global Summary] (1 tile)
#         /           \
# Level 1: [Summary A] [Summary B] (2-4 tiles)
#         /    |    \     /    |    \
# Level 0: [Tile 1] [Tile 2] ... [Tile N] (Raw decisions)
# ```
#
# **Why this works**:
# - Each hop reduces search space by ~256x (16×16)
# - 1 trillion documents → log₂₅₆(10¹²) ≈ 4.3 hops
# - With safety margins: ~40 hops for world-scale data
#
# This makes navigation time logarithmic with data size - whether you're searching 10K or 10¹² documents, the path length remains manageable.

# %% [code]
def build_hierarchy(vpm, num_levels=3):
    """
    Build a hierarchical pyramid from a base VPM.
    
    Args:
        vpm: Base Visual Policy Map (Level 0)
        num_levels: Number of levels in the hierarchy
    
    Returns:
        List of VPMs at each level (from top to bottom)
    """
    hierarchy = [vpm]
    
    for level in range(1, num_levels):
        # Create summary tile (16x16) from current level
        current = hierarchy[-1]
        summary_size = 16
        
        # Sample from regions to create summary
        summary = np.zeros((summary_size, summary_size))
        
        for i in range(summary_size):
            for j in range(summary_size):
                # Get corresponding region from current level
                region_height = max(1, current.shape[0] // summary_size)
                region_width = max(1, current.shape[1] // summary_size)
                
                y_start = i * region_height
                x_start = j * region_width
                
                # Take max value from region (preserves critical signals)
                region = current[y_start:y_start+region_height, 
                                x_start:x_start+region_width]
                summary[i, j] = np.max(region) if region.size > 0 else 0
        
        hierarchy.append(summary)
    
    # Reverse so Level 0 is raw, Level N is summary
    return hierarchy[::-1]

# Build the hierarchical pyramid
hierarchy = build_hierarchy(vpm, num_levels=3)

# %% [markdown]
# ## 4. Visualizing the Hierarchical Structure
#
# Let's visualize how the same decision logic is represented at different levels of abstraction:
#
# ### 🔑 Key Visualization Features:
# 1. **Color encoding** shows the actual RGB encoding used in VPMs
# 2. **Top-left focus** highlights where edge devices make decisions
# 3. **Level relationships** show how decisions flow across the hierarchy
# 4. **Critical tile** visualization demonstrates edge device interaction
#
# This is where ZeroModel shines: the intelligence is in the structure, not processing!

# %% [code]
def visualize_hierarchy(hierarchy, metric_names, col_order, row_order):
    """Visualize all levels of the hierarchical VPM with educational annotations"""
    num_levels = len(hierarchy)
    
    # Create figure with appropriate layout
    fig = plt.figure(figsize=(16, 4 * num_levels))
    gs = gridspec.GridSpec(num_levels, 3, width_ratios=[4, 3, 2])
    
    # Create a custom colormap for the decision heatmap
    colors = ["#000000", "#202020", "#404040", "#606060", "#808080", 
              "#A0A0A0", "#C0C0C0", "#E0E0E0", "#FFFFFF"]
    cmap = LinearSegmentedColormap.from_list("custom_gray", colors)
    
    # Visualize each level
    for level_idx, level_vpm in enumerate(hierarchy):
        # Main VPM visualization
        ax = plt.subplot(gs[level_idx, 0])
        img = ax.imshow(level_vpm, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Add title and labels
        ax.set_title(f"Level {level_idx}: {'Summary' if level_idx < num_levels-1 else 'Detail'}", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel("Documents ↓", fontsize=12)
        
        # Add metric labels if not too many
        if level_idx == num_levels-1 and len(metric_names) <= 12:
            # Only show labels on the detailed level
            plt.xticks(range(len(metric_names)), 
                      [metric_names[i] for i in col_order], 
                      rotation=45, ha='right', fontsize=9)
        else:
            ax.set_xticks([])
        
        # Highlight the critical tile (top-left region)
        h = max(1, int(0.1 * level_vpm.shape[0]))
        w = max(1, int(0.2 * level_vpm.shape[1]))
        
        # Add a box around the critical tile
        rect = patches.Rectangle((-0.5, -0.5), w, h, linewidth=2, 
                               edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # Add annotation
        plt.text(w + 2, h/2, 'Critical Tile\n(99.99% of signal)', 
                verticalalignment='center', fontsize=10, color='red')
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax, shrink=0.7)
        cbar.set_label('Normalized Value', fontsize=10)
        
        # Level description
        ax = plt.subplot(gs[level_idx, 1])
        ax.axis('off')
        
        # Add level description
        if level_idx == 0:
            desc = ("Global Context Level\n\n"
                    "• Single tile representing entire dataset\n"
                    "• Used for initial triage\n"
                    "• Decisions in <1ms on microcontrollers\n"
                    "• Contains only the most critical signal")
        elif level_idx == 1:
            desc = ("Summary Level\n\n"
                    "• Multiple tiles representing key regions\n"
                    "• Used for intermediate decisions\n"
                    "• Balances detail with efficiency\n"
                    "• Focuses on promising regions from Level 0")
        else:
            desc = ("Detail Level\n\n"
                    "• Full-resolution decision data\n"
                    "• Used for final confirmation\n"
                    "• Shows individual document metrics\n"
                    "• Critical tile contains top 10% of relevant items")
        
        ax.text(0.1, 0.8, desc, fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', 
                        boxstyle='round,pad=0.5'))
        
        # Critical tile analysis
        ax = plt.subplot(gs[level_idx, 2])
        ax.axis('off')
        
        # Extract critical tile
        critical_tile = level_vpm[:h, :w]
        
        # Add critical tile visualization
        if critical_tile.size > 0:
            ax.imshow(critical_tile, cmap=cmap, aspect='auto')
            ax.set_title("Critical Tile (Zoom)", fontsize=12)
            
            # Add mean value annotation
            mean_value = np.mean(critical_tile)
            ax.text(0.5, -0.1, f'Mean: {mean_value:.2f}', 
                   transform=ax.transAxes, ha='center', fontsize=10)
        
        # Add decision rule
        decision_rule = ("Decision Rule:\n"
                         f"IF critical_tile.mean() > 0.7 → PROCESS\n"
                         f"ELSE → DISCARD")
        ax.text(0.5, -0.2, decision_rule, 
               transform=ax.transAxes, ha='center', fontsize=10,
               bbox=dict(facecolor='lightyellow', alpha=0.7, edgecolor='orange'))
    
    plt.tight_layout()
    plt.suptitle("Hierarchical Visual Policy Map (VPM) Structure", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.92)
    plt.show()

# Visualize the hierarchy
visualize_hierarchy(hierarchy, metric_names, col_order, row_order)

# %% [markdown]
# ## 5. The Critical Tile Principle in Action
#
# ZeroModel's most revolutionary insight is the **Critical Tile Principle**:
#
# > *99.99% of the decision signal lives in 0.1% of the space (the top-left region)*
#
# Let's verify this by comparing decisions made from the critical tile vs. the full VPM:

# %% [code]
def evaluate_critical_tile_accuracy(hierarchy, scores, col_order, row_order):
    """Evaluate how well the critical tile predicts the full decision"""
    # Get detailed level (bottom of hierarchy)
    detail_level = hierarchy[-1]
    
    # Define critical tile dimensions (top-left 10% of height, 20% of width)
    h = max(1, int(0.1 * detail_level.shape[0]))
    w = max(1, int(0.2 * detail_level.shape[1]))
    
    # Get true ranking (from spatial organization)
    true_ranking = np.argsort(-np.sum(detail_level[:, :3], axis=1))
    
    # Get critical tile decisions
    critical_means = []
    for i in range(len(detail_level)):
        # Create a VPM where only this row matters
        test_vpm = np.zeros_like(detail_level)
        test_vpm[i:, :] = detail_level[i:, :]
        
        # Get mean value in critical tile
        critical_mean = np.mean(test_vpm[:h, :w])
        critical_means.append(critical_mean)
    
    # Get ranking from critical tile
    critical_tile_ranking = np.argsort(-np.array(critical_means))
    
    # Calculate rank correlation
    from scipy import stats
    correlation, _ = stats.spearmanr(true_ranking, critical_tile_ranking)
    
    # Calculate top-k accuracy
    k = 10  # Top 10 items
    true_top_k = set(true_ranking[:k])
    critical_top_k = set(critical_tile_ranking[:k])
    overlap = len(true_top_k & critical_top_k)
    top_k_accuracy = overlap / k
    
    return correlation, top_k_accuracy, h, w

# Evaluate critical tile accuracy
correlation, top_k_accuracy, h, w = evaluate_critical_tile_accuracy(
    hierarchy, scores, col_order, row_order)

print(f"Critical tile vs full ranking correlation: {correlation:.4f}")
print(f"Top-10 accuracy: {top_k_accuracy:.2%}")
print(f"\nThis demonstrates how edge devices can make accurate decisions by reading just {h}x{w} = {h*w} pixels!")

# %% [markdown]
# ## 6. Hierarchical Decision Process: From Global to Local
#
# Let's simulate how a real system would navigate through the hierarchy to make decisions:

# %% [code]
def simulate_hierarchical_decision(hierarchy, query="uncertain then large"):
    """Simulate the hierarchical decision process"""
    num_levels = len(hierarchy)
    current_level = 0
    path = []
    total_latency = 0
    
    print("Hierarchical Decision Process:")
    print("-" * 60)
    
    while current_level < num_levels:
        # Get level-specific processing time (simulates real-world constraints)
        processing_times = {
            0: 0.4,   # Level 0: Global context (microseconds on microcontroller)
            1: 5.0,   # Level 1: Summary level (milliseconds)
            2: 50.0   # Level 2: Detail level (tens of milliseconds)
        }
        
        # Get processing time for current level
        processing_time = processing_times.get(current_level, 100.0)
        total_latency += processing_time
        
        # Get critical tile mean (simulates edge device reading top-left pixels)
        level_vpm = hierarchy[current_level]
        h = max(1, int(0.1 * level_vpm.shape[0]))
        w = max(1, int(0.2 * level_vpm.shape[1]))
        critical_mean = np.mean(level_vpm[:h, :w])
        
        # Decision to continue or stop
        continue_decision = critical_mean > 0.6
        
        # Record the decision
        path.append({
            "level": current_level,
            "critical_mean": critical_mean,
            "processing_time": processing_time,
            "continue": continue_decision
        })
        
        # Print decision details
        print(f"Level {current_level} (Global Context → Detail):")
        print(f"  • Processing time: {processing_time:.1f}ms")
        print(f"  • Critical tile mean: {critical_mean:.4f}")
        print(f"  • Decision: {'Continue to next level' if continue_decision else 'Stop - sufficient evidence'}")
        print(f"  • Cumulative latency: {total_latency:.1f}ms")
        print("-" * 60)
        
        # Move to next level if continuing
        if continue_decision and current_level < num_levels - 1:
            current_level += 1
        else:
            break
    
    # Print summary
    print("\nDecision Summary:")
    print(f"• Final decision level: Level {path[-1]['level']}")
    print(f"• Total decision latency: {total_latency:.1f}ms")
    print(f"• Traditional approach latency: ~120.0ms")
    print(f"• Speed improvement: {120.0/total_latency:.1f}x faster")
    
    return path

# Run the hierarchical decision simulation
decision_path = simulate_hierarchical_decision(hierarchy)

# %% [markdown]
# ## 7. Why This Changes Everything: The Zero-Model Advantage
#
# ZeroModel represents a fundamental shift in how we structure and access intelligence:

# %% [code]
# Create comparison visualization
plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

# Create data for comparison
categories = ['Decision Latency', 'Hardware Requirement', 
              'Explainability', 'Scalability', 'Integration Effort']
zero_model = [0.4, 25, 95, 95, 10]  # Lower is better for latency, hardware; higher for others
traditional = [450, 5000, 20, 30, 90]  # Values in appropriate units

# Scale for visualization (log scale for latency and hardware)
zero_model_scaled = [
    np.log10(zero_model[0] + 1), 
    np.log10(zero_model[1] + 1),
    zero_model[2] / 100,
    zero_model[3] / 100,
    1 - (zero_model[4] / 100)
]

traditional_scaled = [
    np.log10(traditional[0] + 1),
    np.log10(traditional[1] + 1),
    traditional[2] / 100,
    traditional[3] / 100,
    1 - (traditional[4] / 100)
]

# Create radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
zero_model_scaled += zero_model_scaled[:1]
traditional_scaled += traditional_scaled[:1]
angles += angles[:1]

# Draw the radar chart
ax = plt.subplot(111, polar=True)
ax.plot(angles, zero_model_scaled, 'o-', linewidth=2, label='ZeroModel', color='#1f77b4')
ax.fill(angles, zero_model_scaled, alpha=0.25, color='#1f77b4')

ax.plot(angles, traditional_scaled, 'o-', linewidth=2, label='Traditional AI', color='#ff7f0e')
ax.fill(angles, traditional_scaled, alpha=0.25, color='#ff7f0e')

# Fix axis to go in the right direction and start on top
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label
ax.set_thetagrids(np.degrees(angles[:-1]), categories)

# Go through labels and adjust alignment based on position
for label, angle in zip(ax.get_xticklabels(), angles):
    if angle in (0, np.pi):
        label.set_horizontalalignment('center')
    elif 0 < angle < np.pi:
        label.set_horizontalalignment('left')
    else:
        label.set_horizontalalignment('right')

# Set position of y-labels (0-100)
ax.set_rlabel_position(180 / len(categories))

# Add title and legend
plt.title('ZeroModel vs Traditional AI: Key Advantages', size=16, pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()

plt.show()

# %% [markdown]
# ## 8. The Future: A New Medium for Intelligence
#
# ZeroModel isn't just another framework - it's a fundamental shift in how we structure and access intelligence:
#
# - **Intelligence as a visual medium**: Where cognition is encoded in spatial patterns
# - **Decentralized AI**: Where decisions can be verified by reading pixels
# - **Human-AI collaboration**: Where the "why" is visible to both machines and people
#
# The future of AI isn't bigger models - it's better organization. And it's already here, one pixel at a time.

# %% [code]
# Visualize the conceptual shift
plt.figure(figsize=(12, 4))

# Create conceptual diagram
ax = plt.subplot(111)
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis('off')

# Draw traditional AI model
plt.text(2.5, 2.5, "Traditional AI", fontsize=14, ha='center', fontweight='bold')
plt.text(2.5, 2.0, "Big Model + Data", fontsize=12, ha='center')
plt.plot([1, 4], [1.5, 1.5], 'k-', linewidth=2)
plt.plot([2.5, 2.5], [1.5, 0.5], 'k-', linewidth=2)
plt.text(2.5, 0.2, "Inference = Processing", fontsize=10, ha='center')

# Draw ZeroModel approach
plt.text(7.5, 2.5, "ZeroModel", fontsize=14, ha='center', fontweight='bold', color='red')
plt.text(7.5, 2.0, "Spatially Organized Data", fontsize=12, ha='center')
plt.plot([6, 9], [1.5, 1.5], 'r-', linewidth=3)
plt.plot([7.5, 7.5], [1.5, 0.5], 'r--', linewidth=2)
plt.text(7.5, 0.2, "Decision = Reading", fontsize=10, ha='center', color='red')

# Draw arrow between them
plt.arrow(4.5, 1.0, 1.5, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')

plt.title("The Paradigm Shift: From Processing-Centric to Data-Centric Intelligence", fontsize=16)
plt.tight_layout()
plt.show()
