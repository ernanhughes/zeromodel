# iris_example.py
"""
Example Application: Iris Flower Classification using ZeroModel Intelligence

This script demonstrates how to use ZeroModel to process the Iris dataset,
define a task (e.g., "Find Setosa flowers"), and generate a Visual Policy Map (VPM).
The VPM image spatially organizes the data such that the most relevant documents
(flowers) and metrics (attributes) for the task are positioned in the top-left corner.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from zeromodel.core import ZeroModel # Adjust import path if needed (e.g., from zeromodel import ZeroModel)
from zeromodel.config import get_config

def load_and_prepare_iris_data():
    """
    Load the Iris dataset and prepare it for ZeroModel processing.
    Returns:
        score_matrix (np.ndarray): Normalized feature matrix (samples x features).
        metric_names (List[str]): Names of the features.
        target_names (List[str]): Names of the Iris species.
        y (np.ndarray): True labels for reference.
    """
    print("--- Loading and Preparing Iris Data ---")
    # 1. Load the dataset
    iris = load_iris()
    X_raw = iris.data  # Raw feature data (150 samples x 4 features)
    y = iris.target     # True class labels (0, 1, 2)
    target_names = list(iris.target_names) # ['setosa', 'versicolor', 'virginica']
    metric_names = list(iris.feature_names) # ['sepal length (cm)', 'sepal width (cm)', ...]
    print(f"Loaded Iris dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features.")
    print(f"Features: {metric_names}")
    print(f"Classes: {target_names}")

    # 2. Data Preparation (Normalization)
    # ZeroModel often works best with data normalized to a [0, 1] range.
    # We can use Min-Max scaling for this.
    # For demonstration, we'll also show standardization as an alternative.
    print("\nNormalizing feature data to [0, 1] range using Min-Max scaling...")
    
    # Min-Max Scaling: X_scaled = (X - X_min) / (X_max - X_min)
    col_min = X_raw.min(axis=0)
    col_max = X_raw.max(axis=0)
    col_range = col_max - col_min
    # Avoid division by zero if a column has constant value (though not the case for Iris)
    col_range = np.where(col_range == 0, 1, col_range) 
    score_matrix = (X_raw - col_min) / col_range
    
    print(f"Original data range per feature: min={X_raw.min(axis=0)}, max={X_raw.max(axis=0)}")
    print(f"Normalized data range per feature: min={score_matrix.min(axis=0)}, max={score_matrix.max(axis=0)}")
    
    # --- Optional: Demonstrate Standardization ---
    # Standardization (Z-score): X_standardized = (X - mean) / std
    # scaler = StandardScaler()
    # score_matrix_standardized = scaler.fit_transform(X_raw)
    # print(f"Standardized data (mean~0, std~1) range per feature: min={score_matrix_standardized.min(axis=0)}, max={score_matrix_standardized.max(axis=0)}")
    # Note: Standardization can produce negative values and values outside [0,1].
    # While ZeroModel can handle this, Min-Max scaling to [0,1] often aligns better
    # with the typical VPM interpretation (0=black/dark, 1=white/bright).
    # --- End Optional ---
    
    print("--- Data Preparation Complete ---\n")
    return score_matrix, metric_names, target_names, y

def define_and_process_task(score_matrix, metric_names, task_description, task_sql_query):
    """
    Define a task for ZeroModel, process the data, and generate the VPM.
    Args:
        score_matrix (np.ndarray): Normalized data.
        metric_names (List[str]): Feature names.
        task_description (str): Human-readable task description.
        task_sql_query (str): SQL query defining the sorting/ordering logic.
    Returns:
        zeromodel (ZeroModel): The prepared ZeroModel instance.
        vpm_image (np.ndarray): The encoded VPM image.
    """
    print(f"--- Processing Task: '{task_description}' ---")
    print(f"SQL Query: {task_sql_query}")
    
    # 1. Initialize ZeroModel
    # Precision determines internal bit depth, default is usually fine (e.g., 8 or 16).
    # default_output_precision='float32' makes the VPM a float32 array by default.
    get_config("core").update({"precision": 16})
    zeromodel = ZeroModel(metric_names)
    get_config("core").update({"precision": 8})
    print(f"Initialized ZeroModel with metrics: {metric_names}")

    # 2. Prepare the data using the SQL task
    # This is the core step: it loads data into DuckDB, applies the SQL sorting,
    # and organizes the internal sorted_matrix.
    try:
        zeromodel.prepare(score_matrix, task_sql_query)
        print("Data prepared successfully using the SQL task.")
    except Exception as e:
        print(f"Error during ZeroModel preparation: {e}")
        raise

    # 3. Encode the processed data into a Visual Policy Map (VPM)
    # The VPM is an image array where structure encodes task logic.
    # We request a float32 output for visualization clarity.
    try:
        vpm_image = zeromodel.encode(output_precision='float32')
        print(f"VPM encoded successfully. Shape: {vpm_image.shape}, Dtype: {vpm_image.dtype}")
    except Exception as e:
        print(f"Error during VPM encoding: {e}")
        raise

    print("--- Task Processing Complete ---\n")
    return zeromodel, vpm_image

def visualize_vpm(vpm_image, title="Iris Visual Policy Map"):
    """
    Visualize the generated VPM using Matplotlib.
    Args:
        vpm_image (np.ndarray): The VPM image array (H x W x 3).
        title (str): Title for the plot.
    """
    print(f"--- Visualizing VPM: '{title}' ---")
    if vpm_image is None:
        print("No VPM image to visualize.")
        return

    plt.figure(figsize=(8, 6))
    # Display the VPM. Since it's float32 [0, 1], we don't need a specific cmap for grayscale,
    # but 'viridis' or 'gray' can be used. 'gray' might be more intuitive for [0,1] data.
    # As it's an RGB image, imshow handles it correctly.
    plt.imshow(vpm_image, cmap='viridis') # or cmap='gray'
    plt.title(title)
    plt.xlabel("Metrics (Pixels, 3 metrics per pixel)")
    plt.ylabel("Documents (Samples)")
    # Add a colorbar to show the intensity scale [0, 1]
    cbar = plt.colorbar(label='Normalized Score Intensity [0.0, 1.0]')
    # The colorbar helps interpret pixel brightness.
    # Darker (closer to 0.0) might mean lower priority/value for that metric/doc.
    # Brighter (closer to 1.0) means higher priority/value.
    plt.tight_layout()
    # Save the image
    filename = title.lower().replace(" ", "_").replace(":", "") + ".png"
    plt.savefig(filename)
    print(f"VPM image saved as '{filename}'.")
    plt.show()
    print("--- VPM Visualization Complete ---\n")

def analyze_results(zeromodel, y_true, target_names, task_description):
    """
    Analyze the results of the ZeroModel decision.
    Args:
        zeromodel (ZeroModel): The prepared and processed ZeroModel instance.
        y_true (np.ndarray): True labels.
        target_names (List[str]): Names of classes.
        task_description (str): The task that was performed.
    """
    print(f"--- Analyzing Results for Task: '{task_description}' ---")
    try:
        # Get the top decision from ZeroModel
        # This looks at the top-left region of the sorted_matrix/VPM.
        doc_index, relevance_score = zeromodel.get_decision()
        print("ZeroModel Decision:")
        print(f"  - Top Document Index: {doc_index}")
        print(f"  - Relevance Score: {relevance_score:.4f} (Range: 0.0 to 1.0)")
        print(f"  - Interpreted as: The {task_description} is most likely document {doc_index}.")

        # Reference the true label for context (this is for demonstration only)
        if doc_index < len(y_true):
            true_label_index = y_true[doc_index]
            true_species = target_names[true_label_index]
            print(f"  - True Label for Doc {doc_index}: Class {true_label_index} ({true_species})")
        else:
            print(f"  - Warning: Document index {doc_index} is out of bounds for true labels (size {len(y_true)}).")

        # --- Example: Get a Critical Tile for Edge Device Simulation ---
        print("\nSimulating Edge Device Decision with Critical Tile:")
        # Get a small tile (e.g., 3x3 pixels) from the top-left
        tile_bytes = zeromodel.get_critical_tile(tile_size=3, precision='float32')
        # The tile contains width, height, x/y offsets, and pixel data.
        # An edge device would parse this.
        tile_width = tile_bytes[0]
        tile_height = tile_bytes[1]
        print(f"  - Retrieved Critical Tile: {tile_width}x{tile_height} pixels")
        print(f"  - Tile Data Size: {len(tile_bytes)} bytes")
        # A simple edge check: is the very first pixel (top-left of tile) "bright enough"?
        # The interpretation depends on the task. For "Find Setosa" (which might be low in some metrics),
        # a different logic might be needed. Let's assume high relevance means bright top-left.
        # The actual first pixel data starts after the 4-byte header.
        if len(tile_bytes) > 4:
            # For float32 tile, interpreting raw bytes is complex.
            # Let's just check the relevance score from get_decision instead.
            # A simple rule: if relevance > 0.7, consider it a strong signal.
            edge_decision_threshold = 0.7
            if relevance_score > edge_decision_threshold:
                print(f"  - Edge Decision: ✅ Strong signal detected (Relevance {relevance_score:.4f} > {edge_decision_threshold}).")
            else:
                print(f"  - Edge Decision: ❓ Weak signal (Relevance {relevance_score:.4f} <= {edge_decision_threshold}). Request higher resolution tile or cloud processing.")
        else:
             print("  - Edge Decision: Insufficient tile data.")
        # --- End Edge Device Simulation ---

    except Exception as e:
        print(f"Error during result analysis: {e}")
    print("--- Result Analysis Complete ---\n")


def print_top_ranked_documents(zeromodel, score_matrix, y_true, target_names, num_docs=10):
    """
    Prints the top-ranked documents from the ZeroModel's sorted_matrix,
    showing their original indices, true labels, and feature values.
    Args:
        zeromodel (ZeroModel): The prepared ZeroModel instance.
        score_matrix (np.ndarray): The original normalized input data.
        y_true (np.ndarray): The original true labels.
        target_names (List[str]): Names of the classes.
        num_docs (int): Number of top documents to print.
    """
    print(f"--- Top {num_docs} Ranked Documents (Based on Task) ---")
    if zeromodel.sorted_matrix is None:
        print("ZeroModel has not been prepared yet.")
        return

    if zeromodel.doc_order is None:
        print("Document order is not available.")
        # Fallback: assume identity order if not explicitly set by sorter
        doc_order = np.arange(zeromodel.sorted_matrix.shape[0])
    else:
        doc_order = zeromodel.doc_order

    if zeromodel.metric_order is None:
         print("Warning: Metric order is not available, using original metric names for display.")
         # Use original metric names
         display_metric_names = zeromodel.metric_names
    else:
        # Reorder metric names to match the sorted matrix columns
        display_metric_names = [zeromodel.metric_names[i] for i in zeromodel.metric_order]


    print(f"{'Rank':<5} {'Orig Idx':<9} {'True Label':<15} {'Sepal L':<8} {'Sepal W':<8} {'Petal L':<8} {'Petal W':<8}")
    print("-" * 70)
    for i in range(min(num_docs, zeromodel.sorted_matrix.shape[0])):
        # Get the index of this document in the original score_matrix/y_true
        original_index = doc_order[i]
        # Get the true label index and name
        true_label_idx = y_true[original_index]
        true_label_name = target_names[true_label_idx]
        # Get the sorted feature values for this document
        sorted_features = zeromodel.sorted_matrix[i, :4] # Assuming first 4 metrics are the iris features
        # Get the original feature values for comparison/context
        original_features = score_matrix[original_index, :4]

        print(f"{i+1:<5} {original_index:<9} {true_label_name:<15} "
              f"{sorted_features[0]:<8.4f} {sorted_features[1]:<8.4f} "
              f"{sorted_features[2]:<8.4f} {sorted_features[3]:<8.4f}")
        # Optional: Print original values in a second line for direct comparison
        # print(f"      {'(orig)':<9} {'':<15} "
        #       f"{original_features[0]:<8.4f} {original_features[1]:<8.4f} "
        #       f"{original_features[2]:<8.4f} {original_features[3]:<8.4f}")

    print("-" * 70)
    print("Legend:")
    print("  Rank: Position in the sorted ZeroModel matrix (1 = highest priority for task)")
    print("  Orig Idx: Index of this document in the original input data")
    print("  True Label: The actual Iris species of the document")
    print("  Sepal L/W, Petal L/W: Normalized feature values (sorted by task logic)")
    print("--- End of Top Ranked Documents ---\n")


def main():
    """Main function to run the Iris example."""
    print("="*50)
    print("🤖 ZeroModel Intelligence: Iris Dataset Example")
    print("="*50)
    
    # 1. Load and prepare the data
    score_matrix, metric_names, target_names, y_true = load_and_prepare_iris_data()

    # 2. Define a task: "Find Setosa flowers"
    # Setosa flowers are known to have distinctive sepal and petal characteristics.
    # Let's craft an SQL query that prioritizes features typical of Setosa.
    # From domain knowledge or data inspection, Setosa tends to have:
    # - Smaller petal length and width.
    # - Sometimes larger sepal width relative to length.
    # We can create a score that rewards these characteristics.
    # For this example, let's define a task that looks for low petal measurements.
    # We'll sort by Petal Width ascending (low PW first) then Petal Length ascending.
    task_desc_find_setosa = "Find Setosa flowers (prioritize low petal size)"
    # SQL query to sort documents: Low Petal Width first, then Low Petal Length
    task_sql_find_setosa = """
    SELECT * FROM virtual_index 
    ORDER BY "petal width (cm)" ASC, "petal length (cm)" ASC
    """
    # Note: Column names in the SQL query must match exactly with metric_names.
    # DuckDB is case-sensitive for quoted identifiers.

    # 3. Process the task and generate the VPM
    zm_setosa, vpm_setosa = define_and_process_task(
        score_matrix, metric_names, task_desc_find_setosa, task_sql_find_setosa
    )

    print_top_ranked_documents(zm_setosa, score_matrix, y_true, target_names, num_docs=10)
    # This prints the top 10 documents based on the task logic.

    # 4. Visualize the VPM
    visualize_vpm(vpm_setosa, title=f"Iris VPM: {task_desc_find_setosa}")

    # 5. Analyze the results
    analyze_results(zm_setosa, y_true, target_names, task_desc_find_setosa)

    # --- Example 2: A different task ---
    print("\n" + "="*50)
    print("🔄 Running a Second Task Example")
    print("="*50)

    # Define a different task: "Find flowers with large sepals"
    task_desc_large_sepals = "Find flowers with large sepals"
    # Sort by Sepal Length descending, then Sepal Width descending
    task_sql_large_sepals = """
    SELECT * FROM virtual_index 
    ORDER BY "sepal length (cm)" DESC, "sepal width (cm)" DESC
    """

    zm_large_sepals, vpm_large_sepals = define_and_process_task(
        score_matrix, metric_names, task_desc_large_sepals, task_sql_large_sepals
    )

    print("\n" + "="*50)
    print(f"Printing top documents for task: '{task_desc_large_sepals}'")
    print("="*50)
    print_top_ranked_documents(zm_large_sepals, score_matrix, y_true, target_names, num_docs=10)

    visualize_vpm(vpm_large_sepals, title=f"Iris VPM: {task_desc_large_sepals}")

    analyze_results(zm_large_sepals, y_true, target_names, task_desc_large_sepals)
    # --- End Example 2 ---

    print("="*50)
    print("✅ Iris Example Completed")
    print("="*50)
    print("\n🧠 Key Takeaway:")
    print("The structure of the VPM image encodes the task logic.")
    print("In the 'Find Setosa' VPM, documents identified as Setosa by the sorting")
    print("criteria should appear clustered towards the top rows.")
    print("Similarly, the 'large sepals' VPM will show those flowers at the top.")
    print("The intelligence lies in how the data is organized, not in complex processing.")

if __name__ == "__main__":
    main()
