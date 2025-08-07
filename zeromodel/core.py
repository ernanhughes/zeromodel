# zeromodel/core.py
"""
Zero-Model Intelligence Encoder/Decoder with DuckDB SQL Processing.

This module provides the core functionality for transforming high-dimensional
policy evaluation data into spatially-optimized visual maps where the
intelligence is in the data structure itself, not in processing.
"""

import logging
import re
from typing import List, Tuple, Dict, Any, Optional

import duckdb
import numpy as np
from zeromodel.normalizer import DynamicNormalizer

# Import the package logger
logger = logging.getLogger(__name__)  # This will be 'zeromodel.core'
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed output

# zeromodel/core.py (Relevant parts updated)
# (Assuming other imports and logger setup are already present)

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
import duckdb

# Assuming DynamicNormalizer is imported or defined elsewhere in the file
# from .normalizer import DynamicNormalizer

logger = logging.getLogger(__name__)

class ZeroModel:
    """
    Zero-Model Intelligence encoder/decoder with DuckDB SQL processing.

    This class transforms high-dimensional policy evaluation data into
    spatially-optimized visual maps where:
    - Position = Importance (top-left = most relevant)
    - Color = Value (darker = higher priority)
    - Structure = Task logic

    The intelligence is in the data structure itself, not in processing.
    """

    def __init__(self, metric_names: List[str], precision: int = 8, default_output_precision: str = 'float32'):
        """
        Initialize ZeroModel encoder with DuckDB SQL processing.

        Args:
            metric_names: Names of all metrics being tracked.
            precision: Bit precision for internal processing (4-16). (Legacy/kept for compatibility)
            default_output_precision: Default dtype for encode() output.
                                      Supported: 'uint8', 'float16', 'float32', 'float64'.
                                      This affects the default output of encode() and get_critical_tile().
        Raises:
            ValueError: If metric_names is empty or default_output_precision is invalid.
        """
        logger.debug(f"Initializing ZeroModel with metrics: {metric_names}, precision: {precision}, default_output_precision: {default_output_precision}")
        if not metric_names:
            error_msg = "metric_names list cannot be empty."
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Validate default_output_precision
        valid_precisions = ['uint8', 'float16', 'float32', 'float64']
        if default_output_precision not in valid_precisions:
             logger.warning(f"Invalid default_output_precision '{default_output_precision}'. Must be one of {valid_precisions}. Defaulting to 'float32'.")
             default_output_precision = 'float32'
             
        self.metric_names = list(metric_names) # Ensure it's a list
        self.precision = max(4, min(16, precision)) # Legacy precision param
        # --- NEW ATTRIBUTE: Default Output Precision ---
        self.default_output_precision = default_output_precision
        # --- END NEW ATTRIBUTE ---
        # ... (rest of __init__ attributes: sorted_matrix, doc_order, etc.) ...
        self.sorted_matrix: Optional[np.ndarray] = None
        self.doc_order: Optional[np.ndarray] = None
        self.metric_order: Optional[np.ndarray] = None
        # ... (task attributes) ...
        self.task: str = "default"
        self.task_config: Optional[Dict[str, Any]] = None
        # Initialize DuckDB connection
        self.duckdb_conn: duckdb.DuckDBPyConnection = self._init_duckdb()
        # Initialize Dynamic Normalizer
        self.normalizer = DynamicNormalizer(self.metric_names)
        logger.info(f"ZeroModel initialized with {len(self.metric_names)} metrics. Default output precision: {self.default_output_precision}.")

    def encode(self, output_precision: Optional[str] = None) -> np.ndarray:
        """
        Encode the processed data into a full visual policy map (VPM).

        The VPM is a structured image where:
        - Position encodes relevance/importance (top-left = most relevant)
        - Color/value encodes metric scores (darker = higher value)
        - Structure encodes task logic (organization based on SQL query)

        This method now supports configurable output precision for resolution independence.

        Args:
            output_precision: Desired output dtype as a string.
                              Overrides `self.default_output_precision` if provided.
                              Supported: 'uint8', 'float16', 'float32', 'float64'.
                              If None, uses `self.default_output_precision`.

        Returns:
            np.ndarray: RGB image array of shape [height, width, 3].
                        Dtype is determined by `output_precision` or `self.default_output_precision`.
                        
        Raises:
            ValueError: If `process` has not been called successfully yet.
        """
        logger.debug(f"Encoding VPM using vectorized operations. Requested output precision: {output_precision}")
        if self.sorted_matrix is None:
            error_msg = "Data not processed yet. Call process() or prepare() first."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # --- Determine Final Output Precision ---
        final_precision = output_precision if output_precision is not None else self.default_output_precision
        valid_precisions = ['uint8', 'uint16', 'float16', 'float32', 'float64']
        if final_precision not in valid_precisions:
             logger.warning(f"Invalid output_precision '{final_precision}'. Must be one of {valid_precisions}. Using default '{self.default_output_precision}'.")
             final_precision = self.default_output_precision
        # Map string to actual numpy dtype
        precision_dtype_map = {
            'uint8': np.uint8,
            'uint16': np.uint16, # <-- CRITICAL: Ensure this line exists
            'float16': np.float16,
            'float32': np.float32, # Default internal working precision for normalized data
            'float64': np.float64,
        }
        target_dtype = precision_dtype_map.get(final_precision) # Use .get for safety
        if target_dtype is None:
            # Handle error if mapping failed unexpectedly
            logger.error(f"Failed to map final_precision '{final_precision}' to a numpy dtype.")
            target_dtype = np.float32 # Fallback
            final_precision = 'float32' # Sync fallback string
        logger.debug(f"Encoding VPM with target output dtype: {target_dtype}")
        # --- End Determine Final Output Precision ---

        n_docs, n_metrics = self.sorted_matrix.shape
        logger.debug(f"Encoding matrix of shape {n_docs}x{n_metrics}")
        # Calculate required width (3 metrics per pixel, ceiling division)
        width = (n_metrics + 2) // 3
        logger.debug(f"Calculated VPM width: {width} pixels")

        # --- Vectorized Encoding (on normalized data) ---
        # 1. The self.sorted_matrix should already be normalized [0.0, 1.0] float64/float32
        #    from the DynamicNormalizer within process/prepare.
        #    Let's ensure it's float32 for internal consistency if it's not already.
        internal_working_dtype = np.float32
        if self.sorted_matrix.dtype != internal_working_dtype:
            logger.debug(f"Casting internal sorted_matrix from {self.sorted_matrix.dtype} to {internal_working_dtype} for encoding.")
            internal_sorted_matrix = self.sorted_matrix.astype(internal_working_dtype)
        else:
            internal_sorted_matrix = self.sorted_matrix
        logger.debug(f"Internal sorted_matrix dtype for encoding: {internal_sorted_matrix.dtype}")

        # 2. Pad the data to make the number of metrics a multiple of 3
        padding_needed = (3 - n_metrics % 3) % 3
        if padding_needed > 0:
            logger.debug(f"Padding sorted_matrix with {padding_needed} zeros.")
            padded_data = np.pad(internal_sorted_matrix, ((0, 0), (0, padding_needed)), mode='constant', constant_values=0.0)
        else:
            padded_data = internal_sorted_matrix
        logger.debug(f"Padded data shape: {padded_data.shape}")

        # 3. Reshape the padded data directly into the image format [n_docs, width, 3]
        try:
            # Reshape from [n_docs, n_metrics_padded] to [n_docs, width, 3]
            img_data = padded_data.reshape(n_docs, width, 3)
            logger.debug(f"Data reshaped to image format: {img_data.shape}")
        except ValueError as e:
            logger.error(f"Error reshaping data for encoding: {e}. Check matrix dimensions.")
            raise ValueError(f"Cannot reshape data of shape {padded_data.shape} to ({n_docs}, {width}, 3). Padding might be incorrect.") from e

        # 4. Convert normalized float values [0.0, 1.0] to the target output dtype
        #    We use the vpm_logic.denormalize_vpm helper for this if available,
        #    or implement the logic directly here for clarity.
        #    Let's assume vpm_logic.denormalize_vpm is available.
        # --- Use vpm_logic.denormalize_vpm if available ---
        try:
            # Try importing the helper function
            from .vpm_logic import denormalize_vpm
            logger.debug("Using vpm_logic.denormalize_vpm for output conversion.")
            img = denormalize_vpm(img_data, output_type=target_dtype)
        except ImportError:
            # Fallback if vpm_logic is not available or denormalize_vpm isn't found
            logger.debug("vpm_logic.denormalize_vpm not found. Using direct conversion.")
            if target_dtype == np.uint8:
                # Convert [0.0, 1.0] float to [0, 255] uint8
                img = np.clip(img_data * 255.0, 0, 255).astype(target_dtype)
            else:
                # For float outputs, clamp to [0.0, 1.0] and cast
                img = np.clip(img_data, 0.0, 1.0).astype(target_dtype)
        # --- End Conversion ---
        logger.debug(f"Final VPM image array shape: {img.shape}, dtype: {img.dtype}")
        # --- End Vectorized Encoding ---
        
        logger.info(f"VPM encoded successfully using vectorization. Shape: {img.shape}, Output Dtype: {img.dtype}")
        return img

    def get_critical_tile(self, tile_size: int = 3, precision: Optional[str] = None) -> bytes:
        """
        Get critical tile for edge devices (top-left section).

        Args:
            tile_size: Size of tile to extract (default 3x3).
            precision: Desired output precision for the tile data as a string.
                        Supported: 'uint8', 'float16', 'float32', 'float64'.
                        If None, uses `self.default_output_precision`.
                        Note: The returned bytes represent the raw data of the tile
                        in the specified precision. Interpretation depends on the receiver.

        Returns:
            bytes: Compact byte representation of the tile.
                   Format: [width][height][x_offset][y_offset][pixel_data...]
                   Pixel data is flattened row-major, channel-interleaved (RGBRGB...).
                   
        Raises:
            ValueError: If `process` has not been called successfully yet.
        """
        logger.debug(f"Extracting critical tile of size {tile_size}x{tile_size}, requested precision: {precision}")
        if self.sorted_matrix is None:
            error_msg = "Data not processed yet. Call process() or prepare() first."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # --- Determine Tile Precision ---
        final_tile_precision = precision if precision is not None else self.default_output_precision
        valid_precisions = ['uint8', 'float16', 'float32', 'float64']
        if final_tile_precision not in valid_precisions:
             logger.warning(f"Invalid tile precision '{final_tile_precision}'. Must be one of {valid_precisions}. Using default '{self.default_output_precision}'.")
             final_tile_precision = self.default_output_precision

        precision_dtype_map = {
            'uint8': np.uint8,
            'float16': np.float16,
            'float32': np.float32,
            'float64': np.float64,
        }
        target_tile_dtype = precision_dtype_map[final_tile_precision]
        logger.debug(f"Critical tile will be extracted and converted to dtype: {target_tile_dtype}")
        # --- End Determine Tile Precision ---

        # --- Existing tile extraction logic (mostly) ---
        n_docs, n_metrics = self.sorted_matrix.shape
        actual_tile_height = min(tile_size, n_docs)
        actual_tile_width_metrics = min(tile_size * 3, n_metrics)
        actual_tile_width_pixels = (actual_tile_width_metrics + 2) // 3
        # Extract the top-left tile data from the sorted_matrix
        tile_data = self.sorted_matrix[:actual_tile_height, :actual_tile_width_metrics]
        logger.debug(f"Extracted raw tile data shape: {tile_data.shape}")

        # --- Convert Tile Data to Requested Precision ---
        # The tile_data is assumed to be the normalized float matrix from sorted_matrix.
        # We need to convert it to the target_tile_dtype.
        try:
            from .vpm_logic import denormalize_vpm, normalize_vpm
            # Ensure tile_data is normalized float first (it should be, but safeguard)
            normalized_tile_data = normalize_vpm(tile_data) # Should be no-op if already normalized
            # Convert to target dtype
            converted_tile_data = denormalize_vpm(normalized_tile_data, output_type=target_tile_dtype)
        except (ImportError, ModuleNotFoundError):
            logger.debug("vpm_logic helpers not found for tile conversion. Using direct method.")
            # Direct conversion logic (duplicate of encode's fallback)
            if target_tile_dtype == np.uint8:
                converted_tile_data = np.clip(tile_data * 255.0, 0, 255).astype(target_tile_dtype)
            else: # float types
                converted_tile_data = np.clip(tile_data, 0.0, 1.0).astype(target_tile_dtype)
        logger.debug(f"Converted tile data to target dtype {target_tile_dtype}. Shape: {converted_tile_data.shape}")
        # --- End Convert Tile Data ---

        # --- Build Tile Bytes ---
        tile_bytes = bytearray()
        # Standard Header (using actual dimensions)
        tile_bytes.append(actual_tile_width_pixels & 0xFF) # Width in pixels
        tile_bytes.append(actual_tile_height & 0xFF)       # Height in docs
        tile_bytes.append(0 & 0xFF)                        # X offset (always 0 for top-left)
        tile_bytes.append(0 & 0xFF)                        # Y offset (always 0 for top-left)
        logger.debug("Appended tile header bytes.")

        # Pixel Data (flattened, 1 value per channel element)
        # Iterate based on the converted tile data shape and dtype
        # Flatten the data in row-major, channel-interleaved order (C-style flatten is default)
        flattened_pixel_data = converted_tile_data.flatten() # Shape becomes (H * W * C,)
        logger.debug(f"Flattened pixel data for tile: {flattened_pixel_data.shape}")

        # --- Pack Pixel Data Bytes ---
        # Handle different dtypes for byte conversion
        if target_tile_dtype == np.uint8:
            # Data is already bytes, just extend the bytearray
            tile_bytes.extend(flattened_pixel_data.tobytes())
        else:
            # For float types, we need to convert each element to bytes
            # Use numpy's tobytes method with native byte order ('<')
            # The receiver must know the dtype to interpret these bytes correctly.
            # This packs the raw bytes of the float values.
            tile_bytes.extend(flattened_pixel_data.tobytes()) # .tobytes() handles endianness (native by default)
        # --- End Pack Pixel Data Bytes ---
                
        result_bytes = bytes(tile_bytes)
        logger.info(f"Critical tile extracted. Size: {len(result_bytes)} bytes. Output precision: {target_tile_dtype}.")
        return result_bytes

    def get_decision(self, context_size: int = 3) -> Tuple[int, float]:
        """
        Get top decision with contextual understanding.
        NOTE: This method operates on the internal sorted_matrix (normalized float).
        It should produce a relevance score between 0.0 and 1.0.
        """
        logger.debug(f"Making decision with context size {context_size}")
        if self.sorted_matrix is None:
            error_msg = "Data not processed yet. Call process() or prepare() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if context_size <= 0:
            error_msg = f"context_size must be positive, got {context_size}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_docs, n_metrics = self.sorted_matrix.shape
        # Determine actual context window size
        actual_context_docs = min(context_size, n_docs)
        actual_context_metrics = min(context_size * 3, n_metrics) # Width in metrics
        logger.debug(f"Actual decision context: {actual_context_docs} docs x {actual_context_metrics} metrics")

        # Get context window (top-left region) - operates on normalized float data
        context = self.sorted_matrix[:actual_context_docs, :actual_context_metrics]
        logger.debug(f"Context data shape for decision: {context.shape}")

        # Calculate contextual relevance (weighted by position) - on normalized data
        weights = np.zeros_like(context, dtype=np.float64) # Use float64 for calculation
        for i in range(context.shape[0]): # Iterate through rows (docs)
            for j in range(context.shape[1]): # Iterate through columns (metrics)
                # j represents metric index, so j/3 gives approximate pixel x-coordinate
                pixel_x_coord = j / 3.0
                distance = np.sqrt(float(i)**2 + pixel_x_coord**2)
                # Example weight function: linear decrease
                weight = max(0.0, 1.0 - distance * 0.3)
                weights[i, j] = weight
        logger.debug("Calculated positional weights for context.")

        # Calculate weighted relevance - on normalized data
        sum_weights = np.sum(weights, dtype=np.float64)
        if sum_weights > 0.0:
            weighted_sum = np.sum(context * weights, dtype=np.float64)
            weighted_relevance = weighted_sum / sum_weights
        else:
            logger.warning("Sum of weights is zero. Assigning relevance score 0.0.")
            weighted_relevance = 0.0
        weighted_relevance = float(weighted_relevance) # Ensure it's a standard float
        logger.debug(f"Calculated weighted relevance score: {weighted_relevance:.4f}")

        # Get top document index from the *original* order
        top_doc_idx_in_original = 0
        if self.doc_order is not None and len(self.doc_order) > 0:
            top_doc_idx_in_original = int(self.doc_order[0])
        else:
            logger.warning("doc_order is not available or empty. Defaulting top document index to 0.")
        
        logger.info(f"Decision made: Document index {top_doc_idx_in_original}, Relevance {weighted_relevance:.4f}")
        # Return index (int) and relevance (float, 0.0-1.0)
        return (top_doc_idx_in_original, weighted_relevance)


    def _init_duckdb(self) -> duckdb.DuckDBPyConnection:
        """Initialize DuckDB connection with virtual index table structure."""
        logger.debug("Initializing DuckDB connection for ZeroModel.")
        conn = duckdb.connect(database=':memory:')
        # Create virtual index table schema based on initial metric names
        columns = ", ".join([f'"{col}" FLOAT' for col in self.metric_names])
        conn.execute(f"CREATE TABLE virtual_index (row_id INTEGER, {columns})")
        # The analysis logic needs to work with real data.
        logger.debug("DuckDB virtual_index table schema created.")
        return conn

    def normalize(self, score_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize the score matrix using the DynamicNormalizer.

        Args:
            score_matrix: 2D NumPy array of shape [documents x metrics].

        Returns:
            np.ndarray: Normalized score matrix.
        """
        logger.debug(f"Normalizing score matrix with shape {score_matrix.shape}")
        return self.normalizer.normalize(score_matrix)

    def prepare(self, score_matrix: np.ndarray, sql_query: str, nonlinearity_hint: Optional[str] = None) -> None:
        """
        Public entry point to prepare the ZeroModel with data and task in one step.
        Optionally applies specific non-linear feature transformations based on a hint.

        Args:
            score_matrix: A 2D NumPy array of shape [documents x metrics].
            sql_query: A string containing the SQL query for sorting.
            nonlinearity_hint: An optional string hint specifying the type of non-linearity
                            to apply. Supported hints:
                            - None (default): No additional features.
                            - 'auto': Apply a standard set of common non-linear features 
                                        (products, differences, squares).
                            - 'xor': Apply features specifically helpful for XOR-like problems
                                        (product of first two metrics, absolute difference).
                            - 'radial': Apply features helpful for radial/distance-based problems
                                        (distance from center, angle).
                            Example: prepare(data, "SELECT ...", nonlinearity_hint='xor')
        """
        logger.info(f"Preparing ZeroModel with data shape {score_matrix.shape}, query: '{sql_query}', nonlinearity_hint: {nonlinearity_hint}")
        original_metric_names = self.metric_names

        # --- Input Validation ---
        if score_matrix is None:
            error_msg = "score_matrix cannot be None."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.ndim != 2:
            error_msg = f"score_matrix must be 2D, got shape {score_matrix.shape}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.shape[1] != len(original_metric_names):
            error_msg = (f"Number of columns in score_matrix ({score_matrix.shape[1]}) must match "
                        f"the number of metrics initialized ({len(original_metric_names)}).")
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not sql_query or not isinstance(sql_query, str):
            error_msg = "sql_query must be a non-empty string."
            logger.error(error_msg)
            raise ValueError(error_msg)
        # --- End Input Validation ---

        # --- 1. Dynamic Normalization ---
        try:
            logger.debug("Updating DynamicNormalizer with new data ranges.")
            self.normalizer.update(score_matrix)
            logger.debug("Normalizer updated. Applying normalization to the data.")
            normalized_data = self.normalizer.normalize(score_matrix)
            logger.debug("Data normalized successfully.")
        except Exception as e:
            logger.error(f"Failed during normalization step: {e}")
            raise Exception(f"Error during data normalization: {e}") from e
        # --- End Dynamic Normalization ---

        # --- 2. Hint-Based Feature Engineering ---
        processed_data = normalized_data
        effective_metric_names = original_metric_names
        if nonlinearity_hint is not None:
            try:
                logger.debug(f"Applying non-linear feature engineering based on hint: '{nonlinearity_hint}'")
                
                # --- Feature Engineering Logic Based on Hint ---
                engineered_features = []
                engineered_names = []
                n_metrics = normalized_data.shape[1]

                hint_lower = nonlinearity_hint.lower() if nonlinearity_hint else ''

                if hint_lower in ['xor', 'radial']:
                    
                    if hint_lower in ['xor'] and n_metrics >= 2:
                        # XOR-like features: Product and Absolute Difference of first two metrics
                        m1, m2 = normalized_data[:, 0], normalized_data[:, 1]
                        engineered_features.append(m1 * m2)
                        engineered_names.append(f"hint_product_{original_metric_names[0]}_{original_metric_names[1]}")
                        engineered_features.append(np.abs(m1 - m2))
                        engineered_names.append(f"hint_abs_diff_{original_metric_names[0]}_{original_metric_names[1]}")
                        logger.debug("Added XOR-like features (product, abs_diff).")

                    if hint_lower in ['radial'] and n_metrics >= 2:
                        # Radial features: Distance and Angle (assuming first two metrics are X, Y)
                        x, y = normalized_data[:, 0], normalized_data[:, 1]
                        # Assume center is (0.5, 0.5) as common in [0,1] normalized data
                        center_x, center_y = 0.5, 0.5 
                        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        angle = np.arctan2(y - center_y, x - center_x)
                        engineered_features.append(distance)
                        engineered_names.append(f"hint_radial_distance")
                        engineered_features.append(angle)
                        engineered_names.append(f"hint_radial_angle")
                        logger.debug("Added radial features (distance, angle).")

                        # Inside zeromodel/core.py -> ZeroModel.prepare()
                        # Locate the section: if hint_lower in ['auto', 'xor', 'radial']:
                        # Then, inside that, find: if hint_lower == 'auto':

                if hint_lower == 'auto':
                    # --- Robust 'auto' feature generation based on INPUT for this prepare() call ---
                    # Use the original metric names and data shape passed to THIS prepare() call.
                    # This prevents issues if self.metric_names was modified by a previous prepare() call
                    # on the same instance.
                    num_original_metrics = len(original_metric_names) # This should be 3 for the failing test
                    logger.debug(f"Generating 'auto' features for {num_original_metrics} original input metrics.")

                    # 1. Pairwise Products: All unique pairs among the first min(3, N) metrics
                    n_prod_metrics = min(3, num_original_metrics)
                    product_count = 0
                    # Generate unique pairs (i, j) where i < j
                    for i in range(n_prod_metrics):
                        # j should start from i+1 and be less than n_prod_metrics
                        for j in range(i + 1, n_prod_metrics):
                            # Safety check to ensure we are accessing valid columns in the input data
                            # (though this should always be true given how n_prod_metrics is calculated)
                            if i < normalized_data.shape[1] and j < normalized_data.shape[1]:
                                engineered_features.append(normalized_data[:, i] * normalized_data[:, j])
                                engineered_names.append(f"auto_product_{original_metric_names[i]}_{original_metric_names[j]}")
                                product_count += 1
                            else:
                                logger.warning(f"Index out of bounds for product feature: i={i}, j={j}, data_cols={normalized_data.shape[1]}")
                    logger.debug(f"Added {product_count} pairwise product features.")

                    # 2. Squares: First min(2, N) metrics
                    n_square_metrics = min(2, num_original_metrics)
                    square_count = 0
                    for i in range(n_square_metrics):
                        # Safety check
                        if i < normalized_data.shape[1]:
                            engineered_features.append(normalized_data[:, i] ** 2)
                            engineered_names.append(f"auto_square_{original_metric_names[i]}")
                            square_count += 1
                        else:
                            logger.warning(f"Index out of bounds for square feature: i={i}, data_cols={normalized_data.shape[1]}")
                    logger.debug(f"Added {square_count} square features.")
                    
                    total_added = product_count + square_count
                    logger.info(f"Auto hint: Successfully added {total_added} features for {num_original_metrics} input metrics. Expected: 5 for 3 inputs.")
                            # --- End Robust 'auto' feature generation ---
                        # --- End replacement block ---
                else:
                    if hint_lower: # Only warn if a hint was actually provided
                        logger.warning(f"Unknown nonlinearity_hint '{nonlinearity_hint}'. No features added.")

                # --- End Feature Engineering Logic Based on Hint ---

                if engineered_features:
                    processed_data = np.column_stack([normalized_data] + engineered_features)
                    effective_metric_names = original_metric_names + engineered_names
                    logger.info(f"Added {len(engineered_names)} non-linear features based on hint '{nonlinearity_hint}'. New data shape: {processed_data.shape}")
                    logger.debug(f"Effective metric names are now: {effective_metric_names}") 
                else:
                    logger.debug("No new features added based on hint (e.g., insufficient metrics).")
                    
            except Exception as e:
                logger.error(f"Hint-based feature engineering failed: {e}. Using normalized data only.")
                processed_data = normalized_data
                effective_metric_names = original_metric_names
        else:
            logger.debug("No nonlinearity_hint provided. Using normalized data without additional features.")
        # --- End Hint-Based Feature Engineering ---

        # --- 3. Ensure DuckDB table schema matches *effective* metric names ---
        try:
            current_schema_cursor = self.duckdb_conn.execute("PRAGMA table_info(virtual_index)")
            current_columns_info = current_schema_cursor.fetchall()
            expected_columns = ["row_id"] + list(effective_metric_names)
            current_column_names = [col_info[1] for col_info in current_columns_info]

            if current_column_names != expected_columns:
                logger.debug(f"DuckDB schema mismatch. Expected: {expected_columns}, Found: {current_column_names}. Recreating virtual_index table.")
                self.duckdb_conn.execute("DROP TABLE IF EXISTS virtual_index")
                columns_def = ", ".join([f'"{col}" FLOAT' for col in effective_metric_names])
                self.duckdb_conn.execute(f"CREATE TABLE virtual_index (row_id INTEGER, {columns_def})")
                logger.debug(f"Recreated virtual_index table with schema: {expected_columns}")
            else:
                logger.debug("DuckDB virtual_index schema matches effective metric names.")
        except Exception as e:
            logger.error(f"Error checking or recreating DuckDB schema: {e}")
            raise Exception(f"Failed to ensure DuckDB schema: {e}") from e
        # --- End Schema Check ---

        # --- 4. Load PROCESSED (potentially engineered) data into DuckDB ---
        logger.debug("Loading PROCESSED score_matrix data into DuckDB virtual_index table.")
        try:
            self.duckdb_conn.execute("DELETE FROM virtual_index") # Clear existing data

            metric_columns_str = ", ".join([f'"{name}"' for name in effective_metric_names])
            placeholders_str = ", ".join(["?"] * len(effective_metric_names))
            insert_sql = f"INSERT INTO virtual_index (row_id, {metric_columns_str}) VALUES (?, {placeholders_str})"

            # Insert the PROCESSED data
            for row_id, row_data in enumerate(processed_data):
                self.duckdb_conn.execute(insert_sql, [row_id] + row_data.tolist())

            logger.debug(f"Successfully loaded {processed_data.shape[0]} processed rows into DuckDB.")
        except Exception as e:
            logger.error(f"Failed to load PROCESSED data into DuckDB: {e}")
            raise Exception(f"Error loading data into DuckDB: {e}") from e
        # --- End Data Loading ---

        # --- Update Instance Metric Names ---
        # Crucially, update the instance's metric_names so get_metadata() and other
        # parts of the class that rely on self.metric_names are consistent with the
        # data that was actually processed and sorted.
        # Store the original names in case they are needed
        self._original_metric_names = self.metric_names
        self.metric_names = effective_metric_names
        logger.debug(f"Updated instance metric_names to reflect processed features: {len(self.metric_names)} metrics.")
        # --- End Update Instance Metric Names ---

        # --- 5. Set SQL Task ---
        logger.debug("Setting SQL task.")
        try:
            # The SQL task can now reference the original or the new engineered metrics
            self._set_sql_task(sql_query)
            logger.debug("SQL task set successfully.")
        except Exception as e:
            logger.error(f"Failed to set SQL task: {e}")
            raise Exception(f"Error setting SQL task: {e}") from e
        # --- End Set Task ---

        # --- 6. Apply Spatial Organization ---
        logger.debug("Applying spatial organization based on SQL analysis.")
        try:
            if self.task_config and "analysis" in self.task_config:
                analysis = self.task_config["analysis"]
                # Apply organization using the PROCESSED data
                self.sorted_matrix, self.metric_order, self.doc_order = self._apply_sql_organization(
                    processed_data, # Pass the processed matrix that was loaded into DuckDB
                    analysis
                )
                logger.debug("Spatial organization applied successfully.")
            else:
                error_msg = "Task configuration or analysis missing after set_sql_task."
                logger.error(error_msg)
                raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Failed to apply spatial organization: {e}")
            raise Exception(f"Error applying spatial organization: {e}") from e
        # --- End Apply Organization ---
            
        logger.info("ZeroModel preparation complete. Ready for encode/get_decision/etc.")

    def _analyze_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Analyze SQL query using DuckDB to determine sorting orders.

        This determines:
        1. Metric Order: The sequence of columns based on ORDER BY.
        2. Primary Sort Metric: The first metric used for row sorting.
        """
        logger.debug(f"Analyzing SQL query for spatial organization: {sql_query}")

        # --- Metric Order Analysis (using a robust DuckDB method) ---
        # Strategy: Insert test data into virtual_index such that running the
        # user's query reveals the column order.
        # We'll insert N rows, where N = number of metrics.
        # Row i will have a high value (e.g., 1.0) in metric column i, and low/zero elsewhere.
        # When the query (e.g., ORDER BY metric1 DESC, metric2 ASC) runs,
        # the order of the returned row_ids will tell us the importance/sort order of columns.

        metric_count = len(self.metric_names)
        if metric_count == 0:
            logger.warning("No metrics provided for analysis. Using default order.")
            metric_order = []
        else:
            try:
                # Clear any existing data in virtual_index for clean analysis
                self.duckdb_conn.execute("DELETE FROM virtual_index")

                # Insert test data: Row i highlights metric i
                # In prepare
                data_for_db = [(i, *row) for i, row in enumerate(metric_count)]
                self.duckdb_conn.executemany(
                    "INSERT INTO virtual_index VALUES (?, " + ", ".join(["?"]*len(self.metric_names)) + ")",
                    data_for_db
                )

                # Execute the user's query against this test data
                # We only care about the order of rows returned, which reflects the ORDER BY logic
                result_cursor = self.duckdb_conn.execute(sql_query)
                result_rows = result_cursor.fetchall()

                # Extract the row_id sequence from the results (first column)
                # This sequence indicates the order in which metrics (represented by rows)
                # should be prioritized according to the ORDER BY clause.
                returned_row_ids = [row[0] for row in result_rows]

                # The order of row_ids IS the metric order.
                metric_order = returned_row_ids

                logger.debug(
                    f"Metric order determined by DuckDB analysis: {metric_order}"
                )

                # Cleanup test data
                self.duckdb_conn.execute("DELETE FROM virtual_index")

            except Exception as e:
                logger.error(f"Error during DuckDB-based metric order analysis: {e}")
                # Fallback to default order (original sequence) if analysis fails
                metric_order = list(range(metric_count))
                logger.warning(f"Falling back to default metric order: {metric_order}")

        # --- Primary Sort Metric Analysis (Simpler) ---
        # Identify the first metric in the ORDER BY clause for document sorting.
        # A simple regex can often work for this specific, limited purpose.
        # It's less brittle than full parsing because we only need the FIRST metric name.
        primary_sort_metric_index = None
        primary_sort_metric_name = None
        order_by_match = re.search(r"ORDER\s+BY\s+(\w+)", sql_query, re.IGNORECASE)
        if order_by_match:
            first_metric_in_order_by = order_by_match.group(1)
            try:
                primary_sort_metric_index = self.metric_names.index(
                    first_metric_in_order_by
                )
                primary_sort_metric_name = first_metric_in_order_by
                logger.debug(
                    f"Primary sort metric identified: '{primary_sort_metric_name}' (index {primary_sort_metric_index})"
                )
            except ValueError:
                logger.warning(
                    f"Metric '{first_metric_in_order_by}' from ORDER BY not found in metric_names. Document sorting might be incorrect."
                )
        else:
            logger.info(
                "No ORDER BY clause found or first metric could not be parsed. Using first metric for document sorting."
            )
            if metric_count > 0:
                primary_sort_metric_index = 0
                primary_sort_metric_name = self.metric_names[0]

        analysis_result = {
            "metric_order": metric_order,
            "primary_sort_metric_index": primary_sort_metric_index,
            "primary_sort_metric_name": primary_sort_metric_name,
            "original_query": sql_query,
        }
        logger.info(f"SQL query analysis complete: {analysis_result}")
        return analysis_result

    # 3. Revise _apply_sql_organization to use the analysis results correctly
    def _apply_sql_organization(
        self, data: np.ndarray, analysis: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply spatial organization based on SQL query analysis results from DuckDB.

        Args:
            data: The input score matrix (normalized).
            analysis: The result dictionary from `_analyze_query`.

        Returns:
            Tuple of (sorted_matrix, metric_order, doc_order)
        """
        logger.debug(f"Applying SQL-based organization. Data shape: {data.shape}")
        metric_count = data.shape[1]

        # --- 1. Apply Metric Ordering ---
        raw_metric_order = analysis.get("metric_order", [])
        # Ensure indices are valid
        valid_metric_order = [
            idx for idx in raw_metric_order if 0 <= idx < metric_count
        ]
        # Handle potential mismatch (e.g., analysis returned fewer indices)
        if len(valid_metric_order) != metric_count:
            logger.warning(
                f"Analysis metric order ({valid_metric_order}) length mismatch. Appending remaining metrics."
            )
            remaining_indices = [
                i for i in range(metric_count) if i not in valid_metric_order
            ]
            valid_metric_order.extend(remaining_indices)

        if not valid_metric_order:
            logger.info("No valid metric order from analysis. Using default order.")
            valid_metric_order = list(range(metric_count))

        logger.debug(f"Final validated metric order: {valid_metric_order}")
        # Reorder columns (metrics)
        sorted_matrix_by_metrics = data[:, valid_metric_order]

        # --- 2. Apply Document Ordering ---
        primary_sort_idx = analysis.get("primary_sort_metric_index")
        doc_order = np.arange(data.shape[0])  # Default order

        if primary_sort_idx is not None and 0 <= primary_sort_idx < metric_count:
            # Map the primary sort index to the new column order
            try:
                # Find the new column index corresponding to the primary sort metric
                new_column_index_for_primary_sort = valid_metric_order.index(
                    primary_sort_idx
                )
                logger.debug(
                    f"Sorting documents by metric index {primary_sort_idx} which is now column {new_column_index_for_primary_sort}"
                )
                # Sort rows (documents) by the value in this column (descending)
                doc_order = np.argsort(
                    sorted_matrix_by_metrics[:, new_column_index_for_primary_sort]
                )[::-1]
                logger.debug(
                    f"Document order indices calculated: {doc_order[:10]}..."
                )  # Log first 10
            except ValueError:
                logger.error(
                    f"Failed to find primary sort metric index {primary_sort_idx} in reordered metric list. Using default doc order."
                )
        else:
            logger.warning(
                "Primary sort metric index invalid or not found. Using default document order."
            )

        # Reorder rows (documents)
        final_sorted_matrix = sorted_matrix_by_metrics[doc_order, :]

        logger.info(
            f"SQL-based organization applied. Final matrix shape: {final_sorted_matrix.shape}"
        )
        return final_sorted_matrix, np.array(valid_metric_order), doc_order

    def _set_sql_task(self, sql_query: str):
        """
        Set task using SQL query for spatial organization.

        Args:
            sql_query: SQL query defining the task (e.g., sorting criteria).
                       Example: "SELECT * FROM virtual_index ORDER BY uncertainty DESC, size ASC"

        Raises:
            ValueError: If the SQL query is invalid or cannot be analyzed.
        """
        logger.info(f"Setting SQL task: {sql_query}")
        # Analyze query to determine spatial organization
        analysis = self._analyze_query(sql_query)
        # Store task configuration
        self.task = "sql_task"
        self.task_config = {"sql_query": sql_query, "analysis": analysis}
        logger.debug(f"SQL task set. Analysis: {analysis}")

    def _analyze_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Analyze SQL query by running it against DuckDB virtual_index table *with data*.
        This determines the actual sorting order of documents based on the query.
        """
        logger.debug(f"Analyzing SQL query with data: {sql_query}")
        try:
            # Execute query against virtual index table which now contains the actual data
            # We need to get the row_id to know the order
            # Modify the query to select row_id first if it's a SELECT *
            # A more robust way is to explicitly select row_id
            if sql_query.strip().upper().startswith("SELECT *"):
                # Simple replacement - assumes no complex SELECT clauses
                modified_query = sql_query.replace("SELECT *", "SELECT row_id", 1)
            else:
                # If not SELECT *, assume row_id is accessible or construct a way to get it
                # This is trickier. Let's assume the user's query implicitly sorts the data
                # and we just run it and hope the order of results corresponds to row_id order.
                # Or, we could wrap it: SELECT row_id FROM (user_query)
                modified_query = f"SELECT row_id FROM ({sql_query}) AS user_sorted_view"
            
            logger.debug(f"Executing modified query for analysis: {modified_query}")
            result_cursor = self.duckdb_conn.execute(modified_query)
            result_rows = result_cursor.fetchall()
            
            # Extract the row_id sequence from the results
            # This sequence indicates the order in which documents should be arranged.
            doc_order_from_query = [row[0] for row in result_rows] 
            
            logger.debug(f"Document order determined by DuckDB query execution: {doc_order_from_query[:10]}...") # Log first 10
            
            # For metric ordering, if the original logic required it from the dummy row,
            # and we are not changing column order based on the query (which is complex),
            # we can default to the original metric order or derive it simply.
            # Let's default to original order for now, or try to parse the first ORDER BY metric.
            import re
            metric_order_indices = list(range(len(self.metric_names))) # Default
            primary_sort_metric_name = None
            primary_sort_metric_index = None
            order_by_match = re.search(r"ORDER\s+BY\s+(\w+)", sql_query, re.IGNORECASE)
            if order_by_match:
                first_metric_in_order_by = order_by_match.group(1)
                try:
                    primary_sort_metric_index = self.metric_names.index(first_metric_in_order_by)
                    primary_sort_metric_name = first_metric_in_order_by
                    logger.debug(f"Primary sort metric identified from query: '{primary_sort_metric_name}' (index {primary_sort_metric_index})")
                    # If you *did* want to reorder columns based on this, you'd set metric_order_indices
                    # But let's keep it simple and assume column order is fixed by input.
                except ValueError:
                    logger.warning(f"Metric '{first_metric_in_order_by}' from ORDER BY not found in metric_names.")

            analysis_result = {
                "doc_order": doc_order_from_query,
                "metric_order": metric_order_indices, # Keep original column order for now
                "primary_sort_metric_index": primary_sort_metric_index,
                "primary_sort_metric_name": primary_sort_metric_name,
                "original_query": sql_query
            }
            logger.info(f"SQL query analysis (with data) complete.")
            return analysis_result
        except Exception as e:
            logger.error(f"Error during DuckDB-based query analysis: {e}")
            # Re-raise to be caught by process
            raise ValueError(f"Invalid SQL query execution: {str(e)}") from e


    def _apply_sql_organization(self,
                            data: np.ndarray,
                            analysis: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply spatial organization based on SQL query analysis results from DuckDB (with data).
        
        Args:
            The original input score matrix.
            analysis: The result dictionary from `_analyze_query` (now based on real data).
            
        Returns:
            Tuple of (sorted_matrix, metric_order, doc_order)
        """
        logger.debug(f"Applying SQL-based organization (with data). Data shape: {data.shape}")
        
        # --- 1. Apply Document Ordering ---
        doc_order_list = analysis.get("doc_order", [])
        # Validate doc_order indices
        num_docs = data.shape[0]
        valid_doc_order = [idx for idx in doc_order_list if 0 <= idx < num_docs]
        # Handle potential mismatch (e.g., analysis returned fewer/more indices)
        # DuckDB should return all row_ids if the query selects all, sorted.
        # If it returns a subset, that's the subset we want.
        # If it returns duplicates or extras, we filter.
        if not valid_doc_order:
            logger.warning("Document order from analysis is empty or invalid. Using default order.")
            valid_doc_order = list(range(num_docs))
            
        logger.debug(f"Final validated document order (first 10): {valid_doc_order[:10]}...")
        # Reorder rows (documents)
        sorted_matrix_by_docs = data[valid_doc_order, :]

        # --- 2. Apply Metric Ordering (if needed) ---
        # For now, assuming metric/column order is fixed by the input and analysis
        # just provides the doc order. Metric reordering based on complex SQL on columns
        # is complex. Let's keep columns in original order unless analysis explicitly provides a different order.
        raw_metric_order = analysis.get("metric_order", list(range(data.shape[1])))
        metric_count = data.shape[1]
        valid_metric_order = [idx for idx in raw_metric_order if 0 <= idx < metric_count]
        if len(valid_metric_order) != metric_count:
            logger.warning(f"Analysis metric order length mismatch. Appending remaining metrics.")
            remaining_indices = [i for i in range(metric_count) if i not in valid_metric_order]
            valid_metric_order.extend(remaining_indices)
        if not valid_metric_order:
            logger.info("No valid metric order from analysis. Using default order.")
            valid_metric_order = list(range(metric_count))
            
        logger.debug(f"Final validated metric order: {valid_metric_order}")
        # Reorder columns (metrics) - apply to the already row-sorted matrix
        final_sorted_matrix = sorted_matrix_by_docs[:, valid_metric_order]
        
        logger.info(f"SQL-based organization (with data) applied. Final matrix shape: {final_sorted_matrix.shape}")
        return final_sorted_matrix, np.array(valid_metric_order), np.array(valid_doc_order)


    # Inside the get_metadata method
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the current encoding state."""
        logger.debug("Retrieving metadata.")
        metadata = {
            "task": self.task,
            "task_config": self.task_config,
            "metric_order": self.metric_order.tolist() if self.metric_order is not None else [],
            "doc_order": self.doc_order.tolist() if self.doc_order is not None else [],
            # --- Use the potentially updated self.metric_names ---
            "metric_names": self.metric_names, # This should now be effective_metric_names after prepare()
            "precision": self.precision,
        }
        logger.debug(f"Metadata retrieved: {metadata}")
        return metadata

    def _apply_default_organization(self, score_matrix: np.ndarray):
        """
        Apply default ordering to the score matrix:
        - Documents in original order
        - Metrics in original order
        """
        logger.info("Applying default organization: preserving original order.")

        self.sorted_matrix = score_matrix.copy()
        self.metric_order = np.arange(score_matrix.shape[1])  # [0, 1, 2, ...]
        self.doc_order = np.arange(score_matrix.shape[0])     # [0, 1, 2, ...]

        # Optionally update metadata if needed
        self.metadata = {
            "task": "default",
            "precision": self.precision,
            "metric_names": self.metric_names,
            "metric_order": self.metric_order.tolist(),
            "doc_order": self.doc_order.tolist()
        }


# --- Example usage or test code (optional, remove or comment out for library module) ---
# if __name__ == "__main__":
#     # This would typically be in a separate test or example script
#     pass
