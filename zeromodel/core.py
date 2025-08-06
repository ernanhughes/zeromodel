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

    def __init__(self, metric_names: List[str], precision: int = 8):
        """
        Initialize ZeroModel encoder with DuckDB SQL processing.

        Args:
            metric_names: Names of all metrics being tracked.
            precision: Bit precision for encoding (4-16).

        Raises:
            ValueError: If metric_names is empty or precision is invalid.
        """
        logger.debug(
            f"Initializing ZeroModel with metrics: {metric_names}, precision: {precision}"
        )
        if not metric_names:
            error_msg = "metric_names list cannot be empty."
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.metric_names = list(metric_names)  # Ensure it's a list
        self.precision = max(4, min(16, precision))
        self.sorted_matrix: Optional[np.ndarray] = None
        self.doc_order: Optional[np.ndarray] = None
        self.metric_order: Optional[np.ndarray] = None
        self.task: str = "default"
        self.task_config: Optional[Dict[str, Any]] = None
        # Initialize DuckDB connection
        self.duckdb_conn: duckdb.DuckDBPyConnection = self._init_duckdb()
        self.normalizer = DynamicNormalizer(metric_names)
        logger.info(f"ZeroModel initialized with {len(self.metric_names)} metrics.")

    def _init_duckdb(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(database=':memory:')
        columns = ", ".join([f'"{col}" FLOAT' for col in self.metric_names])
        conn.execute(f"CREATE TABLE virtual_index (row_id INTEGER, {columns})")
        # Insert a single row with index values (0, 1, 2, ...)
        values = [0] + list(range(len(self.metric_names))) # For 4 metrics: [0, 0, 1, 2, 3]
        placeholders = ", ".join(["?"] * (len(self.metric_names) + 1))
        conn.execute(f"INSERT INTO virtual_index VALUES ({placeholders})", values) # <-- This line should insert the row
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

    def prepare(self, score_matrix: np.ndarray, sql_query: str) -> None:
        """
        Public entry point to prepare the ZeroModel with data and task in one step.

        This function handles updating the normalizer with the new data's range,
        normalizing the data, loading it into DuckDB, setting the SQL task,
        analyzing the query based on the loaded data, and applying the spatial
        organization. It uses the metric_names provided during ZeroModel initialization.

        Args:
            score_matrix: A 2D NumPy array of shape [documents x metrics]
                        containing the raw score data. It will be normalized in-place.
                        The number of columns (metrics) must match the length of
                        the metric_names list provided during ZeroModel initialization.
            sql_query: A string containing the SQL query. This query MUST operate
                    on a table named 'virtual_index' and should reference the
                    column names provided in self.metric_names (set during __init__).
                    Example: "SELECT * FROM virtual_index ORDER BY metric_a DESC"

        Raises:
            ValueError: If inputs are invalid (None, wrong shapes, mismatched dimensions).
            Exception: If there's an error during any preparation step.
        """
        logger.info(f"Preparing ZeroModel with data shape {score_matrix.shape} and query: '{sql_query}'")
        metric_names = self.metric_names # Use the instance's metric names

        # --- Input Validation ---
        if score_matrix is None:
            error_msg = "score_matrix cannot be None."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.ndim != 2:
            error_msg = f"score_matrix must be 2D, got shape {score_matrix.shape}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Check if the number of columns in score_matrix matches the initialized metric count
        if score_matrix.shape[1] != len(metric_names):
            error_msg = (f"Number of columns in score_matrix ({score_matrix.shape[1]}) must match "
                        f"the number of metrics initialized ({len(metric_names)}).")
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
            # Update the normalizer's internal min/max based on this batch
            self.normalizer.update(score_matrix)
            logger.debug("Normalizer updated. Applying normalization to the data.")
            # Normalize the score_matrix using the (potentially updated) min/max values
            score_matrix = self.normalizer.normalize(score_matrix)
            logger.debug("Data normalized successfully.")
        except Exception as e:
            logger.error(f"Failed during normalization step: {e}")
            raise Exception(f"Error during data normalization: {e}") from e
        # --- End Dynamic Normalization ---

        # --- 2. Ensure DuckDB table schema matches self.metric_names ---
        # Check if the DuckDB table schema matches the expected metric names.
        # If not, recreate the table. This handles potential schema drift or initial setup.
        try:
            current_schema_cursor = self.duckdb_conn.execute("PRAGMA table_info(virtual_index)")
            current_columns_info = current_schema_cursor.fetchall()
            # Expected columns based on instance metric_names: row_id, metric1, metric2, ...
            expected_columns = ["row_id"] + list(metric_names)
            current_column_names = [col_info[1] for col_info in current_columns_info] # col_info[1] is name

            if current_column_names != expected_columns:
                logger.debug(f"DuckDB schema mismatch. Expected: {expected_columns}, Found: {current_column_names}. Recreating virtual_index table.")
                self.duckdb_conn.execute("DROP TABLE IF EXISTS virtual_index")
                columns_def = ", ".join([f'"{col}" FLOAT' for col in metric_names])
                self.duckdb_conn.execute(f"CREATE TABLE virtual_index (row_id INTEGER, {columns_def})")
                logger.debug(f"Recreated virtual_index table with schema: {expected_columns}")
            else:
                logger.debug("DuckDB virtual_index schema matches self.metric_names.")
        except Exception as e:
            logger.error(f"Error checking or recreating DuckDB schema: {e}")
            raise Exception(f"Failed to ensure DuckDB schema: {e}") from e
        # --- End Schema Check ---

        # --- 3. Load NORMALIZED data into DuckDB ---
        logger.debug("Loading NORMALIZED score_matrix data into DuckDB virtual_index table.")
        try:
            # Clear any existing data in the table to ensure a clean state for this preparation
            self.duckdb_conn.execute("DELETE FROM virtual_index")

            # Prepare the SQL INSERT statement using the instance's metric names
            metric_columns_str = ", ".join([f'"{name}"' for name in metric_names])
            placeholders_str = ", ".join(["?"] * len(metric_names))
            insert_sql = f"INSERT INTO virtual_index (row_id, {metric_columns_str}) VALUES (?, {placeholders_str})"

            # Insert the NORMALIZED data row by row
            # Consider using executemany for potentially better performance on large datasets
            for row_id, row_data in enumerate(score_matrix):
                self.duckdb_conn.execute(insert_sql, [row_id] + row_data.tolist())

            logger.debug(f"Successfully loaded {score_matrix.shape[0]} normalized rows into DuckDB.")
        except Exception as e:
            logger.error(f"Failed to load NORMALIZED data into DuckDB: {e}")
            raise Exception(f"Error loading data into DuckDB: {e}") from e
        # --- End Data Loading ---

        # --- 4. Set SQL Task ---
        logger.debug("Setting SQL task.")
        try:
            # This will call _analyze_query.
            # Because we just loaded the data, the analysis should work correctly on the
            # loaded, normalized data, determining the correct doc_order and metric_order.
            self._set_sql_task(sql_query)
            logger.debug("SQL task set successfully.")
        except Exception as e:
            logger.error(f"Failed to set SQL task: {e}")
            raise Exception(f"Error setting SQL task: {e}") from e
        # --- End Set Task ---

        # --- 5. Apply Spatial Organization ---
        logger.debug("Applying spatial organization based on SQL analysis.")
        try:
            # After set_sql_task, self.task_config should contain the analysis results.
            if self.task_config and "analysis" in self.task_config:
                analysis = self.task_config["analysis"]
                # Apply the organization logic (sorting documents/metrics) using the
                # analysis results and the NORMALIZED score_matrix.
                # This populates self.sorted_matrix, self.doc_order, self.metric_order.
                self.sorted_matrix, self.metric_order, self.doc_order = self._apply_sql_organization(
                    score_matrix, # Pass the normalized matrix that was loaded into DuckDB
                    analysis
                )
                logger.debug("Spatial organization applied successfully.")
            else:
                error_msg = "Task configuration or analysis missing after set_sql_task. This indicates a problem with the task setup or analysis phase."
                logger.error(error_msg)
                raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Failed to apply spatial organization: {e}")
            raise Exception(f"Error applying spatial organization: {e}") from e
        # --- End Apply Organization ---

        logger.info("ZeroModel preparation complete. Ready for encode/get_decision/get_critical_tile/etc.")


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

    @DeprecationWarning
    def set_sql_task(self, sql_query: str):
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


    @DeprecationWarning
    def process(self, score_matrix: np.ndarray) -> None:
        """
        Process a score matrix to prepare for encoding.
        This involves loading the data into DuckDB and applying SQL-based organization.

        Args:
            score_matrix: 2D array of shape [documents × metrics]. Should be normalized.
        """
        logger.info(f"Processing score matrix of shape: {score_matrix.shape}")
        if score_matrix is None:
            error_msg = "score_matrix cannot be None."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.ndim != 2:
            error_msg = f"score_matrix must be 2D, got shape {score_matrix.shape}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.shape[1] != len(self.metric_names):
            error_msg = (f"score_matrix column count ({score_matrix.shape[1]}) "
                        f"must match metric_names count ({len(self.metric_names)}).")
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.size == 0:
            logger.warning("Received empty score_matrix. Result will be empty.")
            self.sorted_matrix = np.empty((0, 0))
            self.metric_order = np.array([], dtype=int)
            self.doc_order = np.array([], dtype=int)
            return

        # --- Load data into DuckDB ---
        logger.debug("Loading score_matrix data into DuckDB virtual_index table.")
        try:
            # 1. Clear any existing data (including the initial test row)
            self.duckdb_conn.execute("DELETE FROM virtual_index")
            
            # 2. Prepare INSERT statement
            metric_columns = ", ".join([f'"{name}"' for name in self.metric_names])
            placeholders = ", ".join(["?"] * len(self.metric_names))
            insert_sql = f"INSERT INTO virtual_index (row_id, {metric_columns}) VALUES (?, {placeholders})"
            
            # 3. Insert data row by row
            # Using executemany might be more efficient for large datasets
            # data_for_db = [(i, *row) for i, row in enumerate(score_matrix)]
            # self.duckdb_conn.executemany(insert_sql, data_for_db)
            # For clarity and potential logging, inserting one by one or in small batches might be better initially
            for row_id, row_data in enumerate(score_matrix):
                # Ensure row_data is a list or tuple for DuckDB
                self.duckdb_conn.execute(insert_sql, [row_id] + row_data.tolist())
            
            logger.debug(f"Loaded {score_matrix.shape[0]} rows into DuckDB.")
        except Exception as e:
            logger.error(f"Failed to load data into DuckDB: {e}")
            raise ValueError(f"Error loading data into DuckDB: {e}") from e

        # --- Apply SQL-based spatial organization ---
        doc_order = None
        metric_order = None # Or default order if no specific reordering is derived
        
        if self.task_config and self.task_config.get("analysis"):
            logger.warning("task_config.analysis found, but data is now in DuckDB. Re-analyzing based on DuckDB query results.")
            # The analysis might have been done earlier, but now we have data.
            # We should re-run the analysis or adjust the logic.
            # Let's assume set_sql_task was called and we have the SQL query.
            sql_query = self.task_config.get("sql_query")
            if sql_query:
                logger.debug("Re-analyzing SQL task with data loaded.")
                analysis = self._analyze_query(sql_query) # This will now work on real data
                # _apply_sql_organization will use this analysis
                self.sorted_matrix, self.metric_order, self.doc_order = self._apply_sql_organization(
                    score_matrix, # Pass the original matrix
                    analysis
                )
            else:
                logger.error("SQL task set but no query found in task_config.")
                # Fall back to default
                self._apply_default_organization(score_matrix)
        elif self.task_config and self.task_config.get("sql_query"):
            # SQL task was set, but analysis wasn't run or stored correctly
            # Let's run analysis and apply organization now that data is loaded
            sql_query = self.task_config["sql_query"]
            logger.info(f"Applying SQL task: {sql_query} to loaded data.")
            try:
                analysis = self._analyze_query(sql_query)
                self.sorted_matrix, self.metric_order, self.doc_order = self._apply_sql_organization(
                    score_matrix, # Pass the original matrix
                    analysis
                )
            except Exception as e:
                logger.error(f"Failed to apply SQL task during processing: {e}")
                # Fall back to default organization
                self._apply_default_organization(score_matrix)
        else:
            logger.info("No SQL task set. Using default organization.")
            self._apply_default_organization(score_matrix)
            
        logger.debug(f"Processing complete. Sorted matrix shape: {self.sorted_matrix.shape}")


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

    def encode(self) -> np.ndarray:
        """
        Encode the processed data into a full visual policy map.

        Returns:
            RGB image array of shape [height, width, 3].

        Raises:
            ValueError: If `process` has not been called successfully yet.
        """
        logger.debug("Encoding VPM...")
        if self.sorted_matrix is None:
            error_msg = "Data not processed yet. Call process() first."
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_docs, n_metrics = self.sorted_matrix.shape
        logger.debug(f"Encoding matrix of shape {n_docs}x{n_metrics}")
        # Calculate required width (3 metrics per pixel, ceiling division)
        width = (n_metrics + 2) // 3
        logger.debug(f"Calculated VPM width: {width} pixels")

        # Pad to multiple of 3
        padded = np.pad(self.sorted_matrix, ((0,0), (0, (3 - n_metrics % 3) % 3)))
        
        # Reshape directly to image format
        img = padded.reshape(n_docs, -1, 3)
        return (img * 255).astype(np.uint8)

    def get_critical_tile(self, tile_size: int = 3) -> bytes:
        """
        Get critical tile for edge devices (top-left section).

        Args:
            tile_size: Size of tile to extract (default 3x3).

        Returns:
            Compact byte representation of the tile.

        Raises:
            ValueError: If `process` has not been called successfully yet.
        """
        logger.debug(f"Extracting critical tile of size {tile_size}x{tile_size}")
        if self.sorted_matrix is None:
            error_msg = "Data not processed yet. Call process() first."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if tile_size <= 0:
            error_msg = f"tile_size must be positive, got {tile_size}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_docs, n_metrics = self.sorted_matrix.shape
        # Determine actual tile dimensions (cannot exceed matrix dimensions)
        actual_tile_height = min(tile_size, n_docs)
        actual_tile_width_metrics = min(
            tile_size * 3, n_metrics
        )  # Width in terms of metrics
        # Calculate width in pixels
        actual_tile_width_pixels = (actual_tile_width_metrics + 2) // 3
        logger.debug(
            f"Actual tile dimensions: {actual_tile_height} docs x {actual_tile_width_pixels} pixels ({actual_tile_width_metrics} metrics)"
        )

        # Get top-left section (most relevant documents & metrics)
        # Note: The original code used tile_size*3 for columns, which might be incorrect
        # if n_metrics < tile_size*3. We use actual_tile_width_metrics.
        tile_data = self.sorted_matrix[:actual_tile_height, :actual_tile_width_metrics]
        logger.debug(f"Extracted tile data shape: {tile_data.shape}")

        # Convert to compact byte format
        tile_bytes = bytearray()
        tile_bytes.append(actual_tile_width_pixels & 0xFF)  # Actual width in pixels
        tile_bytes.append(actual_tile_height & 0xFF)  # Actual height in docs
        tile_bytes.append(0 & 0xFF)  # X offset (always 0 for top-left)
        tile_bytes.append(0 & 0xFF)  # Y offset (always 0 for top-left)
        logger.debug("Appended tile header bytes.")

        # Add pixel data (1 byte per channel)
        # Iterate based on the actual extracted data dimensions
        for i in range(actual_tile_height):
            for j in range(actual_tile_width_metrics):  # Iterate through metrics
                pixel_x = j // 3
                channel = j % 3
                if i < tile_data.shape[0] and j < tile_data.shape[1]:
                    # Clamp and convert value (assuming [0,1] input)
                    normalized_value = np.clip(tile_data[i, j], 0.0, 1.0)
                    byte_value = int(normalized_value * 255) & 0xFF
                    tile_bytes.append(byte_value)
                else:
                    # This case should ideally not happen with the slicing above,
                    # but added for robustness.
                    tile_bytes.append(0)  # Padding
                    logger.warning(
                        f"Padding added at pixel ({i}, metric {j}) during tile extraction."
                    )
        logger.info(f"Critical tile extracted. Size: {len(tile_bytes)} bytes.")
        return bytes(tile_bytes)

    def get_decision(self, context_size: int = 3) -> Tuple[int, float]:
        """
        Get top decision with contextual understanding.

        Args:
            context_size: Size of context window to consider (NxN metrics area).

        Returns:
            Tuple of (document_index, relevance_score).

        Raises:
            ValueError: If `process` has not been called successfully yet.
        """
        logger.debug(f"Making decision with context size {context_size}")
        if self.sorted_matrix is None:
            error_msg = "Data not processed yet. Call process() first."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if context_size <= 0:
            error_msg = f"context_size must be positive, got {context_size}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_docs, n_metrics = self.sorted_matrix.shape
        # Determine actual context window size
        actual_context_docs = min(context_size, n_docs)
        actual_context_metrics = min(context_size * 3, n_metrics)  # Width in metrics
        logger.debug(
            f"Actual decision context: {actual_context_docs} docs x {actual_context_metrics} metrics"
        )

        # Get context window (top-left region)
        context = self.sorted_matrix[:actual_context_docs, :actual_context_metrics]
        logger.debug(f"Context data shape for decision: {context.shape}")

        # Calculate contextual relevance (weighted by position)
        weights = np.zeros_like(
            context, dtype=np.float64
        )  # Use float64 for calculation
        for i in range(context.shape[0]):  # Iterate through rows (docs)
            for j in range(context.shape[1]):  # Iterate through columns (metrics)
                # Weight decreases with distance from top-left (0,0)
                # j represents metric index, so j/3 gives approximate pixel x-coordinate
                pixel_x_coord = j / 3.0
                distance = np.sqrt(float(i) ** 2 + pixel_x_coord**2)
                # Example weight function: linear decrease
                weight = max(0.0, 1.0 - distance * 0.3)
                weights[i, j] = weight
        logger.debug("Calculated positional weights for context.")

        # Calculate weighted relevance
        # Use np.sum with dtype=np.float64 for precision in summation
        sum_weights = np.sum(weights, dtype=np.float64)
        if sum_weights > 0.0:
            weighted_sum = np.sum(context * weights, dtype=np.float64)
            weighted_relevance = weighted_sum / sum_weights
        else:
            logger.warning("Sum of weights is zero. Assigning relevance score 0.0.")
            weighted_relevance = 0.0
        weighted_relevance = float(weighted_relevance)  # Ensure it's a standard float
        logger.debug(f"Calculated weighted relevance score: {weighted_relevance:.4f}")

        # Get top document index from the *original* order
        # self.doc_order[0] is the index of the document that ended up in the first row
        # after sorting.
        top_doc_idx_in_original = 0
        if self.doc_order is not None and len(self.doc_order) > 0:
            top_doc_idx_in_original = int(self.doc_order[0])
        else:
            logger.warning(
                "doc_order is not available or empty. Defaulting top document index to 0."
            )


        # Print top 5 rows of sorted matrix and doc order for debugging
        logger.debug("Top 5 rows of sorted_matrix:")
        for i, row in enumerate(self.sorted_matrix[:5]):
            logger.debug(f"Row {i}: {row.tolist()}")

        logger.debug(f"Top 5 document indices (doc_order): {self.doc_order[:5].tolist()}")

        logger.info(
            f"Decision made: Document index {top_doc_idx_in_original}, Relevance {weighted_relevance:.4f}"
        )
        return top_doc_idx_in_original, weighted_relevance

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the current encoding state."""
        logger.debug("Retrieving metadata.")
        metadata = {
            "task": self.task,
            "task_config": self.task_config,
            "metric_order": self.metric_order.tolist()
            if self.metric_order is not None
            else [],
            "doc_order": self.doc_order.tolist() if self.doc_order is not None else [],
            "metric_names": self.metric_names,
            "precision": self.precision,
            # Add more metadata if needed
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
