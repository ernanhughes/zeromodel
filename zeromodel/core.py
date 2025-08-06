"""
Zero-Model Intelligence Encoder/Decoder with Virtual Index SQL Processing
This module provides the core functionality for transforming high-dimensional
policy evaluation data into spatially-optimized visual maps where the
intelligence is in the data structure itself, not in processing.
"""

import sqlite3
import numpy as np
import re
from typing import List, Tuple, Dict, Optional, Any
from zeromodel.normalizer import DynamicNormalizer

class ZeroModel:
    """
    Zero-Model Intelligence encoder/decoder with virtual index SQL processing.
    
    This class transforms high-dimensional policy evaluation data into
    spatially-optimized visual maps where:
    - Position = Importance (top-left = most relevant)
    - Color = Value (darker = higher priority)
    - Structure = Task logic
    
    The intelligence is in the data structure itself, not in processing.
    """
    
    def __init__(self, metric_names: List[str], precision: int = 8):
        """
        Initialize ZeroModel encoder with virtual index SQL processing.
        
        Args:
            metric_names: Names of all metrics being tracked
            precision: Bit precision for encoding (4-16)
        """
        self.metric_names = metric_names
        self.precision = max(4, min(16, precision))
        self.normalizer = DynamicNormalizer(metric_names)
        self.sorted_matrix = None
        self.doc_order = None
        self.metric_order = None
        self.task = "default"
        self.task_config = None
        
        # Create virtual index database
        self.virtual_conn = self._create_virtual_index_db()
    
    def _create_virtual_index_db(self) -> sqlite3.Connection:
        """Create a virtual database with index values for query analysis"""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Create table with proper column names
        columns = ", ".join([f'"{col}" INTEGER' for col in self.metric_names])
        cursor.execute(f"CREATE TABLE data (row_id INTEGER, {columns})")
        
        # Insert index values (we only need one row of indices)
        # This row will be used to determine column ordering
        values = [0] + list(range(len(self.metric_names)))
        placeholders = ", ".join(["?"] * (len(self.metric_names) + 1))
        cursor.execute(f"INSERT INTO data VALUES ({placeholders})", values)
        
        conn.commit()
        return conn
    
    def set_sql_task(self, sql_query: str):
        """
        Set task using SQL query for spatial organization.
        
        Args:
            sql_query: SQL query defining the task
            
        Example:
            model.set_sql_task("SELECT * FROM data ORDER BY uncertainty DESC, size ASC")
        """
        # Analyze query to determine spatial organization
        analysis = self._analyze_query(sql_query)
        
        # Store task configuration
        self.task = "sql_task"
        self.task_config = {
            "sql_query": sql_query,
            "analysis": analysis
        }
    
    def _analyze_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Analyze SQL query by running it against the virtual index database.
        
        This directly gives us the ordering we need for spatial organization.
        """
        cursor = self.virtual_conn.cursor()
        
        # Clean and validate the query
        cleaned_query = self._clean_and_validate_query(sql_query)
        
        try:
            # Execute query to get column ordering information
            cursor.execute(cleaned_query)
            result = cursor.fetchone()  # We only need one row for analysis
            
            if result is None:
                raise ValueError("Query returned no results")
            
            # Extract column ordering from the result
            # The result contains [row_id, col1, col2, ...]
            # Where the values are the indices that would produce the ordering
            column_indices = list(result[1:])  # Skip row_id
            
            # Determine metric ordering (which columns are most important)
            metric_order = np.argsort(column_indices)
            
            # For document ordering, we need to run a different query
            # We'll create a virtual document table with the same structure
            doc_conn = sqlite3.connect(":memory:")
            doc_cursor = doc_conn.cursor()
            
            # Create document table with all metric columns
            columns = ", ".join([f'doc_id INTEGER PRIMARY KEY'] + 
                            [f'{col} INTEGER' for col in self.metric_names])
            doc_cursor.execute(f"CREATE TABLE documents ({columns})")
            
            # Insert document IDs with values designed to test ordering
            n_metrics = len(self.metric_names)
            n_docs = min(100, n_metrics * 10)  # Enough documents to test ordering
            
            for doc_id in range(n_docs):
                values = [doc_id]
                for i, metric in enumerate(self.metric_names):
                    # For the first metric (uncertainty): descending order
                    if i == 0:
                        values.append(n_docs - doc_id)
                    # For the second metric (size): ascending order with wrap-around
                    elif i == 1:
                        values.append((doc_id + 1) % n_docs)
                    # For other metrics: default to document ID
                    else:
                        values.append(doc_id)
                placeholders = ", ".join(["?"] * (n_metrics + 1))
                doc_cursor.execute(f"INSERT INTO documents VALUES ({placeholders})", values)
            
            doc_conn.commit()
            
            # Execute the original query against this virtual document table
            # Just need to fix the table name if it's referenced
            doc_query = cleaned_query.replace("FROM data", "FROM documents")
            doc_query = doc_query.replace("FROM DATA", "FROM documents")
            
            # Execute to get document ordering
            doc_cursor.execute(doc_query)
            ordered_doc_ids = [row[0] for row in doc_cursor.fetchall()]
            
            doc_conn.close()
            
            return {
                "metric_order": metric_order.tolist(),
                "doc_order": ordered_doc_ids,
                "original_query": cleaned_query
            }
            
        except sqlite3.Error as e:
            raise ValueError(f"Invalid SQL query: {str(e)}") from e
            
    def _clean_and_validate_query(self, sql_query: str) -> str:
        """Clean and validate the SQL query"""
        # Remove any semicolons
        sql_query = sql_query.strip().rstrip(";")
        
        # Validate basic structure
        if not sql_query.upper().startswith("SELECT"):
            raise ValueError("Query must be a SELECT statement")
            
        # Ensure it's querying the 'data' table
        if "FROM" not in sql_query.upper():
            raise ValueError("Query must specify a FROM clause")
            
        # Basic SQL injection prevention
        if "--" in sql_query or "/*" in sql_query:
            raise ValueError("Comments not allowed in queries")
            
        return sql_query
    
    def process(self, score_matrix: np.ndarray) -> None:
        """
        Process a score matrix to prepare for encoding.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics]
        """
        # Update normalizer with new data
        self.normalizer.update(score_matrix)
        
        # Normalize scores
        normalized = self.normalizer.normalize(score_matrix)
        
        # Sort by task relevance
        if self.task_config and self.task_config.get("analysis"):
            self.sorted_matrix, self.metric_order, self.doc_order = self._apply_sql_organization(
                normalized, 
                self.task_config["analysis"]
            )
        else:
            # Default organization (if no task set)
            self.sorted_matrix = normalized
            self.metric_order = np.arange(normalized.shape[1])
            self.doc_order = np.arange(normalized.shape[0])
    
    def _apply_sql_organization(self, 
                               normalized: np.ndarray, 
                               analysis: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply spatial organization based on SQL query analysis"""
        # Apply metric ordering
        metric_order = np.array(analysis["metric_order"])
        sorted_by_metric = normalized[:, metric_order]
        
        # Apply document ordering
        # We need to map the virtual document IDs to our actual documents
        # Take the top N document IDs from the analysis
        n_docs = min(len(analysis["doc_order"]), normalized.shape[0])
        virtual_doc_order = np.array(analysis["doc_order"][:n_docs])
        
        # Create a mapping from virtual IDs to actual indices
        # Since our virtual IDs were sequential (0, 1, 2, ...),
        # the mapping is straightforward
        actual_doc_order = np.argsort(virtual_doc_order)
        
        # Sort documents
        sorted_matrix = sorted_by_metric[actual_doc_order]
        
        return sorted_matrix, metric_order, actual_doc_order
    
    def encode(self) -> np.ndarray:
        """
        Encode the processed data into a full visual policy map.
        
        Returns:
            RGB image array of shape [height, width, 3]
        """
        if self.sorted_matrix is None:
            raise ValueError("Data not processed yet. Call process() first.")
        
        n_docs, n_metrics = self.sorted_matrix.shape
        
        # Calculate required width (3 metrics per pixel)
        width = (n_metrics + 2) // 3  # Ceiling division
        
        # Create image array
        img = np.zeros((n_docs, width, 3), dtype=np.uint8)
        
        # Fill pixels with normalized scores (0-255)
        for i in range(n_docs):
            for j in range(n_metrics):
                pixel_x = j // 3
                channel = j % 3
                img[i, pixel_x, channel] = int(self.sorted_matrix[i, j] * 255)
        
        return img
    
    def get_critical_tile(self, tile_size: int = 3) -> bytes:
        """
        Get critical tile for edge devices (top-left section).
        
        Args:
            tile_size: Size of tile to extract (default 3x3)
        
        Returns:
            Compact byte representation of the tile
        """
        if self.sorted_matrix is None:
            raise ValueError("Data not processed yet. Call process() first.")
        
        # Get top-left section (most relevant documents & metrics)
        tile_data = self.sorted_matrix[:tile_size, :tile_size*3]
        
        # Convert to compact byte format
        tile_bytes = bytearray()
        tile_bytes.append(tile_size)  # Width
        tile_bytes.append(tile_size)  # Height
        tile_bytes.append(0)  # X offset
        tile_bytes.append(0)  # Y offset
        
        # Add pixel data (1 byte per channel)
        for i in range(tile_size):
            for j in range(tile_size * 3):  # 3 channels per pixel
                if i < tile_data.shape[0] and j < tile_data.shape[1]:
                    tile_bytes.append(int(tile_data[i, j] * 255))
                else:
                    tile_bytes.append(0)  # Padding
        
        return bytes(tile_bytes)
    
    def get_decision(self, context_size: int = 3) -> Tuple[int, float]:
        """
        Get top decision with contextual understanding, handling limited metrics.
        
        Args:
            context_size: Size of context window to consider
            
        Returns:
            (document_index, relevance_score)
        """
        if self.sorted_matrix is None:
            raise ValueError("Data not processed yet. Call process() first.")
        
        # Get context window (top-left region)
        # Handle case where we don't have enough metrics
        n_metrics = self.sorted_matrix.shape[1]
        context_width = min(context_size * 3, n_metrics)
        
        context = self.sorted_matrix[:context_size, :context_width]
        
        # Calculate contextual relevance (weighted by position)
        weights = np.zeros_like(context)
        for i in range(context.shape[0]):
            for j in range(context.shape[1]):
                # Weight decreases with distance from top-left
                distance = np.sqrt(i**2 + (j/3)**2)
                weights[i, j] = max(0, 1.0 - distance * 0.3)
        
        # Calculate weighted relevance
        weighted_relevance = np.sum(context * weights) / np.sum(weights)
        
        # Get top document index from sorted order
        top_doc_idx = self.doc_order[0] if len(self.doc_order) > 0 else 0
        
        return top_doc_idx, weighted_relevance

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the current encoding state"""
        return {
            "task": self.task,
            "task_config": self.task_config,
            "metric_order": self.metric_order.tolist() if self.metric_order is not None else [],
            "doc_order": self.doc_order.tolist() if self.doc_order is not None else [],
            "metric_names": self.metric_names,
            "precision": self.precision
        }

class HierarchicalVPM:
    """
    Hierarchical Visual Policy Map (HVPM) implementation with virtual index SQL support.
    
    This class creates a multi-level decision map system where:
    - Level 0: Strategic overview (small, low-res, for edge devices)
    - Level 1: Tactical view (medium resolution)
    - Level 2: Operational detail (full resolution)
    
    The hierarchical structure enables:
    - Efficient decision-making at multiple abstraction levels
    - Resource-adaptive processing (edge vs cloud)
    - Multi-stage decision pipelines
    - Visual exploration of policy landscapes
    """
    
    def __init__(self, 
                 metric_names: List[str],
                 num_levels: int = 3,
                 zoom_factor: int = 3,
                 precision: int = 8):
        """
        Initialize the hierarchical VPM system.
        
        Args:
            metric_names: Names of all metrics being tracked
            num_levels: Number of hierarchical levels (default 3)
            zoom_factor: Zoom factor between levels (default 3)
            precision: Bit precision for encoding (4-16)
        """
        self.metric_names = metric_names
        self.num_levels = num_levels
        self.zoom_factor = zoom_factor
        self.precision = precision
        self.levels = []
        self.metadata = {
            "version": "1.0",
            "temporal_axis": False,
            "levels": num_levels,
            "zoom_factor": zoom_factor
        }
    
    def process(self, 
                score_matrix: np.ndarray, 
                task: str,
                temporal_data: Optional[List[np.ndarray]] = None):
        """
        Process score matrix into hierarchical visual policy maps.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics]
            task: SQL query defining the task
            temporal_data: Optional list of score matrices over time
        """
        # Update metadata
        self.metadata["task"] = task
        self.metadata["documents"] = score_matrix.shape[0]
        self.metadata["metrics"] = score_matrix.shape[1]
        
        # Process temporal data if provided
        if temporal_data:
            self.metadata["temporal_axis"] = True
            self.metadata["time_points"] = len(temporal_data)
        
        # Clear existing levels
        self.levels = []
        
        # Create ZeroModel instance
        zeromodel = ZeroModel(self.metric_names, precision=self.precision)
        
        # Set task
        zeromodel.set_sql_task(task)
        
        # Process the base data
        zeromodel.process(score_matrix)
        
        # Create base level (Level 2: Full detail)
        base_level = self._create_base_level(zeromodel, score_matrix)
        self.levels.append(base_level)
        
        # Create higher levels (Level 1, Level 0)
        current_data = score_matrix
        for level in range(1, self.num_levels):
            # Reduce resolution by zoom factor
            num_docs = max(5, current_data.shape[0] // self.zoom_factor)
            num_metrics = max(3, current_data.shape[1] // self.zoom_factor)
            
            # Generate clustered data for this level
            clustered_data = self._cluster_data(current_data, num_docs, num_metrics)
            
            # Create level with same task configuration
            level_data = self._create_level(clustered_data, level, zeromodel.task_config)
            self.levels.insert(0, level_data)  # Insert at beginning (Level 0 first)
            
            # Use this level as basis for next higher level
            current_data = clustered_data
    
    def _cluster_data(self, 
                     data: np.ndarray, 
                     num_docs: int, 
                     num_metrics: int) -> np.ndarray:
        """
        Cluster data for higher-level views.
        
        Args:
            data: Input data matrix
            num_docs: Target number of document clusters
            num_metrics: Target number of metric clusters
        
        Returns:
            Clustered data matrix
        """
        docs, metrics = data.shape
        # Handle edge case where we have fewer items than clusters
        num_docs = min(num_docs, docs)
        num_metrics = min(num_metrics, metrics)
        # Create document clusters
        doc_clusters = []
        for i in range(num_docs):
            start_idx = i * docs // num_docs
            end_idx = (i + 1) * docs // num_docs
            if start_idx < end_idx:  # Ensure we have data to average
                doc_clusters.append(np.mean(data[start_idx:end_idx], axis=0))
            else:
                doc_clusters.append(data[start_idx])
        clustered_docs = np.array(doc_clusters)
        # Create metric clusters
        metric_clusters = []
        for j in range(num_metrics):
            start_idx = j * metrics // num_metrics
            end_idx = (j + 1) * metrics // num_metrics
            if start_idx < end_idx:  # Ensure we have data to average
                metric_clusters.append(np.mean(clustered_docs[:, start_idx:end_idx], axis=1))
            else:
                metric_clusters.append(clustered_docs[:, start_idx])
        return np.column_stack(metric_clusters)
    
    def _create_base_level(self, zeromodel: ZeroModel, score_matrix: np.ndarray) -> Dict[str, Any]:
        """Create the base level (highest detail)"""
        return {
            "level": self.num_levels - 1,
            "type": "base",
            "zeromodel": zeromodel,
            "vpm": zeromodel.encode(),
            "metadata": {
                "documents": score_matrix.shape[0],
                "metrics": score_matrix.shape[1],
                "sorted_docs": zeromodel.doc_order.tolist(),
                "sorted_metrics": zeromodel.metric_order.tolist()
            }
        }
    
    def _create_level(self, 
                     clustered_data: np.ndarray, 
                     level: int,
                     task_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a higher-level (more abstract) view"""
        # Create a simplified metric set for this level
        level_metrics = [
            f"cluster_{i}" for i in range(clustered_data.shape[1])
        ]
        # Process with simplified metrics
        zeromodel = ZeroModel(level_metrics, precision=self.precision)
        
        # Apply the same task configuration
        if task_config:
            # We need to recreate the SQL task with the new metric names
            original_query = task_config["analysis"]["original_query"]
            # Simple replacement of metric names
            new_query = original_query
            for i, old_name in enumerate(task_config["analysis"]["metric_names"]):
                if old_name in self.metric_names:
                    new_name = level_metrics[i % len(level_metrics)]
                    new_query = new_query.replace(old_name, new_name)
            
            zeromodel.set_sql_task(new_query)
        else:
            # Fallback to default task
            zeromodel.set_sql_task(f"SELECT * FROM data ORDER BY {level_metrics[0]} DESC")
        
        zeromodel.process(clustered_data)
        
        return {
            "level": level,
            "type": "clustered",
            "zeromodel": zeromodel,
            "vpm": zeromodel.encode(),
            "metadata": {
                "documents": clustered_data.shape[0],
                "metrics": clustered_data.shape[1],
                "sorted_docs": zeromodel.doc_order.tolist(),
                "sorted_metrics": zeromodel.metric_order.tolist()
            }
        }
    
    def get_level(self, level: int) -> Dict[str, Any]:
        """Get data for a specific level"""
        if level < 0 or level >= self.num_levels:
            raise ValueError(f"Level must be between 0 and {self.num_levels-1}")
        return self.levels[level]
    
    def get_tile(self, 
                level: int, 
                x: int = 0, 
                y: int = 0, 
                width: int = 3, 
                height: int = 3) -> bytes:
        """
        Get a tile from a specific level for edge devices.
        
        Args:
            level: Hierarchical level (0 = most abstract)
            x, y: Top-left corner of tile
            width, height: Dimensions of tile
        
        Returns:
            Compact byte representation of the tile
        """
        level_data = self.get_level(level)
        zeromodel = level_data["zeromodel"]
        # Get critical tile
        return zeromodel.get_critical_tile(tile_size=max(width, height))
    
    def get_decision(self, level: int) -> Tuple[int, float, int]:
        """
        Get top decision from a specific level.
        
        Returns:
            (level, document_index, relevance_score)
        """
        level_data = self.get_level(level)
        doc_idx, relevance = level_data["zeromodel"].get_decision()
        return (level, doc_idx, relevance)
    
    def zoom_in(self, level: int, doc_idx: int, metric_idx: int) -> int:
        """
        Determine the next level to zoom into based on current selection.
        
        Args:
            level: Current hierarchical level
            doc_idx: Selected document index
            metric_idx: Selected metric index
        
        Returns:
            Next level to zoom into (level+1, or same level if already at base)
        """
        if level >= self.num_levels - 1:
            return level  # Already at most detailed level
        return level + 1
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get complete metadata for the hierarchical map"""
        return self.metadata