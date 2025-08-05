"""
SQLTaskProcessor - Zero-Model Intelligence with SQL-based task specification

This class replaces natural language task specification with SQL queries, enabling
Zero-Model Intelligence to work with ANY tabular data source without predefined metrics.

Key features:
- Completely data-agnostic (works with robot sensors, weather stations, ML outputs)
- Uses SQLite in-memory for virtual table indexing
- Preserves ZeroMI's core innovation (spatial organization as intelligence)
- Requires only column headers as configuration
- Works within SQLite's limits (which are extremely generous for this use case)
"""

import sqlite3
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import json
import re
from zeromodel.core import ZeroModel

class SQLTaskProcessor:
    """
    Process SQL queries to determine spatial organization for Zero-Model Intelligence.
    
    This class enables ZeroMI to work with ANY tabular data source by:
    1. Creating a virtual table with index values (no semantic understanding needed)
    2. Executing SQL queries against this virtual table
    3. Analyzing query results to determine spatial organization
    4. Applying the organization to real data
    
    Example usage:
        processor = SQLTaskProcessor(column_names=["col_0", "col_1", "col_2"])
        task = processor.create_task("SELECT * FROM data ORDER BY col_1 DESC LIMIT 100")
        zeromodel = ZeroModel(column_names)
        zeromodel.process_with_task(score_matrix, task)
    """
    
    def __init__(self, metric_names: List[str]):
        """
        Initialize with column names (headers).
        
        Args:
            column_names: List of column names from your data source
        """
        self.metric_names = metric_names
        self.virtual_conn = self._create_virtual_table(len(metric_names))

    
    def _create_virtual_table(self, num_columns: int) -> sqlite3.Connection:
        """Create an in-memory virtual table with index values only"""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Create table with proper column names
        columns = ", ".join([f'"{col}" INTEGER' for col in self.metric_names])
        cursor.execute(f"CREATE TABLE data (row_id INTEGER, {columns})")
        
        # Insert a single row with index values (we only need structure for query analysis)
        # Using just one row is sufficient for query planning/analysis
        values = [0] + [0] * num_columns  # row_id + column values
        placeholders = ", ".join(["?"] * (num_columns + 1))
        cursor.execute(f"INSERT INTO data VALUES ({placeholders})", values)
        
        conn.commit()
        return conn
    
    def analyze_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Analyze SQL query to determine spatial organization parameters.
        
        Args:
            sql_query: SQL query defining the task
            
        Returns:
            Dictionary with spatial organization parameters
        """
        cursor = self.virtual_conn.cursor()
        
        # Validate and clean the query
        cleaned_query = self._clean_and_validate_query(sql_query)
        
        try:
            # Get query plan to understand sorting and filtering
            cursor.execute(f"EXPLAIN QUERY PLAN {cleaned_query}")
            query_plan = cursor.fetchall()
            
            # Analyze ORDER BY clause
            order_by_cols = self._extract_order_by(sql_query)
            
            # Analyze WHERE clause for filtering
            where_cols = self._extract_where_columns(sql_query)
            
            # Analyze SELECT clause for column importance
            select_cols = self._extract_select_columns(sql_query)
            
            # Determine column importance weights
            weights = self._calculate_column_weights(order_by_cols, where_cols, select_cols)
            
            # Determine document ordering
            doc_order = self._determine_document_order(order_by_cols)
            
            return {
                "weights": weights,
                "doc_order": doc_order,
                "order_by_columns": order_by_cols,
                "where_columns": where_cols,
                "select_columns": select_cols,
                "query_plan": query_plan,
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
    
    def _extract_order_by(self, sql_query: str) -> List[Tuple[str, str]]:
        """Extract ORDER BY columns and directions"""
        order_by_match = re.search(r"ORDER BY\s+([^(;]+)", sql_query, re.IGNORECASE)
        if not order_by_match:
            return []
        
        order_by_clause = order_by_match.group(1).strip()
        columns = []
        
        for part in order_by_clause.split(","):
            part = part.strip()
            if " " in part:
                col, direction = part.split(maxsplit=1)
                direction = direction.upper()
                if direction not in ["ASC", "DESC"]:
                    direction = "ASC"  # Default to ASC if invalid
            else:
                col = part
                direction = "ASC"
            
            # Clean column name (remove quotes, etc.)
            col = col.strip('"[]`')
            
            if col in self.metric_names:
                columns.append((col, direction))
        
        return columns
    
    def _extract_where_columns(self, sql_query: str) -> List[str]:
        """Extract columns used in WHERE clause"""
        where_match = re.search(r"WHERE\s+([^;]+?)(?:\s+ORDER BY|\s+GROUP BY|$)", 
                               sql_query, re.IGNORECASE)
        if not where_match:
            return []
        
        where_clause = where_match.group(1).strip()
        columns = []
        
        # Simple regex to find column names (could be enhanced for complex cases)
        for col in self.metric_names:
            if re.search(r"\b" + re.escape(col) + r"\b", where_clause, re.IGNORECASE):
                columns.append(col)
        
        return columns
    
    def _extract_select_columns(self, sql_query: str) -> List[str]:
        """Extract columns in SELECT clause"""
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_query, re.IGNORECASE)
        if not select_match:
            return []
        
        select_clause = select_match.group(1).strip()
        
        # Handle special cases
        if select_clause == "*":
            return self.metric_names.copy()
        
        # Extract column names
        columns = []
        for part in select_clause.split(","):
            part = part.strip()
            # Remove aliases and functions
            col_match = re.search(r"([\w_]+)\b", part)
            if col_match:
                col = col_match.group(1)
                if col in self.metric_names:
                    columns.append(col)
        
        return columns
    
    def _calculate_column_weights(self, 
                                order_by: List[Tuple[str, str]], 
                                where_cols: List[str],
                                select_cols: List[str]) -> Dict[str, float]:
        """
        Calculate column importance weights based on query structure.
        
        Importance hierarchy:
        1. ORDER BY columns (highest priority)
        2. WHERE clause columns (medium priority)
        3. SELECT columns (baseline priority)
        """
        weights = {col: 0.0 for col in self.metric_names}
        
        # ORDER BY columns get highest priority
        for i, (col, direction) in enumerate(order_by):
            # Weight decreases with position in ORDER BY
            weights[col] = 0.7 * (1.0 - i * 0.1)
        
        # WHERE clause columns get medium priority
        for col in where_cols:
            if col not in [ob[0] for ob in order_by]:  # Don't override ORDER BY
                weights[col] = max(weights[col], 0.3)
        
        # SELECT columns get baseline priority
        for col in select_cols:
            if col not in where_cols and col not in [ob[0] for ob in order_by]:
                weights[col] = max(weights[col], 0.1)
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            for col in weights:
                weights[col] = weights[col] / total
        
        return weights
    
    def _determine_document_order(self, order_by: List[Tuple[str, str]]) -> List[int]:
        """
        Determine document ordering based on ORDER BY clause.
        
        Note: In the virtual table approach, we don't actually sort data here.
        This is just a placeholder for the ordering logic that will be applied
        to real data later.
        """
        # In a real implementation, this would return the ordering pattern
        # For now, we just return a placeholder
        return [0]  # Will be replaced during actual processing
    
    def create_task(self, sql_query: str) -> Dict[str, Any]:
        """
        Create a task configuration from a SQL query.
        
        Args:
            sql_query: SQL query defining the task
            
        Returns:
            Task configuration dictionary
        """
        analysis = self.analyze_query(sql_query)
        return {
            "name": f"sql_task_{hash(sql_query) % 1000000}",
            "config": {
                "sql_query": sql_query,
                "weights": analysis["weights"],
                "order_by": analysis["order_by_columns"],
                "where_cols": analysis["where_columns"],
                "select_cols": analysis["select_columns"]
            },
            "analysis": analysis
        }
    
    def apply_to_data(self, 
                     score_matrix: np.ndarray, 
                     task: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply task configuration to real data to create spatial organization.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics]
            task: Task configuration from create_task()
            
        Returns:
            (sorted_matrix, metric_order, doc_order)
        """
        weights = task["config"]["weights"]
        
        # Calculate metric importance
        metric_importance = np.array([weights.get(col, 0) for col in self.metric_names])
        metric_order = np.argsort(metric_importance)[::-1]  # Most important first
        
        # Sort metrics
        sorted_by_metric = score_matrix[:, metric_order]
        
        # Calculate document relevance (weighted sum)
        doc_relevance = np.zeros(score_matrix.shape[0])
        for i in range(len(metric_importance)):
            doc_relevance += metric_importance[i] * sorted_by_metric[:, i]
        
        # Sort documents
        doc_order = np.argsort(doc_relevance)[::-1]  # Most relevant first
        
        # Final sorted matrix
        sorted_matrix = sorted_by_metric[doc_order]
        
        return sorted_matrix, metric_order, doc_order
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage or transmission"""
        return {
            "column_names": self.metric_names,
            "version": "1.0"
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SQLTaskProcessor':
        """Deserialize from dictionary"""
        return cls(column_names=data["column_names"])


class HierarchicalSQLTaskProcessor:
    """
    Process hierarchical SQL queries to determine spatial organization at each level.
    
    This class extends SQLTaskProcessor to support different queries at each hierarchy level:
    - Level 0 (Strategic): Broad categorization (e.g., document types)
    - Level 1 (Tactical): Intermediate organization (e.g., topics/themes)
    - Level 2 (Operational): Detailed sorting (e.g., specific attributes)
    
    Example usage:
        processor = HierarchicalSQLTaskProcessor(column_names)
        task = processor.create_task({
            0: "SELECT * FROM data ORDER BY document_type ASC",
            1: "SELECT * FROM data ORDER BY topic DESC",
            2: "SELECT * FROM data ORDER BY uncertainty DESC, size ASC"
        })
        zeromodel.process_with_task(score_matrix, task)
    """
    
    def __init__(self, column_names: List[str], num_levels: int = 3):
        """
        Initialize with column names and number of hierarchy levels.
        
        Args:
            column_names: List of column names from your data source
            num_levels: Number of hierarchical levels (default: 3)
        """
        self.column_names = column_names
        self.num_levels = num_levels
        self.virtual_conn = self._create_virtual_table(len(column_names))
        self.level_processors = {}
        
        # Create processors for each level
        for level in range(num_levels):
            self.level_processors[level] = SQLTaskProcessor(column_names)
    
    def create_task(self, level_queries: Dict[int, str]) -> Dict[str, Any]:
        """
        Create a hierarchical task configuration from SQL queries at each level.
        
        Args:
            level_queries: Dictionary mapping level numbers to SQL queries
            
        Returns:
            Hierarchical task configuration
        """
        # Validate level queries
        for level in level_queries.keys():
            if level < 0 or level >= self.num_levels:
                raise ValueError(f"Level must be between 0 and {self.num_levels-1}")
        
        # Process each level's query
        level_tasks = {}
        for level in range(self.num_levels):
            if level in level_queries:
                level_tasks[level] = self.level_processors[level].create_task(level_queries[level])
        
        return {
            "name": f"hierarchical_sql_task_{hash(str(level_queries)) % 1000000}",
            "level_tasks": level_tasks,
            "level_queries": level_queries
        }
    
    def apply_to_data(self, 
                     score_matrix: np.ndarray, 
                     task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply hierarchical task configuration to create multi-level spatial organization.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics]
            task: Hierarchical task configuration from create_task()
            
        Returns:
            List of level processing results (indexed by level)
        """
        level_results = []
        
        # Process each level
        current_data = score_matrix
        for level in range(self.num_levels):
            if level in task["level_tasks"]:
                # Apply task configuration for this level
                result = self.level_processors[level].apply_to_data(
                    current_data, 
                    task["level_tasks"][level]
                )
                sorted_matrix, metric_order, doc_order = result
                
                # Store results for this level
                level_results.append({
                    "level": level,
                    "sorted_matrix": sorted_matrix,
                    "metric_order": metric_order,
                    "doc_order": doc_order
                })
                
                # Use this level as basis for next higher level
                current_data = sorted_matrix
            else:
                # If no query for this level, just pass data through
                level_results.append({
                    "level": level,
                    "sorted_matrix": current_data,
                    "metric_order": np.arange(current_data.shape[1]),
                    "doc_order": np.arange(current_data.shape[0])
                })
        
        return level_results