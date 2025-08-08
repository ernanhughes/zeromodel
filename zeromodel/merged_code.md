<!-- Merged Python Code Files -->


## File: __init__.py

`python
# zeromodel/__init__.py
"""
Zero-Model Intelligence (zeromodel) - Standalone package for cognitive compression
"""

from .core import ZeroModel
from .config import init_config
from .hierarchical import HierarchicalVPM
from .edge import EdgeProtocol
from .hierarchical_edge import HierarchicalEdgeProtocol
from .normalizer import DynamicNormalizer
from .transform import get_critical_tile, transform_vpm

_version__ = "1.0.4"
__all__ = [
    "ZeroModel",
    "init_config",
    "HierarchicalVPM",
    "DynamicNormalizer",
    "transform_vpm",
    "get_critical_tile",
    "EdgeProtocol",
    "HierarchicalEdgeProtocol"
]

``n

## File: config.py

`python
# zeromodel/config.py
"""
ZeroModel Unified Configuration System

This module provides a comprehensive configuration system that:
- Merges default, environment, and user configurations
- Automatically configures logging based on settings
- Supports edge/cloud deployment scenarios
- Enables DuckDB bypass for simple queries
- Provides a clean API for accessing configuration

The system is designed to be:
- Simple: One-stop configuration for the entire system
- Flexible: Works for both library and application contexts
- Robust: Handles missing or invalid configuration gracefully
- Extensible: Easy to add new configuration options
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Initialize the base logger early so we can log config loading
logger = logging.getLogger("zeromodel.config")
logger.addHandler(logging.NullHandler())  # Prevent "no handler" warnings

DEFAULT_CONFIG = {
    # Core processing configuration
    "core": {
        "use_duckdb": True,
        "duckdb_bypass_threshold": 0.5,  # ms
        "precision": 8,
        "normalize_inputs": True,
        "nonlinearity_handling": "auto",
        "cache_preprocessed_vpm": True,
        "max_cached_tasks": 100,
    },
    
    # Edge deployment settings
    "edge": {
        "enabled": False,
        "default_tile_size": 3,
        "output_precision": "uint8",
        "max_memory_usage": 25 * 1024,  # 25KB in bytes
    },
    
    # Hierarchical VPM settings
    "hierarchical": {
        "num_levels": 3,
        "zoom_factor": 3,
        "wavelet_type": "haar",
    },
    
    # Logging configuration
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "handlers": [
            {
                "type": "console",
                "level": "DEBUG",
            },
            {
                "type": "file",
                "level": "DEBUG",
                "filename": "zeromodel.log",
                "max_bytes": 10 * 1024 * 1024,  # 10MB
                "backup_count": 5,
            }
        ]
    },
    
    # Advanced features
    "advanced": {
        "metric_discovery": False,
        "metric_discovery_interval": 3600,  # seconds
    }
}

def detect_deployment_environment() -> str:
    """Detect if we're running in edge or cloud environment"""
    if os.environ.get("ZERO_MODEL_EDGE", "false").lower() == "true":
        return "edge"
    elif os.environ.get("ZERO_MODEL_CLOUD", "false").lower() == "true":
        return "cloud"
    return "auto"

def get_edge_aware_defaults(env: str = "auto") -> Dict[str, Any]:
    """Return configuration defaults based on deployment environment"""
    if env == "edge":
        return {
            "core": {
                "use_duckdb": False,
                "precision": 8,
                "output_precision": "uint8",
                "edge": {
                    "enabled": True,
                    "default_tile_size": 3,
                    "max_memory_usage": 25 * 1024,
                }
            }
        }
    elif env == "cloud":
        return {
            "core": {
                "use_duckdb": True,
                "precision": 16,
                "output_precision": "float32",
                "edge": {
                    "enabled": False,
                    "default_tile_size": 5,
                }
            }
        }
    return {}

def load_user_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load user configuration from file if it exists"""
    # Try common locations
    possible_paths = [
        config_path,
        "zeromodel.yaml",
        "config/zeromodel.yaml",
        os.path.expanduser("~/.zeromodel/config.yaml"),
        os.path.expanduser("~/zeromodel.yaml")
    ]
    
    for path in possible_paths:
        if path and Path(path).exists():
            try:
                with open(path, 'r') as f:
                    logger.debug(f"Loading user configuration from {path}")
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")
    
    logger.debug("No user configuration file found")
    return {}

def setup_logging(config: Dict[str, Any]) -> None:
    """Configure logging based on the configuration settings"""
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO").upper()
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Remove existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Create formatter
    formatter = logging.Formatter(log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    # Add configured handlers
    for handler_config in log_config.get("handlers", []):
        handler_type = handler_config.get("type", "console").lower()
        handler_level = handler_config.get("level", "INFO").upper()
        
        try:
            if handler_type == "console":
                handler = logging.StreamHandler()
                handler.setLevel(getattr(logging, handler_level, logging.INFO))
                handler.setFormatter(formatter)
                root_logger.addHandler(handler)
                logger.debug("Added console logging handler")
                
            elif handler_type == "file":
                filename = handler_config.get("filename", "zeromodel.log")
                max_bytes = handler_config.get("max_bytes", 10 * 1024 * 1024)
                backup_count = handler_config.get("backup_count", 5)
                
                # Create directory if needed
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                
                from logging.handlers import RotatingFileHandler
                handler = RotatingFileHandler(
                    filename, 
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
                handler.setLevel(getattr(logging, handler_level, logging.DEBUG))
                handler.setFormatter(formatter)
                root_logger.addHandler(handler)
                logger.debug(f"Added file logging handler: {filename}")
                
            elif handler_type == "null":
                handler = logging.NullHandler()
                handler.setLevel(getattr(logging, handler_level, logging.INFO))
                root_logger.addHandler(handler)
                logger.debug("Added null logging handler")
                
        except Exception as e:
            logger.error(f"Failed to create {handler_type} logging handler: {e}")
    
    logger.info(f"Logging configured at level: {log_level}")
    logger.debug("Configuration details:")
    for key, value in config.items():
        logger.debug(f"  {key}: {value}")

def resolve_config(user_config: Optional[Dict[str, Any]] = None, 
                  env: str = "auto") -> Dict[str, Any]:
    """Resolve final configuration with smart defaults and logging setup"""
    # Start with base defaults
    config = DEFAULT_CONFIG.copy()
    
    # Apply environment-specific defaults
    env = env if env != "auto" else detect_deployment_environment()
    env_defaults = get_edge_aware_defaults(env)
    
    # Deep merge environment defaults
    def deep_merge(target: Dict, source: Dict) -> None:
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                deep_merge(target[key], value)
            else:
                target[key] = value
    
    deep_merge(config, env_defaults)
    
    # Apply user config
    user_config = user_config or load_user_config()
    deep_merge(config, user_config)
    
    # Smart DuckDB bypass detection
    if config["core"]["use_duckdb"] == "auto":
        # If we're in edge mode or precision is low, bypass DuckDB
        if config["edge"]["enabled"] or config["core"]["precision"] <= 8:
            config["core"]["use_duckdb"] = False
        else:
            config["core"]["use_duckdb"] = True
    
    # Setup logging based on resolved config
    setup_logging(config)
    
    # Log the final configuration (safely, without sensitive data)
    logger.debug("Configuration resolved successfully")
    logger.debug(f"Deployment environment: {env}")
    logger.debug(f"Core processing: use_duckdb={config['core']['use_duckdb']}, precision={config['core']['precision']}")
    logger.debug(f"Edge mode: {'enabled' if config['edge']['enabled'] else 'disabled'}")
    
    return config

def get_config_value(config: Dict[str, Any], 
                     section: str, 
                     key: str, 
                     default: Any = None) -> Any:
    """Safely get a configuration value with section and key"""
    try:
        return config[section][key]
    except (KeyError, TypeError):
        return default

class ConfigManager:
    """Singleton configuration manager for ZeroModel"""
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def initialize(self, 
                   user_config: Optional[Dict[str, Any]] = None,
                   env: str = "auto") -> None:
        """Initialize the configuration manager"""
        if self._config is None:
            self._config = resolve_config(user_config, env)
            logger.info("Configuration manager initialized")
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a configuration value by section and key(s)"""
        if self._config is None:
            raise RuntimeError("Configuration manager not initialized. Call initialize() first.")
        
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def set(self, value: Any, *keys: str) -> None:
        """Set a configuration value (use with caution)"""
        if self._config is None:
            raise RuntimeError("Configuration manager not initialized. Call initialize() first.")
        
        current = self._config
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        logger.debug(f"Configuration updated: {'.'.join(keys)} = {value}")
        
        # Special handling for logging changes
        if keys == ("logging",):
            setup_logging(self._config)
    
    def reload(self) -> None:
        """Reload configuration from user config file"""
        user_config = load_user_config()
        self._config = resolve_config(user_config)
        logger.info("Configuration reloaded from user config file")

# Global configuration manager instance
config_manager = ConfigManager()

def init_config(user_config: Optional[Dict[str, Any]] = None, env: str = "auto") -> None:
    """Initialize the global configuration"""
    config_manager.initialize(user_config, env)

def get_config(*keys: str, default: Any = None) -> Any:
    """Get a configuration value from the global configuration"""
    return config_manager.get(*keys, default=default)

def set_config(value: Any, *keys: str) -> None:
    """Set a configuration value in the global configuration"""
    config_manager.set(value, *keys)

# Initialize with defaults immediately
try:
    init_config()
    logger.debug("Global configuration initialized")
except Exception as e:
    logger.error(f"Failed to initialize global configuration: {e}")
    # Fall back to basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("zeromodel.config")
    logger.info("Basic logging initialized due to configuration error")

``n

## File: core.py

`python
# zeromodel/core.py
"""
Zero-Model Intelligence Encoder/Decoder with DuckDB SQL Processing.

This module provides the core functionality for transforming high-dimensional
policy evaluation data into spatially-optimized visual maps where the
intelligence is in the data structure itself, not in processing.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from zeromodel.normalizer import DynamicNormalizer
from zeromodel.config import init_config
from zeromodel.encoder import VPMEncoder
from zeromodel.feature_engineer import FeatureEngineer
from zeromodel.duckdb_adapter import DuckDBAdapter
from zeromodel.organization import SqlOrganizationStrategy

logger = logging.getLogger(__name__)  

precision_dtype_map = {
    'uint8': np.uint8,
    'uint16': np.uint16,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}

init_config()

DATA_NOT_PROCESSED_ERR = "Data not processed yet. Call process() or prepare() first."

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

    def __init__(
        self,
        metric_names: List[str],
        precision: int = 8,
        default_output_precision: str = "float32",
    ):
        """Initialize the ZeroModel core components."""
        logger.debug(
            "Initializing ZeroModel with metrics: %s, precision: %s, default_output_precision: %s",
            metric_names,
            precision,
            default_output_precision,
        )
        if not metric_names:
            raise ValueError("metric_names list cannot be empty.")
        valid_precisions = ["uint8", "float16", "float32", "float64"]
        if default_output_precision not in valid_precisions:
            logger.warning(
                "Invalid default_output_precision '%s'. Must be one of %s. Defaulting to 'float32'.",
                default_output_precision,
                valid_precisions,
            )
            default_output_precision = "float32"

        # Core attributes
        self.metric_names = list(metric_names)
        self.effective_metric_names = list(metric_names)
        self.precision = max(4, min(16, precision))  # legacy internal precision
        self.default_output_precision = default_output_precision
        self.sorted_matrix: Optional[np.ndarray] = None
        self.doc_order: Optional[np.ndarray] = None
        self.metric_order: Optional[np.ndarray] = None
        self.task: str = "default"
        self.task_config: Optional[Dict[str, Any]] = None

        # Components
        self.duckdb = DuckDBAdapter(self.effective_metric_names)
        self.normalizer = DynamicNormalizer(self.effective_metric_names)
        self._encoder = VPMEncoder(default_output_precision)
        self._feature_engineer = FeatureEngineer()
        # Pluggable organization strategy (default SQL-backed)
        self._org_strategy = SqlOrganizationStrategy(self.duckdb)

        logger.info(
            "ZeroModel initialized with %d metrics. Default output precision: %s.",
            len(self.effective_metric_names),
            self.default_output_precision,
        )

    def encode(self, output_precision: Optional[str] = None) -> np.ndarray:
        if self.sorted_matrix is None:
            raise ValueError(DATA_NOT_PROCESSED_ERR)
        return self._encoder.encode(self.sorted_matrix, output_precision)

    def get_critical_tile(self, tile_size: int = 3, precision: Optional[str] = None) -> bytes:
        if self.sorted_matrix is None:
            raise ValueError(DATA_NOT_PROCESSED_ERR)
        return self._encoder.get_critical_tile(self.sorted_matrix, tile_size=tile_size, precision=precision)

    def get_decision(self, context_size: int = 3) -> Tuple[int, float]:
        """
        Get top decision with contextual understanding.
        NOTE: This method operates on the internal sorted_matrix (normalized float).
        It should produce a relevance score between 0.0 and 1.0.
        """
        logger.debug(f"Making decision with context size {context_size}")
        if self.sorted_matrix is None:
            error_msg = DATA_NOT_PROCESSED_ERR
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
        logger.debug(f"Actual decision context: {actual_context_docs} docs x {actual_context_metrics} metrics")

        # Get context window (top-left region) - operates on normalized float data
        context = self.sorted_matrix[:actual_context_docs, :actual_context_metrics]
        logger.debug(f"Context data shape for decision: {context.shape}")

        if context.size == 0:
            logger.warning("Empty context window. Returning default decision (0, 0.0).")
            top_doc_idx_in_original = int(self.doc_order[0]) if (self.doc_order is not None and len(self.doc_order) > 0) else 0
            return (top_doc_idx_in_original, 0.0)

        # Vectorized positional weight calculation
        # Rows (docs) indices
        row_indices = np.arange(actual_context_docs, dtype=np.float64).reshape(-1, 1)
        # Column (metric) indices -> convert to approximate pixel x-coordinate (metric groups of 3)
        col_indices = np.arange(actual_context_metrics, dtype=np.float64)
        pixel_x_coords = col_indices / 3.0
        # Broadcast to grid
        distances = np.sqrt(row_indices**2 + pixel_x_coords**2)
        weights = np.clip(1.0 - distances * 0.3, 0.0, None)

        sum_weights = weights.sum(dtype=np.float64)
        if sum_weights > 0.0:
            weighted_sum = np.sum(context * weights, dtype=np.float64)
            weighted_relevance = float(weighted_sum / sum_weights)
        else:
            logger.warning("Sum of weights is zero after vectorized computation. Assigning relevance score 0.0.")
            weighted_relevance = 0.0

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


    # _init_duckdb removed: handled by DuckDBAdapter.ensure_schema

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

    def _validate_prepare_inputs(self, score_matrix: np.ndarray, sql_query: str) -> None:
        """Validate inputs for prepare(); mutate effective metric names if needed."""
        original_metric_names = self.effective_metric_names
        if score_matrix is None:
            raise ValueError("score_matrix cannot be None.")
        if not isinstance(score_matrix, np.ndarray):
            raise ValueError(f"score_matrix must be a NumPy ndarray, got {type(score_matrix)}.")
        if score_matrix.ndim != 2:
            raise ValueError(f"score_matrix must be 2D, got shape {score_matrix.shape}.")
        if not sql_query or not isinstance(sql_query, str):
            raise ValueError("sql_query must be a non-empty string.")
        if not np.isfinite(score_matrix).all():
            nan_count = int(np.isnan(score_matrix).sum())
            pos_inf_count = int(np.isposinf(score_matrix).sum())
            neg_inf_count = int(np.isneginf(score_matrix).sum())
            raise ValueError(
                "score_matrix contains non-finite values: "
                f"NaN={nan_count}, +inf={pos_inf_count}, -inf={neg_inf_count}. "
                "Clean or impute these values before calling prepare()."
            )
        # Column count adaptation
        _, n_cols = score_matrix.shape
        if n_cols != len(original_metric_names):
            logger.warning(
                "Column count mismatch: %d (expected) vs %d (received).",
                len(original_metric_names), n_cols,
            )
            if n_cols > len(original_metric_names):
                added = [f"col_{i}" for i in range(len(original_metric_names), n_cols)]
                new_names = list(original_metric_names) + added
            else:
                new_names = list(original_metric_names[:n_cols])
            self.effective_metric_names = new_names
            self.normalizer = DynamicNormalizer(new_names)

    def prepare(self, score_matrix: np.ndarray, sql_query: str, nonlinearity_hint: Optional[str] = None) -> None:
        """Prepare model with data, optional feature engineering, and SQL-driven organization."""
        logger.info(
            f"Preparing ZeroModel with data shape {score_matrix.shape}, query: '{sql_query}', nonlinearity_hint: {nonlinearity_hint}"
        )
        self._validate_prepare_inputs(score_matrix, sql_query)
        original_metric_names = list(self.effective_metric_names)

        # --- 1. Dynamic Normalization ---
        try:
            logger.debug("Updating DynamicNormalizer with new data ranges.")
            self.normalizer.update(score_matrix)
            logger.debug("Normalizer updated. Applying normalization to the data.")
            normalized_data = self.normalizer.normalize(score_matrix)
            logger.debug("Data normalized successfully.")
        except Exception as e:
            logger.error(f"Failed during normalization step: {e}")
            raise RuntimeError(f"Error during data normalization: {e}") from e
        # --- End Dynamic Normalization ---

        # --- 2. Hint-Based Feature Engineering (delegated) ---
        processed_data, effective_metric_names = self._feature_engineer.apply(
            nonlinearity_hint, normalized_data, list(original_metric_names)
        )
        if processed_data is not normalized_data:
            logger.info(
                "Feature engineering added %d new metrics (total now %d)",
                processed_data.shape[1] - normalized_data.shape[1],
                processed_data.shape[1],
            )
        # --- End Feature Engineering ---

        # --- 3. Update Instance Metric Names (after feature engineering) ---
        # Crucially, update the instance's metric_names so get_metadata() and other
        # parts of the class that rely on self.metric_names are consistent with the
        # data that was actually processed and sorted.
        # Store the original names in case they are needed
        self._original_metric_names = self.effective_metric_names
        self.effective_metric_names = effective_metric_names
        logger.debug(f"Updated instance metric_names to reflect processed features: {len(self.effective_metric_names)} metrics.")
        # --- End Update Instance Metric Names ---

        # --- 4. Set organization task & apply via strategy ---
        try:
            self._org_strategy.set_task(sql_query)
            logger.debug("Organization task set on strategy. Executing organize().")
            final_matrix, metric_order, doc_order, analysis = self._org_strategy.organize(
                processed_data, self.effective_metric_names
            )
            self.sorted_matrix = final_matrix
            self.metric_order = metric_order
            self.doc_order = doc_order
            self.task = self._org_strategy.name + "_task"
            self.task_config = {"sql_query": sql_query, "analysis": analysis}
            logger.debug("Organization strategy applied successfully.")
        except Exception as e:  # noqa: broad-except
            logger.error(f"Failed during organization strategy execution: {e}")
            raise RuntimeError(f"Error during organization strategy: {e}") from e
        # --- End Strategy Organization ---
            
        logger.info("ZeroModel preparation complete. Ready for encode/get_decision/etc.")

    # Legacy _set_sql_task and _apply_sql_organization removed in favor of strategy abstraction


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
            "metric_names": self.effective_metric_names, # This should now be effective_metric_names after prepare()
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
            "metric_names": self.effective_metric_names,
            "metric_order": self.metric_order.tolist(),
            "doc_order": self.doc_order.tolist()
        }


# --- Example usage or test code (optional, remove or comment out for library module) ---
# if __name__ == "__main__":
#     # This would typically be in a separate test or example script
#     pass
``n

## File: duckdb_adapter.py

`python
"""DuckDB adapter encapsulating schema management, data loading, and query analysis."""
from __future__ import annotations
import logging
from typing import List, Dict, Any
import duckdb

logger = logging.getLogger(__name__)

class DuckDBAdapter:
    def __init__(self, metric_names: List[str]):
        self._conn = duckdb.connect(database=':memory:')
        self._create_table(metric_names)

    # ---------------- Internal helpers -----------------
    def _create_table(self, metric_names: List[str]):
        cols = ", ".join([f'"{c}" FLOAT' for c in metric_names])
        self._conn.execute(f"CREATE TABLE virtual_index (row_id INTEGER, {cols})")
        logger.debug("DuckDB virtual_index table created with %d metrics.", len(metric_names))

    # ---------------- Public API -----------------------
    def ensure_schema(self, metric_names: List[str]):
        try:
            cur = self._conn.execute("PRAGMA table_info(virtual_index)")
            info = cur.fetchall()
            expected = ["row_id"] + list(metric_names)
            current = [r[1] for r in info]
            if current != expected:
                logger.debug("Recreating DuckDB schema. Expected %s, found %s", expected, current)
                self._conn.execute("DROP TABLE IF EXISTS virtual_index")
                self._create_table(metric_names)
        except Exception as e:  # noqa: broad-except
            raise RuntimeError(f"Failed ensuring DuckDB schema: {e}") from e

    def load_matrix(self, matrix, metric_names: List[str]):
        try:
            self._conn.execute("DELETE FROM virtual_index")
            col_list = ", ".join([f'"{m}"' for m in metric_names])
            placeholders = ", ".join(["?"] * len(metric_names))
            sql = f"INSERT INTO virtual_index (row_id, {col_list}) VALUES (?, {placeholders})"
            for rid, row in enumerate(matrix):
                self._conn.execute(sql, [rid] + row.tolist())
            logger.debug("Loaded %d rows into DuckDB.", matrix.shape[0])
        except Exception as e:  # noqa: broad-except
            raise RuntimeError(f"Failed loading data into DuckDB: {e}") from e

    def analyze_query(self, sql_query: str, metric_names: List[str]) -> Dict[str, Any]:
        import re
        try:
            if sql_query.strip().upper().startswith("SELECT *"):
                modified = sql_query.replace("SELECT *", "SELECT row_id", 1)
            else:
                modified = f"SELECT row_id FROM ({sql_query}) AS user_sorted_view"
            result = self._conn.execute(modified).fetchall()
            doc_order = [r[0] for r in result]
            metric_order = list(range(len(metric_names)))
            primary_sort_metric_name = None
            primary_sort_metric_index = None
            match = re.search(r"ORDER\s+BY\s+(\w+)", sql_query, re.IGNORECASE)
            if match:
                candidate = match.group(1)
                if candidate in metric_names:
                    primary_sort_metric_name = candidate
                    primary_sort_metric_index = metric_names.index(candidate)
            return {
                'doc_order': doc_order,
                'metric_order': metric_order,
                'primary_sort_metric_name': primary_sort_metric_name,
                'primary_sort_metric_index': primary_sort_metric_index,
                'original_query': sql_query,
            }
        except Exception as e:  # noqa: broad-except
            raise ValueError(f"Invalid SQL query execution: {e}") from e

    @property
    def connection(self):  # Expose if low-level access needed
        return self._conn

__all__ = ["DuckDBAdapter"]
``n

## File: edge.py

`python
# zeromodel/edge.py
"""
Edge Device Protocol

This module provides the EdgeProtocol class which implements a minimal
protocol for edge devices with <25KB memory. It handles:
- Receiving policy map tiles from a proxy
- Making decisions based on the tile
- Sending back results with minimal overhead
"""

import struct
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class EdgeProtocol:
    """
    Communication protocol for zeromodel edge devices with <25KB memory.
    
    This implements a minimal protocol that:
    - Works with tiny memory constraints
    - Requires minimal processing
    - Survives network transmission
    - Enables zero-model decision making
    
    Designed to work with as little as 180 bytes of code on the device.
    """
    
    # Protocol version (1 byte)
    PROTOCOL_VERSION = 1
    
    # Message types (1 byte each)
    MSG_TYPE_REQUEST = 0x01
    MSG_TYPE_TILE = 0x02
    MSG_TYPE_DECISION = 0x03
    MSG_TYPE_ERROR = 0x04
    
    # Maximum tile size (for memory constraints)
    MAX_TILE_WIDTH = 3
    MAX_TILE_HEIGHT = 3

    @staticmethod
    def create_request(task_description: str) -> bytes:
        """
        Create a request message for the edge proxy.
        
        Args:
            task_description: Natural language task description
        
        Returns:
            Binary request message
            
        Raises:
            ValueError: If task_description is None.
        """
        logger.debug(f"Creating request for task: '{task_description}'")
        if task_description is None:
             error_msg = "Task description cannot be None"
             logger.error(error_msg)
             raise ValueError(error_msg)

        task_bytes = task_description.encode('utf-8')
        original_len = len(task_bytes)
        if original_len > 255:
            logger.warning(f"Task description ({original_len} bytes) exceeds 255 bytes, truncating.")
            task_bytes = task_bytes[:255] # Truncate if too long
        elif original_len == 0:
             logger.info("Creating request with an empty task description.")

        message = struct.pack(
            f"BBB{len(task_bytes)}s",
            EdgeProtocol.PROTOCOL_VERSION,
            EdgeProtocol.MSG_TYPE_REQUEST,
            len(task_bytes), # Length of the task description
            task_bytes
        )
        logger.debug(f"Request message created, size: {len(message)} bytes")
        return message
    
    @staticmethod
    def parse_tile(tile_data: bytes) -> Tuple[int, int, int, int, bytes]:
        """
        Parse a tile message from the proxy.
        
        Args:
            tile_data: Binary tile data (at least 4 bytes: width, height, x_offset, y_offset)
        
        Returns:
            Tuple containing (width, height, x_offset, y_offset, pixels_data)
            
        Raises:
            ValueError: If tile_data is invalid (e.g., too short, invalid dimensions).
        """
        header_size = 4
        logger.debug(f"Parsing tile data, received size: {len(tile_data)} bytes")
        if not tile_data or len(tile_data) < header_size:
            error_msg = f"Invalid tile format: data too short (expected >= {header_size} bytes, got {len(tile_data)})."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        width = tile_data[0]
        height = tile_data[1]
        x_offset = tile_data[2]
        y_offset = tile_data[3]
        pixels_data = tile_data[header_size:] # Remaining bytes are pixel data
        
        logger.debug(f"Parsed tile header: width={width}, height={height}, x_offset={x_offset}, y_offset={y_offset}")

        # Validate dimensions strictly based on protocol limits
        # Note: The original logic modified width/height if they exceeded limits,
        # but this can be confusing. It's often better to reject invalid data.
        # However, to maintain potential compatibility with the original intent,
        # we can log warnings but proceed, assuming the *data* conforms to the limits.
        # A stricter approach would be:
        # if not (0 < width <= EdgeProtocol.MAX_TILE_WIDTH) or not (0 < height <= EdgeProtocol.MAX_TILE_HEIGHT):
        #    error_msg = f"Invalid tile dimensions: width={width} (max {EdgeProtocol.MAX_TILE_WIDTH}), height={height} (max {EdgeProtocol.MAX_TILE_HEIGHT})"
        #    logger.error(error_msg)
        #    raise ValueError(error_msg)

        if width > EdgeProtocol.MAX_TILE_WIDTH:
            logger.warning(f"Tile width ({width}) exceeds MAX_TILE_WIDTH ({EdgeProtocol.MAX_TILE_WIDTH}). Processing with max width.")
            # width = EdgeProtocol.MAX_TILE_WIDTH # Original logic - commented for clarity on strictness
        if height > EdgeProtocol.MAX_TILE_HEIGHT:
            logger.warning(f"Tile height ({height}) exceeds MAX_TILE_HEIGHT ({EdgeProtocol.MAX_TILE_HEIGHT}). Processing with max height.")
            # height = EdgeProtocol.MAX_TILE_HEIGHT # Original logic - commented for clarity on strictness

        # Optional: Check if the actual pixel data length matches the expected size
        # expected_pixel_count = width * height * 3 # Assuming 3 channels per pixel
        # if len(pixels_data) < expected_pixel_count:
        #     logger.warning(f"Tile pixel data ({len(pixels_data)} bytes) is shorter than expected ({expected_pixel_count} bytes).")

        result = (width, height, x_offset, y_offset, pixels_data)
        logger.debug(f"Tile parsed successfully: {result[:4]} + {len(pixels_data)} pixel bytes")
        return result
    
    @staticmethod
    def make_decision(tile_message_data: bytes) -> bytes:
        """
        Process a tile message and make a decision.
        Assumes the tile_message_data includes the full message structure,
        including the header bytes if it came directly from a received message.
        If it's just the payload from a MSG_TYPE_TILE message, it should start
        with width, height, x_offset, y_offset.

        For simplicity, and based on the original `parse_tile` logic,
        we'll assume `tile_message_data` is the payload starting with
        width, height, x_offset, y_offset.

        Args:
            tile_message_data: Binary tile data (payload).
        
        Returns:
            Binary decision message (MSG_TYPE_DECISION).
        """
        logger.debug(f"Making decision based on tile data ({len(tile_message_data)} bytes)")
        try:
            width, height, x_offset, y_offset, pixels_data = EdgeProtocol.parse_tile(tile_message_data)
        except ValueError as e:
            logger.error(f"Failed to parse tile for decision making: {e}")
            # Return an error message instead of raising, if the protocol expects messages back
            return EdgeProtocol.create_error(10, f"Tile Parse Error: {str(e)}")

        # Simple decision logic: check top-left pixel value (R channel)
        # Check if we have at least one pixel's R channel data
        # Pixel data is [R0, G0, B0, R1, G1, B1, ...]
        # Top-left pixel R is at index 0
        is_relevant = 0 # Default to not relevant
        if len(pixels_data) > 0:
            top_left_r_value = pixels_data[0]
            # Decision: is this "dark enough" to be relevant?
            is_relevant = 1 if top_left_r_value < 128 else 0
            logger.debug(f"Top-left pixel R value: {top_left_r_value}, Decision (is_relevant): {is_relevant}")
        else:
             logger.warning("Tile pixel data is empty. Defaulting decision to not relevant.")
             # is_relevant remains 0

        # Create decision message
        # Format: [version][type][decision][reserved]
        decision_message = struct.pack("BBBB", 
                          EdgeProtocol.PROTOCOL_VERSION,
                          EdgeProtocol.MSG_TYPE_DECISION,
                          is_relevant,
                          0)  # Reserved byte
        logger.info(f"Decision made: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
        logger.debug(f"Decision message created, size: {len(decision_message)} bytes")
        return decision_message
    
    @staticmethod
    def create_error(code: int, message: str = "") -> bytes:
        """
        Create an error message.
        
        Args:
            code: Error code (1 byte recommended)
            message: Optional error message (max ~252 chars)
        
        Returns:
            Binary error message (MSG_TYPE_ERROR)
        """
        logger.debug(f"Creating error message: code={code}, message='{message}'")
        if code < 0 or code > 255: # Assuming 1-byte code
             logger.warning(f"Error code {code} is outside typical 0-255 range for 1-byte field.")
        msg_bytes = message.encode('utf-8')[:252]  # Leave room for headers (1+1+1+252 = 255 max)
        
        error_message = struct.pack(
            f"BBB{len(msg_bytes)}s",
            EdgeProtocol.PROTOCOL_VERSION,
            EdgeProtocol.MSG_TYPE_ERROR,
            code & 0xFF, # Ensure code fits in 1 byte
            msg_bytes
        )
        logger.debug(f"Error message created, size: {len(error_message)} bytes")
        return error_message
``n

## File: encoder.py

`python
"""VPM (Visual Policy Map) encoding utilities.

This module contains the VPMEncoder class which is responsible for:
 - Converting a normalized, spatially-organized score matrix into an RGB image tensor
 - Handling padding of metric channels to 3-channel pixels
 - Converting to requested output precision (uint8/uint16/float16/float32/float64)
 - Extracting a critical top-left tile as a compact byte payload

It deliberately knows nothing about DuckDB, feature engineering, or normalization
pipelines; it operates purely on already-prepared numpy arrays.
"""
from __future__ import annotations
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

precision_dtype_map = {
    'uint8': np.uint8,
    'uint16': np.uint16,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}

class VPMEncoder:
    """Stateless encoder for turning normalized matrices into VPM images/tiles."""
    def __init__(self, default_output_precision: str = 'float32'):
        valid = set(precision_dtype_map.keys())
        if default_output_precision not in valid:
            logger.warning("Invalid default_output_precision '%s'. Falling back to 'float32'.", default_output_precision)
            default_output_precision = 'float32'
        self.default_output_precision = default_output_precision

    def encode(self, sorted_matrix: np.ndarray, output_precision: Optional[str] = None) -> np.ndarray:
        if sorted_matrix is None:
            raise ValueError("sorted_matrix cannot be None.")
        if sorted_matrix.ndim != 2:
            raise ValueError(f"sorted_matrix must be 2D, got shape {sorted_matrix.shape}.")
        n_docs, n_metrics = sorted_matrix.shape
        if n_docs == 0 or n_metrics == 0:
            raise ValueError("sorted_matrix cannot have zero docs or metrics.")
        final_precision = output_precision or self.default_output_precision
        if final_precision not in precision_dtype_map:
            logger.warning("Unsupported output_precision '%s'. Using default '%s'.", final_precision, self.default_output_precision)
            final_precision = self.default_output_precision
        target_dtype = precision_dtype_map[final_precision]
        matrix = sorted_matrix.astype(np.float32, copy=False)
        width = (n_metrics + 2) // 3
        padding = (3 - (n_metrics % 3)) % 3
        if padding:
            matrix = np.pad(matrix, ((0, 0), (0, padding)), mode='constant', constant_values=0.0)
        try:
            img_data = matrix.reshape(n_docs, width, 3)
        except ValueError as e:
            raise ValueError(f"Cannot reshape data of shape {matrix.shape} to ({n_docs}, {width}, 3).") from e
        try:
            from .vpm_logic import denormalize_vpm  # local import to avoid cycle
            img = denormalize_vpm(img_data, output_type=target_dtype)
        except Exception:
            if target_dtype == np.uint8:
                img = np.clip(img_data * 255.0, 0, 255).astype(target_dtype)
            elif target_dtype == np.uint16:
                img = np.clip(img_data * 65535.0, 0, 65535).astype(target_dtype)
            else:
                img = np.clip(img_data, 0.0, 1.0).astype(target_dtype)
        logger.debug("Encoded VPM image: shape=%s dtype=%s (precision=%s)", img.shape, img.dtype, final_precision)
        return img

    def get_critical_tile(self, sorted_matrix: np.ndarray, tile_size: int = 3, precision: Optional[str] = None) -> bytes:
        if sorted_matrix is None:
            raise ValueError("sorted_matrix cannot be None.")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        n_docs, n_metrics = sorted_matrix.shape
        if n_docs == 0 or n_metrics == 0:
            raise ValueError("sorted_matrix cannot have zero docs or metrics.")
        final_precision = precision or self.default_output_precision
        if final_precision not in precision_dtype_map:
            logger.warning("Unsupported tile precision '%s'. Using default '%s'.", final_precision, self.default_output_precision)
            final_precision = self.default_output_precision
        target_dtype = precision_dtype_map[final_precision]
        actual_h = min(tile_size, n_docs)
        tile_metrics_w = min(tile_size * 3, n_metrics)
        pixel_w = (tile_metrics_w + 2) // 3
        tile_slice = sorted_matrix[:actual_h, :tile_metrics_w].astype(np.float32, copy=False)
        try:
            from .vpm_logic import denormalize_vpm, normalize_vpm
            tile_norm = normalize_vpm(tile_slice)
            tile_converted = denormalize_vpm(tile_norm, output_type=target_dtype)
        except Exception:
            if target_dtype == np.uint8:
                tile_converted = np.clip(tile_slice * 255.0, 0, 255).astype(target_dtype)
            elif target_dtype == np.uint16:
                tile_converted = np.clip(tile_slice * 65535.0, 0, 65535).astype(target_dtype)
            else:
                tile_converted = np.clip(tile_slice, 0.0, 1.0).astype(target_dtype)
        payload = bytearray()
        payload.append(pixel_w & 0xFF)
        payload.append(actual_h & 0xFF)
        payload.append(0)
        payload.append(0)
        payload.extend(tile_converted.flatten().tobytes())
        logger.debug("Extracted critical tile: tile_size=%d actual=(%d,%d) precision=%s bytes=%d", tile_size, actual_h, pixel_w, final_precision, len(payload))
        return bytes(payload)

__all__ = ["VPMEncoder"]
``n

## File: feature_engine.py

`python
# zeromodel/feature_engine.py
"""
Feature Engineering for ZeroModel to handle non-linear patterns.
"""

import logging
import numpy as np
from typing import List, Dict, Callable, Any

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    Handles non-linear feature transformations for ZeroModel.
    """
    def __init__(self):
        """
        Initializes the feature engine with a registry of transformation functions.
        """
        self.transform_registry: Dict[str, Callable] = {
            "XOR": self._xor_transform,
            "RADIAL": self._radial_transform,
            "PRODUCT": self._product_transform,
            "DIFFERENCE": self._difference_transform,
            # Add more as needed
        }
        self.applied_transforms: List[Dict[str, Any]] = [] # Track transformations
        self.engineered_metric_names: List[str] = []

    def transform(self, matrix: np.ndarray, pattern_type: str, original_metric_names: List[str]) -> np.ndarray:
        """
        Applies a registered transformation to the input matrix.

        Args:
            matrix: The input score matrix (normalized).
            pattern_type: The type of transformation to apply (key in registry).
            original_metric_names: The names of the original metrics.

        Returns:
            np.ndarray: The transformed matrix (original + new features).
        """
        logger.debug(f"Applying feature transformation: {pattern_type}")
        transform_fn = self.transform_registry.get(pattern_type)
        
        if transform_fn is None:
            logger.debug(f"No specific transform for '{pattern_type}', using identity.")
            transform_fn = self._identity_transform

        try:
            # Store original shape for metadata
            original_docs, original_metrics = matrix.shape
            
            # Apply transformation
            transformed_matrix = transform_fn(matrix, original_metric_names)
            
            # Record transformation details
            self.applied_transforms.append({
                "type": pattern_type,
                "input_shape": (original_docs, original_metrics),
                "output_shape": transformed_matrix.shape,
                "timestamp": np.datetime64('now')
            })
            
            logger.info(f"Feature transformation '{pattern_type}' applied. New shape: {transformed_matrix.shape}")
            return transformed_matrix
            
        except Exception as e:
            logger.error(f"Error applying transformation '{pattern_type}': {e}")
            # Fallback to identity transform
            identity_result = self._identity_transform(matrix, original_metric_names)
            self.applied_transforms.append({
                "type": "IDENTITY_FALLBACK",
                "input_shape": matrix.shape,
                "output_shape": identity_result.shape,
                "error": str(e),
                "timestamp": np.datetime64('now')
            })
            return identity_result

    def _identity_transform(self, matrix: np.ndarray, metric_names: List[str]) -> np.ndarray:
        """Passes the matrix through unchanged."""
        self.engineered_metric_names = list(metric_names) # No new names
        logger.debug("Applied identity transformation.")
        return matrix

    def _xor_transform(self, matrix: np.ndarray, metric_names: List[str]) -> np.ndarray:
        """Generates features helpful for XOR-like patterns."""
        if matrix.shape[1] < 2:
            logger.warning("XOR transform needs at least 2 metrics. Returning original.")
            self.engineered_metric_names = list(metric_names)
            return matrix
            
        # Assume first two metrics are the primary coordinates for XOR
        m1, m2 = matrix[:, 0], matrix[:, 1]
        
        # Standard XOR-separating features
        product = m1 * m2
        abs_diff = np.abs(m1 - m2)
        
        # Stack original and new features
        result = np.column_stack([matrix, product, abs_diff])
        
        # Update metric names
        self.engineered_metric_names = list(metric_names) + ["xor_product", "xor_abs_diff"]
        logger.debug("Applied XOR feature transformation.")
        return result

    def _radial_transform(self, matrix: np.ndarray, metric_names: List[str]) -> np.ndarray:
        """Generates radial/distance-based features."""
        if matrix.shape[1] < 2:
            logger.warning("Radial transform needs at least 2 metrics. Returning original.")
            self.engineered_metric_names = list(metric_names)
            return matrix

        # Assume first two metrics are X, Y coordinates centered at 0.5
        x, y = matrix[:, 0], matrix[:, 1]
        center_x, center_y = 0.5, 0.5
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        angle = np.arctan2(y - center_y, x - center_x)
        
        result = np.column_stack([matrix, distance, angle])
        self.engineered_metric_names = list(metric_names) + ["radial_distance", "radial_angle"]
        logger.debug("Applied radial feature transformation.")
        return result

    def _product_transform(self, matrix: np.ndarray, metric_names: List[str]) -> np.ndarray:
        """Adds pairwise products of metrics."""
        if matrix.shape[1] < 2:
            logger.warning("Product transform needs at least 2 metrics. Returning original.")
            self.engineered_metric_names = list(metric_names)
            return matrix
            
        cols = matrix.shape[1]
        new_features = []
        new_names = []
        # Simple pairwise products for first few metrics
        for i in range(min(3, cols)):
            for j in range(i+1, min(4, cols)):
                new_features.append(matrix[:, i] * matrix[:, j])
                new_names.append(f"product_{metric_names[i]}_{metric_names[j]}")
        
        if new_features:
            result = np.column_stack([matrix] + new_features)
            self.engineered_metric_names = list(metric_names) + new_names
        else:
            result = matrix
            self.engineered_metric_names = list(metric_names)
            
        logger.debug(f"Applied product feature transformation. Added {len(new_names)} features.")
        return result

    def _difference_transform(self, matrix: np.ndarray, metric_names: List[str]) -> np.ndarray:
        """Adds pairwise absolute differences of metrics."""
        if matrix.shape[1] < 2:
            logger.warning("Difference transform needs at least 2 metrics. Returning original.")
            self.engineered_metric_names = list(metric_names)
            return matrix
            
        cols = matrix.shape[1]
        new_features = []
        new_names = []
        # Simple pairwise differences for first few metrics
        for i in range(min(3, cols)):
            for j in range(i+1, min(4, cols)):
                new_features.append(np.abs(matrix[:, i] - matrix[:, j]))
                new_names.append(f"abs_diff_{metric_names[i]}_{metric_names[j]}")
        
        if new_features:
            result = np.column_stack([matrix] + new_features)
            self.engineered_metric_names = list(metric_names) + new_names
        else:
            result = matrix
            self.engineered_metric_names = list(metric_names)
            
        logger.debug(f"Applied difference feature transformation. Added {len(new_names)} features.")
        return result

    def get_metric_names(self) -> List[str]:
        """Gets the names of the metrics after the last transformation."""
        return self.engineered_metric_names

    def get_transformation_log(self) -> List[Dict[str, Any]]:
        """Gets a log of all applied transformations."""
        return self.applied_transforms

    def clear_log(self):
        """Clears the transformation log."""
        self.applied_transforms.clear()
``n

## File: feature_engineer.py

`python
"""Feature engineering strategies for ZeroModel.

Encapsulates hint-based non-linear feature generation so the core model
remains focused on orchestration.
"""
from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Dict, Callable
import numpy as np

logger = logging.getLogger(__name__)

FeatureResult = Tuple[np.ndarray, List[str]]  # (augmented_matrix, new_metric_names)

class FeatureEngineer:
    """Applies optional non-linear feature transformations based on a hint string."""
    def __init__(self) -> None:
        # Registry pattern for easy extension
        self._strategies: Dict[str, Callable[[np.ndarray, List[str]], FeatureResult]] = {
            'xor': self._xor_features,
            'radial': self._radial_features,
            'auto': self._auto_features,
        }

    # ------------------------ Public API ------------------------
    def apply(self, hint: Optional[str], data: np.ndarray, metric_names: List[str]) -> FeatureResult:
        """Apply feature engineering based on hint.

        Args:
            hint: Optional string ('xor', 'radial', 'auto', ...)
            data: Normalized base matrix [docs x metrics]
            metric_names: Base metric names
        Returns:
            (processed_matrix, effective_metric_names)
        """
        if hint is None:
            logger.debug("No feature engineering hint provided; returning original data.")
            return data, list(metric_names)
        key = hint.lower().strip()
        strategy = self._strategies.get(key)
        if strategy is None:
            logger.warning("Unknown nonlinearity_hint '%s'. No features added.", hint)
            return data, list(metric_names)
        try:
            augmented, new_names = strategy(data, metric_names)
            if augmented is data:  # No change
                return data, list(metric_names)
            return augmented, new_names
        except Exception as e:  # noqa: broad-except - defensive; fallback to original
            logger.error("Feature engineering strategy '%s' failed: %s. Falling back to base data.", key, e)
            return data, list(metric_names)

    # --------------------- Strategy Implementations ---------------------
    def _xor_features(self, data: np.ndarray, names: List[str]) -> FeatureResult:
        if data.shape[1] < 2:
            logger.debug("Not enough metrics for xor features (<2).")
            return data, names
        m1, m2 = data[:, 0], data[:, 1]
        feats = [m1 * m2, np.abs(m1 - m2)]
        feat_names = [f"hint_product_{names[0]}_{names[1]}", f"hint_abs_diff_{names[0]}_{names[1]}"]
        augmented = np.column_stack([data] + feats)
        return augmented, names + feat_names

    def _radial_features(self, data: np.ndarray, names: List[str]) -> FeatureResult:
        if data.shape[1] < 2:
            logger.debug("Not enough metrics for radial features (<2).")
            return data, names
        x, y = data[:, 0], data[:, 1]
        cx = cy = 0.5
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        angle = np.arctan2(y - cy, x - cx)
        feats = [distance, angle]
        feat_names = ["hint_radial_distance", "hint_radial_angle"]
        augmented = np.column_stack([data] + feats)
        return augmented, names + feat_names

    def _auto_features(self, data: np.ndarray, names: List[str]) -> FeatureResult:
        n_orig = len(names)
        if n_orig == 0:
            return data, names
        engineered_features = []
        engineered_names: List[str] = []
        # Pairwise products among first min(3, n_orig)
        n_prod = min(3, n_orig)
        for i in range(n_prod):
            for j in range(i + 1, n_prod):
                if j < data.shape[1]:
                    engineered_features.append(data[:, i] * data[:, j])
                    engineered_names.append(f"auto_product_{names[i]}_{names[j]}")
        # Squares of first min(2, n_orig)
        n_sq = min(2, n_orig)
        for i in range(n_sq):
            if i < data.shape[1]:
                engineered_features.append(data[:, i] ** 2)
                engineered_names.append(f"auto_square_{names[i]}")
        if not engineered_features:
            logger.debug("Auto hint produced no additional features.")
            return data, names
        augmented = np.column_stack([data] + engineered_features)
        logger.info("Auto hint added %d features (expected ~5 when n>=3).", len(engineered_features))
        return augmented, names + engineered_names

__all__ = ["FeatureEngineer"]
``n

## File: hierarchical_edge.py

`python
# zeromodel/hierarchical_edge.py
"""
Hierarchical Edge Device Protocol

This module provides the communication protocol for edge devices
to interact with hierarchical visual policy maps.
"""

import struct
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class HierarchicalEdgeProtocol:
    """
    Protocol for edge devices to interact with hierarchical VPMs.
    
    This implements a minimal protocol that:
    - Works with tiny memory constraints (<25KB)
    - Handles hierarchical navigation
    - Enables zero-model intelligence at the edge
    """
    
    # Protocol version (1 byte)
    PROTOCOL_VERSION = 1
    
    # Message types (1 byte each)
    MSG_TYPE_REQUEST = 0x01
    MSG_TYPE_TILE = 0x02
    MSG_TYPE_DECISION = 0x03
    MSG_TYPE_ZOOM = 0x04 # Request or indication to change hierarchical level
    MSG_TYPE_ERROR = 0x05
    
    # Maximum tile size (for memory constraints)
    MAX_TILE_WIDTH = 3
    MAX_TILE_HEIGHT = 3
    
    @staticmethod
    def create_request(task_description: str, level: int = 0) -> bytes:
        """
        Create a request message for the edge proxy.
        
        Args:
            task_description: Natural language task description
            level: Starting hierarchical level (0 = most abstract)
        
        Returns:
            Binary request message
            
        Raises:
            ValueError: If task_description is None or level is invalid.
        """
        logger.debug(f"Creating request for task: '{task_description}', starting level: {level}")
        if task_description is None:
             error_msg = "Task description cannot be None"
             logger.error(error_msg)
             raise ValueError(error_msg)
        if not (0 <= level <= 255): # Assuming 1-byte level
             error_msg = f"Level must be between 0 and 255, got {level}"
             logger.error(error_msg)
             raise ValueError(error_msg)

        task_bytes = task_description.encode('utf-8')
        original_len = len(task_bytes)
        max_task_len = 253 # 256 - 3 bytes for version, type, level
        if original_len > max_task_len:
            logger.warning(f"Task description ({original_len} bytes) exceeds {max_task_len} bytes, truncating.")
            task_bytes = task_bytes[:max_task_len] # Truncate if too long
        elif original_len == 0:
             logger.info("Creating request with an empty task description.")

        message = struct.pack(
            f"BBBB{len(task_bytes)}s",
            HierarchicalEdgeProtocol.PROTOCOL_VERSION,
            HierarchicalEdgeProtocol.MSG_TYPE_REQUEST,
            level & 0xFF, # Ensure level fits in 1 byte
            len(task_bytes),
            task_bytes
        )
        logger.debug(f"Request message created, size: {len(message)} bytes")
        return message
    
    @staticmethod
    def parse_tile(tile_data: bytes) -> Tuple[int, int, int, int, int, bytes]:
        """
        Parse a tile message from the proxy.
        
        Args:
            tile_data: Binary tile data 
                          Format: [level][width][height][x_offset][y_offset][...pixels...]
        
        Returns:
            Tuple containing (level, width, height, x_offset, y_offset, pixels_data)
            
        Raises:
            ValueError: If tile_data is invalid (e.g., too short).
        """
        header_size = 5 # level + width + height + x_offset + y_offset
        logger.debug(f"Parsing tile data, received size: {len(tile_data)} bytes")
        if not tile_data or len(tile_data) < header_size:
            error_msg = f"Invalid tile format: data too short (expected >= {header_size} bytes, got {len(tile_data)})."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        level = tile_data[0]
        width = tile_data[1]
        height = tile_data[2]
        x_offset = tile_data[3]
        y_offset = tile_data[4]
        pixels_data = tile_data[header_size:] # Remaining bytes are pixel data
        
        logger.debug(f"Parsed tile header: level={level}, width={width}, height={height}, x_offset={x_offset}, y_offset={y_offset}")

        # Validate dimensions strictly based on protocol limits
        if width > HierarchicalEdgeProtocol.MAX_TILE_WIDTH:
            logger.warning(f"Tile width ({width}) exceeds MAX_TILE_WIDTH ({HierarchicalEdgeProtocol.MAX_TILE_WIDTH}). Processing with max width.")
            width = HierarchicalEdgeProtocol.MAX_TILE_WIDTH # Original logic commented for clarity
        if height > HierarchicalEdgeProtocol.MAX_TILE_HEIGHT:
            logger.warning(f"Tile height ({height}) exceeds MAX_TILE_HEIGHT ({HierarchicalEdgeProtocol.MAX_TILE_HEIGHT}). Processing with max height.")
            height = HierarchicalEdgeProtocol.MAX_TILE_HEIGHT # Original logic commented for clarity

        result = (level, width, height, x_offset, y_offset, pixels_data)
        logger.debug(f"Tile parsed successfully: {result[:5]} + {len(pixels_data)} pixel bytes")
        return result
    
    @staticmethod
    def make_decision(tile_message_data: bytes) -> bytes:
        """
        Process a tile message and make a decision.
        Assumes the tile_message_data includes the full tile message structure
        starting with level, width, height, x_offset, y_offset.

        Args:
            tile_message_data: Binary tile message data (payload).
        
        Returns:
            Binary decision message (MSG_TYPE_DECISION).
        """
        logger.debug(f"Making decision based on tile data ({len(tile_message_data)} bytes)")
        try:
            level, width, height, x_offset, y_offset, pixels_data = HierarchicalEdgeProtocol.parse_tile(tile_message_data)
        except ValueError as e:
            logger.error(f"Failed to parse tile for decision making: {e}")
            # Return an error message instead of raising
            return HierarchicalEdgeProtocol.create_error(11, f"Tile Parse Error: {str(e)}")

        # Simple decision logic: check top-left pixel value (R channel)
        # Check if we have at least one pixel's R channel data
        # Pixel data is [R0, G0, B0, R1, G1, B1, ...]
        # Top-left pixel R is at index 0
        is_relevant = 0 # Default to not relevant
        if len(pixels_data) > 0:
            top_left_r_value = pixels_data[0]
            # Decision: is this "dark enough" to be relevant?
            is_relevant = 1 if top_left_r_value < 128 else 0
            logger.debug(f"Top-left pixel R value: {top_left_r_value}, Decision (is_relevant): {is_relevant}")
        else:
             logger.warning("Tile pixel data is empty. Defaulting decision to not relevant.")
             # is_relevant remains 0

        # Create decision message
        # Format: [version][type][level][decision][reserved]
        decision_message = struct.pack("BBBBB",
                          HierarchicalEdgeProtocol.PROTOCOL_VERSION,
                          HierarchicalEdgeProtocol.MSG_TYPE_DECISION,
                          level & 0xFF, # Ensure level fits in 1 byte
                          is_relevant,
                          0)  # Reserved byte
        logger.info(f"Decision made at level {level}: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")
        logger.debug(f"Decision message created, size: {len(decision_message)} bytes")
        return decision_message
    
    @staticmethod
    def request_zoom(tile_message_data: bytes, direction: str = "in") -> bytes:
        """
        Request to zoom in or out from the level indicated in the current tile.

        Args:
            tile_message_data: Binary tile message data (payload) to get the current level.
            direction: "in" (towards detail) or "out" (towards abstraction).
        
        Returns:
            Binary zoom request message (MSG_TYPE_ZOOM).
            
        Raises:
            ValueError: If direction is invalid or tile data cannot be parsed.
        """
        logger.debug(f"Creating zoom request: direction='{direction}'")
        if direction not in ["in", "out"]:
            error_msg = f"Invalid zoom direction '{direction}'. Must be 'in' or 'out'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            current_level, _, _, _, _, _ = HierarchicalEdgeProtocol.parse_tile(tile_message_data)
        except ValueError as e:
            logger.error(f"Failed to parse tile to determine current level for zoom: {e}")
            # Return an error message instead of raising
            return HierarchicalEdgeProtocol.create_error(12, f"Zoom Parse Error: {str(e)}")

        # Determine new level
        # Assuming Level 0 = Most Abstract, Level 2 = Most Detailed (common convention)
        new_level = current_level
        if direction == "in":
            # Zooming in means going to a more detailed level (higher number)
            new_level = min(2, current_level + 1) # Assuming max 3 levels (0, 1, 2)
            logger.debug(f"Zoom IN requested: Level {current_level} -> {new_level}")
        elif direction == "out":
            # Zooming out means going to a more abstract level (lower number)
            new_level = max(0, current_level - 1)
            logger.debug(f"Zoom OUT requested: Level {current_level} -> {new_level}")

        # Create zoom message
        # Format: [version][type][current_level][new_level]
        zoom_message = struct.pack("BBBB",
                          HierarchicalEdgeProtocol.PROTOCOL_VERSION,
                          HierarchicalEdgeProtocol.MSG_TYPE_ZOOM,
                          current_level & 0xFF, # Ensure level fits in 1 byte
                          new_level & 0xFF      # Ensure level fits in 1 byte
                         )
        logger.debug(f"Zoom request message created, size: {len(zoom_message)} bytes")
        return zoom_message
    
    @staticmethod
    def create_error(code: int, message: str = "") -> bytes:
        """
        Create an error message.
        
        Args:
            code: Error code (1 byte recommended)
            message: Optional error message (max ~252 chars)
        
        Returns:
            Binary error message (MSG_TYPE_ERROR)
        """
        logger.debug(f"Creating error message: code={code}, message='{message}'")
        if code < 0 or code > 255: # Assuming 1-byte code
             logger.warning(f"Error code {code} is outside typical 0-255 range for 1-byte field.")
        # Max message length considering 3 header bytes (Version, Type, Code)
        max_msg_len = 253 
        msg_bytes = message.encode('utf-8')[:max_msg_len] 
        
        error_message = struct.pack(
            f"BBB{len(msg_bytes)}s",
            HierarchicalEdgeProtocol.PROTOCOL_VERSION,
            HierarchicalEdgeProtocol.MSG_TYPE_ERROR,
            code & 0xFF, # Ensure code fits in 1 byte
            msg_bytes
        )
        logger.debug(f"Error message created, size: {len(error_message)} bytes")
        return error_message

# --- Configure logging for this module ---
# This should ideally be done once in your main application.
# Placing it here ensures logs appear if this script is run directly.
if __name__ == "__main__":
    # Example configuration - adjust as needed for your application
    logging.basicConfig(
        level=logging.DEBUG, # Adjust level (DEBUG, INFO, WARNING, ERROR)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() # Output to console
            # logging.FileHandler('hierarchical_edge_protocol.log') # Optional: Output to a file
        ]
    )
    logger.info("Logging configured for HierarchicalEdgeProtocol module.")
``n

## File: hierarchical.py

`python
# zeromodel/hierarchical.py
"""
Hierarchical Visual Policy Map (HVPM) implementation with DuckDB SQL support.

This module provides the HierarchicalVPM class for creating multi-level
decision maps, enabling efficient processing across different resource
constraints (edge vs. cloud).
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# Import the core ZeroModel
# Make sure the path is correct based on your package structure
from .core import ZeroModel

# Create a logger for this module
logger = logging.getLogger(__name__)

class HierarchicalVPM:
    """
    Hierarchical Visual Policy Map (HVPM) implementation with DuckDB SQL support.

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
            metric_names: Names of all metrics being tracked.
            num_levels: Number of hierarchical levels (default 3).
            zoom_factor: Zoom factor between levels (default 3).
            precision: Bit precision for encoding (4-16).
            
        Raises:
            ValueError: If inputs are invalid (e.g., num_levels <= 0).
        """
        logger.debug(f"Initializing HierarchicalVPM with metrics: {metric_names}, levels: {num_levels}, zoom: {zoom_factor}")
        if num_levels <= 0:
            error_msg = f"num_levels must be positive, got {num_levels}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if zoom_factor <= 1:
             error_msg = f"zoom_factor must be greater than 1, got {zoom_factor}."
             logger.error(error_msg)
             raise ValueError(error_msg)
        if precision < 4 or precision > 16:
             logger.warning(f"Precision {precision} is outside recommended range 4-16. Clamping.")
             precision = max(4, min(16, precision))

        self.metric_names = list(metric_names)
        self.num_levels = num_levels
        self.zoom_factor = zoom_factor
        self.precision = precision
        self.levels: List[Dict[str, Any]] = [] # Store level data
        self.metadata: Dict[str, Any] = {
            "version": "1.0",
            "temporal_axis": False,
            "levels": num_levels,
            "zoom_factor": zoom_factor,
            "metric_names": self.metric_names # Add for reference
        }
        logger.info(f"HierarchicalVPM initialized with {num_levels} levels.")

    def process(self, score_matrix: np.ndarray, task: str):
        """
        Process score matrix into hierarchical visual policy maps using the new prepare() method.

        Args:
            score_matrix: 2D array of shape [documents × metrics].
            task: SQL query defining the task.

        Raises:
            ValueError: If inputs are invalid or processing fails.
        """
        logger.info(f"Processing hierarchical VPM for task: '{task}' with data shape {score_matrix.shape}")
        if score_matrix is None:
            error_msg = "score_matrix cannot be None."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.ndim != 2:
            error_msg = f"score_matrix must be 2D, got shape {score_matrix.shape}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not task:
             logger.warning("Task string is empty. Proceeding with empty task.")
             # Consider if an empty task is valid or should raise an error
             # For now, let ZeroModel.prepare handle it. 

        # Update metadata
        self.metadata["task"] = task
        self.metadata["documents"] = score_matrix.shape[0]
        self.metadata["metrics"] = score_matrix.shape[1]
        logger.debug(f"Updated metadata: documents={score_matrix.shape[0]}, metrics={score_matrix.shape[1]}")

        # Clear existing levels
        self.levels = []
        logger.debug("Cleared existing levels.")

        # --- Level Creation using prepare() ---
        # Create ZeroModel instance for the base (highest detail) level
        logger.debug("Creating base ZeroModel instance.")
        base_zeromodel = ZeroModel(self.metric_names, precision=self.precision)
        
        # --- Use prepare() instead of set_sql_task() + process() ---
        # Prepare the base level ZeroModel with data and task in one step
        try:
            base_zeromodel.prepare(score_matrix, task) # <-- CHANGED HERE
            logger.debug("Base ZeroModel prepared with data and task.")
        except Exception as e:
            logger.error(f"Failed to prepare base ZeroModel: {e}")
            raise ValueError(f"Error preparing base ZeroModel level: {e}") from e
        # --- End of change ---

        # Create base level (Level N-1: Full detail, where N is num_levels)
        base_level_index = self.num_levels - 1
        base_level = self._create_base_level(base_zeromodel, score_matrix) # Pass original data for metadata if needed
        # Store level data. Levels list will be ordered [Level 0, Level 1, ..., Level N-1]
        self.levels.append(base_level) 
        logger.debug(f"Base level ({base_level_index}) created and added.")

        # Create higher levels (Level N-2, Level N-3, ..., Level 0)
        current_data = score_matrix
        # Iterate from level 1 up to num_levels-1 (0-indexed levels)
        # Level 0 is the most abstract (smallest)
        for relative_level in range(1, self.num_levels): 
            level_index = self.num_levels - 1 - relative_level # Absolute level index (0 to N-2)
            logger.debug(f"Creating level {level_index} (relative {relative_level})")
            
            # Calculate target dimensions for clustering
            target_docs = max(1, int(np.ceil(current_data.shape[0] / self.zoom_factor)))
            target_metrics = max(1, int(np.ceil(current_data.shape[1] / self.zoom_factor)))
            logger.debug(f"Clustering to target size: docs={target_docs}, metrics={target_metrics}")

            # Perform clustering
            clustered_data = self._cluster_data(current_data, target_docs, target_metrics)
            logger.debug(f"Clustered data shape: {clustered_data.shape}")

            # Create the level data structure
            # Pass the clustered data and the base task
            level_data = self._create_level(clustered_data, task, level_index) # <-- CHANGED: Pass task, not base_task_config
            # Insert at the beginning of the list to maintain order [L0, L1, L2, ...]
            self.levels.insert(0, level_data) 
            logger.debug(f"Level {level_index} created and added.")

            # Update data for next iteration (cluster the clustered data)
            current_data = clustered_data

        logger.info("Hierarchical VPM processing complete.")


    def _cluster_data(self, data: np.ndarray, num_docs: int, num_metrics: int) -> np.ndarray:
        """
        Cluster data for higher-level (more abstract) views.

        Args:
             Input data matrix of shape [docs, metrics].
            num_docs: Target number of document clusters.
            num_metrics: Target number of metric clusters.

        Returns:
            Clustered data matrix of shape [num_docs, num_metrics].
        """
        logger.debug(f"Clustering data of shape {data.shape} to {num_docs} docs x {num_metrics} metrics.")
        docs, metrics = data.shape

        # Handle edge case where we have fewer items than target clusters
        num_docs = min(num_docs, docs)
        num_metrics = min(num_metrics, metrics)
        logger.debug(f"Adjusted clustering targets: docs={num_docs}, metrics={num_metrics}")

        if docs == 0 or metrics == 0:
            logger.warning("Input data for clustering is empty. Returning empty array.")
            return np.empty((num_docs, num_metrics)) # Or np.zeros?

        # --- Document Clustering ---
        doc_clusters = []
        for i in range(num_docs):
            # Calculate slice indices for this cluster
            start_idx = int(i * docs / num_docs) # Use int() or floor?
            end_idx = int((i + 1) * docs / num_docs)
            # Ensure start < end to avoid empty slices
            end_idx = max(start_idx + 1, end_idx) 
            # Ensure end_idx doesn't exceed the array
            end_idx = min(end_idx, docs)

            if start_idx < end_idx:
                # Average the rows (documents) in this slice
                cluster_mean = np.mean(data[start_idx:end_idx], axis=0)
                doc_clusters.append(cluster_mean)
                logger.debug(f"Document cluster {i}: rows {start_idx}-{end_idx-1} -> mean")
            else:
                # Fallback if slice is somehow invalid (should be rare with adjustments)
                logger.warning(f"Invalid document slice [{start_idx}:{end_idx}] for cluster {i}. Using first row.")
                fallback_idx = min(start_idx, docs - 1)
                doc_clusters.append(data[fallback_idx])

        if not doc_clusters:
             logger.error("No document clusters created. This should not happen.")
             # Return an array of zeros or handle error?
             clustered_docs = np.zeros((num_docs, metrics))
        else:
             clustered_docs = np.array(doc_clusters)
        logger.debug(f"Document clustering resulted in shape: {clustered_docs.shape}")

        # --- Metric Clustering ---
        # Note: The original code clustered metrics based on the *already clustered docs*.
        # This means metric clustering happens on the `clustered_docs` array.
        metric_clusters = []
        effective_metrics = clustered_docs.shape[1] # Use shape of clustered_docs
        for j in range(num_metrics):
            start_idx = int(j * effective_metrics / num_metrics)
            end_idx = int((j + 1) * effective_metrics / num_metrics)
            end_idx = max(start_idx + 1, end_idx)
            end_idx = min(end_idx, effective_metrics)

            if start_idx < end_idx:
                # Average the columns (metrics) in this slice
                # np.mean(..., axis=1) averages along columns, resulting in a 1D array per slice
                # np.column_stack then combines these 1D arrays into columns
                cluster_mean = np.mean(clustered_docs[:, start_idx:end_idx], axis=1)
                metric_clusters.append(cluster_mean)
                logger.debug(f"Metric cluster {j}: cols {start_idx}-{end_idx-1} -> mean")
            else:
                 logger.warning(f"Invalid metric slice [{start_idx}:{end_idx}] for cluster {j}. Using first col.")
                 fallback_idx = min(start_idx, effective_metrics - 1)
                 metric_clusters.append(clustered_docs[:, fallback_idx])

        if not metric_clusters:
             logger.error("No metric clusters created. This should not happen.")
             # Return an array matching the doc cluster shape
             final_clustered_data = np.zeros((clustered_docs.shape[0], num_metrics))
        else:
             # np.column_stack takes a sequence of 1-D arrays and stacks them as columns
             final_clustered_data = np.column_stack(metric_clusters)
        logger.debug(f"Metric clustering resulted in shape: {final_clustered_data.shape}")
        
        return final_clustered_data

    def _create_base_level(self, zeromodel: ZeroModel, score_matrix: np.ndarray) -> Dict[str, Any]:
        """Create the base level (highest detail) using the prepared ZeroModel."""
        level_index = self.num_levels - 1
        logger.debug(f"Creating base level data structure (Level {level_index}).")
        # The zeromodel passed in should already be prepared.
        # We can now safely call encode(), get_metadata() etc.
        try:
            vpm_image = zeromodel.encode() # Uses zeromodel.sorted_matrix
            logger.debug(f"Encoded base level VPM image of shape {vpm_image.shape}.")
        except Exception as e:
            logger.error(f"Failed to encode VPM for base level: {e}")
            # Depending on requirements, re-raise or handle?
            raise # Re-raise 

        base_level_data = {
            "level": level_index,
            "type": "base",
            "zeromodel": zeromodel, # Store the prepared ZeroModel instance
            "vpm": vpm_image,
            "metadata": {
                "documents": score_matrix.shape[0], # Use original shape for metadata
                "metrics": score_matrix.shape[1],
                "sorted_docs": zeromodel.doc_order.tolist() if zeromodel.doc_order is not None else [],
                "sorted_metrics": zeromodel.metric_order.tolist() if zeromodel.metric_order is not None else [],
                # Add wavelet info if applicable
                # "wavelet_level": 0 
            }
        }
        logger.debug(f"Base level data structure created.")
        return base_level_data

    def _create_level(self,
                      approx_data: np.ndarray, # Renamed from clustered_data for clarity
                      task: str,              # Pass the task string
                      level_index: int) -> Dict[str, Any]: # Absolute level index (0, 1, 2, ...)
        """Create a higher-level (more abstract) view using prepare()."""
        logger.debug(f"Creating level data structure (Level {level_index}).")
        num_metrics_in_clustered_data = approx_data.shape[1]
        # Create a simplified metric set for this level
        level_metrics = [
            f"cluster_{i}" for i in range(num_metrics_in_clustered_data)
        ]
        logger.debug(f"Generated level metric names: {level_metrics}")

        # --- Create and Prepare ZeroModel for this level ---
        # Process with simplified metrics using a new ZeroModel instance
        level_zeromodel = ZeroModel(level_metrics, precision=self.precision)
        logger.debug("Created ZeroModel instance for this level.")

        # --- Use prepare() for this level's ZeroModel ---
        # Prepare the level's ZeroModel with its clustered data and the task
        # The task might need adaptation (see notes below), but for now, pass it as is
        # or use a default sorting strategy for clustered levels.
        try:
            # Option 1: Use the same task (might not be ideal for clustered data)
            # level_zeromodel.prepare(approx_data, task) 
            
            # Option 2: Use a simple default task for higher levels (often better)
            # E.g., sort by the first "cluster" metric which represents the aggregated importance
            # of the original metrics that formed this cluster.
            default_level_task = f"SELECT * FROM virtual_index ORDER BY {level_metrics[0]} DESC"
            level_zeromodel.prepare(approx_data, default_level_task) # <-- CHANGED HERE
            logger.debug(f"Prepared level {level_index} ZeroModel with default task: {default_level_task}")
        except Exception as e:
             logger.error(f"Failed to prepare ZeroModel for level {level_index}: {e}")
             raise ValueError(f"Error preparing ZeroModel for level {level_index}: {e}") from e
        # --- End of change ---

        try:
            level_vpm = level_zeromodel.encode() # Uses level_zeromodel.sorted_matrix
            logger.debug(f"Encoded level {level_index} VPM image of shape {level_vpm.shape}.")
        except Exception as e:
            logger.error(f"Failed to encode VPM for level {level_index}: {e}")
            raise # Re-raise

        level_data = {
            "level": level_index,
            "type": "clustered", # Or "approximated"
            "zeromodel": level_zeromodel, # Store the prepared ZeroModel instance
            "vpm": level_vpm,
            "metadata": {
                "documents": approx_data.shape[0],
                "metrics": approx_data.shape[1],
                "sorted_docs": level_zeromodel.doc_order.tolist() if level_zeromodel.doc_order is not None else [],
                "sorted_metrics": level_zeromodel.metric_order.tolist() if level_zeromodel.metric_order is not None else [],
                # "wavelet_level": level_index # If wavelets are used
            }
        }
        logger.debug(f"Level {level_index} data structure created.")
        return level_data

    def get_level(self, level_index: int) -> Dict[str, Any]:
        """
        Get data for a specific level.

        Args:
            level_index: The hierarchical level index (0 = most abstract).

        Returns:
            Dictionary containing level data.
            
        Raises:
            ValueError: If level_index is invalid.
        """
        logger.debug(f"Retrieving data for level {level_index}.")
        if not (0 <= level_index < self.num_levels):
            error_msg = f"Level index must be between 0 and {self.num_levels - 1}, got {level_index}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # self.levels is ordered [Level 0, Level 1, ..., Level N-1]
        # So, index directly corresponds to level_index
        level_data = self.levels[level_index]
        logger.debug(f"Level {level_index} data retrieved.")
        return level_data

    def get_tile(self,
                 level_index: int,
                 x: int = 0,
                 y: int = 0,
                 width: int = 3,
                 height: int = 3) -> bytes:
        """
        Get a tile from a specific level for edge devices.

        Args:
            level_index: Hierarchical level (0 = most abstract).
            x, y: Top-left corner of tile (currently ignored, gets top-left).
            width, height: Dimensions of tile.

        Returns:
            Compact byte representation of the tile.
            
        Raises:
            ValueError: If level_index is invalid or tile parameters are bad.
        """
        logger.debug(f"Getting tile from level {level_index} (x={x}, y={y}, w={width}, h={height}).")
        # Note: x, y are currently ignored by ZeroModel.get_critical_tile, which always gets top-left.
        # Future enhancement could involve slicing the VPM image based on x,y.
        
        level_data = self.get_level(level_index)
        zeromodel_instance: ZeroModel = level_data["zeromodel"]
        
        # Determine tile size - use the larger of width/height, or a specific logic
        # The original code used max(width, height). Let's stick to that for now,
        # though it might be better to get a width x height tile.
        # get_critical_tile currently only takes one size parameter (square tile).
        tile_size = max(width, height) 
        logger.debug(f"Determined tile size for request: {tile_size}")

        # Get critical tile (currently always top-left)
        tile_bytes = zeromodel_instance.get_critical_tile(tile_size=tile_size)
        logger.info(f"Tile retrieved from level {level_index}. Size: {len(tile_bytes)} bytes.")
        return tile_bytes

    def get_decision(self, level_index: Optional[int] = None) -> Tuple[int, int, float]:
        """
        Get top decision from a specific level.

        Args:
            level_index: The level to get the decision from.
                         If None, gets the decision from the most detailed level (highest index).

        Returns:
            Tuple of (level_index, document_index, relevance_score).
            
        Raises:
            ValueError: If level_index is invalid.
        """
        if level_index is None:
            level_index = self.num_levels - 1 # Default to most detailed level
        logger.debug(f"Getting decision from level {level_index}.")

        level_data = self.get_level(level_index)
        zeromodel_instance: ZeroModel = level_data["zeromodel"]
        doc_idx, relevance = zeromodel_instance.get_decision()
        logger.info(f"Decision from level {level_index}: Doc {doc_idx}, Relevance {relevance:.4f}.")
        return (level_index, doc_idx, relevance)

    def zoom_in(self, level_index: int, doc_idx: int, metric_idx: int) -> int:
        """
        Determine the next level to zoom into based on current selection.

        Args:
            level_index: Current hierarchical level.
            doc_idx: Selected document index (relative to the current level's data).
            metric_idx: Selected metric index (relative to the current level's data).

        Returns:
            Next level index to zoom into (level_index+1, or same level if already at base).
        """
        logger.debug(f"Calculating zoom-in from level {level_index}, doc {doc_idx}, metric {metric_idx}.")
        if level_index >= self.num_levels - 1:
            logger.info("Already at the most detailed level. Staying at current level.")
            return level_index  # Already at most detailed level
        next_level = level_index + 1
        logger.debug(f"Zooming in to level {next_level}.")
        return next_level

    def get_metadata(self) -> Dict[str, Any]:
        """Get complete metadata for the hierarchical map."""
        logger.debug("Retrieving hierarchical VPM metadata.")
        # Ensure levels metadata is current
        level_info = []
        for level_data in self.levels:
            level_info.append({
                "level": level_data["level"],
                "type": level_data["type"],
                "documents": level_data["metadata"]["documents"],
                "metrics": level_data["metadata"]["metrics"]
            })
        self.metadata["level_details"] = level_info
        logger.debug("Hierarchical VPM metadata retrieved.")
        return self.metadata
``n

## File: metadata.py

`python
# zeromodel/metadata.py
"""
Compact Metadata Handling

This module provides functions for encoding and decoding metadata in a compact
binary format that survives image processing operations. This is critical
for the self-describing nature of zeromodel maps.
"""

import logging
from typing import Dict, List

# Create a logger for this module
logger = logging.getLogger(__name__)

def encode_metadata(task_weights: Dict[str, float], 
                   metric_names: List[str],
                   version: int = 1) -> bytes:
    """
    Encode metadata into compact binary format (<100 bytes).
    
    The encoded metadata includes:
    - A version number
    - A simple hash representing the task (based on metric names and weights)
    - The relative importance of each metric according to task_weights
    
    Args:
        task_weights: A dictionary mapping metric names (from metric_names) 
                      to their weights (floats, typically 0.0 to 1.0).
        metric_names: A list of all metric names. The order is significant.
        version: Metadata format version (integer).
    
    Returns:
        bytes: Compact binary representation of the metadata.
        
    Raises:
        ValueError: If inputs are invalid (e.g., None, negative version).
    """
    logger.debug(f"Encoding metadata: version={version}, metrics={len(metric_names)}")
    if task_weights is None:
        logger.error("task_weights cannot be None")
        raise ValueError("task_weights cannot be None")
    if metric_names is None:
        logger.error("metric_names cannot be None")
        raise ValueError("metric_names cannot be None")
    if version < 0:
        logger.error(f"Version must be non-negative, got {version}")
        raise ValueError(f"Version must be non-negative, got {version}")

    metadata = bytearray()

    # 1. Version (1 byte)
    metadata.append(version & 0xFF) # Ensure fits in 1 byte
    logger.debug(f"Encoded version: {version & 0xFF}")

    # 2. Task ID hash (4 bytes)
    task_hash = 0
    # Iterate through metric_names to ensure consistent order and only consider relevant metrics
    for metric in metric_names: 
        weight = task_weights.get(metric, 0.0)
        if weight > 0:
            # Simple hash based on metric name and weight
            # hash() can return negative values, ensure positive for XOR
            name_hash = hash(metric) & 0xFFFFFFFFFFFFFFFF # Treat as unsigned 64-bit if needed, or just use abs
            weight_int = int(weight * 1000) # Scale weight for hashing, using int for consistency
            task_hash ^= (abs(name_hash) ^ weight_int) # Use abs to handle negative hash
    task_hash &= 0xFFFFFFFF  # Keep as 32-bit unsigned
    metadata.extend(task_hash.to_bytes(4, 'big'))
    logger.debug(f"Encoded task hash: {task_hash:#010x}")

    # 3. Metric importance (4 bits per metric, 2 metrics per byte)
    # Assumes metric_names provides the order.
    # Pads with 0 (importance 0) if metric_names has an odd count.
    for i in range(0, len(metric_names), 2):
        byte_val = 0
        # Handle first metric in the pair (high 4 bits)
        if i < len(metric_names):
            metric1 = metric_names[i]
            # Get weight, defaulting to 0 if not found or if weight is None
            raw_weight1 = task_weights.get(metric1, 0.0) 
            # Clamp raw weight to [0.0, 1.0] to be safe
            clamped_weight1 = max(0.0, min(1.0, raw_weight1))
            # Scale to 0-15 integer
            weight_val1 = int(clamped_weight1 * 15) 
            byte_val |= (weight_val1 & 0x0F) << 4 # Mask to 4 bits, shift to high nibble
            logger.debug(f"Mapped metric '{metric1}' (weight {raw_weight1}) to nibble {weight_val1 & 0x0F}")

        # Handle second metric in the pair (low 4 bits)
        if i + 1 < len(metric_names):
            metric2 = metric_names[i+1]
            raw_weight2 = task_weights.get(metric2, 0.0)
            clamped_weight2 = max(0.0, min(1.0, raw_weight2))
            weight_val2 = int(clamped_weight2 * 15)
            byte_val |= (weight_val2 & 0x0F) # Mask to 4 bits, place in low nibble
            logger.debug(f"Mapped metric '{metric2}' (weight {raw_weight2}) to nibble {weight_val2 & 0x0F}")
            
        metadata.append(byte_val & 0xFF) # Ensure byte_val fits in a byte
    
    result_bytes = bytes(metadata)
    logger.info(f"Metadata encoded successfully. Size: {len(result_bytes)} bytes")
    return result_bytes

def decode_metadata(metadata_bytes: bytes, 
                  metric_names: List[str]) -> Dict[str, float]:
    """
    Decode compact binary metadata back to task weights.
    
    Args:
        metadata_bytes: Binary metadata produced by encode_metadata.
        metric_names: The list of metric names, used to map decoded weights back.
    
    Returns:
        Dict[str, float]: A dictionary mapping metric names to their decoded weights.
    """
    logger.debug(f"Decoding metadata: size={len(metadata_bytes)} bytes, expected metrics={len(metric_names)}")
    if not metadata_bytes:
        logger.warning("Empty metadata bytes provided. Returning default weights.")
        return {m: 0.5 for m in metric_names}
    if metric_names is None:
         logger.error("metric_names cannot be None for decoding.")
         # Returning empty dict might be better than defaulting here, but matching original logic somewhat.
         return {} 

    if len(metadata_bytes) < 5:
        # Not enough data for header, return defaults
        logger.warning("Metadata too short (<5 bytes). Returning default weights.")
        return {m: 0.5 for m in metric_names}
    
    version = metadata_bytes[0]
    # Optional: Check version if multiple versions are expected in the future
    if version != 1: # Assuming version 1 is expected for now
         logger.info(f"Decoded metadata version: {version}. Expected version 1. Proceeding.")
    else:
         logger.debug(f"Decoded metadata version: {version}")

    try:
        task_hash = int.from_bytes(metadata_bytes[1:5], 'big')
        logger.debug(f"Decoded task hash: {task_hash:#010x}")
    except (IndexError, ValueError) as e: # Catch potential errors in slicing/conversion
         logger.error(f"Error decoding task hash from metadata: {e}. Using default weights.")
         return {m: 0.5 for m in metric_names}

    weights = {}
    # Iterate through metric_names to assign weights in the correct order
    for i, metric in enumerate(metric_names):
        byte_idx = 5 + (i // 2) # Calculate which byte contains this metric's data
        if byte_idx >= len(metadata_bytes):
            # Ran out of metadata bytes, assign default weight
            logger.warning(f"Metadata exhausted before weight for metric '{metric}' (index {i}). Assigning default 0.5.")
            weights[metric] = 0.5
            continue
            
        # Determine if this metric's data is in the high or low nibble
        shift = 4 if i % 2 == 0 else 0 # High nibble (4 bits shifted) or low nibble (0 bits shifted)
        # Extract the 4-bit value
        weight_val_nibble = (metadata_bytes[byte_idx] >> shift) & 0x0F 
        # Convert 4-bit value (0-15) back to float weight (0.0-1.0)
        weight = weight_val_nibble / 15.0 
        weights[metric] = weight
        logger.debug(f"Decoded metric '{metric}': nibble {weight_val_nibble:#x} -> weight {weight:.4f}")
    
    logger.info("Metadata decoded successfully.")
    return weights

def get_metadata_size(metric_count: int) -> int:
    """
    Calculate approximate metadata size in bytes.
    
    The size consists of:
    - 1 byte for the version
    - 4 bytes for the task hash
    - N bytes for metric importance (2 metrics per byte)
    
    Args:
        metric_count: Number of metrics
    
    Returns:
        int: Estimated total metadata size in bytes.
    """
    if metric_count < 0:
        logger.warning(f"Negative metric_count {metric_count} provided. Returning size for 0 metrics.")
        metric_count = 0
    # 5 bytes header (version + task_hash) + ceiling division for metric bytes
    size = 5 + (metric_count + 1) // 2 
    logger.debug(f"Calculated metadata size for {metric_count} metrics: {size} bytes")
    return size
``n

## File: normalizer.py

`python
# zeromodel/normalizer.py
"""
Dynamic Range Adaptation

This module provides the DynamicNormalizer class which handles normalization
of scores to handle value drift over time. This is critical for long-term
viability of the zeromodel system as score distributions may change.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

class DynamicNormalizer:
    """
    Handles dynamic normalization of scores to handle value drift over time.
    
    This is critical because:
    - Score ranges may change as policies improve
    - New documents may have scores outside previous ranges
    - Normalization must be consistent across time
    
    The normalizer tracks min/max values for each metric and updates them
    incrementally as new data arrives using exponential smoothing.
    """
    
    def __init__(self, metric_names: List[str], alpha: float = 0.1, *, allow_non_finite: bool = False):
        """
        Initialize the normalizer.
        
        Args:
            metric_names: Names of all metrics being tracked. Order is preserved.
            alpha: Smoothing factor for updating min/max (0.0-1.0).
                   Lower values mean slower adaptation to changes,
                   higher values mean faster adaptation.
                   
        Raises:
            ValueError: If metric_names is None/empty, or alpha is not between 0 and 1.
        """
        logger.debug(f"Initializing DynamicNormalizer with metrics: {metric_names}, alpha: {alpha}")
        if not metric_names:
            error_msg = "metric_names list cannot be None or empty."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not (0.0 <= alpha <= 1.0):
            error_msg = f"Alpha must be between 0.0 and 1.0, got {alpha}."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.metric_names = list(metric_names) # Ensure it's a list
        self.alpha = float(alpha)  # Smoothing factor, ensure float
        self.allow_non_finite = allow_non_finite
        # Initialize with values that will be updated on first data
        self.min_vals = {m: float('inf') for m in self.metric_names}
        self.max_vals = {m: float('-inf') for m in self.metric_names}
        logger.info(f"DynamicNormalizer initialized for {len(self.metric_names)} metrics.")
    
    def update(self, score_matrix: np.ndarray) -> None:
        """
        Update min/max values based on new data using exponential smoothing.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics].
                          Metrics must correspond to self.metric_names in order.
                          
        Raises:
            ValueError: If score_matrix is None, not 2D, or columns don't match metric_names.
        """
        if score_matrix is None:
            error_msg = "score_matrix cannot be None."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.ndim != 2:
            error_msg = f"score_matrix must be 2D, got {score_matrix.ndim}D shape {score_matrix.shape}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.shape[1] != len(self.metric_names):
            error_msg = (f"score_matrix column count ({score_matrix.shape[1]}) "
                         f"must match metric_names count ({len(self.metric_names)}).")
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.size == 0:  # Empty array
            logger.warning("Received empty score_matrix. Skipping update.")
            return

        # Validate finite values unless explicitly allowed
        if not self.allow_non_finite and not np.isfinite(score_matrix).all():
            nan_count = int(np.isnan(score_matrix).sum())
            inf_count = int(np.isinf(score_matrix).sum())
            error_msg = (
                f"score_matrix contains non-finite values (NaN={nan_count}, Inf={inf_count}). "
                "Set allow_non_finite=True or clean the data."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Updating normalizer with data shape: {score_matrix.shape}")
        num_docs = score_matrix.shape[0]

        # Vectorized min/max across columns
        col_mins = np.min(score_matrix, axis=0)
        col_maxs = np.max(score_matrix, axis=0)

        for i, metric in enumerate(self.metric_names):
            current_min = float(col_mins[i])
            current_max = float(col_maxs[i])
            if np.isinf(self.min_vals[metric]):  # First observation
                self.min_vals[metric] = current_min
                self.max_vals[metric] = current_max
                logger.debug(f"Initial range for metric '{metric}': [{current_min:.6f}, {current_max:.6f}]")
            else:
                old_min = self.min_vals[metric]
                old_max = self.max_vals[metric]
                self.min_vals[metric] = float((1 - self.alpha) * old_min + self.alpha * current_min)
                self.max_vals[metric] = float((1 - self.alpha) * old_max + self.alpha * current_max)
                logger.debug(
                    f"Metric '{metric}' range update: min {old_min:.6f}->{self.min_vals[metric]:.6f}, "
                    f"max {old_max:.6f}->{self.max_vals[metric]:.6f} (batch min={current_min:.6f}, max={current_max:.6f})"
                )
        logger.info(f"Normalizer updated successfully with {num_docs} documents.")

    
    def normalize(self, score_matrix: np.ndarray, *, as_float32: bool = False) -> np.ndarray:
        """
        Normalize scores to [0,1] range using current min/max.
        
        Args:
            score_matrix: 2D array of shape [documents × metrics].
                          Metrics must correspond to self.metric_names in order.
        
        Returns:
            np.ndarray: Normalized score matrix of the same shape as input,
                        with values scaled to [0, 1].
                        
        Raises:
            ValueError: If score_matrix is None, not 2D, or columns don't match metric_names.
        """
        if score_matrix is None:
            error_msg = "score_matrix cannot be None for normalization."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.ndim != 2:
            error_msg = f"score_matrix must be 2D for normalization, got {score_matrix.ndim}D shape {score_matrix.shape}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if score_matrix.shape[1] != len(self.metric_names):
            error_msg = (f"score_matrix column count ({score_matrix.shape[1]}) "
                         f"must match metric_names count ({len(self.metric_names)}) for normalization.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Normalizing score matrix of shape: {score_matrix.shape}")
        normalized = np.zeros_like(score_matrix, dtype=np.float64) # Use float64 for precision during calculation
        num_docs = score_matrix.shape[0]
        
        for i, metric in enumerate(self.metric_names):
            min_val = self.min_vals[metric]
            max_val = self.max_vals[metric]
            range_val = max_val - min_val
            
            if range_val > 0.0: # Normal case: valid range
                normalized[:, i] = (score_matrix[:, i] - min_val) / range_val
                logger.debug(f"Normalized metric '{metric}' using range [{min_val:.6f}, {max_val:.6f}]")
            else:
                # Handle case where min == max (constant metric)
                # Assign a default value, typically 0.5 as in the original
                normalized[:, i] = 0.5 
                if np.isinf(min_val): # Truly uninitialized (shouldn't happen if update called first)
                     logger.warning(f"Metric '{metric}' appears uninitialized (inf range). Assigned 0.5.")
                else: # Genuine constant value
                     logger.debug(f"Metric '{metric}' has constant value ({min_val}). Assigned 0.5.")
        
        logger.info(f"Normalization completed for {num_docs} documents.")
        if as_float32:
            return normalized.astype(np.float32)
        return normalized 
    
    def get_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Get current min/max ranges for all metrics.
        
        Returns:
            Dict[str, Tuple[float, float]]: A dictionary mapping metric names
            to their (min, max) tuples.
        """
        ranges = {m: (float(self.min_vals[m]), float(self.max_vals[m])) for m in self.metric_names}
        logger.debug(f"Retrieved current ranges: {ranges}")
        return ranges
``n

## File: organization.py

`python
"""Organization strategies for arranging documents and metrics.

Provides a pluggable abstraction so different ordering backends (SQL/DuckDB,
text specification, heuristic, etc.) can be swapped without changing the core
ZeroModel pipeline.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class BaseOrganizationStrategy:
    """Abstract base for spatial organization strategies."""
    name: str = "base"

    def set_task(self, spec: str):  # pragma: no cover - interface
        raise NotImplementedError

    def organize(self, matrix: np.ndarray, metric_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:  # pragma: no cover - interface
        """Return (sorted_matrix, metric_order, doc_order, analysis_dict)."""
        raise NotImplementedError

class SqlOrganizationStrategy(BaseOrganizationStrategy):
    """SQL-based organization using a DuckDBAdapter-like object.

    Adapter must expose:
        ensure_schema(metric_names: List[str])
        load_matrix(matrix: np.ndarray, metric_names: List[str])
        analyze_query(sql_query: str, metric_names: List[str]) -> Dict[str, Any]
    """
    name = "sql"

    def __init__(self, adapter):
        self.adapter = adapter
        self._sql_query: Optional[str] = None
        self._analysis: Optional[Dict[str, Any]] = None

    def set_task(self, spec: str):
        if not spec or not isinstance(spec, str):
            raise ValueError("SQL task spec must be a non-empty string.")
        self._sql_query = spec

    def organize(self, matrix: np.ndarray, metric_names: List[str]):
        if self._sql_query is None:
            raise RuntimeError("SQL task has not been set before organize().")
        # Ensure schema reflects current metric names then load data
        self.adapter.ensure_schema(metric_names)
        self.adapter.load_matrix(matrix, metric_names)
        # Analyze query
        analysis = self.adapter.analyze_query(self._sql_query, metric_names)
        # Apply ordering
        doc_order_list = analysis.get("doc_order", [])
        num_docs = matrix.shape[0]
        valid_doc_order = [idx for idx in doc_order_list if 0 <= idx < num_docs]
        if not valid_doc_order:
            valid_doc_order = list(range(num_docs))
        sorted_by_docs = matrix[valid_doc_order, :]
        raw_metric_order = analysis.get("metric_order", list(range(matrix.shape[1])))
        metric_count = matrix.shape[1]
        valid_metric_order = [i for i in raw_metric_order if 0 <= i < metric_count]
        if len(valid_metric_order) != metric_count:
            remaining = [i for i in range(metric_count) if i not in valid_metric_order]
            valid_metric_order.extend(remaining)
        if not valid_metric_order:
            valid_metric_order = list(range(metric_count))
        final_matrix = sorted_by_docs[:, valid_metric_order]
        self._analysis = analysis
        return final_matrix, np.array(valid_metric_order), np.array(valid_doc_order), analysis

__all__ = [
    "BaseOrganizationStrategy",
    "SqlOrganizationStrategy",
]
``n

## File: transform.py

`python
# zeromodel/transform.py
"""
Transformation Pipeline

This module provides functions for dynamically transforming visual policy maps
to prioritize specific metrics for different tasks. This enables the same
underlying data to be used for multiple decision contexts.
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def transform_vpm(
    vpm: np.ndarray,
    metric_names: List[str],
    target_metrics: List[str],
    *,
    return_mapping: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[int], List[int]]:
    """
    Transform a Visual Policy Map (VPM) to prioritize specific metrics.
    
    This reorders the metrics (columns) in the VPM so that the target metrics
    appear first. It then sorts the documents (rows) based on the value in the
    first of these target metrics, descending. This makes the most relevant
    information appear in the top-left of the resulting image.

    Args:
        vpm: Visual policy map as an RGB image array of shape [height, width, 3].
             Values are expected to be in the range [0, 255].
        metric_names: List of original metric names corresponding to the columns
                      of the data *before* it was encoded into the VPM.
                      Length should match the total number of metrics represented.
        target_metrics: List of metric names to prioritize and move to the front.
                        Metrics not found in `metric_names` are ignored.

    Returns:
        If return_mapping=False (default):
            np.ndarray: Transformed VPM (RGB image array of the same shape as input).
        If return_mapping=True:
            (transformed_vpm, new_metric_order, sorted_row_indices)
        
    Raises:
        ValueError: If inputs are invalid (e.g., None, incorrect shapes, mismatched dimensions).
    """
    logger.debug(f"Transforming VPM. Shape: {vpm.shape if vpm is not None else 'None'}, "
                 f"Target metrics: {target_metrics}")
    
    # Input validation
    if vpm is None:
        error_msg = "Input VPM cannot be None."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if vpm.ndim != 3 or vpm.shape[2] != 3:
        error_msg = f"VPM must be a 3D RGB array (H, W, 3), got shape {vpm.shape}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if metric_names is None:
        error_msg = "metric_names cannot be None."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if target_metrics is None:  # Allow empty list, but not None
        target_metrics = []
        logger.info("target_metrics was None, treating as empty list.")

    height, width, channels = vpm.shape
    if channels != 3:
        # Redundant check, but good for clarity
        error_msg = f"VPM must have 3 channels (RGB), got {channels}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    total_metrics_in_vpm = width * 3
    if len(metric_names) != total_metrics_in_vpm:
        logger.warning(f"Mismatch: VPM width*3 ({total_metrics_in_vpm}) != len(metric_names) ({len(metric_names)}). "
                       f"Proceeding with VPM width*3 as metric count.")
        # Use VPM dimensions for processing, log warning about mismatch
        actual_metric_count = total_metrics_in_vpm
        # Truncate or pad metric_names conceptually for indexing, but warn
        if len(metric_names) < actual_metric_count:
             logger.info("metric_names list is shorter than VPM metrics. Padding conceptually for indexing.")
        # We'll use actual_metric_count for processing based on VPM
    else:
        actual_metric_count = len(metric_names)

    if actual_metric_count == 0:
        logger.info("No metrics to transform. Returning original VPM.")
        return vpm.copy() # Return a copy to avoid accidental mutation

    # 1. Extract metrics from the VPM image (vectorized)
    # reshape vpm to (H, W*3) interleaving channels consistent with encoding assumption
    flat_metrics = vpm.astype(np.float32).reshape(height, width * 3) / 255.0
    metrics_normalized = flat_metrics[:, :actual_metric_count]
    logger.debug(
        f"Extracted metrics array shape: {metrics_normalized.shape} (vectorized)"
    )

    # 2. Determine the new order of metrics
    # Find indices of target metrics within the available metrics
    metric_indices_to_prioritize = []
    for m in target_metrics:
        try:
            # Find index in the provided metric_names list
            idx = metric_names.index(m)
            # Check if this index is valid for the actual data extracted from VPM
            if idx < actual_metric_count: 
                metric_indices_to_prioritize.append(idx)
            else:
                logger.warning(f"Target metric '{m}' (index {idx}) is beyond the metric count in VPM ({actual_metric_count}). Ignoring.")
        except ValueError:
            logger.warning(f"Target metric '{m}' not found in provided metric_names. Ignoring.")

    # Create the new column order: prioritized metrics first, then the rest
    remaining_indices = [i for i in range(actual_metric_count) 
                        if i not in metric_indices_to_prioritize]
    new_metric_order = metric_indices_to_prioritize + remaining_indices
    logger.debug(f"Calculated new metric order: {new_metric_order}")

    # 3. Reorder the columns (metrics) of the extracted data
    reordered_metrics_normalized = metrics_normalized[:, new_metric_order]
    logger.debug(f"Reordered metrics array shape: {reordered_metrics_normalized.shape}")

    # 4. Sort rows (documents) by the value in the first prioritized metric (descending)
    if len(metric_indices_to_prioritize) > 0:
        sort_key_column = 0 # First column after reordering is the first target metric
        sort_key_values = reordered_metrics_normalized[:, sort_key_column]
        # Get indices that would sort the array descending (highest values first)
        sorted_row_indices = np.argsort(sort_key_values)[::-1] 
        transformed_metrics_normalized = reordered_metrics_normalized[sorted_row_indices]
        logger.debug(f"Sorted rows by metric index {new_metric_order[0]} (original name: {metric_names[new_metric_order[0]] if new_metric_order[0] < len(metric_names) else 'N/A'})")
    else:
        logger.info("No valid target metrics found for sorting. Returning reordered metrics without row sorting.")
        transformed_metrics_normalized = reordered_metrics_normalized
        sorted_row_indices = np.arange(height) # Identity sort if no sorting

    # 5. Re-encode the transformed data back into an RGB image
    # Create output image array
    # 5. Re-encode transformed metrics back to RGB layout
    # Start from zeros; pad metrics if not multiple of 3 for safe reshape
    padded_cols = int(np.ceil(actual_metric_count / 3) * 3)
    pad_needed = padded_cols - actual_metric_count
    if pad_needed:
        pad_block = np.zeros((height, pad_needed), dtype=transformed_metrics_normalized.dtype)
        metrics_padded = np.concatenate([transformed_metrics_normalized, pad_block], axis=1)
    else:
        metrics_padded = transformed_metrics_normalized
    rgb = (np.clip(metrics_padded, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    transformed_vpm = rgb.reshape(height, -1, 3)[:, :width, :]
    logger.info(
        f"VPM transformation complete. Output shape: {transformed_vpm.shape}. Reordered {len(new_metric_order)} metrics."
    )
    if return_mapping:
        return transformed_vpm, new_metric_order, sorted_row_indices.tolist()
    return transformed_vpm

def get_critical_tile(
    vpm: np.ndarray,
    tile_size: int = 3,
    *,
    include_dtype: bool = False,
) -> bytes:
    """
    Extract a critical tile (top-left section) from a visual policy map.

    Args:
        vpm: Visual policy map as an RGB image array of shape [height, width, 3].
        tile_size: Desired size of the square tile (NxN pixels). Defaults to 3.

    Returns:
        bytes: Compact byte representation of the tile.
               Format: [width][height][x_offset][y_offset][(dtype_code?)][pixel_data...]
               If include_dtype=True, a 1-byte dtype code (0=uint8) is inserted after offsets.
               
    Raises:
        ValueError: If inputs are invalid (e.g., None VPM, negative tile_size).
    """
    logger.debug(f"Extracting critical tile. VPM shape: {vpm.shape if vpm is not None else 'None'}, tile_size: {tile_size}")
    
    # Input validation
    if vpm is None:
        error_msg = "Input VPM cannot be None for tile extraction."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if vpm.ndim != 3 or vpm.shape[2] != 3:
        error_msg = f"VPM must be a 3D RGB array (H, W, 3), got shape {vpm.shape}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if tile_size <= 0:
        error_msg = f"tile_size must be positive, got {tile_size}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    vpm_height, vpm_width, _ = vpm.shape

    # Determine the actual tile dimensions (cannot exceed VPM dimensions)
    actual_tile_width = min(tile_size, vpm_width)
    actual_tile_height = min(tile_size, vpm_height)
    logger.debug(f"Actual tile dimensions: {actual_tile_width}x{actual_tile_height}")

    # Convert to compact byte format
    tile_bytes = bytearray()
    tile_bytes.append(actual_tile_width & 0xFF)   # Width (1 byte)
    tile_bytes.append(actual_tile_height & 0xFF)  # Height (1 byte)
    tile_bytes.append(0)                          # X offset (1 byte, always 0 for top-left)
    tile_bytes.append(0)                          # Y offset (1 byte, always 0 for top-left)
    if include_dtype:
        # Currently only uint8 supported (code 0). Extend mapping as needed.
        tile_bytes.append(0)
    logger.debug("Appended tile header bytes.")

    # Add pixel data (1 byte per channel, R,G,B for each pixel)
    # Iterate over the actual tile area within VPM bounds
    sub = vpm[:actual_tile_height, :actual_tile_width, :].astype(np.uint8)
    tile_bytes.extend(sub.flatten().tolist())
            # logger.debug(f"Added pixel ({x},{y}): R={r_value}, G={g_value}, B={b_value}") # Very verbose

    result_bytes = bytes(tile_bytes)
    logger.info(f"Critical tile extracted successfully. Size: {len(result_bytes)} bytes.")
    return result_bytes

``n

## File: utils.py

`python
# zeromodel/utils.py
"""
Utility Functions

This module provides helper functions used throughout the zeromodel package.
"""

from typing import Any

import numpy as np

__all__ = [
    "quantize",
    "dct",
    "idct",
]


def _select_dtype_for_precision(precision: int):
    """Return an integer dtype able to hold the given precision (bits)."""
    if precision <= 8:
        return np.uint8
    if precision <= 16:
        return np.uint16
    if precision <= 32:
        return np.uint32
    return np.uint64


def quantize(value: Any, precision: int) -> Any:
    """Quantize values to specified bit precision (assumes input in [0,1]).

    Clamps input to [0,1] then scales to integer range. Chooses an appropriate
    unsigned integer dtype based on precision.

    Args:
        value: Scalar or ndarray of floats in any range (will be clipped to [0,1]).
        precision: Bit precision (4-32 typical). Values <1 raise, >64 truncated to 64.

    Returns:
        Quantized integer array / scalar of appropriate dtype.
    """
    if not isinstance(precision, int):
        raise TypeError("precision must be an int")
    if precision < 1:
        raise ValueError("precision must be >= 1")
    if precision > 64:
        precision = 64  # cap
    dtype = _select_dtype_for_precision(precision)
    max_val = (1 << precision) - 1 if precision < 64 else np.iinfo(dtype).max
    if isinstance(value, np.ndarray):
        clipped = np.clip(value, 0.0, 1.0)
        scaled = np.round(clipped * max_val)
        return scaled.astype(dtype)
    # Scalar path
    v = float(value)
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return int(round(v * max_val))

def dct(matrix: np.ndarray, norm: str = 'ortho', axis: int = -1) -> np.ndarray:
    """Compute a DCT-II along a chosen axis (minimal, SciPy-free).

    Based on the standard definition:
        X_n = sum_{k=0}^{N-1} x_k * cos[ pi/N * (k + 0.5) * n ]

    Orthonormal scaling (norm='ortho') matches scipy.fft.dct(type=2, norm='ortho').

    Complexity is O(N^2); intended for small edge scenarios.
    """
    x = np.asarray(matrix, dtype=np.float64)
    x = np.moveaxis(x, axis, -1)
    N = x.shape[-1]
    if N == 0:
        return matrix.copy()
    k = np.arange(N, dtype=np.float64)
    n = k  # reuse variable for clarity
    cos_table = np.cos(np.pi / N * (k + 0.5)[:, None] * n[None, :])  # shape (N,N)
    # Perform tensordot over last axis of x with first axis of cos_table
    out = np.tensordot(x, cos_table, axes=([-1], [0]))  # shape (..., N)
    if norm == 'ortho':
        out[..., 0] *= np.sqrt(1.0 / N)
        out[..., 1:] *= np.sqrt(2.0 / N)
    out = np.moveaxis(out, -1, axis)
    return out.astype(np.float32, copy=False)

def idct(matrix: np.ndarray, norm: str = 'ortho', axis: int = -1) -> np.ndarray:
    """Compute an IDCT (inverse of DCT-II) aka DCT-III along axis.

    For norm='ortho' this inverts ``dct(..., norm='ortho')`` numerically.
    Complexity O(N^2); intended for small inputs.
    """
    X = np.asarray(matrix, dtype=np.float64)
    X = np.moveaxis(X, axis, -1)
    N = X.shape[-1]
    if N == 0:
        return matrix.copy()
    n = np.arange(N, dtype=np.float64)
    k = n  # reuse
    cos_table = np.cos(np.pi / N * (n + 0.5)[:, None] * k[None, :])  # (N,N)
    Y = X.copy()
    if norm == 'ortho':
        Y[..., 0] *= np.sqrt(1.0 / N)
        Y[..., 1:] *= np.sqrt(2.0 / N)
    else:
        # Undo scaling expected for unnormalized forward (approximate)
        Y[..., 0] *= 1.0 / (N / 2.0)
    out = np.tensordot(Y, cos_table.T, axes=([-1], [0]))  # shape (..., N)
    out = np.moveaxis(out, -1, axis)
    return out.astype(np.float32, copy=False)
``n

## File: vpm_logic.py

`python
# zeromodel/vpm_logic.py
"""
Visual Policy Maps enable a new kind of symbolic mathematics.

Each VPM is a spatially organized array of scalar values encoding task-relevant priorities.
By composing them using logical operators (AND, OR, NOT, NAND, etc.), we form a new symbolic system
where reasoning becomes image composition, and meaning is distributed across space.

These operators allow tiny edge devices to perform sophisticated reasoning by querying
regions of interest in precomputed VPMs. Just like NAND gates enable classical computation,
VPM logic gates enable distributed visual intelligence.

This is not just fuzzy logic. This is **Visual Symbolic Math**.
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def normalize_vpm(vpm: np.ndarray) -> np.ndarray:
    """
    Ensures a VPM is in the normalized float [0.0, 1.0] range.
    Handles conversion from uint8, uint16, float16, float32, float64.
    """
    logger.debug(f"Normalizing VPM of dtype {vpm.dtype} and shape {vpm.shape}")
    if np.issubdtype(vpm.dtype, np.integer):
        # Integer types: normalize based on max value for the dtype
        dtype_info = np.iinfo(vpm.dtype)
        max_val = dtype_info.max
        min_val = dtype_info.min
        # Handle signed integers if necessary, but VPMs are typically unsigned
        if min_val < 0:
            logger.warning(
                f"VPM dtype {vpm.dtype} is signed. Normalizing assuming 0-min_val range."
            )
            range_val = max_val - min_val
            return ((vpm.astype(np.float64) - min_val) / range_val).astype(np.float32)
        else:
            # Unsigned integer
            return (vpm.astype(np.float64) / max_val).astype(np.float32)
    else:  # Floating point types
        # Assume already in [0, 1] or close enough. Clip for safety.
        return np.clip(vpm, 0.0, 1.0).astype(np.float32)


def denormalize_vpm(vpm: np.ndarray, output_type=np.uint8, assume_normalized: bool = True) -> np.ndarray:
    """Convert a (normalized) VPM to a specified dtype.

    Args:
        vpm: Input VPM. If not already float in [0,1] set ``assume_normalized=False``.
        output_type: Target numpy dtype.
        assume_normalized: If False, will first run ``normalize_vpm``.
    """
    logger.debug(f"Denormalizing VPM to dtype {output_type} (assume_normalized={assume_normalized})")
    data = vpm if assume_normalized else normalize_vpm(vpm)
    if np.issubdtype(output_type, np.integer):
        dtype_info = np.iinfo(output_type)
        max_val = dtype_info.max
        min_val = dtype_info.min
        scaled_vpm = np.clip(data * max_val, min_val, max_val)
        return scaled_vpm.astype(output_type)
    clipped_vpm = np.clip(data, 0.0, 1.0)
    return clipped_vpm.astype(output_type)


# ---------------- Internal Helpers ---------------- #
def _ensure_same_shape(a: np.ndarray, b: np.ndarray, op: str) -> None:
    if a.shape != b.shape:
        logger.error(f"VPM {op}: Shape mismatch. a: {a.shape}, b: {b.shape}")
        raise ValueError(f"VPMs must have the same shape for {op.upper()}. Got {a.shape} and {b.shape}")


def _normalize_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return normalize_vpm(a), normalize_vpm(b)


def vpm_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical OR operation (fuzzy union) on two VPMs.
    The result highlights areas relevant to EITHER input VPM by taking the element-wise maximum.
    Assumes VPMs are normalized to the range [0, 1] (float).

    Args:
        a (np.ndarray): First VPM (normalized float).
        b (np.ndarray): Second VPM (normalized float, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the OR operation (normalized float).
    """
    logger.debug(f"Performing VPM OR operation on shapes {a.shape} and {b.shape}")
    _ensure_same_shape(a, b, "or")
    a_norm, b_norm = _normalize_pair(a, b)
    result = np.maximum(a_norm, b_norm)
    logger.debug("VPM OR operation completed.")
    return result  # Already normalized float32


def vpm_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical AND operation (fuzzy intersection) on two VPMs.
    The result highlights areas relevant to BOTH input VPMs by taking the element-wise minimum.
    Assumes VPMs are normalized to the range [0, 1] (float).

    Args:
        a (np.ndarray): First VPM (normalized float).
        b (np.ndarray): Second VPM (normalized float, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the AND operation (normalized float).
    """
    logger.debug(f"Performing VPM AND operation on shapes {a.shape} and {b.shape}")
    _ensure_same_shape(a, b, "and")
    a_norm, b_norm = _normalize_pair(a, b)
    result = np.minimum(a_norm, b_norm)
    logger.debug("VPM AND operation completed.")
    return result  # Already normalized float32


def vpm_not(a: np.ndarray) -> np.ndarray:
    """
    Performs a logical NOT operation on a VPM.
    Inverts the relevance/priority represented in the VPM.
    Assumes VPMs are normalized to the range [0, 1] (float).

    Args:
        a (np.ndarray): Input VPM (normalized float).

    Returns:
        np.ndarray: The resulting inverted VPM (normalized float).
    """
    logger.debug(
        f"Performing VPM NOT operation on shape {a.shape} with dtype {a.dtype}"
    )
    # Normalize input to ensure consistency
    a_norm = normalize_vpm(a)
    # Invert: 1.0 - value
    result = 1.0 - a_norm
    logger.debug("VPM NOT operation completed.")
    return result  # Already normalized float32


def vpm_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical difference operation (A - B) on two VPMs.
    Result highlights areas important to A but NOT to B.
    Functionally equivalent to `vpm_and(a, vpm_not(b))` but uses clipping.

    Args:
        a (np.ndarray): First VPM (minuend, normalized float).
        b (np.ndarray): Second VPM (subtrahend, normalized float, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the difference operation (normalized float).
    """
    logger.debug(
        f"Performing VPM SUBTRACT (A - B) operation on shapes {a.shape} and {b.shape}"
    )
    _ensure_same_shape(a, b, "subtract")
    a_norm, b_norm = _normalize_pair(a, b)
    # Subtract and clip to [0, 1] to ensure valid range
    result = np.clip(a_norm - b_norm, 0.0, 1.0)
    logger.debug("VPM SUBTRACT operation completed.")
    return result


def vpm_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a simple additive operation on two VPMs (A + B), clipping the result
    to ensure it remains in the valid range [0, 1].

    Args:
        a (np.ndarray): First VPM (normalized float).
        b (np.ndarray): Second VPM (normalized float, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the additive operation (normalized float).
    """
    logger.debug(
        f"Performing VPM ADD (A + B) operation on shapes {a.shape} and {b.shape}"
    )
    _ensure_same_shape(a, b, "add")
    a_norm, b_norm = _normalize_pair(a, b)
    # Add and clip to [0, 1]
    result = np.clip(a_norm + b_norm, 0.0, 1.0)
    logger.debug("VPM ADD operation completed.")
    return result


def vpm_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical XOR (exclusive OR) operation on two VPMs.
    Result highlights areas relevant to A OR B, but NOT BOTH.
    Functionally equivalent to `vpm_or(vpm_diff(a, b), vpm_diff(b, a))`.

    Args:
        a (np.ndarray): First VPM (normalized float).
        b (np.ndarray): Second VPM (normalized float, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the XOR operation (normalized float).
    """
    logger.debug(f"Performing VPM XOR operation on shapes {a.shape} and {b.shape}")
    _ensure_same_shape(a, b, "xor")
    a_norm, b_norm = _normalize_pair(a, b)
    # Calculate (A AND NOT B) OR (B AND NOT A)
    a_and_not_b = vpm_subtract(a_norm, b_norm)  # Use normalized inputs
    b_and_not_a = vpm_subtract(b_norm, a_norm)
    result = vpm_or(a_and_not_b, b_and_not_a)  # vpm_or also uses normalized inputs
    logger.debug("VPM XOR operation completed.")
    return result


def vpm_nand(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a NAND operation: NOT(AND(a, b)).
    Universal gate for constructing any logic circuit.

    Returns:
        np.ndarray: Result of NAND (normalized float).
    """
    # Normalize inputs
    a_norm, b_norm = _normalize_pair(a, b)
    return vpm_not(vpm_and(a_norm, b_norm))


def vpm_nor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a NOR operation: NOT(OR(a, b)).
    Also a universal logic gate.

    Returns:
        np.ndarray: Result of NOR (normalized float).
    """
    # Normalize inputs
    a_norm, b_norm = _normalize_pair(a, b)
    return vpm_not(vpm_or(a_norm, b_norm))


def vpm_resize(img, target_shape):
    """
    Drop-in replacement for scipy.ndimage.zoom(img, zoom=(h/w), order=1),
    for 2D or 3D (HWC) images using bilinear interpolation.
    """
    import numpy as np

    in_h, in_w = img.shape[:2]
    out_h, out_w = target_shape
    channels = img.shape[2] if img.ndim == 3 else 1

    scale_h = in_h / out_h
    scale_w = in_w / out_w

    # Match scipy.ndimage.zoom coordinate mapping
    row_idx = np.arange(out_h) * scale_h
    col_idx = np.arange(out_w) * scale_w

    row0 = np.floor(row_idx).astype(int)
    col0 = np.floor(col_idx).astype(int)
    row1 = np.clip(row0 + 1, 0, in_h - 1)
    col1 = np.clip(col0 + 1, 0, in_w - 1)

    wy = (row_idx - row0).reshape(-1, 1)
    wx = (col_idx - col0).reshape(1, -1)

    row0 = np.clip(row0, 0, in_h - 1)
    col0 = np.clip(col0, 0, in_w - 1)

    if img.ndim == 2:
        img = img[:, :, None]

    out = np.empty((out_h, out_w, channels), dtype=np.float32)

    for c in range(channels):
        I00 = img[row0[:, None], col0[None, :], c]
        I01 = img[row0[:, None], col1[None, :], c]
        I10 = img[row1[:, None], col0[None, :], c]
        I11 = img[row1[:, None], col1[None, :], c]

        top = I00 * (1 - wx) + I01 * wx
        bottom = I10 * (1 - wx) + I11 * wx
        out[..., c] = top * (1 - wy) + bottom * wy

    return out if channels > 1 else out[..., 0]


def vpm_concat_horizontal(vpm1: np.ndarray, vpm2: np.ndarray) -> np.ndarray:
    """
    Concatenate VPMs horizontally (side-by-side).
    Assumes VPMs are normalized floats. Handles height mismatch by cropping.

    Args:
        vpm1 (np.ndarray): Left VPM (normalized float).
        vpm2 (np.ndarray): Right VPM (normalized float).

    Returns:
        np.ndarray: Horizontally concatenated VPM (normalized float).
    """
    logger.debug(
        f"Horizontally concatenating VPMs of shapes {vpm1.shape} and {vpm2.shape}"
    )
    # Normalize inputs
    v1 = normalize_vpm(vpm1)
    v2 = normalize_vpm(vpm2)

    # Ensure same height by cropping the taller one
    min_height = min(v1.shape[0], v2.shape[0])
    v1_crop = v1[:min_height, :, :] if v1.ndim == 3 else v1[:min_height, :]
    v2_crop = v2[:min_height, :, :] if v2.ndim == 3 else v2[:min_height, :]

    # Concatenate along width axis (axis=1)
    try:
        result = np.concatenate((v1_crop, v2_crop), axis=1)
        logger.debug(f"Horizontal concatenation result shape: {result.shape}")
        return result  # Already normalized float32
    except ValueError as e:
        logger.error(f"Failed to concatenate VPMs horizontally: {e}")
        raise ValueError(f"VPMs could not be concatenated horizontally: {e}") from e


def vpm_concat_vertical(vpm1: np.ndarray, vpm2: np.ndarray) -> np.ndarray:
    """
    Concatenate VPMs vertically (stacked).
    Assumes VPMs are normalized floats. Handles width mismatch by cropping.

    Args:
        vpm1 (np.ndarray): Top VPM (normalized float).
        vpm2 (np.ndarray): Bottom VPM (normalized float).

    Returns:
        np.ndarray: Vertically concatenated VPM (normalized float).
    """
    logger.debug(
        f"Vertically concatenating VPMs of shapes {vpm1.shape} and {vpm2.shape}"
    )
    # Normalize inputs
    v1 = normalize_vpm(vpm1)
    v2 = normalize_vpm(vpm2)

    # Ensure same width by cropping the wider one
    min_width = min(v1.shape[1], v2.shape[1])
    v1_crop = v1[:, :min_width, :] if v1.ndim == 3 else v1[:, :min_width]
    v2_crop = v2[:, :min_width, :] if v2.ndim == 3 else v2[:, :min_width]

    # Concatenate along height axis (axis=0)
    try:
        result = np.concatenate((v1_crop, v2_crop), axis=0)
        logger.debug(f"Vertical concatenation result shape: {result.shape}")
        return result  # Already normalized float32
    except ValueError as e:
        logger.error(f"Failed to concatenate VPMs vertically: {e}")
        raise ValueError(f"VPMs could not be concatenated vertically: {e}") from e


def query_top_left(vpm: np.ndarray, context_size: int = 1) -> float:
    """
    Queries the top-left region of a VPM for a relevance score.
    This provides a simple, aggregated measure of relevance for the entire VPM.
    Now resolution-independent by using relative context_size.

    Args:
        vpm (np.ndarray): The VPM to query (assumed normalized float internally).
        context_size (int): The size of the top-left square region to consider (NxN).
                           Must be a positive integer. Interpreted relative to VPM size.

    Returns:
        float: An aggregate relevance score (mean) from the top-left region.
    """
    # Normalize input to ensure consistency for internal processing
    vpm_norm = normalize_vpm(vpm)
    logger.debug(
        f"Querying top-left region of VPM (shape: {vpm_norm.shape}) with context size {context_size}"
    )
    if vpm_norm.ndim < 2:
        logger.error("VPM must be at least 2D for top-left query.")
        raise ValueError("VPM must be at least 2D.")
    if not isinstance(context_size, int) or context_size <= 0:
        logger.error(
            f"Invalid context_size: {context_size}. Must be a positive integer."
        )
        raise ValueError("context_size must be a positive integer.")

    height, width = vpm_norm.shape[:2]  # Handle both 2D and 3D (H, W) or (H, W, C)
    # Make context_size relative and bounded
    actual_context_h = min(context_size, height)
    actual_context_w = min(context_size, width)

    top_left_region = vpm_norm[:actual_context_h, :actual_context_w]
    # Simple aggregation: mean. Could be max, weighted, etc.
    score = np.mean(top_left_region)
    logger.debug(
        f"Top-left query score (mean of {actual_context_h}x{actual_context_w} region): {score:.4f}"
    )
    return float(score)


def create_interesting_map(
    quality_vpm: np.ndarray, novelty_vpm: np.ndarray, uncertainty_vpm: np.ndarray
) -> np.ndarray:
    """
    Creates a composite 'interesting' VPM based on the logic:
    (Quality AND NOT Uncertainty) OR (Novelty AND NOT Uncertainty)

    Args:
        quality_vpm (np.ndarray): VPM representing quality (will be normalized).
        novelty_vpm (np.ndarray): VPM representing novelty (will be normalized).
        uncertainty_vpm (np.ndarray): VPM representing uncertainty (will be normalized).

    Returns:
        np.ndarray: The 'interesting' VPM (normalized float).
    """
    logger.info("Creating 'interesting' composite VPM.")
    try:
        # Normalize inputs implicitly via vpm_not/vpm_and/vpm_or
        anti_uncertainty = vpm_not(uncertainty_vpm)
        good_map = vpm_and(quality_vpm, anti_uncertainty)
        exploratory_map = vpm_and(novelty_vpm, anti_uncertainty)
        interesting_map = vpm_or(good_map, exploratory_map)
        logger.info("'Interesting' VPM created successfully.")
        return interesting_map
    except Exception as e:
        logger.error(f"Failed to create 'interesting' VPM: {e}")
        raise
``n
