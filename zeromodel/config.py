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

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Initialize the base logger early so we can log config loading
logger = logging.getLogger("zeromodel.config")
logger.addHandler(logging.NullHandler())  # Prevent "no handler" warnings

DEFAULT_CONFIG = {
    # Core processing configuration
    "core": {
        "use_duckdb": False,
        "duckdb_bypass_threshold": 0.5,  # ms
        "precision": 8,
        "normalize_inputs": True,
        "nonlinearity_handling": "auto",  # Options: "auto", "none", "force"
        "cache_preprocessed_vpm": True,
        "max_cached_tasks": 100,
        "default_output_precision": "float32",
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
        "level": "DEBUG",
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
            },
        ],
    },
    # Advanced features
    "advanced": {
        "metric_discovery": False,
        "metric_discovery_interval": 3600,  # seconds
    },
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
                },
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
                },
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
        os.path.expanduser("~/zeromodel.yaml"),
    ]

    for path in possible_paths:
        if path and Path(path).exists():
            try:
                with open(path, "r") as f:
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
    formatter = logging.Formatter(
        log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

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
                    filename, maxBytes=max_bytes, backupCount=backup_count
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


def resolve_config(
    user_config: Optional[Dict[str, Any]] = None, env: str = "auto"
) -> Dict[str, Any]:
    """Resolve final configuration with smart defaults and logging setup"""
    # Start with base defaults
    config = DEFAULT_CONFIG.copy()

    # Apply environment-specific defaults
    env = env if env != "auto" else detect_deployment_environment()
    env_defaults = get_edge_aware_defaults(env)

    # Deep merge environment defaults
    def deep_merge(target: Dict, source: Dict) -> None:
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
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
    logger.debug(
        f"Core processing: use_duckdb={config['core']['use_duckdb']}, precision={config['core']['precision']}"
    )
    logger.debug(f"Edge mode: {'enabled' if config['edge']['enabled'] else 'disabled'}")

    return config


def get_config_value(
    config: Dict[str, Any], section: str, key: str, default: Any = None
) -> Any:
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

    def initialize(
        self, user_config: Optional[Dict[str, Any]] = None, env: str = "auto"
    ) -> None:
        """Initialize the configuration manager"""
        if self._config is None:
            self._config = resolve_config(user_config, env)
            logger.info("Configuration manager initialized")

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a configuration value by section and key(s)"""
        if self._config is None:
            raise RuntimeError(
                "Configuration manager not initialized. Call initialize() first."
            )

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
            raise RuntimeError(
                "Configuration manager not initialized. Call initialize() first."
            )

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


def init_config(
    user_config: Optional[Dict[str, Any]] = None, env: str = "auto"
) -> None:
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

