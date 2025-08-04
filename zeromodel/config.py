"""
Zero-Model Intelligence Configuration System

This module handles loading and validating configuration from YAML files.
"""

import os
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'zeromodel_config.yaml')
DEFAULT_CONFIG = {
    'zeromodel': {
        'precision': 8,
        'task_sorter': {
            'ib_threshold': 0.7,
            'adaptive_threshold': True,
            'semantic_groups': {
                'uncertainty': ['uncertainty', 'confidence', 'ambiguity', 'doubt'],
                'size': ['size', 'length', 'scale', 'magnitude'],
                'quality': ['quality', 'score', 'rating', 'value'],
                'novelty': ['novelty', 'diversity', 'originality', 'innovation']
            }
        },
        'hierarchical': {
            'num_levels': 3,
            'zoom_factor': 3,
            'wavelet': 'bior6.8',
            'max_levels': None,
            'soft_thresholding': True
        },
        'edge': {
            'context_size': 3,
            'critical_tile_size': 3,
            'fallback_to_clustering': True
        }
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file, with fallback to default config.
    
    Args:
        config_path: Path to YAML configuration file. If None, uses default path.
    
    Returns:
        Dictionary containing configuration
    """
    # Use provided path or default path
    path_to_use = config_path or DEFAULT_CONFIG_PATH
    
    # Try to load from file
    if path_to_use and os.path.exists(path_to_use):
        try:
            with open(path_to_use, 'r') as f:
                config = yaml.safe_load(f)
                # Validate structure
                if 'zeromodel' not in config:
                    raise ValueError("Config file must contain 'zeromodel' section")
                return config
        except Exception as e:
            print(f"Warning: Failed to load config from {path_to_use}: {str(e)}")
    
    # Return default config if file loading failed
    print(f"Using default configuration (could not load from {path_to_use})")
    return DEFAULT_CONFIG.copy()

def get_config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get a nested configuration value.
    
    Args:
        config: Configuration dictionary
        *keys: Keys to traverse the nested structure
        default: Default value if path doesn't exist
    
    Returns:
        The configuration value or default
    """
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, False otherwise
    """
    zeromodel = config.get('zeromodel', {})
    
    # Validate precision
    precision = zeromodel.get('precision', 8)
    if not (4 <= precision <= 16):
        print(f"Warning: precision must be between 4-16, got {precision}")
        return False
    
    # Validate hierarchical parameters
    hierarchical = zeromodel.get('hierarchical', {})
    num_levels = hierarchical.get('num_levels', 3)
    zoom_factor = hierarchical.get('zoom_factor', 3)
    if num_levels < 1:
        print(f"Warning: num_levels must be at least 1, got {num_levels}")
        return False
    if zoom_factor < 2:
        print(f"Warning: zoom_factor should be at least 2, got {zoom_factor}")
        return False
    
    # Validate edge parameters
    edge = zeromodel.get('edge', {})
    context_size = edge.get('context_size', 3)
    critical_tile_size = edge.get('critical_tile_size', 3)
    if context_size < 1 or critical_tile_size < 1:
        print(f"Warning: context_size and critical_tile_size must be at least 1")
        return False
    
    return True