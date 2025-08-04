# zeromodel/symbolic.py
from typing import Dict, List

import numpy as np


class SymbolicInterpreter:
    """
    Translates between spatial patterns in Visual Policy Maps and symbolic logic.
    
    This implements the neural-symbolic bridge that turns pixels into meaningful symbols.
    """
    
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.symbol_templates = self._create_symbol_templates()
    
    def _create_symbol_templates(self) -> Dict[str, np.ndarray]:
        """Create templates for common symbolic patterns"""
        templates = {}
        
        # Top-left cluster pattern (high priority)
        high_priority = np.zeros((3, 3))
        high_priority[0, 0] = 1.0  # Top-left is most important
        high_priority[0, 1] = 0.7
        high_priority[1, 0] = 0.7
        templates["HIGH_PRIORITY"] = high_priority
        
        # Gradient pattern (ranking)
        ranking = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                ranking[i, j] = 1.0 - (i + j) / 4.0
        templates["RANKING"] = ranking
        
        return templates
    
    def interpret_tile(self, tile: np.ndarray) -> List[str]:
        """
        Interpret a critical tile as symbolic logic.
        
        Args:
            tile: Critical tile from Visual Policy Map
            
        Returns:
            List of symbolic predicates describing the tile
        """
        symbols = []
        
        # Match against templates
        for symbol, template in self.symbol_templates.items():
            # Resize template to match tile size
            resized_template = self._resize_template(template, tile.shape[0], tile.shape[1])
            
            # Calculate similarity
            similarity = np.sum(tile * resized_template) / (
                np.linalg.norm(tile) * np.linalg.norm(resized_template) + 1e-10
            )
            
            # Threshold for symbol recognition
            if similarity > 0.7:
                symbols.append(symbol)
        
        return symbols
    
    def _resize_template(self, template: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize template to match target dimensions"""
        # Simple nearest-neighbor resize for edge compatibility
        h_ratio = target_h / template.shape[0]
        w_ratio = target_w / template.shape[1]
        
        resized = np.zeros((target_h, target_w))
        for i in range(target_h):
            for j in range(target_w):
                orig_i = int(i / h_ratio)
                orig_j = int(j / w_ratio)
                resized[i, j] = template[orig_i, orig_j]
        
        return resized
    
    def to_symbolic_rule(self, symbols: List[str], metric_names: List[str]) -> str:
        """
        Convert symbols to human-readable rule.
        
        Args:
            symbols: List of symbolic predicates
            metric_names: Names of metrics in the policy map
            
        Returns:
            Human-readable rule
        """
        if "HIGH_PRIORITY" in symbols:
            # Determine which metric is most important (leftmost column)
            primary_metric = metric_names[0] if metric_names else "document"
            return f"Prioritize documents with high {primary_metric}"
        
        if "RANKING" in symbols:
            return "Documents are ranked by relevance"
        
        return "General policy evaluation"