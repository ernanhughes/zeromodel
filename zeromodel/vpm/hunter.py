"""
Visual Policy Map (VPM) Hunter Module

Implements a hierarchical search algorithm for locating optimal targets in VPMs using a
coarse-to-fine strategy. The hunter progressively refines its search area based on confidence
metrics, mimicking how humans zoom in on important regions when examining complex data.

Key Features:
- Adaptive zooming strategy for hierarchical VPMs
- Area-of-Interest (AOI) refinement for base ZeroModels
- Confidence-based stopping conditions
- Detailed audit trail for explainability
"""

import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from zeromodel import HierarchicalVPM, ZeroModel

logger = logging.getLogger(__name__)

class VPMHunter:
    """
    Heat-seeking search agent for Visual Policy Maps (VPMs).
    
    Implements a multi-resolution search strategy that:
    1. Starts at coarse resolution to identify promising regions
    2. Progressively zooms into higher-resolution views
    3. Stops when confidence threshold is reached or maximum steps taken
    
    Supports both hierarchical VPMs (multi-level) and base ZeroModels (single-level).
    
    Attributes:
        vpm_source (Union[HierarchicalVPM, ZeroModel]): Source VPM to search
        tau (float): Confidence threshold for stopping (0.0-1.0)
        max_steps (int): Maximum search iterations
        aoi_size_sequence (Tuple[int, ...]): AOI sizes for base ZeroModel refinement
        is_hierarchical (bool): Source type flag
        num_levels (int): Number of levels in hierarchical source
    """

    def __init__(
        self,
        vpm_source: Union[HierarchicalVPM, ZeroModel],
        tau: float = 0.75,
        max_steps: int = 6,
        aoi_size_sequence: Tuple[int, ...] = (9, 5, 3, 1),
    ):
        """
        Initialize the VPM hunter with search parameters.
        
        Args:
            vpm_source: Source VPM (hierarchical or base)
            tau: Confidence threshold for stopping (0.0-1.0)
            max_steps: Maximum number of search iterations
            aoi_size_sequence: Sequence of AOI sizes for base ZeroModel refinement
            
        Raises:
            TypeError: For invalid vpm_source type
        """
        # Validate source type
        if not isinstance(vpm_source, (HierarchicalVPM, ZeroModel)):
            raise TypeError("vpm_source must be HierarchicalVPM or ZeroModel")
            
        # Configure search parameters
        self.vpm_source = vpm_source
        self.tau = max(0.0, min(1.0, tau))
        self.max_steps = max(1, max_steps)
        self.aoi_size_sequence = tuple(max(1, s) for s in aoi_size_sequence)
        
        # Determine source type properties
        self.is_hierarchical = isinstance(vpm_source, HierarchicalVPM)
        self.num_levels = getattr(vpm_source, "num_levels", 1)
        
        logger.info(
            f"VPMHunter initialized for {'HierarchicalVPM' if self.is_hierarchical else 'ZeroModel'} "
            f"with {self.num_levels} levels. Confidence threshold: {self.tau}, Max steps: {self.max_steps}"
        )

    def hunt(self, initial_level: int = 0) -> Tuple[Union[int, Tuple[int, int]], float, List[Dict[str, Any]]]:
        """
        Execute the hierarchical search for optimal targets.
        
        The search progresses through three phases:
        1. Initialization: Set starting level/AOI
        2. Iterative Refinement: Retrieve tile, make decision, record audit
        3. Termination: Return best candidate when conditions met
        
        Args:
            initial_level: Starting level for hierarchical VPMs
            
        Returns:
            Tuple containing:
            - target_identifier: Document index (base) or (level, index) (hierarchical)
            - confidence: Final confidence score (0.0-1.0)
            - audit: Step-by-step search record
        """
        audit: List[Dict[str, Any]] = []  # Audit trail
        steps = 0                         # Step counter
        level = initial_level if self.is_hierarchical else 0
        current_aoi = self.aoi_size_sequence[0] if not self.is_hierarchical else None
        final_doc_idx = -1                # Final document index
        final_confidence = 0.0            # Final confidence score
        
        # --- ITERATIVE SEARCH LOOP ---
        while steps < self.max_steps:
            # --- TILE RETRIEVAL ---
            if self.is_hierarchical:
                # Hierarchical VPM: Get 3x3 tile from current level
                tile = self.vpm_source.get_tile(level, width=3, height=3)
                # Get ZeroModel instance for decision making
                zm = self.vpm_source.get_level(level)["zeromodel"]
                # Make decision using level-specific model
                doc_idx, confidence = zm.get_decision_by_metric(0)
            else:
                # Base ZeroModel: Get critical tile from current AOI
                size = self.aoi_size_sequence[min(steps, len(self.aoi_size_sequence)-1)]
                tile = self.vpm_source.extract_critical_tile(metric_idx=0, size=size)
                # Make decision using main model
                doc_idx, confidence = self.vpm_source.get_decision_by_metric(0)
            
            # Update final state
            final_doc_idx = doc_idx
            final_confidence = confidence
            
            # --- TILE SCORING ---
            score = self._score_tile_ndarray(tile)
            
            # --- AUDIT RECORDING ---
            audit.append({
                "step": steps + 1,
                "level": level,
                "aoi_size": current_aoi,
                "tile_shape": tuple(tile.shape),
                "tile_score": float(score),
                "confidence": float(confidence),
                "doc_index": int(doc_idx),
            })
            logger.debug(
                f"Step {steps+1}: level={level}, AOI={current_aoi}, "
                f"tile_score={score:.4f}, confidence={confidence:.4f}, doc={doc_idx}"
            )
            
            # --- TERMINATION CHECK ---
            # Condition 1: Confidence threshold reached
            # Condition 2: Maximum steps reached
            if confidence >= self.tau or steps + 1 >= self.max_steps:
                logger.info(
                    f"Stopping at step {steps+1}: "
                    f"Confidence {'exceeded threshold' if confidence >= self.tau else 'max steps reached'}"
                )
                break
            
            # --- SEARCH REFINEMENT ---
            if self.is_hierarchical and level < self.num_levels - 1:
                # Hierarchical: Zoom to next level
                level += 1
                logger.debug(f"Zooming to level {level}")
            elif not self.is_hierarchical:
                # Base: Move to next AOI size in sequence
                nxt = min(steps + 1, len(self.aoi_size_sequence) - 1)
                current_aoi = self.aoi_size_sequence[nxt]
                logger.debug(f"Refining AOI size to {current_aoi}")
                
            steps += 1
        
        # --- RESULT FORMATTING ---
        target = (level, final_doc_idx) if self.is_hierarchical else final_doc_idx
        logger.info(
            f"Hunt completed in {steps+1} steps. "
            f"Target: {target}, Confidence: {final_confidence:.4f}"
        )
        
        return target, final_confidence, audit

    @staticmethod
    def _score_tile_ndarray(tile: np.ndarray) -> float:
        """
        Score a tile based on weighted spatial importance.
        
        Emphasizes top-left regions using a radial weighting scheme that decays with
        distance from the origin, matching ZeroModel's decision bias.
        
        Args:
            tile: VPM tile as uint16 array (H, W, 3)
            
        Returns:
            Weighted average intensity (0.0-1.0)
        """
        # Validate input
        if not isinstance(tile, np.ndarray) or tile.ndim != 3 or tile.shape[2] != 3:
            logger.warning(
                "Invalid tile format. Expected (H, W, 3) array, got %s",
                getattr(tile, "shape", type(tile)))
            return 0.0
        
        # Convert to normalized float [0, 1]
        x = tile.astype(np.float32) / 65535.0
        
        # Create radial distance weights (decay from top-left)
        H, W, _ = x.shape
        yy = np.arange(H, dtype=np.float32)[:, None]  # Vertical coordinates
        xx = np.arange(W, dtype=np.float32)[None, :]  # Horizontal coordinates
        dist = np.sqrt(yy**2 + xx**2)                 # Euclidean distance from origin
        
        # Compute weights: inverse distance weighting
        # - Closer to top-left → higher weight
        # - 0.15 controls decay rate (higher = steeper decay)
        w = 1.0 / (1.0 + 0.15 * dist)
        w /= (w.sum() + 1e-9)  # Normalize to sum=1
        
        # Calculate channel-mean intensity
        channel_mean = x.mean(axis=2)
        
        # Compute weighted average
        return float((channel_mean * w).sum())