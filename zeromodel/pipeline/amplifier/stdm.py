# zeromodel/pipeline/stages/amplifiers/stdm.py
"""
STDM (Spatio-Temporal Decision Making) amplifier stage.

This implements ZeroModel's "intelligence lives in the data structure" principle
by learning optimal metric weights and organization before VPM encoding.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage
from zeromodel.vpm.stdm import gamma_operator, learn_w, top_left_mass

logger = logging.getLogger(__name__)

class STDMAmplifier(PipelineStage):
    """STDM amplifier stage for ZeroModel."""
    
    name = "stdm"
    category = "amplifier"
    
    def __init__(self, **params):
        super().__init__(**params)
        self.Kc = params.get("Kc", 12)
        self.Kr = params.get("Kr", 48)
        self.alpha = params.get("alpha", 0.97)
        self.u_mode = params.get("u_mode", "mirror_w")
        self.iters = params.get("iters", 120)
        self.step = params.get("step", 8e-3)
        self.l2 = params.get("l2", 2e-3)
    
    def validate_params(self):
        """Validate STDM parameters."""
        if self.Kc <= 0:
            raise ValueError("Kc must be positive")
        if self.Kr <= 0:
            raise ValueError("Kr must be positive")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0,1]")
    
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply STDM amplification to a VPM.
        
        This is the "visual amplifier" that surfaces hidden signals in model outputs.
        """
        context = self._get_context(context)
        
        # Handle different VPM dimensions
        if vpm.ndim == 2:
            # Single matrix - treat as single time step
            series = [vpm]
        elif vpm.ndim == 3:
            # Time series of matrices
            series = [vpm[t] for t in range(vpm.shape[0])]
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")
        
        try:
            M = series[0].shape[1]
            w_star = np.ones(M) / np.sqrt(M)  # Default equal weights
            
            # Only learn weights if we have meaningful data
            if len(series) > 1 or series[0].shape[0] > 1:
                try:
                    # Learn optimal weights that maximize top-left signal
                    w_star = learn_w(
                        series=series,
                        Kc=self.Kc,
                        Kr=self.Kr,
                        u_mode=self.u_mode,
                        alpha=self.alpha,
                        l2=self.l2,
                        iters=self.iters,
                        step=self.step,
                        seed=0
                    )
                    
                    # Check if optimization succeeded
                    if np.allclose(w_star, w_star[0]):
                        logger.warning("STDM optimization failed - learned uniform weights")
                        raise ValueError("Optimization failed")
                        
                except Exception as e:
                    logger.warning(f"STDM weight learning failed: {e}, using variance-based weights")
                    # Use variance of each metric as weights
                    stacked = np.stack(series, axis=0)
                    w_star = np.var(stacked, axis=(0,1))
                    # Add small random noise to break ties
                    w_star += np.random.normal(0, 1e-6, M)
                    w_star = w_star / (np.linalg.norm(w_star) + 1e-12)
            else:
                logger.info("Insufficient data for weight learning, using equal weights")
            
            # Reorder to concentrate signal in top-left
            u_fn = lambda t, Xt: w_star
            Ys, col_orders, row_orders = gamma_operator(series, u_fn=u_fn, w=w_star, Kc=self.Kc)
            
            # Apply learned weights to amplify signal
            Ys = [np.maximum(0.0, Y * w_star[None, :]) for Y in Ys]
            
            # Calculate diagnostics
            tl_mass = np.mean([top_left_mass(Y, Kr=self.Kr, Kc=self.Kc, alpha=self.alpha) for Y in Ys])
            
            # Convert back to VPM format
            if len(Ys) > 1:
                processed_vpm = np.stack(Ys, axis=0)
            else:
                processed_vpm = Ys[0]
            
            # Calculate weight variance for diagnostics
            weight_variance = float(np.var(w_star))
            
            metadata = {
                "w_star": w_star.tolist(),
                "tl_mass_avg": float(tl_mass),
                "Kc": self.Kc,
                "Kr": self.Kr,
                "alpha": self.alpha,
                "u_mode": self.u_mode,
                "stage": "stmd_amplifier",
                "input_shape": vpm.shape,
                "output_shape": processed_vpm.shape,
                "weight_variance": weight_variance,
                "optimization_success": weight_variance > 1e-6
            }
            
            return processed_vpm, metadata
            
        except Exception as e:
            logger.error(f"STDM amplification failed: {e}")
            # Return original VPM and error metadata
            return vpm, {"error": str(e), "stage": "stmd_amplifier", "optimization_success": False}