# zeromodel/pipeline/filter/wavelet.py
"""
Wavelet filter for ZeroModel.

This implements ZeroModel's "robust under pressure" principle:
"Versioned headers, spillover-safe metadata, and explicit logical width vs physical padding
keep tiles valid as they scale."
"""

from typing import Any, Dict, Tuple

import numpy as np
import pywt

from zeromodel.pipeline.base import PipelineStage


class WaveletFilter(PipelineStage):
    """Wavelet filter stage for ZeroModel."""
    
    name = "wavelet"
    category = "filter"
    
    def __init__(self, **params):
        super().__init__(**params)
        self.wavelet = params.get("wavelet", "haar")
        self.level = params.get("level", 3)
        self.mode = params.get("mode", "soft")
    
    def validate_params(self):
        """Validate wavelet parameters."""
        if self.level <= 0:
            raise ValueError("level must be positive")
        if self.mode not in ["soft", "hard"]:
            raise ValueError("mode must be 'soft' or 'hard'")
    
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply wavelet denoising to a VPM.
        
        This removes noise while preserving important signal features.
        """
        context = self._get_context(context)
       
        # Handle different VPM dimensions
        if vpm.ndim == 2:
            # Single matrix
            coeffs = pywt.wavedec2(vpm, self.wavelet, level=self.level)
            
            # Apply thresholding
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(vpm.size))
            coeffs = [pywt.threshold(c, threshold, mode=self.mode) for c in coeffs]
            
            # Reconstruct
            processed_vpm = pywt.waverec2(coeffs, self.wavelet)
            
        elif vpm.ndim == 3:
            # Time series - apply to each frame
            processed_frames = []
            for t in range(vpm.shape[0]):
                coeffs = pywt.wavedec2(vpm[t], self.wavelet, level=self.level)
                threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(vpm[t].size))
                coeffs = [pywt.threshold(c, threshold, mode=self.mode) for c in coeffs]
                frame = pywt.waverec2(coeffs, self.wavelet)
                processed_frames.append(frame)
            
            processed_vpm = np.stack(processed_frames, axis=0)
            
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")
        
        # Ensure same shape as input
        if processed_vpm.shape != vpm.shape:
            # Crop or pad to match
            slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(processed_vpm.shape, vpm.shape))
            processed_vpm = processed_vpm[slices]
            # Pad if necessary
            pad_width = [(0, max(0, s2-s1)) for s1, s2 in zip(processed_vpm.shape, vpm.shape)]
            processed_vpm = np.pad(processed_vpm, pad_width, mode='constant')
        
        metadata = {
            "wavelet": self.wavelet,
            "level": self.level,
            "mode": self.mode,
            "input_shape": vpm.shape,
            "output_shape": processed_vpm.shape,
            "denoising_applied": True
        }
        
        return processed_vpm, metadata