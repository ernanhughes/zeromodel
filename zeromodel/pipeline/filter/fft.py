# zeromodel/pipeline/stages/filters/fft.py
"""
FFT (Fast Fourier Transform) filter stage for ZeroModel.

This implements ZeroModel's "edge ↔ cloud symmetry" principle:
"The same artifact works everywhere - from microcontrollers to data centers."
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)

class FFTFilter(PipelineStage):
    """FFT filter stage for ZeroModel."""
    
    name = "fft"
    category = "filter"
    
    def __init__(self, **params):
        super().__init__(**params)
        self.filter_type = params.get("filter_type", "bandpass")
        self.low_freq = params.get("low_freq", 0.1)
        self.high_freq = params.get("high_freq", 0.4)
        self.normalize = params.get("normalize", True)
    
    def validate_params(self):
        """Validate FFT parameters."""
        if self.filter_type not in ["lowpass", "highpass", "bandpass", "bandstop"]:
            raise ValueError("filter_type must be one of: lowpass, highpass, bandpass, bandstop")
        if self.low_freq < 0 or self.low_freq >= 0.5:
            raise ValueError("low_freq must be in [0, 0.5)")
        if self.high_freq <= 0 or self.high_freq > 0.5:
            raise ValueError("high_freq must be in (0, 0.5]")
        if self.low_freq >= self.high_freq:
            raise ValueError("low_freq must be less than high_freq")
    
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply FFT filtering to a VPM.
        
        This removes frequency components outside the specified range,
        effectively filtering out periodic noise or enhancing periodic signals.
        """
        context = self._get_context(context)
        
        # Handle different VPM dimensions
        if vpm.ndim == 2:
            # Single matrix
            processed_vpm = self._process_single(vpm)
        elif vpm.ndim == 3:
            # Time series - apply to each frame
            processed_frames = [self._process_single(vpm[t]) for t in range(vpm.shape[0])]
            processed_vpm = np.stack(processed_frames, axis=0)
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")
        
        # Calculate diagnostics
        input_mean = float(np.mean(vpm))
        output_mean = float(np.mean(processed_vpm))
        mean_change = output_mean - input_mean
        
        metadata = {
            "filter_type": self.filter_type,
            "low_freq": self.low_freq,
            "high_freq": self.high_freq,
            "normalize": self.normalize,
            "input_shape": vpm.shape,
            "output_shape": processed_vpm.shape,
            "mean_change": mean_change,
            "fft_applied": True
        }
        
        return processed_vpm, metadata
    
    def _process_single(self, matrix: np.ndarray) -> np.ndarray:
        """Apply FFT filtering to a single matrix."""
        try:
            # Apply 2D FFT
            fft = np.fft.fft2(matrix)
            fft_shifted = np.fft.fftshift(fft)
            
            # Create frequency domain mask
            h, w = matrix.shape
            center_h, center_w = h // 2, w // 2
            
            # Create frequency grid
            y = np.linspace(-0.5, 0.5, h)
            x = np.linspace(-0.5, 0.5, w)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            
            # Create filter mask
            mask = np.ones_like(R)
            if self.filter_type == "lowpass":
                mask[R > self.high_freq] = 0
            elif self.filter_type == "highpass":
                mask[R < self.low_freq] = 0
            elif self.filter_type == "bandpass":
                mask[(R < self.low_freq) | (R > self.high_freq)] = 0
            elif self.filter_type == "bandstop":
                mask[(R >= self.low_freq) & (R <= self.high_freq)] = 0
            
            # Apply mask
            fft_filtered = fft_shifted * mask
            
            # Inverse FFT
            fft_unshifted = np.fft.ifftshift(fft_filtered)
            filtered = np.fft.ifft2(fft_unshifted).real
            
            # Normalize if requested
            if self.normalize:
                filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-12)
                filtered = filtered * (matrix.max() - matrix.min()) + matrix.min()
            
            return filtered
            
        except Exception as e:
            logger.warning(f"FFT filtering failed: {e}, returning original")
            return matrix