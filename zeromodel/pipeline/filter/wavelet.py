#  zeromodel/pipeline/filter/wavelet.py
"""
Wavelet filter stage for ZeroModel.

This implements ZeroModel's "robust under pressure" principle:
"Versioned headers, spillover-safe metadata, and explicit logical width vs physical padding
keep tiles valid as they scale."
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pywt

from zeromodel.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)


class WaveletFilter(PipelineStage):
    """Wavelet filter stage for ZeroModel."""

    name = "wavelet"
    category = "filter"

    def __init__(self, **params):
        super().__init__(**params)
        self.wavelet = params.get("wavelet", "haar")
        self.level = params.get("level", 3)
        self.mode = params.get("mode", "soft")
        self.threshold_factor = params.get("threshold_factor", 2.0)

    def validate_params(self):
        """Validate wavelet parameters."""
        if self.level <= 0:
            raise ValueError("level must be positive")
        if self.mode not in ["soft", "hard"]:
            raise ValueError("mode must be 'soft' or 'hard'")
        if self.threshold_factor <= 0:
            raise ValueError("threshold_factor must be positive")

    def process(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply wavelet denoising to a VPM.

        This removes noise while preserving important signal features.
        """
        context = self.get_context(context)

        # Handle different VPM dimensions
        if vpm.ndim == 2:
            # Single matrix
            processed_vpm = self._process_single(vpm)
        elif vpm.ndim == 3:
            # Time series - apply to each frame
            processed_frames = [
                self._process_single(vpm[t]) for t in range(vpm.shape[0])
            ]
            processed_vpm = np.stack(processed_frames, axis=0)
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")

        # Calculate diagnostics
        input_energy = np.sum(vpm**2)
        output_energy = np.sum(processed_vpm**2)
        energy_ratio = output_energy / (input_energy + 1e-12)

        metadata = {
            "wavelet": self.wavelet,
            "level": self.level,
            "mode": self.mode,
            "threshold_factor": self.threshold_factor,
            "input_shape": vpm.shape,
            "output_shape": processed_vpm.shape,
            "energy_ratio": float(energy_ratio),
            "denoising_applied": True,
        }

        return processed_vpm, metadata

    def _process_single(self, matrix: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising to a single matrix."""
        try:
            # Apply wavelet transform
            coeffs = pywt.wavedec2(matrix, self.wavelet, level=self.level)

            # Calculate threshold based on noise level
            if coeffs and len(coeffs) > 1:
                # Use the detail coefficients to estimate noise
                noise_std = np.std(coeffs[-1])
                threshold = (
                    self.threshold_factor * noise_std * np.sqrt(2 * np.log(matrix.size))
                )
            else:
                threshold = 0.1  # Fallback threshold

            # Apply thresholding
            if self.mode == "soft":
                coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
            else:  # hard
                coeffs = [pywt.threshold(c, threshold, mode="hard") for c in coeffs]

            # Reconstruct
            denoised = pywt.waverec2(coeffs, self.wavelet)

            # Ensure same shape as input
            if denoised.shape != matrix.shape:
                # Crop or pad to match
                slices = tuple(
                    slice(0, min(s1, s2))
                    for s1, s2 in zip(denoised.shape, matrix.shape)
                )
                denoised = denoised[slices]
                pad_width = [
                    (0, max(0, s2 - s1)) for s1, s2 in zip(denoised.shape, matrix.shape)
                ]
                denoised = np.pad(denoised, pad_width, mode="constant")

            return denoised

        except Exception as e:
            logger.warning(f"Wavelet filtering failed: {e}, returning original")
            return matrix