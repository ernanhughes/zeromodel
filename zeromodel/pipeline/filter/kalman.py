# zeromodel/pipeline/stages/filters/kalman.py
"""
Kalman filter stage for ZeroModel.

This implements ZeroModel's "planet-scale navigation that feels flat" principle:
"Whether it's 10K docs or a trillion, you descend in dozens of steps, not millions."
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)


class KalmanFilter(PipelineStage):
    """Kalman filter stage for ZeroModel."""

    name = "kalman"
    category = "filter"

    def __init__(self, **params):
        super().__init__(**params)
        self.process_noise = params.get("process_noise", 1e-4)
        self.measurement_noise = params.get("measurement_noise", 1e-2)
        self.initial_estimate_error = params.get("initial_estimate_error", 1.0)

    def validate_params(self):
        """Validate Kalman filter parameters."""
        if self.process_noise <= 0:
            raise ValueError("process_noise must be positive")
        if self.measurement_noise <= 0:
            raise ValueError("measurement_noise must be positive")
        if self.initial_estimate_error <= 0:
            raise ValueError("initial_estimate_error must be positive")

    def process(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Kalman filtering to a time-series VPM.

        This smooths the signal over time, reducing noise while preserving trends.
        """
        context = self._get_context(context)

        if vpm.ndim != 3:
            logger.warning(
                "Kalman filter requires 3D VPM (time series), returning original"
            )
            return vpm, {
                "warning": "Kalman filter requires 3D VPM",
                "kalman_applied": False,
            }

        T, N, M = vpm.shape

        # Initialize Kalman filter parameters
        Q = self.process_noise  # Process noise covariance
        R = self.measurement_noise  # Measurement noise covariance
        P = self.initial_estimate_error  # Estimate error covariance

        # Process each pixel independently
        filtered_vpm = np.zeros_like(vpm)

        for i in range(N):
            for j in range(M):
                # Extract time series for this pixel
                measurements = vpm[:, i, j]

                # Initialize state estimate
                x_hat = measurements[0]  # Initial estimate
                P_k = P  # Initial error covariance

                # Apply Kalman filter
                filtered_vpm[0, i, j] = x_hat  # First value is the same

                for k in range(1, T):
                    # Prediction step
                    x_hat_minus = x_hat  # Assume constant model
                    P_minus = P_k + Q

                    # Update step
                    K = P_minus / (P_minus + R)  # Kalman gain
                    x_hat = x_hat_minus + K * (measurements[k] - x_hat_minus)
                    P_k = (1 - K) * P_minus

                    filtered_vpm[k, i, j] = x_hat

        # Calculate diagnostics
        noise_reduction = float(np.mean((vpm - filtered_vpm) ** 2))

        metadata = {
            "process_noise": self.process_noise,
            "measurement_noise": self.measurement_noise,
            "initial_estimate_error": self.initial_estimate_error,
            "input_shape": vpm.shape,
            "output_shape": filtered_vpm.shape,
            "noise_reduction": noise_reduction,
            "kalman_applied": True,
        }
        
        return filtered_vpm, metadata