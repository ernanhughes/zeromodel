# zeromodel/pipeline/stages/amplifiers/pca.py
"""
PCA (Principal Component Analysis) amplifier stage for ZeroModel.

This implements ZeroModel's "intelligence lives in the data structure" principle
by learning the principal components that explain most variance in the data.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA

from zeromodel.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)


class PCAAmplifier(PipelineStage):
    """PCA amplifier stage for ZeroModel."""

    name = "pca"
    category = "amplifier"

    def __init__(self, **params):
        super().__init__(**params)
        self.n_components = params.get("n_components", 10)
        self.whiten = params.get("whiten", False)
        self.explained_variance_ratio = params.get("explained_variance_ratio", None)

    def validate_params(self):
        """Validate PCA parameters."""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if self.explained_variance_ratio is not None and not (
            0 < self.explained_variance_ratio <= 1
        ):
            raise ValueError("explained_variance_ratio must be in (0,1]")

    def process(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply PCA to amplify signal in a VPM.

        This transforms the data to a new coordinate system where the first few components
        capture most of the variance, effectively amplifying the most important signals.
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
            # Stack all matrices for PCA fitting
            stacked = np.vstack(series)
            N, M = stacked.shape

            # Determine number of components
            n_components = self.n_components
            if self.explained_variance_ratio is not None:
                # Fit PCA to determine components needed for variance ratio
                temp_pca = PCA()
                temp_pca.fit(stacked)
                cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= self.explained_variance_ratio) + 1
                n_components = max(1, min(n_components, M))

            n_components = min(n_components, N, M)

            # Fit PCA on the stacked data
            pca = PCA(n_components=n_components, whiten=self.whiten)
            pca.fit(stacked)

            # Transform each matrix
            transformed_series = []
            for matrix in series:
                # Project to principal components
                transformed = pca.transform(matrix)

                # Inverse transform to get amplified version
                if self.whiten:
                    # When whitened, need to scale back
                    transformed = pca.inverse_transform(transformed)
                else:
                    # Scale by explained variance
                    explained_variance = pca.explained_variance_[:n_components]
                    scaling = np.sqrt(explained_variance)
                    scaled_transformed = transformed * scaling[None, :]
                    transformed = pca.inverse_transform(scaled_transformed)

                transformed_series.append(transformed)

            # Convert back to VPM format
            if len(transformed_series) > 1:
                processed_vpm = np.stack(transformed_series, axis=0)
            else:
                processed_vpm = transformed_series[0]

            # Calculate diagnostics
            explained_variance_ratio = pca.explained_variance_ratio_
            total_variance_explained = float(np.sum(explained_variance_ratio))

            metadata = {
                "n_components": n_components,
                "whiten": self.whiten,
                "explained_variance_ratio": explained_variance_ratio.tolist(),
                "total_variance_explained": total_variance_explained,
                "components_shape": pca.components_.shape,
                "input_shape": vpm.shape,
                "output_shape": processed_vpm.shape,
                "amplification_applied": True,
            }

            return processed_vpm, metadata

        except Exception as e:
            logger.error(f"PCA amplification failed: {e}")
            # Return original VPM and error metadata
            return vpm, {"error": str(e), "stage": "pca_amplifier"}