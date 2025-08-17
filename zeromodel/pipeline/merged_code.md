<!-- Merged Python Code Files -->


## File: __init__.py

`python
``n

## File: amplifier\__init__.py

`python
``n

## File: amplifier\pca.py

`python
# zeromodel/pipeline/stages/amplifiers/pca.py
"""
PCA (Principal Component Analysis) amplifier stage for ZeroModel.

This implements ZeroModel's "intelligence lives in the data structure" principle
by learning the principal components that explain most variance in the data.
"""

from typing import Dict, Any, Tuple
import numpy as np
from zeromodel.pipeline.base import PipelineStage
from sklearn.decomposition import PCA
import logging

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
    
    def _validate_params(self):
        """Validate PCA parameters."""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if self.explained_variance_ratio is not None and not (0 < self.explained_variance_ratio <= 1):
            raise ValueError("explained_variance_ratio must be in (0,1]")
    
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply PCA to amplify signal in a VPM.
        
        This transforms the data to a new coordinate system where the first few components
        capture most of the variance, effectively amplifying the most important signals.
        """
        context = self._get_context(context)
        self._record_provenance(context, self.name, self.params)
        
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
                "amplification_applied": True
            }
            
            return processed_vpm, metadata
            
        except Exception as e:
            logger.error(f"PCA amplification failed: {e}")
            # Return original VPM and error metadata
            return vpm, {"error": str(e), "stage": "pca_amplifier"}
``n

## File: amplifier\stdm.py

`python
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

                    # Safety net: if optimizer ever returns near-uniform, rebuild from series
                    if np.var(w_star) < 1e-8:
                        logger.warning("STDMAmplifier: learned weights near-uniform; applying series-based fallback")
                        # deterministic fallback identical to learn_w’s heuristic
                        col_mean = np.mean([Xt.mean(axis=0) for Xt in series], axis=0)
                        col_std  = np.mean([np.sqrt(np.var(Xt, axis=0) + 1e-12) for Xt in series], axis=0)
                        w_star = 0.6 * col_mean + 0.4 * col_std
                        w_star = np.maximum(0.0, w_star ** 1.3)
                        w_star = w_star / (np.linalg.norm(w_star) + 1e-12)

                    
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
``n

## File: base.py

`python
# zeromodel/pipeline/base.py
"""
Base classes for ZeroModel pipeline stages.

Implements ZeroModel's "dumb pipe" communication model:
"Because the core representation is a standardized image (VPM tile),
the communication protocol becomes extremely simple and universally understandable."
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

class PipelineStage(ABC):
    """
    Base class for all ZeroModel pipeline stages.
    
    This implements ZeroModel's "intelligence lives in the data structure" principle:
    The processing is minimal - the intelligence is in how the data is organized.
    """
    
    name: str = "base"
    category: str = "base"
    
    def __init__(self, **params):
        self.params = params
        
    @abstractmethod
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a VPM and return (transformed_vpm, metadata).
        
        Args:
            vpm: Input VPM as numpy array
            context: Optional context dictionary with pipeline state
            
        Returns:
            (transformed_vpm, metadata) - Enhanced VPM and diagnostic metadata
        """
        pass
    
    @abstractmethod
    def validate_params(self):
        """Validate stage parameters."""
        pass
    
    def _get_context(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get or create context dictionary."""
        if context is None:
            context = {}
        if 'provenance' not in context:
            context['provenance'] = []
        return context
   
``n

## File: combiner\and.py

`python
# zeromodel/pipeline/combiner/and.py
"""
Logical AND combiner for ZeroModel.

This implements ZeroModel's "symbolic logic in the data" principle:
Instead of running a neural model, we run fuzzy logic on structured images.
"""

from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage


class AndCombiner(PipelineStage):
    """Logical AND combiner stage for ZeroModel."""
    
    name = "and"
    category = "combiner"
    
    def __init__(self, **params):
        super().__init__(**params)
        self.threshold = params.get("threshold", 0.5)
    
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply logical AND to a VPM (assuming multiple channels represent different conditions).
        
        This creates a compound reasoning structure through pixel-wise arithmetic.
        """
        context = self._get_context(context)
        
        if vpm.ndim < 3:
            # Not enough dimensions for AND operation
            return vpm, {"warning": "VPM has <3 dimensions, skipping AND operation"}
        
        # Apply AND across channels (last dimension)
        # Convert to binary using threshold, then AND
        binary_vpm = (vpm > self.threshold).astype(float)
        processed_vpm = np.all(binary_vpm, axis=-1).astype(float)
        
        metadata = {
            "threshold": self.threshold,
            "input_shape": vpm.shape,
            "output_shape": processed_vpm.shape,
            "operation": "AND",
            "channels_combined": vpm.shape[-1]
        }
        
        return processed_vpm, metadata
``n

## File: executor.py

`python
# zeromodel/pipeline/executor.py
import importlib
import time
import numpy as np
from typing import Any, Dict, List, Tuple
import logging
from zeromodel.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)

class PipelineExecutor:
    def __init__(self, stages: List[Dict[str, Any]]):
        self.stages = stages
        logger.info(f"PipelineExecutor initialized with {len(stages)} stages")

    def _load_stage(self, stage_path: str, params: Dict[str, Any]) -> PipelineStage:
        # Supports "pkg/subpkg.ClassName" or "pkg/subpkg" (uses first PipelineStage subclass)
        if "." in stage_path:
            pkg, clsname = stage_path.rsplit(".", 1)
        else:
            pkg, clsname = stage_path, None

        module_path = f"zeromodel.pipeline.{pkg.replace('/', '.')}"
        module = importlib.import_module(module_path)

        cls = getattr(module, clsname) if clsname else None
        if cls is None:
            # fallback: pick first subclass of PipelineStage in the module
            for attr in module.__dict__.values():
                if isinstance(attr, type) and issubclass(attr, PipelineStage) and attr is not PipelineStage:
                    cls = attr
                    break
        if cls is None:
            raise ImportError(f"No PipelineStage subclass found in {module_path}")

        inst = cls(**params)
        return inst

    def _init_context(self, context: Dict[str, Any] | None) -> Dict[str, Any]:
        ctx = {} if context is None else dict(context)
        ctx.setdefault("provenance", [])
        ctx.setdefault("pipeline_start_time", np.datetime64("now"))
        ctx.setdefault("stats", {})
        return ctx

    def _record(self, ctx: Dict[str, Any], **event):
        ctx["provenance"].append({"timestamp": np.datetime64("now"), **event})

    def run(self, vpm: np.ndarray, context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self._init_context(context)
        cur = vpm

        for i, spec in enumerate(self.stages):
            stage_path = spec["stage"]
            params = spec.get("params", {})

            # stage start event
            self._record(ctx,
                kind="stage_start",
                stage=stage_path,
                index=i,
                params=params,
                input_shape=tuple(cur.shape),
            )

            t0 = time.time()
            try:
                stage = self._load_stage(stage_path, params)
                stage.validate_params()

                out, meta = stage.process(cur, ctx)
                dt = time.time() - t0

                # stage end (success)
                self._record(ctx,
                    kind="stage_end",
                    stage=stage_path,
                    index=i,
                    ok=True,
                    elapsed_sec=dt,
                    output_shape=tuple(out.shape),
                    metadata=meta or {},
                )

                # per-stage convenience block
                ctx[f"stage_{i}"] = {
                    "stage": stage_path,
                    "params": params,
                    "elapsed_sec": dt,
                    "input_shape": tuple(cur.shape),
                    "output_shape": tuple(out.shape),
                    "metadata": meta or {},
                }
                cur = out

            except Exception as e:
                dt = time.time() - t0
                logger.exception(f"Stage {stage_path} failed")
                # stage end (failure)
                self._record(ctx,
                    kind="stage_end",
                    stage=stage_path,
                    index=i,
                    ok=False,
                    elapsed_sec=dt,
                    error=str(e),
                )
                ctx[f"stage_{i}_error"] = {
                    "stage": stage_path,
                    "error": str(e),
                    "elapsed_sec": dt,
                    "timestamp": np.datetime64("now"),
                }
                # passthrough: keep cur unchanged and continue

        # final stats
        ctx["final_stats"] = {
            "vpm_shape": tuple(cur.shape),
            "vpm_min": float(np.min(cur)),
            "vpm_max": float(np.max(cur)),
            "vpm_mean": float(np.mean(cur)),
            "pipeline_stages": len(self.stages),
            "total_execution_time": float(sum(
                ctx.get(f"stage_{i}", {}).get("elapsed_sec", 0.0)
                for i in range(len(self.stages))
            )),
        }
        return cur, ctx
``n

## File: filter\__init__.py

`python
``n

## File: filter\fft.py

`python
# zeromodel/pipeline/stages/filters/fft.py
"""
FFT (Fast Fourier Transform) filter stage for ZeroModel.

This implements ZeroModel's "edge ↔ cloud symmetry" principle:
"The same artifact works everywhere - from microcontrollers to data centers."
"""

from typing import Dict, Any, Tuple
import numpy as np
from zeromodel.pipeline.base import PipelineStage
import logging

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
``n

## File: filter\kalman.py

`python
# zeromodel/pipeline/stages/filters/kalman.py
"""
Kalman filter stage for ZeroModel.

This implements ZeroModel's "planet-scale navigation that feels flat" principle:
"Whether it's 10K docs or a trillion, you descend in dozens of steps, not millions."
"""

from typing import Dict, Any, Tuple
import numpy as np
from zeromodel.pipeline.base import PipelineStage
import logging

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
    
    def _validate_params(self):
        """Validate Kalman filter parameters."""
        if self.process_noise <= 0:
            raise ValueError("process_noise must be positive")
        if self.measurement_noise <= 0:
            raise ValueError("measurement_noise must be positive")
        if self.initial_estimate_error <= 0:
            raise ValueError("initial_estimate_error must be positive")
    
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Kalman filtering to a time-series VPM.
        
        This smooths the signal over time, reducing noise while preserving trends.
        """
        context = self._get_context(context)
        self._record_provenance(context, self.name, self.params)
        
        if vpm.ndim != 3:
            logger.warning("Kalman filter requires 3D VPM (time series), returning original")
            return vpm, {"warning": "Kalman filter requires 3D VPM", "kalman_applied": False}
        
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
            "kalman_applied": True
        }
        
        return filtered_vpm, metadata
``n

## File: filter\morphological.py

`python
# zeromodel/pipeline/stages/filters/morphological.py
"""
Morphological filter stage for ZeroModel.

This implements ZeroModel's "symbolic logic in the data" principle:
Instead of running a neural model, we run fuzzy logic on structured images.
"""

from typing import Dict, Any, Tuple
import numpy as np
from zeromodel.pipeline.base import PipelineStage
from scipy import ndimage

class MorphologicalFilter(PipelineStage):
    """Morphological filter stage for ZeroModel."""
    
    name = "morphological"
    category = "filter"
    
    def __init__(self, **params):
        super().__init__(**params)
        self.operation = params.get("operation", "opening")
        self.kernel_size = params.get("kernel_size", 3)
        self.iterations = params.get("iterations", 1)
    
    def validate_params(self):
        """Validate morphological parameters."""
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.operation not in ["opening", "closing", "erosion", "dilation"]:
            raise ValueError("operation must be one of: opening, closing, erosion, dilation")
    
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply morphological operations to a VPM.
        
        This enhances or suppresses features based on their shape and size.
        """
        context = self._get_context(context)
        
        # Create structuring element (square kernel)
        kernel = np.ones((self.kernel_size, self.kernel_size))
        
        # Handle different VPM dimensions
        if vpm.ndim == 2:
            # Single matrix
            processed_vpm = self._apply_morphology(vpm, kernel)
        elif vpm.ndim == 3:
            # Time series - apply to each frame
            processed_frames = [self._apply_morphology(vpm[t], kernel) for t in range(vpm.shape[0])]
            processed_vpm = np.stack(processed_frames, axis=0)
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")
        
        metadata = {
            "operation": self.operation,
            "kernel_size": self.kernel_size,
            "iterations": self.iterations,
            "input_shape": vpm.shape,
            "output_shape": processed_vpm.shape,
            "morphology_applied": True
        }
        
        return processed_vpm, metadata
    
    def _apply_morphology(self, matrix: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply morphological operation to a single matrix."""
        # Normalize to [0,1] for consistent processing
        matrix_min, matrix_max = matrix.min(), matrix.max()
        if matrix_max > matrix_min:
            normalized = (matrix - matrix_min) / (matrix_max - matrix_min)
        else:
            normalized = matrix.copy()
        
        # Apply morphological operation
        if self.operation == "opening":
            result = ndimage.morphology.binary_opening(normalized, structure=kernel, iterations=self.iterations)
        elif self.operation == "closing":
            result = ndimage.morphology.binary_closing(normalized, structure=kernel, iterations=self.iterations)
        elif self.operation == "erosion":
            result = ndimage.morphology.binary_erosion(normalized, structure=kernel, iterations=self.iterations)
        elif self.operation == "dilation":
            result = ndimage.morphology.binary_dilation(normalized, structure=kernel, iterations=self.iterations)
        
        # Restore original scale
        if matrix_max > matrix_min:
            result = result * (matrix_max - matrix_min) + matrix_min
        else:
            result = result * matrix_max
        
        return result
``n

## File: filter\wavelet.py

`python
# zeromodel/pipeline/stages/filters/wavelet.py
"""
Wavelet filter stage for ZeroModel.

This implements ZeroModel's "robust under pressure" principle:
"Versioned headers, spillover-safe metadata, and explicit logical width vs physical padding
keep tiles valid as they scale."
"""

from typing import Dict, Any, Tuple
import numpy as np
from zeromodel.pipeline.base import PipelineStage
import pywt
import logging

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
    
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply wavelet denoising to a VPM.
        
        This removes noise while preserving important signal features.
        """
        context = self._get_context(context)
        self._record_provenance(context, self.name, self.params)
        
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
        input_energy = np.sum(vpm ** 2)
        output_energy = np.sum(processed_vpm ** 2)
        energy_ratio = output_energy / (input_energy + 1e-12)
        
        metadata = {
            "wavelet": self.wavelet,
            "level": self.level,
            "mode": self.mode,
            "threshold_factor": self.threshold_factor,
            "input_shape": vpm.shape,
            "output_shape": processed_vpm.shape,
            "energy_ratio": float(energy_ratio),
            "denoising_applied": True
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
                threshold = self.threshold_factor * noise_std * np.sqrt(2 * np.log(matrix.size))
            else:
                threshold = 0.1  # Fallback threshold
            
            # Apply thresholding
            if self.mode == "soft":
                coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            else:  # hard
                coeffs = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]
            
            # Reconstruct
            denoised = pywt.waverec2(coeffs, self.wavelet)
            
            # Ensure same shape as input
            if denoised.shape != matrix.shape:
                # Crop or pad to match
                slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(denoised.shape, matrix.shape))
                denoised = denoised[slices]
                pad_width = [(0, max(0, s2-s1)) for s1, s2 in zip(denoised.shape, matrix.shape)]
                denoised = np.pad(denoised, pad_width, mode='constant')
            
            return denoised
            
        except Exception as e:
            logger.warning(f"Wavelet filtering failed: {e}, returning original")
            return matrix
``n

## File: organizer\__init__.py

`python
``n

## File: organizer\top_left_sort.py

`python
# zeromodel/pipeline/organizer/top_left_sort.py
"""
Top-left sorter for ZeroModel.

This implements ZeroModel's "top-left rule" for signal concentration:
The most important information is always in the top-left corner.
"""

from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage


class TopLeftSorter(PipelineStage):
    """Top-left sorter stage for ZeroModel."""
    
    name = "top_left_sort"
    category = "organizer"
    
    def __init__(self, **params):
        super().__init__(**params)
        self.metric = params.get("metric", "variance")
        self.Kc = params.get("Kc", 12)
    
    def validate_params(self):
        """Validate stage parameters."""
        pass

    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reorder VPM to concentrate signal in top-left corner.
        
        This implements ZeroModel's "intelligence lives in the data structure" principle:
        The processing is minimal - the intelligence is in how the data is organized.
        """
        context = self._get_context(context)
        
        if vpm.ndim == 2:
            # Single matrix
            processed_vpm = self._sort_matrix(vpm)
        elif vpm.ndim == 3:
            # Time series - apply to each frame
            processed_frames = [self._sort_matrix(vpm[t]) for t in range(vpm.shape[0])]
            processed_vpm = np.stack(processed_frames, axis=0)
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")
        
        metadata = {
            "metric": self.metric,
            "Kc": self.Kc,
            "input_shape": vpm.shape,
            "output_shape": processed_vpm.shape,
            "reordering_applied": True
        }
        
        return processed_vpm, metadata
    
    def _sort_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Sort matrix to concentrate signal in top-left."""
        # Calculate column importance based on selected metric
        if self.metric == "variance":
            col_importance = np.var(matrix, axis=0)
        elif self.metric == "mean":
            col_importance = np.mean(matrix, axis=0)
        elif self.metric == "sum":
            col_importance = np.sum(matrix, axis=0)
        else:
            col_importance = np.var(matrix, axis=0)  # Default
        
        # Sort columns by importance (descending)
        col_order = np.argsort(-col_importance)
        matrix_sorted = matrix[:, col_order]
        
        # Calculate row importance based on top-Kc columns
        Kc_actual = min(self.Kc, matrix_sorted.shape[1])
        if Kc_actual > 0:
            row_importance = np.sum(matrix_sorted[:, :Kc_actual], axis=1)
        else:
            row_importance = np.sum(matrix_sorted, axis=1)
        
        # Sort rows by importance (descending)
        row_order = np.argsort(-row_importance)
        matrix_sorted = matrix_sorted[row_order, :]
        
        return matrix_sorted
``n
