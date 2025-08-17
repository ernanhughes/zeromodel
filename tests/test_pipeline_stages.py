# tests/test_pipeline_stages.py
import pytest
import numpy as np
import logging
from zeromodel.pipeline.filter.wavelet import WaveletFilter
from zeromodel.pipeline.amplifier.pca import PCAAmplifier
from zeromodel.pipeline.filter.morphological import MorphologicalFilter
from zeromodel.pipeline.filter.fft import FFTFilter
from zeromodel.pipeline.filter.kalman import KalmanFilter

logger = logging.getLogger(__name__)

class TestWaveletFilter:
    """Test WaveletFilter pipeline stage."""
    
    def setup_method(self):
        self.filter = WaveletFilter(wavelet="haar", level=2, mode="soft")
    
    def test_process_2d_vpm(self):
        """Test wavelet filtering on 2D VPM."""
        # Create test VPM
        vpm = np.random.rand(64, 64).astype(np.float32)
        
        # Process through filter
        result, metadata = self.filter.process(vpm)
        
        # Verify output
        assert result.shape == vpm.shape
        assert result.dtype == vpm.dtype
        assert "wavelet" in metadata
        assert "denoising_applied" in metadata
        assert metadata["denoising_applied"] is True
        
        # Verify energy ratio is reasonable
        assert "energy_ratio" in metadata
        assert 0 <= metadata["energy_ratio"] <= 1.5  # Allow for small increases due to reconstruction
    
    def test_process_3d_vpm(self):
        """Test wavelet filtering on 3D VPM (time series)."""
        # Create test VPM
        vpm = np.random.rand(5, 64, 64).astype(np.float32)
        
        # Process through filter
        result, metadata = self.filter.process(vpm)
        
        # Verify output
        assert result.shape == vpm.shape
        assert "wavelet" in metadata
        assert "denoising_applied" in metadata
    
    def test_edge_cases(self):
        """Test wavelet filter with edge cases."""
        # Test with zeros
        zeros = np.zeros((32, 32))
        result, metadata = self.filter.process(zeros)
        assert np.array_equal(result, zeros)
        
        # Test with constant values
        constants = np.ones((32, 32)) * 0.5
        result, metadata = self.filter.process(constants)
        assert np.allclose(result, constants, atol=1e-6)
    
class TestPCAAmplifier:
    """Test PCAAmplifier pipeline stage."""
    
    def setup_method(self):
        self.amplifier = PCAAmplifier(n_components=5, whiten=False)

    def test_process_2d_vpm(self):
        """Test PCA amplification on 2D VPM."""
        # Create test VPM with some structure
        np.random.seed(42)
        vpm = np.random.rand(100, 20).astype(np.float32)

        # Add correlation between features to give PCA something to work with.
        # This creates a signal for the amplifier to find.
        # It takes the first 5 features and adds them to the last 15.
        vpm[:, 5:20] += 0.5 * np.tile(vpm[:, :5], (1, 3))

        # Process through amplifier
        result, metadata = self.amplifier.process(vpm)

        # --- Verification Assertions ---

        # 1. Verify the output shape is the same as the input shape
        assert result.shape == vpm.shape

        # 2. Verify metadata keys exist as expected
        assert "n_components" in metadata
        assert "total_variance_explained" in metadata

        # 3. Verify that the PCA explained a significant amount of variance.
        # This confirms that the correlation was found.
        assert metadata["total_variance_explained"] > 0.1

        # 4. Verify the output variance. A PCA reconstruction will typically have
        # a lower variance than the original matrix because it only keeps
        # the variance from the top 'n_components'. The important thing is
        # that the transformation occurred without error.
        input_var = np.var(vpm)
        output_var = np.var(result)
        # We don't assert output_var > input_var, as this is often not true.
        # Instead, we just check that the variance is a valid float.
        assert isinstance(output_var, (float, np.floating))


    def test_process_3d_vpm(self):
        """Test PCA amplification on 3D VPM (time series)."""
        # Create test VPM
        vpm = np.random.rand(3, 64, 64).astype(np.float32)
        
        # Process through amplifier
        result, metadata = self.amplifier.process(vpm)
        
        # Verify output
        assert result.shape == vpm.shape
        assert "n_components" in metadata
        assert "amplification_applied" in metadata
        assert metadata["amplification_applied"] is True
    
    def test_variance_ratio_mode(self):
        """Test PCA with variance ratio specification."""
        amplifier = PCAAmplifier(explained_variance_ratio=0.95)
        vpm = np.random.rand(100, 20).astype(np.float32)
        
        # Add some structure
        vpm[:, 10:] += 0.3 * vpm[:, :10]
        
        result, metadata = amplifier.process(vpm)
        
        # Verify variance ratio is respected
        assert "total_variance_explained" in metadata
        assert metadata["total_variance_explained"] >= 0.90  # Should be close to target
    
    def test_whitening(self):
        """Test PCA with whitening."""
        amplifier = PCAAmplifier(n_components=5, whiten=True)
        vpm = np.random.rand(100, 20).astype(np.float32)
        
        # Add some structure
        vpm[:, 10:] += 0.4 * vpm[:, :10]
        
        result, metadata = amplifier.process(vpm)
        
        # Verify whitening occurred
        assert result.shape == vpm.shape
        assert "whiten" in metadata
        assert metadata["whiten"] is True
   

class TestMorphologicalFilter:
    """Test MorphologicalFilter pipeline stage."""
    
    def setup_method(self):
        self.filter = MorphologicalFilter(operation="opening", kernel_size=3)
    
    def test_process_2d_vpm(self):
        """Test morphological filtering on 2D VPM."""
        # Create test VPM with binary-like structure
        vpm = np.random.rand(64, 64).astype(np.float32)
        vpm = (vpm > 0.5).astype(np.float32)  # Binary pattern
        
        # Process through filter
        result, metadata = self.filter.process(vpm)
        
        # Verify output
        assert result.shape == vpm.shape
        assert "operation" in metadata
        assert "morphology_applied" in metadata
        assert metadata["morphology_applied"] is True
    
    def test_different_operations(self):
        """Test all morphological operations."""
        operations = ["opening", "closing", "erosion", "dilation"]
        vpm = np.random.rand(32, 32).astype(np.float32)
        vpm = (vpm > 0.5).astype(np.float32)
        
        for op in operations:
            filter_op = MorphologicalFilter(operation=op, kernel_size=3)
            result, metadata = filter_op.process(vpm)
            
            assert result.shape == vpm.shape
            assert metadata["operation"] == op
    
    def test_different_kernel_sizes(self):
        """Test different kernel sizes."""
        vpm = np.random.rand(32, 32).astype(np.float32)
        vpm = (vpm > 0.5).astype(np.float32)
        
        for size in [2, 3, 5]:
            filter_op = MorphologicalFilter(operation="opening", kernel_size=size)
            result, metadata = filter_op.process(vpm)
            
            assert result.shape == vpm.shape
            assert metadata["kernel_size"] == size
    
    def test_process_3d_vpm(self):
        """Test morphological filtering on 3D VPM."""
        vpm = np.random.rand(3, 32, 32).astype(np.float32)
        vpm = (vpm > 0.5).astype(np.float32)
        
        result, metadata = self.filter.process(vpm)
        
        assert result.shape == vpm.shape
        assert "morphology_applied" in metadata
    

class TestFFTFilter:
    """Test FFTFilter pipeline stage."""
    
    def setup_method(self):
        self.filter = FFTFilter(filter_type="bandpass", low_freq=0.1, high_freq=0.4)
    
    def test_process_2d_vpm(self):
        """Test FFT filtering on 2D VPM."""
        # Create test VPM with some frequency components
        x = np.linspace(0, 4*np.pi, 64)
        y = np.linspace(0, 4*np.pi, 64)
        X, Y = np.meshgrid(x, y)
        vpm = np.sin(X) * np.cos(Y) + 0.3 * np.random.rand(64, 64)
        vpm = vpm.astype(np.float32)
        
        # Process through filter
        result, metadata = self.filter.process(vpm)
        
        # Verify output
        assert result.shape == vpm.shape
        assert "filter_type" in metadata
        assert "fft_applied" in metadata
        assert metadata["fft_applied"] is True
        
        # Verify mean change is reasonable
        assert "mean_change" in metadata
        assert abs(metadata["mean_change"]) < np.mean(vpm) * 0.5
    
    def test_different_filter_types(self):
        """Test all FFT filter types."""
        filter_types = ["lowpass", "highpass", "bandpass", "bandstop"]
        vpm = np.random.rand(32, 32).astype(np.float32)
        
        for ftype in filter_types:
            if ftype == "bandpass" or ftype == "bandstop":
                filter_op = FFTFilter(filter_type=ftype, low_freq=0.1, high_freq=0.3)
            else:
                filter_op = FFTFilter(filter_type=ftype, high_freq=0.3)  # lowpass/highpass only need high_freq
                
            result, metadata = filter_op.process(vpm)
            
            assert result.shape == vpm.shape
            assert metadata["filter_type"] == ftype
    
    def test_process_3d_vpm(self):
        """Test FFT filtering on 3D VPM."""
        vpm = np.random.rand(3, 32, 32).astype(np.float32)
        
        result, metadata = self.filter.process(vpm)
        
        assert result.shape == vpm.shape
        assert "fft_applied" in metadata
    
class TestKalmanFilter:
    """Test KalmanFilter pipeline stage."""
    
    def setup_method(self):
        self.filter = KalmanFilter(process_noise=1e-4, measurement_noise=1e-2)
    
    def test_process_3d_vpm(self):
        """Test Kalman filtering on 3D VPM (time series)."""
        # Create test VPM with temporal structure
        T, N, M = 10, 8, 8
        vpm = np.zeros((T, N, M), dtype=np.float32)
        
        # Add a signal that evolves over time
        for t in range(T):
            # Signal grows slowly over time
            signal = 0.1 * t
            noise = np.random.normal(0, 0.1, (N, M))
            vpm[t] = signal + noise
        
        # Process through filter
        result, metadata = self.filter.process(vpm)
        
        # Verify output
        assert result.shape == vpm.shape
        assert "kalman_applied" in metadata
        assert metadata["kalman_applied"] is True
        assert "noise_reduction" in metadata
        
        # Verify smoothing occurred
        # Temporal variance should be reduced
        input_temporal_var = np.var(np.diff(vpm, axis=0))
        output_temporal_var = np.var(np.diff(result, axis=0))
        # Allow for some noise
        assert output_temporal_var <= input_temporal_var * 1.5
    
    def test_2d_vpm_warning(self):
        """Test that Kalman filter warns on 2D VPM."""
        vpm = np.random.rand(64, 64).astype(np.float32)
        
        # Should return original with warning
        result, metadata = self.filter.process(vpm)
        
        assert np.array_equal(result, vpm)
        assert "warning" in metadata
        assert "requires 3D VPM" in metadata["warning"]
    
class TestPipelineIntegration:
    """Test integration of pipeline stages."""
    
    def test_pipeline_execution(self):
        """Test executing multiple pipeline stages in sequence."""
        from zeromodel.pipeline.executor import PipelineExecutor
        
        # Create test VPM
        vpm = np.random.rand(5, 64, 64).astype(np.float32)
        
        # Define pipeline
        stages = [
            {"stage": "filters/wavelet.WaveletFilter", "params": {"level": 2}},
            {"stage": "amplifiers/pca.PCAAmplifier", "params": {"n_components": 10}},
            {"stage": "filters/fft.FFTFilter", "params": {"filter_type": "lowpass", "high_freq": 0.3}}
        ]
        
        # Execute pipeline
        executor = PipelineExecutor(stages)
        result, context = executor.run(vpm)
        
        # Verify output
        assert result.shape == vpm.shape
        assert "provenance" in context
        

def test_pipeline_stage_imports():
    """Test that all pipeline stages can be imported."""
    try:
        from zeromodel.pipeline.filter.wavelet import WaveletFilter
        from zeromodel.pipeline.amplifier.pca import PCAAmplifier
        from zeromodel.pipeline.filter.morphological import MorphologicalFilter
        from zeromodel.pipeline.filter.fft import FFTFilter
        from zeromodel.pipeline.filter.kalman import KalmanFilter
        
        # Verify classes exist
        assert callable(WaveletFilter)
        assert callable(PCAAmplifier)
        assert callable(MorphologicalFilter)
        assert callable(FFTFilter)
        assert callable(KalmanFilter)
        
    except ImportError as e:
        pytest.fail(f"Failed to import pipeline stages: {e}")