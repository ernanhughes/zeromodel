# tests/test_gif_logging.py
"""
Test case demonstrating GIF logging in ZeroModel pipeline.
"""

import logging
import os
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from zeromodel.provenance import create_vpf, embed_vpf
from zeromodel.pipeline.executor import PipelineExecutor
from zeromodel.vpm.encoder import VPMEncoder

logger = logging.getLogger(__name__)

class TestGifLogging:
    """Test GIF logging functionality in ZeroModel pipeline."""
    
    def setup_method(self):
        """Setup test data and cleanup previous test files."""
        # Clean up previous test files
        test_files = [
            "pipeline_animation.gif", 
            "test_vpm_tile.png",
            "gif_frame_preview.png"
        ]
        for file in test_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass
        
        # Create test VPM data
        np.random.seed(42)
        self.T, self.N, self.M = 5, 100, 20
        self.vpm_data = self._create_synthetic_data()
        
        logger.info(f"Created test VPM data: shape={self.vpm_data.shape}")
    
    def _create_synthetic_data(self):
        """Create synthetic VPM data with structured signal."""
        # Create true signal pattern - first 5 metrics are important
        w_true = np.zeros(self.M)
        active_indices = [0, 1, 2, 3, 4]
        w_true[active_indices] = [0.8, 0.7, 0.6, 0.5, 0.4]
        w_true = w_true / (np.linalg.norm(w_true) + 1e-12)
        
        series = []
        # Create binary target with 30% positive rate
        y = np.random.binomial(1, 0.3, self.N)
        
        for t in range(self.T):
            # Base noise
            base = np.random.normal(0, 0.4, (self.N, self.M))
            
            # Signal correlated with y and w_true
            signal_strength = np.random.uniform(0.9, 1.1)
            signal = np.outer(y, w_true) * signal_strength
            
            # Additional noise
            noise = np.random.normal(0, 0.3, (self.N, self.M))
            
            # Combine and ensure non-negative
            X_t = np.maximum(0.0, base + signal + noise)
            series.append(X_t)
        
        return np.stack(series, axis=0)
    
    def _verify_gif_structure(self, gif_path):
        """Verify the GIF has correct structure and frames."""
        assert os.path.exists(gif_path), f"GIF file not created: {gif_path}"
        
        # Open GIF and verify properties
        with Image.open(gif_path) as gif:
            # Check it's actually a GIF
            assert gif.format == 'GIF', f"Expected GIF format, got {gif.format}"
            
            # Get frame count
            frame_count = 0
            try:
                while True:
                    gif.seek(frame_count)
                    frame_count += 1
            except EOFError:
                pass  # End of sequence
            
            # We expect 2 frames per stage (before/after) + 1 for initial state
            expected_frames = len(self.stages) * 2 + 1
            logger.warning(f"Expected {expected_frames} frames, got {frame_count}")
            
            # Verify frame dimensions
            first_frame = gif.copy()
            assert first_frame.size[0] > 0 and first_frame.size[1] > 0, "Frame has invalid dimensions"
            
            # Save first frame for inspection
            first_frame.save("gif_frame_preview.png")
            logger.info(f"GIF verified: {frame_count} frames, size={first_frame.size}")
    
    def _analyze_frame_content(self, gif_path):
        """Analyze frame content to ensure meaningful changes."""
        with Image.open(gif_path) as gif:
            frames = []
            try:
                frame_idx = 0
                while True:
                    gif.seek(frame_idx)
                    # Force RGB to avoid P-mode single-channel frames
                    frame_array = np.array(gif.convert("RGB"))
                    frames.append(frame_array)
                    frame_idx += 1
            except EOFError:
                pass
        
        # Analyze frame differences
        total_changes = []
        for i in range(1, len(frames)):
            # Calculate pixel difference between consecutive frames
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            total_changes.append(diff)
            logger.debug(f"Frame {i-1}→{i} change: {diff:.2f}")
        
        # Ensure there are meaningful changes (not all zeros)
        assert len(total_changes) > 0, "No frames in GIF"
        assert np.mean(total_changes) > 1.0, "Frames show no meaningful changes"
        
        logger.info(f"Average frame change: {np.mean(total_changes):.2f}")
    
    def test_gif_logging_complete_pipeline(self):
        """Test GIF logging with complete pipeline execution."""
        logger.info("Starting GIF logging test")
        
        # Define pipeline stages
        self.stages = [
            {
                "stage": "amplifier.stdm.STDMAmplifier",
                "params": {
                    "Kc": 12, 
                    "Kr": 48, 
                    "alpha": 0.97,
                    "iters": 50,  # Reduced for faster testing
                    "step": 8e-3
                }
            },
            {
                "stage": "organizer.top_left.TopLeft",
                "params": {
                    "metric": "variance", 
                    "Kc": 12
                }
            },
        ]
        
        # Configure context with GIF logging
        context = {
            "enable_gif": True,
            "gif_path": "pipeline_animation.gif",
            "gif_scale": 4,
            "gif_fps": 4,
            "gif_max_frames": 100,
            "gif_strip_h": 40,
        }
        
        logger.info("Executing pipeline with GIF logging enabled")
        
        # Execute pipeline
        executor = PipelineExecutor(self.stages)
        result, context_out = executor.run(self.vpm_data, context=context)
        
        # Verify output
        assert result.shape == self.vpm_data.shape, "Pipeline should preserve VPM shape"
        assert "gif_saved" in context_out, "GIF should be saved"
        assert context_out["gif_saved"] == "pipeline_animation.gif", "GIF saved to wrong path"
        
        # Verify GIF was created and has correct structure
        self._verify_gif_structure("pipeline_animation.gif")
        
        # Analyze frame content
        self._analyze_frame_content("pipeline_animation.gif")
        
        logger.info("GIF logging test completed successfully")
    
    def test_gif_logging_with_vpf_embedding(self):
        """Test GIF logging with VPF embedding in final output."""
        logger.info("Starting GIF logging with VPF test")
        
        # First, create animated GIF of pipeline
        self.stages = [
            {
                "stage": "amplifier.stdm.STDMAmplifier",
                "params": {
                    "Kc": 12, 
                    "Kr": 48, 
                    "alpha": 0.97,
                    "iters": 30,
                    "step": 8e-3
                }
            },
            {
                "stage": "organizer.top_left.TopLeft",
                "params": {
                    "metric": "variance", 
                    "Kc": 12
                }
            },
        ]
        
        # Enable GIF logging
        context = {
            "enable_gif": True,
            "gif_path": "pipeline_with_vpf.gif",
            "gif_scale": 3,
            "gif_fps": 3,
            "gif_strip_h": 30,
        }
        
        # Execute pipeline with GIF logging
        executor = PipelineExecutor(self.stages)
        result, context_out = executor.run(self.vpm_data, context=context)
        
        # Create VPF for final output
        vpf = create_vpf(
            pipeline={
                "graph_hash": "sha3:gif-demo", 
                "step": "pipeline-with-animation"
            },
            model={
                "id": "zero-1.0", 
                "assets": {}
            },
            determinism={
                "seed": 42, 
                "rng_backends": ["numpy"]
            },
            params={
                "pipeline_stages": len(self.stages),
                "gif_logging": True
            },
            inputs={
                "data_shape": self.vpm_data.shape
            },
            metrics={
                "frames_generated": context_out.get("gif_saved", False),
                "final_tl_mass": 0.0  # Could calculate from result
            },
            lineage={
                "parents": []
            },
        )
        
        # Create final VPM image
        vpm_img = VPMEncoder('float32').encode(result[0])  # Use first time slice
        vpm_pil = Image.fromarray((vpm_img * 255).astype(np.uint8))
        
        # Embed VPF in final image
        final_png = embed_vpf(vpm_pil, vpf, mode="stripe")
        
        # Save final output
        with open("test_vpm_tile.png", "wb") as f:
            f.write(final_png)
        
        # Verify everything worked
        assert os.path.exists("pipeline_with_vpf.gif"), "Animation GIF not created"
        assert os.path.exists("test_vpm_tile.png"), "Final VPM tile not created"
        
        # Verify VPF can be extracted
        from zeromodel.provenance import extract_vpf
        extracted_vpf, meta = extract_vpf(final_png)
        assert extracted_vpf["pipeline"]["step"] == "pipeline-with-animation"
        
        logger.info("GIF logging with VPF test completed successfully")
    
    def test_gif_logging_edge_cases(self):
        """Test GIF logging with edge cases."""
        logger.info("Starting GIF logging edge case test")
        
        # Test with minimal data
        minimal_vpm = np.random.rand(2, 2, 2).astype(np.float32)
        
        self.stages = [
            {
                "stage": "amplifier.stdm.STDMAmplifier",
                "params": {
                    "Kc": 1, 
                    "Kr": 1, 
                    "alpha": 0.97
                }
            }
        ]
        
        # Test with very small GIF
        context = {
            "enable_gif": True,
            "gif_path": "minimal_pipeline.gif",
            "gif_scale": 1,
            "gif_fps": 1,
            "gif_strip_h": 10,
        }
        
        executor = PipelineExecutor(self.stages)
        result, context_out = executor.run(minimal_vpm, context=context)
        
        assert os.path.exists("minimal_pipeline.gif"), "Minimal GIF not created"
        
        # Test with disabled GIF logging
        context_disabled = {"enable_gif": False}
        result2, context_out2 = executor.run(minimal_vpm, context=context_disabled)
        
        assert "gif_saved" not in context_out2, "GIF should not be saved when disabled"
        
        logger.info("GIF logging edge case test completed")
    
    def teardown_method(self):
        """Clean up test files after each test."""
        test_files = [
            "pipeline_animation.gif", 
            "pipeline_with_vpf.gif",
            "minimal_pipeline.gif",
            "test_vpm_tile.png",
            "gif_frame_preview.png"
        ]
        
        for file in test_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    logger.debug(f"Cleaned up test file: {file}")
                except Exception as e:
                    logger.warning(f"Could not remove {file}: {e}")

# Additional utility test to demonstrate the "AI heartbeat" concept
def test_ai_heartbeat_animation():
    from io import BytesIO

    from zeromodel.utils import tile_to_pil
    logger.info("Creating AI heartbeat animation")

    tiles = []
    for step in range(8):
        scores = np.random.rand(64, 64).astype(np.float32) * (step + 1) / 8.0
        decay = 0.95 ** np.add.outer(np.arange(64), np.arange(64))
        scores = scores * decay
        tiles.append(scores)  # keep as array; convert below

    pil_frames = [tile_to_pil(t) for t in tiles]

    # Optional: quantize to palette for smaller GIF (or keep RGB; both work)
    use_palette = False  # set True to reduce size
    frames_out = ( [im.convert("P", palette=Image.ADAPTIVE, colors=256) for im in pil_frames]
                   if use_palette else
                   [im.convert("RGB") for im in pil_frames] )

    buf = BytesIO()
    frames_out[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames_out[1:],
        duration=200,
        loop=0,
        disposal=2,
        optimize=True,
    )
    buf.seek(0)
        
    # Save to file
    with open("ai_heartbeat.gif", "wb") as f:
        f.write(buf.getvalue())
    
    # Verify GIF
    assert os.path.exists("ai_heartbeat.gif"), "AI heartbeat GIF not created"
    with Image.open("ai_heartbeat.gif") as gif:
        frame_count = 0
        try:
            while True:
                gif.seek(frame_count)
                frame_count += 1
        except EOFError:
            pass
        assert frame_count == 8, f"Expected 8 frames, got {frame_count}"
    
    logger.info("AI heartbeat animation created successfully")
    logger.info("This demonstrates ZeroModel's 'See AI think' principle - "
               "you can literally watch the reasoning unfold frame by frame.")