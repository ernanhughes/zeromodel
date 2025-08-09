import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("zeromemory.gif_logger")

class TrainingHeartbeatVisualizer:
    """
    Creates an animated GIF showing the "heartbeat" of training using ZeroMemory's VPM snapshots.
    
    The visualization has two components:
    - TOP: Current VPM tile showing the most relevant metrics (spatial representation)
    - BOTTOM: Timeline strip with sparklines of key metrics and alert indicators
    
    Designed to work seamlessly with ZeroMemory for intuitive monitoring of training dynamics.
    """
    
    def __init__(self, 
                 max_frames: int = 2000,
                 vpm_scale: int = 6,        # Scale factor for VPM visualization
                 strip_height: int = 40,    # Height of the timeline strip
                 bg_color: Tuple[int, int, int] = (10, 10, 12),
                 font_size: int = 9):
        """
        Initialize the heartbeat visualizer.
        
        Args:
            max_frames: Maximum number of frames to store (avoids memory issues)
            vpm_scale: Scale factor for VPM visualization (higher = more visible)
            strip_height: Height of the timeline strip at the bottom
            bg_color: Background color (RGB)
            font_size: Font size for labels
        """
        self.frames = []  # Store VPM frames (HxWx3 uint8)
        self.meta = []    # Store metrics for each frame
        self.max_frames = max(10, max_frames)  # Minimum 10 frames
        self.vpm_scale = max(1, vpm_scale)
        self.strip_height = max(20, strip_height)
        self.bg_color = bg_color
        self.font_size = font_size
        self.font = None  # Will be initialized on first use
        
        logger.info(f"Initialized TrainingHeartbeatVisualizer with max_frames={max_frames}, "
                    f"vpm_scale={vpm_scale}, strip_height={strip_height}")
    
    def _ensure_font(self):
        """Initialize font on first use (avoids PIL import issues early)"""
        if self.font is None:
            try:
                self.font = ImageFont.truetype("arial", self.font_size)
            except:
                try:
                    self.font = ImageFont.truetype("DejaVuSans", self.font_size)
                except:
                    self.font = ImageFont.load_default()
    
    def add_frame(self, 
                 vpm_uint8: np.ndarray, 
                 metrics: Dict[str, Any],
                 step: Optional[int] = None):
        """
        Add a frame to the visualization.
        
        Args:
            vpm_uint8: HxWx3 uint8 array from ZeroMemory.snapshot_vpm()
            metrics: Dictionary containing training metrics
            step: Optional step number (if not in metrics)
        """
        # Handle early training case where VPM might be empty
        if vpm_uint8 is None or vpm_uint8.size == 0:
            logger.warning("Skipping frame: empty VPM")
            return
            
        # Ensure proper VPM format
        if vpm_uint8.ndim != 3 or vpm_uint8.shape[2] != 3:
            logger.warning(f"Skipping frame: invalid VPM shape {vpm_uint8.shape}")
            return
            
        # Handle frame limit
        if len(self.frames) >= self.max_frames:
            # Simple decimation: keep every other frame
            if len(self.frames) % 2 == 0:
                self.frames.pop(0)
                self.meta.pop(0)
            else:
                return  # Skip this frame
        
        # Store the frame
        self.frames.append(vpm_uint8.copy())
        
        # Process metrics
        timestamp = time.time()
        step_val = step if step is not None else metrics.get("step", len(self.meta))
        
        # Extract alerts (ensure boolean values)
        alerts = metrics.get("alerts", {})
        if not isinstance(alerts, dict):
            alerts = {}
        
        # Normalize metric values for consistent processing
        processed_metrics = {
            "timestamp": timestamp,
            "step": step_val,
            "loss": float(metrics.get("loss", np.nan)),
            "val_loss": float(metrics.get("val_loss", np.nan)),
            "acc": float(metrics.get("acc", np.nan)),
            "val_acc": float(metrics.get("val_acc", np.nan)),
            "lr": float(metrics.get("lr", np.nan)),
            "alerts": {
                "overfitting": bool(alerts.get("overfitting", False)),
                "underfitting": bool(alerts.get("underfitting", False)),
                "drift": bool(alerts.get("drift", False)),
                "saturation": bool(alerts.get("saturation", False)),
                "instability": bool(alerts.get("instability", False))
            }
        }
        
        self.meta.append(processed_metrics)
        logger.debug(f"Added frame #{len(self.frames)} at step {step_val}")

    def _compose_frame(self, 
                      vpm: np.ndarray, 
                      history: List[Dict[str, Any]]) -> Image.Image:
        """
        Compose a single frame with VPM on top and timeline at bottom.
        
        Args:
            vpm: Current VPM (HxWx3 uint8)
            history: Historical metrics for the timeline
            
        Returns:
            PIL Image ready for animation
        """
        # --- Top section: Scaled VPM ---
        H, W, _ = vpm.shape
        scaled_w = W * self.vpm_scale
        scaled_h = H * self.vpm_scale
        
        # Convert to PIL image and scale
        vpm_img = Image.fromarray(vpm, mode="RGB")
        scaled_vpm = vpm_img.resize(
            (scaled_w, scaled_h), 
            resample=Image.NEAREST
        )
        
        # --- Bottom section: Timeline strip ---
        K = min(300, len(history))  # Show last 300 points
        timeline_w = scaled_w
        timeline_h = self.strip_height
        
        # Create timeline background
        timeline = Image.new("RGB", (timeline_w, timeline_h), self.bg_color)
        draw = ImageDraw.Draw(timeline)
        
        # Prepare metric data for visualization
        def normalize_series(values):
            """Normalize series to [0,1] range, handling NaNs"""
            arr = np.array(values, dtype=np.float32)
            mask = np.isfinite(arr)
            
            if mask.sum() < 2:  # Not enough valid points
                return np.full_like(arr, np.nan)
                
            valid_vals = arr[mask]
            min_val = np.percentile(valid_vals, 5)
            max_val = np.percentile(valid_vals, 95)
            
            if max_val - min_val < 1e-8:
                return np.where(mask, 0.5, np.nan)
                
            # Normalize to [0,1]
            normalized = np.full_like(arr, np.nan)
            normalized[mask] = np.clip((valid_vals - min_val) / (max_val - min_val), 0, 1)
            return normalized
        
        # Extract and normalize metric series
        losses = normalize_series([d["loss"] for d in history[-K:]])
        val_losses = normalize_series([d["val_loss"] for d in history[-K:]])
        accs = normalize_series([d["acc"] for d in history[-K:]])
        val_accs = normalize_series([d["val_acc"] for d in history[-K:]])
        
        # Draw sparklines
        def draw_sparkline(values, y_offset, color, width=1):
            """Draw a sparkline on the timeline"""
            if len(values) < 2:
                return
                
            # Convert to points
            points = []
            for i, val in enumerate(values):
                if np.isnan(val):
                    continue
                    
                x = int(i * (timeline_w - 1) / (len(values) - 1))
                y = int(y_offset + (1.0 - val) * (timeline_h / 4 - 4))
                points.append((x, y))
            
            # Draw line
            if len(points) > 1:
                draw.line(points, fill=color, width=width)
        
        # Draw all sparklines
        h4 = timeline_h // 4
        draw_sparkline(losses, 1 + 0*h4, (220, 120, 120))     # Training loss (red)
        draw_sparkline(val_losses, 1 + 1*h4, (120, 180, 220))  # Validation loss (blue)
        draw_sparkline(accs, 1 + 2*h4, (140, 220, 140))        # Training accuracy (green)
        draw_sparkline(val_accs, 1 + 3*h4, (220, 180, 120))    # Validation accuracy (orange)
        
        # Draw alert indicators
        for i, data in enumerate(history[-K:]):
            x = int(i * (timeline_w - 1) / max(1, K - 1))
            
            # Overfitting: red triangle at bottom
            if data["alerts"]["overfitting"]:
                draw.polygon([
                    (x, timeline_h - 4),
                    (x - 3, timeline_h),
                    (x + 3, timeline_h)
                ], fill=(255, 80, 80))
            
            # Drift: yellow line
            if data["alerts"]["drift"]:
                draw.line([(x, timeline_h - 10), (x, timeline_h - 5)], 
                         fill=(255, 200, 80), width=1)
            
            # Instability: white zigzag
            if data["alerts"]["instability"]:
                draw.line([(x-2, timeline_h - 8), (x, timeline_h - 4), (x+2, timeline_h - 8)], 
                         fill=(220, 220, 220), width=1)
        
        # Add labels
        self._ensure_font()
        label_y = timeline_h - 12
        draw.text((5, 0), "Loss", fill=(200, 150, 150), font=self.font)
        draw.text((5, h4), "Val Loss", fill=(150, 180, 220), font=self.font)
        draw.text((5, 2*h4), "Accuracy", fill=(150, 220, 150), font=self.font)
        draw.text((5, 3*h4), "Val Acc", fill=(220, 200, 150), font=self.font)
        
        # Add current step indicator
        current_step = history[-1]["step"] if history else 0
        step_text = f"Step: {current_step}"
        text_width = draw.textlength(step_text, font=self.font)
        draw.text((timeline_w - text_width - 5, 2), step_text, 
                 fill=(200, 200, 200), font=self.font)
        
        # --- Combine sections ---
        panel = Image.new("RGB", (scaled_w, scaled_h + timeline_h), self.bg_color)
        panel.paste(scaled_vpm, (0, 0))
        panel.paste(timeline, (0, scaled_h))
        
        return panel

    def save_gif(self, 
                path: str = "training_heartbeat.gif",
                fps: float = 6.0,
                optimize: bool = True,
                loop: int = 0) -> str:
        """
        Save the collected frames as an animated GIF.
        
        Args:
            path: Output file path
            fps: Frames per second
            optimize: Whether to optimize the GIF (reduces size)
            loop: Loop count (0 = infinite)
            
        Returns:
            Path to the saved GIF
        """
        if not self.frames:
            logger.error("No frames to save - call add_frame() first")
            raise RuntimeError("No frames to save")
            
        logger.info(f"Creating GIF with {len(self.frames)} frames at {fps} fps")
        
        # Compose panels
        panels = []
        for i in range(len(self.frames)):
            try:
                panel = self._compose_frame(self.frames[i], self.meta[:i+1])
                panels.append(panel)
            except Exception as e:
                logger.error(f"Failed to compose frame {i}: {str(e)}")
                # Skip problematic frame but continue
                continue
        
        if not panels:
            logger.error("No valid panels created")
            raise RuntimeError("Failed to create any valid panels")
            
        # Convert to palette images for smaller file size
        palette_images = []
        for panel in panels:
            try:
                palette_images.append(
                    panel.convert("P", palette=Image.ADAPTIVE, colors=256)
                )
            except Exception as e:
                logger.error(f"Failed to convert panel to palette: {str(e)}")
                # Fallback to original if conversion fails
                palette_images.append(panel)
        
        # Save GIF
        duration_ms = int(1000 / max(1, fps))
        try:
            palette_images[0].save(
                path,
                save_all=True,
                append_images=palette_images[1:],
                duration=duration_ms,
                loop=loop,
                optimize=optimize,
                disposal=2
            )
            logger.info(f"Successfully saved GIF to {path} ({len(panels)} frames)")
            return path
        except Exception as e:
            logger.error(f"Failed to save GIF: {str(e)}")
            raise

    def clear(self):
        """Clear all collected frames and metrics."""
        self.frames = []
        self.meta = []
        logger.debug("Cleared all frames and metrics")