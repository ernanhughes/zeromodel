# ZeroModel Intelligence (ZeroModel)

[![PyPI version](https://badge.fury.io/py/zeromodel.svg)](https://badge.fury.io/py/zeromodel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ZeroModel Intelligence** is a paradigm-shifting approach that embeds decision logic into the *structure* of data itself. Instead of making models smarter, ZeroModel makes data structures intelligent.

> **The intelligence isn't in the processing—it's in the data structure itself.**

## 🧠 Core Concept

ZeroModel transforms high-dimensional policy evaluation data into spatially-optimized **Visual Policy Maps (VPMs)** where:

- **Position = Importance** (top-left = most relevant)
- **Color = Value** (darker = higher priority)
- **Structure = Task logic** (spatial organization encodes decision workflow based on user-defined SQL)

This enables **zero-model intelligence** on devices with <25KB memory and unlocks **Visual Symbolic Reasoning**.

## 🚀 Quick Start

```bash
pip install zeromodel
```

```python
from zeromodel.core import ZeroModel
import numpy as np

# 1. Prepare your data (documents × metrics)
# Example: 100 items scored on 5 criteria
score_matrix = np.random.rand(100, 5)
metric_names = ["uncertainty", "size", "quality", "novelty", "coherence"]

# 2. Initialize ZeroModel
zeromodel = ZeroModel(metric_names)

# 3. Define your task using SQL and process the data in one step
# The intelligence comes from how you sort the data for your task.
sql_task = "SELECT * FROM virtual_index ORDER BY quality DESC, uncertainty ASC"

# Process data for the task. This handles normalization and spatial organization.
# Use nonlinearity hints for complex tasks (e.g., XOR-like patterns).
zeromodel.prepare(score_matrix, sql_task, nonlinearity_hint='auto')

# 4. Get the Visual Policy Map (VPM) - a structured image
vpm = zeromodel.encode() # Returns a NumPy array (H x W x 3)

# 5. Make decisions by inspecting the structured VPM
doc_idx, relevance = zeromodel.get_decision()
print(f"Top document index: {doc_idx}, Relevance score: {relevance:.4f}")

# 6. For edge devices: get a small, critical tile
tile_bytes = zeromodel.get_critical_tile(tile_size=3) # Returns compact bytes
```

## 🧬 Hierarchical Reasoning

Handle large datasets and multi-resolution decisions with `HierarchicalVPM`:

```python
from zeromodel.hierarchical import HierarchicalVPM

# Create a hierarchical structure (e.g., 3 levels)
hvpm = HierarchicalVPM(metric_names, num_levels=3, zoom_factor=3, precision=16)

# Process data with the same SQL task
hvpm.process(score_matrix, sql_task) # Internally uses ZeroModel.prepare

# Access different levels of detail
base_level_vpm = hvpm.get_level(2)["vpm"]  # Level 2: Most detailed
strategic_vpm = hvpm.get_level(0)["vpm"]   # Level 0: Most abstract

# Get a tile from a specific level for an edge device
edge_tile_bytes = hvpm.get_tile(level_index=0, width=3, height=3)

# Make a decision (defaults to most detailed level)
level, doc_idx, relevance = hvpm.get_decision()
```

## 🔮 Visual Symbolic Reasoning

Combine VPMs like logic gates to create new, complex decision criteria:

```python
from zeromodel.vpm_logic import vpm_and, vpm_or, vpm_not, query_top_left

# Prepare VPMs for different sub-tasks
high_quality_model = ZeroModel(metric_names)
high_quality_model.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY quality DESC")
high_quality_vpm = high_quality_model.encode()

low_uncertainty_model = ZeroModel(metric_names)
low_uncertainty_model.prepare(score_matrix, "SELECT * FROM virtual_index ORDER BY uncertainty ASC")
low_uncertainty_vpm = low_uncertainty_model.encode()

# Compose VPMs: Find items that are High Quality AND Low Uncertainty
# The result is a new VPM representing this combined concept.
good_and_cert_vpm = vpm_and(high_quality_vpm, low_uncertainty_vpm)

# Query the composed VPM
composite_score = query_top_left(good_and_cert_vpm, context_size=3)
print(f"Score for 'High Quality AND Low Uncertainty' items: {composite_score:.4f}")

# This enables complex reasoning: (A AND NOT B) OR (C AND D) as VPM operations.
```

## 💡 Edge Device Example (Pseudocode)

Tiny devices can make intelligent decisions using minimal code by processing small VPM tiles.

```python
# --- On a powerful server ---
hvpm = HierarchicalVPM(metric_names)
hvpm.process(large_score_matrix, "SELECT * ORDER BY my_metric DESC")
tile_bytes_level_0 = hvpm.get_tile(level_index=0, width=3, height=3)
send_to_edge_device(tile_bytes_level_0)

# --- On the tiny edge device (e.g., microcontroller) ---
def process_tile_simple(tile_bytes):
    # Parse header (simplified)
    # tile_data = parse_tile_protocol(tile_bytes)
    # width, height, x, y, pixel_data = tile_data

    # Simple decision: check the very first pixel's Red channel
    # (Assumes pixel data is [R, G, B, R, G, B, ...])
    first_pixel_red_value = tile_bytes[4] # 4 header bytes, 5th byte is first R
    is_relevant = 1 if first_pixel_red_value < 128 else 0
    return is_relevant

# result = process_tile_simple(received_tile_bytes)
# if result == 1: perform_action()
```

## 📚 Running Tests

Ensure you have `pytest` installed (`pip install pytest`).

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_core.py -v

# Run a specific test function
pytest tests/test_xor.py::test_xor_validation -v -s
```

## 🌐 Website

Check out our website at [zeromi.org](https://zeromodel.org) for tutorials, examples, and community resources.

## 📄 Citation

If you use ZeroModel in your research, please cite:

```text
@article{zeromodel2025,
  title={Zero-Model Intelligence: Spatially-Optimized Decision Maps for Resource-Constrained AI},
  author={Ernan Hughes},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
