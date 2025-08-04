# Zero-Model Intelligence (ZeroModel)

[![PyPI version](https://badge.fury.io/py/zeromodel.svg)](https://badge.fury.io/py/zeromodel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Zero-Model Intelligence** is a paradigm-shifting approach that embeds decision logic into data structure itself. Instead of making models smarter, zeromodel makes data structures intelligent.

> **The intelligence isn't in the processing—it's in the data structure itself.**

## 🧠 Core Concept

zeromodel transforms high-dimensional policy evaluation data into spatially-optimized visual maps where:

- **Position = Importance** (top-left = most relevant)
- **Color = Value** (darker = higher priority)
- **Structure = Task logic** (spatial organization encodes decision workflow)

This enables **zero-model intelligence** on devices with <25KB memory.

## 🚀 Quick Start

```bash
pip install zeromodel
```

```python
from zeromodel import ZeroModel
import numpy as np

# Initialize with metric names
metric_names = ["uncertainty", "size", "quality", "novelty", "coherence"]
zeromodel = ZeroModel(metric_names)

# Generate or load your score matrix (documents × metrics)
score_matrix = np.random.rand(100, 5)  # Example data

# Process for a specific task
zeromodel.set_task("Find uncertain large documents")
zeromodel.process(score_matrix)

# Get visual policy map
vpm = zeromodel.encode()

# For edge devices: get critical tile
tile = zeromodel.get_critical_tile()

# Get top decision
doc_idx, relevance = zeromodel.get_decision()
```

📚 Documentation
See the full documentation for detailed usage instructions.

💡 Edge Device Example (Lua)

```lua
-- 180 bytes of code - works on 25KB memory devices
function process_tile(tile_data)
    -- Parse tile: [width, height, x, y, pixels...]
    local width = string.byte(tile_data, 1)
    local height = string.byte(tile_data, 2)
    local x = string.byte(tile_data, 3)
    local y = string.byte(tile_data, 4)
    
    -- Decision rule: is top-left pixel "dark enough"?
    local top_left = string.byte(tile_data, 5)
    return top_left < 128
end
```

🌐 Website
Check out our website at [zeromi.org](https://zeromi.org) for tutorials, examples, and community resources.

📄 Citation
If you use zeromodel in your research, please cite:

```text
@article{zeromodel2025,
  title={Zero-Model Intelligence: Spatially-Optimized Decision Maps for Resource-Constrained AI},
  author={Ernan Hughes},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

