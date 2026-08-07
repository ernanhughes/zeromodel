# Fixed-Camera State Claims Dataset Scaffold

This example is the seed dataset for `ZM-PER-STATE-001`: trustworthy observation-to-state compilation for a bounded fixed-camera domain.

It uses locally rendered canonical panel fixtures rather than stock or internet images. The renderer knows the exact state for each panel image and writes the ground truth beside the generated fixtures.

Run:

```powershell
python examples/fixed_camera_state_claims/panel_renderer.py
```

Compile one rendered or captured panel image into typed evidence:

```python
from pathlib import Path
from examples.fixed_camera_state_claims.evidence_compiler import PanelObservationEvidenceCompiler

compiler = PanelObservationEvidenceCompiler.load_default()
observation, report = compiler.compile_image(
    Path("examples/fixed_camera_state_claims/captures/development/state-001/canonical-panel-fixture-01.png")
)
```

Generated structure:

```text
examples/fixed_camera_state_claims/
├── panel_renderer.py
├── evidence_compiler.py
├── policy.json
├── states.json
├── regions.json
├── calibration.json
├── canonical/
├── captures/
│   ├── development/
│   ├── calibration/
│   └── evaluation/
└── manifests/
    ├── development.jsonl
    ├── calibration.jsonl
    └── evaluation.jsonl
```

The generated PNG files are canonical panel fixtures. The development, calibration and evaluation folders are logical partitions until real camera images are added; they are not independent optical capture sessions.

Real phone or webcam captures should later be added under the same state/session folders with conditions such as `clean-front`, `bright-room`, `dark-room`, `left-angle`, `right-angle`, `mild-blur`, `screen-glare`, `partial-occlusion`, `cropped-edge`, and `greater-distance`.

The deterministic evidence compiler reads declared visual channels rather than OCR:

- corner anchors for panel validity and registration;
- LED colour for power;
- active marker position for mode;
- three-segment bar fill for temperature;
- door geometry occupancy;
- alarm diamond fill.

The current compiler is validated against canonical fixtures and deliberate synthetic occlusions. It is not yet a held-out real-camera benchmark.
