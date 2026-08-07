# Fixed-Camera State Claims Dataset Scaffold

This example is the seed dataset for `ZM-PER-STATE-001`: trustworthy observation-to-state compilation for a bounded fixed-camera domain.

It uses a locally rendered machine status panel rather than stock or internet images. The renderer knows the exact state for each panel image and writes the ground truth beside the generated screenshots.

Run:

```powershell
python examples/fixed_camera_state_claims/panel_renderer.py
```

Generated structure:

```text
examples/fixed_camera_state_claims/
├── panel_renderer.py
├── policy.json
├── states.json
├── captures/
│   ├── development/
│   ├── calibration/
│   └── evaluation/
└── manifests/
    ├── development.jsonl
    ├── calibration.jsonl
    └── evaluation.jsonl
```

The generated PNG files are canonical computer renderings. Real phone or webcam captures should later be added under the same state/session folders with conditions such as `clean-front`, `bright-room`, `dark-room`, `left-angle`, `right-angle`, `mild-blur`, `screen-glare`, `partial-occlusion`, `cropped-edge`, and `greater-distance`.

The first compiler phase should use hand-authored evidence fixtures against these state declarations. Image decoding belongs in the next phase.
