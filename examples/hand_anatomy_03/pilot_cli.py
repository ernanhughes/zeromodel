"""Run the ZM-HAND-03a development pilot (needs the dev crops + segmentation output).

python -m examples.hand_anatomy_03.pilot_cli
"""

import json
from pathlib import Path
from examples.hand_anatomy_03.real_pipeline import run_pilot

if __name__ == "__main__":
    sel = json.load(open("examples/hand_anatomy_03/data/vbench_dev_selection.json"))[
        "images"
    ]
    r = run_pilot(
        sel,
        Path("artifacts/zm_hand_03/vbench_dev"),
        Path("artifacts/zm_hand_03/seg/dev"),
        Path("artifacts/zm_hand_03/pilot"),
    )
    print(r["wall_seconds"], sum(c["eligible"] for c in r["cases"]), len(r["cases"]))
