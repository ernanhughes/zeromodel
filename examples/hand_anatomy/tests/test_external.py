import json

import numpy as np
from PIL import Image
import pytest

from examples.hand_anatomy import synthetic as sy
from examples.hand_anatomy._common import UnsupportedInput
from examples.hand_anatomy.assessment import DecisionContract
from examples.hand_anatomy.external import evaluate_external, import_cases


def _case(root, name, mask, label):
    d = root / name
    d.mkdir(parents=True)
    Image.fromarray((mask * 255).astype(np.uint8), "L").save(d / "mask.png")
    Image.fromarray(np.stack([mask * 200] * 3, axis=-1).astype(np.uint8)).save(d / "image.png")
    (d / "label.json").write_text(json.dumps({"label": label, "defect": None, "source": "unit-test"}))


def test_import_validates_layout_and_labels(tmp_path, small_lib):
    with pytest.raises(UnsupportedInput):
        import_cases(tmp_path / "missing")
    with pytest.raises(UnsupportedInput):
        import_cases(tmp_path)  # exists but empty
    mask = sy.make_independent("test", "open_palm", 0, False).mask
    _case(tmp_path, "a", mask, "valid")
    assert import_cases(tmp_path)[0]["label"] == "valid"
    (tmp_path / "a" / "label.json").write_text(json.dumps({"label": "maybe"}))
    with pytest.raises(UnsupportedInput):
        import_cases(tmp_path)


def test_evaluate_external_runs_both_methods_on_same_cases(tmp_path, small_lib):
    _case(tmp_path, "a", sy.make_independent("test", "open_palm", 0, False).mask, "valid")
    _case(tmp_path, "b", sy.make_independent("test", "large_protrusion", 0, True).mask, "malformed")
    out = evaluate_external(tmp_path, small_lib, DecisionContract(), 0.74, tmp_path / "out")
    assert out["n_cases"] == 2
    for key in ("structured_evidence", "iou_baseline"):
        s = out[key]
        assert s["tp"] + s["fp"] + s["tn"] + s["fn"] + s["abstained_malformed"] + s["abstained_valid"] == 2
    assert (tmp_path / "out" / "external_results.json").is_file()
