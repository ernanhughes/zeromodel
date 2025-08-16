"""
Provenance demo: snapshot a trained model → embed VPF →
strip footer → restore weights → verify identical predictions.
"""

import hashlib
import json
import numpy as np
from io import BytesIO
from PIL import Image
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from zeromodel.metadata import read_all_metadata

import logging


from zeromodel.images.vpf import (
    embed_vpf,
    create_vpf,
    extract_vpf,
    verify_vpf,
)

from zeromodel.images.core import (
    tensor_to_vpm,
    vpm_to_tensor,
)

logger = logging.getLogger(__name__)


def sha3_hex(b: bytes) -> str:
    return hashlib.sha3_256(b).hexdigest()


def test_provenance_model():
    # 1) Data + model
    X, y = make_moons(n_samples=400, noise=0.15, random_state=42)
    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    # 2) Minimal "fittable" state for restore
    weights = {
        "coef_": model.coef_.copy(),
        "intercept_": model.intercept_.copy(),
        "classes_": model.classes_.copy(),
        "n_features_in_": X.shape[1],
    }

    # 3) Snapshot state → VPM image
    vpm_img = tensor_to_vpm(weights, min_size=(128, 128))

    # 4) Build VPF
    vpf = create_vpf(
        pipeline={"graph_hash": "sha3:sklearn-demo", "step": "logreg.fit"},
        model={"id": "sklearn-logreg", "assets": {}},
        determinism={"seed": 0, "rng_backends": ["numpy"]},
        params={"max_iter": 2000, "solver": "lbfgs"},
        inputs={"X_sha3": sha3_hex(X.tobytes()), "y_sha3": sha3_hex(y.tobytes())},
        metrics={"train_accuracy": float(acc)},
        lineage={"parents": []},
    )

    # 5) Embed provenance (metrics stripe optional here)
    png_with_footer = embed_vpf(vpm_img, vpf, mode="stripe")

    # 6) Extract + verify provenance
    vpf_out, meta = extract_vpf(png_with_footer)
    assert verify_vpf(vpf_out, png_with_footer), "VPF verification failed."

    # 7) Strip footer to recover the core VPM PNG, then restore model state
    idx = png_with_footer.rfind(b"ZMVF")
    core_png = png_with_footer[:idx]
    restored_vpm_img = Image.open(BytesIO(core_png)).convert("RGB")
    restored_state = vpm_to_tensor(restored_vpm_img)

    # 8) Rehydrate a fresh model with restored state
    m2 = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)
    # inject minimal fitted attributes
    m2.classes_ = restored_state["classes_"]
    m2.n_features_in_ = int(restored_state["n_features_in_"])
    m2.coef_ = restored_state["coef_"]
    m2.intercept_ = restored_state["intercept_"]

    # 9) Prove identical predictions
    y2 = m2.predict(X)
    acc2 = accuracy_score(y, y2)

    logger.info(f"Original acc:  {acc:.4f}")
    logger.info(f"Restored acc:  {acc2:.4f}")
    logger.info(f"Predictions identical: {np.array_equal(y_pred, y2)}")

    meta = read_all_metadata(png_with_footer)
    print(meta.vpm)  # core info (legacy chunks/whatever VPMMetadata parses)
    print(meta.provenance)  # VPF dict, core_sha3, has_tensor_vpm
    logger.info(f"Meta vpm: {meta.vpm}")
    logger.info(f"\n\n Hi Meta Provenance: {json.dumps(meta.provenance.vpf, indent=2)}")
    assert np.array_equal(y_pred, y2), "Restored model predictions differ."
