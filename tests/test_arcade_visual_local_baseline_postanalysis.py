from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "arcade_visual_local_baseline_postanalysis.py"
    )
    spec = importlib.util.spec_from_file_location(
        "arcade_visual_local_baseline_postanalysis_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gate_bucket_decomposes_distance_and_margin_failures() -> None:
    module = _load_module()
    accepted = {
        "decision": {
            "accepted": True,
            "nearest_score": 0.1,
            "ambiguity_measure": 0.6,
            "trace": {
                "distance_threshold": 0.2,
                "required_conflicting_action_margin": 0.5,
            },
        }
    }
    margin_only = {
        "decision": {
            "accepted": False,
            "nearest_score": 0.1,
            "ambiguity_measure": 0.4,
            "trace": {
                "distance_threshold": 0.2,
                "required_conflicting_action_margin": 0.5,
            },
        }
    }
    distance_only = {
        "decision": {
            "accepted": False,
            "nearest_score": 0.3,
            "ambiguity_measure": 0.6,
            "trace": {
                "distance_threshold": 0.2,
                "required_conflicting_action_margin": 0.5,
            },
        }
    }
    both = {
        "decision": {
            "accepted": False,
            "nearest_score": 0.3,
            "ambiguity_measure": 0.4,
            "trace": {
                "distance_threshold": 0.2,
                "required_conflicting_action_margin": 0.5,
            },
        }
    }
    assert module._gate_bucket(accepted) == "accepted"
    assert module._gate_bucket(margin_only) == "margin_only"
    assert module._gate_bucket(distance_only) == "distance_only"
    assert module._gate_bucket(both) == "both"


def test_decoupled_selection_prefers_conservative_threshold_after_coverage() -> None:
    module = _load_module()
    candidate_a = {
        "accepted_exact_row_precision": 1.0,
        "benign_coverage": 0.1,
        "accepted_exact_row_recall": 0.1,
        "top1_benign_row_accuracy": 1.0,
        "threshold": 0.07,
        "ambiguity_margin": 0.67,
        "distance_quantile": 0.0,
    }
    candidate_b = {
        "accepted_exact_row_precision": 1.0,
        "benign_coverage": 0.1,
        "accepted_exact_row_recall": 0.1,
        "top1_benign_row_accuracy": 1.0,
        "threshold": 0.02,
        "ambiguity_margin": 0.67,
        "distance_quantile": 0.5,
    }
    assert module._decoupled_selection_key(candidate_b) > module._decoupled_selection_key(candidate_a)
