from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v3_dataset_is_deterministic_and_fresh_against_v2() -> None:
    root = Path(__file__).resolve().parents[1]
    v2 = _load_module(
        "arcade_visual_address_benchmark_v2_test",
        root / "examples" / "arcade_visual_address_benchmark.py",
    )
    v3 = _load_module(
        "arcade_visual_local_evidence_benchmark_v3_test",
        root / "examples" / "arcade_visual_local_evidence_benchmark.py",
    )
    dataset_v2 = v2.build_arcade_benchmark_dataset(variants_per_family=1, ood_examples_per_family=1)
    dataset_a = v3.build_arcade_local_evidence_dataset(variants_per_family=1)
    dataset_b = v3.build_arcade_local_evidence_dataset(variants_per_family=1)

    assert dataset_a.manifest.digest == dataset_b.manifest.digest
    assert dataset_a.manifest.digest != dataset_v2.manifest.digest

    v2_final = {
        record.observation_digest
        for record in dataset_v2.manifest.records
        if record.split == "final_evaluation"
    }
    v3_final = {
        record.observation_digest
        for record in dataset_a.manifest.records
        if record.split == "final_evaluation"
    }
    assert v3_final
    assert v2_final.isdisjoint(v3_final)


def test_v3_dataset_declares_required_families_and_metadata() -> None:
    module = _load_module(
        "arcade_visual_local_evidence_benchmark_v3_required_test",
        Path(__file__).resolve().parents[1]
        / "examples"
        / "arcade_visual_local_evidence_benchmark.py",
    )
    dataset = module.build_arcade_local_evidence_dataset(variants_per_family=1)
    dataset.manifest.validate()
    family_ids = {family.family_id for family in dataset.manifest.families}
    required = {
        "final-translation-heldout-v3",
        "final-translation-photometric-v3",
        "final-translation-occlusion-v3",
        "final-translation-critical-v3",
        "final-same-action-wrong-row-v3",
        "final-conflicting-action-near-v3",
        "final-compositional-invalid-v3",
        "final-information-impossible-v3",
        "final-beyond-bounds-translation-v3",
    }
    assert required.issubset(family_ids)
    final_records = [
        record
        for record in dataset.manifest.records
        if record.split == "final_evaluation"
    ]
    assert any(record.evaluation_role == module.IMPOSSIBILITY_CONTROL for record in final_records)
    assert any(record.family_id == "final-conflicting-action-near-v3" for record in final_records)
    for record in final_records[:25]:
        assert "distinguishable" in record.metadata
        assert "information_theoretic_impossible" in record.metadata
        assert "contains_conflicting_action_evidence" in record.metadata
        assert "transformation_parameters" in record.metadata
