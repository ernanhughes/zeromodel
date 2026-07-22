from __future__ import annotations

import pytest

from zeromodel.artifacts import ReportAdapterContractDTO
from zeromodel.artifacts.report_dto import compute_report_adapter_contract_id
from zeromodel.core.artifact import VPMValidationError


def _base_kwargs() -> dict:
    return {
        "adapter_id": "writer.ai_artifact",
        "adapter_version": "1.0.0",
        "report_kind": "writer-ai-artifact",
        "subject_kind": "sentence",
        "dimension_namespace": "writer.ai_artifact",
        "compatibility_id": "writer-ai-artifact/v1",
    }


def test_contract_identity_is_deterministic():
    kwargs = _base_kwargs()
    first = compute_report_adapter_contract_id(**kwargs)
    second = compute_report_adapter_contract_id(**kwargs)
    assert first == second
    ReportAdapterContractDTO(contract_id=first, **kwargs)  # constructs without raising


def test_changing_adapter_version_changes_contract_identity():
    kwargs = _base_kwargs()
    original = compute_report_adapter_contract_id(**kwargs)
    bumped = dict(kwargs, adapter_version="1.1.0")
    changed = compute_report_adapter_contract_id(**bumped)
    assert original != changed


def test_empty_adapter_id_is_rejected():
    kwargs = dict(_base_kwargs(), adapter_id="")
    contract_id = compute_report_adapter_contract_id(**kwargs)
    with pytest.raises(VPMValidationError):
        ReportAdapterContractDTO(contract_id=contract_id, **kwargs)


def test_fabricated_contract_id_is_rejected():
    kwargs = _base_kwargs()
    with pytest.raises(VPMValidationError):
        ReportAdapterContractDTO(contract_id="sha256:" + "a" * 64, **kwargs)


def test_invalid_missing_value_semantics_is_rejected():
    kwargs = _base_kwargs()
    kwargs["missing_value_semantics"] = "zero"
    contract_id = compute_report_adapter_contract_id(**kwargs)
    with pytest.raises(VPMValidationError, match="missing_value_semantics"):
        ReportAdapterContractDTO(contract_id=contract_id, **kwargs)


def test_invalid_duplicate_value_semantics_is_rejected():
    kwargs = _base_kwargs()
    kwargs["duplicate_value_semantics"] = "last_wins"
    contract_id = compute_report_adapter_contract_id(**kwargs)
    with pytest.raises(VPMValidationError, match="duplicate_value_semantics"):
        ReportAdapterContractDTO(contract_id=contract_id, **kwargs)
