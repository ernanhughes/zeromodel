from __future__ import annotations

import zeromodel.perception as perception


def test_p18g_is_exposed_from_package_root() -> None:
    expected = {
        "ActivePromotionStateDTO",
        "PromotionActivationPolicyDTO",
        "PromotionActivationAuditFindingDTO",
        "PromotionActivationAuditReportDTO",
        "PromotionActivationAdmissionDTO",
        "PromotionRollbackPlanDTO",
        "PromotionActivationReceiptDTO",
        "PromotionActivationBundleDTO",
        "PromotionActivationStore",
        "InMemoryPromotionActivationStore",
        "SqlitePromotionActivationStore",
        "PerceptionPromotionActivationError",
        "PromotionRollbackRequestDTO",
        "PromotionRollbackPolicyDTO",
        "PromotionRollbackAuditDTO",
        "PromotionRollbackAdmissionDTO",
        "PromotionRollbackReceiptDTO",
        "PromotionRollbackBundleDTO",
        "audit_promotion_activation",
        "audit_promotion_rollback",
        "authorize_promotion_activation",
        "authorize_promotion_rollback",
        "build_promotion_activation_bundle",
        "execute_promotion_activation",
        "execute_promotion_rollback",
        "PROMOTION_ACTIVATION_BUNDLE_VERSION",
        "PROMOTION_ACTIVE_STATE_VERSION",
        "PROMOTION_ROLLBACK_RECEIPT_VERSION",
        "PROMOTION_SQLITE_ACTIVATION_SCHEMA_VERSION",
    }

    assert expected <= set(perception.__all__)
    for name in expected:
        assert getattr(perception, name) is not None
