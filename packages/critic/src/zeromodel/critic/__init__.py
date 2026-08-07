from __future__ import annotations

from zeromodel.critic.compiler import compile_critic_readout
from zeromodel.critic.dto import (
    CriticContractDTO,
    CriticEvaluationResultDTO,
    CriticEvaluationSetDTO,
    CriticFeatureBatchDTO,
    CriticFeatureContributionDTO,
    CriticFeatureDTO,
    CriticFeatureSpecDTO,
    CriticFitSpecDTO,
    CriticItemScoreDTO,
    CriticLabelBatchDTO,
    CriticReadoutArtifactDTO,
    CriticScoreReceiptDTO,
    CriticScoreRequestDTO,
    CriticScoreResultDTO,
    CriticThresholdContractDTO,
)
from zeromodel.critic.errors import (
    CriticCalibrationError,
    CriticContractMismatchError,
    CriticEvaluationError,
    CriticFeatureSchemaMismatchError,
    CriticPayloadTooLargeError,
    CriticReadoutIntegrityError,
    CriticReplayMismatchError,
    CriticValidationError,
)
from zeromodel.critic.evaluation import (
    auroc,
    budget_selection_metrics,
    brier_score,
    evaluate_binary_critic,
    expected_calibration_error,
    grouped_selection_metrics,
)
from zeromodel.critic.linear import CompiledCriticReadout
from zeromodel.critic.persistence import load_critic_readout_aggregate
from zeromodel.critic.persistence import (
    load_critic_evaluation_result_aggregate,
    load_critic_evaluation_set_aggregate,
    store_critic_evaluation_result,
    store_critic_evaluation_set,
)
from zeromodel.critic.portable import (
    export_portable_critic,
    load_portable_critic,
    score_portable,
)
from zeromodel.critic.promotion import (
    CriticPromotionDecisionDTO,
    CriticPromotionPolicyDTO,
    evaluate_promotion,
)
from zeromodel.critic.receipts import build_critic_score_receipt, replay_critic_score
from zeromodel.critic.scoring import score_critic
from zeromodel.critic.triage import rank_by_critic, triage_by_budget
from zeromodel.critic.views import build_critic_score_vpm

CRITIC_PACKAGE_VERSION = "1.2.0"

__all__ = [
    "CRITIC_PACKAGE_VERSION",
    "CompiledCriticReadout",
    "CriticCalibrationError",
    "CriticContractDTO",
    "CriticContractMismatchError",
    "CriticEvaluationResultDTO",
    "CriticEvaluationSetDTO",
    "CriticEvaluationError",
    "CriticFeatureBatchDTO",
    "CriticFeatureContributionDTO",
    "CriticFeatureDTO",
    "CriticFeatureSchemaMismatchError",
    "CriticFeatureSpecDTO",
    "CriticFitSpecDTO",
    "CriticItemScoreDTO",
    "CriticLabelBatchDTO",
    "CriticPayloadTooLargeError",
    "CriticPromotionDecisionDTO",
    "CriticPromotionPolicyDTO",
    "CriticReadoutArtifactDTO",
    "CriticReadoutIntegrityError",
    "CriticReplayMismatchError",
    "CriticScoreReceiptDTO",
    "CriticScoreRequestDTO",
    "CriticScoreResultDTO",
    "CriticThresholdContractDTO",
    "CriticValidationError",
    "auroc",
    "brier_score",
    "budget_selection_metrics",
    "build_critic_score_receipt",
    "build_critic_score_vpm",
    "compile_critic_readout",
    "evaluate_binary_critic",
    "evaluate_promotion",
    "expected_calibration_error",
    "export_portable_critic",
    "grouped_selection_metrics",
    "load_critic_readout_aggregate",
    "load_critic_evaluation_result_aggregate",
    "load_critic_evaluation_set_aggregate",
    "load_portable_critic",
    "rank_by_critic",
    "replay_critic_score",
    "score_critic",
    "score_portable",
    "store_critic_evaluation_result",
    "store_critic_evaluation_set",
    "triage_by_budget",
]
