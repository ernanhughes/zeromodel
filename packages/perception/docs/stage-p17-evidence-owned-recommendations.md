# Stage P17 — Evidence-Owned Operational Recommendations

## Objective

Convert governed operational health evidence into an immutable, inspectable recommendation without allowing the recommendation layer to mutate model lifecycle state.

## Inputs

`recommend_operational_response` requires:

- one `OperationalHealthReportDTO` produced by the statistically gated P16E path;
- one immutable `ModelLifecycleSnapshotDTO`;
- the active model's `ModelCompatibilityContractDTO`;
- explicitly supplied compatibility contracts for historical candidates.

The health report, active pointer, current compatibility contract, and active promoted model must describe the same operational state.

## Recommendation states

- `insufficient_evidence`: at least one governed health finding is underpowered, so action is withheld;
- `no_action`: adequately supported health evidence is healthy;
- `investigate`: drift is adequately supported, but no compatible previously active candidate is available;
- `rollback_candidate`: drift is adequately supported and the most recently active compatible historical model is identified for operator review.

## Safety contract

P17 does not:

- execute rollback;
- activate, supersede, or deactivate a model;
- invent a missing compatibility contract;
- recommend action from insufficient evidence;
- treat lifecycle history alone as rollback eligibility;
- claim causal diagnosis from threshold-based drift evidence.

Any lifecycle mutation remains an explicit operator action through the P12/P16F governed lifecycle APIs.

## Determinism

Recommendation identity includes the health report, lifecycle snapshot, active pointer revision, current compatibility contract, assessed compatibility artifacts, selected target when present, rationale, semantics, and version.

Historical candidates are considered in reverse activation recency. The resulting immutable assessment collection is stored in canonical model-identity order.
