"""Identity-owned video action-set contracts."""

BENCHMARK_VERSION = "zeromodel-video-action-set-reachability-benchmark/v1"
EPISODE_PLAN_VERSION = "zeromodel-video-action-set-sealed-episode-plan/v1"
FRAME_SHAPE = (16, 28)
GENERATOR_VERSION = "zeromodel-video-action-set-reachability-generator/v1"
OBSERVATION_OPERATION_CHAIN_VERSION = "zeromodel-video-observation-operation-chain/v1"
PROVIDER_OBSERVATION_BOUNDARY_VERSION = (
    "zeromodel-video-action-set-provider-observation-boundary/v1"
)
REACHABILITY_TILE_DIGEST = (
    "sha256:fef2bc5fd795bb92d3bd564bccdc2d32e1b23319aba55dffed5e0391e795a5df"
)
REACHABILITY_TILE_VERSION = "zeromodel-video-policy-reachability-tile/v1"
SEED_DERIVATION_VERSION = "zeromodel-video-action-set-seed-derivation/v1"

__all__ = [
    "BENCHMARK_VERSION",
    "EPISODE_PLAN_VERSION",
    "FRAME_SHAPE",
    "GENERATOR_VERSION",
    "OBSERVATION_OPERATION_CHAIN_VERSION",
    "PROVIDER_OBSERVATION_BOUNDARY_VERSION",
    "REACHABILITY_TILE_DIGEST",
    "REACHABILITY_TILE_VERSION",
    "SEED_DERIVATION_VERSION",
]
