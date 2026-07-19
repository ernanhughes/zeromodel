# Video Action-Set Reachability Runtime Amendment v1

This amendment freezes the allowed runtime optimization space for the prospective action-set reachability benchmark without changing any scientific or identity-bearing semantics.

## Runtime Principle

Runtime optimization may change execution strategy, data layout, cache structure, batching, vectorization, checkpointing, and parallel scheduling. It may not change the scientific meaning or identity-bearing output of a provider.

## Provider Semantic Invariants

For every reference versus optimized comparison, the following must remain identical:

- the same `112` row IDs
- the same quantized score for every row
- the same complete ranking
- the same tie groups
- the same semantic winner status
- the same score-vector digest
- the same ranking digest

Raw floating-point diagnostics may differ only when the quantized evidence, ranking, and tie semantics remain identical.

For `P3-B3`, direct frozen `B3` construction remains the semantic source of truth.

For `P2`, the registration search space and deterministic registration tie-breaking order remain frozen exactly.

## Benchmark Invariants

Optimization may not change:

- episode IDs
- frame IDs
- observation pixels
- split membership
- episode counts
- frame counts
- transformation parameters
- expected rows
- expected actions
- actual executed actions
- invalid-input classifications
- temporal-negative classifications
- control classifications

## Final-Data Invariant

Optimization may not:

- materialize final pixels
- score final frames
- inspect final expected rows
- inspect final expected actions

## Permitted Optimizations

- vectorized candidate scoring
- batched candidate scoring
- precomputed candidate tensors
- precomputed masks
- precomputed denominators
- precomputed regional statistics
- shared immutable provider setup
- content-addressed caching
- atomic evidence shards
- checkpoint/resume
- process-level parallelism
- memory mapping
- deterministic merge of independent shards

## Prohibited Optimizations

- candidate pruning
- approximate nearest-neighbour search
- early exit based on expected row
- lower-resolution substitutes
- reduced registration search
- changed masks
- changed region weights
- changed tie epsilon
- changed quantization
- changed score formula

## Central Invariant

Optimization may make the benchmark faster and resumable. It may not change a single identity-bearing quantized score, semantic ranking, tie group, episode, observation, or split.
