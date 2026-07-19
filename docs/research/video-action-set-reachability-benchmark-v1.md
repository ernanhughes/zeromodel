# Video Action-Set Reachability Benchmark v1

Version: `zeromodel-video-action-set-reachability-benchmark/v1`

This contract freezes the prospective evidence-preserving benchmark instrument for action-set and reachability analysis before any benchmark identities or observations are generated.

## Scientific Hypotheses

### H1 — Action Equivalence

Some visual row-identification errors preserve the exact policy action.

### H2 — Action-Unanimous Candidate Sets

A bounded row candidate set may support safe policy execution when all supported rows map to one action, even when exact row identity is not established.

### H3 — Invalid-Input Separation

Correct action unanimity on valid observations does not imply valid-input membership. Invalid-input action support must be measured separately.

### H4 — Reachability Composition

Intersecting current visual evidence with a separately identified reachability artifact may reduce row and action ambiguity without injecting unsupported rows.

### H5 — Evidence Preservation

Every reported candidate-set or temporal finding must be reproducible from committed per-observation score vectors and composition traces.

## Providers

- primary provider: `P3-B3 prospective joint fit`
- baseline provider: `P1 prospective normalized pixel similarity`
- baseline provider: `P2 prospective registered local correlation`

The primary scientific conclusion will be based on `P3-B3`. The baseline providers contextualize the historical System B and R1 action-gap signal.

## Versions

- benchmark: `zeromodel-video-action-set-reachability-benchmark/v1`
- generator: `zeromodel-video-action-set-reachability-generator/v1`
- evidence schema: `zeromodel-video-complete-row-evidence/v1`
- episode schema: `zeromodel-video-policy-episode/v1`
- score quantizer: `zeromodel-video-score-quantizer/v1`
- ranking schema: `zeromodel-video-complete-ranking/v1`
- provider P1: `zeromodel-video-prospective-normalized-pixel/v1`
- provider P2: `zeromodel-video-prospective-local-correlation/v1`
- provider P3: `zeromodel-video-prospective-b3-joint-fit/v1`
- candidate-set protocol: `zeromodel-video-prospective-candidate-set/v1`
- conformal protocol: `zeromodel-video-prospective-episode-conformal/v1`
- reachability composition: `zeromodel-video-prospective-reachability-composition/v1`
- phase-access audit: `zeromodel-video-prospective-phase-access/v1`

## Split Unit

The independent split unit is `episode`. Frames from one episode must never cross splits.

### Calibration Split

- `112` valid episodes
- `4` frames per episode
- `448` valid frames

### Architecture-Selection Split

- `112` valid episodes
- `56` distinguishable frame-invalid episodes
- `56` temporal-negative episodes
- `28` information-theoretic control episodes
- `4` frames per episode
- total episodes: `252`
- total frames: `1008`

### Final Split

The final split is frozen with the same episode counts and frame counts as architecture selection. The final split plan may be sealed and hashed in this block, but its observations must not be materialized or scored.

## Valid Episode Construction

- one valid episode begins from each of the `112` policy rows
- episode length is `4` frames
- every valid frame preserves:
  - current row
  - expected policy action
  - actual executed action
  - next row
  - transition source
  - transition choice index

For ordinary valid episodes, `actual executed action = exact policy action for current row`, but expected and actual action remain separate fields.

## Valid Observation Families

The valid family schedule is balanced deterministically over:

- `exact`
- `bounded_photometric`
- `bounded_translation`
- `bounded_translation_photometric`
- `bounded_translation_occlusion`
- `compound_bounded`

Every valid split must represent every family, every policy row, and every action.

## Distinguishable Frame-Invalid Episodes

Per selection/final split:

- `28` conflicting-action splice episodes
- `28` critical-evidence corruption episodes

These preserve no valid expected row and no valid expected action. Collisions with valid observations must be audited exhaustively.

## Temporal-Negative Episodes

Per selection/final split, freeze `14` episodes each of:

- `reordered_frames`
- `stale_repeated_frame`
- `impossible_transition`
- `declared_gap_or_unknown_action`

Unknown action or declared gap must force temporal reset in later replay blocks.

## Information-Theoretic Controls

Per selection/final split, freeze `28` control episodes. Controls record pixel-identical or legitimately equivalent collisions and remain outside distinguishable-invalid denominators.

## Complete Evidence Schema

For every provider and observation preserve:

- benchmark version
- generator version
- benchmark identity
- split
- episode ID
- clip ID
- frame ID
- sequence number
- episode seed
- frame seed
- family
- expected disposition
- expected row or null
- expected action or null
- actual executed action or null
- action-known flag
- gap declaration
- source observation IDs
- transformation parameters
- provider ID
- provider version
- provider contract digest
- policy artifact ID
- reachability tile digest
- all `112` row IDs
- all `112` raw scores or distances
- all `112` quantized scores
- complete ordered ranking
- explicit tie groups
- winner row
- winner quantized score
- runner-up row
- runner-up quantized score
- score-vector digest
- ranking digest
- observation-pixel digest

No score vector may omit a policy row. No ranking may silently break a numerical tie.

## Score Quantization

- quantization version: `zeromodel-video-score-quantizer/v1`
- quantization scale: `1,000,000`
- valid quantized range: `0` through `1,000,000`

For a bounded similarity `s`:

`s_clamped = min(1, max(0, s))`

`q = floor(s_clamped * 1,000,000 + 0.5)`

Store both raw float score and quantized integer score. Identity, ordering, tie groups, and later candidate-set membership use the quantized integer. Any non-finite score invalidates provider evidence.

## Ranking and Ties

Rank by descending quantized similarity. Distance-native providers must convert to bounded similarity before quantization.

When multiple rows share the same quantized score:

- keep them in one semantic tie group
- retain every row
- sort lexically only inside the persisted trace
- lexical order may not create semantic uniqueness
- top-k reconstruction must include the full boundary tie group

## Provider Formulas

### P1 — Normalized Pixel Similarity

`normalized absolute error = sum(valid_pixel_weight * abs(observation - candidate)) / (255 * sum(valid_pixel_weight))`

`similarity = 1 - normalized absolute error`

### P2 — Registered Local Correlation

Per region:

`regional similarity = max(0, 1 - normalized_correlation_distance / 2)`

Registration bounds and tie semantics reuse the frozen v3 registration constraints where applicable.

### P3 — B3 Joint Fit

`B3 = 1 - sum(region_weight * mask * abs(observation - candidate) / 255) / sum(region_weight * mask)`

The B3 scoring mechanism remains unchanged. The prospective wrapper gets a new provider identity because its evidence-record contract is new.

## Candidate-Set Methods Frozen Now, Not Run

### Fixed Top-k

`k in {1, 2, 3, 5}` with complete boundary tie inclusion.

### Fixed Score Gaps

Quantized similarity gaps in:

`{0, 2500, 5000, 10000, 25000, 50000}`

### Episode-Level Conformal

Freeze `alpha in {0.10, 0.05, 0.01}` with episode-level nonconformity defined as the maximum over valid frames of `1 - true_row_similarity`.

### Maximum Executable Set Size

Freeze maximum executable candidate-set size at `3`.

## Action Image

For any future row set `V`:

`ActionImage(V) = { policy_action(row) : row in V }`

Later interpretation rules:

- `V empty -> no_sufficient_evidence`
- `|ActionImage(V)| = 1 and |V| <= 3 -> action_unanimous_candidate_set`
- `|ActionImage(V)| > 1 -> action_ambiguous_candidate_set`
- `|V| > 3 -> candidate_set_too_large`

## Selection Risk Framework Frozen Now

Later feasibility will use episode-level failure events:

- valid wrong-action episode
- invalid-support episode
- temporal safety episode

Selection is not run in this block.

## Reachability Composition Frozen Now

Bind:

- reachability tile version: `zeromodel-video-policy-reachability-tile/v1`
- reachability tile digest: `sha256:fef2bc5fd795bb92d3bd564bccdc2d32e1b23319aba55dffed5e0391e795a5df`

Later replay contract:

`B_t = retained visual row belief after frame t`

`a_t = actual executed action after frame t`

`R_(t+1) = union over row in B_t of T(row, a_t)`

`B_(t+1) = V_(t+1) intersection R_(t+1)`

Require `B_(t+1) subset of V_(t+1)`. Reachability may remove rows and may never inject rows absent from current visual evidence.

## Output Directory

All prospective benchmark outputs for this block are rooted at:

`docs/results/video-action-set-reachability-benchmark-v1/`

## Absolute Boundaries

This block must not:

- materialize or score the final split
- run top-k utility, score-gap utility, conformal calibration, candidate-set selection, reachability replay, or final evaluation
- tune providers from benchmark outcomes
- train or fine-tune a model
- mutate parent audit artifacts, parent reachability artifacts, or the frozen v3 artifacts

## Central Invariant

Every future claim about action-equivalent candidate sets or reachability-constrained beliefs must be reconstructible from a committed 112-row score vector, complete ranking, explicit tie groups, exact policy identity, exact sequence identity, and actual executed action.
