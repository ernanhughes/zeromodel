# Manual representation inventory (Step 2, written before any compiler code)

Every representation below was hand-selected in stages 1-3. This table is the
compiler's ground truth for "what should a correct compilation look like,"
and the source for this experiment's rediscovery tests (cooldown, tank).

| # | Requirement | Manual region | Manual resolution | Manual aggregation | Manual decoder | Known failure (if any) |
|---|---|---|---|---|---|---|
| 1 | Arcade tank presence | rows 11-13, any column | 4x1px (coarse) | P18A per-tile mean, threshold | presence/absence via `changed_field_ids` | none |
| 2 | Arcade alien presence | rows 2-4, any column | 4x1px (coarse) | P18A per-tile mean, threshold | presence/absence | none (no strict expectation; monitored only) |
| 3 | Arcade cooldown presence | rows 7-8, rightmost tile | 4x1px (coarse) | P18A per-tile mean, threshold | presence/absence | none |
| 4 | Arcade background presence | everything else | 4x1px (coarse) | P18A per-tile mean, threshold | presence/absence | none |
| 5 | Arcade tank position/direction/magnitude | tank band, all 7 columns | 1x1px (fine) | **mean** per column over 3 row-tiles | argmax column, alive threshold 0.05 | **max aggregation tied the true column with a 1px bleed column** (tank's 5px base is 1px wider than its 4px cell); fixed by switching to mean |
| 6 | Arcade cooldown value | rows 7-8, rightmost 2 cols | 1x1px (fine) | mean over exactly the 4 real cooldown pixels | nearest-canonical-level (ready=40, blocked=160, tol=15/255) | **4x1px tile diluted 2 real pixels with 2 always-zero background pixels**, halving decoded intensity (40->20, 160->80), matching neither level; fixed by 1x1px resolution |
| 7 | Arcade alien target identity | alien band, any column | 1x1px (fine) | mean per column | argmax column (gives a *position*, not a *persistent identity*) | **hidden**: the true next-in-queue alien is never rendered; only the current front target is drawn. Not a resolution problem -- the evidence does not exist in the frame. |
| 8 | Warehouse robot/crate presence | full interior grid (9 cells, robot+crate share the region) | 1x1px (compiler-selected) | **mode** over each 6x6 cell's 36 pixels | nearest-canonical-level, 6-way (empty/goal_ring/wall/crate/robot) + real-object-vs-background attribution rule | **(a)** initial 10/255 tolerance collided wall(60) with goal_ring(50) (10 apart), misclassifying by dict-iteration order; fixed with nearest-match + 4/255 tolerance. **(b)** a vacated real-object cell was initially misattributed to "background"; fixed by attributing to the real object whenever either before/after type is real |
| 9 | Warehouse door state | door cell, bar sub-region only (rows 0-5, cols 2-3 of the cell) | 1x1px (fine) | mean over the 12-pixel bar sub-region only | nearest-canonical-level (closed=full bar, open=half bar) | **direct analogue of #6**: whole-cell mode classification reads the cell's majority-empty background (4 of 6 columns always 0) and misclassifies every door state as "empty"; fixed with a dedicated sub-region decoder |
| 10 | Warehouse battery value | 3 fixed segment pixels in the battery strip | 1x1px | single representative pixel per segment | threshold count of lit segments | none |
| 11 | Warehouse robot position/direction/magnitude | full interior grid | 1x1px | mode per cell (reused occupant classifier) | argmax cell classified "robot" | none -- the robot glyph fills its cell uniformly with no neighbor bleed, so mode/mean/max would all agree here |
| 12 | Warehouse crate identity | 3 fixed 2x2px corner sub-positions per cell | 1x1px (fine; needed to isolate 2x2 sub-regions) | none (direct per-pixel min over each 2x2 patch) | prefix dot-count (1/2/3 dots -> identity 0/1/2) | measured, not a resolution failure: 25.0% correctness against ground truth on a set that mixes correctly-rendered and deliberately-corrupted markers; the decoder reads exactly what is rendered, but several fault categories render the wrong marker by construction |
| 13 | Warehouse push relation | robot decoded position + newly-appearing crate cells | 1x1px | n/a (relation over two decoders) | Manhattan-adjacency check between new crate cell and robot before/after position | **irreducible, renderer-level**: robot always lands exactly on the crate's pre-push cell; a crate that fails to follow is z-order-occluded (painted over) for any pixel-based system, at any resolution |

## What this predicts for the compiler

- Requirements 1-4, 10, 11: any reasonable coarse-or-finer candidate should compile cleanly; no known failure to rediscover.
- Requirement 5 (tank position): the compiler must independently reject a **max**-aggregation candidate for ambiguity and select **mean** -- this is the primary rediscovery test.
- Requirement 6 (cooldown value) and 9 (door state): the compiler must independently reject a coarse (>=2px covering background) candidate for reconstruction loss and select a fine, sub-region-isolated candidate -- the secondary rediscovery test.
- Requirement 7 (alien identity): the compiler must report `insufficient_observability`, not `insufficient_representation` -- the evidence is absent from the frame, not poorly encoded.
- Requirement 12 (crate identity): the compiler can only be evaluated on whether it selects the best *available* candidate; ground-truth correctness on deliberately-corrupted instances is a property of the dataset, not the representation, and must not be conflated with `insufficient_representation`.
- Requirement 13 (push relation): out of scope for per-property candidate compilation as specified here (it is a relation over two already-compiled decoders); recorded so the limitation is not silently repeated as if newly discovered.
