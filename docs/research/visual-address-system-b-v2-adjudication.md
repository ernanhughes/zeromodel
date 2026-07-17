# System B v2 adjudication

**Status date:** 17 July 2026  
**Result directory:** `docs/results/visual-address-system-b-v2/`

## Decision

System B does not provide a viable bounded Level 1 operating region under the repaired calibration protocol.

The selected calibration rule was:

1. reject any candidate with a distinguishable rejection-calibration false accept;
2. reject any candidate with a conflicting-action acceptance;
3. among remaining candidates maximize accepted exact-row precision;
4. break ties by benign coverage;
5. break further ties by exact-row recall;
6. break further ties by deterministic quantile ordering.

The only feasible candidate under that rule was quantile `1.0`.

That candidate:

- achieved zero distinguishable false accepts on rejection calibration;
- collapsed benign calibration coverage to `1 / 1344`;
- rejected every benign final-evaluation case;
- produced final benign coverage `0 / 1344`;
- retained strong raw top-1 ranking before rejection:
  - top-1 benign exact-row accuracy: `1008 / 1344 = 75.0%`
  - top-1 benign action accuracy: `1302 / 1344 = 96.875%`

This is Outcome `C`: no feasible operating point.

## Strongest supported claim

The current normalized-pixel representation still ranks many benign observations correctly, but no calibration-selected operating point simultaneously preserves useful benign acceptance and zero distinguishable false acceptance.

## Unsupported claim

The repository does not support the claim that calibrated normalized pixels provide a useful bounded Level 1 visual reader for the arcade fixture.

## Next action

Proceed to a local baseline showdown, starting with:

1. bounded registration plus local comparison;
2. translation-equivariant local template matching;
3. deterministic connected-components and geometry extraction.

The translation family should be treated as the first mechanistic target because:

- final translation top-1 action accuracy remained high;
- final translation top-1 exact-row accuracy fell to `168 / 336`;
- the selected safe operating point rejected all translated benign cases.

That combination points to a locality and alignment failure, not a complete loss of task information.
