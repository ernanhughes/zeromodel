# P16B Scope Boundary

This repair slice is limited to dataset partition ownership and exact duplicate leakage.

It does not yet:

- create manifest-owned validation or test partition DTOs;
- change calibration or untouched-test APIs;
- detect perceptual near-duplicates;
- quantify temporal-window overlap;
- repair floating-point model identity;
- change P16 operational health statistics;
- implement P17 recommendations.
