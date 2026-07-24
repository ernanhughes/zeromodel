# Selected image evidence

These PNGs are copied byte-for-byte from the local result archive. The complete image set remains in the preserved local archive.

| File | Variant/state | Reason | SHA-256 |
|---|---|---|---|
| `labelled-blocked-control.png` | `labelled-v1` · `tank=6 | target=none | cooldown=1` | Labelled blocked/no-target control; predicted exactly. | `592e6655929b58eea7589d6f8672388e48ac7046829f332ebd168c65e160c99b` |
| `unlabelled-blocked-failure.png` | `unlabelled-v1` · `tank=6 | target=none | cooldown=1` | Same bounded state without labels; cooldown was read as READY but action remained STAY. | `e6c04f9eb0af8fbfd05bc3b7aeacf537154be03730393362ad0fed9a04771ac5` |
| `unlabelled-action-changing.png` | `unlabelled-v1` · `tank=3 | target=3 | cooldown=1` | Baseline action-changing case: predicted tank 4 and READY instead of tank 3 and BLOCKED. | `3b89d349b810f42263c6d109af31cb40ce493dd214c5304b6bac4827643a674a` |
| `cooldown-shape-action-changing.png` | `cooldown-shape-v1` · `tank=3 | target=3 | cooldown=1` | Shape overlay did not repair cooldown and retained the baseline action-changing error. | `0b3444d4dc760f64cbd6c95cf1863250d773d5ec92145c9d95e828740f304d25` |
| `cooldown-redundant-action-changing.png` | `cooldown-redundant-v1` · `tank=3 | target=1 | cooldown=1` | Redundant marker caused target absence and a second policy-changing error. | `664d91aeb2c15c693c78a6cb8a6f13849944b3dbe38b957d73b74a6db50b88d4` |
| `lane-enhanced-rejected-column-7.png` | `lane-enhanced-v1` · `tank=6 | target=none | cooldown=1` | Model returned TANK_COLUMN: 7; separator emphasis created an invalid eighth-boundary interpretation. | `6c15f924db9c2334add62a1706271c17241961ef0cc0d698bc6bdfdf938d3d6b` |
| `lane-enhanced-action-changing.png` | `lane-enhanced-v1` · `tank=0 | target=0 | cooldown=0` | Target shifted from lane 0 to lane 1, changing FIRE to RIGHT. | `b22a27c2b171eb1e695809476a4db2bed39aa79646768586ad8850924d2e1e66` |
