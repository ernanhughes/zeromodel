"""Skeleton-based digit-branch counting (numpy only; Zhang-Suen thinning).

Heuristic evidence with a declared applicability limit: fingers that merge in
the silhouette become one skeleton branch, so counts are only meaningful
RELATIVE to the same measurement on the aligned template.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from examples.hand_anatomy.comparison import connected_components, palm_center

MIN_BRANCH_PIXELS = 6
PALM_RADIUS_FACTOR = 1.3


def thin(mask: np.ndarray, max_iter: int = 80) -> np.ndarray:
    img = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:]
            p6, p7, p8, p9 = img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2], img[:-2, :-2]
            c = img[1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(n.astype(np.int16) for n in nb)
            a = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                m1, m2 = (p2 * p4 * p6) == 0, (p4 * p6 * p8) == 0
            else:
                m1, m2 = (p2 * p4 * p8) == 0, (p2 * p6 * p8) == 0
            rem = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & m1 & m2
            if rem.any():
                img[1:-1, 1:-1][rem] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1] > 0


def digit_branches(mask: np.ndarray) -> Dict[str, Any]:
    """Skeleton components lying beyond the palm circle (finger/forearm-like branches)."""
    (cy, cx), r0 = palm_center(mask)
    skel = thin(mask)
    ys, xs = np.nonzero(skel)
    far = np.zeros_like(skel)
    keep = np.hypot(ys - cy, xs - cx) > PALM_RADIUS_FACTOR * r0
    far[ys[keep], xs[keep]] = True
    _, areas = connected_components(far, min_area=MIN_BRANCH_PIXELS)
    return {"count": len(areas), "lengths": sorted(areas, reverse=True), "palm_radius": r0}
