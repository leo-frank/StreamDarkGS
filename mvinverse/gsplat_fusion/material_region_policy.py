"""Shared material-region boundary policy.

The label filler and generated-region cleanup must agree on what constitutes a
material edge.  Keeping this small policy independent avoids a split stage
accepting a boundary that the merge stage immediately rejects (or vice versa).
"""
from __future__ import annotations

import numpy as np


def boundary_evidence(
    inside: np.ndarray,
    outside: np.ndarray,
    tolerance: float,
) -> tuple[int, float, float]:
    """Return boundary length, strong-edge fraction and median color contrast."""
    if inside.size == 0 or outside.size == 0:
        return 0, 0.0, 0.0
    contrast = np.linalg.norm(
        np.asarray(inside, dtype=np.float32)
        - np.asarray(outside, dtype=np.float32), axis=1,
    )
    threshold = max(float(tolerance), 0.0) * 0.5
    return (
        int(contrast.size),
        float(np.mean(contrast > threshold)),
        float(np.median(contrast)),
    )


def weak_boundary(
    length: int,
    strong_fraction: float,
    mean_contrast: float,
    tolerance: float,
    *,
    weak_fraction: float = 0.90,
) -> bool:
    """Whether a shared boundary is safe for a generated-region merge."""
    if length <= 0:
        return False
    threshold = max(float(tolerance), 0.0) * 0.5
    return (
        strong_fraction <= 1.0 - float(weak_fraction)
        and mean_contrast <= threshold
    )


def coherent_split(
    area: int,
    local_contrast: float,
    strong_fraction: float,
    tolerance: float,
    min_pixels: int,
) -> bool:
    """Whether a color outlier has enough boundary evidence to stay separate."""
    tolerance = max(float(tolerance), 0.0)
    if area >= int(min_pixels):
        return local_contrast >= tolerance * 0.5 and strong_fraction >= 0.25
    return local_contrast >= tolerance and strong_fraction >= 0.60
