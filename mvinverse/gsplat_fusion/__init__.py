"""Gaussian-map construction and first-hit RGB-D completion."""

from .rgbd import (
    FrameGaussians,
    GaussianMapState,
    RGBDGaussianBuilderConfig,
    RGBDGaussianFusionConfig,
    build_frame_gaussians,
    fuse_frame_gaussians,
)
from .types import PinholeCamera

__all__ = [
    "build_frame_gaussians",
    "FrameGaussians",
    "fuse_frame_gaussians",
    "GaussianMapState",
    "PinholeCamera",
    "RGBDGaussianBuilderConfig",
    "RGBDGaussianFusionConfig",
]
