from __future__ import annotations

import torch

from .rgbd import (
    FrameGaussians,
    GaussianMapState,
    RGBDGaussianFusionConfig,
    _confidence_depth_creation_mask,
)


def _empty_stats(total: int) -> dict[str, int]:
    return {
        "created": 0,
        "creation_suppressed": 0,
        "covered_creation_skipped": 0,
        "create_confidence_skipped": 0,
        "create_depth_skipped": 0,
        "merged": 0,
        "total": total,
    }


def _append_frame_gaussians(
    state: GaussianMapState,
    frame: FrameGaussians,
    create_mask: torch.Tensor,
) -> GaussianMapState:
    rows = create_mask.nonzero(as_tuple=False).squeeze(-1)
    if rows.numel() == 0:
        return state

    device = state.means_world.device
    confidence = frame.confidence.to(device)[rows]

    def append(name: str, source: torch.Tensor) -> torch.Tensor:
        current = getattr(state, name)
        return torch.cat((current, source.to(device)[rows]), dim=0)

    fields = {
        "means_world": append("means_world", frame.means_world),
        "colors": append("colors", frame.albedo),
        "albedo": append("albedo", frame.albedo),
        "roughness": append("roughness", frame.roughness),
        "metallic": append("metallic", frame.metallic),
        "normals_world": append("normals_world", frame.normals_world),
        "scales": append("scales", frame.scales),
        "opacities": append("opacities", frame.opacities),
        "confidence_sum": torch.cat((state.confidence_sum, confidence), dim=0),
        "update_count": torch.cat(
            (state.update_count, torch.ones_like(confidence)), dim=0
        ),
    }

    return GaussianMapState(**fields)


def _render_coverage(
    state: GaussianMapState,
    frame: FrameGaussians,
    config: RGBDGaussianFusionConfig,
) -> torch.Tensor:
    from .gsplat_adapter import render_gaussian_map_association

    device = state.means_world.device
    rendered = render_gaussian_map_association(
        state,
        frame.camera,
        device=device,
        backend="gsplat_2dgs",
        planar_scale=float(config.render_planar_scale),
        thickness_scale=float(config.render_thickness_scale),
    )
    coverage = rendered["coverage"].to(device=device, dtype=state.means_world.dtype)
    return coverage[frame.sample_y.to(device), frame.sample_x.to(device)]


def fuse_frame_gaussians(
    state: GaussianMapState,
    frame: FrameGaussians,
    config: RGBDGaussianFusionConfig | None = None,
    allow_create: bool = True,
) -> tuple[GaussianMapState, dict[str, int]]:
    config = config or RGBDGaussianFusionConfig()
    count = frame.means_world.shape[0]
    if count == 0:
        stats = _empty_stats(state.means_world.shape[0])
        return state, stats

    device = state.means_world.device
    frame_means = frame.means_world.to(device)
    frame_confidence = frame.confidence.to(device)
    creation_gate, gate_stats, _ = _confidence_depth_creation_mask(
        frame_means,
        frame_confidence,
        frame.camera,
        config,
    )
    if state.means_world.shape[0] == 0:
        coverage = torch.zeros(count, device=device, dtype=frame_means.dtype)
    else:
        coverage = _render_coverage(state, frame, config)

    covered = coverage >= max(float(config.first_hit_coverage_threshold), 0.0)
    creation_candidate = creation_gate & ~covered
    create_mask = creation_candidate & bool(allow_create)
    state = _append_frame_gaussians(state, frame, create_mask)

    stats = _empty_stats(state.means_world.shape[0])
    stats.update(gate_stats)
    stats["created"] = int(create_mask.sum().item())
    stats["creation_suppressed"] = int((creation_candidate & ~create_mask).sum().item())
    stats["covered_creation_skipped"] = int(covered.sum().item())
    return state, stats
