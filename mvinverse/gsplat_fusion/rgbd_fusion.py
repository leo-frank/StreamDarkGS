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
        "front_depth_created": 0,
        "covered_same_or_behind_skipped": 0,
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
        "scales": append("scales", frame.scales),
        "opacities": append("opacities", frame.opacities),
        "confidence_sum": torch.cat((state.confidence_sum, confidence), dim=0),
        "update_count": torch.cat(
            (state.update_count, torch.ones_like(confidence)), dim=0
        ),
    }
    current_quats = state.quats
    if current_quats is None:
        current_quats = torch.zeros(
            (state.means_world.shape[0], 4),
            device=device,
            dtype=frame.quats.dtype,
        )
        current_quats[:, 0] = 1.0
    fields["quats"] = torch.cat(
        (current_quats.to(device), frame.quats.to(device)[rows]), dim=0
    )

    return GaussianMapState(**fields)


def _render_coverage_and_depth(
    state: GaussianMapState,
    frame: FrameGaussians,
    config: RGBDGaussianFusionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    depth = rendered["depth"].to(device=device, dtype=state.means_world.dtype)
    if depth.dim() == 3:
        depth = depth[0]
    sample_y = frame.sample_y.to(device)
    sample_x = frame.sample_x.to(device)
    return coverage[sample_y, sample_x], depth[sample_y, sample_x]


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
    creation_gate, gate_stats, current_depth = _confidence_depth_creation_mask(
        frame_means,
        frame_confidence,
        frame.camera,
        config,
    )
    if state.means_world.shape[0] == 0:
        coverage = torch.zeros(count, device=device, dtype=frame_means.dtype)
        rendered_depth = torch.zeros_like(coverage)
    else:
        coverage, rendered_depth = _render_coverage_and_depth(state, frame, config)

    covered = coverage >= max(float(config.first_hit_coverage_threshold), 0.0)
    valid_depth_pair = (
        torch.isfinite(current_depth)
        & torch.isfinite(rendered_depth)
        & (current_depth > 1e-6)
        & (rendered_depth > 1e-6)
    )
    margin = max(float(config.front_depth_relative_margin), 0.0)
    sufficiently_in_front = (
        covered
        & valid_depth_pair
        & (current_depth < rendered_depth * (1.0 - margin))
    )
    # Low map coverage retains the original completion behavior.  With strong
    # coverage, only a clearly closer observation may introduce a new surface;
    # similar or farther Pi3 depths are conservatively suppressed.
    creation_candidate = creation_gate & (~covered | sufficiently_in_front)
    create_mask = creation_candidate & bool(allow_create)
    state = _append_frame_gaussians(state, frame, create_mask)

    stats = _empty_stats(state.means_world.shape[0])
    stats.update(gate_stats)
    stats["created"] = int(create_mask.sum().item())
    stats["creation_suppressed"] = int((creation_candidate & ~create_mask).sum().item())
    stats["covered_creation_skipped"] = int(covered.sum().item())
    stats["front_depth_created"] = int((create_mask & sufficiently_in_front).sum().item())
    stats["covered_same_or_behind_skipped"] = int(
        (creation_gate & covered & ~sufficiently_in_front).sum().item()
    )
    return state, stats
