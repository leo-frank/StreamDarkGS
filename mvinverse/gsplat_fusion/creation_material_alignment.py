from __future__ import annotations

import torch
import torch.nn.functional as F

from .gsplat_adapter import render_gaussian_map_association
from .rgbd import GaussianMapState


def _resize_channel(value: torch.Tensor | None, size: tuple[int, int]) -> torch.Tensor | None:
    if value is None:
        return None
    if value.dim() == 2:
        value = value.unsqueeze(0)
    if value.shape[-2:] == size:
        return value
    return F.interpolate(
        value.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _fit_color_clusters(
    current: torch.Tensor,
    valid: torch.Tensor,
    cluster_count: int,
    max_samples: int,
) -> torch.Tensor | None:
    height, width = current.shape[-2:]
    features = current.permute(1, 2, 0).reshape(-1, 3).float()
    valid_features = features[valid.reshape(-1)]
    count = int(valid_features.shape[0])
    clusters = min(max(int(cluster_count), 1), count)
    if clusters <= 1:
        return None

    if count > max_samples:
        sample_indices = torch.linspace(
            0, count - 1, steps=max_samples, device=valid_features.device
        ).round().long()
        fit_features = valid_features[sample_indices]
    else:
        fit_features = valid_features

    centers = torch.empty((clusters, 3), device=fit_features.device, dtype=fit_features.dtype)
    first = torch.argsort(fit_features.mean(dim=1))[fit_features.shape[0] // 2]
    centers[0] = fit_features[first]
    distances = torch.sum((fit_features - centers[0]) ** 2, dim=1)
    for cluster_index in range(1, clusters):
        next_index = torch.argmax(distances)
        centers[cluster_index] = fit_features[next_index]
        distances = torch.minimum(
            distances,
            torch.sum((fit_features - centers[cluster_index]) ** 2, dim=1),
        )

    for _ in range(8):
        labels = torch.argmin(torch.cdist(fit_features, centers), dim=1)
        updated = centers.clone()
        for cluster_index in range(clusters):
            selected = labels == cluster_index
            if selected.any():
                updated[cluster_index] = fit_features[selected].median(dim=0).values
        if torch.max(torch.abs(updated - centers)).item() < 1e-4:
            centers = updated
            break
        centers = updated

    return torch.argmin(torch.cdist(features, centers), dim=1).view(height, width)


def align_creation_material_to_map(
    *,
    state: GaussianMapState,
    camera,
    albedo: torch.Tensor,
    roughness: torch.Tensor | None,
    metallic: torch.Tensor | None,
    device: str,
    coverage_threshold: float = 0.6,
    cluster_count: int = 6,
    cluster_min_pixels: int = 2048,
    cluster_sample_pixels: int = 50_000,
    global_max_log_offset: float = 0.25,
    cluster_max_log_offset: float = 10.0,
    planar_scale: float = 1.1,
    thickness_scale: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, object]]:
    stats: dict[str, object] = {"applied": False, "valid_pixels": 0}
    if state.means_world.shape[0] == 0:
        stats["reason"] = "empty_map"
        return albedo, roughness, metallic, stats

    rendered = render_gaussian_map_association(
        state=state,
        camera=camera,
        device=device,
        backend="gsplat_2dgs",
        planar_scale=planar_scale,
        thickness_scale=thickness_scale,
    )
    size = tuple(albedo.shape[-2:])
    reference = _resize_channel(rendered["albedo"], size).to(albedo)
    coverage = _resize_channel(rendered["coverage"], size).squeeze(0).to(albedo)
    current = albedo.float().clamp(1e-4, 1.0)
    reference = reference.float().clamp(1e-4, 1.0)
    valid = (
        torch.isfinite(current).all(dim=0)
        & torch.isfinite(reference).all(dim=0)
        & torch.isfinite(coverage)
        & (coverage >= coverage_threshold)
        & (current.mean(dim=0) > 0.03)
        & (reference.mean(dim=0) > 0.03)
    )
    valid_count = int(valid.sum().item())
    stats["valid_pixels"] = valid_count
    if valid_count < 512:
        stats["reason"] = "too_few_valid_pixels"
        return albedo, roughness, metallic, stats

    log_difference = torch.log(reference) - torch.log(current)
    global_offset = log_difference[:, valid].median(dim=1).values.clamp(
        -abs(global_max_log_offset), abs(global_max_log_offset)
    )
    correction = global_offset.view(3, 1, 1).expand_as(current).clone()
    labels = _fit_color_clusters(current, valid, cluster_count, cluster_sample_pixels)
    used_clusters = 0
    if labels is not None:
        residual = log_difference - global_offset.view(3, 1, 1)
        for cluster_index in range(int(labels.max().item()) + 1):
            selected = valid & (labels == cluster_index)
            if int(selected.sum().item()) < cluster_min_pixels:
                continue
            cluster_offset = global_offset + residual[:, selected].median(dim=1).values
            cluster_offset = cluster_offset.clamp(
                -abs(cluster_max_log_offset), abs(cluster_max_log_offset)
            )
            correction[:, labels == cluster_index] = cluster_offset.view(3, 1)
            used_clusters += 1

    correction = correction.clamp(-abs(cluster_max_log_offset), abs(cluster_max_log_offset))
    aligned = torch.exp(torch.log(current) + correction).clamp(0.0, 1.0).to(albedo.dtype)
    stats.update(
        {
            "applied": True,
            "global_offset": [float(value) for value in global_offset.detach().cpu()],
            "used_clusters": used_clusters,
            "mean_abs_delta": float((aligned.float() - albedo.float()).abs().mean().item()),
        }
    )
    return aligned, roughness, metallic, stats
