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
    cluster_count: int,
    max_samples: int,
    spatial_weight: float,
) -> torch.Tensor | None:
    height, width = current.shape[-2:]
    color_features = current.permute(1, 2, 0).reshape(-1, current.shape[0]).float()
    y_coords = torch.linspace(
        0.0, 1.0, steps=height, device=current.device, dtype=torch.float32
    )
    x_coords = torch.linspace(
        0.0, 1.0, steps=width, device=current.device, dtype=torch.float32
    )
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    spatial_features = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
    features = torch.cat(
        (color_features, spatial_features * max(float(spatial_weight), 0.0)),
        dim=1,
    )
    count = int(features.shape[0])
    clusters = min(max(int(cluster_count), 1), count)
    if clusters <= 1:
        return None

    if count > max_samples:
        sample_indices = torch.linspace(
            0, count - 1, steps=max_samples, device=features.device
        ).round().long()
        fit_features = features[sample_indices]
    else:
        fit_features = features

    centers = torch.empty(
        (clusters, fit_features.shape[1]),
        device=fit_features.device,
        dtype=fit_features.dtype,
    )
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


def _smooth_cluster_labels(
    labels: torch.Tensor,
    cluster_count: int,
    kernel_size: int,
) -> torch.Tensor:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0 or kernel_size == 1:
        return labels
    one_hot = F.one_hot(labels.long(), num_classes=cluster_count)
    one_hot = one_hot.permute(2, 0, 1).float().unsqueeze(0)
    scores = F.avg_pool2d(
        one_hot,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )
    return scores.squeeze(0).argmax(dim=0)


def _sam_masks_to_labels(
    masks: list[dict[str, object]],
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    labels = torch.full(size, -1, dtype=torch.long, device=device)
    ordered_masks = sorted(masks, key=lambda item: int(item.get("area", 0)))
    for mask_index, mask_data in enumerate(ordered_masks):
        segmentation = torch.as_tensor(
            mask_data["segmentation"], dtype=torch.bool, device=device
        )
        if segmentation.shape != size:
            segmentation = F.interpolate(
                segmentation.float().unsqueeze(0).unsqueeze(0),
                size=size,
                mode="nearest",
            ).squeeze(0).squeeze(0).bool()
        labels[(labels < 0) & segmentation] = mask_index
    return labels


def _build_region_labels(
    *,
    source: str,
    current: torch.Tensor,
    cluster_count: int,
    cluster_sample_pixels: int,
    cluster_spatial_weight: float,
    cluster_smoothing_kernel_size: int,
    sam_mask_generator: object | None,
) -> torch.Tensor | None:
    if source == "sam":
        if sam_mask_generator is None:
            raise ValueError("SAM region source requires a mask generator")
        region_image = current
        if region_image.shape[0] == 1:
            region_image = region_image.expand(3, -1, -1)
        sam_image = (
            region_image.detach()
            .float()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .mul(255.0)
            .byte()
            .cpu()
            .numpy()
        )
        sam_masks = sam_mask_generator.generate(sam_image)
        return _sam_masks_to_labels(
            sam_masks,
            tuple(current.shape[-2:]),
            current.device,
        )
    if source != "kmeans":
        raise ValueError(f"Unknown material region source: {source}")
    labels = _fit_color_clusters(
        current,
        cluster_count,
        cluster_sample_pixels,
        cluster_spatial_weight,
    )
    if labels is None:
        return None
    return _smooth_cluster_labels(
        labels,
        cluster_count=int(labels.max().item()) + 1,
        kernel_size=cluster_smoothing_kernel_size,
    )


def _align_region_values(
    *,
    current: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
    labels: torch.Tensor | None,
    cluster_min_pixels: int,
    global_max_offset: float,
    cluster_max_offset: float,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    current_log = torch.log(current.float().clamp(1e-4, 1.0))
    reference_log = torch.log(reference.float().clamp(1e-4, 1.0))
    channel_valid = (
        valid
        & torch.isfinite(current).all(dim=0)
        & torch.isfinite(reference).all(dim=0)
    )
    difference = reference_log - current_log
    global_offset = difference[:, channel_valid].median(dim=1).values.clamp(
        -abs(global_max_offset), abs(global_max_offset)
    )
    correction = global_offset.view(-1, 1, 1).expand_as(current_log).clone()

    used_clusters = 0
    if labels is not None:
        for cluster_index in range(int(labels.max().item()) + 1):
            cluster_all = labels == cluster_index
            cluster_valid = channel_valid & cluster_all
            if int(cluster_valid.sum().item()) < cluster_min_pixels:
                continue
            cluster_offset = difference[:, cluster_valid].median(dim=1).values.clamp(
                -abs(cluster_max_offset), abs(cluster_max_offset)
            )
            correction[:, cluster_all] = cluster_offset.view(-1, 1)
            used_clusters += 1

    correction = correction.clamp(
        -abs(cluster_max_offset), abs(cluster_max_offset)
    ) * max(float(strength), 0.0)
    aligned = torch.exp(current_log + correction).clamp(0.0, 1.0)
    return aligned.to(current.dtype), global_offset, used_clusters


def _align_constant_regions(
    *, current: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor,
    labels: torch.Tensor | None, cluster_min_pixels: int,
) -> tuple[torch.Tensor, int]:
    """Use one robust scalar value for each material region."""
    aligned = current.clone()
    if labels is None:
        labels = torch.zeros(current.shape[-2:], dtype=torch.long, device=current.device)
    aligned_regions = 0
    for region_index in range(int(labels.max().item()) + 1):
        region = labels == region_index
        region_current = region & torch.isfinite(current[0])
        if int(region_current.sum().item()) < int(cluster_min_pixels):
            continue
        region_valid = region & valid
        current_median = current[0, region_current].float().median()
        if int(region_valid.sum().item()) >= int(cluster_min_pixels):
            region_value = reference[0, region_valid].float().median()
        else:
            region_value = current_median
        aligned[0, region] = region_value.clamp(0.0, 1.0).to(aligned.dtype)
        aligned_regions += 1
    return aligned, aligned_regions


def align_creation_material_to_map(
    *,
    state: GaussianMapState,
    camera,
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    device: str,
    coverage_threshold: float = 0.6,
    cluster_count: int = 6,
    cluster_min_pixels: int = 2048,
    cluster_sample_pixels: int = 50_000,
    cluster_spatial_weight: float = 0.2,
    cluster_smoothing_kernel_size: int = 5,
    region_source: str = "kmeans",
    sam_mask_generator: object | None = None,
    global_max_log_offset: float = 0.25,
    cluster_max_log_offset: float = 10.0,
    planar_scale: float = 1.1,
    thickness_scale: float = 0.05,
    roughness_mode: str = "region_constant",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
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

    labels = _build_region_labels(
        source=region_source,
        current=current,
        cluster_count=cluster_count,
        cluster_sample_pixels=cluster_sample_pixels,
        cluster_spatial_weight=cluster_spatial_weight,
        cluster_smoothing_kernel_size=cluster_smoothing_kernel_size,
        sam_mask_generator=sam_mask_generator,
    )
    if labels is not None:
        labels = labels.to(device=valid.device, dtype=torch.long)
        stats["albedo_cluster_labels"] = labels.detach().cpu()
        stats["albedo_cluster_valid"] = valid.detach().cpu()

    aligned, global_offset, used_clusters = _align_region_values(
        current=albedo,
        reference=reference,
        valid=valid,
        labels=labels,
        cluster_min_pixels=cluster_min_pixels,
        global_max_offset=global_max_log_offset,
        cluster_max_offset=cluster_max_log_offset,
        strength=1.0,
    )
    reference_roughness = _resize_channel(rendered["roughness"], size).to(roughness)
    roughness_valid = (
        torch.isfinite(roughness).all(dim=0)
        & torch.isfinite(reference_roughness).all(dim=0)
        & torch.isfinite(coverage)
        & (coverage >= coverage_threshold)
    )
    if roughness_mode not in {"offset", "region_constant"}:
        raise ValueError(
            f"Unsupported roughness alignment mode: {roughness_mode}. "
            "Expected 'offset' or 'region_constant'."
        )
    roughness_offset = roughness.new_zeros(1)
    if roughness_mode == "region_constant":
        aligned_roughness, roughness_clusters = _align_constant_regions(
            current=roughness,
            reference=reference_roughness,
            valid=roughness_valid,
            labels=labels,
            cluster_min_pixels=cluster_min_pixels,
        )
    else:
        aligned_roughness, roughness_offset, roughness_clusters = _align_region_values(
            current=roughness,
            reference=reference_roughness,
            valid=roughness_valid,
            labels=labels,
            cluster_min_pixels=cluster_min_pixels,
            global_max_offset=global_max_log_offset,
            cluster_max_offset=cluster_max_log_offset,
            strength=1.0,
        )
    if labels is not None:
        stats["roughness_cluster_labels"] = labels.detach().cpu()
        stats["roughness_cluster_valid"] = roughness_valid.detach().cpu()

    reference_metallic = _resize_channel(rendered["metallic"], size).to(metallic)
    metallic_valid = (
        torch.isfinite(metallic).all(dim=0)
        & torch.isfinite(reference_metallic).all(dim=0)
        & torch.isfinite(coverage)
        & (coverage >= coverage_threshold)
    )
    aligned_metallic, metallic_clusters = _align_constant_regions(
        current=metallic,
        reference=reference_metallic,
        valid=metallic_valid,
        labels=labels,
        cluster_min_pixels=cluster_min_pixels,
    )
    if labels is not None:
        stats["metallic_cluster_labels"] = labels.detach().cpu()
        stats["metallic_cluster_valid"] = metallic_valid.detach().cpu()
    stats.update(
        {
            "applied": True,
            "global_offset": [float(value) for value in global_offset.detach().cpu()],
            "used_clusters": used_clusters,
            "mean_abs_delta": float(
                (aligned.float() - albedo.float()).abs().mean().item()
            ),
            "roughness_global_offset": float(roughness_offset.item()),
            "roughness_used_clusters": roughness_clusters,
            "roughness_mean_abs_delta": float(
                (aligned_roughness.float() - roughness.float()).abs().mean().item()
            ),
            "metallic_used_clusters": metallic_clusters,
            "metallic_mean_abs_delta": float(
                (aligned_metallic.float() - metallic.float()).abs().mean().item()
            ),
        }
    )
    return aligned, aligned_roughness, aligned_metallic, stats
