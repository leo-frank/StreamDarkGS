from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .gsplat_adapter import render_gaussian_map_association
from .rgbd import GaussianMapState


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _timer_start(device: torch.device) -> float:
    _sync_device(device)
    return time.perf_counter()


def _record_timing(
    timings: dict[str, float],
    name: str,
    start_time: float,
    device: torch.device,
) -> None:
    _sync_device(device)
    timings[name] = (time.perf_counter() - start_time) * 1000.0


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


def _fill_unassigned_labels(
    labels: torch.Tensor,
    *,
    image: torch.Tensor | np.ndarray,
    max_distance: float,
    new_region_min_pixels: int,
    new_region_max_std: float,
    max_color_distance: float,
    debug_outputs: dict[str, object] | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    max_distance = float(max_distance)
    stats = {
        "seam_filled_pixels": 0,
        "new_region_pixels": 0,
        "new_region_count": 0,
        "merged_region_pixels": 0,
        "merged_region_count": 0,
        "color_rejected_pixels": 0,
    }
    if not bool((labels < 0).any()):
        return labels, stats
    labels_np = labels.detach().cpu().numpy().astype(np.int32, copy=True)
    valid = labels_np >= 0
    if not bool(valid.any()):
        return labels, stats
    if isinstance(image, torch.Tensor):
        image_float = (
            image.detach()
            .float()
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        image_u8 = np.round(image_float * 255.0).astype(np.uint8)
    else:
        image_array = np.asarray(image)
        if image_array.dtype == np.uint8:
            image_u8 = image_array.copy()
            image_float = image_u8.astype(np.float32) / 255.0
        else:
            image_float = np.asarray(image_array, dtype=np.float32)
            if image_float.max(initial=0.0) > 1.0:
                image_float = image_float / 255.0
            image_u8 = np.round(np.clip(image_float, 0.0, 1.0) * 255.0).astype(
                np.uint8
            )
    missing = ~valid
    nearest_distance = cv2.distanceTransform(
        missing.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_3,
    )
    missing_component_count, _, _, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), connectivity=8
    )
    deep_core = missing & (nearest_distance > max_distance)
    component_count, components, component_stats, _ = cv2.connectedComponentsWithStats(
        deep_core.astype(np.uint8), connectivity=8
    )
    next_label = int(labels_np.max()) + 1
    markers = np.zeros(labels_np.shape, dtype=np.int32)
    markers[valid] = labels_np[valid] + 1
    label_colors: dict[int, np.ndarray] = {}
    for label_index in np.unique(labels_np[valid]):
        label_mask = valid & (labels_np == label_index)
        label_colors[int(label_index)] = np.median(image_float[label_mask], axis=0)
    for component_index in range(1, component_count):
        component_area = int(component_stats[component_index, cv2.CC_STAT_AREA])
        if component_area < int(new_region_min_pixels):
            continue
        core = components == component_index
        core_std = float(image_float[core].std(axis=0).mean())
        if core_std <= float(new_region_max_std):
            markers[core] = next_label + 1
            label_colors[next_label] = np.median(image_float[core], axis=0)
            next_label += 1
            stats["new_region_count"] += 1

    if debug_outputs is not None:
        debug_outputs["deep_core"] = deep_core.copy()
        debug_outputs["new_region_markers"] = markers.copy() - 1

    watershed_markers = cv2.watershed(
        np.ascontiguousarray(image_u8[..., ::-1]), markers.copy()
    )
    if debug_outputs is not None:
        debug_outputs["watershed_raw"] = watershed_markers.copy() - 1
    filled_np = watershed_markers - 1
    unresolved = watershed_markers <= 0
    if bool(unresolved.any()):
        resolved = watershed_markers > 0
        _, nearest = cv2.distanceTransformWithLabels(
            unresolved.astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        resolved_y, resolved_x = np.nonzero(resolved)
        nearest_index = nearest - 1
        can_fill = unresolved & (nearest_index >= 0)
        filled_np[can_fill] = filled_np[
            resolved_y[nearest_index[can_fill]], resolved_x[nearest_index[can_fill]]
        ]
    filled_np[valid] = labels_np[valid]
    if debug_outputs is not None:
        debug_outputs["watershed_candidates"] = filled_np.copy()

    # Watershed supplies a spatially connected candidate. Accept that candidate only
    # when its albedo is compatible with the original SAM region (or new-region core).
    candidate_colors = np.zeros_like(image_float)
    for label_index, color in label_colors.items():
        candidate_colors[filled_np == label_index] = color
    candidate_color_distance = np.linalg.norm(
        image_float - candidate_colors, axis=2
    )
    rejected = missing & (candidate_color_distance > float(max_color_distance))
    stats["color_rejected_pixels"] = int(rejected.sum())
    if debug_outputs is not None:
        debug_outputs["color_rejected"] = rejected.copy()
        debug_outputs["candidate_color_distance"] = candidate_color_distance.copy()

    if bool(rejected.any()):
        accepted_np = filled_np.copy()
        accepted_np[rejected] = -1
        rejected_count, rejected_components, rejected_stats, _ = (
            cv2.connectedComponentsWithStats(
                rejected.astype(np.uint8), connectivity=8
            )
        )
        component_records: list[dict[str, object]] = []
        if debug_outputs is not None:
            debug_outputs["color_accepted_labels"] = accepted_np.copy()
            debug_outputs["rejected_components"] = rejected_components.copy()
        neighborhood_kernel = np.ones((3, 3), dtype=np.uint8)
        for component_index in range(1, rejected_count):
            component = rejected_components == component_index
            component_area = int(
                rejected_stats[component_index, cv2.CC_STAT_AREA]
            )
            component_color = np.median(image_float[component], axis=0)
            border = cv2.dilate(
                component.astype(np.uint8), neighborhood_kernel, iterations=1
            ).astype(bool) & ~component
            adjacent_labels = np.unique(accepted_np[border])
            adjacent_labels = adjacent_labels[adjacent_labels >= 0]

            best_label = -1
            best_distance = float("inf")
            for label_index in adjacent_labels:
                color = label_colors.get(int(label_index))
                if color is None:
                    continue
                color_distance = float(np.linalg.norm(component_color - color))
                if color_distance < best_distance:
                    best_distance = color_distance
                    best_label = int(label_index)

            if best_label >= 0 and best_distance <= float(max_color_distance):
                accepted_np[component] = best_label
                component_records.append(
                    {
                        "component": component_index,
                        "action": "merge_color_match",
                        "label": best_label,
                        "area": component_area,
                        "color_distance": best_distance,
                    }
                )
                continue

            component_std = float(image_float[component].std(axis=0).mean())
            if (
                component_area >= int(new_region_min_pixels)
                and component_std <= float(new_region_max_std)
            ):
                accepted_np[component] = next_label
                label_colors[next_label] = component_color
                component_records.append(
                    {
                        "component": component_index,
                        "action": "create_new_label",
                        "label": next_label,
                        "area": component_area,
                        "color_distance": best_distance,
                    }
                )
                next_label += 1
                stats["new_region_count"] += 1
                continue

            # Tiny ambiguous remnants cannot remain unlabelled. Merge them into the
            # most color-compatible spatial neighbor, even when it exceeds the gate.
            if best_label >= 0:
                accepted_np[component] = best_label
                assigned_label = best_label
                action = "merge_small_fallback"
            else:
                accepted_np[component] = filled_np[component]
                assigned_label = int(filled_np[component][0])
                action = "restore_watershed_fallback"
            component_records.append(
                {
                    "component": component_index,
                    "action": action,
                    "label": assigned_label,
                    "area": component_area,
                    "color_distance": best_distance,
                }
            )
        filled_np = accepted_np
        filled_np[valid] = labels_np[valid]
        if debug_outputs is not None:
            debug_outputs["component_records"] = component_records
    if debug_outputs is not None:
        debug_outputs["final_labels"] = filled_np.copy()

    seam = missing & (nearest_distance <= max_distance)
    new_region = missing & (filled_np >= int(labels_np.max()) + 1)
    merged = missing & ~new_region
    stats["seam_filled_pixels"] = int(seam.sum())
    stats["new_region_pixels"] = int(new_region.sum())
    stats["merged_region_pixels"] = int(merged.sum())
    stats["merged_region_count"] = max(
        int(missing_component_count - 1 - stats["new_region_count"]), 0
    )

    filled = torch.from_numpy(filled_np).to(device=labels.device, dtype=labels.dtype)
    return filled, stats


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
    cluster_min_pixels: int,
    sam_fill_max_distance: float,
    sam_new_region_max_std: float,
    sam_max_color_distance: float,
    sam_mask_generator: object | None,
    timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if source == "sam":
        if sam_mask_generator is None:
            raise ValueError("SAM region source requires a mask generator")
        device = current.device
        start_time = _timer_start(device)
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
        if timings is not None:
            _record_timing(timings, "sam_prepare_image_ms", start_time, device)
        start_time = _timer_start(device)
        sam_masks = sam_mask_generator.generate(sam_image)
        if timings is not None:
            _record_timing(timings, "sam_generate_ms", start_time, device)
            timings["sam_mask_count"] = float(len(sam_masks))
        start_time = _timer_start(device)
        labels = _sam_masks_to_labels(
            sam_masks,
            tuple(current.shape[-2:]),
            current.device,
        )
        original_sam_mask = labels >= 0
        labels, fill_stats = _fill_unassigned_labels(
            labels,
            image=sam_image,
            max_distance=sam_fill_max_distance,
            new_region_min_pixels=cluster_min_pixels,
            new_region_max_std=sam_new_region_max_std,
            max_color_distance=sam_max_color_distance,
        )
        if timings is not None:
            timings.update(
                {f"sam_{key}": float(value) for key, value in fill_stats.items()}
            )
            _record_timing(timings, "sam_labels_ms", start_time, device)
        return labels, original_sam_mask
    if source != "kmeans":
        raise ValueError(f"Unknown material region source: {source}")
    device = current.device
    start_time = _timer_start(device)
    labels = _fit_color_clusters(
        current,
        cluster_count,
        cluster_sample_pixels,
        cluster_spatial_weight,
    )
    if timings is not None:
        _record_timing(timings, "kmeans_fit_ms", start_time, device)
    if labels is None:
        return None, None
    start_time = _timer_start(device)
    labels = _smooth_cluster_labels(
        labels,
        cluster_count=int(labels.max().item()) + 1,
        kernel_size=cluster_smoothing_kernel_size,
    )
    if timings is not None:
        _record_timing(timings, "kmeans_smooth_ms", start_time, device)
    return labels, torch.ones_like(labels, dtype=torch.bool)


def _align_region_values(
    *,
    current: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
    labels: torch.Tensor | None,
    alignment_sample_mask: torch.Tensor | None,
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
    if alignment_sample_mask is not None:
        channel_valid &= alignment_sample_mask
    difference = reference_log - current_log
    if bool(channel_valid.any()):
        global_offset = difference[:, channel_valid].median(dim=1).values.clamp(
            -abs(global_max_offset), abs(global_max_offset)
        )
    else:
        global_offset = current_log.new_zeros(current_log.shape[0])
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
    labels: torch.Tensor | None, alignment_sample_mask: torch.Tensor | None,
    cluster_min_pixels: int,
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
        if alignment_sample_mask is not None:
            region_valid &= alignment_sample_mask
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
    sam_fill_max_distance: float = 5.0,
    sam_new_region_max_std: float = 0.08,
    sam_max_color_distance: float = 0.15,
    sam_mask_generator: object | None = None,
    global_max_log_offset: float = 0.25,
    cluster_max_log_offset: float = 10.0,
    planar_scale: float = 1.1,
    thickness_scale: float = 0.05,
    roughness_mode: str = "region_constant",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    timings: dict[str, float] = {}
    stats: dict[str, object] = {
        "applied": False,
        "valid_pixels": 0,
        "timings_ms": timings,
    }
    timer_device = torch.device(device)
    if state.means_world.shape[0] == 0:
        stats["reason"] = "empty_map"
        return albedo, roughness, metallic, stats

    start_time = _timer_start(timer_device)
    rendered = render_gaussian_map_association(
        state=state,
        camera=camera,
        device=device,
        backend="gsplat_2dgs",
        planar_scale=planar_scale,
        thickness_scale=thickness_scale,
    )
    _record_timing(timings, "render_map_ms", start_time, timer_device)
    start_time = _timer_start(timer_device)
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
    _record_timing(timings, "valid_mask_ms", start_time, timer_device)
    if valid_count < 512:
        stats["reason"] = "too_few_valid_pixels"
        return albedo, roughness, metallic, stats

    start_time = _timer_start(timer_device)
    labels, alignment_sample_mask = _build_region_labels(
        source=region_source,
        current=current,
        cluster_count=cluster_count,
        cluster_sample_pixels=cluster_sample_pixels,
        cluster_spatial_weight=cluster_spatial_weight,
        cluster_smoothing_kernel_size=cluster_smoothing_kernel_size,
        cluster_min_pixels=cluster_min_pixels,
        sam_fill_max_distance=sam_fill_max_distance,
        sam_new_region_max_std=sam_new_region_max_std,
        sam_max_color_distance=sam_max_color_distance,
        sam_mask_generator=sam_mask_generator,
        timings=timings,
    )
    _record_timing(timings, "build_labels_total_ms", start_time, timer_device)
    if labels is not None:
        start_time = _timer_start(timer_device)
        labels = labels.to(device=valid.device, dtype=torch.long)
        stats["albedo_cluster_labels"] = labels.detach().cpu()
        stats["albedo_cluster_valid"] = valid.detach().cpu()
        if alignment_sample_mask is not None:
            alignment_sample_mask = alignment_sample_mask.to(
                device=valid.device, dtype=torch.bool
            )
            stats["albedo_cluster_original_sam"] = alignment_sample_mask.detach().cpu()
        _record_timing(timings, "debug_label_cpu_copy_ms", start_time, timer_device)

    start_time = _timer_start(timer_device)
    aligned, global_offset, used_clusters = _align_region_values(
        current=albedo,
        reference=reference,
        valid=valid,
        labels=labels,
        alignment_sample_mask=alignment_sample_mask,
        cluster_min_pixels=cluster_min_pixels,
        global_max_offset=global_max_log_offset,
        cluster_max_offset=cluster_max_log_offset,
        strength=1.0,
    )
    _record_timing(timings, "align_albedo_ms", start_time, timer_device)
    start_time = _timer_start(timer_device)
    reference_roughness = _resize_channel(rendered["roughness"], size).to(roughness)
    roughness_valid = (
        torch.isfinite(roughness).all(dim=0)
        & torch.isfinite(reference_roughness).all(dim=0)
        & torch.isfinite(coverage)
        & (coverage >= coverage_threshold)
    )
    _record_timing(timings, "roughness_valid_ms", start_time, timer_device)
    if roughness_mode not in {"offset", "region_constant"}:
        raise ValueError(
            f"Unsupported roughness alignment mode: {roughness_mode}. "
            "Expected 'offset' or 'region_constant'."
        )
    roughness_offset = roughness.new_zeros(1)
    start_time = _timer_start(timer_device)
    if roughness_mode == "region_constant":
        aligned_roughness, roughness_clusters = _align_constant_regions(
            current=roughness,
            reference=reference_roughness,
            valid=roughness_valid,
            labels=labels,
            alignment_sample_mask=alignment_sample_mask,
            cluster_min_pixels=cluster_min_pixels,
        )
    else:
        aligned_roughness, roughness_offset, roughness_clusters = _align_region_values(
            current=roughness,
            reference=reference_roughness,
            valid=roughness_valid,
            labels=labels,
            alignment_sample_mask=alignment_sample_mask,
            cluster_min_pixels=cluster_min_pixels,
            global_max_offset=global_max_log_offset,
            cluster_max_offset=cluster_max_log_offset,
            strength=1.0,
        )
    _record_timing(timings, "align_roughness_ms", start_time, timer_device)
    if labels is not None:
        stats["roughness_cluster_labels"] = labels.detach().cpu()
        stats["roughness_cluster_valid"] = roughness_valid.detach().cpu()

    start_time = _timer_start(timer_device)
    reference_metallic = _resize_channel(rendered["metallic"], size).to(metallic)
    metallic_valid = (
        torch.isfinite(metallic).all(dim=0)
        & torch.isfinite(reference_metallic).all(dim=0)
        & torch.isfinite(coverage)
        & (coverage >= coverage_threshold)
    )
    _record_timing(timings, "metallic_valid_ms", start_time, timer_device)
    start_time = _timer_start(timer_device)
    aligned_metallic, metallic_clusters = _align_constant_regions(
        current=metallic,
        reference=reference_metallic,
        valid=metallic_valid,
        labels=labels,
        alignment_sample_mask=alignment_sample_mask,
        cluster_min_pixels=cluster_min_pixels,
    )
    _record_timing(timings, "align_metallic_ms", start_time, timer_device)
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
