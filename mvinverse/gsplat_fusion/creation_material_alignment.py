from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .gsplat_adapter import render_gaussian_map_association
from .material_region_policy import boundary_evidence, coherent_split
from .material_region_cleanup import cleanup_small_regions
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


def _coherent_color_outliers(
    mask: np.ndarray, colors: np.ndarray, outliers: np.ndarray,
    distances: np.ndarray, tolerance: float, min_pixels: int, *, force_extreme: bool = True,
) -> np.ndarray:
    """Keep color outliers that have spatial/edge evidence of a separate region.

    Weak-boundary texture may stay with its parent. Strong small patches are
    preserved irrespective of area; deviations beyond 2*tolerance always split.
    Only boundaries inside the current region count, not unrelated nearby masks.
    """
    retained = np.zeros(mask.shape, dtype=bool)
    count, components, boxes, _ = cv2.connectedComponentsWithStats(outliers.astype(np.uint8), connectivity=8)
    for index in range(1, count):
        x, y, w, h, area = boxes[index]
        x0, y0 = max(int(x) - 1, 0), max(int(y) - 1, 0)
        x1, y1 = min(int(x + w) + 1, mask.shape[1]), min(int(y + h) + 1, mask.shape[0])
        part = components[y0:y1, x0:x1] == index
        if force_extreme and float(distances[y0:y1, x0:x1][part].max()) > 2 * tolerance:
            retained[y0:y1, x0:x1][part] = True
            continue
        roi = colors[y0:y1, x0:x1]
        valid = mask[y0:y1, x0:x1]
        strong_count, boundary_count = 0, 0
        inside_colors, outside_colors = [], []
        for a, b, va, vb, ca, cb in (
            (part[:, :-1], part[:, 1:], valid[:, :-1], valid[:, 1:], roi[:, :-1], roi[:, 1:]),
            (part[:-1], part[1:], valid[:-1], valid[1:], roi[:-1], roi[1:]),
        ):
            border = (a != b) & va & vb
            if not border.any():
                continue
            left, right = ca[border], cb[border]
            count, fraction, _ = boundary_evidence(left, right, tolerance)
            boundary_count += count
            strong_count += int(round(fraction * count))
            side = a[border, None]
            inside_colors.append(np.where(side, left, right))
            outside_colors.append(np.where(side, right, left))
        strong_fraction = strong_count / max(boundary_count, 1)
        if not boundary_count or strong_fraction < (.25 if area >= min_pixels else .6):
            continue
        local_contrast = 0.0
        if boundary_count:
            local_contrast = float(np.linalg.norm(
                np.median(np.concatenate(inside_colors), axis=0)
                - np.median(np.concatenate(outside_colors), axis=0)))
        if coherent_split(area, local_contrast, strong_fraction, tolerance, min_pixels):
            retained[y0:y1, x0:x1][part] = True
    return retained


def _color_connected_parts(
    mask: np.ndarray, colors: np.ndarray, tolerance: float, *, material_min_pixels: int = 0,
):
    """Partition strict colors or a bounded hierarchy of coherent color regions.

    With material_min_pixels=0 every pixel must satisfy the original color gate.
    Material mode splits only spatially coherent changes, for at most three
    levels. Weak-edge texture is retained even beyond the strict color gate.
    Retained outliers are excluded from fitting downstream, not from correction.
    """
    y, x = np.nonzero(mask)
    pending = [(y, x, 0)] if y.size else []
    tolerance = max(float(tolerance), 0.0)
    while pending:
        y, x, depth = pending.pop()
        # Material mode has a bounded coarse hierarchy, never a per-pixel
        # color-fitting recursion. Weak texture stays with the parent region.
        if material_min_pixels > 0 and depth >= 3:
            yield y, x
            continue
        values = colors[y, x]
        center = np.median(values, axis=0)
        color_distances = np.linalg.norm(values - center, axis=1)
        compatible = color_distances <= tolerance
        if compatible.all():
            yield y, x
            continue
        if not compatible.any():
            # An even split between distinct colors can put the median in empty
            # color space. Pick an observed color to guarantee progress.
            center = values[np.argmin(color_distances)]
            color_distances = np.linalg.norm(values - center, axis=1)
            compatible = color_distances <= tolerance
        y0, x0 = int(y.min()), int(x.min())
        shape = (int(y.max()) - y0 + 1, int(x.max()) - x0 + 1)
        if material_min_pixels > 0:
            local_mask = np.zeros(shape, dtype=bool)
            local_mask[y - y0, x - x0] = True
            outliers = np.zeros(shape, dtype=bool)
            outliers[y[~compatible] - y0, x[~compatible] - x0] = True
            distances = np.zeros(shape, dtype=np.float32)
            distances[y - y0, x - x0] = color_distances
            retained = _coherent_color_outliers(
                local_mask, colors[y0:y0 + shape[0], x0:x0 + shape[1]],
                outliers, distances, tolerance, material_min_pixels, force_extreme=False,
            )
            material_compatible = ~retained[y - y0, x - x0]
            if material_compatible.all():
                # No coherent material edge: do not restart strict pixel splits.
                yield y, x
                continue
            else:
                compatible = material_compatible
        for selected in (compatible, ~compatible):
            if not selected.any():
                continue
            local = np.zeros(shape, dtype=np.uint8)
            local[y[selected] - y0, x[selected] - x0] = 1
            count, components, boxes, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
            for i in range(1, count):
                bx, by, bw, bh = boxes[i, :4]
                py, px = np.nonzero(components[by:by + bh, bx:bx + bw] == i)
                pending.append((py + by + y0, px + bx + x0, depth + 1))


def _lookup_label_colors(
    labels: np.ndarray, label_colors: dict[int, np.ndarray], dtype: np.dtype,
) -> np.ndarray:
    """Gather representative colors once, leaving unknown/negative labels zero.

    Normal SAM IDs use a dense lookup table. Compact sparse external IDs before
    lookup so a large label value cannot allocate an unnecessarily large table.
    """
    if not label_colors:
        return np.zeros((*labels.shape, 3), dtype=dtype)
    ids = np.array(sorted(label_colors), dtype=np.int64)
    colors = np.asarray([label_colors[int(i)] for i in ids], dtype=dtype)
    if ids[0] >= 0 and ids[-1] < max(4096, 4 * len(ids)):
        table = np.zeros((int(ids[-1]) + 2, 3), dtype=dtype)
        table[ids] = colors
        # The final entry is a zero sentinel, never Python's negative indexing.
        indices = np.where((labels >= 0) & (labels <= ids[-1]), labels, len(table) - 1)
    else:
        table = np.concatenate((colors, np.zeros((1, 3), dtype=dtype)))
        indices = np.searchsorted(ids, labels)
        matched = (indices < len(ids)) & (ids[np.minimum(indices, len(ids) - 1)] == labels)
        indices = np.where(matched & (labels >= 0), indices, len(ids))
    return table[indices]


def _assign_sam_seams(
    raw: np.ndarray, guide: np.ndarray, distance: np.ndarray,
    watershed_labels: np.ndarray, max_distance: float, color_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find local SAM owners for narrow missing bands, without crossing edges.

    Only original SAM pixels can donate an owner. Straight short paths must stay
    color-compatible with that local donor and cannot cross a different SAM ID.
    No newly filled pixel becomes a donor (avoids color/label chaining). Ambiguous
    proposals from materially different colors are left to the existing fill.
    """
    candidates = (raw < 0) & (distance <= max_distance) & (max_distance > 0)
    owner = np.full(raw.shape, -1, np.int32)
    ys, xs = np.nonzero(candidates)
    if not ys.size or not (raw >= 0).any():
        return candidates, owner
    height, width = raw.shape
    radius = min(int(np.ceil(max_distance)), max(height, width))
    offsets = sorted(
        (float(np.hypot(dy, dx)), dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if 0 < np.hypot(dy, dx) <= max_distance
    )
    best = np.full(ys.size, np.inf)
    second = np.full(ys.size, np.inf)
    ids = np.full(ys.size, -1, np.int32)
    best_color = np.zeros((ys.size, 3), np.float32)
    second_color = np.zeros_like(best_color)
    target_colors = guide[ys, xs]
    tolerance = max(float(color_tolerance), 0.0)
    for spatial_distance, dy, dx in offsets:
        sy, sx = ys + dy, xs + dx
        selected = np.flatnonzero((sy >= 0) & (sy < height) & (sx >= 0) & (sx < width))
        selected = selected[raw[sy[selected], sx[selected]] >= 0]
        if not selected.size:
            continue
        donor_ids = raw[sy[selected], sx[selected]]
        colors = guide[sy[selected], sx[selected]]
        color_distance = np.linalg.norm(target_colors[selected] - colors, axis=1)
        compatible = color_distance <= tolerance
        if not compatible.any():
            continue
        selected = selected[compatible]
        donor_ids, colors = donor_ids[compatible], colors[compatible]
        color_distance = color_distance[compatible]
        compatible = np.ones(selected.size, dtype=bool)
        previous = target_colors[selected]
        steps = max(abs(dy), abs(dx))
        for step in range(1, steps + 1):
            py = ys[selected] + int(round(dy * step / steps))
            px = xs[selected] + int(round(dx * step / steps))
            path_color = guide[py, px]
            path_label = raw[py, px]
            compatible &= ((path_label < 0) | (path_label == donor_ids))
            compatible &= np.linalg.norm(path_color - previous, axis=1) <= tolerance
            compatible &= np.linalg.norm(path_color - colors, axis=1) <= tolerance
            previous = path_color
            if not compatible.any():
                break
        selected, donor_ids, colors = selected[compatible], donor_ids[compatible], colors[compatible]
        score = color_distance[compatible] / max(tolerance, 1e-6) + .15 * spatial_distance / max_distance
        different = donor_ids != ids[selected]
        better = (score < best[selected]) | (
            (np.abs(score - best[selected]) <= 1e-8)
            & (donor_ids == watershed_labels[ys[selected], xs[selected]]) & different
        )
        replaced = selected[better & different]
        second[replaced] = best[replaced]
        second_color[replaced] = best_color[replaced]
        alternate = ~better & different & (score < second[selected])
        second[selected[alternate]] = score[alternate]
        second_color[selected[alternate]] = colors[alternate]
        best[selected[better]] = score[better]
        best_color[selected[better]] = colors[better]
        ids[selected[better]] = donor_ids[better]
    found = ids >= 0
    unambiguous = ~np.isfinite(second)
    compared = found & np.isfinite(second)
    unambiguous[compared] = (
        (second[compared] - best[compared] >= .05)
        | (np.linalg.norm(best_color[compared] - second_color[compared], axis=1) <= tolerance * .25)
    )
    accepted = found & unambiguous
    owner[ys[accepted], xs[accepted]] = ids[accepted]
    return candidates, owner




def _local_rejected_postprocess(
    labels: np.ndarray,
    raw_labels: np.ndarray,
    image: np.ndarray,
    rejected: np.ndarray,
    *,
    label_colors: dict[int, np.ndarray],
    next_label: int,
    max_color_distance: float,
    material_min_pixels: int = 0,
    texture_outliers: np.ndarray | None = None,
    component_data=None,
    debug_outputs=None,
) -> tuple[np.ndarray, int, int]:
    """Process rejected components in local boxes; reuse their connected components."""
    output = labels.copy()
    accepted = output.copy()
    accepted[rejected] = -1
    if component_data is None:
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            rejected.astype(np.uint8), connectivity=8
        )
    else:
        count, components, stats = component_data
    records = []
    created = 0
    merged = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    for index in range(1, count):
        x, y, w, h = stats[index, :4]
        y0, y1 = max(int(y) - 1, 0), min(int(y + h) + 1, output.shape[0])
        x0, x1 = max(int(x) - 1, 0), min(int(x + w) + 1, output.shape[1])
        component = components[y0:y1, x0:x1] == index
        component_image = image[y0:y1, x0:x1]
        values = component_image[component]
        border = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool) & ~component
        adjacent = np.unique(accepted[y0:y1, x0:x1][border])
        adjacent = adjacent[adjacent >= 0]
        best, distance = -1, float("inf")
        for label_id in adjacent.tolist():
            color = label_colors.get(int(label_id))
            if color is None:
                continue
            candidate = float(np.linalg.norm(values - color, axis=1).max())
            if candidate < distance:
                best, distance = int(label_id), candidate
        if best >= 0 and distance <= float(max_color_distance):
            accepted[y0:y1, x0:x1][component] = best
            merged += int(component.sum())
            if debug_outputs is not None:
                records.append(dict(component=index, action="merge_color_match", label=best,
                                    area=int(component.sum()), color_distance=distance))
            continue
        for ys, xs in _color_connected_parts(
            component, component_image, max_color_distance,
            material_min_pixels=material_min_pixels,
        ):
            accepted[y0 + ys, x0 + xs] = next_label
            label_colors[next_label] = np.median(component_image[ys, xs], axis=0)
            if debug_outputs is not None:
                records.append(dict(component=index, action="create_new_label", label=next_label,
                                    area=int(ys.size), color_distance=distance))
            if texture_outliers is not None and material_min_pixels > 0:
                texture_outliers[y0 + ys, x0 + xs] = (
                    np.linalg.norm(component_image[ys, xs] - label_colors[next_label], axis=1)
                    > float(max_color_distance)
                )
            next_label += 1
            created += int(ys.size)
    accepted[raw_labels >= 0] = raw_labels[raw_labels >= 0]
    if debug_outputs is not None:
        debug_outputs["component_records"] = records
    return accepted, created, merged


def _fill_unassigned_labels(
    labels: torch.Tensor,
    *,
    image: torch.Tensor | np.ndarray,
    max_distance: float,
    new_region_min_pixels: int,
    new_region_max_std: float,
    max_color_distance: float,
    debug_outputs: dict[str, object] | None = None,
    assign_seams: bool = True,
    seam_outputs: dict[str, np.ndarray] | None = None,
    cleanup_min_pixels: int = 64,
    material_split_min_pixels: int = 64,
) -> tuple[torch.Tensor, dict[str, int | float]]:
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
        if seam_outputs is not None:
            seam_outputs["candidates"] = np.zeros(tuple(labels.shape), dtype=bool)
            seam_outputs["assigned"] = np.zeros(tuple(labels.shape), dtype=bool)
        if assign_seams and debug_outputs is not None:
            debug_outputs["sam_seam_owner"] = np.full(tuple(labels.shape), -1, dtype=np.int32)
        return labels, stats
    stage_started_at = time.perf_counter()
    labels_np = labels.detach().cpu().numpy().astype(np.int32, copy=True)
    valid = labels_np >= 0
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
    # Compare material-scale colors, not individual cloth-texture pixels. This
    # bilateral guide preserves strong color edges; the albedo itself is unchanged
    # and watershed still runs on the original image.
    image_float = cv2.bilateralFilter(
        np.ascontiguousarray(image_float, dtype=np.float32),
        7, max(min(float(max_color_distance) / 3.0, 0.05), 1e-6), 3.0,
    )
    missing = ~valid
    texture_outliers = np.zeros(labels_np.shape, dtype=bool)
    stats["timing_cpu_prepare_ms"] = (
        time.perf_counter() - stage_started_at
    ) * 1000.0

    stage_started_at = time.perf_counter()
    nearest_distance = cv2.distanceTransform(
        missing.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_3,
    )
    stats["timing_distance_transform_ms"] = (
        time.perf_counter() - stage_started_at
    ) * 1000.0

    stage_started_at = time.perf_counter()
    missing_component_count, _, _, _ = cv2.connectedComponentsWithStats(
        missing.astype(np.uint8), connectivity=8
    )
    deep_core = missing & (nearest_distance > max_distance)
    component_count, components, component_stats, _ = cv2.connectedComponentsWithStats(
        deep_core.astype(np.uint8), connectivity=8
    )
    stats["timing_connected_components_ms"] = (
        time.perf_counter() - stage_started_at
    ) * 1000.0

    stage_started_at = time.perf_counter()
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
        x, y, w, h = component_stats[component_index, :4]
        core = components[y:y + h, x:x + w] == component_index
        core_image = image_float[y:y + h, x:x + w]
        core_values = core_image[core]
        core_std = float(core_values.std(axis=0).mean())
        core_center = np.median(core_values, axis=0)
        # Large gaps own their seeds even when textured. Split heterogeneous
        # cores instead of allowing unrelated SAM seeds to swallow the gap.
        parts = (
            [np.nonzero(core)]
            if core_std <= float(new_region_max_std)
            and np.linalg.norm(core_values - core_center, axis=1).max()
            <= float(max_color_distance)
            else _color_connected_parts(core, core_image, max_color_distance,
                                        material_min_pixels=material_split_min_pixels)
        )
        for ys, xs in parts:
            markers[ys + y, xs + x] = next_label + 1
            label_colors[next_label] = (
                core_center if ys.size == core_values.shape[0]
                else np.median(core_image[ys, xs], axis=0)
            )
            if material_split_min_pixels > 0:
                texture_outliers[ys + y, xs + x] = (
                    np.linalg.norm(core_image[ys, xs] - label_colors[next_label], axis=1)
                    > float(max_color_distance)
                )
            next_label += 1
            stats["new_region_count"] += 1
    if not bool((markers > 0).any()):
        # No SAM masks, or an image too small to contain a large core.
        for ys, xs in _color_connected_parts(missing, image_float, max_color_distance,
                                            material_min_pixels=material_split_min_pixels):
            markers[ys, xs] = next_label + 1
            label_colors[next_label] = np.median(image_float[ys, xs], axis=0)
            if material_split_min_pixels > 0:
                texture_outliers[ys, xs] = (
                    np.linalg.norm(image_float[ys, xs] - label_colors[next_label], axis=1)
                    > float(max_color_distance)
                )
            next_label += 1
            stats["new_region_count"] += 1
    stats["timing_region_color_stats_ms"] = (
        time.perf_counter() - stage_started_at
    ) * 1000.0

    if debug_outputs is not None:
        debug_outputs["deep_core"] = deep_core.copy()
        debug_outputs["new_region_markers"] = markers.copy() - 1

    stage_started_at = time.perf_counter()
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
    stats["timing_watershed_ms"] = (
        time.perf_counter() - stage_started_at
    ) * 1000.0
    if debug_outputs is not None:
        debug_outputs["watershed_candidates"] = filled_np.copy()

    seam_assigned = np.zeros(labels_np.shape, dtype=bool)
    if assign_seams:
        stage_started_at = time.perf_counter()
        seam_candidates, seam_owner = _assign_sam_seams(
            labels_np, image_float, nearest_distance, filled_np,
            max_distance, max_color_distance,
        )
        seam_assigned = seam_owner >= 0
        filled_np[seam_assigned] = seam_owner[seam_assigned]
        if seam_outputs is not None:
            seam_outputs.update(candidates=seam_candidates, assigned=seam_assigned)
        if debug_outputs is not None:
            debug_outputs["sam_seam_owner"] = seam_owner
        stats["seam_candidate_pixels"] = int(seam_candidates.sum())
        stats["seam_assigned_pixels"] = int(seam_assigned.sum())
        stats["timing_seam_assignment_ms"] = (time.perf_counter() - stage_started_at) * 1000

    # Watershed supplies a spatially connected candidate. Accept that candidate only
    # when its albedo is compatible with the original SAM region (or new-region core).
    stage_started_at = time.perf_counter()
    candidate_colors = _lookup_label_colors(filled_np, label_colors, image_float.dtype)
    candidate_color_distance = np.linalg.norm(
        image_float - candidate_colors, axis=2
    )
    # Locally supported seams need not match the median of a whole textured SAM
    # region. They inherit its ID but will not supply fitting samples.
    # Only retain texture for its actual seed owner; never transfer that trust
    # to a different watershed candidate. These pixels will not fit the offset.
    texture_outliers &= filled_np == (markers - 1)
    rejected = missing & ~seam_assigned & ~texture_outliers & (candidate_color_distance > float(max_color_distance))
    stats["color_rejected_pixels"] = int(rejected.sum())
    if debug_outputs is not None:
        debug_outputs["color_rejected"] = rejected.copy()
        debug_outputs["candidate_color_distance"] = candidate_color_distance.copy()

    rejected_components = None
    rejected_stats = None
    rejected_count = 0
    if bool(rejected.any()):
        accepted_np = filled_np.copy()
        accepted_np[rejected] = -1
        rejected_count, rejected_components, rejected_stats, _ = (
            cv2.connectedComponentsWithStats(
                rejected.astype(np.uint8), connectivity=8
            )
        )
    stats["timing_rejected_analysis_ms"] = (
        time.perf_counter() - stage_started_at
    ) * 1000.0

    stage_started_at = time.perf_counter()
    accepted_np = filled_np.copy()
    accepted_np[rejected] = -1
    if debug_outputs is not None and bool(rejected.any()):
        debug_outputs["color_accepted_labels"] = accepted_np.copy()
        debug_outputs["rejected_components"] = rejected_components.copy() if rejected_components is not None else np.zeros_like(rejected)
    if bool(rejected.any()):
        filled_np, created_pixels, merged_pixels = _local_rejected_postprocess(
            filled_np, labels_np, image_float, rejected,
            label_colors=label_colors, next_label=next_label,
            max_color_distance=max_color_distance,
            material_min_pixels=material_split_min_pixels,
            texture_outliers=texture_outliers,
            component_data=(rejected_count, rejected_components, rejected_stats),
            debug_outputs=debug_outputs,
        )
        stats["new_region_count"] += int(created_pixels > 0)
        stats["merged_region_pixels"] = int(merged_pixels)
    if cleanup_min_pixels > 0:
        stats["timing_component_merge_ms"] = (time.perf_counter() - stage_started_at) * 1000.0
        cleanup_start = time.perf_counter()
        if debug_outputs is not None:
            debug_outputs["before_cleanup_labels"] = filled_np.copy()
        filled_np, cleanup_stats = cleanup_small_regions(
            filled_np, image_float, original_max_label=int(labels_np.max()),
            label_colors=label_colors, min_pixels=cleanup_min_pixels,
            color_tolerance=max_color_distance,
        )
        stats.update(cleanup_stats)
        stats["timing_region_cleanup_ms"] = (time.perf_counter() - cleanup_start) * 1000.0
    if "timing_component_merge_ms" not in stats:
        stats["timing_component_merge_ms"] = (time.perf_counter() - stage_started_at) * 1000.0
    if debug_outputs is not None:
        debug_outputs["final_labels"] = filled_np.copy()
    if material_split_min_pixels > 0:
        stats["texture_excluded_pixels"] = int(texture_outliers.sum())
        if seam_outputs is not None:
            seam_outputs["texture_outliers"] = texture_outliers

    seam = missing & (nearest_distance <= max_distance)
    new_region = missing & (filled_np >= int(labels_np.max()) + 1)
    merged = missing & ~new_region
    stats["seam_filled_pixels"] = int(seam.sum())
    stats["new_region_pixels"] = int(new_region.sum())
    stats["new_region_count"] = int(np.unique(filled_np[new_region]).size)
    stats["merged_region_pixels"] = int(merged.sum())
    stats["merged_region_count"] = max(
        int(missing_component_count - 1 - stats["new_region_count"]), 0
    )

    _sync_device(labels.device)
    stage_started_at = time.perf_counter()
    filled = torch.from_numpy(filled_np).to(device=labels.device, dtype=labels.dtype)
    _sync_device(labels.device)
    stats["timing_result_to_device_ms"] = (
        time.perf_counter() - stage_started_at
    ) * 1000.0
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
    debug_outputs: dict[str, object] | None = None,
    timings: dict[str, float] | None = None,
    sam_seam_assignment: bool = True,
    seam_outputs: dict[str, np.ndarray] | None = None,
    sam_cleanup_min_pixels: int = 64,
    sam_material_split_min_pixels: int = 64,
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
            internal_timings = getattr(sam_mask_generator, "last_timings_ms", None)
            if isinstance(internal_timings, dict):
                timings.update(
                    {
                        f"sam_internal_{key}": float(value)
                        for key, value in internal_timings.items()
                    }
                )
        start_time = _timer_start(device)
        labels = _sam_masks_to_labels(
            sam_masks,
            tuple(current.shape[-2:]),
            current.device,
        )
        if debug_outputs is not None:
            debug_outputs["sam2_raw_labels"] = labels.detach().cpu().numpy()
        if timings is not None:
            _record_timing(timings, "sam_masks_to_labels_ms", start_time, device)
        original_sam_mask = labels >= 0
        start_time = _timer_start(device)
        labels, fill_stats = _fill_unassigned_labels(
            labels,
            image=sam_image,
            max_distance=sam_fill_max_distance,
            new_region_min_pixels=cluster_min_pixels,
            new_region_max_std=sam_new_region_max_std,
            max_color_distance=sam_max_color_distance,
            debug_outputs=debug_outputs,
            assign_seams=sam_seam_assignment,
            seam_outputs=seam_outputs,
            cleanup_min_pixels=sam_cleanup_min_pixels,
            material_split_min_pixels=sam_material_split_min_pixels,
        )
        if timings is not None:
            _record_timing(timings, "sam_fill_labels_ms", start_time, device)
            timings.update(
                {f"sam_{key}": float(value) for key, value in fill_stats.items()}
            )
            timings["sam_labels_ms"] = (
                timings["sam_masks_to_labels_ms"] + timings["sam_fill_labels_ms"]
            )
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


@dataclass
class _RegionAlignmentStats:
    """Per-call cache. Labels/sampling masks must not change after preparation."""

    ids: list[int]
    inverse: torch.Tensor
    counts: list[int]
    comparable_counts: list[int]
    samples: tuple[torch.Tensor, ...]
    centers: torch.Tensor
    current_log: torch.Tensor
    difference: torch.Tensor
    channel_valid: torch.Tensor
    rejected: torch.Tensor


def _prepare_region_alignment_stats(
    current, reference, valid, labels, alignment_sample_mask, comparable_mask,
    min_pixels,
) -> _RegionAlignmentStats:
    """Group fitting samples once; compute exact medians only for viable regions.

    Compact IDs also support sparse and negative labels without a huge table.
    No spatial/color segmentation decisions are made here.
    """
    current_log = current.float().clamp(1e-4, 1.0).log()
    difference = reference.float().clamp(1e-4, 1.0).log() - current_log
    finite = torch.isfinite(current).all(dim=0) & torch.isfinite(reference).all(dim=0)
    channel_valid = valid & finite
    if alignment_sample_mask is not None:
        channel_valid &= alignment_sample_mask
    comparable = (comparable_mask if comparable_mask is not None else channel_valid) & finite
    ids, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    inverse = inverse.reshape(labels.shape)
    sample_ids = inverse[channel_valid]
    counts = torch.bincount(sample_ids, minlength=ids.numel())
    comparable_counts = torch.bincount(inverse[comparable], minlength=ids.numel())
    # One host transfer for bookkeeping, instead of .item() per label per pass.
    records = torch.stack((ids, counts, comparable_counts), dim=1).cpu().tolist()
    id_list = [row[0] for row in records]
    count_list = [row[1] for row in records]
    comparable_list = [row[2] for row in records]
    order = torch.argsort(sample_ids, stable=True)
    grouped = difference[:, channel_valid][:, order]
    samples = torch.split(grouped, count_list, dim=1)
    centers = difference.new_zeros((len(records), difference.shape[0]))
    for index, (label_id, count, _) in enumerate(records):
        if label_id >= 0 and count >= max(int(min_pixels), 1):
            centers[index] = samples[index].median(dim=1).values
    return _RegionAlignmentStats(
        id_list, inverse, count_list, comparable_list, samples, centers,
        current_log, difference, channel_valid,
        torch.zeros(len(records), dtype=torch.bool, device=labels.device),
    )


def _align_region_values(
    *,
    current: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
    labels: torch.Tensor | None,
    alignment_sample_mask: torch.Tensor | None,
    comparable_mask: torch.Tensor | None,
    cluster_min_pixels: int,
    min_valid_ratio: float,
    global_max_offset: float,
    cluster_max_offset: float,
    strength: float,
    application_outputs: dict[str, torch.Tensor] | None = None,
    region_stats: _RegionAlignmentStats | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if region_stats is not None:
        cached = region_stats
        channel_valid = cached.channel_valid & valid
        if bool(channel_valid.any()):
            global_offset = cached.difference[:, channel_valid].median(dim=1).values.clamp(
                -abs(global_max_offset), abs(global_max_offset)
            )
        else:
            global_offset = cached.current_log.new_zeros(current.shape[0])
        eligible = torch.tensor([
            label_id >= 0 and count >= cluster_min_pixels
            and count / max(comparable_count, 1) >= max(float(min_valid_ratio), 0.0)
            for label_id, count, comparable_count in zip(
                cached.ids, cached.counts, cached.comparable_counts)
        ], device=current.device, dtype=torch.bool) & ~cached.rejected
        table = cached.centers.clamp(-abs(cluster_max_offset), abs(cluster_max_offset))
        table = torch.where(eligible[:, None], table, torch.zeros_like(table))
        table = table * max(float(strength), 0.0)
        correction = table[cached.inverse].permute(2, 0, 1)
        aligned = torch.exp(cached.current_log + correction).clamp(0.0, 1.0)
        if application_outputs is not None:
            application_outputs.update(
                applied_mask=eligible[cached.inverse], log_correction=correction)
        return aligned.to(current.dtype), global_offset, int(eligible.sum().item())
    current_log = torch.log(current.float().clamp(1e-4, 1.0))
    reference_log = torch.log(reference.float().clamp(1e-4, 1.0))
    channel_valid = (
        valid
        & torch.isfinite(current).all(dim=0)
        & torch.isfinite(reference).all(dim=0)
    )
    if alignment_sample_mask is not None:
        channel_valid &= alignment_sample_mask
    comparable = comparable_mask if comparable_mask is not None else channel_valid
    comparable = comparable & torch.isfinite(current).all(dim=0) & torch.isfinite(reference).all(dim=0)
    difference = reference_log - current_log
    if bool(channel_valid.any()):
        global_offset = difference[:, channel_valid].median(dim=1).values.clamp(
            -abs(global_max_offset), abs(global_max_offset)
        )
    else:
        global_offset = current_log.new_zeros(current_log.shape[0])
    correction = torch.zeros_like(current_log)
    applied_mask = torch.zeros_like(valid, dtype=torch.bool) if application_outputs is not None else None
    min_valid_ratio = max(float(min_valid_ratio), 0.0)

    used_clusters = 0
    if labels is not None:
        for cluster_index in range(int(labels.max().item()) + 1):
            cluster_all = labels == cluster_index
            cluster_valid = channel_valid & cluster_all
            comparable_count = int((comparable & cluster_all).sum().item())
            valid_count = int(cluster_valid.sum().item())
            valid_ratio = valid_count / max(comparable_count, 1)
            if valid_count < cluster_min_pixels or valid_ratio < min_valid_ratio:
                continue
            cluster_offset = difference[:, cluster_valid].median(dim=1).values.clamp(
                -abs(cluster_max_offset), abs(cluster_max_offset)
            )
            correction[:, cluster_all] = cluster_offset.view(-1, 1)
            if applied_mask is not None:
                applied_mask[cluster_all] = True
            used_clusters += 1
    elif int(channel_valid.sum().item()) >= cluster_min_pixels:
        valid_ratio = int(channel_valid.sum().item()) / max(int(comparable.sum().item()), 1)
        if valid_ratio >= min_valid_ratio:
            correction[:] = global_offset.view(-1, 1, 1)
            if applied_mask is not None:
                applied_mask[:] = True

    correction = correction.clamp(
        -abs(cluster_max_offset), abs(cluster_max_offset)
    ) * max(float(strength), 0.0)
    aligned = torch.exp(current_log + correction).clamp(0.0, 1.0)
    if application_outputs is not None:
        application_outputs.update(applied_mask=applied_mask, log_correction=correction)
    return aligned.to(current.dtype), global_offset, used_clusters


def _align_constant_regions(
    *, current: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor,
    labels: torch.Tensor | None, alignment_sample_mask: torch.Tensor | None,
    comparable_mask: torch.Tensor | None, min_valid_ratio: float,
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
        comparable = region if comparable_mask is None else (region & comparable_mask)
        valid_ratio = int(region_valid.sum().item()) / max(int(comparable.sum().item()), 1)
        if (
            int(region_valid.sum().item()) >= int(cluster_min_pixels)
            and valid_ratio >= max(float(min_valid_ratio), 0.0)
        ):
            region_value = reference[0, region_valid].float().median()
        else:
            continue
        aligned[0, region] = region_value.clamp(0.0, 1.0).to(aligned.dtype)
        aligned_regions += 1
    return aligned, aligned_regions


def _check_region_residuals(current, reference, valid, labels, min_pixels, threshold, *, log_space,
                            region_stats: _RegionAlignmentStats | None = None):
    """Gate whole regions by the 80th percentile of centered correction residuals."""
    rejected = torch.zeros_like(valid)
    records = []
    if threshold <= 0:
        return valid, rejected, records
    if region_stats is not None:
        if not log_space:
            raise ValueError("Shared region statistics require log-space residuals")
        cached = region_stats
        indices, scores = [], []
        for index, (label_id, count) in enumerate(zip(cached.ids, cached.counts)):
            if label_id < 0 or count < max(int(min_pixels), 1):
                continue
            centered = (cached.samples[index] - cached.centers[index, :, None]).abs()
            scores.append(torch.quantile(centered, 0.8, dim=1).max())
            indices.append(index)
        if scores:
            values = torch.stack(scores)
            cached.rejected[indices] = values > threshold
            for index, score in zip(indices, values.cpu().tolist()):
                records.append({"label_id": cached.ids[index], "pixels": cached.counts[index],
                                "residual_p80": score, "rejected": score > threshold})
        rejected = cached.rejected[cached.inverse]
        return valid & cached.channel_valid & ~rejected, rejected, records
    if labels is None:
        labels = torch.zeros_like(valid, dtype=torch.long)
    valid = valid & torch.isfinite(current).all(dim=0) & torch.isfinite(reference).all(dim=0)
    lhs, rhs = current.float(), reference.float()
    if log_space:
        lhs, rhs = lhs.clamp(1e-4, 1).log(), rhs.clamp(1e-4, 1).log()
    difference = rhs - lhs
    for label_id in torch.unique(labels[labels >= 0]).tolist():
        region = labels == label_id
        samples = valid & region
        count = int(samples.sum().item())
        if count < max(int(min_pixels), 1):
            continue
        values = difference[:, samples]
        center = values.median(dim=1, keepdim=True).values
        # Use the worst color channel; a coherent color/exposure shift passes.
        score = float(torch.quantile((values - center).abs(), 0.8, dim=1).max().item())
        reject = score > threshold
        if reject:
            rejected |= region
        records.append({"label_id": label_id, "pixels": count,
                        "residual_p80": score, "rejected": reject})
    return valid & ~rejected, rejected, records


def _depth_sample_mask(current, reference, coverage_valid, relative_tolerance, edge_threshold):
    """Reject depth disagreement and 3x3 neighborhoods crossing depth/visibility edges."""
    valid = coverage_valid & torch.isfinite(current) & torch.isfinite(reference)
    valid &= (current > 0) & (reference > 0)
    if relative_tolerance > 0:
        valid &= (current - reference).abs() <= relative_tolerance * current
    if edge_threshold > 0:
        for depth, support in ((current, torch.ones_like(valid)), (reference, coverage_valid)):
            finite = torch.isfinite(depth) & (depth > 0) & support
            safe = torch.where(finite, depth, torch.zeros_like(depth))[None, None]
            high = F.max_pool2d(safe, 3, 1, 1)[0, 0]
            low = -F.max_pool2d(-safe, 3, 1, 1)[0, 0]
            neighborhood = F.avg_pool2d(finite.float()[None, None], 3, 1, 1)[0, 0] >= 1.0
            valid &= neighborhood & ((high - low) <= edge_threshold * depth)
    return valid


def _region_interior(labels, sample_mask, radius):
    interior = labels >= 0
    if sample_mask is not None:
        interior &= sample_mask
    if radius <= 0:
        return interior
    kernel = 2 * radius + 1
    # Work on compact labels, avoiding float precision loss for large IDs.
    _, compact = torch.unique(labels, return_inverse=True)
    values = compact.reshape(labels.shape).float()[None, None]
    high = F.max_pool2d(values, kernel, 1, radius)[0, 0]
    low = -F.max_pool2d(-values, kernel, 1, radius)[0, 0]
    complete = F.avg_pool2d(interior.float()[None, None], kernel, 1, radius)[0, 0] >= 1.0
    return interior & complete & (high == low)


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
    min_valid_ratio: float = 0.5,
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
    current_depth: torch.Tensor | None = None,
    depth_relative_tolerance: float = 0.05,
    depth_edge_threshold: float = 0.05,
    mask_erode_radius: int = 2,
    residual_log_threshold: float = 0.30,
    residual_scalar_threshold: float = 0.20,
    debug_region_labels: bool = False,
    sam_seam_assignment: bool = True,
    sam_cleanup_min_pixels: int = 64,
    sam_material_split_min_pixels: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Align albedo only; pass roughness and metallic through unchanged.

    roughness_mode and residual_scalar_threshold remain accepted for compatibility
    with existing callers, but no scalar-material alignment is performed.
    """
    timings: dict[str, float] = {}
    stats: dict[str, object] = {
        "applied": False,
        "alignment_channels": "albedo",
        "roughness_alignment_applied": False,
        "metallic_alignment_applied": False,
        "roughness_global_offset": 0.0,
        "roughness_used_clusters": 0,
        "roughness_mean_abs_delta": 0.0,
        "metallic_used_clusters": 0,
        "metallic_mean_abs_delta": 0.0,
        "valid_pixels": 0,
        "timings_ms": timings,
    }
    if debug_region_labels:
        stats["albedo_alignment_applied"] = torch.zeros(albedo.shape[-2:], dtype=torch.bool)
        stats["albedo_correction"] = torch.zeros(albedo.shape, dtype=torch.float32)
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
    sample_valid = torch.isfinite(coverage) & (coverage >= coverage_threshold)
    stats["coverage_pixels"] = int(sample_valid.sum().item())
    if current_depth is not None:
        def depth_at_size(value):
            return F.interpolate(value.reshape(1, 1, *value.shape[-2:]).float(),
                                 size=size, mode="nearest")[0, 0].to(albedo.device)
        sample_valid &= _depth_sample_mask(
            depth_at_size(current_depth), depth_at_size(rendered["depth"]),
            sample_valid, depth_relative_tolerance, depth_edge_threshold,
        )
    stats["geometry_pixels"] = int(sample_valid.sum().item())
    current = albedo.float().clamp(1e-4, 1.0)
    reference = reference.float().clamp(1e-4, 1.0)
    valid = (
        sample_valid
        &
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
    comparable_albedo = valid.clone()

    start_time = _timer_start(timer_device)
    region_debug_outputs: dict[str, object] | None = {} if debug_region_labels else None
    seam_outputs: dict[str, np.ndarray] = {}
    labels, original_region_mask = _build_region_labels(
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
        debug_outputs=region_debug_outputs,
        timings=timings,
        sam_seam_assignment=sam_seam_assignment,
        seam_outputs=seam_outputs,
        sam_cleanup_min_pixels=sam_cleanup_min_pixels,
        sam_material_split_min_pixels=sam_material_split_min_pixels,
    )
    _record_timing(timings, "build_labels_total_ms", start_time, timer_device)
    if original_region_mask is not None:
        stats["albedo_cluster_original_sam"] = original_region_mask.detach().cpu()
    if region_debug_outputs is not None:
        stats["region_debug_outputs"] = region_debug_outputs
    if labels is not None:
        # Final labels include postprocessed regions. The original SAM mask is
        # diagnostic only; every final region may supply interior samples.
        sample_valid &= _region_interior(labels, None, max(int(mask_erode_radius), 0))
    seam_assigned = torch.zeros_like(valid)
    if "assigned" in seam_outputs:
        seam_assigned = torch.from_numpy(seam_outputs["assigned"]).to(device=valid.device)
        # Fit only on reliable interiors, but apply the fitted correction to the
        # whole region, including these locally assigned seams. Large newly
        # segmented regions remain eligible for the normal sampling checks.
        sample_valid &= ~seam_assigned
        comparable_albedo &= ~seam_assigned
    if "texture_outliers" in seam_outputs:
        texture_outliers = torch.from_numpy(seam_outputs["texture_outliers"]).to(device=valid.device)
        sample_valid &= ~texture_outliers
        comparable_albedo &= ~texture_outliers
        stats["sam_texture_excluded_pixels"] = int(texture_outliers.sum().item())
        if debug_region_labels:
            stats["sam_texture_outliers"] = texture_outliers.detach().cpu()
    valid &= sample_valid
    stats["valid_pixels"] = int(valid.sum().item())
    stats["sample_pixels"] = int(sample_valid.sum().item())
    alignment_sample_mask = sample_valid
    start_time = _timer_start(timer_device)
    region_stats = None
    if labels is not None:
        region_stats = _prepare_region_alignment_stats(
            albedo, reference, valid, labels, alignment_sample_mask,
            comparable_albedo, cluster_min_pixels,
        )
    _record_timing(timings, "region_stats_ms", start_time, timer_device)
    start_time = _timer_start(timer_device)
    valid, albedo_rejected, residual_records = _check_region_residuals(
        albedo, reference, valid, labels, cluster_min_pixels,
        residual_log_threshold, log_space=True, region_stats=region_stats,
    )
    _record_timing(timings, "albedo_residual_check_ms", start_time, timer_device)
    stats["albedo_residual_regions"] = residual_records
    stats["albedo_residual_rejected"] = sum(r["rejected"] for r in residual_records)
    stats["albedo_residual_rejected_mask"] = albedo_rejected.detach().cpu()
    stats["valid_pixels"] = int(valid.sum().item())
    if labels is not None:
        start_time = _timer_start(timer_device)
        labels = labels.to(device=valid.device, dtype=torch.long)
        stats["albedo_cluster_labels"] = labels.detach().cpu()
        stats["albedo_cluster_valid"] = valid.detach().cpu()
        if alignment_sample_mask is not None:
            alignment_sample_mask = alignment_sample_mask.to(
                device=valid.device, dtype=torch.bool
            )
        _record_timing(timings, "debug_label_cpu_copy_ms", start_time, timer_device)

    start_time = _timer_start(timer_device)
    application_outputs: dict[str, torch.Tensor] = {}
    aligned, global_offset, used_clusters = _align_region_values(
        current=albedo,
        reference=reference,
        valid=valid,
        labels=labels,
        alignment_sample_mask=alignment_sample_mask,
        comparable_mask=comparable_albedo,
        cluster_min_pixels=cluster_min_pixels,
        min_valid_ratio=min_valid_ratio,
        global_max_offset=global_max_log_offset,
        cluster_max_offset=cluster_max_log_offset,
        strength=1.0,
        application_outputs=application_outputs,
        region_stats=region_stats,
    )
    _record_timing(timings, "align_albedo_ms", start_time, timer_device)
    # Rejected regions must not inherit the global fallback correction.
    aligned[:, albedo_rejected] = albedo[:, albedo_rejected]
    applied_mask = application_outputs["applied_mask"] & ~albedo_rejected
    log_correction = application_outputs["log_correction"]
    log_correction[:, albedo_rejected] = 0
    stats["albedo_applied_pixels"] = int(applied_mask.sum().item())
    stats["sam_seam_corrected_pixels"] = int((seam_assigned & applied_mask).sum().item())
    stats["sam_seam_assigned_pixels"] = int(seam_assigned.sum().item())
    if debug_region_labels:
        stats["albedo_alignment_applied"] = applied_mask.detach().cpu()
        stats["albedo_correction"] = log_correction.detach().cpu()
        if "candidates" in seam_outputs:
            candidates = torch.from_numpy(seam_outputs["candidates"])
            stats["sam_seam_candidates"] = candidates
            stats["sam_seam_assigned"] = seam_assigned.detach().cpu()
            stats["sam_seam_uncorrected"] = candidates & ~applied_mask.detach().cpu()
    stats.update(
        {
            "applied": True,
            "global_offset": [float(value) for value in global_offset.detach().cpu()],
            "used_clusters": used_clusters,
            "mean_abs_delta": float(
                (aligned.float() - albedo.float()).abs().mean().item()
            ),
        }
    )
    return aligned, roughness, metallic, stats
