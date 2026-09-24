"""Conservative adjacency-based cleanup of newly created material labels."""
from __future__ import annotations

import heapq

import numpy as np

from .material_region_policy import weak_boundary


def cleanup_small_regions(
    labels: np.ndarray,
    guide: np.ndarray,
    *,
    original_max_label: int,
    label_colors: dict[int, np.ndarray],
    min_pixels: int,
    color_tolerance: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Merge small generated regions only across weak, color-compatible borders.

    Original SAM IDs (and pixels already assigned to those IDs) never move.
    A size threshold requests examination, not compulsory deletion. Bounding
    color boxes constrain the *whole merged group*, preventing gradual color
    chaining. Only touching regions can merge; mere diagonal contact is not a
    shared boundary. Region/boundary statistics are updated after each merge.
    """
    stats = {"cleanup_merged_regions": 0, "cleanup_merged_pixels": 0}
    if min_pixels <= 0 or labels.size == 0:
        return labels, stats
    if np.any(labels < 0):
        raise ValueError("Cleanup requires fully assigned labels")
    ids, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    generated = ids > original_max_label
    if not np.any(generated & (counts < min_pixels)):
        return labels, stats
    compact = inverse.reshape(labels.shape)
    order = np.argsort(inverse.ravel(), kind="stable")
    values = guide.reshape(-1, 3)[order]
    starts = np.r_[0, np.cumsum(counts[:-1])]
    sums = np.add.reduceat(values, starts, axis=0, dtype=np.float64)
    lower = np.minimum.reduceat(values, starts, axis=0)
    upper = np.maximum.reduceat(values, starts, axis=0)
    representatives = sums / counts[:, None]
    for i in np.flatnonzero(~generated):
        representatives[i] = label_colors[int(ids[i])]

    # Build each shared border once, collecting its length, total contrast, and
    # number of strong-edge pixels. Most (>=90%) of a merge boundary must be weak.
    edge_limit = max(float(color_tolerance), 0.0) * .5
    keys, contrasts = [], []
    for a, b, ca, cb in (
        (compact[:, :-1], compact[:, 1:], guide[:, :-1], guide[:, 1:]),
        (compact[:-1], compact[1:], guide[:-1], guide[1:]),
    ):
        different = a != b
        keys.append(np.minimum(a[different], b[different]) * len(ids)
                    + np.maximum(a[different], b[different]))
        contrasts.append(np.linalg.norm(ca[different] - cb[different], axis=1))
    pairs, edge_inverse = np.unique(np.concatenate(keys), return_inverse=True)
    contrast = np.concatenate(contrasts)
    lengths = np.bincount(edge_inverse, minlength=len(pairs))
    totals = np.bincount(edge_inverse, weights=contrast, minlength=len(pairs))
    strong = np.bincount(edge_inverse, weights=contrast > edge_limit, minlength=len(pairs))
    graph: list[dict[int, tuple[int, float, int]]] = [{} for _ in ids]
    for pair, length, total, strong_count in zip(pairs, lengths, totals, strong):
        a, b = divmod(int(pair), len(ids))
        record = (int(length), float(total), int(strong_count))
        graph[a][b] = record
        graph[b][a] = record
    parent = np.arange(len(ids))
    queue = [(int(counts[i]), int(i)) for i in np.flatnonzero(generated & (counts < min_pixels))]
    heapq.heapify(queue)
    merged = np.zeros(len(ids), dtype=bool)
    initial_counts = counts.copy()
    tolerance = max(float(color_tolerance), 0.0)
    while queue:
        area, source = heapq.heappop(queue)
        if parent[source] != source or counts[source] != area or area >= min_pixels:
            continue
        best = None
        for target, (length, total, strong_count) in graph[source].items():
            if generated[target] and counts[target] < counts[source]:
                continue
            if not weak_boundary(
                int(length), int(strong_count) / max(int(length), 1),
                float(total) / max(int(length), 1), tolerance,
            ):
                continue
            if generated[target]:
                center = (sums[source] + sums[target]) / (counts[source] + counts[target])
                lo = np.minimum(lower[source], lower[target])
                hi = np.maximum(upper[source], upper[target])
            else:
                # SAM representative colors remain fixed, just as in gap fill.
                # Do not constrain/resegment an existing heterogeneous SAM mask.
                center, lo, hi = representatives[target], lower[source], upper[source]
            bound = np.linalg.norm(np.maximum(np.abs(lo - center), np.abs(hi - center)))
            if bound > tolerance:
                continue
            rank = (float(np.linalg.norm(representatives[source] - representatives[target])),
                    total / length, -int(counts[target]), target)
            if best is None or rank < best:
                best = rank
        if best is None:
            continue
        target = best[-1]
        parent[source] = target
        merged[source] = True
        stats["cleanup_merged_regions"] += 1
        counts[target] += counts[source]
        sums[target] += sums[source]
        lower[target] = np.minimum(lower[target], lower[source])
        upper[target] = np.maximum(upper[target], upper[source])
        if generated[target]:
            representatives[target] = sums[target] / counts[target]
        for neighbor, record in list(graph[source].items()):
            graph[neighbor].pop(source)
            if neighbor == target:
                continue
            old = graph[target].get(neighbor, (0, 0., 0))
            combined = tuple(a + b for a, b in zip(old, record))
            graph[target][neighbor] = combined
            graph[neighbor][target] = combined
            if generated[neighbor] and counts[neighbor] < min_pixels:
                heapq.heappush(queue, (int(counts[neighbor]), neighbor))
        graph[source].clear()
        if generated[target] and counts[target] < min_pixels:
            heapq.heappush(queue, (int(counts[target]), target))

    roots = np.arange(len(ids))
    for i in range(len(ids)):
        while parent[roots[i]] != roots[i]:
            roots[i] = parent[roots[i]]
    # Count original pixels absorbed, not intermediate group sizes repeatedly.
    stats["cleanup_merged_pixels"] = int(initial_counts[roots != np.arange(len(ids))].sum())
    if not merged.any():
        return labels, stats
    remap = ids.copy()
    next_id = original_max_label + 1
    for root in np.flatnonzero((parent == np.arange(len(ids))) & generated):
        remap[root] = next_id
        next_id += 1
    # Compact generated IDs so downstream range(max_label + 1) loops benefit too.
    return remap[roots[inverse]].reshape(labels.shape).astype(labels.dtype), stats
