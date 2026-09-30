"""Compare Pi3 cameras against image feature matches on a saved diagnostic run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def _index(path: Path) -> int:
    return int(re.search(r"_(\d+)\.jpg$", path.name).group(1))


def _project_matches(
    source: np.lib.npyio.NpzFile,
    target: np.lib.npyio.NpzFile,
    xy_source: np.ndarray,
    xy_target: np.ndarray,
    fixed_focal: float | None,
) -> np.ndarray:
    source_fx = float(fixed_focal or source["fx"])
    source_fy = float(fixed_focal or source["fy"])
    target_fx = float(fixed_focal or target["fx"])
    target_fy = float(fixed_focal or target["fy"])
    depth_map = source["depth"]
    confidence = source["confidence"]
    pixels = np.rint(xy_source).astype(int)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < depth_map.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < depth_map.shape[0])
    )
    pixels = pixels[inside]
    xy_source = xy_source[inside]
    xy_target = xy_target[inside]
    z = depth_map[pixels[:, 1], pixels[:, 0]]
    conf = confidence[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(z) & (z > 0) & np.isfinite(conf) & (conf > 0.1)
    z = z[valid]
    xy_source = xy_source[valid]
    xy_target = xy_target[valid]
    points = np.column_stack(
        (
            (xy_source[:, 0] - float(source["cx"])) / source_fx * z,
            (xy_source[:, 1] - float(source["cy"])) / source_fy * z,
            z,
        )
    )
    src_pose = source["camera_to_world"]
    tgt_pose = target["camera_to_world"]
    world = points @ src_pose[:3, :3].T + src_pose[:3, 3]
    camera = (world - tgt_pose[:3, 3]) @ tgt_pose[:3, :3]
    projected = np.column_stack(
        (
            camera[:, 0] / camera[:, 2] * target_fx + float(target["cx"]),
            camera[:, 1] / camera[:, 2] * target_fy + float(target["cy"]),
        )
    )
    error = np.linalg.norm(projected - xy_target, axis=1)
    return error[np.isfinite(error) & (camera[:, 2] > 0)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--fixed-focal-px", type=float, required=True)
    args = parser.parse_args()
    capture_dir = next(args.run_dir.glob("stream_capture_*"))
    files = sorted(capture_dir.glob("*.jpg"), key=_index)
    sift = cv2.SIFT_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    records = []
    for i in range(0, len(files) - 2, 2):
        source = np.load(args.diagnostics_dir / f"geometry_{i:06d}.npz")
        target = np.load(args.diagnostics_dir / f"geometry_{i + 2:06d}.npz")
        source_image = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        target_image = cv2.imread(str(files[i + 2]), cv2.IMREAD_GRAYSCALE)
        if source_image is None or target_image is None:
            continue
        source_h, source_w = source["depth"].shape
        target_h, target_w = target["depth"].shape
        source_image = cv2.resize(source_image, (source_w, source_h), interpolation=cv2.INTER_AREA)
        target_image = cv2.resize(target_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        kp_source, desc_source = sift.detectAndCompute(source_image, None)
        kp_target, desc_target = sift.detectAndCompute(target_image, None)
        if desc_source is None or desc_target is None:
            continue
        matched = matcher.knnMatch(desc_source, desc_target, k=2)
        good = [a for a, b in matched if a.distance < 0.75 * b.distance]
        if len(good) < 16:
            continue
        xy_source = np.float32([kp_source[m.queryIdx].pt for m in good])
        xy_target = np.float32([kp_target[m.trainIdx].pt for m in good])
        _, inliers = cv2.findFundamentalMat(
            xy_source, xy_target, cv2.FM_RANSAC, 1.5, 0.999
        )
        if inliers is None or int(inliers.sum()) < 12:
            continue
        xy_source = xy_source[inliers.ravel() > 0]
        xy_target = xy_target[inliers.ravel() > 0]
        dynamic = _project_matches(source, target, xy_source, xy_target, None)
        fixed = _project_matches(
            source, target, xy_source, xy_target, args.fixed_focal_px
        )
        if min(len(dynamic), len(fixed)) < 8:
            continue
        record = {
            "source_frame": i,
            "target_frame": i + 2,
            "feature_inliers": len(xy_source),
            "depth_matches": len(dynamic),
            "dynamic_median_px": float(np.median(dynamic)),
            "fixed_median_px": float(np.median(fixed)),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    output = args.diagnostics_dir / "feature_reprojection.json"
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} pairs to {output}")


if __name__ == "__main__":
    main()
