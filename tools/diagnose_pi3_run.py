"""Replay saved stream frames through Pi3 and measure geometry consistency."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mvinverse.gsplat_fusion.pi3_preprocess import (  # noqa: E402
    Pi3GeometryStream,
    _recover_focal_shift,
)


def _frame_index(path: Path) -> int:
    match = re.search(r"_(\d+)\.jpg$", path.name)
    if match is None:
        raise ValueError(f"Missing frame index: {path}")
    return int(match.group(1))


def _measure_observation(
    observation, fixed_focal_px: float | None = None
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    camera = observation.camera
    pose = camera.camera_to_world.numpy()
    points = (observation.points_world - pose[:3, 3]) @ pose[:3, :3]
    confidence = observation.confidence.squeeze(0).numpy()
    depth = observation.depth.squeeze(0).numpy()
    valid = (
        np.isfinite(points).all(axis=-1)
        & np.isfinite(confidence)
        & (confidence > 0.1)
        & (depth > 0)
    )
    _, shifts = _recover_focal_shift(
        torch.from_numpy(points[None].copy()),
        mask=torch.from_numpy(valid[None].copy()),
    )
    shift = float(shifts[0])
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    def projection_error(z: np.ndarray, focal_px: float) -> tuple[float, float, int]:
        usable = valid & np.isfinite(z) & (z > 1e-4)
        error = np.sqrt(
            ((points[..., 0] / z * focal_px) - (xx - camera.cx)) ** 2
            + ((points[..., 1] / z * focal_px) - (yy - camera.cy)) ** 2
        )[usable]
        if error.size == 0:
            return float("nan"), float("nan"), 0
        return float(np.median(error)), float(np.quantile(error, 0.9)), int(error.size)

    raw_median, raw_p90, _ = projection_error(depth, camera.fx)
    shifted_median, shifted_p90, shifted_count = projection_error(depth + shift, camera.fx)
    metrics = {
        "fx": float(camera.fx),
        "valid_fraction": float(valid.mean()),
        "median_depth": float(np.median(depth[valid])) if valid.any() else float("nan"),
        "z_shift": shift,
        "raw_reprojection_median_px": raw_median,
        "raw_reprojection_p90_px": raw_p90,
        "shifted_reprojection_median_px": shifted_median,
        "shifted_reprojection_p90_px": shifted_p90,
        "shifted_projectable_pixels": shifted_count,
    }
    if fixed_focal_px is not None:
        normalized_focal = 2 * fixed_focal_px / np.hypot(width, height)
        _, fixed_shifts = _recover_focal_shift(
            torch.from_numpy(points[None].copy()),
            mask=torch.from_numpy(valid[None].copy()),
            focal=torch.tensor([normalized_focal], dtype=torch.float32),
        )
        fixed_shift = float(fixed_shifts[0])
        fixed_median, fixed_p90, fixed_count = projection_error(
            depth + fixed_shift, fixed_focal_px
        )
        metrics.update(
            fixed_focal_px=fixed_focal_px,
            fixed_focal_shift=fixed_shift,
            fixed_focal_reprojection_median_px=fixed_median,
            fixed_focal_reprojection_p90_px=fixed_p90,
            fixed_focal_projectable_pixels=fixed_count,
        )
    return metrics, depth, confidence


def _save_heatmaps(output_dir: Path, index: int, depth: np.ndarray, confidence: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0) & (confidence > 0.1)
    if valid.any():
        near, far = np.quantile(depth[valid], [0.02, 0.98])
        spread = max(float(far - near), 1e-6)
        normalized = np.clip((depth - near) / spread, 0, 1)
        depth_image = np.uint8(255 * (1 - normalized))
        depth_image[~valid] = 0
        Image.fromarray(depth_image).save(output_dir / f"depth_{index:06d}.png")
    confidence_image = np.uint8(255 * np.clip(confidence, 0, 1))
    Image.fromarray(confidence_image).save(output_dir / f"confidence_{index:06d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--pi3-root", type=Path, required=True)
    parser.add_argument("--pi3-ckpt", type=Path, required=True)
    parser.add_argument("--geometry-model", choices=("pi3", "pi3x"), default="pi3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--fixed-focal-px", type=float)
    parser.add_argument("--save-geometry", action="store_true")
    parser.add_argument("--window-size", type=int, default=3)
    args = parser.parse_args()

    capture_dirs = sorted(args.run_dir.glob("stream_capture_*"))
    if len(capture_dirs) != 1:
        raise ValueError(f"Expected one capture directory, found {len(capture_dirs)}")
    capture_dir = capture_dirs[0]
    source_dir = args.image_dir or capture_dir
    files = sorted(source_dir.glob("*.jpg"), key=_frame_index)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise ValueError("No captured JPEG frames found")

    gamma_label = str(args.gamma).replace(".", "p")
    focal_label = f"_fx_{args.fixed_focal_px:g}" if args.fixed_focal_px else ""
    label = f"_{args.label}" if args.label else ""
    label += f"_window_{args.window_size}" if args.window_size != 3 else ""
    output_dir = args.run_dir / f"{args.geometry_model}_diagnostics_gamma_{gamma_label}{focal_label}{label}"
    output_dir.mkdir(exist_ok=True)
    runner = Pi3GeometryStream(
        args.pi3_root,
        ckpt=str(args.pi3_ckpt),
        device="cuda",
        alignment_mode="pointcloud",
        alignment_reference="first",
        overlap_policy="first",
        input_gamma=args.gamma,
        model_name=args.geometry_model,
    )
    records = []
    for start in range(0, len(files), 2):
        window = files[start : start + args.window_size]
        runner.process_window(source_dir, [path.name for path in window])
        for path in window[: min(2, len(window))]:
            observation, _ = runner.resolve_frame(path.stem)
            metrics, depth, confidence = _measure_observation(
                observation, fixed_focal_px=args.fixed_focal_px
            )
            index = _frame_index(path)
            metrics["frame"] = index
            records.append(metrics)
            _save_heatmaps(output_dir, index, depth, confidence)
            if args.save_geometry:
                np.savez_compressed(
                    output_dir / f"geometry_{index:06d}.npz",
                    depth=depth.astype(np.float32),
                    confidence=confidence.astype(np.float32),
                    camera_to_world=observation.camera.camera_to_world.numpy(),
                    fx=float(observation.camera.fx),
                    fy=float(observation.camera.fy),
                    cx=float(observation.camera.cx),
                    cy=float(observation.camera.cy),
                )
            print(f"[{args.geometry_model}-diagnostic] " + json.dumps(metrics), flush=True)
            runner.release_frame(path.stem)

    (output_dir / "metrics.json").write_text(
        json.dumps(records, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[{args.geometry_model}-diagnostic] saved {len(records)} frames to {output_dir}")


if __name__ == "__main__":
    main()
