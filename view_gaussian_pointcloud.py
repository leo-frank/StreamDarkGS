from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Gaussian centers from a saved gaussian_map.pt with Open3D."
    )
    parser.add_argument("--state_path", type=Path, required=True)
    parser.add_argument(
        "--color",
        choices=("albedo", "normal", "confidence", "white"),
        default="albedo",
        help="Point coloring source.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.0,
        help="Optional voxel downsampling size. Zero keeps every Gaussian center.",
    )
    parser.add_argument(
        "--point_size", type=float, default=2.0, help="Open3D point size."
    )
    parser.add_argument(
        "--show_frame", action="store_true", help="Show a world coordinate frame."
    )
    return parser.parse_args()


def load_gaussian_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if "gaussian_state" not in payload:
        raise ValueError(f"Missing gaussian_state in {path}")
    state = payload["gaussian_state"]
    if "means_world" not in state:
        raise ValueError(f"Missing means_world in {path}")
    return state


def make_colors(
    state: dict[str, torch.Tensor], color: str, count: int
) -> np.ndarray:
    if color == "albedo":
        values = state["albedo"].float().clamp(0.0, 1.0)
    elif color == "normal":
        values = state["normals_world"].float() * 0.5 + 0.5
    elif color == "confidence":
        confidence = state["confidence_sum"].float().reshape(-1, 1)
        confidence = confidence / confidence.max().clamp_min(1e-6)
        values = confidence.expand(-1, 3)
    else:
        values = torch.ones((count, 3), dtype=torch.float32)
    return values.cpu().numpy().astype(np.float64, copy=False)


def main() -> None:
    args = parse_args()
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required. Install it with: pip install open3d"
        ) from exc

    state = load_gaussian_state(args.state_path)
    positions = state["means_world"].float().numpy()
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected means_world with shape [N, 3], got {positions.shape}")
    if positions.shape[0] == 0:
        raise ValueError("The Gaussian map contains no points")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(positions)
    point_cloud.colors = o3d.utility.Vector3dVector(
        make_colors(state, args.color, positions.shape[0])
    )
    if args.voxel_size > 0.0:
        point_cloud = point_cloud.voxel_down_sample(args.voxel_size)

    geometries: list[object] = [point_cloud]
    if args.show_frame:
        extent = np.ptp(np.asarray(point_cloud.points), axis=0).max()
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=max(float(extent) * 0.1, 1e-3)
            )
        )

    print(
        f"[view] {args.state_path} points={len(point_cloud.points)} "
        f"color={args.color} voxel_size={args.voxel_size}",
        flush=True,
    )
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="Gaussian point cloud")
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    render_option = visualizer.get_render_option()
    render_option.point_size = max(float(args.point_size), 1.0)
    render_option.background_color = np.zeros(3, dtype=np.float32)
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
