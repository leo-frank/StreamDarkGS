from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mvinverse.gsplat_fusion.creation_material_alignment import (
    align_creation_material_to_map,
)
from mvinverse.gsplat_fusion.live_material_renderer import align_normals_camera
from mvinverse.gsplat_fusion.live_pipeline_io import (
    align_rgbd_to_camera,
    list_images,
    tensor_to_bgr,
)
from mvinverse.gsplat_fusion.material_consensus import MVInverseMaterialStream
from mvinverse.gsplat_fusion.pi3_preprocess import (
    GeometryObservation,
    Pi3GeometryStream,
)
from mvinverse.gsplat_fusion.global_optimization import (
    GlobalOptimizationConfig,
    GlobalOptimizationObservation,
    optimize_gaussian_map_global,
)
from mvinverse.gsplat_fusion.gsplat_adapter import render_gaussian_map_association
from mvinverse.gsplat_fusion.rgbd import (
    GaussianMapState,
    RGBDGaussianBuilderConfig,
    RGBDGaussianFusionConfig,
    build_frame_gaussians,
    fuse_frame_gaussians,
    normals_from_depth,
)
from mvinverse.gsplat_fusion.window_schedule import build_window_schedule
from mvinverse.gsplat_fusion.types import PinholeCamera
from mvinverse.gsplat_fusion.pipeline_profiling import StageClock, PreviewSink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="First-hit Gaussian completion with Pi3 and MVInverse."
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--profile_timing", choices=("off", "wall", "sync"), default="off",
                        help="Opt-in stage timing; sync waits for CUDA at stage boundaries and affects concurrency.")
    parser.add_argument("--profile_viewer", action="store_true",
                        help="Run actual asynchronous preview rendering/JPEG encoding in replay/batch, without HTTP.")
    parser.add_argument("--serve_replay_viewer", action="store_true",
                        help="Serve replay inputs and reconstruction previews over HTTP on --stream_host/--stream_port.")
    parser.add_argument("--preview_after_online_optimization", action="store_true",
                        help="Submit the frame preview only after its online optimization completes.")
    parser.add_argument("--low_latency_pipeline", action="store_true",
                        help="Create on first prediction, preview before optimization, and overlap optimization with next-window inference.")
    parser.add_argument("--preview_noncreation_frames", action="store_true",
                        help="Render non-creation frame poses using the latest immutable map snapshot.")
    parser.add_argument("--save_viewer_dir", default="",
                        help="Optionally save the actual published Albedo/Relit JPEGs and metadata.")
    parser.add_argument("--profile_exit_after_stream", action="store_true",
                        help="Exit after stream processing/export instead of entering the interactive browser.")
    parser.add_argument(
        "--input_mode",
        choices=("batch", "replay", "stream"),
        default="batch",
        help=(
            "batch processes the discovered image sequence as before; replay "
            "feeds it incrementally; stream receives live phone-camera frames."
        ),
    )
    parser.add_argument(
        "--replay_fps",
        type=float,
        default=2.0,
        help=(
            "Simulated input rate for --input_mode replay. Use 0 to feed the "
            "sequence as fast as processing permits."
        ),
    )
    parser.add_argument("--stream_host", default="0.0.0.0")
    parser.add_argument("--stream_port", type=int, default=8765)
    parser.add_argument("--stream_capture_fps", type=float, default=2.0)
    parser.add_argument("--stream_jpeg_quality", type=int, default=85)
    parser.add_argument("--stream_queue_size", type=int, default=120)
    parser.add_argument("--stream_max_frame_mb", type=float, default=12.0)
    parser.add_argument(
        "--stream_bootstrap_frames",
        type=int,
        default=0,
        help=(
            "Run one provisional preview window after this many accepted frames; "
            "0 disables bootstrap. The formal scheduler remains unchanged."
        ),
    )
    parser.add_argument("--stream_certfile", default="")
    parser.add_argument("--stream_keyfile", default="")
    parser.add_argument("--geometry_model", choices=("pi3", "pi3x"), default="pi3",
                        help="Geometry network; pi3 retains the original behavior.")
    parser.add_argument("--pi3_root", required=True)
    parser.add_argument("--pi3_ckpt", default="")
    parser.add_argument("--pi3x_root", default="")
    parser.add_argument("--pi3x_ckpt", default="")
    parser.add_argument("--pi3x_intrinsics", choices=("none", "sidecar"), default="none",
                        help="Pi3X only: condition on per-frame JSON camera intrinsics next to each JPEG.")
    parser.add_argument("--mvinverse_ckpt", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--pose_alignment_mode",
        choices=("pointcloud", "pose_depth"),
        default="pointcloud",
    )
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--window_stride", type=int, default=2)
    parser.add_argument(
        "--pi3_alignment_reference", choices=("first", "latest"), default="first"
    )
    parser.add_argument(
        "--pi3_overlap_policy", choices=("first", "latest"), default="latest"
    )
    parser.add_argument(
        "--mvinverse_overlap_policy",
        choices=("first", "latest", "robust_consensus"),
        default="latest",
    )
    parser.add_argument(
        "--mvinverse_window_align_to_overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align each MVInverse window to cached overlap-frame material predictions.",
    )
    parser.add_argument(
        "--no_mvinverse_window_align_to_overlap",
        dest="mvinverse_window_align_to_overlap",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--mvinverse_window_align_strength", type=float, default=1.0)
    parser.add_argument(
        "--mvinverse_window_align_max_log_offset", type=float, default=0.35
    )
    parser.add_argument("--mvinverse_window_align_min_pixels", type=int, default=512)
    parser.add_argument("--pi3_device", default="cuda")
    parser.add_argument(
        "--pi3_input_gamma", type=float, default=1.0,
        help="Apply gamma to Pi3 input only; values below 1 brighten dark frames.",
    )
    parser.add_argument("--fusion_device", default="cuda")
    parser.add_argument("--mvinverse_device", default="cuda")
    parser.add_argument("--mvinverse_max_long_edge", type=int, default=512)
    parser.add_argument("--input_frame_stride", type=int, default=1)
    parser.add_argument("--fusion_frame_stride", type=int, default=5)
    parser.add_argument("--pixel_stride", type=int, default=1)
    parser.add_argument("--gaussian_scale_xy_multiplier", type=float, default=0.8)
    parser.add_argument("--pi3_min_confidence", type=float, default=0.1)
    parser.add_argument("--creation_min_confidence", type=float, default=0.0)
    parser.add_argument("--creation_max_depth_quantile", type=float, default=1.0)
    parser.add_argument("--first_hit_coverage_threshold", type=float, default=0.95)
    parser.add_argument(
        "--creation_front_depth_relative_margin",
        type=float,
        default=0.05,
        help=(
            "With sufficient map coverage, create a Gaussian only when the new "
            "depth is this fraction closer than the rendered map depth."
        ),
    )
    parser.add_argument("--creation_material_align_to_map", action="store_true")
    parser.add_argument(
        "--creation_material_align_roughness_mode",
        choices=("region_constant", "offset"),
        default="region_constant",
        help="Legacy compatibility option; ignored because only albedo is aligned.",
    )
    parser.add_argument(
        "--creation_material_align_region_source",
        choices=("kmeans", "sam"),
        default="kmeans",
    )
    parser.add_argument(
        "--sam2_ckpt",
        default="third_party/sam2/sam2.1_hiera_large.pt",
    )
    parser.add_argument(
        "--sam2_config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument("--sam2_device", default="cuda")
    parser.add_argument("--sam2_points_per_side", type=int, default=16)
    parser.add_argument("--sam2_pred_iou_thresh", type=float, default=0.8)
    parser.add_argument("--sam2_stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--sam2_mask_threshold", type=float, default=0.0)
    parser.add_argument("--sam2_min_mask_region_area", type=int, default=0)
    parser.add_argument(
        "--creation_material_align_sam_seam_assignment",
        action=argparse.BooleanOptionalAction, default=True,
        help="Assign color-compatible narrow SAM gaps to nearby regions; exclude gaps from fitting but share the region correction.",
    )
    parser.add_argument(
        "--creation_material_align_sam_cleanup_min_pixels", type=int, default=64,
        help="Examine generated regions smaller than this for color/edge-safe merging; 0 disables cleanup. Never merges original SAM labels.",
    )
    parser.add_argument(
        "--creation_material_align_sam_material_split_min_pixels", type=int, default=64,
        help="Use coarse coherent-color splitting (at most 3 levels); 0 enables legacy strict per-pixel splitting. Distinct small patches remain protected.",
    )
    parser.add_argument(
        "--creation_material_align_sam_fill_max_distance",
        type=float,
        default=5.0,
        help="Minimum distance from existing SAM labels for a gap core to seed a new region.",
    )
    parser.add_argument(
        "--creation_material_align_sam_new_region_max_std",
        type=float,
        default=0.08,
        help="Maximum mean albedo channel standard deviation for a remaining SAM gap to become a region.",
    )
    parser.add_argument(
        "--creation_material_align_sam_max_color_distance",
        type=float,
        default=0.15,
        help="Maximum normalized RGB albedo distance for accepting a watershed label.",
    )
    parser.add_argument("--creation_material_align_cluster_count", type=int, default=12)
    parser.add_argument(
        "--creation_material_align_cluster_min_pixels", type=int, default=512
    )
    parser.add_argument(
        "--creation_material_align_min_valid_ratio", type=float, default=0.5,
        help="Minimum final-valid/comparable pixel ratio required per material region.",
    )
    parser.add_argument(
        "--creation_material_align_cluster_sample_pixels", type=int, default=50000
    )
    parser.add_argument(
        "--creation_material_align_cluster_spatial_weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--creation_material_align_cluster_smoothing_kernel_size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--creation_material_align_cluster_residual_strength", type=float, default=1.0
    )
    parser.add_argument(
        "--creation_material_align_cluster_max_log_offset", type=float, default=10.0
    )
    parser.add_argument(
        "--creation_material_align_coverage_threshold", type=float, default=0.6
    )
    parser.add_argument("--creation_material_align_residual_log_threshold", type=float, default=0.30,
                        help="Maximum centered log-correction residual P80; 0 disables.")
    parser.add_argument("--creation_material_align_residual_scalar_threshold", type=float, default=0.20,
                        help="Legacy compatibility option; ignored because scalar materials are not aligned.")
    parser.add_argument("--creation_material_align_depth_relative_tolerance", type=float, default=0.05,
                        help="Maximum relative camera-Z error for alignment samples; 0 disables.")
    parser.add_argument("--creation_material_align_depth_edge_threshold", type=float, default=0.05,
                        help="Maximum relative depth range in a 3x3 neighborhood; 0 disables.")
    parser.add_argument("--creation_material_align_mask_erode_radius", type=int, default=2,
                        help="Alignment sample erosion radius in pixels; 0 disables.")
    parser.add_argument(
        "--creation_material_align_global_strength", type=float, default=1.0
    )
    parser.add_argument(
        "--creation_material_align_global_max_log_offset", type=float, default=0.25
    )
    parser.add_argument("--debug_creation_mvinverse_dir", default="")
    parser.add_argument("--debug_creation_mvinverse_video_fps", type=float, default=10.0)
    parser.add_argument(
        "--debug_optimization_inputs_dir",
        default="",
        help="Save the exact global-optimization supervision targets as PNGs.",
    )
    parser.add_argument(
        "--debug_global_optimization_dir",
        default="",
        help="Save rendered-vs-target images from global optimization steps.",
    )
    parser.add_argument(
        "--debug_global_optimization_interval",
        type=int,
        default=0,
        help="Save global-optimization render debug every N steps; 0 saves only first/last when a debug dir is set.",
    )
    parser.add_argument("--export_relit_after_fusion", action="store_true")
    parser.add_argument("--export_relit_output_dir", default="")
    parser.add_argument(
        "--export_relit_output_size_policy",
        choices=("shrink_to_input",),
        default="shrink_to_input",
    )
    parser.add_argument("--export_render_planar_scale", type=float, default=1.1)
    parser.add_argument("--export_render_thickness_scale", type=float, default=0.05)
    parser.add_argument("--relight_flash_intensity", type=float, default=12.0)
    parser.add_argument("--relight_flash_radius", type=float, default=1.5)
    parser.add_argument("--relight_flash_beam_power", type=float, default=3.0)
    parser.add_argument("--export_relit_ambient", type=float, default=0.02)
    parser.add_argument("--global_optimization", action="store_true")
    parser.add_argument("--online_global_optimization", action="store_true")
    parser.add_argument("--online_global_optimization_steps", type=int, default=50)
    parser.add_argument("--online_global_optimization_interval", type=int, default=1)
    parser.add_argument(
        "--online_global_optimization_window_multiplier",
        type=float,
        default=2.0,
        help=(
            "Online optimization window as a multiple of --window_size; "
            "the resulting number of input-frame indices is rounded up."
        ),
    )
    parser.add_argument(
        "--skip_online_inference",
        action="store_true",
        help="Load the saved pre-optimization map and observations, then run optimization only.",
    )
    parser.add_argument("--global_optimization_steps", type=int, default=1000)
    parser.add_argument(
        "--global_optimization_lr_geometry", type=float, default=1e-4
    )
    parser.add_argument("--global_optimization_lr_albedo", type=float, default=1e-2)
    parser.add_argument("--global_optimization_lr_roughness", type=float, default=1e-2)
    parser.add_argument("--global_optimization_lr_metallic", type=float, default=1e-2)
    parser.add_argument("--global_optimization_lr_opacity", type=float, default=1e-3)
    parser.add_argument(
        "--global_optimization_optimize_pose",
        action="store_true",
        default=False,
        help="Jointly optimize per-frame camera pose increments (disabled by default).",
    )
    parser.add_argument(
        "--disable_global_optimization_pose",
        dest="global_optimization_optimize_pose",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--global_optimization_lr_pose_rotation", type=float, default=1e-4)
    parser.add_argument("--global_optimization_lr_pose_translation", type=float, default=1e-4)
    parser.add_argument("--global_optimization_pose_rotation_reg", type=float, default=1e-2)
    parser.add_argument("--global_optimization_pose_translation_reg", type=float, default=1e-2)
    parser.add_argument(
        "--global_optimization_depth_weight", type=float, default=1.0
    )
    parser.add_argument(
        "--global_optimization_albedo_weight", type=float, default=1.0
    )
    parser.add_argument("--global_optimization_roughness_weight", type=float, default=1.0)
    parser.add_argument("--global_optimization_metallic_weight", type=float, default=0.5)
    parser.add_argument("--global_optimization_normal_weight", type=float, default=1.0)
    parser.add_argument("--global_optimization_alpha_weight", type=float, default=1.0)
    parser.add_argument(
        "--global_optimization_surface_normal_weight",
        type=float,
        default=0.0,
        help="Consistency weight between rendered normals and 2DGS surface normals.",
    )
    parser.add_argument(
        "--global_optimization_surface_normal_depth_edge_threshold",
        type=float,
        default=0.05,
        help="Maximum relative 3x3 rendered-depth variation used by surface-normal loss.",
    )
    parser.add_argument(
        "--global_optimization_position_reg", type=float, default=1e-2
    )
    parser.add_argument("--global_optimization_scale_reg", type=float, default=1e-2)
    parser.add_argument("--global_optimization_opacity_reg", type=float, default=1e-3)
    args = parser.parse_args()
    if args.geometry_model == "pi3" and args.pi3x_intrinsics != "none":
        parser.error("--pi3x_intrinsics requires --geometry_model pi3x")
    if args.serve_replay_viewer and args.input_mode != "replay":
        parser.error('--serve_replay_viewer requires --input_mode replay')
    if args.serve_replay_viewer and args.profile_viewer:
        parser.error('--serve_replay_viewer and --profile_viewer cannot be combined')
    if args.low_latency_pipeline and args.preview_after_online_optimization:
        parser.error('--low_latency_pipeline conflicts with --preview_after_online_optimization')
    return args


def save_state(
    state: GaussianMapState,
    output_path: Path,
    processed_names: set[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gaussian_state": {
            key: value.detach().cpu() for key, value in state.as_dict().items()
        },
        "processed_names": sorted(processed_names),
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)

    from mvinverse.gsplat_fusion.io import export_gaussian_map_state_to_go2dark_ply

    export_gaussian_map_state_to_go2dark_ply(state, output_path.with_suffix(".ply"))


def _save_creation_material_debug(
    output_dir: Path,
    stem: str,
    *,
    albedo: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.png"
    if not cv2.imwrite(str(output_path), tensor_to_bgr(albedo)):
        raise RuntimeError(f"Failed to save material debug image: {output_path}")


def _save_normal_debug(
    output_dir: Path,
    stem: str,
    normal: torch.Tensor,
    label: str,
) -> None:
    normal = F.normalize(normal.float(), dim=0, eps=1e-6)
    normal_image = normal.mul(0.5).add(0.5).clamp(0.0, 1.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_{label}_normal.png"
    if not cv2.imwrite(str(output_path), tensor_to_bgr(normal_image)):
        raise RuntimeError(f"Failed to save normal debug image: {output_path}")


def _save_map_render_debug(
    output_dir: Path,
    stem: str,
    rendered_material: torch.Tensor,
    size: tuple[int, int],
    channel: str = "albedo",
) -> None:
    rendered_material = rendered_material.float().clamp(0.0, 1.0)
    if rendered_material.shape[-2:] != size:
        rendered_material = F.interpolate(
            rendered_material.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "map_render" if channel == "albedo" else f"map_render_{channel}"
    output_path = output_dir / f"{stem}_{suffix}.png"
    if not cv2.imwrite(str(output_path), tensor_to_bgr(rendered_material)):
        raise RuntimeError(f"Failed to save map render debug image: {output_path}")


def _depth_to_color_image(depth: torch.Tensor, near: float, far: float):
    import numpy as np

    depth_array = depth.detach().cpu().float().squeeze().numpy()
    valid = np.isfinite(depth_array) & (depth_array > 1e-6)
    values = np.zeros(depth_array.shape, dtype=np.uint8)
    if valid.any() and far > near:
        normalized = np.clip((depth_array - near) / (far - near), 0.0, 1.0)
        values[valid] = np.round((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    elif valid.any():
        values[valid] = 128
    image = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def _save_material_cluster_debug(
    output_dir: Path,
    stem: str,
    channel: str,
    labels: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    palette = torch.tensor(
        [
            [0.90, 0.15, 0.15],
            [0.15, 0.80, 0.20],
            [0.15, 0.35, 0.95],
            [0.95, 0.75, 0.10],
            [0.75, 0.20, 0.85],
            [0.10, 0.80, 0.80],
            [0.95, 0.45, 0.10],
            [0.55, 0.55, 0.55],
        ],
        dtype=torch.float32,
    )
    labels = labels.long()
    valid_labels = labels >= 0
    safe_labels = labels.clamp_min(0)
    image = palette[safe_labels.remainder(palette.shape[0])].permute(2, 0, 1)
    image[:, ~valid_labels] = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = output_dir / f"{stem}_{channel}_clusters.png"
    if not cv2.imwrite(str(cluster_path), tensor_to_bgr(image)):
        raise RuntimeError(f"Failed to save material cluster debug image: {cluster_path}")

    valid_image = valid.float().expand(3, -1, -1)
    valid_path = output_dir / f"{stem}_{channel}_valid.png"
    if not cv2.imwrite(str(valid_path), tensor_to_bgr(valid_image)):
        raise RuntimeError(f"Failed to save material valid debug image: {valid_path}")


def _save_region_label_debug(
    output_dir: Path,
    stem: str,
    suffix: str,
    labels,
) -> None:
    """Save an intermediate SAM/watershed label map with invalid pixels black."""
    if labels is None:
        return
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    labels = labels.long()
    palette_size = max(64, int(labels.max().item()) + 1 if bool((labels >= 0).any()) else 64)
    palette_index = torch.arange(palette_size, dtype=torch.float32)
    palette = torch.stack(
        (
            0.15 + 0.80 * ((palette_index * 37.0) % 97.0) / 96.0,
            0.15 + 0.80 * ((palette_index * 59.0) % 89.0) / 88.0,
            0.15 + 0.80 * ((palette_index * 83.0) % 83.0) / 82.0,
        ),
        dim=1,
    )
    valid = labels >= 0
    image = palette[labels.clamp_min(0).remainder(palette.shape[0])].permute(2, 0, 1)
    image[:, ~valid] = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_{suffix}.png"
    if not cv2.imwrite(str(output_path), tensor_to_bgr(image)):
        raise RuntimeError(f"Failed to save region label debug image: {output_path}")


class FirstHitPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.fusion_device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("First-hit fusion requires CUDA and gsplat.")

        image_dir = Path(args.image_dir)
        if args.input_mode == "stream":
            image_dir = Path(args.output_path).parent / time.strftime(
                "stream_capture_%Y%m%d_%H%M%S"
            )
            image_dir.mkdir(parents=True, exist_ok=True)
            files: list[str] = []
        else:
            files = list_images(str(image_dir))[:: max(args.input_frame_stride, 1)]
        self.image_dir = str(image_dir)
        if not files and args.input_mode != "stream":
            raise ValueError(f"No images found in {self.image_dir}")
        self.files = files
        self.frame_index = {
            Path(filename).stem: index for index, filename in enumerate(self.files)
        }
        self.output_path = Path(args.output_path)
        self.camera_path = self.output_path.parent / "cameras.json"
        self.cameras = {}
        self.material_debug_dir = (
            Path(args.debug_creation_mvinverse_dir)
            if args.debug_creation_mvinverse_dir
            else None
        )
        self.material_debug_video_frames: dict[str, list[Path]] = {}
        self.material_debug_temp_frames: set[Path] = set()
        self.sam_mask_generator = None
        if args.creation_material_align_region_source == "sam" and not args.skip_online_inference:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
            from mvinverse.gsplat_fusion.timed_sam import TimedSAMMaskGenerator

            checkpoint_path = Path(args.sam2_ckpt)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(__file__).resolve().parent / checkpoint_path
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"SAM2 checkpoint does not exist: {checkpoint_path}"
                )
            sam_device = torch.device(args.sam2_device)
            sam_model = build_sam2(
                args.sam2_config,
                str(checkpoint_path),
                device=str(sam_device),
                apply_postprocessing=True,
            )
            self.sam_mask_generator = TimedSAMMaskGenerator(
                SAM2AutomaticMaskGenerator(
                    sam_model,
                    points_per_side=max(int(args.sam2_points_per_side), 1),
                    pred_iou_thresh=float(args.sam2_pred_iou_thresh),
                    stability_score_thresh=float(args.sam2_stability_score_thresh),
                    mask_threshold=float(args.sam2_mask_threshold),
                    min_mask_region_area=max(int(args.sam2_min_mask_region_area), 0),
                    output_mode="binary_mask",
                    multimask_output=True,
                )
            )
            print(
                f"[sam2] automatic mask generator loaded checkpoint={checkpoint_path} "
                f"config={args.sam2_config} points_per_side={args.sam2_points_per_side} "
                f"pred_iou={args.sam2_pred_iou_thresh} "
                f"stability={args.sam2_stability_score_thresh} "
                "postprocess=True multimask=True",
                flush=True,
            )

        if args.skip_online_inference:
            self.pi3 = None
            self.material = None
        else:
            self.pi3 = Pi3GeometryStream(
                pi3_root=(args.pi3x_root or args.pi3_root) if args.geometry_model == "pi3x" else args.pi3_root,
                ckpt=args.pi3x_ckpt if args.geometry_model == "pi3x" else args.pi3_ckpt,
                device=args.pi3_device,
                alignment_mode=args.pose_alignment_mode,
                alignment_reference=args.pi3_alignment_reference,
                overlap_policy=args.pi3_overlap_policy,
                input_gamma=args.pi3_input_gamma,
                model_name=args.geometry_model,
                intrinsics_mode=args.pi3x_intrinsics if args.geometry_model == "pi3x" else "none",
            )
            self.material = MVInverseMaterialStream(
                ckpt=args.mvinverse_ckpt,
                device=args.mvinverse_device,
                max_long_edge=args.mvinverse_max_long_edge,
                policy=args.mvinverse_overlap_policy,
                align_window_to_overlap=args.mvinverse_window_align_to_overlap,
                window_align_strength=args.mvinverse_window_align_strength,
                window_align_max_log_offset=args.mvinverse_window_align_max_log_offset,
                window_align_min_pixels=args.mvinverse_window_align_min_pixels,
            )
        self.state = GaussianMapState.empty(device=self.device)
        self.processed: set[str] = set()
        self.optimization_observations: list[GlobalOptimizationObservation] = []
        self.online_optimization_history: list[dict[str, object]] = []
        self.stream_server = None
        self.viewer_worker = None
        self._optimization_executor = None
        self._optimization_future = None
        self._optimization_stream = None
        self._last_creation_camera = None
        self._preview_snapshot = None
        self._latest_preview_camera = None

        self.profile_received_at: dict[str, float] = {}
        self.profile_first_preview_at: float | None = None

        self.builder_config = RGBDGaussianBuilderConfig(
            pixel_stride=max(args.pixel_stride, 1),
            min_depth=1e-3,
            max_depth=50.0,
            min_confidence=max(args.pi3_min_confidence, 0.0),
            scale_xy_multiplier=max(args.gaussian_scale_xy_multiplier, 1e-3),
        )
        self.fusion_config = RGBDGaussianFusionConfig(
            creation_min_confidence=max(args.creation_min_confidence, 0.0),
            creation_max_depth_quantile=min(
                max(args.creation_max_depth_quantile, 0.01), 1.0
            ),
            first_hit_coverage_threshold=max(args.first_hit_coverage_threshold, 0.0),
            front_depth_relative_margin=max(
                float(args.creation_front_depth_relative_margin), 0.0
            ),
            render_planar_scale=args.export_render_planar_scale,
            render_thickness_scale=args.export_render_thickness_scale,
        )

    def run(self) -> None:
        try:
            self._run_impl()
        finally:
            if self.viewer_worker is not None:
                self.viewer_worker.close()
                self.viewer_worker = None
            if self.stream_server is not None:
                self.stream_server.close()
                self.stream_server = None

    def _run_impl(self) -> None:
        if self.args.skip_online_inference:
            if not self.args.global_optimization:
                raise ValueError("--skip_online_inference requires --global_optimization")
            self._load_optimization_inputs()
            optimized = self.optimize_global_map()
            if not optimized:
                print(
                    "[global-opt] no optimization update; saving the loaded map unchanged",
                    flush=True,
                )
            save_state(self.state, self.output_path, self.processed)
            if len(self.cameras) == len(self.files):
                self.save_camera_manifest()
            print(
                f"[save] {self.output_path} gaussians={self.state.means_world.shape[0]}",
                flush=True,
            )
            self.state = self.state.to("cpu")
            self.export_relit(
                self.output_path,
                self._relit_output_dir("after_optimization")
                if optimized
                else self._relit_output_dir(),
            )
            return
        if self.args.serve_replay_viewer:
            from mvinverse.gsplat_fusion.async_viewer import LatestOnlyViewerWorker
            from mvinverse.gsplat_fusion.camera_stream_server import CameraStreamServer

            self.stream_server = CameraStreamServer(
                self.image_dir,
                host=self.args.stream_host,
                port=self.args.stream_port,
                queue_size=self.args.stream_queue_size,
                max_frame_bytes=int(self.args.stream_max_frame_mb * 1024 * 1024),
                capture_fps=self.args.stream_capture_fps,
                jpeg_quality=min(max(int(self.args.stream_jpeg_quality), 1), 100),
            )
            self.stream_server.start()
            self.viewer_worker = LatestOnlyViewerWorker(self._render_stream_preview_task)
            self.viewer_worker.start()
            print(f'[viewer] replay: http://{self.args.stream_host}:{self.stream_server.port}/profile?poll=50', flush=True)
        elif self.args.profile_viewer and self.args.input_mode != "stream":
            from mvinverse.gsplat_fusion.async_viewer import LatestOnlyViewerWorker
            self.stream_server = PreviewSink()
            self.viewer_worker = LatestOnlyViewerWorker(self._render_stream_preview_task)
            self.viewer_worker.start()
        processing_started_at = time.perf_counter()
        if self.args.profile_timing != "off":
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            if self.args.input_mode == "stream":
                self._run_stream()
            elif self.args.input_mode == "replay":
                self._run_replay()
            else:
                self._run_batch()
            self._finish_online_optimization()
        finally:
            self._close_online_optimization()

        if self.args.profile_viewer and self.args.input_mode != "stream":
            self.viewer_worker.close()
            self.viewer_worker = None
            self.stream_server = None
        if self.args.profile_timing != "off":
            torch.cuda.synchronize(self.device)
            first_received = min(self.profile_received_at.values()) if self.profile_received_at else None
            first_preview_ms = (
                (self.profile_first_preview_at - first_received) * 1000
                if self.profile_first_preview_at is not None and first_received is not None else -1
            )
            receive_to_done_ms = (time.perf_counter() - first_received) * 1000 if first_received is not None else -1
            print(
                f"[profile-run] processing_ms={(time.perf_counter() - processing_started_at) * 1000:.3f} "
                f"frames={len(self.files)} gaussians={self.state.means_world.shape[0]} "
                f"peak_allocated_mb={torch.cuda.max_memory_allocated(self.device) / 2**20:.1f} "
                f"peak_reserved_mb={torch.cuda.max_memory_reserved(self.device) / 2**20:.1f} "
                f"first_preview_ms={first_preview_ms:.3f} receive_to_done_ms={receive_to_done_ms:.3f}", flush=True,
            )

        self.save_camera_manifest()
        self._close_material_debug_videos()
        optimized = False
        global_optimization_steps = max(int(self.args.global_optimization_steps), 0)
        if self.args.global_optimization and global_optimization_steps == 0:
            print(
                "[global-opt] steps=0; skipping initial map, optimization inputs, "
                "Pi3 depth debug, and offline optimization",
                flush=True,
            )
            self._release_inference_models()
        elif self.args.global_optimization:
            initial_path = self.output_path.with_name(
                f"{self.output_path.stem}_before_optimization{self.output_path.suffix}"
            )
            save_state(self.state, initial_path, self.processed)
            print(f"[save] initial map {initial_path}", flush=True)
            self._save_optimization_inputs()
            self._release_inference_models()
            if self.args.export_relit_after_fusion:
                self.state = self.state.to("cpu")
                torch.cuda.empty_cache()
                self.export_relit(
                    initial_path,
                    self._relit_output_dir("before_optimization"),
                )
                self.state = self.state.to(self.device)
            self.save_pi3_depth_debug()
            optimized = self.optimize_global_map()
            if optimized:
                self.save_camera_manifest()
        if (
            self.args.global_optimization
            and global_optimization_steps > 0
            and not optimized
        ):
            print(
                "[global-opt] no optimization update; saving current online map unchanged",
                flush=True,
            )
        save_state(self.state, self.output_path, self.processed)
        print(
            f"[save] {self.output_path} gaussians={self.state.means_world.shape[0]}",
            flush=True,
        )
        if self.stream_server is not None:
            # Publish one final map preview before the potentially slow offline
            # relit export. The phone can show the result while export continues.
            self._publish_final_map_preview_once()
        self.state = self.state.to("cpu")
        self._release_inference_models()
        self.export_relit(
            self.output_path,
            self._relit_output_dir("after_optimization")
            if optimized
            else self._relit_output_dir(),
        )
        gc.collect()
        torch.cuda.empty_cache()

        if self.stream_server is not None and not self.args.profile_exit_after_stream:
            self._browse_final_map()

    def _browse_final_map(self) -> None:
        from types import SimpleNamespace
        from mvinverse.gsplat_fusion.orbit_camera import orbit_camera

        self.state = self.state.to(self.device)
        base = self.cameras[Path(self.files[-1]).stem].to(torch.device('cpu'))
        # Use the median positive depth as a robust orbit distance, ignoring outliers.
        points = self.state.means_world[::max(1, len(self.state.means_world) // 10000)].detach().cpu()
        depths = (points - base.camera_to_world[:3, 3]) @ base.camera_to_world[:3, 2]
        positive = depths[torch.isfinite(depths) & (depths > 0)]
        distance = max(float(positive.median()) if len(positive) else 1.0, 0.01)
        payload = dict(self.state.as_dict())
        payload['metallic'] = torch.zeros_like(payload['metallic'])
        preview_state = GaussianMapState.from_dict(payload)
        revision = -1
        print('[viewer] final map ready; drag on phone to explore; Ctrl+C to exit', flush=True)
        try:
            while True:
                settings = self.stream_server.viewer_settings()
                if settings['revision'] != revision:
                    camera = orbit_camera(base, distance, settings).to(self.device)
                    self._render_stream_preview_task(SimpleNamespace(payload=(preview_state, camera, settings)))
                    revision = settings['revision']
                    self.stream_server.interactive_ready = True
                time.sleep(0.03)
        except KeyboardInterrupt:
            print('[viewer] closed', flush=True)

    def _publish_final_map_preview_once(self) -> None:
        """Publish the initial final-map view before offline export starts."""
        if self.stream_server is None or self.state.means_world.shape[0] == 0:
            return
        from types import SimpleNamespace
        from mvinverse.gsplat_fusion.orbit_camera import orbit_camera

        from_camera = self.cameras[Path(self.files[-1]).stem].to(torch.device("cpu"))
        points = self.state.means_world[
            :: max(1, len(self.state.means_world) // 10000)
        ].detach().cpu()
        depths = (points - from_camera.camera_to_world[:3, 3]) @ from_camera.camera_to_world[:3, 2]
        positive = depths[torch.isfinite(depths) & (depths > 0)]
        distance = max(float(positive.median()) if len(positive) else 1.0, 0.01)
        payload = dict(self.state.as_dict())
        payload["metallic"] = torch.zeros_like(payload["metallic"])
        preview_state = GaussianMapState.from_dict(payload)
        settings = self.stream_server.viewer_settings()
        camera = orbit_camera(from_camera, distance, settings).to(self.device)
        self._render_stream_preview_task(
            SimpleNamespace(payload=(preview_state, camera, settings))
        )
        self.stream_server.interactive_ready = True
        print("[viewer] final map preview published before offline export", flush=True)

    def _run_batch(self) -> None:
        """Run the original, fully discovered image-sequence schedule."""
        schedule = build_window_schedule(
            self.files,
            window_size=self.args.window_size,
            window_stride=self.args.window_stride,
        )
        for step in schedule:
            self.process_window(step.index, step.frame_names)
            for filename in step.mature_names:
                self.finalize_frame(filename)

    def _run_replay(self) -> None:
        """Reveal files over time and execute windows as soon as they are ready."""
        fps = float(self.args.replay_fps)
        if fps < 0.0:
            raise ValueError("--replay_fps must be non-negative")

        window_size = int(self.args.window_size)
        window_stride = int(self.args.window_stride)
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if window_stride < 1 or window_stride > window_size:
            raise ValueError("window_stride must be in [1, window_size]")

        available: list[str] = []
        next_window_start = 0
        window_index = 0
        finalized: set[str] = set()
        replay_start = time.monotonic()

        for input_index, filename in enumerate(self.files):
            if fps > 0.0:
                target_time = replay_start + input_index / fps
                delay = target_time - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
            available.append(filename)
            self.profile_received_at[Path(filename).stem] = time.perf_counter()
            if self.args.serve_replay_viewer:
                source = Path(self.image_dir) / filename
                if source.suffix.lower() in {'.jpg', '.jpeg'}:
                    jpeg = source.read_bytes()
                else:
                    frame = cv2.imread(str(source))
                    if frame is None:
                        raise ValueError(f'cannot read replay frame: {source}')
                    encoded, buffer = cv2.imencode('.jpg', frame)
                    if not encoded:
                        raise ValueError(f'cannot encode replay frame: {source}')
                    jpeg = buffer.tobytes()
                self.stream_server.publish_input(frame=filename, jpeg=jpeg)
            print(
                f"[replay] frame={filename} received={len(available)}/{len(self.files)}",
                flush=True,
            )

            if next_window_start + window_size <= len(available):
                window = tuple(
                    available[next_window_start : next_window_start + window_size]
                )
                self.process_window(window_index, window)
                mature_end = min(next_window_start + window_stride, len(available))
                for mature_name in available[next_window_start:mature_end]:
                    if mature_name not in finalized:
                        self.finalize_frame(mature_name)
                        finalized.add(mature_name)
                next_window_start += window_stride
                window_index += 1

        # A final short/overlapping window gives the tail frames their latest
        # geometry/material observations, matching the batch scheduler at EOF.
        if next_window_start < len(available):
            final_window = tuple(available[next_window_start:])
            already_observed = all(
                Path(name).stem in self.pi3.geometry_observations
                for name in final_window
            )
            if not already_observed:
                self.process_window(window_index, final_window)

        for filename in available:
            if filename not in finalized:
                self.finalize_frame(filename)
                finalized.add(filename)

    def _run_stream(self) -> None:
        """Receive phone frames and feed the regular sliding-window pipeline."""
        from mvinverse.gsplat_fusion.async_viewer import LatestOnlyViewerWorker
        from mvinverse.gsplat_fusion.camera_stream_server import CameraStreamServer

        server = CameraStreamServer(
            self.image_dir,
            host=self.args.stream_host,
            port=self.args.stream_port,
            queue_size=self.args.stream_queue_size,
            max_frame_bytes=int(self.args.stream_max_frame_mb * 1024 * 1024),
            capture_fps=self.args.stream_capture_fps,
            jpeg_quality=min(max(int(self.args.stream_jpeg_quality), 1), 100),
            certfile=self.args.stream_certfile,
            keyfile=self.args.stream_keyfile,
        )
        server.start()
        self.stream_server = server
        self.viewer_worker = LatestOnlyViewerWorker(self._render_stream_preview_task)
        self.viewer_worker.start()
        display_host = self.args.stream_host
        if display_host in {"0.0.0.0", "::"}:
            display_host = "<本机局域网IP>"
        scheme = "https" if self.args.stream_certfile else "http"
        print(
            f"[stream] phone capture page: {scheme}://{display_host}:{server.port}/",
            flush=True,
        )
        print("[stream] waiting for frames; tap '结束并重建' on the phone to finish", flush=True)

        available: list[str] = []
        next_window_start = 0
        window_index = 0
        finalized: set[str] = set()
        accepted_at: dict[str, float] = {}
        input_stride = max(int(self.args.input_frame_stride), 1)
        bootstrap_frames = max(int(self.args.stream_bootstrap_frames), 0)
        if bootstrap_frames >= int(self.args.window_size):
            raise ValueError("--stream_bootstrap_frames must be below --window_size")
        bootstrap_done = bootstrap_frames == 0
        received_count = 0
        try:
            for stream_frame in server.frames():
                filename = stream_frame.filename
                self.profile_received_at[Path(filename).stem] = stream_frame.received_at
                dequeue_at = time.perf_counter()
                queue_ms = (dequeue_at - stream_frame.received_at) * 1000.0
                current_received_index = received_count
                received_count += 1
                if current_received_index % input_stride != 0:
                    print(f"[stream] frame={filename} skipped_by_input_stride", flush=True)
                    continue
                self.frame_index[Path(filename).stem] = len(self.files)
                self.files.append(filename)
                available.append(filename)
                accepted_at[filename] = dequeue_at
                print(
                    f"[stream-timing] frame={filename} accepted={len(available)} "
                    f"input_queue_ms={queue_ms:.1f} "
                    f"server_queue={server.queue_depth()}",
                    flush=True,
                )
                if not bootstrap_done and len(available) >= bootstrap_frames:
                    self._run_stream_bootstrap(tuple(available[:bootstrap_frames]))
                    bootstrap_done = True
                if next_window_start + self.args.window_size <= len(available):
                    window = tuple(
                        available[
                            next_window_start : next_window_start + self.args.window_size
                        ]
                    )
                    now = time.perf_counter()
                    waits_ms = [
                        (now - accepted_at[name]) * 1000.0 for name in window
                    ]
                    print(
                        f"[window-queue] index={window_index} "
                        f"oldest_wait_ms={max(waits_ms):.1f} "
                        f"newest_wait_ms={min(waits_ms):.1f}",
                        flush=True,
                    )
                    self.process_window(window_index, window)
                    mature_end = min(
                        next_window_start + self.args.window_stride, len(available)
                    )
                    for mature_name in available[next_window_start:mature_end]:
                        if mature_name not in finalized:
                            self.finalize_frame(mature_name)
                            finalized.add(mature_name)
                    next_window_start += self.args.window_stride
                    window_index += 1

            if not available:
                raise ValueError("Live stream finished without any accepted frames")
            if next_window_start < len(available):
                final_window = tuple(available[next_window_start:])
                already_observed = all(
                    Path(name).stem in self.pi3.geometry_observations
                    for name in final_window
                )
                if not already_observed:
                    self.process_window(window_index, final_window)
            for filename in available:
                if filename not in finalized:
                    self.finalize_frame(filename)
                    finalized.add(filename)
        finally:
            try:
                self._finish_online_optimization()
            finally:
                self.viewer_worker.close()
                self.viewer_worker = None

    def _run_stream_bootstrap(self, window: tuple[str, ...]) -> None:
        """Publish a temporary early map without changing the formal map state."""
        started_at = time.perf_counter()
        print(f"[bootstrap] frames={','.join(window)}", flush=True)
        self.process_window(-1, window)

        filename = window[0]
        stem = Path(filename).stem
        geometry, _ = self.pi3.resolve_frame(stem)
        material, _ = self.material.resolve_frame(stem)
        camera = geometry.camera
        image = dict(self.pi3.get_cached_image_tensors(self.image_dir, [filename]))[stem]
        image, depth = align_rgbd_to_camera(image, geometry.depth, camera)
        confidence = geometry.confidence
        if confidence.shape[-2:] != depth.shape[-2:]:
            confidence = F.interpolate(
                confidence.unsqueeze(0), size=depth.shape[-2:], mode="nearest"
            ).squeeze(0)
        image = image.to(self.device)
        depth = depth.to(self.device)
        confidence = confidence.to(self.device)
        depth_normal_world = normals_from_depth(depth, camera).permute(2, 0, 1).to(image)
        depth_normal_camera = torch.einsum(
            "ij,jhw->ihw", camera.world_to_camera[:3, :3].to(image), depth_normal_world
        )
        material_normal_camera = align_normals_camera(
            material["normal"].to(image), depth_normal_camera
        )
        normal_world = F.normalize(
            torch.einsum(
                "ij,jhw->ihw",
                camera.rotation_camera_to_world.to(image),
                material_normal_camera,
            ),
            dim=0,
            eps=1e-6,
        )
        frame = build_frame_gaussians(
            image=image,
            depth=depth,
            camera=camera,
            config=self.builder_config,
            confidence_image=confidence,
            albedo_image=material["albedo"].to(image),
            roughness_image=material["roughness"].to(image),
            metallic_image=material["metallic"].to(image),
            normal_world_image=normal_world,
        )
        provisional_state, stats = fuse_frame_gaussians(
            GaussianMapState.empty(device=self.device),
            frame,
            config=self.fusion_config,
            allow_create=True,
        )
        self._publish_stream_preview(camera, state=provisional_state)

        # Bootstrap predictions must not affect the formal 10/8 schedule when
        # overlap policies are set to first.
        for name in window:
            bootstrap_stem = Path(name).stem
            self.pi3.release_frame(bootstrap_stem)
            self.material.release_frame(bootstrap_stem)
        print(
            f"[bootstrap] published gaussians={provisional_state.means_world.shape[0]} "
            f"created={stats['created']} "
            f"elapsed_ms={(time.perf_counter() - started_at) * 1000.0:.1f}",
            flush=True,
        )

    def process_window(
        self,
        window_index: int,
        window: tuple[str, ...],
    ) -> None:
        window_started_at = time.perf_counter()
        clock = StageClock(self.device)
        print(f"[window] index={window_index} frames={','.join(window)}", flush=True)
        pi3_started_at = time.perf_counter()
        _, pi3_size = self.pi3.process_window(self.image_dir, list(window))
        pi3_ms = (time.perf_counter() - pi3_started_at) * 1000.0
        clock.mark("pi3")
        tensors = self.pi3.get_cached_image_tensors(self.image_dir, list(window))
        clock.mark("cached_images")
        material_started_at = time.perf_counter()
        _, material_input_size, overlap_count = self.material.process_window(
            tensors,
            target_size=pi3_size,
            output_size=pi3_size,
        )
        material_ms = (time.perf_counter() - material_started_at) * 1000.0
        clock.mark("mvinverse")
        if self.stream_server is not None:
            stem = Path(window[-1]).stem
            maps = self.material.last_window_raw_outputs[stem]
            previews = {}
            for key in ("normal", "albedo"):
                value = maps[key].detach().float()
                if key == "normal":
                    value = value * 0.5 + 0.5
                ok, encoded = cv2.imencode(".jpg", tensor_to_bgr(value.clamp(0, 1)))
                if not ok:
                    raise RuntimeError(f"Failed to encode MVInverse {key} preview")
                previews[key] = encoded.tobytes()
            self.stream_server.publish_predictions(
                frame=stem, normal_jpeg=previews["normal"], albedo_jpeg=previews["albedo"]
            )
        clock.mark("prediction_preview")
        print(
            f"[mvinverse] window={window_index} input={material_input_size} "
            f"overlap={overlap_count}",
            flush=True,
        )
        alignment_stats = self.material.last_window_alignment_stats
        if isinstance(alignment_stats, dict) and alignment_stats:
            offsets = alignment_stats.get("offsets", {})
            offset_text = ""
            if isinstance(offsets, dict) and offsets:
                parts = []
                for channel, values in offsets.items():
                    if isinstance(values, list):
                        formatted = ",".join(f"{float(value):.4f}" for value in values)
                        parts.append(f"{channel}=[{formatted}]")
                if parts:
                    offset_text = " " + " ".join(parts)
            print(
                f"[mvinverse-align] window={window_index} "
                f"applied={alignment_stats.get('applied', False)} "
                f"overlap={alignment_stats.get('overlap_frames', 0)} "
                f"valid={alignment_stats.get('valid_pixels', 0)}{offset_text}",
                flush=True,
            )
        self._save_window_material_debug_frames(window_index)
        clock.mark("debug_and_logging")
        clock.report("window", index=window_index, frames=len(window))
        total_ms = (time.perf_counter() - window_started_at) * 1000.0
        print(
            f"[window-timing] index={window_index} pi3_ms={pi3_ms:.1f} "
            f"mvinverse_ms={material_ms:.1f} total_ms={total_ms:.1f}",
            flush=True,
        )
        if self.args.low_latency_pipeline and window_index >= 0:
            self._create_first_predictions(window)

    def _create_first_predictions(self, window: tuple[str, ...]) -> None:
        for filename in window:
            stem = Path(filename).stem
            if (stem not in self.processed and
                    (self.args.preview_noncreation_frames or
                     self.frame_index[stem] % max(self.args.fusion_frame_stride, 1) == 0)):
                self.finalize_frame(filename, retain_observations=True)

    def finalize_frame(self, filename: str, *, retain_observations: bool = False) -> None:
        finalize_started_at = time.perf_counter()
        stem = Path(filename).stem
        if stem in self.processed:
            if not retain_observations:
                self.pi3.release_frame(stem)
                self.material.release_frame(stem)
            return
        creation_frame = self.frame_index[stem] % max(self.args.fusion_frame_stride, 1) == 0
        if creation_frame:
            self._wait_online_optimization()
        geometry, geometry_count = self.pi3.resolve_frame(stem)
        self.cameras[stem] = geometry.camera
        if creation_frame:
            material, material_count = self.material.resolve_frame(stem)
            self.fuse_creation_frame(
                filename,
                geometry,
                material,
                geometry_count=geometry_count,
                material_count=material_count,
            )
        else:
            self.processed.add(stem)
            if self.args.preview_noncreation_frames:
                self._publish_stream_preview(geometry.camera, reuse_snapshot=True)
            print(
                f"[finalize] {stem} creation_frame=False "
                f"geometry_observations={geometry_count}",
                flush=True,
            )
        if not retain_observations:
            self.pi3.release_frame(stem)
            self.material.release_frame(stem)
        print(
            f"[finalize-timing] frame={stem} "
            f"total_ms={(time.perf_counter() - finalize_started_at) * 1000.0:.1f}",
            flush=True,
        )

    def fuse_creation_frame(
        self,
        filename: str,
        geometry: GeometryObservation,
        material: dict[str, torch.Tensor],
        *,
        geometry_count: int,
        material_count: int,
    ) -> None:
        frame_started_at = time.perf_counter()
        clock = StageClock(self.device)
        stem = Path(filename).stem
        camera = geometry.camera
        image_tensors = dict(
            self.pi3.get_cached_image_tensors(self.image_dir, [filename])
        )
        image = image_tensors[stem]
        image, depth = align_rgbd_to_camera(image, geometry.depth, camera)
        confidence = geometry.confidence
        if confidence.shape[-2:] != depth.shape[-2:]:
            confidence = F.interpolate(
                confidence.unsqueeze(0), size=depth.shape[-2:], mode="nearest"
            ).squeeze(0)
        image = image.to(self.device)
        depth = depth.to(self.device)
        confidence = confidence.to(self.device)
        depth_normal_world = (
            normals_from_depth(depth, camera).permute(2, 0, 1).to(image)
        )
        depth_normal_camera = torch.einsum(
            "ij,jhw->ihw",
            camera.world_to_camera[:3, :3].to(image),
            depth_normal_world,
        )
        material_normal_camera = align_normals_camera(
            material["normal"].to(image),
            depth_normal_camera,
        )
        normal_world = F.normalize(
            torch.einsum(
                "ij,jhw->ihw",
                camera.rotation_camera_to_world.to(image),
                material_normal_camera,
            ),
            dim=0,
            eps=1e-6,
        )
        clock.mark("input_depth_normal_prepare")

        if self.material_debug_dir is not None:
            _save_normal_debug(
                self.material_debug_dir,
                stem,
                material["normal"].to(image),
                "mvinverse",
            )
            _save_normal_debug(
                self.material_debug_dir,
                stem,
                depth_normal_camera,
                "depth",
            )
            _save_normal_debug(
                self.material_debug_dir,
                stem,
                material_normal_camera,
                "aligned",
            )
            _save_normal_debug(
                self.material_debug_dir,
                stem,
                normal_world,
                "world",
            )
            self._save_normal_debug_frame(stem, normal_world)

        albedo = material["albedo"].to(image)
        roughness = material["roughness"].to(image)
        metallic = material["metallic"].to(image)

        if self.material_debug_dir is not None:
            if self.state.means_world.shape[0] > 0:
                rendered_map = render_gaussian_map_association(
                    state=self.state,
                    camera=camera,
                    device=str(self.device),
                    backend="gsplat_2dgs",
                    planar_scale=self.args.export_render_planar_scale,
                    thickness_scale=self.args.export_render_thickness_scale,
                )
                _save_map_render_debug(
                    self.material_debug_dir,
                    stem,
                    rendered_map["albedo"].to(image),
                    tuple(albedo.shape[-2:]),
                )
                _save_map_render_debug(
                    self.material_debug_dir,
                    stem,
                    rendered_map["roughness"].to(image),
                    tuple(roughness.shape[-2:]),
                    "roughness",
                )
                _save_map_render_debug(
                    self.material_debug_dir,
                    stem,
                    rendered_map["metallic"].to(image),
                    tuple(metallic.shape[-2:]),
                    "metallic",
                )
            self._save_material_debug_frame(stem, albedo, "before_align", "albedo")
            self._save_material_debug_frame(stem, roughness, "before_align", "roughness")
            self._save_material_debug_frame(stem, metallic, "before_align", "metallic")

        if (
            self.args.creation_material_align_to_map
            and self.state.means_world.shape[0] > 0
        ):
            albedo, roughness, metallic, stats = align_creation_material_to_map(
                state=self.state,
                camera=camera,
                albedo=albedo,
                roughness=roughness,
                metallic=metallic,
                device=str(self.device),
                coverage_threshold=self.args.creation_material_align_coverage_threshold,
                cluster_count=self.args.creation_material_align_cluster_count,
                cluster_min_pixels=self.args.creation_material_align_cluster_min_pixels,
                min_valid_ratio=self.args.creation_material_align_min_valid_ratio,
                cluster_sample_pixels=self.args.creation_material_align_cluster_sample_pixels,
                cluster_spatial_weight=self.args.creation_material_align_cluster_spatial_weight,
                cluster_smoothing_kernel_size=(
                    self.args.creation_material_align_cluster_smoothing_kernel_size
                ),
                region_source=self.args.creation_material_align_region_source,
                sam_fill_max_distance=self.args.creation_material_align_sam_fill_max_distance,
                sam_new_region_max_std=self.args.creation_material_align_sam_new_region_max_std,
                sam_max_color_distance=self.args.creation_material_align_sam_max_color_distance,
                sam_seam_assignment=self.args.creation_material_align_sam_seam_assignment,
                sam_cleanup_min_pixels=self.args.creation_material_align_sam_cleanup_min_pixels,
                sam_material_split_min_pixels=self.args.creation_material_align_sam_material_split_min_pixels,
                sam_mask_generator=self.sam_mask_generator,
                debug_region_labels=self.material_debug_dir is not None,
                current_depth=depth,
                residual_log_threshold=self.args.creation_material_align_residual_log_threshold,
                residual_scalar_threshold=self.args.creation_material_align_residual_scalar_threshold,
                depth_relative_tolerance=self.args.creation_material_align_depth_relative_tolerance,
                depth_edge_threshold=self.args.creation_material_align_depth_edge_threshold,
                mask_erode_radius=self.args.creation_material_align_mask_erode_radius,
                global_max_log_offset=(
                    self.args.creation_material_align_global_max_log_offset
                    * self.args.creation_material_align_global_strength
                ),
                cluster_max_log_offset=(
                    self.args.creation_material_align_cluster_max_log_offset
                    * self.args.creation_material_align_cluster_residual_strength
                ),
                roughness_mode=self.args.creation_material_align_roughness_mode,
                planar_scale=self.args.export_render_planar_scale,
                thickness_scale=self.args.export_render_thickness_scale,
            )
            region_debug_outputs = stats.get("region_debug_outputs")
            if self.material_debug_dir is not None and isinstance(region_debug_outputs, dict):
                _save_region_label_debug(
                    self.material_debug_dir, stem, "albedo_sam2_raw_labels",
                    region_debug_outputs.get("sam2_raw_labels"),
                )
                _save_region_label_debug(
                    self.material_debug_dir, stem, "albedo_watershed_labels",
                    region_debug_outputs.get("watershed_raw"),
                )
                _save_region_label_debug(
                    self.material_debug_dir, stem, "albedo_watershed_candidates",
                    region_debug_outputs.get("watershed_candidates"),
                )
                _save_region_label_debug(
                    self.material_debug_dir, stem, "albedo_final_labels",
                    region_debug_outputs.get("final_labels"),
                )
                _save_region_label_debug(
                    self.material_debug_dir, stem, "sam_seam_owner",
                    region_debug_outputs.get("sam_seam_owner"),
                )
                _save_region_label_debug(
                    self.material_debug_dir, stem, "albedo_before_cleanup_labels",
                    region_debug_outputs.get("before_cleanup_labels"),
                )
            print(
                f"[material-align] {stem} applied={stats['applied']} "
                f"channels=albedo roughness_alignment=off metallic_alignment=off "
                f"valid={stats['valid_pixels']} clusters={stats.get('used_clusters', 0)} "
                f"coverage_samples={stats.get('coverage_pixels', 0)} "
                f"geometry_samples={stats.get('geometry_pixels', 0)} "
                f"interior_samples={stats.get('sample_pixels', 0)} "
                f"seam_assigned={stats.get('sam_seam_assigned_pixels', 0)} "
                f"seam_corrected={stats.get('sam_seam_corrected_pixels', 0)} "
                f"albedo_residual_rejected={stats.get('albedo_residual_rejected', 0)} "
                f"roughness_delta={stats.get('roughness_mean_abs_delta', 0.0):.4f} "
                f"metallic_delta={stats.get('metallic_mean_abs_delta', 0.0):.4f}",
                flush=True,
            )
            timings = stats.get("timings_ms")
            if isinstance(timings, dict) and timings:
                timing_keys = (
                    "render_map_ms",
                    "valid_mask_ms",
                    "sam_prepare_image_ms",
                    "sam_generate_ms",
                    "sam_internal_generate_masks_ms",
                    "sam_internal_set_image_ms",
                    "sam_internal_predict_ms",
                    "sam_internal_batch_postprocess_ms",
                    "sam_internal_crop_postprocess_ms",
                    "sam_internal_encode_records_ms",
                    "sam_internal_crop_count",
                    "sam_internal_batch_count",
                    "sam_internal_predict_count",
                    "sam_labels_ms",
                    "sam_masks_to_labels_ms",
                    "sam_fill_labels_ms",
                    "sam_timing_cpu_prepare_ms",
                    "sam_timing_distance_transform_ms",
                    "sam_timing_connected_components_ms",
                    "sam_timing_region_color_stats_ms",
                    "sam_timing_watershed_ms",
                    "sam_timing_seam_assignment_ms",
                    "sam_timing_region_cleanup_ms",
                    "sam_cleanup_merged_regions",
                    "sam_cleanup_merged_pixels",
                    "sam_texture_excluded_pixels",
                    "sam_seam_candidate_pixels",
                    "sam_seam_assigned_pixels",
                    "sam_timing_rejected_analysis_ms",
                    "sam_timing_component_merge_ms",
                    "sam_timing_result_to_device_ms",
                    "sam_seam_filled_pixels",
                    "sam_new_region_pixels",
                    "sam_new_region_count",
                    "sam_merged_region_pixels",
                    "sam_merged_region_count",
                    "sam_color_rejected_pixels",
                    "build_labels_total_ms",
                    "region_stats_ms",
                    "albedo_residual_check_ms",
                    "debug_label_cpu_copy_ms",
                    "align_albedo_ms",
                    "roughness_valid_ms",
                    "align_roughness_ms",
                    "metallic_valid_ms",
                    "align_metallic_ms",
                    "sam_mask_count",
                )
                timing_text = " ".join(
                    f"{key}={float(timings[key]):.3f}"
                    for key in timing_keys
                    if key in timings
                )
                print(f"[material-align-timing] {stem} {timing_text}", flush=True)
            if self.material_debug_dir is not None:
                for name in ("sam_seam_candidates", "sam_seam_assigned",
                             "albedo_alignment_applied", "sam_seam_uncorrected", "sam_texture_outliers"):
                    mask = stats.get(name)
                    if isinstance(mask, torch.Tensor):
                        path = self.material_debug_dir / f"{stem}_{name}.png"
                        if not cv2.imwrite(str(path), tensor_to_bgr(mask.float().expand(3, -1, -1))):
                            raise RuntimeError(f"Failed to save {path}")
                correction = stats.get("albedo_correction")
                if isinstance(correction, torch.Tensor):
                    # Fixed log scale: gray = identity, lighter = positive,
                    # darker = negative. Save exact values for quantitative checks.
                    path = self.material_debug_dir / f"{stem}_albedo_correction.png"
                    if not cv2.imwrite(str(path), tensor_to_bgr((0.5 + correction / 2).clamp(0, 1))):
                        raise RuntimeError(f"Failed to save {path}")
                    np.save(self.material_debug_dir / f"{stem}_albedo_correction_log.npy",
                            correction.numpy())
                for channel in ("albedo", "roughness", "metallic"):
                    rejected = stats.get(f"{channel}_residual_rejected_mask")
                    if isinstance(rejected, torch.Tensor):
                        path = self.material_debug_dir / f"{stem}_{channel}_residual_rejected.png"
                        if not cv2.imwrite(str(path), tensor_to_bgr(rejected.float().expand(3, -1, -1))):
                            raise RuntimeError(f"Failed to save {path}")
                    cluster_labels = stats.get(f"{channel}_cluster_labels")
                    cluster_valid = stats.get(f"{channel}_cluster_valid")
                    if isinstance(cluster_labels, torch.Tensor) and isinstance(
                        cluster_valid, torch.Tensor
                    ):
                        _save_material_cluster_debug(
                            self.material_debug_dir,
                            stem,
                            channel,
                            cluster_labels,
                            cluster_valid,
                        )
                original_sam = stats.get("albedo_cluster_original_sam")
                if isinstance(original_sam, torch.Tensor):
                    original_path = self.material_debug_dir / f"{stem}_albedo_original_sam.png"
                    original_image = original_sam.float().expand(3, -1, -1)
                    if not cv2.imwrite(str(original_path), tensor_to_bgr(original_image)):
                        raise RuntimeError(f"Failed to save {original_path}")
                    alignment_samples = stats["albedo_cluster_valid"]
                    samples_path = (
                        self.material_debug_dir
                        / f"{stem}_albedo_alignment_samples.png"
                    )
                    samples_image = alignment_samples.float().expand(3, -1, -1)
                    if not cv2.imwrite(str(samples_path), tensor_to_bgr(samples_image)):
                        raise RuntimeError(f"Failed to save {samples_path}")

        if self.material_debug_dir is not None:
            self._save_material_debug_frame(stem, albedo, "aligned", "albedo")
            self._save_material_debug_frame(stem, roughness, "aligned", "roughness")
            self._save_material_debug_frame(stem, metallic, "aligned", "metallic")

        if self.args.global_optimization or self.args.online_global_optimization:
            valid_depth = (
                torch.isfinite(depth)
                & (depth >= self.builder_config.min_depth)
                & (depth <= self.builder_config.max_depth)
                & torch.isfinite(confidence)
                & (confidence >= self.builder_config.min_confidence)
            )
            self.optimization_observations.append(
                GlobalOptimizationObservation(
                    image_name=stem,
                    camera=camera.to(torch.device("cpu")),
                    depth=depth.detach().cpu(),
                    albedo=albedo.detach().cpu(),
                    roughness=roughness.detach().cpu(),
                    metallic=metallic.detach().cpu(),
                    normal_world=normal_world.detach().cpu(),
                    valid_depth=valid_depth.detach().cpu(),
                    normal_camera=material_normal_camera.detach().cpu(),
                    depth_confidence=confidence.detach().cpu(),
                )
            )

        clock.mark("material_alignment_and_observation")
        preparation_ms = (time.perf_counter() - frame_started_at) * 1000.0
        build_started_at = time.perf_counter()
        frame = build_frame_gaussians(
            image=image,
            depth=depth,
            camera=camera,
            config=self.builder_config,
            confidence_image=confidence,
            albedo_image=albedo,
            roughness_image=roughness,
            metallic_image=metallic,
            normal_world_image=normal_world,
        )
        build_ms = (time.perf_counter() - build_started_at) * 1000.0
        clock.mark("build")
        fusion_started_at = time.perf_counter()
        self.state, stats = fuse_frame_gaussians(
            self.state,
            frame,
            config=self.fusion_config,
            allow_create=True,
        )
        fusion_ms = (time.perf_counter() - fusion_started_at) * 1000.0
        clock.mark("fusion")
        self.processed.add(stem)
        print(
            f"[fusion] {stem} creation_frame=True "
            f"geometry_observations={geometry_count} "
            f"material_observations={material_count} "
            f"candidates={frame.num_confidence_kept}/{frame.num_candidates} "
            f"created={stats['created']} covered={stats['covered_creation_skipped']} "
            f"front_created={stats['front_depth_created']} "
            f"same_or_behind={stats['covered_same_or_behind_skipped']} "
            f"formal={self.state.means_world.shape[0]}",
            flush=True,
        )
        viewer_submit_started_at = time.perf_counter()
        if not self.args.preview_after_online_optimization:
            self._publish_stream_preview(camera)
        viewer_submit_ms = (time.perf_counter() - viewer_submit_started_at) * 1000.0
        clock.mark("snapshot_and_submit")
        optimization_started_at = time.perf_counter()
        if self.args.low_latency_pipeline and self.args.online_global_optimization:
            self._submit_online_optimization(stem)
        else:
            self.optimize_global_map_online(stem)
        online_optimization_ms = (
            time.perf_counter() - optimization_started_at
        ) * 1000.0
        clock.mark("online_optimization_submit" if self.args.low_latency_pipeline and self.args.online_global_optimization else "online_optimization")
        self._last_creation_camera = camera
        if self.args.preview_after_online_optimization:
            viewer_submit_started_at = time.perf_counter()
            self._publish_stream_preview(camera)
            viewer_submit_ms = (time.perf_counter() - viewer_submit_started_at) * 1000.0
            clock.mark("snapshot_and_submit_after_optimization")
        clock.report("creation", frame=stem, gaussians=self.state.means_world.shape[0])
        print(
            f"[frame-timing] frame={stem} preparation_ms={preparation_ms:.1f} "
            f"build_ms={build_ms:.1f} fusion_ms={fusion_ms:.1f} "
            f"viewer_submit_ms={viewer_submit_ms:.1f} "
            f"{'online_optimization_submit_ms' if self.args.low_latency_pipeline and self.args.online_global_optimization else 'online_optimization_ms'}={online_optimization_ms:.1f} "
            f"total_ms={(time.perf_counter() - frame_started_at) * 1000.0:.1f}",
            flush=True,
        )

    def _publish_stream_preview(
        self,
        camera: PinholeCamera,
        *,
        state: GaussianMapState | None = None,
        reuse_snapshot: bool = False,
    ) -> None:
        source_state = self._preview_snapshot if reuse_snapshot else (self.state if state is None else state)
        if (
            self.stream_server is None
            or self.viewer_worker is None
            or source_state is None
            or source_state.means_world.shape[0] == 0
        ):
            return
        snapshot_started_at = time.perf_counter()
        if reuse_snapshot:
            snapshot = source_state
        else:
            snapshot_payload = {
                key: value.detach().clone()
                for key, value in source_state.as_dict().items()
            }
            snapshot_payload["metallic"] = torch.zeros_like(snapshot_payload["metallic"])
            snapshot = GaussianMapState.from_dict(snapshot_payload)
            if self.args.preview_noncreation_frames and state is None:
                self._preview_snapshot = snapshot
        self._latest_preview_camera = camera
        version, dropped = self.viewer_worker.submit(
            (snapshot, camera.to(self.device), None,
             {"frame": camera.image_name, "received_at": self.profile_received_at.get(camera.image_name)})
        )
        print(
            f"[viewer-submit] version={version} "
            f"frame={camera.image_name} reused_snapshot={reuse_snapshot} "
            f"snapshot_ms={(time.perf_counter() - snapshot_started_at) * 1000.0:.1f} "
            f"dropped_stale={dropped}",
            flush=True,
        )

    def _render_stream_preview_task(self, task) -> None:
        from export_gaussian_map_relit_views import _render_relit_view

        state, camera = task.payload[:2]
        if self.stream_server is None:
            return
        clock = StageClock(self.device)
        settings = (task.payload[2] if len(task.payload) > 2 else None) or self.stream_server.viewer_settings()
        light_dir = torch.tensor(
            [settings["light_x"], settings["light_y"], -1.0], dtype=torch.float32
        )
        preview_point_lights = [
            (
                torch.tensor([-0.65, -0.75, 0.15], dtype=torch.float32),
                torch.ones(3, dtype=torch.float32),
                3.2,
                1.0,
            ),
            (
                torch.tensor([0.75, -0.35, 0.35], dtype=torch.float32),
                torch.ones(3, dtype=torch.float32),
                1.8,
                1.3,
            ),
            (
                torch.tensor([0.0, -1.0, 0.85], dtype=torch.float32),
                torch.ones(3, dtype=torch.float32),
                2.2,
                0.9,
            ),
        ]
        with torch.no_grad():
            outputs = _render_relit_view(
                state,
                camera,
                self.device,
                "gsplat_2dgs",
                None,
                "multi_point",
                light_dir,
                torch.ones(3, dtype=torch.float32),
                self.args.relight_flash_intensity,
                self.args.relight_flash_radius,
                self.args.relight_flash_beam_power,
                True,
                1.0,
                1.0,
                False,
                0.0,
                self.args.export_render_planar_scale,
                self.args.export_render_thickness_scale,
                relight_point_lights=preview_point_lights,
                relight_light_energy_scale=0.2,
                relight_model="mvinverse_diffuse",
                relight_output_encoding="linear",
            )
        clock.mark("render_and_relight")
        cpu_images = {key: tensor_to_bgr(outputs[key]) for key in ("albedo", "relit")}
        clock.mark("device_to_cpu_and_image_convert")
        previews: dict[str, bytes] = {}
        for key in ("albedo", "relit"):
            ok, encoded = cv2.imencode(".jpg", cpu_images[key])
            if ok:
                previews[key] = encoded.tobytes()
        clock.mark("jpeg_encode")
        metadata = task.payload[3] if len(task.payload) > 3 else {}
        if len(previews) == 2:
            self.stream_server.publish_previews(
                albedo_jpeg=previews["albedo"],
                relit_jpeg=previews["relit"],
                metadata=metadata,
            )
            if self.profile_first_preview_at is None:
                self.profile_first_preview_at = time.perf_counter()
        clock.mark("publish")
        source_time = metadata.get("received_at")
        clock.report("viewer", frame=camera.image_name, version=getattr(task, "version", -1),
                     source_age_ms=f"{(time.perf_counter() - source_time) * 1000:.3f}" if source_time else "na",
                     gaussians=state.means_world.shape[0], width=camera.width, height=camera.height)
        if getattr(self.args, "save_viewer_dir", "") and len(previews) == 2:
            from mvinverse.gsplat_fusion.preview_recording import save_preview
            save_started = time.perf_counter()
            save_preview(self.args.save_viewer_dir, camera.image_name,
                         getattr(task, "version", -1), previews, metadata)
            print(f"[viewer-save] frame={camera.image_name} save_ms={(time.perf_counter()-save_started)*1000:.3f}", flush=True)

    def _save_material_debug_frame(
        self,
        stem: str,
        albedo: torch.Tensor,
        label: str,
        channel: str = "albedo",
    ) -> None:
        if self.material_debug_dir is None:
            return
        image_stem = stem if label == "aligned" else f"{stem}_before_align"
        if channel != "albedo":
            image_stem = f"{image_stem}_{channel}"
        _save_creation_material_debug(
            self.material_debug_dir,
            image_stem,
            albedo=albedo,
        )
        video_label = label if channel == "albedo" else f"{channel}_{label}"
        self.material_debug_video_frames.setdefault(video_label, []).append(
            self.material_debug_dir / f"{image_stem}.png"
        )

    def _save_window_material_debug_frames(self, window_index: int) -> None:
        if self.material_debug_dir is None or self.material is None:
            return
        debug_sets = (
            ("window_before_align", self.material.last_window_raw_outputs),
            ("window_aligned", self.material.last_window_aligned_outputs),
        )
        for label, outputs in debug_sets:
            for stem, maps in outputs.items():
                albedo = maps.get("albedo")
                if albedo is None:
                    continue
                image_stem = f"window_{window_index:04d}_{stem}_{label}"
                temp_dir = Path(
                    tempfile.mkdtemp(
                        prefix="mvinverse_window_material_",
                        dir="/tmp",
                    )
                )
                frame_path = temp_dir / f"{image_stem}.png"
                _save_creation_material_debug(
                    temp_dir,
                    frame_path.stem,
                    albedo=albedo,
                )
                self.material_debug_temp_frames.add(frame_path)
                self.material_debug_video_frames.setdefault(label, []).append(
                    frame_path
                )

    def _save_normal_debug_frame(self, stem: str, normal_world: torch.Tensor) -> None:
        if self.material_debug_dir is None:
            return
        output_path = self.material_debug_dir / f"{stem}_world_normal.png"
        self.material_debug_video_frames.setdefault("normal_world", []).append(
            output_path
        )

    def _close_material_debug_videos(self) -> None:
        if self.material_debug_dir is None:
            return
        for label, frame_paths in self.material_debug_video_frames.items():
            if not frame_paths:
                continue
            first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
            if first_frame is None:
                raise RuntimeError(f"Failed to read material debug frame: {frame_paths[0]}")
            first_height, first_width = first_frame.shape[:2]
            width = first_width + first_width % 2
            height = first_height + first_height % 2
            video_prefix = "normal" if label.startswith("normal_") else "albedo"
            video_label = label.removeprefix("normal_")
            video_path = self.material_debug_dir / f"{video_prefix}_{video_label}.mp4"
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pixel_format", "bgr24",
                "-video_size", f"{width}x{height}",
                "-framerate", f"{max(float(self.args.debug_creation_mvinverse_video_fps), 0.1):g}",
                "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(video_path),
            ]
            try:
                result = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                assert result.stdin is not None
                for frame_path in frame_paths:
                    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise RuntimeError(f"Failed to read material debug frame: {frame_path}")
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    result.stdin.write(frame.tobytes())
                result.stdin.close()
                error = result.stderr.read().decode("utf-8", errors="replace") if result.stderr else ""
                if result.wait() != 0:
                    raise RuntimeError(f"FFmpeg failed to encode {video_path}: {error.strip()}")
                print(
                    f"[material-debug-video] saved {video_path} "
                    f"frames={len(frame_paths)} fps={self.args.debug_creation_mvinverse_video_fps:g}",
                    flush=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("FFmpeg is required for material debug videos") from exc
        for frame_path in self.material_debug_temp_frames:
            try:
                frame_path.unlink(missing_ok=True)
                frame_path.parent.rmdir()
            except OSError:
                pass
        self.material_debug_temp_frames.clear()
        self.material_debug_video_frames.clear()

    def _optimization_observations_path(self) -> Path:
        return self.output_path.with_name(
            f"{self.output_path.stem}_optimization_observations.pt"
        )

    def _save_optimization_inputs(self) -> None:
        payload = {
            "observations": [
                {
                    "image_name": observation.image_name,
                    "camera": {
                        "image_name": observation.camera.image_name,
                        "width": observation.camera.width,
                        "height": observation.camera.height,
                        "fx": observation.camera.fx,
                        "fy": observation.camera.fy,
                        "cx": observation.camera.cx,
                        "cy": observation.camera.cy,
                        "camera_to_world": observation.camera.camera_to_world,
                    },
                    "depth": observation.depth,
                    "albedo": observation.albedo,
                    "roughness": observation.roughness,
                    "metallic": observation.metallic,
                    "normal_world": observation.normal_world,
                    "normal_camera": observation.normal_camera,
                    "depth_confidence": observation.depth_confidence,
                    "valid_depth": observation.valid_depth,
                }
                for observation in self.optimization_observations
            ]
        }
        path = self._optimization_observations_path()
        torch.save(payload, path)
        print(f"[save] optimization inputs {path}", flush=True)
        self._save_optimization_input_debug_images()

    def _save_optimization_input_debug_images(self) -> None:
        if not self.args.debug_optimization_inputs_dir:
            return
        if not self.optimization_observations:
            return

        output_dir = Path(self.args.debug_optimization_inputs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sampled_values = []
        for observation in self.optimization_observations:
            values = observation.depth[
                observation.valid_depth & torch.isfinite(observation.depth)
            ].flatten()
            if values.numel() > 10000:
                indices = torch.linspace(
                    0, values.numel() - 1, 10000, dtype=torch.long
                )
                values = values[indices]
            if values.numel() > 0:
                sampled_values.append(values)
        if sampled_values:
            valid_values = torch.cat(sampled_values).float()
            near, far = torch.quantile(
                valid_values,
                torch.tensor((0.02, 0.98), dtype=torch.float32),
            ).tolist()
        else:
            near, far = 0.0, 1.0

        manifest: dict[str, object] = {
            "source_image_dir": self.image_dir,
            "observation_count": len(self.optimization_observations),
            "depth_visualization_range": {"near": near, "far": far},
            "frames": [],
        }
        frame_paths_by_channel: dict[str, list[Path]] = {
            channel: []
            for channel in (
                "input",
                "depth",
                "depth_confidence",
                "valid_depth",
                "albedo",
                "roughness",
                "metallic",
                "normal_world",
            )
        }

        for observation in self.optimization_observations:
            stem = observation.image_name
            frame_dir = output_dir / stem
            frame_dir.mkdir(parents=True, exist_ok=True)
            size = (int(observation.camera.height), int(observation.camera.width))
            frame_manifest: dict[str, object] = {
                "image_name": stem,
                "camera": {
                    "width": observation.camera.width,
                    "height": observation.camera.height,
                    "fx": observation.camera.fx,
                    "fy": observation.camera.fy,
                    "cx": observation.camera.cx,
                    "cy": observation.camera.cy,
                },
                "paths": {},
            }

            input_filename = f"{stem}.png"
            source_path = Path(self.image_dir) / input_filename
            if not source_path.is_file():
                matches = list(Path(self.image_dir).glob(f"{stem}.*"))
                source_path = matches[0] if matches else source_path
            if source_path.is_file():
                image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if image is not None:
                    if image.shape[:2] != size:
                        image = cv2.resize(
                            image,
                            (size[1], size[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    path = frame_dir / "input.png"
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError(f"Failed to save debug image: {path}")
                    frame_manifest["paths"]["input"] = str(path)
                    frame_paths_by_channel["input"].append(path)

            outputs = {
                "albedo": observation.albedo,
                "roughness": observation.roughness,
                "metallic": observation.metallic,
                "depth_confidence": observation.depth_confidence,
                "valid_depth": observation.valid_depth.float(),
            }
            for channel, tensor in outputs.items():
                path = frame_dir / f"{channel}.png"
                if not cv2.imwrite(str(path), tensor_to_bgr(tensor)):
                    raise RuntimeError(f"Failed to save debug image: {path}")
                frame_manifest["paths"][channel] = str(path)
                frame_paths_by_channel[channel].append(path)

            depth_path = frame_dir / "depth.png"
            if not cv2.imwrite(
                str(depth_path), _depth_to_color_image(observation.depth, near, far)
            ):
                raise RuntimeError(f"Failed to save debug image: {depth_path}")
            frame_manifest["paths"]["depth"] = str(depth_path)
            frame_paths_by_channel["depth"].append(depth_path)

            normal_path = frame_dir / "normal_world.png"
            normal = F.normalize(observation.normal_world.float(), dim=0, eps=1e-6)
            normal_image = normal.mul(0.5).add(0.5).clamp(0.0, 1.0)
            if not cv2.imwrite(str(normal_path), tensor_to_bgr(normal_image)):
                raise RuntimeError(f"Failed to save debug image: {normal_path}")
            frame_manifest["paths"]["normal_world"] = str(normal_path)
            frame_paths_by_channel["normal_world"].append(normal_path)

            manifest["frames"].append(frame_manifest)

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for channel, paths in frame_paths_by_channel.items():
            if paths:
                from export_gaussian_map_relit_views import save_png_video

                save_png_video(paths, output_dir / f"{channel}.mp4", fps=10.0)
        print(
            f"[save] optimization input debug images {output_dir} "
            f"observations={len(self.optimization_observations)}",
            flush=True,
        )

    def _load_optimization_inputs(self) -> None:
        initial_path = self.output_path.with_name(
            f"{self.output_path.stem}_before_optimization{self.output_path.suffix}"
        )
        observations_path = self._optimization_observations_path()
        if not initial_path.is_file():
            raise FileNotFoundError(
                f"Missing pre-optimization map: {initial_path}. "
                "Run once without --skip_online_inference first."
            )
        if not observations_path.is_file():
            raise FileNotFoundError(
                f"Missing optimization observations: {observations_path}. "
                "Run once without --skip_online_inference first."
            )
        state_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
        self.state = GaussianMapState.from_dict(state_payload["gaussian_state"]).to(
            self.device
        )
        self.processed = set(state_payload.get("processed_names", []))
        if self.camera_path.is_file():
            camera_manifest = json.loads(self.camera_path.read_text(encoding="utf-8"))
            self.cameras = {
                str(item["image_name"]): PinholeCamera(**item)
                for item in camera_manifest.get("frames", [])
            }
        payload = torch.load(observations_path, map_location="cpu", weights_only=False)
        self.optimization_observations = []
        for item in payload["observations"]:
            camera_data = item["camera"]
            camera = PinholeCamera(**camera_data)
            self.optimization_observations.append(
                GlobalOptimizationObservation(
                    image_name=item["image_name"],
                    camera=camera,
                    depth=item["depth"],
                    albedo=item["albedo"],
                    roughness=item["roughness"],
                    metallic=item["metallic"],
                    normal_world=item["normal_world"],
                    valid_depth=item["valid_depth"],
                    normal_camera=item.get("normal_camera"),
                    depth_confidence=item.get("depth_confidence"),
                )
            )
        print(
            f"[skip-online] loaded map={initial_path} "
            f"observations={len(self.optimization_observations)}",
            flush=True,
        )
        self._save_optimization_input_debug_images()

    def _build_global_optimization_config(
        self,
        *,
        steps: int,
        debug_render_dir: str | Path | None = None,
    ) -> GlobalOptimizationConfig:
        if debug_render_dir is None:
            debug_render_dir = self.args.debug_global_optimization_dir
        return GlobalOptimizationConfig(
            steps=max(int(steps), 0),
            geometry_learning_rate=max(
                float(self.args.global_optimization_lr_geometry), 0.0
            ),
            albedo_learning_rate=max(
                float(self.args.global_optimization_lr_albedo), 0.0
            ),
            roughness_learning_rate=max(
                float(self.args.global_optimization_lr_roughness), 0.0
            ),
            metallic_learning_rate=max(
                float(self.args.global_optimization_lr_metallic), 0.0
            ),
            opacity_learning_rate=max(
                float(self.args.global_optimization_lr_opacity), 0.0
            ),
            optimize_camera_poses=self.args.global_optimization_optimize_pose,
            pose_rotation_learning_rate=max(
                float(self.args.global_optimization_lr_pose_rotation), 0.0
            ),
            pose_translation_learning_rate=max(
                float(self.args.global_optimization_lr_pose_translation), 0.0
            ),
            pose_rotation_regularization_weight=max(
                float(self.args.global_optimization_pose_rotation_reg), 0.0
            ),
            pose_translation_regularization_weight=max(
                float(self.args.global_optimization_pose_translation_reg), 0.0
            ),
            depth_weight=max(
                float(self.args.global_optimization_depth_weight), 0.0
            ),
            albedo_weight=max(
                float(self.args.global_optimization_albedo_weight), 0.0
            ),
            roughness_weight=max(
                float(self.args.global_optimization_roughness_weight), 0.0
            ),
            metallic_weight=max(
                float(self.args.global_optimization_metallic_weight), 0.0
            ),

            normal_weight=max(
                float(self.args.global_optimization_normal_weight), 0.0
            ),
            surface_normal_weight=max(
                float(self.args.global_optimization_surface_normal_weight), 0.0
            ),
            surface_normal_depth_edge_threshold=max(
                float(
                    self.args.global_optimization_surface_normal_depth_edge_threshold
                ),
                0.0,
            ),
            alpha_weight=max(
                float(self.args.global_optimization_alpha_weight), 0.0
            ),
            position_regularization_weight=max(
                float(self.args.global_optimization_position_reg), 0.0
            ),
            scale_regularization_weight=max(
                float(self.args.global_optimization_scale_reg), 0.0
            ),
            opacity_regularization_weight=max(
                float(self.args.global_optimization_opacity_reg), 0.0
            ),
            planar_scale=self.args.export_render_planar_scale,
            thickness_scale=self.args.export_render_thickness_scale,
            debug_render_dir=str(debug_render_dir) if debug_render_dir else "",
            debug_render_interval=self.args.debug_global_optimization_interval,
        )

    def _wait_online_optimization(self) -> None:
        future = self._optimization_future
        if future is not None:
            started = time.perf_counter()
            future.result()  # Worker synchronizes its stream before completing.
            self._optimization_future = None
            print(f"[optimization-wait] wait_ms={(time.perf_counter()-started)*1000:.3f}", flush=True)

    def _submit_online_optimization(self, stem: str) -> None:
        # Only one writer owns the map. No optimization job is replaced/dropped.
        self._wait_online_optimization()
        ready = None
        if self.device.type == "cuda":
            if self._optimization_stream is None:
                self._optimization_stream = torch.cuda.Stream(device=self.device)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        if self._optimization_executor is None:
            self._optimization_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map-opt")
        self._optimization_future = self._optimization_executor.submit(self._run_online_optimization, stem, ready)

    def _run_online_optimization(self, stem: str, ready) -> None:
        started = time.perf_counter()
        if ready is None:
            self.optimize_global_map_online(stem)
        else:
            with torch.cuda.device(self.device), torch.cuda.stream(self._optimization_stream):
                self._optimization_stream.wait_event(ready)
                try:
                    self.optimize_global_map_online(stem)
                finally:
                    self._optimization_stream.synchronize()
        print(f"[profile-online-worker] frame={stem} online_optimization_ms={(time.perf_counter()-started)*1000:.3f}", flush=True)

    def _finish_online_optimization(self) -> None:
        self._wait_online_optimization()
        if self.args.low_latency_pipeline and self._last_creation_camera is not None:
            latest = self._latest_preview_camera or self._last_creation_camera
            camera = self.cameras.get(latest.image_name, latest)
            self._publish_stream_preview(camera)
            self._last_creation_camera = None

    def _close_online_optimization(self) -> None:
        try:
            self._wait_online_optimization()
        finally:
            if self._optimization_executor is not None:
                self._optimization_executor.shutdown(wait=True)
                self._optimization_executor = None

    def optimize_global_map_online(self, stem: str) -> None:
        if not self.args.online_global_optimization:
            return
        steps = max(int(self.args.online_global_optimization_steps), 0)
        if steps <= 0:
            return
        interval = max(int(self.args.online_global_optimization_interval), 1)
        observation_count = len(self.optimization_observations)
        if observation_count == 0 or observation_count % interval != 0:
            return

        current_frame_index = self.frame_index[stem]
        window_multiplier = max(
            float(self.args.online_global_optimization_window_multiplier), 0.0
        )
        window_size = max(
            int(math.ceil(float(self.args.window_size) * window_multiplier)), 1
        )
        window_start_index = max(current_frame_index - window_size + 1, 0)
        local_observations = [
            observation
            for observation in self.optimization_observations
            if window_start_index
            <= self.frame_index.get(observation.image_name, -1)
            <= current_frame_index
        ]
        if not local_observations:
            return

        debug_render_dir: Path | None = None
        if self.args.debug_global_optimization_dir:
            debug_render_dir = (
                Path(self.args.debug_global_optimization_dir)
                / "online"
                / f"after_{observation_count:04d}_{stem}"
            )
        config = self._build_global_optimization_config(
            steps=steps,
            debug_render_dir=debug_render_dir,
        )
        config.recency_weighted_sampling = True
        start_time = time.perf_counter()
        self.state, history = optimize_gaussian_map_global(
            self.state,
            local_observations,
            config,
            self.device,
        )
        self._sync_optimized_cameras()
        elapsed_seconds = time.perf_counter() - start_time
        entry = {
            "after_frame": stem,
            "observation_count": observation_count,
            "local_observation_count": len(local_observations),
            "local_window_start_index": window_start_index,
            "local_window_end_index": current_frame_index,
            "local_frames": [
                observation.image_name for observation in local_observations
            ],
            "gaussian_count": int(self.state.means_world.shape[0]),
            "steps": steps,
            "elapsed_seconds": elapsed_seconds,
            "history": history,
        }
        self.online_optimization_history.append(entry)
        history_path = self.output_path.parent / "online_global_optimization_loss.json"
        history_path.write_text(
            json.dumps(self.online_optimization_history, indent=2),
            encoding="utf-8",
        )
        print(
            f"[online-global-opt] after={stem} observations={observation_count} "
            f"local_observations={len(local_observations)} "
            f"window=[{window_start_index},{current_frame_index}] "
            f"steps={steps} gaussians={self.state.means_world.shape[0]} "
            f"elapsed={elapsed_seconds:.2f}s history={history_path}",
            flush=True,
        )

    def optimize_global_map(self) -> bool:
        config = self._build_global_optimization_config(
            steps=max(int(self.args.global_optimization_steps), 0),
        )
        start_time = time.perf_counter()
        self.state, history = optimize_gaussian_map_global(
            self.state,
            self.optimization_observations,
            config,
            self.device,
        )
        self._sync_optimized_cameras()
        elapsed_seconds = time.perf_counter() - start_time
        if not history:
            print(
                f"[global-opt] skipped observations={len(self.optimization_observations)} "
                f"steps={config.steps} gaussians={self.state.means_world.shape[0]}",
                flush=True,
            )
            return False
        history_path = self.output_path.parent / "global_optimization_loss.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"[global-opt] finished observations={len(self.optimization_observations)} "
            f"gaussians={self.state.means_world.shape[0]} "
            f"elapsed={elapsed_seconds:.2f}s "
            f"({elapsed_seconds / 60.0:.2f}min) history={history_path}",
            flush=True,
        )
        return bool(history)

    def _sync_optimized_cameras(self) -> None:
        """Publish optimized observation poses to rendering/export consumers."""
        for observation in self.optimization_observations:
            self.cameras[observation.image_name] = observation.camera

    def save_pi3_depth_debug(self) -> None:
        if not self.optimization_observations:
            return
        depth_dir = self.output_path.parent / "pi3_depth_before_optimization"
        depth_dir.mkdir(parents=True, exist_ok=True)
        sampled_values = []
        for observation in self.optimization_observations:
            values = observation.depth[
                observation.valid_depth & torch.isfinite(observation.depth)
            ].flatten()
            if values.numel() > 10000:
                indices = torch.linspace(
                    0, values.numel() - 1, 10000, dtype=torch.long
                )
                values = values[indices]
            sampled_values.append(values)
        valid_values = torch.cat(sampled_values)
        if valid_values.numel() == 0:
            return
        near, far = torch.quantile(
            valid_values.float(),
            torch.tensor((0.02, 0.98), dtype=torch.float32),
        ).tolist()
        frame_paths = []
        for observation in self.optimization_observations:
            path = depth_dir / f"{observation.image_name}_pi3_depth.png"
            if not cv2.imwrite(
                str(path), _depth_to_color_image(observation.depth, near, far)
            ):
                raise RuntimeError(f"Failed to save Pi3 depth image: {path}")
            frame_paths.append(path)
        from export_gaussian_map_relit_views import save_png_video

        save_png_video(frame_paths, depth_dir / "pi3_depth.mp4", fps=10.0)
        print(
            f"[pi3-depth] saved {len(frame_paths)} frames to {depth_dir} "
            f"range near={near:.4f} far={far:.4f}",
            flush=True,
        )

    def _release_inference_models(self) -> None:
        self.material = None
        self.pi3 = None
        self.sam_mask_generator = None
        gc.collect()
        torch.cuda.empty_cache()

    def save_camera_manifest(self) -> None:
        frames = []
        for filename in self.files:
            camera = self.cameras[Path(filename).stem]
            frames.append(
                {
                    "image_name": camera.image_name,
                    "width": camera.width,
                    "height": camera.height,
                    "fx": camera.fx,
                    "fy": camera.fy,
                    "cx": camera.cx,
                    "cy": camera.cy,
                    "camera_to_world": camera.camera_to_world.tolist(),
                }
            )
        self.camera_path.parent.mkdir(parents=True, exist_ok=True)
        self.camera_path.write_text(
            json.dumps({"frames": frames}, indent=2), encoding="utf-8"
        )

    def _relit_output_dir(self, suffix: str = "") -> Path:
        output_dir = (
            Path(self.args.export_relit_output_dir)
            if self.args.export_relit_output_dir
            else self.output_path.parent / "relit_flash_full"
        )
        if suffix:
            return output_dir.with_name(f"{output_dir.name}_{suffix}")
        return output_dir

    def export_relit(self, state_path: Path, output_dir: Path) -> None:
        if not self.args.export_relit_after_fusion:
            return
        # Export the same camera-relative light and display material as the phone.
        settings = (
            self.stream_server.viewer_settings()
            if self.stream_server is not None
            else {"light_x": -0.25, "light_y": -0.35, "ambient": 0.05}
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "export_gaussian_map_relit_views.py"),
            "--state_path",
            str(state_path),
            "--image_dir",
            self.image_dir,
            "--camera_path",
            str(self.camera_path),
            "--output_dir",
            str(output_dir),
            "--device",
            str(self.device),
            "--backend",
            "gsplat_2dgs",
            "--relight_mode",
            "multi_point",
            "--relight_model",
            "mvinverse_diffuse",
            "--relight_output_encoding",
            "linear",
            "--relight_light_color=1.0,1.0,1.0",
            "--relight_light_energy_scale", "0.2",
            "--no_relight_apply_tonemap",
            "--relight_specular_scale", "0.0",
            "--relight_roughness_scale", "1.0",
            "--force_zero_metallic",
            "--relight_flash_intensity",
            str(self.args.relight_flash_intensity),
            "--relight_flash_radius",
            str(self.args.relight_flash_radius),
            "--relight_flash_beam_power",
            str(self.args.relight_flash_beam_power),
            "--relight_ambient",
            "0.0",
            "--render_planar_scale",
            str(self.args.export_render_planar_scale),
            "--render_thickness_scale",
            str(self.args.export_render_thickness_scale),
            "--output_size_policy",
            self.args.export_relit_output_size_policy,
        ]
        print(f"[export] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    os.environ["STREAMDARKGS_PROFILE_TIMING"] = args.profile_timing
    started = time.perf_counter()
    pipeline = FirstHitPipeline(args)
    if args.profile_timing != "off":
        torch.cuda.synchronize(pipeline.device)
        print(f"[profile-startup] initialization_ms={(time.perf_counter() - started) * 1000:.3f}", flush=True)
    pipeline.run()


if __name__ == "__main__":
    main()
