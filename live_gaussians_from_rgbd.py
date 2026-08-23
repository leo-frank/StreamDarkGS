from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="First-hit Gaussian completion with Pi3 and MVInverse."
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--pi3_root", required=True)
    parser.add_argument("--pi3_ckpt", default="")
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
    parser.add_argument("--pi3_device", default="cuda")
    parser.add_argument("--fusion_device", default="cuda")
    parser.add_argument("--mvinverse_device", default="cuda")
    parser.add_argument("--mvinverse_max_long_edge", type=int, default=512)
    parser.add_argument("--input_frame_stride", type=int, default=1)
    parser.add_argument("--fusion_frame_stride", type=int, default=5)
    parser.add_argument("--pixel_stride", type=int, default=1)
    parser.add_argument("--gaussian_scale_xy_multiplier", type=float, default=0.8)
    parser.add_argument("--pi3_min_confidence", type=float, default=0.3)
    parser.add_argument("--creation_min_confidence", type=float, default=0.0)
    parser.add_argument("--creation_max_depth_quantile", type=float, default=1.0)
    parser.add_argument("--first_hit_coverage_threshold", type=float, default=0.95)
    parser.add_argument("--creation_material_align_to_map", action="store_true")
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
    parser.add_argument("--creation_material_align_cluster_count", type=int, default=12)
    parser.add_argument(
        "--creation_material_align_cluster_min_pixels", type=int, default=512
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
    parser.add_argument(
        "--creation_material_align_global_strength", type=float, default=1.0
    )
    parser.add_argument(
        "--creation_material_align_global_max_log_offset", type=float, default=0.25
    )
    parser.add_argument("--debug_creation_mvinverse_dir", default="")
    parser.add_argument("--debug_creation_mvinverse_video_fps", type=float, default=10.0)
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
    parser.add_argument("--global_optimization_steps", type=int, default=1000)
    parser.add_argument(
        "--global_optimization_lr_geometry", type=float, default=1e-4
    )
    parser.add_argument("--global_optimization_lr_albedo", type=float, default=1e-2)
    parser.add_argument("--global_optimization_lr_opacity", type=float, default=1e-3)
    parser.add_argument(
        "--global_optimization_depth_weight", type=float, default=1.0
    )
    parser.add_argument(
        "--global_optimization_albedo_weight", type=float, default=1.0
    )
    parser.add_argument(
        "--global_optimization_position_reg", type=float, default=1e-2
    )
    parser.add_argument("--global_optimization_scale_reg", type=float, default=1e-2)
    parser.add_argument("--global_optimization_opacity_reg", type=float, default=1e-3)
    parser.add_argument(
        "--no_export_relit_apply_tonemap",
        action="store_false",
        dest="export_relit_apply_tonemap",
        default=True,
    )
    return parser.parse_args()


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


def _save_map_render_debug(
    output_dir: Path,
    stem: str,
    rendered_albedo: torch.Tensor,
    size: tuple[int, int],
) -> None:
    rendered_albedo = rendered_albedo.float().clamp(0.0, 1.0)
    if rendered_albedo.shape[-2:] != size:
        rendered_albedo = F.interpolate(
            rendered_albedo.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_map_render.png"
    if not cv2.imwrite(str(output_path), tensor_to_bgr(rendered_albedo)):
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
    cluster_path = output_dir / f"{stem}_material_clusters.png"
    if not cv2.imwrite(str(cluster_path), tensor_to_bgr(image)):
        raise RuntimeError(f"Failed to save material cluster debug image: {cluster_path}")

    valid_image = valid.float().expand(3, -1, -1)
    valid_path = output_dir / f"{stem}_material_valid.png"
    if not cv2.imwrite(str(valid_path), tensor_to_bgr(valid_image)):
        raise RuntimeError(f"Failed to save material valid debug image: {valid_path}")


class FirstHitPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.fusion_device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("First-hit fusion requires CUDA and gsplat.")

        self.image_dir = str(Path(args.image_dir))
        files = list_images(self.image_dir)[:: max(args.input_frame_stride, 1)]
        if not files:
            raise ValueError(f"No images found in {self.image_dir}")
        self.files = tuple(files)
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
        self.sam_mask_generator = None
        if args.creation_material_align_region_source == "sam":
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2

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
            )
            self.sam_mask_generator = SAM2AutomaticMaskGenerator(sam_model)
            print(
                f"[sam2] automatic mask generator loaded checkpoint={checkpoint_path}",
                flush=True,
            )

        self.pi3 = Pi3GeometryStream(
            pi3_root=args.pi3_root,
            ckpt=args.pi3_ckpt,
            device=args.pi3_device,
            alignment_mode=args.pose_alignment_mode,
            alignment_reference=args.pi3_alignment_reference,
            overlap_policy=args.pi3_overlap_policy,
        )
        self.material = MVInverseMaterialStream(
            ckpt=args.mvinverse_ckpt,
            device=args.mvinverse_device,
            max_long_edge=args.mvinverse_max_long_edge,
            policy=args.mvinverse_overlap_policy,
        )
        self.state = GaussianMapState.empty(device=self.device)
        self.processed: set[str] = set()
        self.optimization_observations: list[GlobalOptimizationObservation] = []

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
            render_planar_scale=args.export_render_planar_scale,
            render_thickness_scale=args.export_render_thickness_scale,
        )

    def run(self) -> None:
        schedule = build_window_schedule(
            self.files,
            window_size=self.args.window_size,
            window_stride=self.args.window_stride,
        )
        for step in schedule:
            self.process_window(step.index, step.frame_names)
            for filename in step.mature_names:
                self.finalize_frame(filename)

        self.save_camera_manifest()
        self._close_material_debug_videos()
        if self.args.global_optimization:
            initial_path = self.output_path.with_name(
                f"{self.output_path.stem}_before_optimization{self.output_path.suffix}"
            )
            save_state(self.state, initial_path, self.processed)
            print(f"[save] initial map {initial_path}", flush=True)
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
            self.optimize_global_map()
        save_state(self.state, self.output_path, self.processed)
        print(
            f"[save] {self.output_path} gaussians={self.state.means_world.shape[0]}",
            flush=True,
        )
        self.state = self.state.to("cpu")
        self._release_inference_models()
        self.export_relit(
            self.output_path,
            self._relit_output_dir("after_optimization")
            if self.args.global_optimization
            else self._relit_output_dir(),
        )
        gc.collect()
        torch.cuda.empty_cache()

    def process_window(
        self,
        window_index: int,
        window: tuple[str, ...],
    ) -> None:
        print(f"[window] index={window_index} frames={','.join(window)}", flush=True)
        _, pi3_size = self.pi3.process_window(self.image_dir, list(window))
        tensors = self.pi3.get_cached_image_tensors(self.image_dir, list(window))
        _, material_input_size, overlap_count = self.material.process_window(
            tensors,
            target_size=pi3_size,
            output_size=pi3_size,
        )
        print(
            f"[mvinverse] window={window_index} input={material_input_size} "
            f"overlap={overlap_count}",
            flush=True,
        )

    def finalize_frame(self, filename: str) -> None:
        stem = Path(filename).stem
        geometry, geometry_count = self.pi3.resolve_frame(stem)
        self.cameras[stem] = geometry.camera
        creation_frame = (
            self.frame_index[stem] % max(self.args.fusion_frame_stride, 1) == 0
        )
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
            print(
                f"[finalize] {stem} creation_frame=False "
                f"geometry_observations={geometry_count}",
                flush=True,
            )
        self.pi3.release_frame(stem)
        self.material.release_frame(stem)

    def fuse_creation_frame(
        self,
        filename: str,
        geometry: GeometryObservation,
        material: dict[str, torch.Tensor],
        *,
        geometry_count: int,
        material_count: int,
    ) -> None:
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
            self._save_material_debug_frame(stem, albedo, "before_align")

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
                cluster_sample_pixels=self.args.creation_material_align_cluster_sample_pixels,
                cluster_spatial_weight=self.args.creation_material_align_cluster_spatial_weight,
                cluster_smoothing_kernel_size=(
                    self.args.creation_material_align_cluster_smoothing_kernel_size
                ),
                region_source=self.args.creation_material_align_region_source,
                sam_mask_generator=self.sam_mask_generator,
                region_image=albedo,
                global_max_log_offset=(
                    self.args.creation_material_align_global_max_log_offset
                    * self.args.creation_material_align_global_strength
                ),
                cluster_max_log_offset=(
                    self.args.creation_material_align_cluster_max_log_offset
                    * self.args.creation_material_align_cluster_residual_strength
                ),
                planar_scale=self.args.export_render_planar_scale,
                thickness_scale=self.args.export_render_thickness_scale,
            )
            print(
                f"[material-align] {stem} applied={stats['applied']} "
                f"valid={stats['valid_pixels']} clusters={stats.get('used_clusters', 0)}",
                flush=True,
            )
            cluster_labels = stats.get("cluster_labels")
            cluster_valid = stats.get("cluster_valid")
            if (
                self.material_debug_dir is not None
                and isinstance(cluster_labels, torch.Tensor)
                and isinstance(cluster_valid, torch.Tensor)
            ):
                _save_material_cluster_debug(
                    self.material_debug_dir,
                    stem,
                    cluster_labels,
                    cluster_valid,
                )

        if self.material_debug_dir is not None:
            self._save_material_debug_frame(stem, albedo, "aligned")

        if self.args.global_optimization:
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
                    valid_depth=valid_depth.detach().cpu(),
                )
            )

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
        self.state, stats = fuse_frame_gaussians(
            self.state,
            frame,
            config=self.fusion_config,
            allow_create=True,
        )
        self.processed.add(stem)
        print(
            f"[fusion] {stem} creation_frame=True "
            f"geometry_observations={geometry_count} "
            f"material_observations={material_count} "
            f"candidates={frame.num_confidence_kept}/{frame.num_candidates} "
            f"created={stats['created']} covered={stats['covered_creation_skipped']} "
            f"formal={self.state.means_world.shape[0]}",
            flush=True,
        )

    def _save_material_debug_frame(
        self,
        stem: str,
        albedo: torch.Tensor,
        label: str,
    ) -> None:
        if self.material_debug_dir is None:
            return
        image_stem = stem if label == "aligned" else f"{stem}_before_align"
        _save_creation_material_debug(
            self.material_debug_dir,
            image_stem,
            albedo=albedo,
        )
        self.material_debug_video_frames.setdefault(label, []).append(
            self.material_debug_dir / f"{image_stem}.png"
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
            video_path = self.material_debug_dir / f"albedo_{label}.mp4"
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
        self.material_debug_video_frames.clear()

    def optimize_global_map(self) -> None:
        config = GlobalOptimizationConfig(
            steps=max(int(self.args.global_optimization_steps), 0),
            geometry_learning_rate=max(
                float(self.args.global_optimization_lr_geometry), 0.0
            ),
            albedo_learning_rate=max(
                float(self.args.global_optimization_lr_albedo), 0.0
            ),
            opacity_learning_rate=max(
                float(self.args.global_optimization_lr_opacity), 0.0
            ),
            depth_weight=max(
                float(self.args.global_optimization_depth_weight), 0.0
            ),
            albedo_weight=max(
                float(self.args.global_optimization_albedo_weight), 0.0
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
        )
        start_time = time.perf_counter()
        self.state, history = optimize_gaussian_map_global(
            self.state,
            self.optimization_observations,
            config,
            self.device,
        )
        elapsed_seconds = time.perf_counter() - start_time
        history_path = self.output_path.parent / "global_optimization_loss.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"[global-opt] finished observations={len(self.optimization_observations)} "
            f"gaussians={self.state.means_world.shape[0]} "
            f"elapsed={elapsed_seconds:.2f}s "
            f"({elapsed_seconds / 60.0:.2f}min) history={history_path}",
            flush=True,
        )

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
            "flash",
            "--relight_flash_intensity",
            str(self.args.relight_flash_intensity),
            "--relight_flash_radius",
            str(self.args.relight_flash_radius),
            "--relight_flash_beam_power",
            str(self.args.relight_flash_beam_power),
            "--relight_ambient",
            str(self.args.export_relit_ambient),
            "--render_planar_scale",
            str(self.args.export_render_planar_scale),
            "--render_thickness_scale",
            str(self.args.export_render_thickness_scale),
            "--output_size_policy",
            self.args.export_relit_output_size_policy,
        ]
        if not self.args.export_relit_apply_tonemap:
            command.append("--no_relight_apply_tonemap")
        print(f"[export] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


def main() -> None:
    FirstHitPipeline(parse_args()).run()


if __name__ == "__main__":
    main()
