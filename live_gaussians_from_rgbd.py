from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
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
    parser.add_argument("--creation_material_align_cluster_count", type=int, default=6)
    parser.add_argument(
        "--creation_material_align_cluster_min_pixels", type=int, default=2048
    )
    parser.add_argument(
        "--creation_material_align_cluster_sample_pixels", type=int, default=50000
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
    parser.add_argument("--debug_creation_mvinverse_thumb_width", type=int, default=256)
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


def _save_material_debug(
    output_dir: Path,
    window_index: int,
    proposals: dict[str, object],
    width: int,
) -> None:
    for channel in ("albedo", "normal", "roughness", "metallic"):
        panels: list[np.ndarray] = []
        for stem, proposal in proposals.items():
            maps = proposal.as_dict()
            value = maps[channel]
            if channel == "normal":
                value = value * 0.5 + 0.5
            panel = tensor_to_bgr(value)
            scale = max(width, 32) / max(panel.shape[1], 1)
            panel = cv2.resize(
                panel,
                (max(width, 32), max(int(round(panel.shape[0] * scale)), 1)),
                interpolation=cv2.INTER_AREA,
            )
            cv2.putText(
                panel,
                stem,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
        if panels:
            path = output_dir / f"window_{window_index:04d}_{channel}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.concatenate(panels, axis=1))


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
        save_state(self.state, self.output_path, self.processed)
        print(
            f"[save] {self.output_path} gaussians={self.state.means_world.shape[0]}",
            flush=True,
        )
        self.export_relit()
        self.state = self.state.to("cpu")
        self.material = None
        self.pi3 = None
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
        proposals, material_input_size, overlap_count = self.material.process_window(
            tensors,
            target_size=pi3_size,
            output_size=pi3_size,
        )
        if self.material_debug_dir is not None:
            _save_material_debug(
                self.material_debug_dir,
                window_index,
                proposals,
                self.args.debug_creation_mvinverse_thumb_width,
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

    def export_relit(self) -> None:
        if not self.args.export_relit_after_fusion:
            return
        output_dir = (
            Path(self.args.export_relit_output_dir)
            if self.args.export_relit_output_dir
            else self.output_path.parent / "relit_flash_full"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "export_gaussian_map_relit_views.py"),
            "--state_path",
            str(self.output_path),
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
