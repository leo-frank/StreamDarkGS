from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mvinverse.gsplat_fusion.live_pipeline_io import (
    list_images,
    load_state,
    tensor_to_bgr,
)
from mvinverse.gsplat_fusion.gsplat_adapter import (
    _get_activated_splat_params,
    _make_intrinsics,
    _make_viewmat,
    as_chw_normal_map,
    ensure_local_gsplat_path,
    gaussian_map_to_splats,
)
from mvinverse.gsplat_fusion.io import load_camera_manifest
from mvinverse.gsplat_fusion.live_material_renderer import _pbr_relight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render relit gaussian-map views for every input image/camera."
    )
    parser.add_argument(
        "--state_path", type=str, required=True, help="Saved gaussian_map.pt path."
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory of input RGB images."
    )
    parser.add_argument(
        "--camera_path", type=str, required=True, help="Camera manifest JSON."
    )
    parser.add_argument(
        "--pose_override_path",
        type=str,
        default="",
        help="Optional pose override JSON.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save relit images."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Rendering device.")
    parser.add_argument(
        "--backend",
        type=str,
        default="gsplat_2dgs",
        choices=("simple", "auto", "gsplat", "gsplat_2dgs"),
        help="Gaussian preview backend used to render material buffers.",
    )
    parser.add_argument(
        "--gsplat_root",
        type=str,
        default="",
        help="Optional isolated gsplat checkout path.",
    )
    parser.add_argument(
        "--relight_mode",
        type=str,
        default="directional",
        choices=("flash", "directional"),
        help="Relight mode. 'flash' uses camera-centered flash with depth-based attenuation; 'directional' uses a distant directional light.",
    )
    parser.add_argument(
        "--relight_light_dir",
        type=str,
        default="-0.25,-0.35,-1.0",
        help="Camera-space direction from the surface toward the directional light.",
    )
    parser.add_argument(
        "--relight_light_color",
        type=str,
        default="1.0,1.0,1.0",
        help="RGB light color.",
    )
    parser.add_argument(
        "--relight_ambient", type=float, default=0.05, help="Ambient term."
    )
    parser.add_argument(
        "--relight_flash_intensity",
        type=float,
        default=4.0,
        help="Flash intensity multiplier.",
    )
    parser.add_argument(
        "--relight_flash_radius", type=float, default=2.0, help="Flash radius."
    )
    parser.add_argument(
        "--relight_flash_beam_power", type=float, default=4.0, help="Flash beam power."
    )
    parser.add_argument(
        "--relight_tone_gamma",
        type=float,
        default=2.2,
        help="Gamma after tone mapping.",
    )
    parser.add_argument(
        "--relight_apply_tonemap",
        action="store_true",
        default=True,
        help="Apply tone mapping.",
    )
    parser.add_argument(
        "--no_relight_apply_tonemap",
        action="store_false",
        dest="relight_apply_tonemap",
        help="Disable tone mapping.",
    )
    parser.add_argument(
        "--relight_specular_scale", type=float, default=1.0, help="Specular scale."
    )
    parser.add_argument(
        "--render_planar_scale",
        type=float,
        default=1.8,
        help="2DGS planar scale multiplier used during export. Lower values are sharper but may reveal holes.",
    )
    parser.add_argument(
        "--render_thickness_scale",
        type=float,
        default=0.05,
        help="2DGS thickness scale multiplier used during export.",
    )
    parser.add_argument(
        "--limit", type=int, default=-1, help="Only export the first N images."
    )
    parser.add_argument(
        "--output_size_policy",
        type=str,
        default="manifest",
        choices=("manifest", "shrink_to_input"),
        help=(
            "Output resolution policy. 'manifest' renders at camera manifest size. "
            "'shrink_to_input' renders at the original input image size only when "
            "the input image is smaller than the manifest camera size; larger input "
            "images keep the manifest/Pi3 size."
        ),
    )
    return parser.parse_args()


def _natural_sort_key(path: Path) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
        if part
    )


def _fit_video_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    if (frame_width, frame_height) == (width, height):
        return frame
    scale = min(width / frame_width, height / frame_height)
    resized_width = max(int(round(frame_width * scale)), 1)
    resized_height = max(int(round(frame_height * scale)), 1)
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((height, width, 3), dtype=frame.dtype)
    offset_x = (width - resized_width) // 2
    offset_y = (height - resized_height) // 2
    canvas[
        offset_y : offset_y + resized_height, offset_x : offset_x + resized_width
    ] = resized
    return canvas


def save_png_video(image_paths: list[Path], output_path: Path, fps: float) -> int:
    sorted_paths = sorted(image_paths, key=_natural_sort_key)
    if not sorted_paths:
        raise ValueError("No PNG frames matched the requested video pattern")
    if fps <= 0:
        raise ValueError(f"Video FPS must be positive, got {fps}")

    first_frame = cv2.imread(str(sorted_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read video frame: {sorted_paths[0]}")
    first_height, first_width = first_frame.shape[:2]
    width = first_width + first_width % 2
    height = first_height + first_height % 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    try:
        encoder = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    if encoder.stdin is None or encoder.stderr is None:
        encoder.kill()
        encoder.wait()
        raise RuntimeError("Failed to open FFmpeg input pipe")

    try:
        for frame_path in sorted_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read video frame: {frame_path}")
            encoder.stdin.write(_fit_video_frame(frame, width, height).tobytes())
        encoder.stdin.close()
        error_output = encoder.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = encoder.wait()
    except Exception:
        encoder.kill()
        encoder.wait()
        temporary_path.unlink(missing_ok=True)
        raise

    if return_code != 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg failed to encode {output_path}: {error_output}")

    shutil.copyfile(temporary_path, output_path)
    temporary_path.unlink(missing_ok=True)

    print(
        f"[video] {output_path} frames={len(sorted_paths)} fps={fps:g} "
        f"size={width}x{height} codec=h264 first={sorted_paths[0].name} "
        f"last={sorted_paths[-1].name}",
        flush=True,
    )
    return len(sorted_paths)


def _camera_for_output_size(camera, image_path: Path, policy: str):
    if policy == "manifest":
        return camera
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return camera
    input_h, input_w = image.shape[:2]
    if input_w < int(camera.width) or input_h < int(camera.height):
        return camera.scaled_to(int(input_w), int(input_h))
    return camera


def _parse_vec3(text: str) -> torch.Tensor:
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {text}")
    return torch.tensor(values, dtype=torch.float32)


def _depth_to_bgr(
    depth: torch.Tensor,
    near: float,
    far: float,
) -> np.ndarray:
    depth_array = depth.detach().cpu().float().squeeze().numpy()
    valid = np.isfinite(depth_array) & (depth_array > 1e-6)
    visualization = np.zeros(depth_array.shape, dtype=np.uint8)
    if valid.any():
        if far > near:
            normalized = np.clip((depth_array - near) / (far - near), 0.0, 1.0)
            visualization[valid] = np.round(
                (1.0 - normalized[valid]) * 255.0
            ).astype(np.uint8)
        else:
            visualization[valid] = 128
    colored = cv2.applyColorMap(visualization, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _render_relit_view(
    state,
    camera,
    device: torch.device,
    backend: str,
    gsplat_root: str | None,
    relight_mode: str,
    relight_light_dir: torch.Tensor,
    relight_light_color: torch.Tensor,
    relight_flash_intensity: float,
    relight_flash_radius: float,
    relight_flash_beam_power: float,
    relight_tone_gamma: float,
    relight_apply_tonemap: bool,
    relight_specular_scale: float,
    relight_ambient: float,
    render_planar_scale: float,
    render_thickness_scale: float,
) -> dict[str, torch.Tensor]:
    if backend != "gsplat_2dgs":
        raise ValueError(
            "relight export currently requires --backend gsplat_2dgs for fast GPU normal/depth rendering"
        )

    ensure_local_gsplat_path(gsplat_root)
    from gsplat.rendering import rasterization_2dgs

    splats = gaussian_map_to_splats(
        state,
        device=device,
        orient_to_normals=True,
        planar_scale=max(float(render_planar_scale), 1e-6),
        thickness_scale=max(float(render_thickness_scale), 1e-6),
    )
    activated = _get_activated_splat_params(splats)
    means = activated["means"]
    quats = activated["quats"]
    scales = activated["scales"]
    opacities = activated["opacities"]
    viewmats = _make_viewmat(camera, device, means.dtype).unsqueeze(0)
    Ks = _make_intrinsics(camera, device, means.dtype).unsqueeze(0)

    material_buffer = torch.cat(
        (
            state.albedo.to(device=device, dtype=means.dtype),
            state.roughness.to(device=device, dtype=means.dtype),
            state.metallic.to(device=device, dtype=means.dtype),
        ),
        dim=1,
    )
    (
        render_colors,
        render_alphas,
        render_normals,
        _surf_normals,
        _render_distort,
        _render_median,
        _meta,
    ) = rasterization_2dgs(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=material_buffer,
        viewmats=viewmats,
        Ks=Ks,
        width=camera.width,
        height=camera.height,
        sh_degree=None,
        packed=False,
        render_mode="RGB+ED",
    )
    render_colors = render_colors[0]
    coverage = render_alphas[0, ..., 0]
    valid = coverage > 1e-4
    material = render_colors[..., :5] / coverage.clamp_min(1e-6).unsqueeze(-1)
    material = torch.where(valid.unsqueeze(-1), material, 0.0)
    albedo = material[..., :3].permute(2, 0, 1).clamp(0.0, 1.0)
    roughness = material[..., 3].unsqueeze(0).clamp(0.0, 1.0)
    metallic = material[..., 4].unsqueeze(0).clamp(0.0, 1.0)
    depth = torch.where(valid, render_colors[..., 5], 0.0)
    render_normals = as_chw_normal_map(render_normals)
    normals_world = render_normals.to(device=albedo.device, dtype=albedo.dtype)
    normals_world = torch.where(
        valid.unsqueeze(0), normals_world, torch.zeros_like(normals_world)
    )

    rotation = camera.rotation_camera_to_world.to(
        device=albedo.device, dtype=albedo.dtype
    )
    ys = torch.arange(camera.height, device=device, dtype=means.dtype)
    xs = torch.arange(camera.width, device=device, dtype=means.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x = (grid_x - camera.cx) / camera.fx * depth
    y = (grid_y - camera.cy) / camera.fy * depth
    points_camera = torch.stack([x, y, depth], dim=-1)
    c2w = camera.camera_to_world.to(device=device, dtype=means.dtype)
    points_world = points_camera @ c2w[:3, :3].transpose(0, 1) + c2w[:3, 3]
    camera_center = c2w[:3, 3]
    viewdirs_world = F.normalize(
        camera_center.view(1, 1, 3) - points_world,
        dim=-1,
        eps=1e-6,
    ).permute(2, 0, 1)
    viewdirs_world = torch.where(
        valid.unsqueeze(0), viewdirs_world, torch.zeros_like(viewdirs_world)
    )

    relight_light_dir = relight_light_dir.to(device=albedo.device, dtype=albedo.dtype)
    relight_light_color = relight_light_color.to(
        device=albedo.device, dtype=albedo.dtype
    )

    if relight_mode == "directional":
        light_dir_world = (
            torch.einsum("ij,j->i", rotation, relight_light_dir)
            .view(1, 1, 3)
            .expand(camera.height, camera.width, 3)
        )
        attenuation = torch.ones(
            (1, camera.height, camera.width), dtype=albedo.dtype, device=albedo.device
        )
    else:
        light_vec_world = camera_center.view(1, 1, 3) - points_world
        light_dir_world = F.normalize(light_vec_world, dim=-1, eps=1e-6)
        camera_forward = F.normalize(
            rotation @ torch.tensor([0.0, 0.0, 1.0], dtype=means.dtype, device=device),
            dim=0,
            eps=1e-6,
        )
        beam_dir = F.normalize(
            points_world - camera_center.view(1, 1, 3), dim=-1, eps=1e-6
        )
        beam = (
            (beam_dir * camera_forward.view(1, 1, 3))
            .sum(dim=-1, keepdim=True)
            .clamp_min(0.0)
        )
        beam = beam.pow(max(float(relight_flash_beam_power), 1.0))
        flash_radius = max(float(relight_flash_radius), 1e-3)
        dist2 = light_vec_world.square().sum(dim=-1, keepdim=True)
        attenuation = (
            float(relight_flash_intensity)
            * beam
            / (1.0 + dist2 / (flash_radius * flash_radius))
        )
        attenuation = attenuation.permute(2, 0, 1) * valid.unsqueeze(0)

    relit = _pbr_relight(
        albedo=albedo.to(dtype=torch.float32),
        roughness=roughness.to(dtype=torch.float32),
        metallic=metallic.to(dtype=torch.float32),
        normals=normals_world.to(dtype=torch.float32),
        viewdirs=viewdirs_world.to(dtype=torch.float32),
        light_dir=light_dir_world.to(dtype=torch.float32),
        light_color=relight_light_color.to(dtype=torch.float32),
        light_attenuation=attenuation.to(dtype=torch.float32),
        ambient=float(relight_ambient),
        tone_gamma=float(relight_tone_gamma),
        apply_tonemap=bool(relight_apply_tonemap),
        specular_scale=float(relight_specular_scale),
    )
    relit = relit * valid.unsqueeze(0)
    normals_vis = normals_world * 0.5 + 0.5
    return {
        "relit": relit.clamp(0.0, 1.0),
        "albedo": albedo.clamp(0.0, 1.0),
        "depth": depth.unsqueeze(0),
        "roughness": roughness.expand(3, -1, -1).clamp(0.0, 1.0),
        "metallic": metallic.expand(3, -1, -1).clamp(0.0, 1.0),
        "normal": normals_vis.clamp(0.0, 1.0),
    }


def main() -> None:
    args = parse_args()
    state, _processed = load_state(Path(args.state_path))
    if state.means_world.shape[0] == 0:
        raise RuntimeError(f"Gaussian state is empty: {args.state_path}")

    cameras = load_camera_manifest(
        args.camera_path, pose_override_path=args.pose_override_path or None
    )
    image_filenames = list_images(args.image_dir)
    if args.limit > 0:
        image_filenames = image_filenames[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    relight_light_dir = _parse_vec3(args.relight_light_dir)
    relight_light_color = _parse_vec3(args.relight_light_color)
    render_types = ("relit", "albedo", "depth", "roughness", "metallic", "normal")
    rendered_paths: dict[str, list[Path]] = {
        render_type: [] for render_type in render_types
    }
    depth_frames: list[tuple[str, torch.Tensor]] = []
    from tqdm import tqdm
    for filename in tqdm(image_filenames):
        stem = Path(filename).stem
        if stem not in cameras:
            print(f"[skip] missing camera for {stem}", flush=True)
            continue
        camera = _camera_for_output_size(
            cameras[stem],
            Path(args.image_dir) / filename,
            args.output_size_policy,
        )
        outputs = _render_relit_view(
            state=state,
            camera=camera.to(device),
            device=device,
            backend=args.backend,
            gsplat_root=args.gsplat_root or None,
            relight_mode=args.relight_mode,
            relight_light_dir=relight_light_dir,
            relight_light_color=relight_light_color,
            relight_flash_intensity=args.relight_flash_intensity,
            relight_flash_radius=args.relight_flash_radius,
            relight_flash_beam_power=args.relight_flash_beam_power,
            relight_tone_gamma=args.relight_tone_gamma,
            relight_apply_tonemap=args.relight_apply_tonemap,
            relight_specular_scale=args.relight_specular_scale,
            relight_ambient=args.relight_ambient,
            render_planar_scale=args.render_planar_scale,
            render_thickness_scale=args.render_thickness_scale,
        )
        for render_type in render_types:
            if render_type == "depth":
                depth_frames.append((stem, outputs[render_type].detach().cpu()))
                continue
            image_path = output_dir / f"{stem}_{render_type}.png"
            image = tensor_to_bgr(outputs[render_type])
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Failed to write rendered image: {image_path}")
            rendered_paths[render_type].append(image_path)
        # print(f"[relight] {stem} ({camera.width}x{camera.height})", flush=True)

    if depth_frames:
        sampled_values = []
        for _, depth in depth_frames:
            values = depth[torch.isfinite(depth) & (depth > 1e-6)].flatten()
            if values.numel() > 10000:
                indices = torch.linspace(
                    0, values.numel() - 1, 10000, dtype=torch.long
                )
                values = values[indices]
            sampled_values.append(values)
        valid_values = torch.cat(sampled_values)
        near, far = torch.quantile(
            valid_values,
            torch.tensor((0.02, 0.98), dtype=valid_values.dtype),
        ).tolist()
        for stem, depth in depth_frames:
            image_path = output_dir / f"{stem}_depth.png"
            if not cv2.imwrite(str(image_path), _depth_to_bgr(depth, near, far)):
                raise RuntimeError(f"Failed to write rendered image: {image_path}")
            rendered_paths["depth"].append(image_path)
        print(
            f"[depth] shared visualization range near={near:.4f} far={far:.4f} "
            f"frames={len(depth_frames)}",
            flush=True,
        )

    for render_type in render_types:
        save_png_video(
            rendered_paths[render_type],
            output_dir / f"{render_type}.mp4",
            fps=10.0,
        )


if __name__ == "__main__":
    main()
