from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .types import PinholeCamera


@dataclass
class FrameGaussians:
    image_name: str
    camera: PinholeCamera
    num_candidates: int
    num_confidence_kept: int
    means_world: torch.Tensor
    sample_y: torch.Tensor
    sample_x: torch.Tensor
    colors: torch.Tensor
    albedo: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    confidence: torch.Tensor


@dataclass
class GaussianMapState:
    means_world: torch.Tensor
    colors: torch.Tensor
    albedo: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    confidence_sum: torch.Tensor
    update_count: torch.Tensor
    quats: torch.Tensor | None = None

    @classmethod
    def empty(
        cls,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "GaussianMapState":
        zeros3 = torch.zeros((0, 3), device=device, dtype=dtype)
        zeros4 = torch.zeros((0, 4), device=device, dtype=dtype)
        zeros1 = torch.zeros((0, 1), device=device, dtype=dtype)
        return cls(
            means_world=zeros3.clone(),
            colors=zeros3.clone(),
            albedo=zeros3.clone(),
            roughness=zeros1.clone(),
            metallic=zeros1.clone(),
            scales=zeros3.clone(),
            opacities=zeros1.clone(),
            confidence_sum=zeros1.clone(),
            update_count=zeros1.clone(),
            quats=zeros4.clone(),
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        payload = {
            "means_world": self.means_world,
            "colors": self.colors,
            "albedo": self.albedo,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "scales": self.scales,
            "opacities": self.opacities,
            "confidence_sum": self.confidence_sum,
            "update_count": self.update_count,
        }
        if self.quats is not None:
            payload["quats"] = self.quats
        return payload

    def to(self, device: torch.device | str) -> "GaussianMapState":
        device = torch.device(device)
        quats = self.quats.to(device=device) if self.quats is not None else None
        return GaussianMapState(
            means_world=self.means_world.to(device=device),
            colors=self.colors.to(device=device),
            albedo=self.albedo.to(device=device),
            roughness=self.roughness.to(device=device),
            metallic=self.metallic.to(device=device),
            scales=self.scales.to(device=device),
            opacities=self.opacities.to(device=device),
            confidence_sum=self.confidence_sum.to(device=device),
            update_count=self.update_count.to(device=device),
            quats=quats,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, torch.Tensor]) -> "GaussianMapState":
        required = (
            "means_world",
            "colors",
            "scales",
            "opacities",
            "confidence_sum",
            "update_count",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"GaussianMapState payload missing keys: {missing}")
        colors = payload["colors"]
        albedo = payload.get("albedo")
        roughness = payload.get("roughness")
        metallic = payload.get("metallic")
        if albedo is None:
            albedo = colors.clone()
        if roughness is None:
            roughness = torch.zeros(
                (colors.shape[0], 1), dtype=colors.dtype, device=colors.device
            )
        if metallic is None:
            metallic = torch.zeros(
                (colors.shape[0], 1), dtype=colors.dtype, device=colors.device
            )
        quats = payload.get("quats")
        if quats is None:
            normals_world = payload.get("normals_world")
            if normals_world is not None:
                quats = quats_from_normals(normals_world)
            else:
                quats = torch.zeros(
                    (colors.shape[0], 4), dtype=colors.dtype, device=colors.device
                )
                quats[:, 0] = 1.0
        return cls(
            means_world=payload["means_world"],
            colors=colors,
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            scales=payload["scales"],
            opacities=payload["opacities"],
            confidence_sum=payload["confidence_sum"],
            update_count=payload["update_count"],
            quats=quats,
        )


@dataclass
class RGBDGaussianBuilderConfig:
    pixel_stride: int = 4
    min_depth: float = 1e-3
    max_depth: float = 1e4
    min_confidence: float = 0.0
    scale_xy_multiplier: float = 1.0
    scale_z_multiplier: float = 0.5
    default_opacity: float = 1.0


@dataclass
class RGBDGaussianFusionConfig:
    creation_min_confidence: float = 0.0
    creation_max_depth_quantile: float = 1.0
    first_hit_coverage_threshold: float = 0.95
    render_planar_scale: float = 1.8
    render_thickness_scale: float = 0.05


def _pixel_grid(
    height: int, width: int, stride: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(0, height, stride, device=device, dtype=dtype)
    xs = torch.arange(0, width, stride, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_y, grid_x


def _camera_depth_for_points(
    points_world: torch.Tensor, camera: PinholeCamera
) -> torch.Tensor:
    if points_world.numel() == 0:
        return torch.zeros((0,), device=points_world.device, dtype=points_world.dtype)
    camera = camera.to(points_world.device)
    world_to_camera = camera.world_to_camera.to(
        device=points_world.device, dtype=points_world.dtype
    )
    points_camera = (
        points_world @ world_to_camera[:3, :3].transpose(0, 1) + world_to_camera[:3, 3]
    )
    return points_camera[:, 2]


def _confidence_depth_creation_mask(
    frame_means: torch.Tensor,
    frame_confidence: torch.Tensor,
    camera: PinholeCamera,
    config: RGBDGaussianFusionConfig,
) -> tuple[torch.Tensor, dict[str, int], torch.Tensor]:
    depth = _camera_depth_for_points(frame_means, camera).clamp_min(1e-6)
    mask = torch.ones(
        (frame_means.shape[0],), device=frame_means.device, dtype=torch.bool
    )
    stats = {
        "create_confidence_skipped": 0,
        "create_depth_skipped": 0,
    }
    min_conf = max(float(config.creation_min_confidence), 0.0)
    if min_conf > 0.0:
        confidence_mask = frame_confidence[:, 0] >= min_conf
        stats["create_confidence_skipped"] = int((~confidence_mask).sum().item())
        mask &= confidence_mask
    quantile = float(config.creation_max_depth_quantile)
    if quantile < 1.0 and depth.numel() > 0:
        quantile = min(max(quantile, 0.01), 1.0)
        valid_depth = depth[torch.isfinite(depth) & (depth > 1e-6)]
        if valid_depth.numel() > 0:
            depth_limit = torch.quantile(valid_depth.float(), quantile).to(
                dtype=depth.dtype
            )
            depth_mask = depth <= depth_limit
            stats["create_depth_skipped"] = int((~depth_mask).sum().item())
            mask &= depth_mask
    return mask, stats, depth


def depth_to_world_points(
    depth: torch.Tensor, camera: PinholeCamera, stride: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if depth.dim() != 3 or depth.shape[0] != 1:
        raise ValueError(f"depth must have shape [1, H, W], got {tuple(depth.shape)}")
    depth = depth[0]
    device = depth.device
    dtype = depth.dtype
    height, width = depth.shape
    grid_y, grid_x = _pixel_grid(height, width, stride, device, dtype)
    z = depth[grid_y.long(), grid_x.long()]

    x = (grid_x - camera.cx) / camera.fx * z
    y = (grid_y - camera.cy) / camera.fy * z
    points_camera = torch.stack([x, y, z], dim=-1)

    c2w = camera.camera_to_world.to(device=device, dtype=dtype)
    points_world = points_camera @ c2w[:3, :3].transpose(0, 1) + c2w[:3, 3]
    return points_world, grid_y, grid_x


def normals_from_depth(depth: torch.Tensor, camera: PinholeCamera) -> torch.Tensor:
    points_world, _, _ = depth_to_world_points(depth, camera, stride=1)
    dx = points_world[2:, 1:-1, :] - points_world[:-2, 1:-1, :]
    dy = points_world[1:-1, 2:, :] - points_world[1:-1, :-2, :]
    normals = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1, eps=1e-6)
    normals = F.pad(normals.permute(2, 0, 1), (1, 1, 1, 1), value=0.0).permute(1, 2, 0)
    return normals


def quats_from_normals(normals_world: torch.Tensor) -> torch.Tensor:
    normals_world = F.normalize(normals_world, dim=-1, eps=1e-6)
    z_axis = torch.tensor(
        [0.0, 0.0, 1.0], device=normals_world.device, dtype=normals_world.dtype
    ).expand_as(normals_world)
    dots = (z_axis * normals_world).sum(dim=-1, keepdim=True)
    xyz = torch.cross(z_axis, normals_world, dim=-1)
    quats = torch.cat([1.0 + dots, xyz], dim=-1)

    opposite = dots.squeeze(-1) < -0.9999
    if opposite.any():
        fallback = torch.zeros(
            (int(opposite.sum().item()), 4),
            device=normals_world.device,
            dtype=normals_world.dtype,
        )
        fallback[:, 2] = 1.0
        quats[opposite] = fallback

    small = quats.norm(dim=-1) < 1e-8
    if small.any():
        quats[small, 0] = 1.0
        quats[small, 1:] = 0.0
    return F.normalize(quats, dim=-1, eps=1e-6)


def normals_from_quats(quats: torch.Tensor) -> torch.Tensor:
    quats = F.normalize(quats, dim=-1, eps=1e-6)
    w, x, y, z = quats.unbind(dim=-1)
    return F.normalize(
        torch.stack(
            (
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x.square() + y.square()),
            ),
            dim=-1,
        ),
        dim=-1,
        eps=1e-6,
    )


def build_frame_gaussians(
    image: torch.Tensor,
    depth: torch.Tensor,
    camera: PinholeCamera,
    config: RGBDGaussianBuilderConfig | None = None,
    confidence_image: torch.Tensor | None = None,
    albedo_image: torch.Tensor | None = None,
    roughness_image: torch.Tensor | None = None,
    metallic_image: torch.Tensor | None = None,
    normal_world_image: torch.Tensor | None = None,
) -> FrameGaussians:
    config = config or RGBDGaussianBuilderConfig()
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"image must have shape [3, H, W], got {tuple(image.shape)}")
    if depth.shape[-2:] != image.shape[-2:]:
        raise ValueError("depth and image must share spatial shape")
    if albedo_image is not None and albedo_image.shape != image.shape:
        raise ValueError("albedo_image must share shape with image")
    if confidence_image is not None and confidence_image.shape != (
        1,
        image.shape[1],
        image.shape[2],
    ):
        raise ValueError("confidence_image must have shape [1, H, W] matching image")
    if roughness_image is not None and roughness_image.shape != (
        1,
        image.shape[1],
        image.shape[2],
    ):
        raise ValueError("roughness_image must have shape [1, H, W] matching image")
    if metallic_image is not None and metallic_image.shape != (
        1,
        image.shape[1],
        image.shape[2],
    ):
        raise ValueError("metallic_image must have shape [1, H, W] matching image")
    if normal_world_image is not None and normal_world_image.shape != image.shape:
        raise ValueError("normal_world_image must have shape [3, H, W] matching image")
    device = image.device
    dtype = image.dtype
    camera = camera.to(device)
    normals_world_full = normals_from_depth(depth, camera)
    points_world, grid_y, grid_x = depth_to_world_points(
        depth, camera, stride=config.pixel_stride
    )
    num_candidates = int(grid_y.numel())
    sampled_depth = depth[0, grid_y.long(), grid_x.long()]
    sampled_confidence = (
        confidence_image[:, grid_y.long(), grid_x.long()].permute(1, 2, 0)
        if confidence_image is not None
        else torch.ones((*grid_y.shape, 1), device=device, dtype=dtype)
    )
    albedo_source = albedo_image if albedo_image is not None else image
    sampled_albedo = albedo_source[:, grid_y.long(), grid_x.long()].permute(1, 2, 0)
    sampled_roughness = (
        roughness_image[:, grid_y.long(), grid_x.long()].permute(1, 2, 0)
        if roughness_image is not None
        else torch.zeros((*grid_y.shape, 1), device=device, dtype=dtype)
    )
    sampled_metallic = (
        metallic_image[:, grid_y.long(), grid_x.long()].permute(1, 2, 0)
        if metallic_image is not None
        else torch.zeros((*grid_y.shape, 1), device=device, dtype=dtype)
    )

    if normal_world_image is not None:
        sampled_normals = normal_world_image[:, grid_y.long(), grid_x.long()].permute(
            1, 2, 0
        )
    else:
        sampled_normals = normals_world_full[grid_y.long(), grid_x.long()]

    valid = (
        torch.isfinite(sampled_depth)
        & (sampled_depth >= config.min_depth)
        & (sampled_depth <= config.max_depth)
        & torch.isfinite(sampled_confidence[..., 0])
        & (sampled_confidence[..., 0] >= float(config.min_confidence))
        & torch.isfinite(points_world).all(dim=-1)
        & (sampled_normals.norm(dim=-1) > 1e-6)
    )
    num_confidence_kept = int(valid.sum().item())

    points_world = points_world[valid]
    sampled_confidence = sampled_confidence[valid]
    sampled_albedo = sampled_albedo[valid]
    sampled_roughness = sampled_roughness[valid]
    sampled_metallic = sampled_metallic[valid]
    sampled_normals = sampled_normals[valid]
    sampled_depth = sampled_depth[valid]
    sampled_y = grid_y.long()[valid]
    sampled_x = grid_x.long()[valid]

    if points_world.numel() == 0:
        return FrameGaussians(
            image_name=camera.image_name,
            camera=camera,
            num_candidates=num_candidates,
            num_confidence_kept=num_confidence_kept,
            means_world=torch.zeros((0, 3), device=device, dtype=dtype),
            sample_y=torch.zeros((0,), device=device, dtype=torch.long),
            sample_x=torch.zeros((0,), device=device, dtype=torch.long),
            colors=torch.zeros((0, 3), device=device, dtype=dtype),
            albedo=torch.zeros((0, 3), device=device, dtype=dtype),
            roughness=torch.zeros((0, 1), device=device, dtype=dtype),
            metallic=torch.zeros((0, 1), device=device, dtype=dtype),
            quats=torch.zeros((0, 4), device=device, dtype=dtype),
            scales=torch.zeros((0, 3), device=device, dtype=dtype),
            opacities=torch.zeros((0, 1), device=device, dtype=dtype),
            confidence=torch.zeros((0, 1), device=device, dtype=dtype),
        )

    pixel_world_x = sampled_depth / camera.fx * float(config.pixel_stride)
    pixel_world_y = sampled_depth / camera.fy * float(config.pixel_stride)
    scale_xy = 0.5 * (pixel_world_x + pixel_world_y) * config.scale_xy_multiplier
    scale_z = scale_xy * config.scale_z_multiplier
    scales = torch.stack([scale_xy, scale_xy, scale_z], dim=-1).clamp_min(1e-6)

    confidence = sampled_confidence.reshape(-1, 1)
    quats = quats_from_normals(sampled_normals.reshape(-1, 3))
    opacities = torch.full(
        (points_world.shape[0], 1),
        float(config.default_opacity),
        device=device,
        dtype=dtype,
    )

    return FrameGaussians(
        image_name=camera.image_name,
        camera=camera,
        num_candidates=num_candidates,
        num_confidence_kept=num_confidence_kept,
        means_world=points_world.reshape(-1, 3),
        sample_y=sampled_y.reshape(-1),
        sample_x=sampled_x.reshape(-1),
        colors=sampled_albedo.reshape(-1, 3),
        albedo=sampled_albedo.reshape(-1, 3),
        roughness=sampled_roughness.reshape(-1, 1),
        metallic=sampled_metallic.reshape(-1, 1),
        quats=quats,
        scales=scales.reshape(-1, 3),
        opacities=opacities,
        confidence=confidence,
    )


def fuse_frame_gaussians(
    state: GaussianMapState,
    frame: FrameGaussians,
    config: RGBDGaussianFusionConfig | None = None,
    allow_create: bool = True,
) -> tuple[GaussianMapState, dict[str, int]]:
    from .rgbd_fusion import fuse_frame_gaussians as fuse

    return fuse(
        state,
        frame,
        config=config,
        allow_create=allow_create,
    )
