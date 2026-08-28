from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .gsplat_adapter import (
    as_chw_normal_map,
    ensure_local_gsplat_path,
    gaussian_map_to_splats,
)
from .rgbd import GaussianMapState
from .types import PinholeCamera


@dataclass
class GlobalOptimizationObservation:
    image_name: str
    camera: PinholeCamera
    depth: torch.Tensor
    albedo: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor
    normal_world: torch.Tensor
    valid_depth: torch.Tensor


@dataclass
class GlobalOptimizationConfig:
    steps: int = 1000
    geometry_learning_rate: float = 1e-4
    scale_learning_rate: float = 0.005
    albedo_learning_rate: float = 1e-2
    roughness_learning_rate: float = 1e-2
    metallic_learning_rate: float = 1e-2
    opacity_learning_rate: float = 0.05
    rotation_learning_rate: float = 0.001
    depth_weight: float = 1.0
    albedo_weight: float = 1.0
    roughness_weight: float = 1.0
    metallic_weight: float = 1.0
    normal_weight: float = 1.0
    surface_normal_weight: float = 0.0
    alpha_weight: float = 1.0
    position_regularization_weight: float = 1e-2
    scale_regularization_weight: float = 1e-2
    opacity_regularization_weight: float = 1e-3
    planar_scale: float = 1.0
    thickness_scale: float = 1.0
    debug_render_dir: str = ""
    debug_render_interval: int = 0


def _safe_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1e-4, 1.0 - 1e-4))


def _as_chw_normal_map(
    value: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if value.dim() == 4 and value.shape[0] == 1:
        value = value[0]
    if value.dim() == 3 and value.shape == (3, height, width):
        pass
    elif value.dim() == 3 and value.shape == (height, width, 3):
        value = value.permute(2, 0, 1)
    else:
        value = as_chw_normal_map(value)
        if value.shape != (3, height, width):
            raise ValueError(
                f"normal map must resolve to {(3, height, width)}, got {tuple(value.shape)}"
            )
    return F.normalize(value, dim=0, eps=1e-6)


def _tensor_to_bgr(value: torch.Tensor) -> np.ndarray:
    image = value.detach().cpu().float().clamp(0.0, 1.0)
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    rgb = image[:3].permute(1, 2, 0).numpy()
    return np.ascontiguousarray(rgb[..., ::-1] * 255.0).astype(np.uint8)


def _normal_to_bgr(value: torch.Tensor) -> np.ndarray:
    normal = F.normalize(value.detach().cpu().float(), dim=0, eps=1e-6)
    return _tensor_to_bgr(normal.mul(0.5).add(0.5))


def _depth_to_bgr(value: torch.Tensor, near: float, far: float) -> np.ndarray:
    depth = value.detach().cpu().float().squeeze().numpy()
    valid = np.isfinite(depth) & (depth > 1e-6)
    image = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any() and far > near:
        normalized = np.clip((depth - near) / (far - near), 0.0, 1.0)
        image[valid] = np.round((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    elif valid.any():
        image[valid] = 128
    colored = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _write_debug_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save optimization debug image: {path}")


def _write_raw_array(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = value.detach().cpu().float().squeeze().numpy()
    np.save(path, array)


def _write_grayscale_image(
    path: Path,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> dict[str, float]:
    array = value.detach().cpu().float().squeeze().numpy()
    finite = np.isfinite(array)
    image = np.zeros(array.shape, dtype=np.uint8)
    finite_values = array[finite]
    if finite_values.size == 0:
        _write_debug_image(path, image)
        return {"min": 0.0, "max": 0.0, "scale": float(scale or 0.0)}

    if scale is None:
        scale_value = float(np.percentile(finite_values, 99.0))
        scale_value = max(scale_value, 1e-6)
    else:
        scale_value = max(float(scale), 1e-6)

    normalized = np.clip(array / scale_value, 0.0, 1.0)
    image[finite] = np.round(normalized[finite] * 255.0).astype(np.uint8)
    _write_debug_image(path, image)
    return {
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "mean": float(finite_values.mean()),
        "p50": float(np.percentile(finite_values, 50.0)),
        "p95": float(np.percentile(finite_values, 95.0)),
        "p99": float(np.percentile(finite_values, 99.0)),
        "scale": scale_value,
    }




def _normals_from_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    quaternions = F.normalize(quaternions, dim=-1, eps=1e-6)
    w, x, y, z = quaternions.unbind(dim=-1)
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


def _camera_tensors(
    camera: PinholeCamera,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera_to_world = camera.camera_to_world.to(device=device, dtype=dtype)
    viewmat = torch.linalg.inv(camera_to_world).unsqueeze(0)
    intrinsics = torch.eye(3, device=device, dtype=dtype)
    intrinsics[0, 0] = camera.fx
    intrinsics[1, 1] = camera.fy
    intrinsics[0, 2] = camera.cx
    intrinsics[1, 2] = camera.cy
    return viewmat, intrinsics.unsqueeze(0)


def _render_observation(
    splats: torch.nn.ParameterDict,
    albedo_logits: torch.nn.Parameter,
    roughness_logits: torch.nn.Parameter,
    metallic_logits: torch.nn.Parameter,
    observation: GlobalOptimizationObservation,
    config: GlobalOptimizationConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from gsplat.rendering import rasterization_2dgs

    means = splats["means"]
    quaternions = F.normalize(splats["quats"], dim=-1, eps=1e-6)
    scales = torch.exp(splats["scales"])
    scales_xy = scales[:, :2] * max(float(config.planar_scale), 1e-6)
    scale_z = scales_xy.mean(dim=-1, keepdim=True) * max(
        float(config.thickness_scale), 1e-6
    )
    render_scales = torch.cat((scales_xy, scale_z), dim=-1)
    opacities = torch.sigmoid(splats["opacities"]).reshape(-1)
    material = torch.cat(
        (
            torch.sigmoid(albedo_logits),
            torch.sigmoid(roughness_logits),
            torch.sigmoid(metallic_logits),
        ),
        dim=1,
    )
    viewmats, intrinsics = _camera_tensors(
        observation.camera,
        means.device,
        means.dtype,
    )
    rendered, alphas, rendered_normals, surf_normals, _, _, _ = rasterization_2dgs(
        means=means,
        quats=quaternions,
        scales=render_scales,
        opacities=opacities,
        colors=material,
        viewmats=viewmats,
        Ks=intrinsics,
        width=observation.camera.width,
        height=observation.camera.height,
        sh_degree=None,
        packed=False,
        render_mode="RGB+ED",
    )
    rendered = rendered[0]
    coverage = alphas[0, ..., 0]
    material = rendered[..., :5] / coverage.clamp_min(1e-6).unsqueeze(-1)
    material = torch.where(
        (coverage > 1e-4).unsqueeze(-1), material, torch.zeros_like(material)
    )
    rendered_normals = _as_chw_normal_map(
        rendered_normals,
        observation.camera.height,
        observation.camera.width,
    )
    surf_normals = _as_chw_normal_map(
        surf_normals,
        observation.camera.height,
        observation.camera.width,
    )
    return (
        material[..., :3].permute(2, 0, 1).clamp(0.0, 1.0),
        rendered[..., 5].unsqueeze(0),
        material[..., 3].unsqueeze(0).clamp(0.0, 1.0),
        material[..., 4].unsqueeze(0).clamp(0.0, 1.0),
        rendered_normals,
        surf_normals,
        alphas[0, ..., 0],
    )


def _data_loss(
    rendered_albedo: torch.Tensor,
    rendered_depth: torch.Tensor,
    rendered_roughness: torch.Tensor,
    rendered_metallic: torch.Tensor,
    rendered_normals: torch.Tensor,
    rendered_alpha: torch.Tensor,
    observation: GlobalOptimizationObservation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_depth = observation.depth.to(
        device=rendered_depth.device,
        dtype=rendered_depth.dtype,
    )
    target_albedo = observation.albedo.to(
        device=rendered_albedo.device,
        dtype=rendered_albedo.dtype,
    )
    target_roughness = observation.roughness.to(
        device=rendered_roughness.device, dtype=rendered_roughness.dtype
    )
    target_metallic = observation.metallic.to(
        device=rendered_metallic.device, dtype=rendered_metallic.dtype
    )
    target_normals = observation.normal_world.to(
        device=rendered_normals.device, dtype=rendered_normals.dtype
    )
    valid_target_normal = target_normals.norm(dim=0) > 1e-6
    target_normals = F.normalize(target_normals, dim=0, eps=1e-6)
    valid = observation.valid_depth.to(device=rendered_depth.device)
    depth_residual = (rendered_depth - target_depth).abs() / target_depth.clamp_min(1e-3)
    depth_loss = depth_residual[valid].mean()
    albedo_mask = (valid & (rendered_alpha > 1e-4)).expand_as(target_albedo)
    albedo_loss = (rendered_albedo - target_albedo).abs()[albedo_mask].mean()
    material_mask = valid & (rendered_alpha > 1e-4)
    if material_mask.any():
        roughness_loss = (rendered_roughness - target_roughness).abs()[material_mask].mean()
        metallic_loss = (rendered_metallic - target_metallic).abs()[material_mask].mean()
        albedo_loss = (rendered_albedo - target_albedo).abs()[material_mask.expand_as(target_albedo)].mean()
    else:
        zero = rendered_albedo.sum() * 0.0
        albedo_loss = roughness_loss = metallic_loss = zero
    normal_valid = valid.squeeze(0) if valid.dim() == 3 else valid
    normal_valid = normal_valid & valid_target_normal & (rendered_alpha > 1e-4)
    cosine = (rendered_normals * target_normals).sum(dim=0).clamp(-1.0, 1.0)
    normal_loss = (1.0 - cosine)[normal_valid].mean()
    target_alpha = torch.ones_like(rendered_alpha)
    alpha_loss = (rendered_alpha.clamp(0.0, 1.0) - target_alpha).abs().mean()
    return depth_loss, albedo_loss, roughness_loss, metallic_loss, normal_loss, alpha_loss


def _regularization_loss(
    splats: torch.nn.ParameterDict,
    initial: dict[str, torch.Tensor],
    config: GlobalOptimizationConfig,
) -> torch.Tensor:
    initial_scale = torch.exp(initial["scales"]).mean(dim=-1, keepdim=True).clamp_min(1e-4)
    position = ((splats["means"] - initial["means"]) / initial_scale).square().mean()
    scale = (splats["scales"] - initial["scales"]).square().mean()
    opacity = (splats["opacities"] - initial["opacities"]).square().mean()
    return (
        float(config.position_regularization_weight) * position
        + float(config.scale_regularization_weight) * scale
        + float(config.opacity_regularization_weight) * opacity
    )


def _surface_normal_consistency_loss(
    rendered_normals: torch.Tensor,
    surf_normals: torch.Tensor,
    rendered_alpha: torch.Tensor,
) -> torch.Tensor:
    rendered_normals = F.normalize(rendered_normals, dim=0, eps=1e-6)
    surf_normals = F.normalize(surf_normals, dim=0, eps=1e-6)
    cosine = (rendered_normals * surf_normals).sum(dim=0).clamp(-1.0, 1.0)
    residual = 1.0 - cosine
    valid = (
        torch.isfinite(residual)
        & torch.isfinite(rendered_normals).all(dim=0)
        & torch.isfinite(surf_normals).all(dim=0)
        & (rendered_alpha > 1e-4)
    )
    if not valid.any():
        return residual.sum() * 0.0
    return residual[valid].mean()


def _save_optimization_debug_step(
    *,
    output_dir: Path,
    step: int,
    observation_index: int,
    observation: GlobalOptimizationObservation,
    rendered_albedo: torch.Tensor,
    rendered_depth: torch.Tensor,
    rendered_roughness: torch.Tensor,
    rendered_metallic: torch.Tensor,
    rendered_normals: torch.Tensor,
    surf_normals: torch.Tensor,
    rendered_alpha: torch.Tensor,
    losses: dict[str, float],
) -> None:
    target_depth = observation.depth.to(
        device=rendered_depth.device, dtype=rendered_depth.dtype
    )
    target_albedo = observation.albedo.to(
        device=rendered_albedo.device, dtype=rendered_albedo.dtype
    )
    target_roughness = observation.roughness.to(
        device=rendered_roughness.device, dtype=rendered_roughness.dtype
    )
    target_metallic = observation.metallic.to(
        device=rendered_metallic.device, dtype=rendered_metallic.dtype
    )
    target_normals = observation.normal_world.to(
        device=rendered_normals.device, dtype=rendered_normals.dtype
    )
    valid_depth = observation.valid_depth.to(device=rendered_depth.device)
    material_mask = torch.ones_like(rendered_alpha, dtype=torch.bool)

    frame_dir = output_dir / f"step_{step:06d}_{observation.image_name}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    target_values = target_depth[valid_depth & torch.isfinite(target_depth)]
    rendered_values = rendered_depth[
        material_mask & torch.isfinite(rendered_depth)
    ]
    if target_values.numel() > 0 and rendered_values.numel() > 0:
        depth_values = torch.cat((target_values.flatten(), rendered_values.flatten()))
        near, far = torch.quantile(
            depth_values.float(),
            torch.tensor((0.02, 0.98), device=depth_values.device),
        ).tolist()
    else:
        near, far = 0.0, 1.0

    _write_debug_image(frame_dir / "target_albedo.png", _tensor_to_bgr(target_albedo))
    _write_debug_image(
        frame_dir / "rendered_albedo.png", _tensor_to_bgr(rendered_albedo)
    )
    _write_debug_image(
        frame_dir / "error_albedo_l1.png",
        _tensor_to_bgr((rendered_albedo - target_albedo).abs()),
    )
    _write_debug_image(
        frame_dir / "target_roughness.png", _tensor_to_bgr(target_roughness)
    )
    _write_debug_image(
        frame_dir / "rendered_roughness.png", _tensor_to_bgr(rendered_roughness)
    )
    _write_debug_image(
        frame_dir / "error_roughness_l1.png",
        _tensor_to_bgr((rendered_roughness - target_roughness).abs()),
    )
    _write_debug_image(
        frame_dir / "target_metallic.png", _tensor_to_bgr(target_metallic)
    )
    _write_debug_image(
        frame_dir / "rendered_metallic.png", _tensor_to_bgr(rendered_metallic)
    )
    _write_debug_image(
        frame_dir / "error_metallic_l1.png",
        _tensor_to_bgr((rendered_metallic - target_metallic).abs()),
    )
    _write_debug_image(
        frame_dir / "target_normal_world.png", _normal_to_bgr(target_normals)
    )
    _write_debug_image(
        frame_dir / "rendered_normal_world.png", _normal_to_bgr(rendered_normals)
    )
    _write_debug_image(
        frame_dir / "rendered_surf_normal_world.png", _normal_to_bgr(surf_normals)
    )
    target_normals = F.normalize(target_normals.float(), dim=0, eps=1e-6)
    rendered_normals = F.normalize(rendered_normals.float(), dim=0, eps=1e-6)
    surf_normals = F.normalize(surf_normals.float(), dim=0, eps=1e-6)
    normal_error = 1.0 - (rendered_normals * target_normals).sum(dim=0).clamp(-1.0, 1.0)
    _write_debug_image(
        frame_dir / "error_normal_cosine.png",
        _tensor_to_bgr((normal_error / 2.0).unsqueeze(0)),
    )
    surface_normal_error = 1.0 - (rendered_normals * surf_normals).sum(dim=0).clamp(
        -1.0, 1.0
    )
    _write_debug_image(
        frame_dir / "error_surface_normal_cosine.png",
        _tensor_to_bgr((surface_normal_error / 2.0).unsqueeze(0)),
    )
    _write_debug_image(frame_dir / "target_depth.png", _depth_to_bgr(target_depth, near, far))
    _write_debug_image(
        frame_dir / "rendered_depth.png", _depth_to_bgr(rendered_depth, near, far)
    )

    _write_debug_image(
        frame_dir / "valid_depth.png", _tensor_to_bgr(valid_depth.float())
    )
    _write_debug_image(
        frame_dir / "material_mask.png", _tensor_to_bgr(material_mask.float())
    )

    target_alpha = torch.ones_like(rendered_alpha, dtype=rendered_alpha.dtype)
    _write_debug_image(
        frame_dir / "target_alpha.png", _tensor_to_bgr(target_alpha.unsqueeze(0))
    )
    alpha_error = (rendered_alpha.clamp(0.0, 1.0) - target_alpha).abs()
    _write_debug_image(
        frame_dir / "error_alpha_l1.png", _tensor_to_bgr(alpha_error.unsqueeze(0))
    )
    _write_debug_image(
        frame_dir / "rendered_alpha.png", _tensor_to_bgr(rendered_alpha.unsqueeze(0))
    )

    metadata = {
        "step": step,
        "frame": observation.image_name,
        "observation_index": observation_index,
        "losses": losses,
        "depth_visualization_range": {"near": near, "far": far},
    }
    (frame_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def optimize_gaussian_map_global(
    state: GaussianMapState,
    observations: list[GlobalOptimizationObservation],
    config: GlobalOptimizationConfig,
    device: torch.device | str,
) -> tuple[GaussianMapState, list[dict[str, float | int | str]]]:
    if not observations or config.steps <= 0 or state.means_world.shape[0] == 0:
        return state, []

    ensure_local_gsplat_path()
    device = torch.device(device)
    splats = gaussian_map_to_splats(
        state,
        device=device,
        orient_to_normals=True,
    )
    splats["quats"].requires_grad_(True)
    splats["colors"].requires_grad_(False)
    albedo_logits = torch.nn.Parameter(_safe_logit(state.albedo.to(device=device)))
    roughness_logits = torch.nn.Parameter(_safe_logit(state.roughness.to(device=device)))
    metallic_logits = torch.nn.Parameter(_safe_logit(state.metallic.to(device=device)))
    initial = {
        name: splats[name].detach().clone()
        for name in ("means", "scales", "opacities")
    }
    optimizer = torch.optim.Adam(
        [
            {"params": [splats["means"]], "lr": float(config.geometry_learning_rate)},
            {"params": [splats["scales"]], "lr": float(config.scale_learning_rate)},
            {"params": [splats["quats"]], "lr": float(config.rotation_learning_rate)},
            {"params": [albedo_logits], "lr": float(config.albedo_learning_rate)},
            {"params": [roughness_logits], "lr": float(config.roughness_learning_rate)},
            {"params": [metallic_logits], "lr": float(config.metallic_learning_rate)},
            {"params": [splats["opacities"]], "lr": float(config.opacity_learning_rate)},
        ]
    )


    history: list[dict[str, float | int | str]] = []
    debug_render_dir = Path(config.debug_render_dir) if config.debug_render_dir else None
    debug_render_interval = max(int(config.debug_render_interval), 0)
    observation_indices = torch.randint(
        len(observations),
        (int(config.steps),),
        device="cpu",
    ).tolist()
    for step, observation_index in enumerate(observation_indices):
        observation = observations[observation_index]
        optimizer.zero_grad(set_to_none=True)
        (
            rendered_albedo,
            rendered_depth,
            rendered_roughness,
            rendered_metallic,
            rendered_normals,
            surf_normals,
            rendered_alpha,
        ) = _render_observation(
            splats,
            albedo_logits,
            roughness_logits,
            metallic_logits,
            observation,
            config,
        )
        (
            depth_loss,
            albedo_loss,
            roughness_loss,
            metallic_loss,
            normal_loss,
            alpha_loss,
        ) = _data_loss(
            rendered_albedo,
            rendered_depth,
            rendered_roughness,
            rendered_metallic,
            rendered_normals,
            rendered_alpha,
            observation,
        )
        surface_normal_loss = _surface_normal_consistency_loss(
            rendered_normals,
            surf_normals,
            rendered_alpha,
        )
        regularization = _regularization_loss(splats, initial, config)
        loss = (
            float(config.depth_weight) * depth_loss
            + float(config.albedo_weight) * albedo_loss
            + float(config.roughness_weight) * roughness_loss
            + float(config.metallic_weight) * metallic_loss
            + float(config.normal_weight) * normal_loss
            + float(config.surface_normal_weight) * surface_normal_loss
            + float(config.alpha_weight) * alpha_loss
            + regularization
        )
        loss.backward()
        optimizer.step()

        should_save_debug = (
            debug_render_dir is not None
            and (
                step == 0
                or (debug_render_interval > 0 and (step + 1) % debug_render_interval == 0)
                or step + 1 == config.steps
            )
        )
        if should_save_debug:
            _save_optimization_debug_step(
                output_dir=debug_render_dir,
                step=step + 1,
                observation_index=observation_index,
                observation=observation,
                rendered_albedo=rendered_albedo,
                rendered_depth=rendered_depth,
                rendered_roughness=rendered_roughness,
                rendered_metallic=rendered_metallic,
                rendered_normals=rendered_normals,
                surf_normals=surf_normals,
                rendered_alpha=rendered_alpha,
                losses={
                    "loss": float(loss.detach()),
                    "depth": float(depth_loss.detach()),
                    "albedo": float(albedo_loss.detach()),
                    "roughness": float(roughness_loss.detach()),
                    "metallic": float(metallic_loss.detach()),
                    "normal": float(normal_loss.detach()),
                    "surface_normal": float(surface_normal_loss.detach()),
                    "alpha": float(alpha_loss.detach()),
                    "regularization": float(regularization.detach()),
                },

            )

        if step == 0 or (step + 1) % 10 == 0 or step + 1 == config.steps:
            entry: dict[str, float | int | str] = {
                "step": step + 1,
                "frame": observation.image_name,
                "loss": float(loss.detach()),
                "depth": float(depth_loss.detach()),
                "albedo": float(albedo_loss.detach()),
                "roughness": float(roughness_loss.detach()),
                "metallic": float(metallic_loss.detach()),
                "normal": float(normal_loss.detach()),
                "surface_normal": float(surface_normal_loss.detach()),
                "alpha": float(alpha_loss.detach()),
                "regularization": float(regularization.detach()),
            }
            history.append(entry)
            print(
                f"[global-opt] step={step + 1}/{config.steps} "
                f"frame={observation.image_name} loss={entry['loss']:.6f} "
                f"depth={entry['depth']:.6f} albedo={entry['albedo']:.6f} "
                f"roughness={entry['roughness']:.6f} metallic={entry['metallic']:.6f} "
                f"normal={entry['normal']:.6f} surf_normal={entry['surface_normal']:.6f} "
                f"alpha={entry['alpha']:.6f} reg={entry['regularization']:.6f}",
                flush=True,
            )

    optimized_quats = F.normalize(splats["quats"], dim=-1, eps=1e-6).detach()
    optimized = GaussianMapState(
        means_world=splats["means"].detach(),
        colors=torch.sigmoid(albedo_logits).detach(),
        albedo=torch.sigmoid(albedo_logits).detach(),
        roughness=torch.sigmoid(roughness_logits).detach(),
        metallic=torch.sigmoid(metallic_logits).detach(),
        scales=torch.exp(splats["scales"]).detach(),
        opacities=torch.sigmoid(splats["opacities"]).detach().unsqueeze(-1),
        confidence_sum=state.confidence_sum.to(device).detach(),
        update_count=state.update_count.to(device).detach(),
        quats=optimized_quats,
    )
    return optimized, history
