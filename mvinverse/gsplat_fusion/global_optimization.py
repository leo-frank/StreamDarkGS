from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .gsplat_adapter import ensure_local_gsplat_path, gaussian_map_to_splats
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
    valid_depth: torch.Tensor


@dataclass
class GlobalOptimizationConfig:
    steps: int = 1000
    geometry_learning_rate: float = 1e-4
    albedo_learning_rate: float = 1e-2
    roughness_learning_rate: float = 1e-2
    metallic_learning_rate: float = 1e-2
    opacity_learning_rate: float = 1e-3
    depth_weight: float = 1.0
    albedo_weight: float = 1.0
    roughness_weight: float = 1.0
    metallic_weight: float = 1.0
    position_regularization_weight: float = 1e-2
    scale_regularization_weight: float = 1e-2
    opacity_regularization_weight: float = 1e-3
    planar_scale: float = 1.0
    thickness_scale: float = 1.0


def _safe_logit(value: torch.Tensor) -> torch.Tensor:
    return torch.logit(value.clamp(1e-4, 1.0 - 1e-4))


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    rendered, _, _, _, _, _, _ = rasterization_2dgs(
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
    return (
        rendered[..., :3].permute(2, 0, 1),
        rendered[..., 5].unsqueeze(0),
        rendered[..., 3].unsqueeze(0),
        rendered[..., 4].unsqueeze(0),
    )


def _data_loss(
    rendered_albedo: torch.Tensor,
    rendered_depth: torch.Tensor,
    rendered_roughness: torch.Tensor,
    rendered_metallic: torch.Tensor,
    observation: GlobalOptimizationObservation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    valid = observation.valid_depth.to(device=rendered_depth.device)
    depth_residual = (rendered_depth - target_depth).abs() / target_depth.clamp_min(1e-3)
    depth_loss = depth_residual[valid].mean()
    albedo_mask = valid.expand_as(target_albedo)
    albedo_loss = (rendered_albedo - target_albedo).abs()[albedo_mask].mean()
    material_mask = valid
    roughness_loss = (rendered_roughness - target_roughness).abs()[material_mask].mean()
    metallic_loss = (rendered_metallic - target_metallic).abs()[material_mask].mean()
    return depth_loss, albedo_loss, roughness_loss, metallic_loss


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
    splats["quats"].requires_grad_(False)
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
            {
                "params": [splats["means"], splats["scales"]],
                "lr": float(config.geometry_learning_rate),
            },
            {"params": [albedo_logits], "lr": float(config.albedo_learning_rate)},
            {"params": [roughness_logits], "lr": float(config.roughness_learning_rate)},
            {"params": [metallic_logits], "lr": float(config.metallic_learning_rate)},
            {"params": [splats["opacities"]], "lr": float(config.opacity_learning_rate)},
        ]
    )

    history: list[dict[str, float | int | str]] = []
    observation_indices = torch.randint(
        len(observations),
        (int(config.steps),),
        device="cpu",
    ).tolist()
    for step, observation_index in enumerate(observation_indices):
        observation = observations[observation_index]
        optimizer.zero_grad(set_to_none=True)
        rendered_albedo, rendered_depth, rendered_roughness, rendered_metallic = _render_observation(
            splats,
            albedo_logits,
            roughness_logits,
            metallic_logits,
            observation,
            config,
        )
        depth_loss, albedo_loss, roughness_loss, metallic_loss = _data_loss(
            rendered_albedo,
            rendered_depth,
            rendered_roughness,
            rendered_metallic,
            observation,
        )
        regularization = _regularization_loss(splats, initial, config)
        loss = (
            float(config.depth_weight) * depth_loss
            + float(config.albedo_weight) * albedo_loss
            + float(config.roughness_weight) * roughness_loss
            + float(config.metallic_weight) * metallic_loss
            + regularization
        )
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % 10 == 0 or step + 1 == config.steps:
            entry: dict[str, float | int | str] = {
                "step": step + 1,
                "frame": observation.image_name,
                "loss": float(loss.detach()),
                "depth": float(depth_loss.detach()),
                "albedo": float(albedo_loss.detach()),
                "roughness": float(roughness_loss.detach()),
                "metallic": float(metallic_loss.detach()),
                "regularization": float(regularization.detach()),
            }
            history.append(entry)
            print(
                f"[global-opt] step={step + 1}/{config.steps} "
                f"frame={observation.image_name} loss={entry['loss']:.6f} "
                f"depth={entry['depth']:.6f} albedo={entry['albedo']:.6f} "
                f"roughness={entry['roughness']:.6f} metallic={entry['metallic']:.6f} "
                f"reg={entry['regularization']:.6f}",
                flush=True,
            )

    optimized = GaussianMapState(
        means_world=splats["means"].detach(),
        colors=torch.sigmoid(albedo_logits).detach(),
        albedo=torch.sigmoid(albedo_logits).detach(),
        roughness=torch.sigmoid(roughness_logits).detach(),
        metallic=torch.sigmoid(metallic_logits).detach(),
        normals_world=state.normals_world.to(device).detach(),
        scales=torch.exp(splats["scales"]).detach(),
        opacities=torch.sigmoid(splats["opacities"]).detach().unsqueeze(-1),
        confidence_sum=state.confidence_sum.to(device).detach(),
        update_count=state.update_count.to(device).detach(),
    )
    return optimized, history
